"""Fixed descriptive neural baseline; never supplies labels to the compiler."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AtomV2' / 'Harness'))
from atomv2.data import build_bundle
from atomv2.e7_audit import boundary_panel
from atomv2.panel import load_checkpoint


def main():
    torch.set_num_threads(4)
    target = ROOT / 'AtomV3/results/neural_baseline.json'
    if target.exists():
        raise FileExistsError(target)
    candidates = list((ROOT / 'AtomV2/runs/e4').glob('A14_s1_*/checkpoints/final.pt'))
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one registered A14 seed 1 final')
    checkpoint = candidates[0]
    start = time.perf_counter()
    model, config, step = load_checkpoint(checkpoint.parent.parent, 'final.pt')
    model.eval()
    bundle = build_bundle(config)
    with torch.inference_mode():
        panel = boundary_panel(model, bundle)
    groups = {}
    for name, aggregates in panel['by_group'].items():
        rows = [x for x in panel['cells'].values() if x['group'] == name]
        groups[name] = dict(aggregates, n_tasks=len(rows), n_examples=len(rows) * 400,
                           raw_correct=sum(round(x['raw_acc'] * 400) for x in rows),
                           repaired_correct=sum(round(x['self_bottleneck_acc'] * 400) for x in rows))
    result = {
        'label': 'FIXED_DESCRIPTIVE_BASELINE_NOT_COMPILER_TRAINING',
        'checkpoint': checkpoint.relative_to(ROOT).as_posix(),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'checkpoint_step': step,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
        'groups': groups, 'seconds': time.perf_counter() - start,
        'torch': torch.__version__,
        'limits': 'Known-world pair panel; extra decoder/encoder computation in repaired arm; no neural training.',
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
