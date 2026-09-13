"""Run the prospective Skill Forge protocol sequentially on the local CPU.

This driver writes a source/checkpoint registration before the first teacher
query. It never overwrites a completed stage. Re-running resumes only when the
registered source fingerprint still matches.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / 'results'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    files = sorted(HERE.glob('*.py')) + [HERE / 'PROTOCOL.md', HERE / 'RELATED_WORK.md']
    files += sorted((ROOT / 'AtomV2/Harness/atomv2').glob('*.py'))
    files += [ROOT / 'AtomV2/Harness/splits/split_v2.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def verify_registration():
    record = json.loads((RESULTS / 'REGISTRATION.json').read_text(encoding='utf-8'))
    if record['source_sha256'] != source_hashes():
        raise RuntimeError('Source changed during the experiment')
    for relative, expected in record['checkpoint_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f'Registered checkpoint changed: {relative}')
    return record


def register():
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / 'REGISTRATION.json'
    fingerprint = source_hashes()
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['source_sha256'] != fingerprint:
            raise RuntimeError('Source differs from the prospective registration; refuse silent protocol drift')
        verify_registration()
        return record
    checkpoints = []
    for seed in (0, 1, 2):
        found = list((ROOT / 'AtomV2/runs/e4').glob(f'A14_s{seed}_*/checkpoints/final.pt'))
        if len(found) != 1:
            raise RuntimeError(f'A14 checkpoint ambiguous: {seed}')
        checkpoints += found
    found = list((ROOT / 'AtomV2/runs/e11').glob('A33_s1_*/checkpoints/final.pt'))
    if len(found) != 1:
        raise RuntimeError('A33 checkpoint ambiguous')
    checkpoints += found
    record = {
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'BEFORE_RESULT_BEARING_QUERIES',
        'source_sha256': fingerprint,
        'checkpoint_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in checkpoints},
        'python': platform.python_version(),
        'query_seed': 20260912, 'execution_seed': 20260913,
        'synthetic_controls': ['oracle_s0', 'shuffled_s0', 'wrong_s0'],
        'real_teachers': ['a14_s0', 'a14_s1', 'a14_s2', 'a33_s1'],
    }
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    return record


def stage(name, arguments):
    verify_registration()
    start = time.perf_counter()
    print(f'STAGE {name}', flush=True)
    logdir = RESULTS / 'logs'
    logdir.mkdir(exist_ok=True)
    logfile = logdir / f'{name}.log'
    if logfile.exists():
        logfile = logdir / f'{name}_{int(time.time())}.log'
    with logfile.open('w', encoding='utf-8') as output:
        proc = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            output.write(line)
            output.flush()
            print(line, end='', flush=True)
        code = proc.wait()
    print(f'END {name} code={code} seconds={time.perf_counter() - start:.2f}', flush=True)
    if code:
        raise RuntimeError(f'{name} failed; inspect {logfile}')


def finalize():
    import numpy as np
    verify_registration()
    archive = json.loads((RESULTS / 'verified/archive.json').read_text(encoding='utf-8'))
    runtime = json.loads((RESULTS / 'verified/runtime_verdict.json').read_text(encoding='utf-8'))
    by_control = {teacher: [d for d in archive['diagnostics'] if d['teacher'] == teacher]
                  for teacher in ('oracle', 'shuffled', 'wrong')}
    controls = {
        'exact_teacher_eight_verified': len(by_control['oracle']) == 8 and all(
            d['candidate_accepted'] and d['verification']['exhaustive_pass'] for d in by_control['oracle']),
        'shuffled_none_accepted': len(by_control['shuffled']) == 8 and not any(
            d['candidate_accepted'] for d in by_control['shuffled']),
        'wrong_teacher_eight_candidates_all_rejected': len(by_control['wrong']) == 8 and all(
            d['candidate_accepted'] and not d['verification']['exhaustive_pass'] for d in by_control['wrong']),
        'no_synthetic_control_admitted': not any(d['is_control'] and d['admitted_real_archive']
                                                for d in archive['diagnostics']),
    }
    data_audit = {}
    reference_inputs = None
    for folder in sorted((RESULTS / 'harvest').iterdir()):
        if not folder.is_dir():
            continue
        report = json.loads((folder / 'programs.json').read_text(encoding='utf-8'))
        for line in (folder / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            if sha(folder / relative) != expected:
                raise RuntimeError(f'Harvest checksum mismatch: {folder / relative}')
        with np.load(folder / 'queries.npz') as query:
            disjoint = True
            unique = True
            for cal, val in zip(query['x_cal'], query['x_val']):
                cal_set, val_set = set(map(tuple, cal)), set(map(tuple, val))
                disjoint &= not (cal_set & val_set)
                unique &= len(cal_set) == 2048 and len(val_set) == 2048
            these = (query['x_cal'].copy(), query['x_val'].copy())
            if reference_inputs is None:
                reference_inputs = these
            matched = all(np.array_equal(a, b) for a, b in zip(reference_inputs, these))
        data_audit[report['origin']] = {'cal_val_disjoint': bool(disjoint),
            'unique_fixed_counts': bool(unique), 'inputs_match_other_teachers': bool(matched)}
    acquisition = archive['acquisition_gate']
    union = archive['union_eight_tokens_gate']
    initial = archive['initial_a33_closure']
    primary = archive['primary']
    gap = not initial['contains_reversal'] and all(
        token in primary and primary[token]['teacher'] == 'a14' for token in ('P1', 'P3'))
    gates = dict(controls,
        query_integrity=all(all(row.values()) for row in data_audit.values()),
        two_a14_origins_seven_or_more=acquisition, real_union_eight=union,
        reversal_gap_filled_from_independent_teacher=gap,
        all_depths_exact=bool(runtime.get('all_lengths_exact', False)),
        one_operator_payload_at_most_66_bytes=bool(runtime.get('payload_capacity_gate', False)),
        no_neural_weights_in_runtime=bool(runtime.get('no_neural_weights_gate', False)))
    rig = (controls['exact_teacher_eight_verified'] and controls['no_synthetic_control_admitted']
           and gates['query_integrity'])
    if not rig:
        outcome = 'INVALID_RIG'
    elif all(gates.values()):
        outcome = 'PORTABLE_SKILL_ACQUISITION_SUPPORTED_IN_ATOM_WORLD'
    elif any(archive['teacher_certified_counts'].get(f'a14_s{s}', 0) for s in (0, 1, 2)):
        outcome = 'PARTIAL_ACQUISITION_ONLY'
    else:
        outcome = 'EXTRACTION_FALSIFIED_AT_THIS_BUDGET'
    summary = {'outcome': outcome, 'gates': gates,
        'teacher_certified_counts': archive['teacher_certified_counts'],
        'initial_permutation_closure_count': initial['count'],
        'final_permutation_closure_count': archive['final_real_archive_closure']['count'],
        'archive_payload_bytes_all_origin_variants': archive['archive_payload_bytes'],
        'primary_eight_payload_bytes': sum(x['payload_bytes'] for x in primary.values()),
        'data_audit': data_audit, 'runtime': runtime,
        'scope': 'Fixed separable grammar, six-digit domain, independent exhaustive specification; not general AI validation.',
        'completed_utc': datetime.now(timezone.utc).isoformat()}
    (RESULTS / 'VERDICT.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    files = sorted(p for p in RESULTS.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
    (RESULTS / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(RESULTS).as_posix()}\n'
                                             for p in files), encoding='utf-8')
    print(json.dumps({'outcome': outcome, 'gates': gates,
                      'teacher_certified_counts': summary['teacher_certified_counts']}, indent=2), flush=True)


def main():
    record = register()
    print(json.dumps({'registration_utc': record['registered_utc'],
                      'protocol_sha256': record['source_sha256']['AtomV3/PROTOCOL.md']}), flush=True)
    # Exact and corrupt synthetic controls run before neural teachers, but are
    # result-bearing and subject to the same frozen source manifest.
    for teacher, seed in [('oracle', 0), ('shuffled', 0), ('wrong', 0),
                          ('a33', 1), ('a14', 0), ('a14', 1), ('a14', 2)]:
        if not (RESULTS / 'harvest' / f'{teacher}_s{seed}' / 'programs.json').exists():
            stage(f'harvest_{teacher}_{seed}', [str(HERE / 'harvest.py'),
                  '--teacher', teacher, '--seed', str(seed), '--query-seed', '20260912'])
    if not (RESULTS / 'neural_baseline.json').exists():
        stage('neural_baseline', [str(HERE / 'neural_baseline.py')])
    # Verification and deployment runner is kept separate from the compiler.
    if not (RESULTS / 'verified/runtime_verdict.json').exists():
        stage('verify_archive', [str(HERE / 'verify_archive.py'),
              '--harvest-root', str(RESULTS / 'harvest'),
              '--output-dir', str(RESULTS / 'verified')])
    finalize()
    print('All registered experimental stages completed.', flush=True)


if __name__ == '__main__':
    main()
