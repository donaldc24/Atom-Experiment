# Atom Playground

Read-only CPU inference over completed AtomV2 checkpoints. The backend imports
the existing harness model and canonical operation algebra; it does not train,
compute gradients, or write to `runs/`.

From `AtomV2/`:

```powershell
..\.venv\Scripts\pip.exe install -r demo\requirements.txt
..\.venv\Scripts\python.exe -m uvicorn demo.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. API documentation is available at `/docs`.

Run the read-only acceptance battery with:

```powershell
..\.venv\Scripts\python.exe -m demo.verify
```

The canonical-channel mode runs each learned token-program, decodes the model's
own six-digit boundary prediction, and re-encodes that prediction before the
next token. Raw mode runs the same programs but passes the latent state through
unchanged. Ground truth is independently derived from the harness's canonical
triples and is never supplied to the model.

The execution-budget control is independent of that boundary channel. The
default applies the checkpoint's complete learned micro-step program. The
single-atom option shadow-runs that router, chooses a strict majority when one
route receives more than half the votes (2 of 3 on the standard checkpoints),
or otherwise chooses the hard selection with the highest per-step soft
confidence, then applies the winner once to the incoming state.

The confidence-only single-atom variant ignores vote counts, takes the hard
selection with the largest soft weight at any shadow micro-step, and applies
that route once.

The repeated-winner variant preserves the winning route's multiplicity while
dropping the other selections: a shadow route A1, A1, A2 applies A1 twice. If
no route repeats, it applies the highest-confidence hard selection once.

Clicking a routed or used atom opens its existing final-panel standalone
surface scores. Each score is the exact-match accuracy of one application of
that atom from the canonical code against one P-operation target; the scores
are independent measurements and are not normalized mixture proportions.

The P Routing Atlas tab requires no chain. It runs P1 through P8 independently
as singleton programs from one shared six-digit input and shows every hard
micro-step selection. Because the composer is state-dependent, the atlas is an
input-conditioned trace rather than a claim that each surface op has one fixed
route for all inputs.

The Unique Atom Outcomes tab uses those same P1-P8 singleton routes, removes
duplicate atom IDs within each operation, and independently applies every
remaining atom once to the shared canonical input. It shows each one-atom
decode against that surface operation's truth; PASS is omitted because it is a
route, not an atom.

The Atlas Survey tab repeats that deduplicated one-atom analysis over a
reproducible random input batch and every healthy demo checkpoint (non-smoke
A14/A16 with seen accuracy at least 90%). It reports per-checkpoint, per-op,
per-atom exact-match accuracy, selection frequency, and soft-confidence
distributions overall and by micro-step. The full report, including the random
inputs, can be downloaded as JSON or flattened CSV. Downloads are assembled in
the browser; the server does not write report artifacts or modify `runs/`.
