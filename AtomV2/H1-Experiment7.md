# H1 Experiment 7: Simplify Before Translating

## Can one-step atoms become directly reusable and composable without an explicit canonicalizer?

## Why this experiment exists

D1, D2, and D3 together isolate a persistent interface problem in the current three-microstep architecture.

The existing model can learn useful specialist atoms and routed computations. In healthy runs, a first computation often contains the correct semantic result, and decoding then re-encoding that result through the model's shared encoder restores downstream composition almost completely.

However, direct atom-to-atom composition remains poor.

The current architecture contains two major possible causes that have not yet been isolated:

1. **Three routing microsteps per task token.**
   The effective computational unit may be a state-dependent routed sequence of several atoms rather than one independently reusable atom.

2. **An overpermissive residual state interface.**
   The 384-dimensional residual state can preserve the transformed answer, original input, scratch information, and private routing conventions simultaneously.

Before introducing a learned canonical latent interface, Experiment 7 asks whether the interface problem can be removed by simplifying the architecture itself.

The experiment therefore proceeds in stages:

```text
remove microstep depth
        ↓
test direct atom composition
        ↓
restrict interface bandwidth
        ↓
remove residual carry-through
        ↓
test direct atom composition again
```

The experiment is intentionally agnostic about what atoms represent.

An atom may implement:

* a hidden sub-operation,
* a complete surface operation,
* some other learned transformation,
* or any reusable computational role discovered by training.

No semantic granularity is preferred.

The criterion is functional:

> Does the architecture discover independently addressable components that can be reused and directly composed?

This is the property required by the eventual systems objective: storing a large atom library outside peak resident memory and loading only the subset required by the current computation.

---

# Primary question

> Does reducing each task token to one routing decision produce independently reusable atoms that compose directly through the learned latent state?

# Secondary questions

If one-step atoms still fail to compose:

1. Does reducing state bandwidth force a more interoperable representation?
2. Does removing the residual carry path force atoms to emit cleaner states?
3. Do bandwidth restriction and replacement updates work only in combination?
4. If all of these fail while the component computations remain competent, is an explicit shared canonical interface justified?

---

# What Experiment 7 is not testing

Experiment 7 does **not** ask whether atoms recover the seven hidden sub-operations.

It does not require human-interpretable atom semantics.

It does not require that every task use exactly one unique atom.

It does not optimize actual CPU, GPU, or disk paging yet.

It does not introduce a learned translator or canonicalizer.

It asks only whether a simpler architectural contract is enough to create reusable, composable computational units.

---

# Registered base: A6

All Experiment 7 arms descend from A6.

A6 is used instead of A9, A14, or A16 because it contains the mature bounded cosine routing stack without later interface-specific pressures.

Registered inherited properties:

```text
router                  cosine
router alpha            3.6838542861871852
router Gumbel sigma     1.082543687...
tau_backward            1.0
lambda_use              0
state width              384
atom update              residual
training steps           20,000
seeds                    0, 1, 2
```

Original A6 differs from the first E7 arm only in:

```text
microsteps per token     3
```

A6 remains the historical reference and is never rerun unless an implementation-equivalence check requires it.

A14 remains the strongest mature treatment reference but is not used as the causal base for E7.

---

# Stage 1: Remove the microstep confound

## A18: One-step A6

A18 is identical to A6 except:

```text
microsteps per token     1
```

There is exactly one routing decision for each surface-task token.

For a two-token program:

```text
Pa → Pb
```

execution becomes:

```text
encoder(x)
    ↓
router(Pa)
    ↓
one selected atom
    ↓
router(Pb)
    ↓
one selected atom
    ↓
decoder
```

The composer remains state-dependent.

The router remains hard top-1.

The atom library remains unchanged.

The state width remains 384.

Atoms remain residual.

No interface noise, sandbox loss, producer loss, rent, canonicalization, intermediate supervision, or semantic target is introduced.

## Why this arm matters

With three microsteps, the actual reusable unit may be:

```text
token + router + atom i + atom j + atom k
```

A18 removes this ambiguity.

If A18 learns successfully, then every task-token computation is mediated by exactly one selected atom.

The atom therefore becomes the natural pageable computational unit.

---

# A18 run order

## Step 1: mechanism screen

Run:

```text
A18 seed 1
```

Seed 1 is preregistered before observing A18 because the corresponding A6 run is a healthy reference and therefore provides an interpretable matched comparison.

Run the full 20,000-step budget unless:

* loss becomes non-finite,
* parameters become non-finite,
* or another implementation failure invalidates the run.

Poor optimization is a result, not an implementation failure.

## A18 health criterion

A18 seed 1 is considered sufficiently healthy for factorization diagnostics if:

```text
seen hard accuracy >= 80%
```

and the model shows clear competence on trained singleton operations.

The exact singleton panel already used by the harness is reported in full.

The 80% threshold is not a claim of parity with A6. It exists only to prevent interpreting composition diagnostics from a model that never learned its component tasks.

## If A18 seed 1 passes

Run:

```text
A18 seed 0
A18 seed 2
```

A18 then becomes the three-seed one-step baseline.

## If A18 seed 1 fails

Do not proceed directly to interface conclusions.

Record:

```text
ONE-STEP CAPACITY / OPTIMIZATION FAILURE
```

The result means the matched one-step architecture does not provide enough usable computation under the current atom design.

It does **not** prove that one routing decision per module is impossible.

A separate registered capacity-rescue experiment should then increase computation *inside* each pageable atom while keeping one routing decision per token.

That rescue is not folded into E7 after seeing results.

---

# Stage 1 measurements

A18 receives the complete standard evaluation panel plus the relevant D1, D2, and D3 diagnostics.

## Task behavior

Report:

* Seen hard accuracy
* L1 hard accuracy
* L2 hard accuracy
* L3 hard accuracy
* singleton accuracy
* route usage
* atom utilization
* pass usage
* deterministic routing diversity
* stochastic routing diversity

## D1-style boundary panel

For two-token programs, measure after token 1:

* boundary answer decodability,
* boundary distance to canonical state,
* raw continuation accuracy,
* self-bottleneck continuation accuracy.

The self-bottleneck remains:

```text
learned boundary state
        ↓
frozen decoder
        ↓
predicted intermediate digits
        ↓
frozen encoder
        ↓
canonical state
        ↓
continue normally
```

This is diagnostic only.

It is never used during training.

## D2-style atom audit

Because A18 permits exactly one atom call per token, D2 becomes substantially easier to interpret.

For every atom:

1. Evaluate isolated behavior from canonical encoder states.
2. Identify any stable function it performs under the benchmark's observable transformation family.
3. Do not privilege hidden sub-operations over surface operations.
4. Record all sharp functional matches.
5. Test the same atom across states produced by other atoms.
6. Directly chain discovered competent atoms with the composer removed.

For discovered components:

```text
canonical input
    ↓
Atom i
    ↓
Atom j
    ↓
decoder
```

Compare against:

```text
canonical input
    ↓
Atom i
    ↓
decoder → encoder
    ↓
Atom j
    ↓
decoder
```

The first is **raw composition**.

The second is **canonicalized composition**.

## D3-style state-content panel

Measure linear recoverability of:

* the transformed answer,
* the original input,
* any existing registered state-content targets.

This tests whether one-step execution alone reduces the latent "stack" previously observed.

---

# Primary E7 composition metric

For atom pairs where:

1. both component atoms are individually competent, and
2. canonicalized chaining achieves at least 80% accuracy,

define:

```text
Interface Closure Ratio
    =
raw direct-chain accuracy
/
canonicalized direct-chain accuracy
```

This metric asks:

> Given that both components know how to perform their jobs, how much of that capability survives direct latent composition?

Interpretation:

```text
>= 0.80    strong direct closure
0.30-0.80  partial closure
< 0.30     severe interface failure
```

Also report raw percentages.

The ratio must never conceal low absolute accuracy.

---

# Stage 1 outcome bins

## MICROSTEPS WERE THE MAIN CONFOUND

Register if A18 is healthy and direct composition improves dramatically relative to the three-step A6/D2 behavior.

Strong evidence requires:

```text
Interface Closure Ratio >= 0.80
```

on a meaningful set of discovered competent atom pairs.

If reached, stop architectural deprivation experiments temporarily.

The next priority becomes replication and actual lazy-loading memory tests.

## MICROSTEPS HELPED BUT DID NOT SOLVE IT

A18 is healthy and direct composition improves substantially but remains:

```text
Interface Closure Ratio < 0.80
```

Proceed to Stage 2.

## MICROSTEPS NOT CAUSAL

A18 is healthy but direct composition remains near the prior floor while canonicalized composition remains strong.

Proceed to Stage 2.

## ONE-STEP CAPACITY FAILURE

A18 does not learn the underlying tasks sufficiently.

Stop E7 before making interface claims.

---

# Stage 2: Restrict the interface

Stage 2 begins only if A18 is healthy and direct atom composition remains materially impaired.

The purpose is to test two architectural explanations:

```text
excess bandwidth
residual preservation
```

using a registered 2 × 2 design.

---

# Stage 2 arms

| Arm | Microsteps | State width | Atom update | Purpose               |
| --- | ---------: | ----------: | ----------- | --------------------- |
| A18 |          1 |         384 | residual    | control               |
| A19 |          1 |          64 | residual    | bandwidth starvation  |
| A20 |          1 |         384 | replacement | residual-erasure test |
| A21 |          1 |          64 | replacement | combined pressure     |

Everything else remains inherited from A6/A18.

No learned canonicalizer is introduced.

---

# A19: Bandwidth starvation

Change only:

```text
state width: 384 → 64
```

All modules whose dimensions depend mechanically on state width are adjusted accordingly.

The router definition, routing-key dimensionality, exploration regime, task structure, atom count, losses, and training budget otherwise remain unchanged.

## Interpretation

The intervention does **not** assume that 64 real-valued dimensions are literally incapable of encoding the original input plus transformed result.

The claim is weaker:

> Lower state dimensionality increases pressure to represent only information useful for future computation.

The registered mechanistic prediction is:

```text
original-input recoverability decreases
answer recoverability remains high
direct atom composition improves
```

A reduction in original-input recoverability without an improvement in composition is not sufficient.

---

# A20: Replacement atoms

Keep:

```text
state width = 384
```

but change atom application from the existing residual form:

```text
s_new = LayerNorm(s + F_i(s))
```

to:

```text
s_new = LayerNorm(F_i(s))
```

for atom routes.

The selected atom must therefore generate the complete outgoing state rather than adding a delta to an automatically preserved input.

Pass remains the explicit identity/no-operation route.

No other change is made.

## Interpretation

Replacement removes the architectural shortcut by which every atom automatically carries its input forward.

It does not mathematically guarantee erasure.

An atom remains free to relearn identity-preserving behavior if task optimization rewards it.

The question is empirical:

> Does removing free residual preservation make independently learned atom outputs more interoperable?

---

# A21: Narrow replacement state

Combine both interventions:

```text
microsteps = 1
state width = 64
atom update = replacement
```

A21 is the strongest architectural attempt in E7 to obtain a shared direct interface without introducing a translator.

---

# Stage 2 screening order

Run one preregistered seed first:

```text
A19 seed 1
A20 seed 1
A21 seed 1
```

A18 seed 1 is the matched control.

All receive the same full training budget and diagnostic panel.

No treatment is modified after observing another treatment.

---

# Stage 2 health criterion

An intervention is considered task-competent if:

```text
seen hard accuracy >= 80%
```

and component/singleton competence remains sufficient for the D2 direct-chain audit.

A treatment that destroys task competence cannot be credited with improving interoperability.

For example:

```text
raw chain = 90%
```

is meaningless if individual atoms no longer perform the required component transformations.

---

# Stage 2 predictions

## Bandwidth hypothesis

If excessive state capacity creates private latent stacks:

```text
A19 > A18
```

on Interface Closure Ratio, with reduced stale/original recoverability.

## Residual hypothesis

If automatic input carry-through creates the private representation:

```text
A20 > A18
```

on Interface Closure Ratio.

## Interaction hypothesis

If both mechanisms jointly sustain the private dialect:

```text
A21 > A19
A21 > A20
A21 >> A18
```

on direct composition.

---

# Stage 2 replication rule

Do not automatically spend three seeds on all treatment arms.

## If one treatment reaches strong closure

If A19, A20, or A21 seed 1 achieves:

```text
Interface Closure Ratio >= 0.80
```

while remaining task-competent, designate it the candidate winner.

Run the corresponding:

```text
seed 0
seed 2
```

A18 already provides the three-seed control.

The candidate treatment must reproduce the qualitative effect across seeds before the mechanism is considered established.

## If no arm reaches strong closure but one clearly improves composition

Replicate the strongest treatment across seeds 0 and 2.

Do not select solely by L1 or seen accuracy.

Selection is based primarily on:

1. component competence,
2. direct-chain accuracy,
3. Interface Closure Ratio,
4. relevant D3 state-content changes.

## If all three treatments fail

Replicate:

```text
A21 seed 0
A21 seed 2
```

provided A21 seed 1 is task-competent.

A21 is chosen because it removes both registered architectural suspects simultaneously.

If healthy A21 runs across all three seeds still exhibit:

```text
canonicalized composition >= 80%
Interface Closure Ratio < 0.30
```

the architectural simplification route is considered to have failed its strongest registered test.

At that point Experiment 7 terminates.

---

# Experiment 7 final outcome bins

## DIRECT FACTORIZATION

Healthy one-step atoms compose directly with:

```text
Interface Closure Ratio >= 0.80
```

without bandwidth or replacement intervention.

Conclusion:

> Three-microstep routed programs were a major source of the apparent interface pathology. A one-route-per-module architecture can discover directly composable components.

## BANDWIDTH-CLOSED

A19 succeeds and reproduces.

Conclusion:

> Restricting interface capacity is sufficient to induce a substantially shared atom language.

## REPLACEMENT-CLOSED

A20 succeeds and reproduces.

Conclusion:

> Automatic residual preservation was a major cause of private state conventions.

## JOINT-CLOSED

Only A21 succeeds and reproduces.

Conclusion:

> Direct atom closure requires both reduced interface capacity and removal of the residual carry path.

## PARTIAL CLOSURE

One or more interventions substantially improve direct composition but do not reach strong closure.

Conclusion:

> Architectural constraints move the interface in the correct direction but are insufficient by themselves.

The next experiment should investigate a minimal shared projection or learned canonical latent interface.

## TRANSLATOR JUSTIFIED

Register if healthy A21 runs reproduce the following pattern:

```text
component competence high
canonicalized composition high
raw direct composition poor
Interface Closure Ratio < 0.30
```

Conclusion:

> The model possesses reusable component computations, but architectural simplification alone does not force a sufficiently shared latent interface. An explicit learned canonicalization mechanism is now justified.

This is the stop condition for further starvation/erasure variants.

Do not continue inventing increasingly arbitrary restrictions after this outcome.

## CAPACITY-LIMITED

One-step and/or 64-dimensional architectures fail to learn the underlying component computations.

Conclusion:

> The intervention removed required computational capacity, so the factorization question is unresolved under that architecture.

Do not interpret low composition as evidence for a translator.

---

# Read-only success criteria for reusable atoms

Experiment 7 does not require semantic correspondence to any particular benchmark decomposition.

A component counts as reusable if the same stored atom can successfully participate in multiple computational contexts.

Evidence may include:

* the same atom performing its learned function across different inputs,
* the same atom participating successfully with multiple predecessor atoms,
* the same atom participating successfully with multiple successor atoms,
* duplicate atoms being substitutable,
* direct composer-free chaining,
* repeated use of the same parameter block across different surface programs.

The strongest evidence is causal direct composition:

```text
Atom A output
      ↓
Atom B input
```

with no task-specific adapter between them.

---

# What would count as the strongest E7 result?

The ideal result is:

```text
microsteps = 1
small reusable atom library
high component accuracy
high raw composer-free pair accuracy
high L1/L2/L3 composition
little or no canonical repair gap
same atom reused in multiple contexts
```

The atom's human interpretation is irrelevant.

It only needs to be:

```text
independently addressable
functionally useful
reusable
composable
```

---

# Relationship to peak resident memory

Experiment 7 does not yet claim a resident-memory reduction.

It establishes whether the unit intended for future paging is computationally legitimate.

The eventual systems architecture is:

```text
always resident:
    encoder / embedding substrate
    router
    routing keys
    shared interface
    current latent state
    small runtime machinery

offloaded:
    large atom library

execution:
    route
      ↓
    load selected atom
      ↓
    execute
      ↓
    evict or cache
      ↓
    route again
```

Moving from three microsteps to one also reduces the number of potential atom loads per logical operation from three to one.

If E7 produces composable atoms, the next experiment should implement true selected-atom-only execution and measure:

* peak GPU-resident parameter memory,
* peak process RSS,
* transfer volume,
* cache hit rate,
* latency,
* throughput,
* model quality versus the fully resident implementation.

That memory experiment should not begin until the functional unit being paged has been established.

---

# Registered run budget

Minimum path if A18 immediately succeeds:

```text
A18 s1
A18 s0
A18 s2

3 runs
```

Typical path if Stage 2 is required:

```text
A18 s0/s1/s2       3
A19 s1             1
A20 s1             1
A21 s1             1
winner s0/s2       2

8 result-bearing runs
```

Strong negative path:

```text
A18 s0/s1/s2       3
A19 s1             1
A20 s1             1
A21 s0/s1/s2       3

8 result-bearing runs
```

No arm receives extra seeds merely because its seed-1 result looks interesting outside the registered selection rules.

---

# Implementation gates

Before launching result-bearing runs:

1. `microsteps = 3` under the refactored implementation must reproduce the existing A6 execution path.
2. A18 must make exactly one route decision per token.
3. A two-token A18 program must therefore execute exactly two token-level routing decisions.
4. No hidden extra atom call may occur inside evaluation or routing.
5. D1/D2/D3 diagnostics must remain read-only.
6. A19 must change state width everywhere consistently.
7. A20 replacement mode must contain no residual addition around the atom output.
8. Pass behavior must remain explicit and unchanged except for unavoidable dimensional bookkeeping.
9. No canonicalization, decoder-reencode path, ground-truth intermediate, or semantic supervision may enter training.
10. Random streams and seed pairing must remain isolated as in prior certified experiments.

Any implementation-gate failure invalidates the run.

Optimization failure is reported as a result.

---

# Reporting order

For every arm, report in this order:

1. implementation validity,
2. task competence,
3. singleton/component competence,
4. raw direct composition,
5. canonicalized direct composition,
6. Interface Closure Ratio,
7. D3 answer recoverability,
8. D3 original-input recoverability,
9. routing/utilization behavior,
10. seed-level results,
11. registered outcome-bin assignment.

Do not lead with L3 alone.

Do not claim factorization merely from specialist atoms.

Do not claim failure merely because atoms do not match hidden primitives.

The central E7 question is:

> **Can independently addressable learned components execute successfully when directly chained through the model's own interface?**

---

# Registered decision tree

```text
A18 s1: A6 with microsteps = 1
                │
                ├── unhealthy
                │      ↓
                │   stop E7
                │   investigate per-atom capacity
                │
                └── healthy
                       ↓
                 A18 s0/s2
                       ↓
               direct closure strong?
                  /           \
                yes            no
                 │              │
                 │        A19/A20/A21 s1
                 │              │
                 │        any strong winner?
                 │           /        \
                 │         yes         no
                 │          │           │
                 │      replicate    replicate A21
                 │      winner       s0/s2
                 │          │           │
                 │          │      raw still fails
                 │          │      while canonical
                 │          │      remains strong?
                 │          │          /     \
                 │          │        yes      no
                 │          │         │        │
                 ↓          ↓         ↓        ↓
             composable   architectural   learned
             atoms found   solution       canonical
                                          interface
                                          justified
```

---

# Final standard

Experiment 7 succeeds scientifically even if every treatment fails.

A positive result would show that direct atom interoperability can emerge from a sufficiently constrained architecture without a translator.

A negative result would eliminate three major alternatives:

```text
microstep co-adaptation
excess state bandwidth
residual state preservation
```

and would provide a principled reason to introduce a learned canonical latent interface rather than adding one speculatively.

Either result sharply narrows the architecture required for scalable atom factorization.
