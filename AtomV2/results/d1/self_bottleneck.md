# D1 addendum: self decode->re-encode bottleneck at the token boundary

Read-only follow-up to the D1 verdict. Mechanism: run token 1 normally, decode
the model's OWN boundary state with its frozen decoder, re-encode the digits
with its own encoder (canonical state), continue token 2 via
execute_from_state. No oracle, no ground truth, no training. "Repair without
the oracle" - predicted by D1's domain analysis (canonical states are
universally in-domain via every program's token-1 role).

| run | trained raw->bn | L1 raw->bn | L2 raw->bn | L3 raw->bn |
|---|---|---|---|---|
| A6_s0  | 0.711->0.535 | 0.028->0.276 | 0.231->0.919 | 0.000->0.735 |
| A6_s1  | 0.981->0.972 | 0.333->0.967 | 0.275->0.972 | 0.000->0.947 |
| A6_s2  | 0.413->0.254 | 0.010->0.135 | 0.105->0.516 | 0.000->0.547 |
| A14_s0 | 0.952->0.941 | 0.565->0.953 | 0.244->0.928 | 0.000->0.969 |
| A14_s1 | 0.988->0.987 | 0.432->0.986 | 0.394->0.983 | 0.000->0.993 |
| A14_s2 | 0.723->0.565 | 0.113->0.280 | 0.252->0.953 | 0.001->0.772 |
| A16_s0 | 0.957->0.961 | 0.528->0.968 | 0.274->0.960 | 0.000->0.980 |
| A16_s1 | 0.975->0.977 | 0.367->0.983 | 0.278->0.978 | 0.000->0.991 |

Readings:
- In every healthy run (A14 s0/s1, A16 s0/s1, A6 s1) the bottleneck takes L3
  from 0.000 to 0.95-0.99 and L1/L2 to 0.93-0.99 while leaving trained
  accuracy intact. The dax wall is an interface-dialect problem, fully
  bypassed by translating every handoff through the canonical channel.
- Where it fails, it fails exactly as P-B predicted: weak seeds (A6 s0/s2,
  A14 s2) have sloppy boundary decodes (0.50-0.75), and the bottleneck
  propagates decode errors the co-adapted continuation used to absorb.
  Accuracy is gated by boundary-decode quality, nothing else.
- Numbers match oracle canonical repair almost exactly (A14 s1: repair 0.996
  vs self-bottleneck 0.993): the model needed no ground truth, only the
  discipline of speaking canonically at boundaries.
- This does NOT answer H1's emergent-language question - it imposes the
  symbol space as the interface rather than discovering a latent one. As a
  mechanism finding it is decisive: per-program computation was correct all
  along; composition fails only at the handoff dialect. E6 design space:
  make the bottleneck trainable (straight-through) or use it as a teacher
  for latent-space closure.
