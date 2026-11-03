# Exact-memory landing statistic (item 4, 2026-11-03) - EXPLORATORY / POST HOC, not pre-registered

Definition: docs/theory/exact_lambda/exact_lambda.tex (PDF out/theory/exact_lambda/exact_lambda.pdf), written before the
computation. Lambda_mem(x0) = log(max over the landing window of |p_k| / max over the window of |c_k|), p_k = d loss / d eps_k
(exact price, all GRU memory paths, one reverse pass per start), c_k the direct one-step source; Lambda*_mem = max over the 1000
training starts. Lambda_scal = the same with the scalar price recursion p_k = c_k + m_k p_{k+1} (direct gain, no memory).
Code theory/t2_exact.py, all 180 runs, CPU about 20 min; out/theory/t2_exact.csv, t2_exact_with_labels.csv.
The pre-registered scalar Lambda* (12.1) stays the primary statistic.

## Results
| statistic | threshold (rule) | traps flagged (of 20) | false alarms (of 160) |
|---|---|---|---|
| scalar Lambda* (pre-registered) | 12.1 | 17 | 0 |
| Lambda_scal (same normalisation as Lambda_mem) | 11.4 (geometric midpoint on LN MSE 3000) | 17 | 0 |
| Lambda_mem (exact memory) | pre-specified rule does not apply: on LN MSE 3000 the largest non-trapped value (16.47, seed 7 rep 1) exceeds the smallest trapped one (11.74, seed 8 rep 2) | - | - |
| Lambda_mem, post hoc | 16.55 (midpoint of the closest gap giving 11/12 hits and 0/18 false alarms on LN MSE 3000) | 14 | 0 |

Missed with the exact statistic at 16.55: LN MSE s8 r2 (11.7; scalar Lambda* 89), slope-from-0 s9 r0 (7.0; 84), s9 r2 (7.0; 45),
and the three traps the scalar statistic also misses: slope-from-0 s4 r1 (2.3; scalar 2.5), s9 r1 (4.1; 3.4), s3 r2 (3.6; 6.5).
Non-trapped maxima per family: LN MSE 3000 16.5, slope from 0 9.4, MSE 12000 4.2, two-stage 4.1, no norm 2.3, static 2.0.

Check against the slide 17 background numbers (price at step 0 of the start with the given x0, repeat 0): LN MSE seed 2,
x0 = +0.022: exact -3.3e33, scalar 2.9e48 (quoted 3e33 and 6e48: same orders). LN MSE seed 8, nearest training start
x0 = 2.368: exact 2.9, scalar 1.4e4 (quoted 3e10 and 6e5: NOT reproduced; the earlier number may come from another start or
repeat; not resolved).

## What it means
- Including the memory makes the statistic worse, not better. The normalisation is not the cause (the scalar statistic with
  the same normalisation keeps 17/20, 0/160); the memory is: on several trapped runs the reference GRU's memory cancels much
  of the amplification that the direct-gain multiplier sees (seed 8 rep 2: 88 -> 11.7).
- This fits the (untested) reading in NOTES_theory.md: DPC is trapped while it visits GRU policies WITHOUT the reference's
  memory compensation, which is what the direct-gain statistic describes. It argues for presenting the scalar Lambda* as a
  statistic of the policy class near the good controller, not of the good controller itself. A direct test would compute
  Lambda_mem along DPC training checkpoints (needs saved checkpoints).
- The three missed slope-from-0 traps are low under every version (2.3-6.5): whatever traps those runs is not visible at the
  reference controller, exact memory or not (consistent with the earlier straight-line finding that their spikes lie on the way
  in, not at the end point).
- Recommendation: keep the scalar Lambda* as the paper's statistic; mention Lambda_mem in a footnote or appendix as a negative
  check ("the exact costate of the reference controller does not predict better").
