# Held-out seeds 10-29: Part 1 (LQR) replication (Q6c)

Written 2026-11-04. Pre-registration: prereg/PREREG_heldout.md (23b5835), committed before any surrogate of seeds 10-29
existed. Surrogates trained locally on CPU, float64, exactly as seeds 0-9 (tools/train_surrogates_local.py, folder
data/dpc_out_s10_29). Timing under load: LN MSE 3000 26 min and static norm 28 min for 20 seeds; two-stage 47 min,
including the labels. All six families would have taken well over 3 h, so only LN MSE 3000, two-stage and static norm were
trained, as the pre-registration allows. LQR study: companion/run_lqr_study.py with LK_SEEDS=10..29 (20 seeds x 11 setpoints x
9 starts; 1760 runs per family and controller with |x0 - r| >= 0.5). Scoring: analysis/heldout_report.py, outputs in out/heldout/.

## Rates against the registered prediction intervals
| family | controller | metric | seeds 10-29 (%) [95 % seed-bootstrap CI] | seeds 0-9 (%) | registered 95 % PI | inside |
|---|---|---|---|---|---|---|
| LN MSE 3000 | LQR fixed | failed | 36.2 [30.2, 42.1] | 26.5 | [18.2, 34.4] | NO |
| LN MSE 3000 | LQR fixed | catastrophic | 17.8 [13.1, 22.7] | 10.8 | [3.1, 19.4] | yes |
| LN MSE 3000 | LQR sched | failed | 29.1 [23.7, 34.9] | 20.5 | [15.2, 25.9] | NO |
| LN MSE 3000 | LQR sched | catastrophic | 8.5 [5.6, 12.1] | 3.4 | [1.6, 5.3] | NO |
| LN two-stage | LQR fixed | failed | 6.2 [1.7, 11.9] | 5.7 | [0.5, 13.7] | yes |
| LN two-stage | LQR fixed | catastrophic | 1.1 [0.0, 3.2] | 0.9 | [0.0, 3.6] | yes |
| LN two-stage | LQR sched | failed | 6.9 [1.9, 13.3] | 3.5 | [0.0, 9.7] | yes |
| LN two-stage | LQR sched | catastrophic | 1.3 [0.0, 3.3] | 0.1 | [0.0, 0.5] | NO |
| static norm | LQR fixed | failed | 1.1 [0.3, 2.2] | 0.6 | [0.0, 1.8] | yes |
| static norm | LQR fixed | catastrophic | 0.0 [0.0, 0.0] | 0.0 | [0.0, 0.0] | yes |
| static norm | LQR sched | failed | 3.3 [0.4, 8.3] | 0.8 | [0.0, 1.9] | NO |
| static norm | LQR sched | catastrophic | 1.9 [0.0, 5.8] | 0.0 | [0.0, 0.0] | NO |

## Outcomes
| id | outcome | numbers |
|---|---|---|
| H1 (rates in their 95 % PI; held if at most 2 of the scored cells outside) | failed | 6 of 12 outside. All six are ABOVE the interval: LN MSE 3000 fixed failed 36.2 % (PI [18.2, 34.4]), rescheduled failed 29.1 % ([15.2, 25.9]), rescheduled catastrophic 8.5 % ([1.6, 5.3]); two-stage rescheduled catastrophic 1.3 % ([0, 0.45]); static norm rescheduled failed 3.3 % ([0, 1.9]) and catastrophic 1.9 % ([0, 0]) |
| H2 (fixed LQR: LN MSE 3000 > static norm, and its cell inside) | failed (by the second condition) | the ordering holds strongly: 36.2 % vs 1.1 %; the LN MSE 3000 cell lies above its interval |
| H3 (every LN kink centre within 0.1 of (0, 0)) | failed | 4 of 40 outside, all slightly: LN MSE 3000 seed 28 (-0.046, +0.141), seed 29 (-0.006, +0.131); two-stage seed 10 (-0.071, -0.136), seed 28 (-0.053, +0.144) |

## What replicates and what does not
- The qualitative Part 1 result replicates on 20 new seeds and is stronger:
  - Fixed LQR fails on 36 % of LN MSE 3000 runs (catastrophic 18 %), against 1.1 % (0 %) for static norm.
  - Two-stage brings it down to 6.2 % (1.1 %), within its interval.
  - The kink geometry is the same: median r*_x 0.34 vs 0.26 on seeds 0-9; median peak input-slope error 20.3 B vs 18.7 B.
- The quantitative predictions fail: the 10 original seeds understated the seed-to-seed spread and the level. Even the
  two-stage intervals, the wider choice registered, were too narrow for the LN MSE 3000 rates, and the rescheduled-LQR
  catastrophic rates of the two kink-free or repaired families were not zero.
- "Every LN kink centre within 0.1 of the origin" (CLAUDE.md 1.2, measured on seeds 0-9) does not hold on 20 more seeds:
  4 of 40 lie at 0.13-0.14 along u. It should read "within 0.15 of (0, 0) on all 30 + 20 seeds" or similar (a change of a
  stated fact, your decision).
- Not trained (no outcome): LN MSE 12000, LN slope from step 0, no norm. Run 7 trains all six families of seeds 10-29 on Colab.
