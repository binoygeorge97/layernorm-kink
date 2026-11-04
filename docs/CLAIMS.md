# Claims ledger (started 2026-11-03)

One row per claim we would put in the paper. CI = 95 % seed-bootstrap percentile interval (10 000 resamples of the 10 seeds,
a seed carries all its runs; `analysis/stats.py`, draws fixed by generator seed 20261103) unless stated. Status:
established / pending Run N (see docs/NEXT_COLAB.md) / corrected on <date> / exploratory (not pre-registered).
Data: B = 0.01 study, `MyDrive/dpc_out` (downloaded 3 Nov), `companion/lqr_study.csv` (regenerated bit-for-bit).

## Kink (Theorem 1)
| # | claim | number [CI] | produced by | status |
|---|---|---|---|---|
| K1 | LN(Ez + b) = V phi(w) exactly (closed-form reduction) | identity error <= 8.2e-15; trained surrogates through the reduction = F_point to 2.2e-15 | theory/verify_thm1.py -> out/logs/verify_thm1.log | established (proof in docs/theory/theory.tex) |
| K2 | every LN kink centre lies within 0.1 of (0, 0) | 10/10 seeds | surrogates.csv (zmin_x, zmin_u) | established |
| K3 | peak input-slope error of LN MSE 3000 surrogates | 6.7 to 201 x B, median 18.7 [9.6, 36.1] | surrogates.csv Ju_sup_rel; CI out of analysis/stats.py draws (computed in CLAIMS session) | established |
| K4 | longer MSE training shrinks but does not remove it | median 2.8 [1.1, 15.4] x B (MSE 12000), max 35 | surrogates.csv | established |
| K5 | wrong-sign arms reach 2 to 7 half-widths on 8 of 10 seeds (10 and 33 on seeds 1, 2); 56-99 % of the wrong-sign area outside the ellipse | per seed, no CI | theory/arm_reach.py | established; corrected 2026-11-03 (was "2 to 4") |
| K6 | setpoint input-slope error ~ 0.49 / d along the setpoint line (fit d^-0.98, about a decade of scatter) | 440 designs | phase_report.fig_setpoint_distance | established, wording softened 2026-11-03 |

## LQR (Part 1)
| # | claim | number [CI] | produced by | status |
|---|---|---|---|---|
| L1 | fixed LQR from LN MSE 3000 fails / is catastrophic | 26.5 [19.7, 33.1] % / 10.8 [4.2, 17.6] % of 880 runs | analysis/stats.py -> out/stats/lqr_rates_boot.csv | established |
| L2 | static norm, same accuracy | 0.6 [0.0, 1.5] % / 0.0 % | same | established |
| L3 | fixed-LQR failure peaks where the setpoint sits on the kink | r = 0: 66 [51, 83] %; r = +-2: 0-1 % | out/stats/lqr_setpoint_rates_boot.csv | established |
| L4 | path colours matter only for the rescheduled LQR (setpoint gain within 20 % of B) | fixed 1.0 / 1.0 / 0 / 0 %; rescheduled 1.1 [0.4, 2.1] / 9.0 [5.1, 15.1] / 27.3 [14.7, 38.5] / 40.0 [25.6, 55.3] % | out/stats/path_colour_boot.csv | established |
| L5 | the input slope carries the LQR damage (swap test, LN MSE 3000) | true B: 13.5 [8.8, 18.3] / 4.1 [0.9, 8.1] %; true A: 29.8 [21.5, 36.9] / 12.2 [4.9, 19.8] %; both true 2.3 [1.4, 3.3] / 0 % | analysis/swap_lqr_runs.py -> out/stats/swap_rates_boot.csv | established |
| L6 | an overestimated B_hat lowers K far less than 1/B_hat; dangerous for B_hat > ~30 B | B_hat = 18.3 B -> K = 3.99 (factor 2.6); K = 3 at 28.6 B | level1_core.dare_K | established; corrected 2026-11-03 (was "K ~ 1/B_hat") |
| L7 | ranking of families does not depend on the failure threshold (5-100 %) | out/stats/threshold_sensitivity.csv | analysis/stats.py | established |

## DPC (Part 2)
| # | claim | number [CI] | produced by | status |
|---|---|---|---|---|
| D1 | LN MSE 3000: DPC fails / is trapped | 13/30 = 43 [13, 73] % / 12/30 = 40 [10, 70] %; seeds 1, 2, 8, 9 in every repeat | out/stats/dpc_rates_boot.csv | established |
| D2 | two-stage, MSE 12000, no norm, static norm: no failure | 0/30 each | same | established |
| D3 | two-stage beats LN MSE (per-seed median gap) | 10/10 seeds, median diff -3.6 [-291.7, -0.7] pp, Wilcoxon p = 0.002, Holm 0.012 | out/stats/paired_claims.csv | established |
| D4 | LN MSE loses to static norm / to no norm | 10/10, p = 0.002 (Holm 0.012) / 8/10, p = 0.010 (Holm 0.039) | same | established |
| D5 | two-stage ties static norm | 6/10, p = 0.43 | same | established |
| D6 | two-stage beats no norm | 9/10 seeds, p = 0.049, NOT significant after Holm (0.098) | same | corrected 2026-11-03 (was stated as significant) |
| D7 | MSE 12000 beats MSE 3000 | 8/10 seeds, p = 0.014, Holm 0.041 | same | corrected 2026-11-03 (was 9/10, p = 0.006, repeat 0 only) |
| D8 | the failure is an optimisation trap: the surrogate prices the good controller correctly | oracle on surrogate 2.48-2.56 vs 2.50 true (seeds 1, 2, 8, 9, rep 0); own controller 3.8x, 3.4x, 7.5x, 750x worse; 20 of 21 failed runs trapped | dpc_diag.csv | established |
| D9 | gradient at the good controller is exploding on trapped pairs | orders of magnitude only: 1e3 to 1e35 vs 0.03-3.5 healthy | theory/t2_test.py (grad_batch) | established as orders of magnitude (not reproducible beyond that) |
| D10 | warm start: plain AdamW pushes the good controller off on most broken pairs; per-start clipping keeps it | 5 of 6 vs 0 of 6 (healthy 0 of 4 each) | analysis/warmstart_harness.py, opt_report.py | established (re-run 3 Nov; LN MSE s2 drifts +9.6 %, just under the 10 % line) |
| D11 | in the WARM START, generic optimiser tricks do not keep it (from scratch see D13): global clip 3 / median of means / minibatch 100 | pushed off 4 / 5 / 4 of 6; unit vote 0 of 6 | same | P2b held, P3b failed (pre-registered); unit and batch not predicted |
| D12 | lr 1e-4 keeps it | 0 of 6 in 100 epochs; 1000-epoch re-run | out/opt/warmstart_lr1e-4_1000ep_*.csv | exploratory, see RESULTS_opt.md |
| D13 | from scratch: unit 0/30 failed; per-start clip 0/30 trapped, 5 model-mismatch failures (both cost about 4x: 120 vs 28 s per seed); global clip rescues 3/4 target seeds (2 traps in 30) and lr 1e-4 4/4 (3 traps), both at the cost of plain; median of means 1/4; two-stage with the plain optimiser best: per-seed median gap 0.04 % vs 0.94-2.42 %, at the cost of plain DPC plus 1000 surrogate training steps | Run 4 | analysis/opt_report.py (218cbbe), analysis/run4_followups.py | P1 held; P2a, P3a, P4 failed; P5a confounded (horizon/noise mismatch, matched control not run); P5b held |

## Theorem 2 and Lambda*
| # | claim | number [CI] | produced by | status |
|---|---|---|---|---|
| T1 | Lambda* > 12.1 separates the GRU traps at B = 0.01 retrospectively (threshold fitted on LN MSE 3000 of the same data) | 17/20 hits = 85 [55, 100] %, 0/160 false alarms [0, 0] | theory/t2_test.py -> out/theory/t2_runs_*.csv | established as a retrospective fit (reproduced 180/180); NOT a validated predictor, see T6-T8 (corrected 2026-11-04) |
| T2 | within the GRU B = 0.01 data the separation does not hinge on 12.1 | any threshold in [8.61, 17.03) gives 17/20 and 0/160; leave one seed out 11/12, 0/18 | out/stats/lambda_threshold_sweep.csv, lambda_loso.csv | established (within that data set only; the zero-false-alarm interval is different on every prospective set: MLP (5.97, 7.36], GRU B = 0.1 (10.73, 14.75], linear > 93.4) |
| T3 | swapping J_u_hat -> B removes the expansion on most LN MSE traps | 9 of 12 below 12.1 (J_x_hat -> A: 0 of 12) | theory/t2_swap.py | established |
| T4 | Lambda* gives an upper bound on the gradient at the good controller | ||grad|| <= 4.7 e^Lambda* on 180/180 GRU runs (max 4.63) and 90/90 linear runs (max 0.41); 89/90 MLP runs (one at 6.1). The factor 4.7 was read off the GRU runs. Linear and MLP have 90 runs because their oracle gradients were computed on 3 of the 6 families (LN MSE 3000, slope from 0, two-stage; 30 runs each). The bound is not tight: of the 28 GRU runs with Lambda* > 5, the gradient is within 10^3 of it on 8 (slope from 0 seed 9 rep 0: 0.55 with Lambda* = 83.5); linear 0 of 15 | t2_runs_*.csv, out/theory/arch_gradients_*.csv | established as an upper bound (an observation, not a theorem for the GRU); NOT an estimate of the gradient's size (corrected 2026-11-04) |
| T5 | memoryless LQR-gain policy: Lambda* <= 8.75 on all 180 -> linear DPC predicted never trapped (P2) | max 8.746; Run 5b: 0 of 180 linear runs trapped | theory/t2_memoryless.py, t2_arch.py --unit, docs/RESULTS_arch.md | held (P2). But the trained linear oracle has gain about 17, not 10.3, and its own Lambda* (up to 93) predicted 12 traps that did not occur (P2' failed) |
| T6 | Lambda* of the MLP oracle predicts MLP traps (prospective, f945a5c) | predicted 0, trapped 40: 0/40 hits, 0/140 false alarms; MLP oracle Lambda* <= 7.4 on all 180 (below the threshold) | analysis/make_prereg_arch.py, arch_report.py | FAILED |
| T7a | Lambda* of the trained linear oracle predicts linear traps (prospective, 322e353) | predicted 12, trapped 0: 12 false alarms (6.7 %) | same | FAILED |
| T8 | Lambda* of the B = 0.1 GRU oracle predicts B = 0.1 traps (prospective, 77d8455) | predicted 1, trapped 7: 1/7 hits, 0/173 false alarms; the augmented Lambda*_aug (91238b4): 0/7 | analysis/b01_report.py | FAILED (hit condition) |
| T7 | exact-memory statistic Lambda*_mem | see RESULTS_t2_exact.md | theory/t2_exact.py | exploratory |

Added 2026-11-04: a trap also needs a tame region for the policy to settle in (theory.tex, remark after Corollary 6). This explains the linear false alarms: the gradient at the linear oracle explodes (1e27-1e34) on all 12, yet two parameters leave no region to fall into. It was not part of the pre-registration. At B = 0.01 the explosion at the good controller was neither necessary nor sufficient for a trap. Not necessary: 6 of the 20 GRU traps had a tame gradient there (0.40-3.7: LN MSE 3000 seed 8 rep 2; slope from 0 seed 3 rep 2, seed 4 rep 1, seed 9 reps 0-2), against >= 1.3e3 on the other 14 and <= 106 on all 160 untrapped runs; and the MLP traps occur with a tame gradient at the MLP oracle (docs/RESULTS_lambda_posthoc.md). Not sufficient: the linear policy. MLP DPC results by family (Run 5a, trapped / failed of 30, median gap): LN MSE 3000 27 / 27 (241 %); slope from 0 12 / 12 (0.25 %); MSE 12000 1 / 1 (0.13 %); two-stage 0 / 0 (0.06 %); no norm 0 / 0 (0.61 %); static norm 0 / 0 (0.12 %). So two-stage training, no norm and static norm remove the MLP traps, as they remove the GRU traps.

## Generality
| # | claim | status |
|---|---|---|
| G1 | the trap is not specific to the GRU (MLP, linear) | pending Runs 1, 2, 5 |
| G2 | traps persist / vanish at ordinary authority B = 0.1 | pending Runs 3, 6 (PREREG_B01 Part 1 committed 8ecc750) |
| G3 | a two-state plant | proposal docs/plant2/proposal.tex (out/plant2/proposal.pdf), awaiting decision |
| G4 | "global clipping does not help" | may NOT be claimed: false from scratch (Run 4: 3 of 4 target seeds rescued, 2 traps in 30); true only in the warm start (corrected 2026-11-04) |
