# Pre-registration: MLP and linear DPC controllers, prospective test of Theorem 2 (Task 1)

Date written: 2026-11-03
Commit hash of this file (fill after committing, then commit again): see "Registered at commit" at the end
Results that exist at the time of writing: the GRU study (dpc_out, 7 models x 10 seeds x 3 repeats); the MLP ORACLE
controllers, repeats 0-2 (Stage A, Colab Run 1, `MyDrive/dpc_out_mlp_stageA.zip` -> `data/dpc_out_mlp`: true-plant cost
2.4969-2.5007 on all 30, analytic saturated LQR 2.4968). The linear oracle does not exist yet (Run 2): this file registers the
MLP part (P1, A1, A3) now; the linear part (P2 from the trained linear oracle, A2, A4, A5) is appended as Part B before the
linear Stage B run. No MLP or linear controller trained through any surrogate exists.

## Question
Is the DPC trap an artefact of the recurrent (GRU) policy, and does the landing amplification Lambda*, computed from the good
(oracle) controller alone, predict which MLP and linear DPC runs are trapped?

## Design (fixed before results)
- Stage B (Colab A100): DPC notebook, `ARCH = "mlp"` then `ARCH = "linear"`, `ONLY = None`, `REPS = [0, 1, 2]`,
  `RUN_ROBUST = False`, `RUN_OPT = False`, `DTYPE = "float64"`, `QUICK = False`, Drive root `MyDrive/dpc_out` (results go to
  `dpc_out_mlp`, `dpc_out_linear`; surrogates read from `dpc_out`). MLP: width 64, depth 2, tanh, u = 10 tanh(.).
  Linear: u = 10 tanh((k x + b) / 10), k, b stored / 10, init gain -5.
- Units: 6 surrogate families x 10 seeds x 3 repeats = 180 runs per architecture.
- Primary outcome: trapped = failed (typical-start gap > 10 %) AND own controller on its surrogate > 1.05 x the oracle
  controller of the same architecture on the same surrogate (dpc_diag.csv of the architecture's folder).
- Predictor: Lambda* of the architecture's oracle (same repeat) through the surrogate, 1000 training starts of that repeat,
  landing window |x| <= 0.2 (theory/t2_arch.py, commit of this file). Threshold 12.1, NOT re-tuned.

## Predictions
| id | prediction | basis | how it is scored |
|---|---|---|---|
| P1 | MLP: trapped iff Lambda* > 12.1; predicted trapped: 0 of 180 runs (list below) | Theorem 2, GRU test 17/20 hits, 0/160 false alarms | confusion matrix; held if hit rate >= 0.75 and false-alarm rate <= 0.05 (if no run is predicted trapped: held if at most 1 of 180 is trapped); partly if one of the two holds |
| P2 | Linear: no run trapped (0 of 180). Trained linear oracle not available yet: its Lambda* is registered in an addendum before the linear Stage B run; linear policy at K* = 10.2875: max 8.75 | memoryless policy with LQR-like gain stays in the safe band | held if 0 trapped; else failed, with the runs listed |
| A1 | (copied, 23 Oct) The MLP reproduces the GRU result: LN MSE fails on seeds 1, 2, 8, 9 and slope-from-0 on 2, 3, 9; two-stage and kink-free on none; trap test > 1 on the failures | GRU result | per-seed failures (majority of repeats) vs the listed seeds |
| A2 | (copied) The linear controller does NOT fail on those seeds | as P2 | failed runs on those seeds |
| A3 | (copied) Trapped MLP policies are flat or inverted inside the kink band, recover the LQR slope outside; healthy ones are a saturated line of slope -10 to -17 | - | D9 policy maps, K_eff at x = 0, 0.5, 2 |
| A4 | (copied) The linear DPC loss minimum through the surrogate is within ~10 % of the true plant's K on every seed | - | D10 |
| A5 | (copied) dL/dK through LN MSE has the wrong sign on a larger share of the gain axis than through kink-free models | - | D11 |
| A6, A7 | (copied) capacity sweep, survivor orientation: not run in this stage | - | not scored here |

Note on P1 vs A1 (written before results): A1 was written on 23 Oct from the GRU result. P1 is the theory's run-level
prediction from the MLP oracle's own closed loop. Where they disagree on a seed (A1 says failed, P1 says Lambda* <= 12.1, or
vice versa), the outcome decides between "the trap belongs to the surrogate" (A1) and "the trap needs a policy whose good
solution amplifies through the kink" (P1). Both are scored.

Two notes written before results. (1) P1 and A1 now make OPPOSITE predictions on LN MSE seeds 1, 2, 8, 9: the MLP oracle's
direct gain at landing is LQR-like (median K_land 10.1-10.8 on every run, against 13-22 for the GRU reference), and its
Lambda* stays below 8 everywhere. (2) The exactness check sum log|m_k| = log|dx_N/dx_0| holds to <= 1e-6 on 179 of 180 runs;
on LN slope-from-0 seed 4 repeat 0, 256 strongly contracting starts underflow (|dx_N/dx_0| = 2e-307 < the 1e-300 floor; sum
log|m| = -916), which concerns contraction, not amplification.

Per-family summary of the MLP predictor (oracle Lambda*):
| family | runs | predicted | max_Lambda | med_K_land |
|---|---|---|---|---|
| LN MSE 3000 | 30 | 0 | 7.36 | 10.47 |
| LN MSE 12000 | 30 | 0 | 3.82 | 10.46 |
| LN slope from 0 | 30 | 0 | 5.97 | 10.47 |
| LN two-stage | 30 | 0 | 2.38 | 10.49 |
| no norm | 30 | 0 | 0.00 | 10.48 |
| static norm | 30 | 0 | 0.00 | 10.48 |

Per-family summary of the linear predictor (linear policy at K* = 10.2875, until the trained linear oracle exists Lambda*):
| family | runs | predicted | max_Lambda | med_K_land |
|---|---|---|---|---|
| LN MSE 3000 | 30 | 0 | 8.75 | 10.28 |
| LN MSE 12000 | 30 | 0 | 5.04 | 10.29 |
| LN slope from 0 | 30 | 0 | 5.62 | 10.28 |
| LN two-stage | 30 | 0 | 2.75 | 10.29 |
| no norm | 30 | 0 | 0.00 | 10.28 |
| static norm | 30 | 0 | 0.00 | 10.28 |

Runs predicted trapped (MLP):
- none

Run-level predictions for both architectures: `prereg/PREREG_arch_predicted.csv`.

## Analysis plan (analysis/arch_report.py, docs/RESULTS_arch.md)
Per architecture x family: failed / trapped of 30, median gap; paired Wilcoxon vs the GRU over seeds (per-seed median over
repeats); confusion matrices of P1 and P2; K_eff table (D9) against the LQR gain 10.29; A1-A5 marked held / failed / partly
with the numbers. If the MLP is not trapped on the GRU-trapped seeds, the paper claim narrows to "recurrent /
high-direct-gain policies", stated plainly.

## Outcomes (fill AFTER results; do not edit anything above this line)
| id | outcome (held / failed / partly) | numbers |
|---|---|---|

## Part B (linear controller): to be appended from the trained linear oracle (Run 2) before the linear Stage B run

Part A (MLP) registered at commit f945a5cd587d2a3cd21423ae0fbc1ba1e21dfece (2026-11-03), before any MLP controller was trained through a surrogate.
