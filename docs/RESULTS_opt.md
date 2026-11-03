# Task 3 - Optimiser baselines (status 2026-11-03)

Pre-registration: `prereg/PREREG_opt.md` (commit c41967c, before any result). From-scratch runs (P1, P2a, P3a, P4, P5) need
Colab Run 4 (`docs/NEXT_COLAB.md`); `analysis/opt_report.py` scores them when `dpc_results.csv` / `dpc_diag.csv` are uploaded.
Done here: the warm-start part (local CPU, `analysis/warmstart_harness.py`, 100 AdamW epochs at N = 200 from the repeat-0 GRU
oracle; figure `out/opt/fig_warmstart.png`, table `out/opt/warmstart_table.csv`).

## Warm start: pushed off the good controller? (true-plant cost > 1.10 x its start at any logged epoch 10-100)
| setting | broken pairs pushed off (of 6) | healthy pairs (of 4) |
|---|---|---|
| plain AdamW | 5 | 0 |
| per-start clip 3 | 0 | 0 |
| unit vote | 0 | 0 |
| global clip 3 | 4 | 0 |
| median of means (10 groups) | 5 | 0 |
| minibatch 100 | 4 | 0 |
| lr 1e-4 | 0 | 0 |
Broken pairs: LN MSE seeds 1, 2, 8, 9; slope-from-0 seeds 2, 9. Healthy: LN MSE 3, 7; two-stage 2, 9.

True-plant cost at epoch 100 on the broken pairs (start 2.56-2.64):
| pair | plain | clip 3 | unit | global 3 | mom | batch 100 | lr 1e-4 |
|---|---|---|---|---|---|---|---|
| LN MSE s1 | 2.60 | 2.59 | 2.61 | 2.63 | 3.09 | 4.49 | 2.56 |
| LN MSE s2 | 2.89 | 2.64 | 2.65 | 2.74 | 3.51 | 3.79 | 2.66 |
| LN MSE s8 | 5.25 | 2.63 | 2.63 | 3.15 | 3.08 | 2.84 | 2.65 |
| LN MSE s9 | 4.70 | 2.65 | 2.67 | 2.66 | 3.02 | 2.74 | 2.65 |
| slope s2 | 5.87 | 2.64 | 2.64 | 2.64 | 4.05 | 6.05 | 2.64 |
| slope s9 | 4.59 | 2.64 | 2.65 | 2.66 | 2.64 | 4.04 | 2.65 |

## Scored predictions
- P2b (global clip 3 pushed off on >= 4 of 6): HELD, 4 of 6 (LN MSE 1, 2, 8, 9). Caveat: by epoch 100 most of them are back
  near the start (2.63-2.74; seed 8 3.15), so global clipping gives large transient kicks rather than a lasting walk uphill;
  the slope-from-0 pairs were not pushed off.
- P3b (median of means stays on >= 5 of 6): FAILED, stays on 1 of 6 (pushed off 5, ends 3.0-4.1). Untested explanation:
  exploding per-start gradients occur in more than half of the 10 groups at some steps, so the coordinate-wise median is
  corrupted too. This contradicts the "one start dominates" reading of the mechanism for these steps; worth checking by
  logging per-group gradient norms.
- P6 (healthy pairs stay under every setting): HELD, 0 of 28.
- Reproduction: plain pushes off 5 of 6 (reference: 5 of 6 plus seed 2; here LN MSE s2 drifts to +9.6 %, just under the line,
  reference reached 3.80; chaotic, pitfall 9); clip 3 keeps all 6 (= reference).
- Not predicted: unit vote keeps all 6 (like clip 3); minibatch 100 pushes off 4 (U1: no prediction); lr 1e-4 keeps all 6
  within 100 epochs, possibly only slower (10x smaller steps); its pre-registered prediction (P4) is about from-scratch runs.

## Reading so far
Bounding each trajectory's vote (clip 3, unit) is what keeps the good controller; generic tricks (global clipping, median of
means, minibatches) do not. Whether per-start clipping also FINDS the good controller from scratch is the Colab Run 4 question.
