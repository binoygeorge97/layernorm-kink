# Pre-registration: RMSNorm surrogates (Q7), Part 1

## Part 1 (written 2026-11-04, BEFORE any RMSNorm surrogate is trained)
Basis: docs/theory/rmsnorm/rmsnorm_kink.pdf (b349d6d). Without centring, the feature map has the Theorem 1 structure with
P = I: centre z* = -(E^T E)^{-1} E^T b, kink size D = sqrt(||c_perp||^2 + H eps) with c_perp the part of b outside range(E);
for the same weights D_RMS >= D_LN, and on LN-trained weights the two differ by at most 4 %. The theory does not predict how
training changes the weights, so the size and failure-rate predictions below are coarse (factor-2 bands), stated as such.

### Design (fixed now)
- Two new configurations (a9b34f2): "RMS MSE (3000)" (as LN MSE 3000 with v = gamma h / sqrt(mean(h^2) + eps) + beta) and
  "RMS J_x + J_u (two-stage)" (as LN two-stage). Seeds 0-9, same data, init, labels and slope-check keys as the LN families,
  float64, local CPU, new output folder data/dpc_out_rms (surrogates and labels; the labels are the B = 0.01 ones and
  are recomputed there, identical by construction).
- Kink metrics from the RMS version of Theorem 1 (level1_core.kink_center with P = I): zmin_x, zmin_u, rstar_x, rstar_u.
- LQR study: companion/run_lqr_study.py on the two RMS families (same grid, rule CLAUDE.md 1.3).
- Comparison values (LN, seeds 0-9, data/dpc_out/surrogates.csv, companion/lqr_study.csv): LN MSE 3000 median
  rstar_x 0.261, rstar_u 0.336; fixed-LQR failed 26.5 %; LN two-stage fixed-LQR failed 5.7 %; static norm 0.6 %.

### Predictions
| id | prediction | how it is scored |
|---|---|---|
| M1 | every RMS kink centre (2 families x 10 seeds) lies within 0.1 of (0, 0): max(abs(zmin_x), abs(zmin_u)) <= 0.1 | held if none outside |
| M2 | RMS MSE 3000 kink size comparable to LN MSE 3000: median rstar_x in [0.13, 0.52] and median rstar_u in [0.17, 0.67] (factor 2 around the LN medians) | held if both inside |
| M3 | fixed-LQR failure rate of RMS MSE 3000 within a factor 2 of LN MSE 3000: in [13.3, 53] % (runs with abs(x0 - r) >= 0.5) | held if inside |
| M4 | two-stage training lowers the fixed-LQR failure rate for RMSNorm as for LayerNorm: RMS two-stage < RMS MSE 3000, and RMS two-stage <= 13.3 % | held if both |
| U1 | no prediction on the rescheduled LQR or on the peak input-slope error | reported |

DPC traps through the RMS surrogates are pre-registered separately (Part 2) after their Lambda* is computed from the existing
B = 0.01 GRU oracles, threshold 12.1, before any DPC-through-RMS run.

## Outcomes Part 1 (fill AFTER training and the LQR study; do not edit anything above this line)

## Outcomes Part 1 (filled 2026-11-04; analysis/rms_report.py, out/rms/summary.csv)
| id | outcome | numbers |
|---|---|---|
| M1 | failed (narrowly) | 1 of 20 RMS centres outside 0.1: max offset 0.109 (all others <= 0.1) |
| M2 | held | RMS MSE 3000 median r*_x 0.233 (band [0.13, 0.52]), r*_u 0.272 ([0.17, 0.67]); LN MSE 3000 0.261, 0.336 |
| M3 | held | RMS MSE 3000 fixed LQR failed 29.5 % (band [13.3, 53]); catastrophic 15.6 % (LN 26.5 / 10.8 %) |
| M4 | held | RMS two-stage fixed failed 4.2 % < 29.5 %, and <= 13.3 % (LN two-stage 5.7 %) |
Peak input-slope error, median: RMS MSE 3000 16.4 B (LN 18.7 B), RMS two-stage 1.43 B (LN 1.19 B). Rescheduled LQR failed /
catastrophic: RMS MSE 3000 22.2 / 4.1 %, RMS two-stage 3.0 / 0.1 %.

## Part 2 (written 2026-11-04, BEFORE any DPC controller is trained through an RMSNorm surrogate): run-level DPC traps
Predictor (as registered for the other runs, threshold NOT re-tuned): direct-gain Lambda* of the existing B = 0.01 GRU oracle
(same seed and repeat) through the local RMS surrogate, theory/t2_test.py (T2_DATA = data/dpc_out_rms, oracle controllers of
data/dpc_out), trapped iff Lambda* > 12.1. Run-level list: prereg/PREREG_rms_predicted.csv. Scoring as R1 of PREREG_B01: held if
hit rate >= 0.75 and false-alarm rate <= 0.05; if no run is predicted trapped, held if at most 1 trapped. Trap definition
CLAUDE.md 1.4.
Caveats stated in advance: Lambda* failed three prospective tests (docs/RESULTS_B01.md, RESULTS_arch.md). The surrogates DPC
will use are trained on Colab (Run 9). If their 3000-step weights differ from the local ones, Lambda* is recomputed on the Colab
surrogates and that is the registered value; this is stated here, before results. The replacement predictor of
PROTOCOL_trap_predictor.md will be added as an addendum, once frozen, if that happens before Run 9.

| family | runs | predicted trapped | max Lambda* | median Lambda* |
|---|---|---|---|---|
| RMS MSE 3000 | 30 | 9 | 155.63 | 4.92 |
| RMS two-stage | 30 | 0 | 1.91 | 0.00 |

Runs predicted trapped:
- RMS MSE 3000, seed 1, repeat 0: Lambda* = 47.71
- RMS MSE 3000, seed 2, repeat 0: Lambda* = 149.54
- RMS MSE 3000, seed 8, repeat 0: Lambda* = 131.20
- RMS MSE 3000, seed 1, repeat 1: Lambda* = 45.11
- RMS MSE 3000, seed 2, repeat 1: Lambda* = 155.63
- RMS MSE 3000, seed 8, repeat 1: Lambda* = 81.70
- RMS MSE 3000, seed 1, repeat 2: Lambda* = 37.89
- RMS MSE 3000, seed 2, repeat 2: Lambda* = 135.43
- RMS MSE 3000, seed 8, repeat 2: Lambda* = 109.14

Also registered: RS1. RMS two-stage has no trapped run (0 of 30).
