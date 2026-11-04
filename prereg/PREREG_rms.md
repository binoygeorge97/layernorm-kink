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
