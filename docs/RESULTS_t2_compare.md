# B4: four versions of the landing amplification on the 180 GRU runs (B = 0.01), side by side

Written 2026-11-04. Retrospective: all four are computed from the oracle GRU (same repeat) through each surrogate; the
registered direct-gain Lambda* (S_seg_max, theory/t2_test.py) stays the primary statistic. Script analysis/b4_table.py,
table out/theory/b4_table.csv.
- direct gain (registered): scalar multipliers m_k = J_x_hat - J_u_hat K_k with K_k = -du_k/dx_k at fixed hidden state,
  from the first landing step, max contiguous sum of log|m_k|.
- exact price Lambda*_mem (theory/t2_exact.py): the exact sensitivity price through all GRU memory paths, normalised by the
  largest direct one-step source, not anchored at landing.
- anchored scalar (theory/t2_aug.py, L_scal_land_star): the direct-gain multipliers with the running product restarted at
  landing (the anchoring of Lambda*_aug, without memory).
- anchored augmented Lambda*_aug (theory/t2_aug.py): operator norm of the product of the 65 x 65 (x, h) Jacobians of the
  closed loop through the surrogate, anchored at landing.

| statistic | traps above 12.1 | false alarms above 12.1 | largest non-trapped value | traps above it (zero-false-alarm) | smallest trapped value |
|---|---|---|---|---|---|
| direct gain (registered) | 17/20 | 0/160 | 8.61 | 17/20 | 2.49 |
| exact price Lambda*_mem | 14/20 | 2/160 | 16.47 | 14/20 | 2.26 |
| anchored scalar | 17/20 | 0/160 | 8.23 | 17/20 | 2.41 |
| anchored augmented Lambda*_aug | 15/20 | 0/160 | 11.21 | 15/20 | 2.27 |

The 20 traps and every run that is a false alarm (> 12.1) under at least one statistic (* = above 12.1):
| family | seed | rep | trapped | gap % | direct-gain Lambda* (registered) | exact price Lambda*_mem | anchored scalar | anchored augmented Lambda*_aug |
|---|---|---|---|---|---|---|---|---|
| LN MSE 3000 | 1 | 0 | yes | 281.6 | 17.03* | 16.63* | 16.10* | 15.97* |
| LN MSE 3000 | 1 | 1 | yes | 71.4 | 18.81* | 16.72* | 17.93* | 16.55* |
| LN MSE 3000 | 1 | 2 | yes | 24.9 | 34.59* | 37.91* | 33.81* | 31.93* |
| LN MSE 3000 | 2 | 0 | yes | 242.1 | 131.14* | 91.89* | 131.14* | 90.53* |
| LN MSE 3000 | 2 | 1 | yes | 432.9 | 158.08* | 28.31* | 158.08* | 28.38* |
| LN MSE 3000 | 2 | 2 | yes | 952.3 | 118.42* | 96.73* | 117.66* | 101.03* |
| LN MSE 3000 | 8 | 0 | yes | 674.7 | 52.41* | 43.04* | 51.59* | 46.50* |
| LN MSE 3000 | 8 | 1 | yes | 24.0 | 66.97* | 28.54* | 65.56* | 28.95* |
| LN MSE 3000 | 8 | 2 | yes | 291.8 | 89.43* | 11.74 | 89.43* | 13.01* |
| LN MSE 3000 | 9 | 0 | yes | 99693.8 | 79.92* | 35.74* | 79.21* | 38.17* |
| LN MSE 3000 | 9 | 1 | yes | 336.8 | 83.96* | 75.37* | 83.96* | 82.38* |
| LN MSE 3000 | 9 | 2 | yes | 178.0 | 38.25* | 33.58* | 36.99* | 44.16* |
| LN slope from 0 | 2 | 0 | yes | 74.0 | 72.09* | 53.34* | 67.43* | 83.09* |
| LN slope from 0 | 2 | 1 | yes | 7256.6 | 94.81* | 21.36* | 91.41* | 23.90* |
| LN slope from 0 | 2 | 2 | yes | 865.3 | 93.85* | 79.97* | 92.42* | 90.54* |
| LN slope from 0 | 3 | 2 | yes | 42.9 | 6.45 | 3.64 | 4.11 | 4.59 |
| LN slope from 0 | 4 | 1 | yes | 275.7 | 2.49 | 2.26 | 2.41 | 2.27 |
| LN slope from 0 | 9 | 0 | yes | 222.7 | 83.51* | 6.99 | 83.34* | 9.38 |
| LN slope from 0 | 9 | 1 | yes | 61.3 | 3.37 | 4.08 | 3.12 | 3.84 |
| LN slope from 0 | 9 | 2 | yes | 637.0 | 45.10* | 6.96 | 45.10* | 7.55 |
| LN MSE 3000 | 0 | 1 | no | 2.1 | 8.61 | 12.39* | 8.23 | 11.21 |
| LN MSE 3000 | 7 | 1 | no | 0.2 | 7.28 | 16.47* | 6.73 | 8.95 |

Separating the two losses. Anchoring alone costs nothing: the anchored scalar flags the same 17 traps as the direct-gain
statistic, with no false alarm, and on the 17 flagged traps its values are within 7 % of it. Memory costs the rest: going from the
anchored scalar to the anchored augmented statistic loses slope-from-0 seed 9 rep 0 (83.3 -> 9.4) and seed 9 rep 2
(45.1 -> 7.6), and brings LN MSE 3000 seed 8 rep 2 down from 89.4 to 13.0 (still flagged, the smallest flagged trap); the exact price
(memory, no anchoring) loses the same two slope-from-0 runs and LN MSE seed 8 rep 2 (11.7), and in addition raises two false
alarms (LN MSE 3000 seed 0 rep 1, 12.4; seed 7 rep 1, 16.5). The three slope-from-0 misses with small values (seed 4 rep 1,
seed 9 rep 1, seed 3 rep 2) are missed by all four statistics.

Why the exact quantity misses traps the direct-gain one catches. On those runs the oracle GRU's hidden state corrects for
the direct-gain expansion: when a step through the kink multiplies the deviation in x by a large factor, the deviation also
enters h, and the next inputs computed from h push back, so the product of the full (x, h) Jacobians grows far less than the
product of the scalar multipliers (seed 9 rep 0 of slope-from-0: e^83 against e^9). That is a property of the ORACLE
controller, whose memory was trained on the true plant and happens to compensate along the oracle's own trajectories. The
trap, however, is about what gradient descent does from the start of training and from nearby controllers, where this
compensation need not exist. The direct-gain statistic measures what the surrogate's slope error does to any policy with
that direct gain, so it is less tied to the oracle's particular memory. This is an interpretation that fits the numbers, not
a tested explanation; a test would be to compute Lambda*_aug along the DPC training path (controllers at intermediate
epochs) rather than at the oracle.
