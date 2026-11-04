# Pre-registration: Part 1 (LQR) replication on held-out seeds 10-29 (B = 0.01)

## Part 1 (written 2026-11-04, BEFORE any surrogate of seeds 10-29 exists)
Results that exist: everything on seeds 0-9. No data set, label, surrogate or LQR result for seeds 10-29 exists.

### Design (fixed now)
- Seeds 10-29, the existing key scheme unchanged (data numpy default_rng(1000 + s), init PRNGKey(s), slope-check draws
  PRNGKey(10000 + s); checked 4 Nov: no overlap with the 40000 / 50000 offsets for s <= 29, rep <= 2).
- Surrogates trained exactly as for seeds 0-9 (dpc_core.train_surrogates, float64), in a NEW output folder
  (data/dpc_out_s10_29; the cache in data/dpc_out is keyed by the seed list and would be overwritten otherwise).
  Families: all six if local CPU training takes under about 3 h, otherwise LN MSE 3000, LN two-stage, static norm;
  families not trained are reported "not trained" (no outcome).
- LQR study: companion/run_lqr_study.py with LK_SEEDS=10,...,29, same 11 setpoints x 9 starts, same metrics; rule CLAUDE.md
  1.3 (|x0 - r| >= 0.5; failed = diverged or gap > 10 %; catastrophic = diverged or gap > 100 %).
- Kink metrics: centre z*, half-widths from Theorem 1 (the surrogates.csv columns zmin_x, zmin_u).

### Predictions
| id | prediction | how it is scored |
|---|---|---|
| H1 | each failed / catastrophic rate (family x controller) of seeds 10-29 lies in its 95 % prediction interval below (two-stage interval, primary) | per cell inside / outside; H1 held if at most 2 of the scored cells lie outside (about 5 % of 24 expected outside by chance), else failed; outside cells listed |
| H2 | fixed LQR, LN MSE 3000 fails more often than static norm (26.5 % vs 0.6 % on seeds 0-9) | rates on seeds 10-29; held if LN MSE 3000 > static norm and its interval cell holds |
| H3 | every LN kink centre (4 LN families x 20 seeds) lies within 0.1 of (0, 0): max(abs(zmin_x), abs(zmin_u)) <= 0.1 | count of centres outside; held if none |

Prediction intervals: analysis/make_prereg_heldout.py (rng 20261104, 10 000 resamples), prereg/PREREG_heldout_intervals.csv.
Primary = two-stage: bootstrap seeds 0-9 (the uncertainty of the rates themselves), then draw the 20 new seeds with
replacement from that resample, pooled rate over their runs. The single-stage version (20 seeds drawn from the fixed seeds
0-9) is shown for reference: it ignores the uncertainty of the 10-seed rates and is NARROWER than the 95 % CI of the mean,
so it is not used for scoring. Cells with no event on seeds 0-9 have the degenerate interval [0, 0]: any event there counts
as outside.

| family | controller | metric | seeds 0-9 (%) | 95 % PI, two-stage (PRIMARY) | single-stage (20 from fixed 0-9) | 95 % CI of the mean (10 seeds) |
|---|---|---|---|---|---|---|
| LN MSE 3000 | LQR fixed | failed | 26.5 | [18.2, 34.4] | [21.7, 31.3] | [19.7, 33.2] |
| LN MSE 3000 | LQR fixed | catastrophic | 10.8 | [3.1, 19.4] | [6.1, 15.7] | [4.2, 17.7] |
| LN MSE 3000 | LQR sched | failed | 20.5 | [15.2, 25.9] | [17.4, 23.6] | [16.0, 24.9] |
| LN MSE 3000 | LQR sched | catastrophic | 3.4 | [1.6, 5.3] | [2.3, 4.5] | [1.8, 5.0] |
| LN MSE 12000 | LQR fixed | failed | 6.5 | [2.7, 11.5] | [4.0, 9.3] | [3.2, 10.6] |
| LN MSE 12000 | LQR fixed | catastrophic | 1.1 | [0.0, 3.6] | [0.1, 2.5] | [0.0, 3.0] |
| LN MSE 12000 | LQR sched | failed | 3.5 | [1.0, 7.9] | [1.7, 5.9] | [1.1, 6.8] |
| LN MSE 12000 | LQR sched | catastrophic | 0.2 | [0.0, 0.9] | [0.0, 0.6] | [0.0, 0.7] |
| LN slope from step 0 | LQR fixed | failed | 5.8 | [1.1, 12.0] | [2.8, 9.1] | [1.7, 10.7] |
| LN slope from step 0 | LQR fixed | catastrophic | 1.4 | [0.0, 4.6] | [0.2, 3.0] | [0.0, 3.6] |
| LN slope from step 0 | LQR sched | failed | 2.8 | [1.4, 4.3] | [2.0, 3.7] | [1.6, 4.1] |
| LN slope from step 0 | LQR sched | catastrophic | 0.0 | [0.0, 0.0] | [0.0, 0.0] | [0.0, 0.0] |
| LN two-stage | LQR fixed | failed | 5.7 | [0.5, 13.7] | [2.1, 9.9] | [0.9, 11.9] |
| LN two-stage | LQR fixed | catastrophic | 0.9 | [0.0, 3.6] | [0.0, 2.3] | [0.0, 2.7] |
| LN two-stage | LQR sched | failed | 3.5 | [0.0, 9.7] | [0.9, 7.0] | [0.0, 8.8] |
| LN two-stage | LQR sched | catastrophic | 0.1 | [0.0, 0.5] | [0.0, 0.3] | [0.0, 0.3] |
| no norm | LQR fixed | failed | 2.6 | [0.3, 7.1] | [0.8, 4.9] | [0.3, 6.0] |
| no norm | LQR fixed | catastrophic | 0.0 | [0.0, 0.0] | [0.0, 0.0] | [0.0, 0.0] |
| no norm | LQR sched | failed | 5.7 | [0.3, 16.5] | [1.2, 11.4] | [0.5, 13.6] |
| no norm | LQR sched | catastrophic | 3.6 | [0.0, 14.5] | [0.0, 9.1] | [0.0, 10.9] |
| static norm | LQR fixed | failed | 0.6 | [0.0, 1.8] | [0.1, 1.2] | [0.0, 1.5] |
| static norm | LQR fixed | catastrophic | 0.0 | [0.0, 0.0] | [0.0, 0.0] | [0.0, 0.0] |
| static norm | LQR sched | failed | 0.8 | [0.0, 1.9] | [0.2, 1.4] | [0.0, 1.7] |
| static norm | LQR sched | catastrophic | 0.0 | [0.0, 0.0] | [0.0, 0.0] | [0.0, 0.0] |

## Outcomes Part 1 (fill AFTER the held-out LQR study; do not edit anything above this line)

## Outcomes Part 1 (filled 2026-11-04; docs/RESULTS_heldout.md)
| id | outcome | numbers |
|---|---|---|
| H1 | failed | 6 of 12 scored cells outside their 95 % PI, all above (LN MSE 3000 fixed failed 36.2 % vs [18.2, 34.4]) |
| H2 | failed (cell outside); ordering held | LN MSE 3000 36.2 % vs static norm 1.1 % |
| H3 | failed | 4 of 40 LN centres outside 0.1 (max offset 0.144) |
Families trained: LN MSE 3000, LN two-stage, static norm (the six would exceed ~3 h of CPU).
