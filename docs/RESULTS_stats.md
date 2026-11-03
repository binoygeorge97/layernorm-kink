# Task 4 - Statistics (2026-11-03, local CPU)

Script: `analysis/stats.py` (+ `analysis/swap_lqr_runs.py` for the per-run swap data). Outputs: `out/stats/*.csv`,
LaTeX tables `out/stats/tab_stats.tex`, full printout `out/logs/stats.log`.
Data: `companion/lqr_study.csv` re-generated bit-for-bit (`out/companion/lqr_study_extra.csv`, which adds `settle_time`);
`MyDrive/dpc_out/dpc_results.csv` and `dpc_diag.csv` (downloaded 3 Nov); Lambda* from the Task 0 re-run of the Theorem 2 test.

## Method in plain words
- The unit that is replicated is the SEED: one seed is one surrogate (its data, initialisation and training). Runs of the
  same seed (11 setpoints x 8 starts for the LQR, 3 training repeats for DPC) are not independent of each other.
- Seed bootstrap: draw 10 seeds from the 10 with replacement, pool all runs of the drawn seeds (a seed drawn twice counts
  twice), recompute the rate; 10 000 draws; the 2.5 and 97.5 percentiles are the 95 % interval. The draws are fixed
  (generator seed 20261103), so every interval is reproducible, and every statistic uses the same draws.
- Paired claims: per seed, the median over the 3 repeats of the typical-start DPC gap; for a pair of families the 10
  per-seed differences; Wilcoxon signed-rank test (exact, two-sided); Holm adjustment over the 6 claims; bootstrap interval
  of the median difference (same draws).

## 4.1 Rates with 95 % seed-bootstrap intervals
Fixed and rescheduled LQR, % of the 880 runs per family with |x0 - r| >= 0.5 (failed = gap > 10 % or diverged;
catastrophic = gap > 100 % or diverged):

| family | fixed failed | fixed catastrophic | rescheduled failed | rescheduled catastrophic |
|---|---|---|---|---|
| LN MSE 3000 | 26.5 [19.7, 33.1] | 10.8 [4.2, 17.6] | 20.5 [16.1, 24.8] | 3.4 [1.8, 5.0] |
| LN MSE 12000 | 6.5 [3.3, 10.6] | 1.1 [0.0, 3.0] | 3.5 [1.1, 6.9] | 0.2 [0.0, 0.7] |
| LN slope from step 0 | 5.8 [1.7, 10.7] | 1.4 [0.0, 3.6] | 2.8 [1.6, 4.1] | 0.0 [0.0, 0.0] |
| LN two-stage | 5.7 [0.9, 12.0] | 0.9 [0.0, 2.7] | 3.5 [0.0, 8.8] | 0.1 [0.0, 0.3] |
| no norm | 2.6 [0.3, 5.9] | 0.0 [0.0, 0.0] | 5.7 [0.5, 13.6] | 3.6 [0.0, 10.9] |
| static norm | 0.6 [0.0, 1.5] | 0.0 [0.0, 0.0] | 0.8 [0.0, 1.7] | 0.0 [0.0, 0.0] |

Reproduces the quoted 26.5 / 10.8 % exactly. A "0.0 [0.0, 0.0]" interval means no run of any seed failed; the bootstrap
cannot give an upper bound above zero then (with 880 runs the exact binomial upper bound would be 0.4 %, but runs are not
independent, so read it as "none observed").
The no-norm rescheduled catastrophic rate (3.6 %) comes from one seed (8), hence its wide interval [0.0, 10.9].

Per setpoint (fig P3), fixed LQR, LN MSE 3000, % failed of 80 runs, r = -2 ... 2: 0, 0, 16 [4, 30], 41 [20, 64],
65 [46, 85], 66 [51, 83], 56 [35, 78], 34 [14, 55], 10 [1, 20], 1 [0, 4], 1 [0, 4]. All families and both controllers:
`out/stats/lqr_setpoint_rates_boot.csv`.

Path colour (4 LN families, setpoint gain within 20 % of B, share of the path's steps with a wrong-sign input slope):

| controller | 0 | 0-2 % | 2-10 % | > 10 % |
|---|---|---|---|---|
| fixed LQR, % failed (n) | 1.0 [0.2, 2.2] (2014) | 1.0 [0.0, 2.3] (204) | 0.0 [0.0, 0.0] (46) | 0.0 [0.0, 0.0] (24) |
| rescheduled LQR | 1.1 [0.4, 2.1] (2083) | 9.0 [5.1, 15.1] (111) | 27.3 [14.7, 38.5] (44) | 40.0 [25.6, 55.3] (50) |

Reproduces 1.0 / 1.0 / 0 / 0 % and 1.1 / 9.0 / 27 / 40 % with the same n. The rescheduled rise is well outside the
fixed-LQR intervals from the 2-10 % bin on.

DPC (GRU), runs failed / trapped out of 30 (10 seeds x 3 repeats):

| family | failed | trapped |
|---|---|---|
| LN MSE 3000 | 13 = 43 % [13, 73] | 12 = 40 % [10, 70] |
| LN slope from step 0 | 8 = 27 % [3, 53] | 8 = 27 % [3, 53] |
| LN MSE 12000, LN two-stage, no norm, static norm | 0 [0, 0] | 0 [0, 0] |

The DPC intervals are wide because failures are a property of the seed: 4 of 10 seeds fail in every repeat. This is the
honest uncertainty with 10 surrogates; more seeds, not more repeats, would narrow it.

LQR swap test (fixed LQR, LN MSE 3000, % failed / catastrophic): both from the surrogate 26.5 [19.7, 33.1] / 10.8 [4.2, 17.6];
true A 29.8 [21.5, 36.9] / 12.2 [4.9, 19.8]; true B 13.5 [8.8, 18.3] / 4.1 [0.9, 8.1]; both true 2.3 [1.4, 3.3] / 0.0.
Equal to `story/swap_lqr_rates.csv` for all 5 families. The intervals for "true A" and "both from surrogate" overlap almost
completely, those for "true B" do not reach the other two: the input slope carries the damage, as stated.

## 4.2 Paired claims (typical-start DPC gap; per-seed median over 3 repeats; n = 10 seeds)
| A | B | seeds where A is better | median difference A - B, pp [95 % CI] | Wilcoxon p | Holm p |
|---|---|---|---|---|---|
| two-stage | LN MSE 3000 | 10 | -3.61 [-291.7, -0.70] | 0.002 | 0.012 |
| LN MSE 3000 | static norm | 0 | +3.05 [+0.71, +291.4] | 0.002 | 0.012 |
| LN MSE 3000 | no norm | 2 | +1.68 [+0.33, +290.4] | 0.010 | 0.039 |
| two-stage | static norm | 6 | -0.05 [-0.46, +0.04] | 0.43 | 0.43 |
| two-stage | no norm | 9 | -0.42 [-0.98, -0.17] | 0.049 | 0.098 |
| MSE 12000 | MSE 3000 | 8 | -3.47 [-291.7, -0.14] | 0.014 | 0.041 |

Two corrections to earlier statements:
- "two-stage beats no norm (9/10, p = 0.049)" does not survive the Holm adjustment over these six claims (0.098). Say:
  "two-stage beats no norm on 9 of 10 seeds (Wilcoxon p = 0.049, not significant after Holm adjustment over six claims)".
- "MSE 12000 better than MSE 3000 on 9/10 seeds, p = 0.006" (NOTES_dpc_level1.md) was repeat 0 only. With all three repeats
  (per-seed median) it is 8 of 10 seeds, p = 0.014 (Holm 0.041): still significant, smaller.
The other claims stand after Holm (two-stage beats LN MSE, LN MSE loses to static and to no norm). The bootstrap intervals
of the median difference are very wide on the LN MSE side because of the catastrophic seeds (gaps of 100 to 1000 %);
the sign is what is robust.

## 4.3 Threshold sensitivity and threshold-free metrics
Failed rate (%) at gap thresholds 5 / 10 / 20 / 50 / 100 % (diverged always counts as failed):

| family | fixed LQR | rescheduled LQR | DPC (GRU, 30 runs) |
|---|---|---|---|
| LN MSE 3000 | 33.4 / 26.5 / 19.6 / 12.5 / 10.8 | 27.7 / 20.5 / 14.9 / 7.3 / 3.4 | 43.3 / 43.3 / 43.3 / 33.3 / 30.0 |
| LN MSE 12000 | 11.3 / 6.5 / 4.2 / 1.4 / 1.1 | 6.8 / 3.5 / 2.5 / 0.8 / 0.2 | 0 at every threshold |
| LN slope from step 0 | 8.6 / 5.8 / 4.1 / 2.6 / 1.4 | 5.2 / 2.8 / 1.8 / 0.5 / 0.0 | 26.7 / 26.7 / 26.7 / 23.3 / 16.7 |
| LN two-stage | 8.5 / 5.7 / 2.6 / 1.6 / 0.9 | 6.0 / 3.5 / 1.8 / 1.0 / 0.1 | 0 at every threshold |
| no norm | 5.7 / 2.6 / 1.1 / 0.1 / 0.0 | 8.4 / 5.7 / 4.3 / 3.8 / 3.6 | 3.3 / 0 / 0 / 0 / 0 |
| static norm | 2.0 / 0.6 / 0.1 / 0.0 / 0.0 | 3.1 / 0.8 / 0.5 / 0.0 / 0.0 | 0 at every threshold |

The ranking of the families does not depend on the threshold; LN MSE 3000 is worst at every threshold for every controller.
DPC failures are not near the 10 % line: the trapped runs have gaps far above 10 % (the rate barely moves between 5 and 20 %).

Threshold-free (fixed / rescheduled LQR): median steady offset |x_inf - r| of non-diverged runs 0.03 / 0.03 for LN MSE 3000,
0.01-0.03 for the others (offsets are a value-error effect present in every family, as in the companions); median settling
time to a 5 % band 63 / 60 steps for LN MSE 3000 vs 51-55 for the others, and the share of runs that never settle in the
band within 200 steps is 25.6 / 20.1 % for LN MSE 3000 against 3.5-11.8 % for the others (static norm 3.6 / 3.5 %).
Settling time = first step after which |x_k - r| <= 0.05 |x0 - r| for the rest of the run (`phase_core.settle_time`).

## 4.4 Robustness of the Lambda* threshold
- Leave one seed out (LN MSE 3000): the threshold is re-derived from the other 9 seeds (geometric midpoint of the largest
  non-trapped and smallest trapped Lambda*) and applied to the held-out seed's 3 runs. Thresholds 11.1 to 18.2 (12.1 in 7
  of 10 folds). Result: 11 of 12 traps flagged, 0 false alarms in 18 non-trapped runs. The one miss: holding out seed 1 raises
  the threshold to 18.2, and seed 1 repeat 0 (Lambda* 17.0, the smallest trapped value) falls just below it.
- Whole range of thresholds: every threshold above 8.61 (the largest Lambda* of the 160 non-trapped runs) gives 0 false
  alarms; every threshold in [8.61, 17.03) flags 17 of the 20 traps. So the reported 17/20 and 0/160 hold for any threshold
  in a factor-2 interval, not only at 12.1. Below 8.61 false alarms start (5 at 6.45, 17 at 3.37); the three missed traps
  (slope-from-0, Lambda* 2.49, 3.37, 6.45) cannot be caught without false alarms.

## What this changes in the claims
- Every rate in the slides and companions now has a seed-level interval (`out/stats/`); quote them as "26.5 % (95 % CI
  20-33 %, seed bootstrap)".
- Correct "two-stage beats no norm" to "9 of 10 seeds, p = 0.049, not significant after Holm" wherever it appears (slide
  notes of the scorecard and NOTES_dpc_level1.md "Paired" lines; I have not edited the slides).
- Lambda*: state the threshold interval [8.6, 17.0) with 17/20 hits and 0/160 false alarms, and the leave-one-seed-out
  11/12, 0/18.
