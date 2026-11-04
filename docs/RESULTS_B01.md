# Task 2a: the scalar plant with ordinary authority, B = 0.1 (Run 6), side by side with B = 0.01

Written 2026-11-04 (Q4). Data: Run 6 (`MyDrive/dpc_out_B01`, dpc_results.csv modified 17:58:58Z, dpc_diag.csv 17:59:21Z),
copied to `data/run6_B01`; surrogates, labels and GRU oracles from Run 3 (Stage A). Analysis: the frozen
`analysis/b01_report.py` (180ec7b), run unchanged: `python analysis/b01_report.py --root data/run6_B01`; outputs and figures
in `out/B01/report/` (report.md, fig_lqr_rates.png, fig_setpoint_law.png, fig_dpc_traps.png, fig_lambda_scatter.png).
Pre-registrations: PREREG_B01 Part 1 (8ecc750) and Part 2 (77d8455, direct-gain Lambda*, primary), PREREG_B01_aug Part 2
(91238b4, augmented Lambda*_aug, secondary; committed after Run 6 began writing, before any Run 6 file was read).

## Checks
- Every family has exactly 30 rows (10 seeds x 3 repeats), no duplicates, a dpc_diag value for each; no rows of any other
  model (the robust block did not run in Run 6).
- B = 0.1 oracle controllers re-fetched from Drive: parameter hashes equal the local copies used for both pre-registrations
  (repeat 0 64200edf..., repeat 1 c9b0a7ac..., repeat 2 02717862...).

## DPC (GRU), failed / trapped of 30
| family | B = 0.01 failed | B = 0.01 trapped | B = 0.1 failed | B = 0.1 trapped | B = 0.1 median gap |
|---|---|---|---|---|---|
| LN MSE 3000 | 13 | 12 | 7 | 7 | 1.6 % |
| LN MSE 12000 | 0 | 0 | 0 | 0 | 0.04 % |
| LN slope from 0 | 8 | 8 | 0 | 0 | 0.02 % |
| LN two-stage | 0 | 0 | 0 | 0 | 0.02 % |
| no norm | 0 | 0 | 0 | 0 | 0.06 % |
| static norm | 0 | 0 | 0 | 0 | 0.03 % |

B = 0.1 trapped runs: LN MSE 3000 seed 1 (repeats 0, 1, 2: gap 10 448 %, 493 %, 20 %), seed 2 (0, 1, 2: 6 425 %, 33 749 %,
3 939 %) and seed 9 repeat 2 (615 %). Own controller vs oracle through the surrogate: 105, 6.1, 1.2 (seed 1), 63, 328, 38
(seed 2), 6.8 (seed 9 r2). Seeds 1 and 2 are the two narrowest B = 0.1 LN MSE 3000 kinks (r*_x 0.125 and 0.117) and have
the largest peak input-slope errors (5.2 and 12.2 B; PREREG_B01 Part 1 table). Every failed run is trapped.

## Scoring of the pre-registered predictions
| id | outcome | numbers |
|---|---|---|
| Q1, Q2, Q3 (Part 1) | held | scored 3 Nov (PREREG_B01) |
| R1 (direct-gain Lambda* > 12.1, primary) | failed (hit condition) | predicted 1 (LN MSE s2 r2), trapped 7: 1 hit, 6 misses, 0 false alarms; hit rate 0.14 < 0.75, false-alarm rate 0 <= 0.05. The registered rule says "held if both"; the frozen script labels one-of-two "partly" (the convention of PREREG_arch), so the report reads "partly". By the PREREG_B01 text it is not held. |
| R2 (two-stage, no norm, static norm: no trap) | held | 0 of 90 |
| aug rule (Lambda*_aug > 12.08, secondary) | failed | predicted 0, trapped 7 (rule: held only if at most 1 trapped) |
Misses with their predictor values: direct-gain Lambda* 2.55, 3.14, 2.07 (seed 1 r0, r1; s2 r0 3.14), 11.45 (s2 r1), 10.12
(s1 r2), 6.45 (s9 r2); augmented 2.30 to 9.12. Two misses lie just below 12.1 (11.45, 10.12).

## LQR (Part 1, from 3 Nov, repeated by the frozen report)
| family | fixed failed / catastrophic, B = 0.01 | B = 0.1 | rescheduled failed / catastrophic, B = 0.01 | B = 0.1 |
|---|---|---|---|---|
| LN MSE 3000 | 26.5 / 10.8 % | 25.1 / 5.6 % | 20.5 / 3.4 % | 15.5 / 3.9 % |
| LN MSE 12000 | 6.5 / 1.1 % | 1.5 / 0.9 % | 3.5 / 0.2 % | 0.8 / 0 % |
| LN slope from 0 | 5.8 / 1.4 % | 1.4 / 0 % | 2.8 / 0 % | 0.6 / 0.1 % |
| LN two-stage | 5.7 / 0.9 % | 0.6 / 0 % | 3.5 / 0.1 % | 0.3 / 0 % |
| no norm | 2.6 / 0 % | 0 / 0 % | 5.7 / 3.6 % | 0 / 0 % |
| static norm | 0.6 / 0 % | 0 / 0 % | 0.8 / 0 % | 0 / 0 % |
Setpoint law (|B_hat/B - 1| against the setpoint distance d in half-widths, four LN families pooled, the script's fit):
0.39 d^-0.88 at B = 0.01, 0.053 d^-0.72 at B = 0.1. Path colour at B = 0.1: almost no LN path with an accurate setpoint gain
crosses wrong-sign slope (8 and 2 runs in the 0-2 % bin, 8 in > 10 %), so the path-colour table is not informative at B = 0.1.

## CLAUDE.md decision rule
Traps do not vanish at B = 0.1: 7 of 30 LN MSE 3000 runs are trapped, on the narrowest-kink seeds, while every kink-free and
slope-trained family is clean. The claim is therefore NOT restricted to low-authority plants, and the two-state plant (Task 2b)
need not be low authority on these grounds. What changes with authority: fewer traps (12 -> 7 on LN MSE 3000) and none on
slope-from-0 (8 -> 0), consistent with the 10x smaller relative slope errors (Q1); the fixed LQR still fails on a quarter of
LN MSE 3000 runs.

## For the paper / corrections
- Lambda*, the landing-amplification statistic computed from the oracle, did not predict the B = 0.1 traps prospectively
  (1 of 7). With Run 5 (MLP 0 of 40, linear 12 false alarms) it has now failed three prospective tests; its GRU B = 0.01
  result (17/20, 0/160) is retrospective. Any claim that Lambda* predicts traps must be removed or labelled retrospective
  (CLAIMS.md, NOTES_theory.md, slides 17-19, companion Theorem 2 text): a change of committed claims, for you to decide.
- The trap itself replicates at B = 0.1 (same family, narrowest-kink seeds), and two-stage training removes it again.
