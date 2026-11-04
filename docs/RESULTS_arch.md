# Task 1.5: MLP and linear DPC controllers (Run 5), scored against PREREG_arch

Written 2026-11-04. Data: Run 5a (MLP, `MyDrive/dpc_out_mlp`, dpc_results.csv on Drive 17:54 CT 3 Nov) and Run 5b (linear,
`MyDrive/dpc_out_linear`, ~20:59 CT 3 Nov), copied to `data/run5_mlp`, `data/run5_linear` (tables, plus the controller files
used below). Script `analysis/arch_report.py` (outputs `out/arch/`), `analysis/linear_landscape.py` (A4),
`analysis/spotcheck_run5.py` (A3 check). Pre-registrations: Part A (MLP) f945a5c, Part B (linear) 322e353; the predicted
CSV is unchanged since those commits (checked by hash of the rows).

## Data checks (A3 of 4 Nov)
- Each of the 7 main models has exactly one row per seed x repeat (10 x 3 = 30), in both folders; no duplicates.
- RUN_ROBUST was left True: "LN, MSE (3000), clipped grads" and "... unit grads" have 10 rows each, all repeat 0. They are
  excluded from every table and confusion matrix below (they are separate model names; the scoring filters on the six
  registered families). surrogates.csv grew because the two robust names got their own 20 rows, which are copies of the
  LN MSE 3000 rows (same surrogate files).
- Oracle rows are the Stage A rows (max abs difference 4.5e-13, CSV round trip), and the oracle files keep their hashes.
- Spot check: LN MSE 3000 MLP repeat 0 and linear repeat 1 re-evaluated on the true plant (and through their own surrogate)
  from the downloaded controller files: cost_test_true, cost_test_own, n_diverged and per-start costs agree to <= 1.1e-14
  relative on all 20 seed rows.

## Counts (trapped = typical-start gap > 10 % AND own controller > 1.05 x oracle controller through the same surrogate)
| family | MLP failed / 30 | MLP trapped / 30 | linear failed / 30 | linear trapped / 30 | GRU trapped / 30 |
|---|---|---|---|---|---|
| LN MSE 3000 | 27 | 27 | 0 | 0 | 12 |
| LN MSE 12000 | 1 | 1 | 0 | 0 | 0 |
| LN slope from 0 | 12 | 12 | 0 | 0 | 8 |
| LN two-stage | 0 | 0 | 0 | 0 | 0 |
| no norm | 0 | 0 | 0 | 0 | 0 |
| static norm | 0 | 0 | 0 | 0 | 0 |

Every failed MLP run is trapped (no model-mismatch failures). MLP trapped runs:
- LN MSE 3000: every seed except seed 4, in all three repeats (27 runs).
- LN slope from 0: seed 2 (r1, r2), 3 (r0, r1, r2), 4 (r0, r2), 8 (r1, r2), 9 (r0, r1, r2).
- LN MSE 12000: seed 1 repeat 2.
Median typical-start gap of the MLP on LN MSE 3000: 241 %. The linear controller never fails: largest gap 0.79 % (LN MSE 3000
seed 8 r0), largest own/oracle ratio through the surrogate 1.0075. Gaps are relative to the oracle of the same architecture
(linear oracle 2.5235, 1.1 % above the saturated LQR; MLP oracle 2.4972).

Paired against the GRU (per-seed median gap over repeats, Wilcoxon over 10 seeds):
| family | MLP better on | MLP p | linear better on | linear p |
|---|---|---|---|---|
| LN MSE 3000 | 2/10 | 0.13 | 10/10 | 0.002 |
| LN MSE 12000 | 7/10 | 0.32 | 10/10 | 0.002 |
| LN slope from 0 | 4/10 | 0.43 | 10/10 | 0.002 |
| LN two-stage | 4/10 | 0.70 | 10/10 | 0.002 |
| no norm | 4/10 | 0.28 | 10/10 | 0.002 |
| static norm | 3/10 | 0.19 | 10/10 | 0.002 |
(The linear "wins" partly reflect that its gap is measured against its own, 1.1 % worse oracle.)

## Confusion matrices of the pre-registered predictions
P1 (MLP, f945a5c): trapped iff Lambda* of the MLP oracle > 12.1; predicted 0 of 180.
| | trapped | not trapped |
|---|---|---|
| predicted trapped | 0 | 0 |
| predicted not trapped | 40 (misses) | 140 |
Missed runs (all 40 trapped runs), Lambda* of the MLP oracle in brackets: LN MSE 3000 s0 (5.1, 1.6, 2.9 for r0, r1, r2),
s1 (1.2, 1.9, 3.1), s2 (5.4, 4.8, 5.6), s3 (3.6, 3.9, 2.8), s5 (2.3, 2.3, 2.3), s6 (0.8, 0.8, 0.8), s7 (4.3, 4.3, 4.3),
s8 (4.5, 4.0, 3.0), s9 (3.1, 5.5, 7.4); LN MSE 12000 s1 r2 (2.1); slope from 0 s3 r0 (3.7), s4 r0 (4.6), s9 r0 (2.5), s2 r1
(1.4), s3 r1 (2.4), s8 r1 (0.6), s9 r1 (3.9), s2 r2 (2.1), s3 r2 (3.4), s4 r2 (0.0), s8 r2 (3.2), s9 r2 (2.3). The largest MLP
oracle Lambda* over all 180 runs is 7.4, so no threshold separates.

P2' (linear, 322e353): trapped iff Lambda* of the trained linear oracle > 12.1; predicted 12 of 180 (LN MSE 3000 seeds 1, 2,
8, 9, every repeat).
| | trapped | not trapped |
|---|---|---|
| predicted trapped | 0 | 12 (false alarms) |
| predicted not trapped | 0 | 168 |
False alarms: LN MSE 3000 s1 (Lambda* 86.5, 92.0, 80.1 for r0, r1, r2), s2 (29.7, 35.0, 44.4), s8 (87.3, 93.4, 83.4),
s9 (75.5, 77.1, 53.2). Their gaps are 0.002 to 0.79 % and own/oracle ratios 0.994 to 1.0075: not near the trap criterion.

## Outcomes
| id | outcome | numbers |
|---|---|---|
| P1 (MLP, Lambda*) | failed | 0 predicted, 40 trapped (rule: held only if at most 1 trapped); hit rate 0/40, false-alarm rate 0/140 |
| A1 (MLP reproduces the GRU seeds) | partly | majority-of-repeats failures: LN MSE 3000 seeds 0, 1, 2, 3, 5, 6, 7, 8, 9 (listed 1, 2, 8, 9: all four, plus five more); slope from 0 seeds 2, 3, 4, 8, 9 (listed 2, 3, 9: all three, plus 4 and 8); two-stage, no norm, static norm: none (held); trap test > 1.05 on every failure (held). The listed seeds fail, but the seed lists do not match |
| P2 (linear, LQR-gain basis) | held | 0 of 180 trapped |
| P2' (linear, trained-oracle Lambda*) | failed | 12 false alarms, no trapped run (false-alarm rate 12/180 = 6.7 % > 5 %; hit rate undefined) |
| A2 (linear does not fail on GRU-trapped seeds) | held | 0 failed runs on LN MSE 3000 seeds 1, 2, 8, 9 and slope-from-0 seeds 2, 3, 4, 9 |
| A3 (MLP policy shapes) | partly | healthy: slope at x = 0.5 is 9.2 to 11.1 (two-stage, median over repeats), at x = 0 6.3 to 21.4, i.e. roughly the LQR line (held, range wider than "10 to 17"). Trapped: NOT "flat or inverted inside the kink band, LQR slope outside": the trapped LN MSE 3000 policies are near-constant saturated inputs over most of [-1, 1] (repeat 0: seed 1 u = +9.8 to +9.9 from x = -1 to +0.5; seed 8 u = -8.5 to -10 from x = -1 to +2; seed 2 u = -8.7 to -9.9 from x = -1 to +0.2, with a bump to +0.6 at x = 0.5), K_eff about 0 at x = 0 and 0.5 (failed for the trapped shape) |
| A4 (linear loss minimum through the surrogate within ~10 % of the true K on every seed) | failed (narrowly) | LN MSE 3000: 9 of 10 seeds within 10 % (seed 1: argmin 14.25 vs 16.75, -14.9 %); for comparison two-stage 9 of 10 (seed 9 -16.2 %), static norm 10 of 10. The tanh-linear loss is flat near its minimum, so the argmin moves easily |
| A5 (wrong-sign share of dL/dK) | not scored | needs the D11 gradient-audit figure / data from Drive, not fetched |
| A6, A7 | not run | as registered |

K_eff = -du/dx (median over repeats; controllers downloaded for LN MSE 3000 and two-stage, all repeats, both architectures,
and the oracles):
| seed | MLP LN MSE 3000, x = 0 | MLP two-stage, x = 0 | MLP oracle, x = 0 | linear LN MSE 3000 | linear two-stage | linear oracle |
|---|---|---|---|---|---|---|
| 0 | -2.2 | 10.4 | 10.5 | 18.1 | 16.6 | 16.9 |
| 1 | 0.4 | 7.0 | 10.5 | 15.7 | 16.6 | 16.8 |
| 2 | -2.2 | 9.8 | 10.6 | 16.9 | 17.0 | 16.9 |
| 3 | 0.0 | 9.0 | 10.6 | 17.5 | 16.5 | 16.8 |
| 4 | 23.0 | 9.9 | 10.5 | 17.3 | 17.0 | 16.9 |
| 5 | -0.0 | 10.7 | 10.5 | 17.4 | 16.5 | 16.9 |
| 6 | -0.1 | 10.1 | 10.7 | 16.8 | 16.7 | 16.9 |
| 7 | 0.1 | 9.1 | 10.6 | 16.9 | 17.3 | 16.9 |
| 8 | 0.0 | 21.4 | 10.5 | 16.4 | 18.8 | 16.9 |
| 9 | 0.1 | 6.3 | 10.5 | 16.1 | 13.9 | 16.9 |
No linear run is trapped, so there is no trapped linear seed for a D10 landscape; the landscape for all LN MSE 3000 seeds is
in `out/arch/linear_landscape.csv` (A4 above).

## Exploratory, not pre-registered: the two robust models left on in Run 5 (repeat 0 only)
RUN_ROBUST was left True, so Run 5 also trained LN MSE 3000 with per-start gradient clipping (norm 3) and with unit
per-start gradients, repeat 0 only. These rows are not part of any table above. Typical-start gap (%) per seed 0-9:
| architecture, aggregation | seed 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | failed / 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP, plain (main run, rep 0) | 220 | 363 | 387 | 2133 | 0.2 | 12.6 | 459 | 17.9 | 366 | 93.8 | 9 |
| MLP, per-start clip 3 | 5.0 | 1.4 | 1.5 | 0.8 | 0.2 | 11.3 | 8.0 | 0.1 | 2.3 | 9.5 | 1 |
| MLP, unit per-start | 1.2 | 5.4 | 0.2 | 0.8 | 0.5 | 11.5 | 2.7 | 1.4 | 0.3 | 3.3 | 1 |
| linear, plain (main run, rep 0) | 0.1 | 0.1 | 0.1 | 0.0 | 0.1 | 0.1 | 0.0 | 0.1 | 0.8 | 0.0 | 0 |
| linear, per-start clip 3 | 0.1 | 0.4 | 0.1 | 0.0 | 0.1 | 0.1 | 0.0 | 0.1 | 0.2 | 0.0 | 0 |
| linear, unit per-start | 0.1 | 0.5 | 0.1 | 0.0 | 0.0 | 0.1 | 0.0 | 0.0 | 0.2 | 0.1 | 0 |
For the MLP, per-start clipping from scratch brings 9 failed seeds down to 1 (seed 5, 11.3 %); dpc_diag.csv has no rows for
these models, so trap status is not available. One repeat, not registered: a hint for PREREG_opt P1 (which concerns the GRU,
Run 4), not evidence.

## What this means for the claims
1. The trap is not an artefact of the recurrent policy: a memoryless MLP is trapped MORE often than the GRU (27 vs 12 of 30
   on LN MSE 3000; 12 vs 8 on slope-from-0), on the same surrogates, with the same kink-free controls clean. The two-stage
   surrogate removes it for the MLP as for the GRU (0 of 30).
2. The 2-parameter linear controller is never trapped, on any surrogate, including those where its own oracle's landing
   amplification is 30 to 93. So a large Lambda* is not sufficient for a trap; whether the optimiser can be pulled away
   depends on the policy class (a 2-parameter tanh-linear policy has a flat, well-located loss minimum, A4).
3. Lambda* as a run-level predictor failed prospectively in both directions: for the MLP it missed every trap (the MLP
   oracle never amplifies: Lambda* <= 7.4 on all 180 runs), for the linear controller it raised 12 false alarms. The GRU
   result (17/20 hits, 0/160) does not transfer to other architectures. The paper may say: "the kink traps gradient-based
   DPC for recurrent and memoryless neural policies; the oracle's landing amplification predicted the GRU traps
   retrospectively, but did not predict MLP or linear traps prospectively". It may not say Lambda* is a general trap
   predictor.
4. Places that need correcting once this is accepted: CLAIMS.md (Lambda* claims, "MLP/linear" claim), NOTES_theory.md
   (predictions for E-A1/E-A2), the slide deck's Theorem 2 / Lambda* slides and the companion text on Lambda* (to be listed
   in the correction pass; nothing edited yet).
