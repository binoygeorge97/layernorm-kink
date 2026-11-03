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

## EXPLORATORY / POST HOC (not pre-registered): why median of means failed (P3b), 2026-11-03
Script `analysis/mom_diagnose.py` (the mom warm start re-run with per-epoch logging; end costs 3.60, 3.51, 3.08, 2.71, 4.05,
2.64 for LN MSE s1, s2, s8, s9 and slope-from-0 s2, s9: the same 5-of-6 pattern as the harness), `out/opt/mom_diagnose.csv`;
follow-up `analysis/mom_moment_test.py`, `out/opt/mom_moment_test.csv`.

1. At the good controller (statistics of the first epoch, before any update) the explosion is widespread, not one start:
   starts with per-start gradient norm > 1e3: LN MSE s1 11, s2 498, s8 136, s9 61; slope-from-0 s2 62, s9 0 (its quiet end
   point). Groups above 10 x the median group norm: 1-4 of 10. Because the exploding starts sit in more than half of the 10
   groups, the median group mean is itself exploded: norm of the median-of-means gradient at epoch 1 = 7 (s1; but 3e42 at
   epoch 3), 2e31 (s2), 1e10 (s8), 2e6 (s9), 4e14 (slope s2). cos(mom, per-start clip 3) at epoch 1: -0.06, -0.03, +0.35,
   -0.55, -0.04. So: "many groups explode at once" holds at the start.
2. The coordinate-wise median is NOT biased when only one group explodes: over the 62 epochs with exactly one group above
   10 x the median, cos(mom, clip 3) has median 0.985 (0.948 for the mean of the clean groups); over the 434 epochs with none,
   1.000. The direction breaks only when >= 2 groups explode (104 epochs, median 0.32).
3. After the first epoch (all pairs except LN MSE s9, which keeps exploding intermittently) no start explodes, all 1000 starts
   have norms 3-40, and mom, the plain mean and the clip-3 direction coincide (cos 1.00), yet the true cost keeps rising
   smoothly. Follow-up (LN MSE s2, s8, 20 epochs): re-initialising AdamW after epoch 1 does not stop the rise (s2 2.64 -> 3.34,
   s8 2.61 -> 3.86, vs 3.34 and 2.99 without reset), so poisoned Adam moments are NOT the explanation; clipping only the
   epoch-1 gradient to norm 3 helps on s2 (2.77) but not s8 (3.98). Over epochs 2-20 the AdamW update has cosine -0.1 to
   -0.2 with the negative gradient in every variant: the steps do not descend although the aggregated gradients are tame and
   consistent. Not established here: why (step size vs local curvature is a candidate; see the lr 1e-4 1000-epoch run).

Answer to the question: the first one. At the good controller tens to hundreds of starts explode at once (LN MSE seed 2:
half of them), more than the median of 10 group means can reject; the median is not biased when only one group explodes.
What the warm start does after the first step is a separate, unexplained phenomenon (tame, agreeing gradients, rising cost).

Meaning for slide 18 ("a chaotic gradient, carried by one start"): the statement "one start holds 61-100 % of ||grad||^2"
remains true, but "carried by one start" overstates it. The gradient is dominated by the single largest start in the squared
norm, while the explosion itself involves 11 to 498 of the 1000 starts on the trapped LN MSE pairs. That is why per-start
clipping (bounds every start) works and a median over 10 groups (needs a majority of clean groups) does not. Proposed slide
text: out/slide_fixes.md, entry 12 (a proposal; nothing on the slide was changed).
Check for the "healthy: none" part (one per-start gradient evaluation at the good controller, healthy pairs): largest
per-start norm 2.96 (LN MSE s3), 5.29 (LN MSE s7; 18 starts above 3), 0.26 (two-stage s2), 1.86 (two-stage s9); no start above
1e3 on any healthy pair.
