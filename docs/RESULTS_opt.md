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

## EXPLORATORY / POST HOC: lr 1e-4 warm start for 1000 epochs (same total step budget as plain at 1e-3 for 100), 2026-11-03
`analysis/warmstart_harness.py --settings lr1e-4 --epochs 1000` (one process per pair, `out/opt/warmstart_lr1e-4_1000ep_*.csv`).
True-plant cost, start -> epoch 1000 (largest value at any logged epoch):
| pair | lr 1e-4, 1000 epochs | plain lr 1e-3, 100 epochs (for comparison) |
|---|---|---|
| LN MSE s1 | 2.556 -> 2.558 (2.574) | pushed off (3.85 at epoch 10, back to 2.60) |
| LN MSE s2 | 2.638 -> 2.656 (2.656) | 2.64 -> 2.89 |
| LN MSE s8 | 2.609 -> 2.648 (2.648) | 2.61 -> 5.25 |
| LN MSE s9 | 2.639 -> 2.651 (2.651) | 2.64 -> 4.70 |
| slope-from-0 s2 | 2.638 -> 2.638 (2.644) | 2.64 -> 5.87 |
| slope-from-0 s9 | 2.639 -> 2.655 (2.655) | 2.64 -> 4.59 |
| LN MSE s3 (healthy) | 2.419 -> 2.429 (2.448) | stays |
| LN MSE s7 (healthy) | 2.597 -> 2.602 (2.603) | stays |
| two-stage s2 (healthy) | 2.638 -> 2.638 (2.639) | stays |
| two-stage s9 (healthy) | 2.639 -> 2.681 (2.681) | stays |
(LN MSE s1, s3, s7 were also completed by the first attempt, stopped for time before it saved; its printed values are identical.)

Answer: "stays", not "slower". With ten times smaller steps and the same total step budget, the good controller is kept on all
6 broken pairs (largest rise +1.5 %, LN MSE s8). Together with item 2 (after the first step the gradients are tame and
consistent, yet lr 1e-3 AdamW steps climb), this points to the STEP SIZE relative to the very local structure near the good
controller: a 1e-3 relative step per weight lands where the surrogate's gradient field is chaotic, a 1e-4 step does not. It
does not answer pre-registered P4 (lr 1e-4 from scratch does not rescue), which is about reaching the good controller from a
random initialisation (Colab Run 4).

---------------------------------------------------------------------------------------------------------------------------
## Run 4: optimiser baselines from scratch (added 2026-11-04; frozen analysis 218cbbe, run unchanged)
Data: MyDrive/dpc_out after Run 4 (dpc_results.csv modified 21:52:41Z, dpc_diag.csv 21:53:06Z), copied to data/run4.
Command: `LK_OUT=out/run4 python analysis/opt_report.py data/run4` (outputs out/run4/opt/).
Checks:
- Every Run 4 setting has 30 rows (10 seeds x 3 repeats), with no duplicates.
- The 210 pre-existing result rows and 180 diag rows are bit-identical to the pre-Run-4 backup (be5e1ca).
- The GRU oracle files are unchanged on Drive (modified 21-22 Oct, sizes equal to the local copies).

GRU through the LN MSE 3000 surrogates; rescued = not failed in >= 2 of 3 repeats, over target seeds 1, 2, 8, 9.
| setting | failed / 30 | trapped / 30 | median gap | best-seed gap | wall time (s per seed) | target seeds rescued |
|---|---|---|---|---|---|---|
| plain | 13 | 12 | 4.05 % | 0.17 % | 28 | 0 |
| per-start clip 3 | 5 | 0 | 1.23 % | 0.13 % | 120 | 3 (not s9) |
| unit per-start | 0 | 0 | 0.94 % | 0.15 % | 121 | 4 |
| global clip 3 | 5 | 2 | 1.98 % | 0.18 % | 28 | 3 (not s1) |
| median of means | 10 | 8 | 1.85 % | 0.18 % | 120 | 1 (s9) |
| batch 100 | 15 | 11 | 12.4 % | 0.21 % | 6 | 0 |
| lr 1e-4 | 3 | 3 | 2.16 % | 0.23 % | 28 | 4 |
| N <= 50 | 17 | 15 | 12.9 % | 2.93 % | 6 | 0 |
| N <= 100 | 15 | 12 | 8.23 % | 0.19 % | 10 | 0 |
| two-stage, plain (reference) | 0 | 0 | 0.04 % | 0.006 % | 28 | 4 |
The failed per-start-clip runs are not trapped: they are model mismatch, gaps 20-55 %, on seeds 5, 8 and 9.

Pre-registered outcomes (PREREG_opt, c41967c; convergence rule 84ecb3a):
| id | prediction | outcome | numbers |
|---|---|---|---|
| P1 | per-start clip 3 rescues >= 3 of 4 | held | 3 of 4 (seeds 1, 2, 8; seed 9 failed in 2 of 3) |
| P2a | global clip 3 rescues <= 1 | failed | 3 of 4 (seed 1 failed in 2 of 3); 2 traps in 30 |
| P3a | median of means rescues >= 3 | failed | 1 of 4 (seed 9); 8 traps in 30 |
| P4 convergence | healthy seeds 0, 3, 4, 6, 7 reach <= 10 % in >= 2 of 3 | converged | 15 of 15 healthy runs <= 10 % (largest gap 2.3 %) |
| P4 | lr 1e-4 rescues <= 1 | failed | 4 of 4; 3 traps in 30 (s8 r0, s2 r2, s9 r2) |
| P5a | N <= 50: <= 3 of 30 trapped | failed | 15 of 30 trapped (more than plain, 12) |
| P5b | N <= 50 worse than two-stage plain | held | median gap 12.9 % vs 0.04 % |
| U1 | batch 100: no prediction | reported | 15 failed, 11 trapped, 0 of 4 targets rescued |
Paired against plain (per-seed median gap, Wilcoxon over 10 seeds): unit better on 9 of 10 (p = 0.004), clip 3 8/10
(p = 0.11), lr 1e-4 7/10 (p = 0.065), median of means 7/10 (p = 0.084), global clip 6/10 (p = 0.19); two-stage 10/10 (p = 0.002).

Reading:
- Bounding each start's influence works from scratch. Unit per-start gradients rescue all four target seeds with no trap
  and no failure in 30 runs. Per-start clip 3 leaves no trap; its 5 failures are model mismatch.
- Two predictions based on "one start sets the direction" failed. Global clipping rescues 3 of 4, and a learning rate 10
  times smaller rescues all 4 (after converging, by the registered rule). So the size of the steps matters too, against
  P2a and P4 and the warm-start results (where lr 1e-4 also stayed).
- Median of means mostly fails (1 of 4), as the B2 diagnosis explained: many starts explode at once.
- Truncating the horizon to N <= 50 or <= 100 does not avoid the trap: 15 and 12 trapped. P5a failed.
- Minibatches of 100 make it worse: 15 failed.
- The repair on the surrogate side remains the strongest: two-stage training, 0 of 30 failed.
Committed claims affected (your decision; not changed): CLAIMS.md / slides that say "global clipping does not help" (CLAUDE.md
1.6 lists it as not yet claimable) and anything implying a smaller learning rate does not help.

## S2: Run 4 follow-ups (added 2026-11-04; ALL POST HOC, written after the Run 4 results were read)
Script `analysis/run4_followups.py` (data/run4) -> out/run4/followups/.

Note on P5a, dated 2026-11-04: P5a is CONFOUNDED until the matched control in (b) is run. The "failed" outcome above stands
as scored by the frozen script, but it should not be read as "truncation does not avoid the trap".

(a) Failed and trapped runs (seed s, repeat r). Healthy seeds are 0 and 3-7.
| setting | failed | trapped | healthy seeds trapped? |
|---|---|---|---|
| plain | s1 r0-2, s2 r0-2, s5 r0, s8 r0-2, s9 r0-2 | the same without s5 r0 | no |
| per-start clip 3 | s5 r0, s5 r1, s8 r0, s9 r1, s9 r2 | none | no |
| unit | none | none | no |
| global clip 3 | s1 r0, s1 r1, s5 r0, s5 r1, s9 r0 | s1 r0, s1 r1 | no |
| median of means | s1 r0-2, s2 r0, s2 r2, s5 r0, s5 r1, s8 r0, s8 r1, s9 r0 | the same without s5 | no |
| batch 100 | s1 r0, s1 r2, s2 r0-2, s5 r0, s5 r1, s6 r0, s6 r1, s8 r0-2, s9 r0-2 | s1 r0, s1 r2, s2 r0-2, s8 r0-2, s9 r0-2 | no (s5, s6 fail, not trapped) |
| lr 1e-4 | s2 r2, s8 r0, s9 r2 | the same | no |
| N <= 50 | s0 r0, s1 r0-1, s2 r0-1, s4 r2, s5 r0-2, s6 r0, s6 r2, s7 r0, s8 r0-2, s9 r1-2 | all but s5 r1, s5 r2 | YES: 6 runs (s0 r0, s4 r2, s5 r0, s6 r0, s6 r2, s7 r0) |
| N <= 100 | s1 r0-2, s2 r0-2, s5 r0-1, s6 r0, s8 r0-2, s9 r0-2 | s1, s2, s8, s9 all repeats | no (s5, s6 fail, not trapped) |
| two-stage plain | none | none | no |
Under N <= 50, 6 healthy-seed runs carry the trapped label. That is a warning sign that, for this setting, the label measures
the horizon mismatch (see b) rather than a trap.

(b) Why P5a (and the trapped counts for N <= 50, N <= 100, batch 100) is confounded. "Trapped" compares DPC's own
controller with the oracle trained at the full horizon (N = 200) on the full batch, both priced through the surrogate at
N = 200. A controller trained only up to N = 50 or on noisy minibatches can lose to that oracle on a non-kinked model too,
purely from horizon or noise mismatch. Cheapest matched control: train the oracle with the same setting through the TRUE
plant ("true plant (oracle), N <= 50", "..., N <= 100", "..., batch 100"; 3 settings x 10 seeds x 3 repeats). Then
(i) the gap of each setting against its matched oracle separates optimiser damage from the horizon/noise penalty, and
(ii) the trapped label is recomputed with the matched oracle priced through the surrogate. Cost on Colab A100, from the
logged wall times (6, 10, 6 s per seed): about 3 + 5 + 3 min of training plus about 5 min of evaluation and the diagnostic,
so ~15-20 min. It needs a small code change: one model list with kind "true" plus those options (no change to defaults;
regression by QUICK). Second option (also ~15 min): the same three settings through the two-stage surrogate, where the
plain optimiser gives 0/30; any "trapped" label there would be mismatch, not a trap. The matched oracle is the better test,
because it repairs the label itself.

(c) Paired per-seed comparisons (median gap over the 3 repeats; Wilcoxon signed-rank over 10 seeds; Holm over these 8).
| comparison | median gaps | seeds where the first is better | p | p (Holm) |
|---|---|---|---|---|
| two-stage plain vs unit | 0.04 % vs 0.94 % | 9 of 10 | 0.020 | 0.078 |
| two-stage plain vs per-start clip 3 | 0.04 % vs 1.32 % | 10 of 10 | 0.002 | 0.016 |
| two-stage plain vs global clip 3 | 0.04 % vs 2.14 % | 10 of 10 | 0.002 | 0.016 |
| two-stage plain vs lr 1e-4 | 0.04 % vs 2.42 % | 10 of 10 | 0.002 | 0.016 |
| plain vs unit | 3.63 % vs 0.94 % | 1 of 10 | 0.004 | 0.020 |
| plain vs per-start clip 3 | 3.63 % vs 1.32 % | 2 of 10 | 0.11 | 0.21 |
| plain vs global clip 3 | 3.63 % vs 2.14 % | 4 of 10 | 0.19 | 0.21 |
| plain vs lr 1e-4 | 3.63 % vs 2.42 % | 3 of 10 | 0.065 | 0.19 |
(Per-seed medians over repeats, so these medians differ from the run-level medians in the Run 4 table.) After Holm, two-stage
plain beats clip 3, global clip 3 and lr 1e-4; against unit it is better on 9 of 10 seeds but p (Holm) = 0.078. Only unit
beats plain after Holm.

(d) Wall time (median train_s_per_seed, s per seed, A100): plain 28, clip 3 120, unit 121, global clip 3 28, median of
means 120, batch 100 6, lr 1e-4 28, N <= 50 6, N <= 100 10, two-stage plain 28. The column is in the Run 4 table above and
in paper/tables/tab_optimiser.tex ("s per seed"). Per-start schemes cost about 4x, because they need per-start gradients.

(e) The 5 per-start-clip failures are model mismatch, not traps. Through the surrogate, the clip-3 controller costs LESS
than the oracle controller on every one; through the true plant it costs 20-55 % more.
| run | gap | own / oracle, through the surrogate | own, oracle through the true plant |
|---|---|---|---|
| s5 r0 | 39.7 % | 2.539 / 2.571 = 0.988 | 3.488, 2.498 |
| s8 r0 | 38.6 % | 2.474 / 2.560 = 0.966 | 3.462, 2.498 |
| s5 r1 | 54.6 % | 2.539 / 2.570 = 0.988 | 3.861, 2.498 |
| s9 r1 | 21.9 % | 2.445 / 2.468 = 0.991 | 3.044, 2.498 |
| s9 r2 | 20.1 % | 2.445 / 2.471 = 0.990 | 2.999, 2.498 |
So per-start clipping finds the surrogate's own optimum; the surrogate is wrong there. (Plain also fails s5 r0 the same way.)

(f) Hypothesis (user's): global clipping helps from scratch but not in the warm start because, without clipping, an exploded
gradient inflates Adam's second-moment estimate v (it decays only with time constant 1/(1 - beta2) = 1000 steps), so the
effective step lr / sqrt(v) is tiny for ~1000 epochs and learning stalls in the trap. Global clipping bounds what enters v.
In the warm start the failure is different: the controller starts at the optimum and is pushed off by the first few
direction-chaotic steps, before any stall can matter; clipping the size does not fix the direction.
Cheapest test: rerun from scratch, plain vs global clip 3, seeds 1, 2, 8, 9, rep 0 (8 runs), logging every 10 epochs
max(v), median(v), the update norm, and ||grad|| before clipping. Prediction: in plain runs that end trapped, median(v)
jumps by orders of magnitude at the first explosion and the update norm collapses for hundreds of epochs; in the global-clip
runs v stays bounded and the update norm does not collapse. Needs a local script (logging hook around the existing trainer;
no notebook change). No Colab needed: CPU ~3-4 h in the background (4 seeds vmapped, full 9000-epoch curriculum, two
settings); on Colab it would be ~10 min but needs a notebook rebuild. I have not run it.
