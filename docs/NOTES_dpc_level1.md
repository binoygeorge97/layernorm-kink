# DPC through the Level 1 surrogates (set up 2026-10-21; full run + repeats on Colab A100, 2026-10-21/22)

Notebook: `claude/dpc/dpc_through_surrogates.ipynb`, built by `claude/dpc/build_dpc_notebook.py` from
`claude/dpc/level1_core.py` (unchanged copy of `claude/level1/level1_core.py`), `claude/dpc/dpc_core.py`,
`claude/dpc/dpc_report.py`. Rebuild the notebook after any code change.

## Design (agreed with the user, "go" on all defaults)
- Follows the user's DPC example (GRU controller, hidden 64; LQR-type loss Q_x = 5, R_u = 0.1, Q_f = 50, / N;
  curriculum 5/10/20/50/100/200 with 1000/2000/2000/1000/1000/2000 epochs = 9000; AdamW 1e-3 (optax default wd 1e-4);
  1000 x0 ~ U[-3, 3], full batch; test on the true plant; controller trained through the true plant = oracle reference).
- Changes from the example: rollout through the WHOLE surrogate (the example extracted W, V after LayerNorm, which is not
  the model); u = 10 tanh(head) (data and actuator |u| <= 10); plain JAX + optax, vmapped over 10 seeds; per seed every
  model's controller has the same init (key 20000 + s) and the same 1000 x0 (key 30000 + s) -> paired.
- Task: regulation to x = 0 (ends at (0, 0), within ~0.1 of every z_min: most kink-exposed). Plant A = 1.03 only.
- Models: oracle; LN MSE 3000; LN MSE 12000; LN J_x + J_u 3000; LN two-stage; no norm; static norm. Surrogates retrained
  with the Level 1 code (seed 0 MSE 3000 reproduced the Kaggle numbers to every digit: 11.3558 x B, rel-MSE 1.2545e-6).
- Evaluation on the true plant, 200 steps: 1000 typical starts (key 12345) -> example's true_lqr_cost; 22 starts
  +-{0.01 ... 3}; per start J, J_dyn (about the controller's own resting point), offset x_inf (mean of last 20 steps);
  gaps vs the oracle on the same seed: gap_test, gap_small (|x0| <= 0.2), gap_small_dyn, gap_large (|x0| >= 1),
  gap_worst (transient only); diverged (|x| > 10), not settled (|x_200| > 0.05); share of (x_k, u_k) inside the kink
  ellipse; cost on own model vs true plant. Sanity: oracle vs analytic LQR with saturation (K = 10.29): 2.497 on the
  1000 starts.
- Figures D1 headline strips, D2 closed loops (example's plot_closed_loop), D3 |x(t)| (example's plot_norm_comparison),
  D4 (x, u) trajectories over the surrogate's gain map + kink ellipse, D5 DPC training curves, D6 kink size vs gap.
- Pre-registered: P1 LN MSE worse than kink-free mainly from small starts; P2 two-stage within the kink-free range;
  P4 MSE 12000 in between, failing mainly on seeds 1, 2, 8, 9.

## Engineering notes
- Backprop through 200 steps x 1000 starts x 10 seeds needed 13.5 GB -> jax.checkpoint on the scan step (only (h, x)
  stored). CPU (2 cores) full-size epoch at N = 200: 6.7 s float64, 3.4 s float32. Whole curriculum ~ 3075
  N=200-epoch equivalents -> ~6 h per model on CPU; GPU estimate 1-3 h for all 7 models.
- GPUs (user, 2026-10-21): Kaggle now only offers T4; Colab offers L4, G4, A100. Recommended: Colab A100 with
  DTYPE = "float64" (fast FP64, same precision as Level 1). L4 / G4 / T4: DTYPE = "float32"; the notebook then sets
  jax_default_matmul_precision = "highest" (no TF32: ~3 significant digits is too coarse near the kink, divisor ~0.03)
  and writes to dpc_out_f32 so float32 and float64 results never mix. Surrogates are always trained in float64.
  Quick test passed in both precisions (float32: 3.6 min on CPU, oracle +7.3 % vs analytic LQR at 120 epochs).
  Results differ between precisions run to run (chaotic training), so compare models only within one precision.
- Quick-mode test (2 seeds, short surrogates, 120 controller epochs) already showed on seed 0 the LN MSE controller
  spiralling inside the kink on the true plant while two-stage and static norm did not (not evidence: undertrained).

## RESULTS - full run (Colab A100, float64, 2026-10-21 ~22:45-23:15, ~35 s per seed per model; `dpc_out_backup.zip`)
Medians over 10 seeds unless stated; gap = true-plant cost / oracle's on the same seed - 1.

Checks
- C1 surrogates = Level 1: slope-trained ones identical to 1e-13; MSE-trained ones identical on 8/10 seeds, but seeds 3, 4
  drifted on this GPU (e.g. two-stage seed 3 peak 0.631 vs 0.496 x B, no-norm seed 3 0.245 vs 2.22). Medians: LN MSE 18.67,
  MSE 12000 2.80 (Level 1 2.3; 12k drift known), slope 3000 7.92, two-stage 1.19 (1.15), no norm 0.77 (0.83), static 0.46.
  Comparisons inside this study are paired on these surrogates, so valid.
- C2 oracle DPC = analytic LQR with saturation: 2.4976 vs 2.4968 (+0.04 %), range 2.4976-2.4977. The GRU/DPC training is
  not the bottleneck; every gap is model error.

Main table (typical starts = 1000 x0 ~ U[-3, 3]; worst = worst of 22 starts, transient only)
| model | gap typical % median [IQR] (max) | gap large starts % | offset |x_inf| | diverged / not settled (of 220) |
|---|---|---|---|---|
| LN MSE 3000 | 17.1 [1.4, 270] (99 700, seed 9) | 5.7 | 0.083 | 20 / 111 |
| LN MSE 12000 | 0.15 [0.09, 0.20] (0.96) | 0.15 | 0.013 | 0 / 22 |
| LN J_x+J_u 3000 | 0.13 [0.06, 0.65] (223, seed 9; 74 seed 2) | 0.12 | 0.011 | 0 / 44 |
| LN two-stage | 0.051 [0.018, 0.11] (1.8, seed 9) | 0.046 | 0.0098 | 0 / 22 |
| no norm | 0.57 [0.28, 1.1] (3.9) | 0.55 | 0.027 | 0 / 37 |
| static norm | 0.11 [0.04, 0.63] (1.1) | 0.12 | 0.015 | 0 / 2 |
Oracle offset 0.0002.

Per seed, LN MSE 3000 typical gap %: 1.2, 282, 242, 2.1, 0.16, 30, 4.5, 0.22, 675, 99 694 (seeds 0-9).
-> catastrophic on EXACTLY seeds 1, 2, 8, 9 = the four narrowest kinks (training points inside: 6.8, 3.5, 20.8, 15.5; the
   5th narrowest, seed 7 with 23, is fine at 0.2 %). Spearman gap vs peak J_u error 0.66 (p = 0.04); vs points inside -0.54
   (p = 0.11, continuous); over all 40 LN model-seeds: gap vs peak 0.57 (p = 1e-4), small-start gap vs peak 0.64.

Paired (A vs B, typical-start gap; seeds A better; Wilcoxon p)
- LN MSE vs static: static better 10/10 (p = 0.002); vs no norm: no norm better 8/10 (p = 0.01). Also on large starts (0/10).
- two-stage vs LN MSE: better 10/10 (p = 0.002), every metric. Slope from step 0 vs LN MSE: 10/10 on typical (p = 0.002).
- two-stage vs static: tie (6/10, p = 0.38; median diff -0.05 pp); vs no norm: better 9/10 (p = 0.049).
- MSE 12000 vs MSE 3000: better 9/10 (p = 0.006); two-stage vs MSE 12000: 7/10, p = 0.23 (n.s.).

Mechanism (failing LN MSE seeds)
- Failure mode (a), not (b): the controller cannot control even its OWN model. Cost on own model ~= on true plant
  (seed 1: 9.61 vs 9.53; seed 2: 8.53 vs 8.54; seed 8: 19.26 vs 19.35; seed 9: 1861 vs 2492). DPC loss on own model stuck
  from the start of the horizon-200 phase (9.84, 8.68, 19.6 vs ~2.6 healthy); seed 9 blew up at horizon 200 (77 at h100 ->
  1790).
- The closed loop settles at a WRONG resting point away from 0 (offset 0.58 seed 1, 1.10 seed 2: x ~ 1.1 held with u ~ -3.3)
  or diverges (seed 9: early u ~ -7 for x < 0 pushes the wrong way, then u = +10 too late, x -> -90). Their trajectories
  spend ~0 % of steps inside the kink ellipse; healthy LN controllers spend 75-85 % (they rest at (0, 0), inside it).
- Fig D4 seed 2 (wide window): the wrong-gain region is a thin strip along u ~ 0 spanning |x| <= 1.5 (the far-field petal
  along the x-direction of a 201 x B kink). Regulation to 0 must end at u = 0, inside the strip. [H] gradients through the
  sign-flipped strip push the policy away; DPC settles in a local minimum that avoids it (holds x ~ 1.1 outside it).
  [2026-10-23: the "local minimum" half of this is WRONG - no barrier exists; see the trap-mechanism section at the end.]
- Slope loss from step 0 fails on seeds 2 and 9 (narrow tall kinks, peaks 28 and 18 x B, 3.1 and 2.2 points inside), same
  pattern (offsets 0.55, 1.06). Consistent with Level 1: it narrows rather than removes the kink.

Predictions
- P1 HOLDS, more strongly than stated: LN MSE worse than kink-free on typical (10/10 vs static) and large starts too. The
  "about the same from large starts" half was WRONG: the failure is global on narrow-kink seeds because the controller
  itself is broken (mode a), not only near-zero transients.
- P2 HOLDS: two-stage within the kink-free range (ties static, beats no norm 9/10). Seed 9 slightly off (1.8 %, offset 0.054).
- P4 WRONG as stated: MSE 12000 is fine on all 10 seeds (max 0.96 %). It is MSE 3000 that fails on seeds 1, 2, 8, 9.
  The narrow-kink seeds were right; the model was wrong (12 000 steps shrink those kinks enough for DPC).
- Small-start metric is weak: every model, kink-free included, has large relative gaps there (static 28 %, no norm 100 %
  transient-only) because the oracle's cost from |x0| <= 0.2 is tiny. Use typical / large / worst-start and failure counts.

Not established / next
- One controller init per seed: whether the LN MSE failures survive other controller inits (keys) or initial-state
  draws is untested -> rerun seeds 1, 2, 8, 9 x 3 controller seeds (cheap: ~4 models x 3 inits).
- Regulation at A = 1.03 only; setpoint tracking and A = 1.07 untested.
- Bug fixed after the run: kink_share columns were dropped from dpc_results.csv (header written by the oracle rows);
  now append_rows widens the file and RP.add_kink_share recomputes it from trajectories.

## Diagnostic on the full run, repeat 0 (computed locally 2026-10-22 from the saved controllers; no training)
Reference controller (trained through the TRUE plant) run through each surrogate, 1000 typical starts:
- On the failing LN MSE seeds the surrogate rates the reference controller like the plant does: 2.55 / 2.51 / 2.56 / 2.48
  (seeds 1, 2, 8, 9) vs 2.50 on the true plant; no divergence (a few not-settled small starts, offsets 0.03-0.06).
- The same surrogate rates the controller DPC found through it 3.8x, 3.4x, 7.5x, 750x WORSE than the reference controller.
- -> OPTIMISATION TRAP, not a misleading model: a good controller existed for the kinked surrogate and its predictions
  would have confirmed it; DPC did not find it. DPC learns from the surrogate's slopes (gradients through F), which the
  kink corrupts, while its predictions stay accurate. This is the paper's thesis in closed loop: MSE-invisible slope errors
  break gradient-based control even when predictions are fine.
- Slope loss from step 0, failing seeds 2 and 9: same pattern (own / reference on the model 1.75 and 3.3).
- Seed 5 of LN MSE (gap 30 %, counted as failed at the 10 % threshold): own / reference on the model 0.99 -> the model
  slightly prefers DPC's controller: mild model mismatch, not a trap.
- Healthy runs (all other models/seeds): own / reference on the model ~1.00 (DPC found an equally good controller).
- Fig D8 (figD8_diagnostic), Fig D7 (figD7_robustness) added. Extension notebook: REPS = [0, 1, 2] (repeat r: controller
  init key 20000 + s + 1000 r, x0 key 30000 + s + 1000 r; r = 0 = the original run, bit-identical, reused from the folder),
  run_diagnostic over all repeats; paired tests use the per-seed median over repeats (n = 10).

## RESULTS - repeats (REPS = 0, 1, 2; Colab A100 float64; `dpc_out_backup (2).zip`, 2026-10-22)
Each controller trained 3 times per seed (repeat r: controller init key 20000 + s + 1000 r, starts key 30000 + s + 1000 r).
- Runs failed (typical-start gap > 10 %), of 30: LN MSE 13 (seeds 1, 2, 8, 9 in all three repeats, seed 5 in one);
  slope from step 0 8 (seeds 2 and 9 in all three, 3 and 4 once); two-stage 0; MSE 12000 0; no norm 0; static 0.
- Paired, per-seed median over repeats (n = 10): two-stage beats LN MSE 10/10 (p = 0.002); LN MSE loses to static 10/10
  (p = 0.002); two-stage vs static tie (6/10, p = 0.43); two-stage beats no norm 9/10 (p = 0.049).
- Diagnostic over all repeats: 20 of the 21 failed runs are the optimisation trap (DPC's own controller rated > 1.05x the
  reference controller on the surrogate; up to 750x). The one exception is LN MSE seed 5 rep 0 (0.99: mild model mismatch).
  The reference controller run through each surrogate costs 0.93-1.06x its true-plant cost on the LN MSE surrogates.
- -> The failures belong to the surrogate (narrow-kink seeds fail in every repeat), not to unlucky DPC training.

## Phase portraits and the setpoint study (2026-10-22; user's idea: time-ordered (x_k, u_k) over the slope maps)
Code: `claude/dpc/phase_core.py`, `claude/dpc/phase_report.py`; notebook `claude/dpc/dpc_phase_tracking.ipynb` (built by
`build_phase_notebook.py`; reads MyDrive/dpc_out, adds phase_lqr.csv, track_*, figs/figP*).
- Figures: P1 per seed/surrogate, columns = setpoints 0 / 0.25 / 1, LQR fixed + rescheduled, each with strips (x - r and the
  surrogate gain along the true path); P2 atlas (6 surrogates x 4 setpoints); P3 gap / failure / offset vs setpoint;
  P4 setpoint gain error vs distance from the kink; P5 regulation DPC (true plant solid, own surrogate dashed; J_u and J_x
  maps); P6 failure vs share of wrong-sign steps on the path; P7 tracking DPC (after the Colab run).
- Overlays: true equilibria u = u_ss(x) (dotted), the surrogate's equilibria F(x, u) = x (dashed), setpoint star, surrogate's
  u_hat(r) diamond, kink ellipse. Symlog axes (linear within |x| <= 0.25, |u| <= 1).
- Observation: every LN surrogate's kink centre is within 0.1 of (0, 0) because the encoder bias starts at zero, so
  regulation to 0 puts the setpoint inside the kink by construction; moving r is a clean experiment.

Controllers (all from a surrogate, run on the TRUE plant, cost around (r, u_ss(r)) = (r, -3 r)):
- LQR fixed: linearise at the surrogate's own equilibrium for r (root of F(r, u) = r nearest the global linear fit's guess;
  up to 3 roots near the kink), K = DARE(A_hat, B_hat), u = clip(u_hat - K (x - r)). Only setpoint slopes matter.
- LQR rescheduled: relinearise at (x_k, u_{k-1}) every step. Path slopes matter.
- Tracking DPC: GRU input (x - r, r), 1000 pairs x0 ~ U[-3, 3], r ~ U[-2, 2], loss around the surrogate's u_hat(r);
  oracle = through the true plant. Keys 50000 + s (init), 60000 + s (pairs). Trained on Colab A100 2026-10-22 (repeat 0).
- Setpoints {-2, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 2} x 9 starts in [-3, 3] x 10 seeds; gaps exclude starts
  within 0.5 of r (tiny reference cost makes any offset look huge).

Findings (LQR, computed locally from the saved surrogates)
1. Setpoint placement (fig P3; % of the 80 runs per setpoint = 10 seeds x 8 starts >= 0.5 from r that fail: gap > 10 %
   or diverged). LQR fixed, LN MSE: 0, 0, 16, 41, 65, 66, 56, 34, 10, 1, 1 % at r = -2 ... 2 (peak at the kink, ~0 beyond
   |r| = 1). Static 0-1 %, no norm 0-4 % everywhere. Slope from step 0: 38 % at r = 0, 10-14 % at +-0.1, ~0 elsewhere.
   MSE 12000: 25 % at r = 0. Two-stage: 22 % at r = 0, 1-5 % elsewhere (its setpoint gain at the centre is still off by
   up to ~4x, which slows a fixed LQR enough to pass the 10 % threshold; regulation DPC on two-stage never failed).
   LQR rescheduled, LN MSE: 26-35 % for |r| <= 0.25, 9-12 % at +-0.5 / 1, 2 % at +-2; others <= 12 %.
2. The 1/distance law (fig P4): |B_hat/B - 1| at the setpoint vs d = setpoint distance from the kink centre in kink
   half-widths, over all LN surrogates, seeds and setpoints: log-log slope -0.98 beyond d = 1 (d x error ~ 0.3-0.7, roughly
   constant) = the petal far field G'(theta) cos(theta) / rho measured through a controller. Wrong-sign setpoint gains occur
   out to d ~ 4 (the petal arms); median error 51 % at d = 1-2, 18 % at 2-4, 5 % at 4-8, 3 % at 8-32.
3. Path vs setpoint (fig P6, LN surrogates, starts >= 0.5 from r): restricting to runs whose setpoint gain is within 20 % of
   B, the FIXED LQR fails on 1.0 / 1.0 / 0 / 0 % of runs as the true path's share of wrong-sign steps grows (0, 0-2, 2-10,
   > 10 %; n = 2014, 204, 46, 24); the RESCHEDULED LQR fails on 1.1 / 9.0 / 27 / 40 % (n = 2083, 111, 44, 50).
   -> The colours along the path matter exactly when the controller uses the slopes along the path. A linear controller
   on a linear true plant does not care what the model says off the setpoint; a gain-scheduled one (and DPC, which learns
   from path gradients) does.
4. Red vs blue: an overestimated gain (red) lowers K (K ~ K_true / g); harmless until K < ~3 (true pole 1.03 - 0.01 K >= 1),
   i.e. g > ~30 (seed 1 r = 0.1: g = 29, K = 3.0, gap 560 %). A wrong-sign setpoint gain (blue) gives K < 0: all 72 such
   LN runs diverged.
5. Rescheduled LQR chatters: when u_{k-1} lands in a sign-flipped band, K flips sign and u jumps across (bang-bang limit
   cycles near the setpoint; figs P1/P2 seed 9 r = -1 and 0.25, seed 6 r = 0.25).
6. Steady offsets are mostly NOT the kink: u_hat(r) is set by the surrogate's VALUE, amplified by 1/B = 100 (output error
   0.005 -> u_hat error 0.5 -> offset ~ 0.07 with K ~ 10). Every model has offsets 0.005-0.03; LN MSE larger near the kink.
   Integral action would remove them; not a kink effect.

Regulation DPC phase portraits (fig P5, saved controllers, repeat 0)
- LN MSE seed 9: all starts driven to u = -10 and x -> -10 (diverged), on the true plant AND through its own surrogate
  (dashed curves follow the solid ones): the controller is bad on its own model = the trap, seen in the phase plane.
- LN MSE seed 2: bang-bang limit cycles between u = +10 and -10 (true plant and own surrogate alike).
- Slope from step 0, seed 2: parks at x ~ 0.7-1 on the true equilibrium line (offset 0.55): a false resting point.
- Two-stage and static: smooth paths to the setpoint; two-stage seed 2 crosses a wrong-sign patch late and still converges.

Next (offered 2026-10-22)
- User runs `dpc_phase_tracking.ipynb` on Colab A100 in MyDrive/dpc_out (LQR part reproduces the numbers above; tracking DPC
  ~30-45 min per repeat; quick mode tested locally end to end, 0 errors, 8 min CPU). Then: does tracking DPC follow the
  fixed or the rescheduled LQR in fig P6b (path colours matter?), and does its failure rate peak at r ~ 0 in fig P3b?
- Candidate fix for offsets (not kink): integral action / offset-free formulation; keep out of the kink story.

## RESULTS - tracking DPC (Colab A100 float64, repeat 0; `dpc_out_phase_backup.zip`, 2026-10-22)
Checks: the Colab LQR study reproduces the local one (median relative cost difference 4e-16; failure flags agree on 99.9 %,
differences only in runs near divergence). Oracle tracking DPC = analytic LQR of the true plant: cost ratio median 1.0002,
90th percentile 1.004, max 1.029; oracle steady offset ~ 0.001. Training is not the bottleneck.

Two kinds of failure, separated (gap vs oracle tracking DPC; runs = setpoint x start pairs with |x0 - r| >= 0.5):
- CATASTROPHIC (gap > 100 % or diverged), % of runs - DPC / LQR fixed / LQR rescheduled:
  LN MSE 26.6 / 10.8 / 3.6; slope from step 0 29.3 / 1.4 / 0.0; MSE 12000 0.2 / 1.1 / 0.2; two-stage 0.0 / 0.9 / 0.1;
  no norm 0 / 0 / 3.6; static 0 / 0 / 0.
- MILD (10-100 %, no divergence) = mostly steady offsets (median offset 0.15 DPC, 0.06 LQR): value error of the surrogate
  x 1/B, present in every model (LN MSE 14-17 %, others 0.1-5 %). Not the kink.

1. Tracking-DPC catastrophic failures follow the SEED, not the setpoint (fig P8 = seeds x setpoints grid): LN MSE seeds 1, 2,
   8, 9 fail on 52-77 % of all runs at every setpoint from -2 to 2 (the SAME four narrow-kink seeds as in regulation);
   slope from step 0 seeds 2, 3, 9 fail on 97-100 % (seed 3 = its narrowest kink, r*_x = 0.037). Every other seed and
   model: <= 2 % (MSE 12000 seed 3: 2 %, LN MSE seed 6: 1 %). Contrast: LQR failures form a vertical band at |r| <= 0.25 on most seeds.
2. Trap test (track_diag.csv; reference = oracle tracking controller run through each surrogate): on all 7 broken
   (model, seed) pairs the surrogate rates the reference controller within 3 % of its true-plant cost (0.99-1.03) but rates
   DPC's own controller 3.5-6.5x worse (LN MSE) and 31x, 66x, 7200x worse (slope from step 0, seeds 2, 3, 9). Healthy
   pairs: trap ~ 1.00. Same optimisation trap as in regulation, now for a reference-conditioned policy.
3. Phase portraits (fig P7): broken tracking controllers park at a false resting point or limit-cycle away from the
   setpoint, on the true plant and through their own surrogate alike (dashed = solid), and their paths AVOID the kink
   (wrong-sign share on broken seeds 0.7 % vs 3.0 % on healthy ones; kink share 0).
4. Path colours (fig P6c, catastrophic only, broken controllers left out, setpoint gain within 20 % of B): fixed LQR 0 / 0 /
   0 / 0 %, tracking DPC 0 / 0 / 0 / 0 % (n = 1759, 86, 69, 22), rescheduled LQR 0 / 0.9 / 2.3 / 12.2 % (n = 2083, 112, 44,
   49) as the path's wrong-sign share goes 0, 0-2, 2-10, > 10 %. A trained DPC policy is a fixed map at run time: like the
   fixed LQR it never consults the model online, so the colours its path crosses do not matter at test time.

Synthesis (paper-worthy framing): WHEN a controller consults the surrogate's slopes decides WHERE the kink hurts.
- Once, at the setpoint (LQR designed at r): damage depends on the setpoint's distance from the kink, error ~ 0.45 / d.
- Online, along the path (rescheduled LQR, and by extension gradient-based MPC): damage grows with the path's time in
  wrong-sign regions (chattering, bang-bang).
- During training, across the training distribution (DPC): damage is global and seed-level, setpoint-independent: on the
  narrowest kinks the gradients trap the policy; the controller is broken everywhere, while the surrogate's predictions
  would have certified a good one. Two-stage removes all of it (0 catastrophic failures in every controller).
Caveats: one tracking repeat; one plant (A = 1.03, linear); offsets need integral action and are kept out of the kink story.

## Trap mechanism: exploding gradients, not a barrier (computed locally 2026-10-23 from the saved repeat-0 controllers)
Code: `phase_core.py` section 7 (`reference_path_colours`, `cost_interpolation`, `closed_loop_stability`); Fig. P9
(figP9_trap_landscape). Surrogates and controllers downloaded from MyDrive/dpc_out; trap ratios reproduced to every digit first
(seed 1: 9.61 / 2.549 = 3.77x; seed 9: 1861 / 2.482 = 750x). All on the 1000 TRAINING starts of repeat 0, horizon 200.
Broken pairs (regulation, repeat 0): LN MSE seeds 1, 2, 8, 9; slope-from-0 seeds 2, 9.

1. No barrier. Along the straight line in weight space from DPC's own controller (alpha = 0) to the reference controller
   (alpha = 1), the SURROGATE's loss is monotone decreasing on all 6 broken pairs (max at alpha = 0: 9.8, 8.6, 19.5, 1782, 4.46,
   7.98 -> 2.6, 2.65, 2.67, 2.62, 2.61, 2.52), and so is the true-plant loss (dashed = solid). The trapped controller is not in a
   loss basin of its own, and it is NOT a stationary point: ||grad L_sur|| at DPC's controller = 48, 29, 69, 31932, 7.5, 37, versus
   1e-4 to 1 on every healthy pair (healthy pairs show barriers of 0-0.2 along the line, the same size on static norm: ordinary
   curvature between two different minima).
2. The gradient explodes on the approach. Along the same line ||grad L_sur|| spikes to 5.6e16 (s2, alpha 0.7), 7.7e53 (s8, 0.65),
   4.5e36 (s9, 0.9), 1.6e45 (slope s2, 0.95), 5.3e26 (slope s9, 0.8), and 1.3e3 at the end point for s1. Kink-free surrogates
   stay below 16 everywhere; healthy LN MSE seeds 0, 3, 4, 5, 7 below 10. Slope-from-0 seed 3 (healthy in regulation, 97 %
   catastrophic in TRACKING) also spikes (5e15 at 0.05, 7e33 at 0.9); LN MSE seed 6 mildly (120 at 0.05).
3. Why: at the reference controller the surrogate's own loss gradient is 1.3e3 (s1), 1.1e35 (s2), 3.1e13 (s8), 1.2e14 (s9),
   2.7e24 (slope s2) vs 0.03-3.5 on all other 44 pairs, while the surrogate's VALUE of that controller's cost is right to 6 %
   (2.44-2.66 vs 2.50). The largest closed-loop Lyapunov exponent of (reference controller + surrogate), (1/N) log|dx_N/dx_0| over
   64 starts, is POSITIVE on exactly those pairs: 0.012 (s1, 1.6 % of starts), 0.418 (s2, 52 %), 0.157 (s8), 0.136 (s9), 0.211
   (slope s2); negative on every other pair (true plant: log 0.927 = -0.076; two-stage -0.02 to -0.07; MSE 12000 seed 2, 35 x B:
   -0.039). exp(200 lyap) reproduces the gradient norms (s2: e^83.6 = 2e36; s8: e^31 = 4e13). Mechanism: at the setpoint the
   surrogate's J_u_hat along the landing path is wrong-sign or tiny (min gain over the last 20 steps: s2 -0.95, s8 -1.68, slope s2
   -6.96 x B; s9 crosses a flipped band with J_x_hat 0.92), so the surrogate believes the GOOD controller is locally unstable;
   the rollout stays bounded (x jitters within 0.02-0.07 of 0, cost unchanged) but backprop through 200 steps multiplies by
   exp(N lyap). The surrogate's cost of the good controller is right; its gradient at the good controller is 1e13-1e35 too
   large and chaotic in direction. Gradient-based DPC cannot enter that basin; AdamW settles where gradients are tame.
4. Where it settles: the trapped controllers rest OUTSIDE the kink (kink share 0) at points where the surrogate's gain is
   CORRECT (median J_u_hat/B over the last 20 steps: 1.02, 1.05, 0.97, 1.00, 0.91, 1.06 for the six broken pairs), i.e. just
   outside the petal arms: x = 0.59, 1.10, 0.75 (s1, s2, s8), 0.55, 1.06 (slope s2, s9), diverging (s9). Along DPC's own path the
   adjoint per-step gradients agree in sign with the true plant's on 99-100 % of steps: the policy found the honest part of the
   surrogate and stopped there.
5. What separates the tall-narrow SURVIVORS (answers the open item): MSE 12000 seed 2 (35 x B) and LN MSE seed 7 (23 x B) have
   positive gain where the reference controller lands (4.9 and 6.2 x B, overestimated but right-signed), Lyapunov -0.039 /
   -0.041, gradient norm 0.41 / 0.74: the wrong-sign arms exist but do not cut the landing line. Peak height is necessary; the
   orientation of the arms relative to the landing path (u ~ -K x near 0) is what decides. Slope-from-0 seed 9 is the one broken
   pair whose end point is tame (0.55, lyap -0.025): its spike (5e26) sits at alpha = 0.8, on the way in.
6. Simple path-colour statistics along the reference path do NOT separate broken from healthy pairs: wrong-sign share 0-8 % on
   all, |J_u/B - 1| > 1 on 70-80 % of steps for healthy wide-kink seeds that rest inside the kink, open-loop adjoint sign flips
   0.04-0.44 overlapping. Only the closed-loop quantity (Lyapunov exponent / gradient norm) separates. The open-loop adjoint
   through J_x_hat alone also "explodes" (ratio 1e5-1e8) on healthy seeds resting inside a wide kink, harmlessly: magnitude error
   with the right direction, which Adam normalises away.
7. Paper statement: "MSE-invisible slope errors break gradient-based control" now has its mechanism: the kink makes the
   surrogate's closed loop under the good policy locally unstable, so the training gradient through a long horizon diverges
   (exp(N lyap)) while the predicted cost stays accurate. Two-stage: Lyapunov negative on 10/10, gradient norms 0.05-0.9.
   Also explains the horizon dependence (traps appeared at horizons 100-200: exp(N lyap) grows with N) and the tracking
   failure of slope-from-0 seed 3 (spiky landscape, survived regulation by luck of the path).
Figure P9 (3 panels): loss along the line (solid surrogate, dashed true), gradient norm along the line, gradient norm at the
reference controller vs its Lyapunov exponent through the surrogate (circled = broke DPC). Data: path_colours.csv, interp.csv,
gradnorm.csv, gradnorm_line.csv, stability.csv (local; regenerate with phase_core section 7 from MyDrive/dpc_out).
Not done: animations of test-time phase portraits (judged not informative: P5/P7 already show the symptom; the cause is the
gradient field, which P9 shows); training-time checkpoints (would show the policy being deflected; optional, ~10 min A100).

Verification of the mechanism (2026-10-23, same saved controllers; `scratchpad/verify.py`, no training)
(i) Horizon scaling at the reference controller, N = 25 / 50 / 100 / 200: s2 4e3 / 6e8 / 5e17 / 1e35 (= e^{0.4 N}, lyap 0.42);
    s8 4 / 39 / 2e6 / 3e13; s9 31 / 2e4 / 2e7 / 1e14; slope s2 0.7 / 8e5 / 5e15 / 3e24; s1 1 / 18 / 2.7e3 / 1.3e3 (marginal);
    survivors (LN MSE s3, s7, MSE 12000 s2, static s2, slope s3, s9) flat at 0.2-1.3 at every N. Explains horizon dependence.
(ii) Per-start share of ||grad||^2 at the reference controller (N = 200): broken pairs ONE start carries 61-100 % (s1 0.94, s2
    0.83, s8 0.999, s9 1.000, slope s2 0.61; top-10 starts 100 %); healthy pairs top start 0.2-1.7 %, top-100 15-44 %. The
    batch gradient over 1000 starts is numerically a single trajectory (the one lingering in the flipped band); the honest
    signal of the other 999 is 1e13-1e35 times smaller. Effective batch size 1.
(iii) Direction, measured: cosine between the surrogate gradient at the reference controller and at a copy with relative
    weight perturbation 1e-6 / 1e-4 / 1e-2. Healthy: 1.000 / 1.000 / 1.000 (smooth field). Broken: s1 -0.73 / -0.69 / +0.37;
    s2 +0.94 / -0.15 / +0.42; s8 +0.95 / -0.34 / -0.33; s9 +0.97 / -0.07 / +0.91; slope s2 -0.98 / -0.94 / +0.98. The direction
    decorrelates at 1e-4 relative changes, below one Adam step (lr 1e-3): each step sees a fresh random direction.
(iv) Repeats 1, 2 (their own reference controllers and starts): LN MSE lyap_max > 0 at seeds 1, 2, 8, 9 in all three repeats;
    s2 and s9 extreme every time (gradnorm 1e35 / 6.6e27 / 2.2e41 and 1.2e14 / 6.6e27 / 5.7e9). CAVEATS: seed 0 has a small
    lyap_max > 0 in reps 1-2 (0.029, 0.017; gradnorm 110, 3.8) and never failed -> the magnitude, not the sign, is the
    criterion (failures from ~1e3 up); rep-2 seed 8 has a tame endpoint (3.7, lyap 0.034) yet failed, like slope-from-0
    seed 9 (0.55): for those the spike is on the line (seed 9: 5e26 at alpha 0.8), not at the endpoint.
    -> Correct statement: the surrogate's gradient field between the init and the good policy contains regions where the
    closed loop is unstable THROUGH THE SURROGATE, with gradient norms >= 1e13; every broken pair shows one on the straight
    line; healthy surrogates never exceed 16. The endpoint Lyapunov exponent is a sufficient diagnostic (5/6), not necessary.
Still open (need training): warm-start DPC at the reference controller on the trapped surrogates (prediction: leaves the
basin within tens of epochs; stays on two-stage/static); robust per-start gradient clipping / median aggregation as the
generic optimiser-side fix (if it rescues seeds 2, 8, 9 the cause is proven and the reviewers' baseline exists). Cheap, no
training: the line-spike statistic for repeats 1-2.

Intervention tests (2026-10-23, local CPU; `scratchpad/run_1.py`, `run_3.py`, `run_2a.py`; data warmstart*.csv,
gradnorm_line_reps12.csv)
T3. Line-spike statistic for repeats 1-2 (their own reference and DPC controllers). Max ||grad L_sur|| along the line,
    LN MSE, seeds 0-9: rep 1: 1e2, 1e59, 3e7, 0.8, 6, 5, 20, 1, 1e64, 1e32; rep 2: 4, 1e51, 2e41, 6, 8, 7, 5, 3, 1e56, 2e42.
    Over all 30 LN MSE runs: every failed run >= 1e3 (11/12 >= 1e7), every healthy run <= 1e2. Complete separation.
    Slope-from-0: seed 2 spikes in every repeat (2e45, 2e23, 1e30); seed 9 only in rep 0 (5e26; reps 1-2: 20, 1e2 despite
    failing); seed 3 spikes in rep 0 (7e33) without failing in regulation. 15 of 18 failed runs caught by the straight line.
T1. WARM START: AdamW started AT the reference controller, 300 epochs at horizon 200, on each surrogate's own loss.
    Broken pairs: the optimiser walks UPHILL on its own objective and plateaus: surrogate loss 2.65 -> 3.70 (LN MSE s2),
    2.67 -> 2.93 (s8), 2.62 -> 2.97 (s9), 2.61 -> 6.55 (slope s2), 2.52 -> 5.77 (slope s9, slower: 100 epochs); true cost
    2.64 -> 3.80, 3.11, 3.22, 6.50, 6.04. Seed 1 is kicked to 3.96 at epoch 10 and recovers to 2.64 (mildest trap).
    Healthy pairs (LN MSE s3, s7; two-stage s2, s9; MSE 12000 s2): within +-0.03 throughout.
    -> A descent method that raises its own loss: the gradient direction is chaotic, not only large. Slope-from-0 seed 9,
    whose endpoint is tame (norm 0.55, cosine 0.999), is still expelled: the spiky region lies within one Adam step
    (1e-3 per weight) of the good controller. Resolves T3's gap: the endpoint can be quiet, the neighbourhood never is.
    From-scratch traps (9-1800) are worse than these plateaus (3-6.5) because from scratch the policy never reaches the basin.
T2a. CLIPPED WARM START (per-start gradients clipped to norm <= 3 = just above the healthy maximum, then averaged; 100 epochs,
    horizon 200, same pairs). Every broken pair STAYS at the good controller: true cost 2.638 -> 2.643 (LN MSE s2; plain
    3.80), 2.61 -> 2.64 (s8; plain 3.11), 2.64 -> 2.65 (s9; plain 3.22), 2.64 -> 2.64 (slope s2; plain 6.50), 2.64 -> 2.64
    (slope s9; plain 6.00); healthy pairs unchanged in both arms. Surrogate loss within 0.05 of the start on all nine pairs.
    -> Bounding each trajectory's vote removes the expulsion: the failure IS the few exploding per-trajectory gradients
    dominating the batch, not the loss landscape. Fig. P10 (figP10_warmstart: plain vs clipped, true cost vs epoch).
    Still open: whether clipping also lets DPC FIND the good controller from scratch (dpc_core.ROBUST_MODELS, 'clip' = 3 and
    'unit' vote; notebook block "Optimiser-side control", A100 run pending). Caveat: clipping is a diagnosis-dependent fix
    (needs to know the per-trajectory gradients can explode and a threshold tuned to the healthy scale); two-stage training
    removes the cause at the model and needs neither.
Status of the mechanism claim after T1-T3 and (i)-(iv): confirmed by intervention for all 6 repeat-0 broken pairs; the
straight-line diagnostic separates 30/30 LN MSE runs and 15/18 slope-from-0 runs; the remaining 3 (slope s9 reps 1-2, s4
once) fail without a spike on the straight line - unexplained, likely spikes along the actual training path.

## Which slope error hurts control: J_x vs J_u (2026-10-24, local; `claude/dpc/story/attribution.py`, no training)
Method: replace ONE of the surrogate's Jacobians by the true value and re-measure. LQR fixed: K = DARE(A_hat, B_hat) vs
DARE(A, B_hat) vs DARE(A_hat, B) vs DARE(A, B), always with the surrogate's own u_hat(r); true plant; same grid as the setpoint
study (11 setpoints x 9 starts x 10 seeds, |x0 - r| >= 0.5). DPC: the GRU reference controller of repeat 0, its surrogate rollout
kept (values unchanged), but inside backprop d/dx (or d/du) of the surrogate replaced by A (or B) via stop-gradient.
LQR fixed, % runs failed / catastrophic:
  LN MSE 3000: both from surrogate 26.5 / 10.8; true A + surrogate B 29.8 / 12.2; surrogate A + true B 13.5 / 4.1; both true 2.3 / 0.
  slope from 0: 5.8 / 1.4; 7.2 / 1.5; 1.8 / 0; 0.3 / 0.   MSE 12000: 6.5 / 1.1; 7.4 / 1.1; 3.1 / 0.2; 1.8 / 0.2.
  two-stage: 5.7 / 0.9; 6.0 / 0.9; 2.3 / 0; 2.3 / 0.   static: 0.6 / 0; 0.6 / 0; 0.5 / 0; 0.5 / 0.
  Setpoint slope errors, LN MSE: |A_hat - A| median 0.007, max 0.50; |B_hat/B - 1| median 0.34, max 87.
  -> J_u error alone reproduces ALL of the LQR damage (fixing A changes nothing); J_x error alone causes about half of it
  (13.5 % failed, 4 % catastrophic) despite being tiny in relative terms: what matters for LQR is A - 1 = 0.03 (the
  unstable part), and A_hat errors of 0.03-0.5 near the kink are 1-17x that. Not additive (26.5 < 29.8 + 13.5): the two
  errors partly cancel in the DARE gain. Residual 2.3 % with both true = u_hat value error (offsets), not slopes.
DPC, ||grad L_sur|| at the reference controller (lyap_max) - surrogate J_x and J_u / true J_x / true J_u:
  LN MSE s1 1.3e3 (0.012) / 1.2e3 (0.010) / 4.1 (-0.017); s2 1.2e35 (0.41) / 4.6e34 (0.43) / 0.35 (-0.04);
  s8 8e15 (0.16) / 1.2e17 (0.17) / 0.81 (-0.03); s9 9e11 (0.11) / 1.4e12 (0.13) / 2.5 (0.003); slope s2 3e31 (0.25) / 6e32 (0.26) / 1.0.
  Healthy LN MSE s0, s3, s7, MSE 12000 s2, two-stage s2: 0.05-0.74 with the surrogate's slopes; making J_u true gives 8.2, 650
  (s3, lyap +0.035), 1.8, 0.27, 0.11.
  -> The DPC gradient explosion is entirely the J_u kink: with the true J_u it disappears on every broken pair; with the true
  J_x it is unchanged. Caveat: a hybrid (true J_u, surrogate J_x) is not a consistent model; on LN MSE s3 it creates a mild
  instability that the surrogate's own J_u had cancelled, so the J_x error is not harmless in isolation, it is just not what
  breaks DPC on these surrogates. Slope s9 (tame endpoint, spike on the way in) is not attributable at the endpoint.
Summary for the paper: both Jacobians carry a kink, and both matter for LQR (J_u dominant, J_x about half as damaging through
the instability margin A - 1); the DPC trap is caused by the J_u kink (low control authority B = 0.01 amplifies it: H3).
