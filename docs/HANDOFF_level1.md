# Handoff: Level 1 (Jacobian loss) -> controls -> Level 2  (written 2026-10-21, last updated 2026-10-23 night)

Read this first in a new chat. Then, as needed: `claude/HANDOFF_smc_paper.md` (paper ladder, section 3),
`claude/NOTES_level1_results.md` (all Level 1 numbers, findings 1-17), and the doc
"Level 1 explained: method, parameters, metrics and results" (https://claude.ai/artifact/M8C52hXNTUbPLAZZs9CJpe;
Claude Docs container {"kind":"project","id":"a2f7cd28-1d4a-4f53-9d3f-3e62455af2a7"}, prose node "299ccfed-a56a",
rev 50 at handoff; read with sinceRev 50 to see later edits). Companion doc (2026-10-21, user's request): "The Student and
the Kink: Level 1 as a fable" (https://claude.ai/artifact/KbzpkWkrG8n7aA8Ug4asj1; container id
96a76b3e-7a04-44cf-a26c-7f0244349184, prose node 9f2b2db2-279b): all Level 1 findings and Figures 1-11 + Anatomy 1, 2, 5
retold in the FIRST PERSON (the user as the PhD student, "my advisor"), one moral per chapter. Keep it in sync when results
change (e.g. 10-seed controls -> Chapter 9; DPC -> a new chapter).

## How the user works (keep doing this)
- Final-year PhD (control + NNs). Goes level by level; wants justifiable/generic parameters, simple Kaggle notebooks
  with figures for the advisor, and to fully understand every method, parameter, metric and figure. Asks many
  "what exactly does X mean" questions: answer concretely, with the math and the actual numbers from the runs,
  and check the code before answering.
- Keeps one reference doc ("Level 1 explained") updated with every explanation and result. Ask/offer to add new
  explanations there; it uses Claude Docs (figures uploaded via Artifact asset upload -> blob create -> media block).
- Vocabulary: "two-stage training" (NOT "retrofit"; old result files say retrofit, load() renames).
  "slope loss" = Jacobian loss. Keep Epanechnikov weighting (user decided).
- Be candid about caveats and correct earlier wrong statements explicitly.
- Kaggle: the user's runs DO restore the previous output folder (labels load from cache in 0.7 s/seed), so re-running
  only trains missing configurations.

## Setup (Level 0/1, unchanged)
Plant x+ = 1.03x + 0.01u (linear, open-loop unstable) in ALL of Level 1; 20 000 transitions, x~U[-4,4], u~U[-10,10],
sigma = 0.01; test 4000 noiseless points. Pre-norm LayerNorm residual block (H = 16, 16->32 GELU->16, skip, readout,
1169 params); static norm (fixed s0) and no norm as kink-free yardsticks. Full-batch Adam lr 3e-3, float64.
Labels: local quadratic WLS, k nearest in std-scaled (x, u), Epanechnikov weights, k by doubling-trick bias +
sandwich variance; k = 1280 on every seed = TOP of grid (40..1280), because bias = 0 on a linear plant (note this).
Loss: MSE (all points) + lambda * mean over M = 2048 uniform draws of [w_x s_x^2 (dF/dx - Jx_hat)^2 + w_u s_u^2 (dF/du - Ju_hat)^2],
s_x = 2.31, s_u = 5.77, lambda = 1. Adam moments reset at each phase switch.
Seed s fixes init weights (key s), data (1000+s), labels, and slope-check draws (10000+s); every configuration
trains its own model per seed -> paired comparisons (Wilcoxon signed-rank).
Metrics: test rel-MSE; sup and RMS |J_u err|/B on 201x201 grid + kink-line points; sup J_x; interior sign-flip
area; LQR-unstable setpoints (801 on u = -3x, Q = 5, R = 0.1, DARE); kink tests (crest factor, centre-energy share,
steepness, worst-point location); training points inside kink = 125*pi*r*_x*r*_u.

## Status
- Level 1 Kaggle run (14 configs x 10 seeds) done and interpreted (NOTES findings 1-16; doc complete).
- 2nd Kaggle run (2026-10-21, `level1_out_backup (4).zip`): longer MSE training on 10 seeds DONE (finding 14 rewritten,
  doc Fig. 10 section + "short version" item 1 corrected). Key result: 4x longer MSE shrinks the LN peak ~7x
  (18.7 -> 2.3) but it stays ~40x above static norm at every length, kink-shaped, and narrow-kink seeds barely move
  (Spearman -0.99); two-stage (4000 steps) beats 12 000 MSE steps 10/10. The 3-seed preview ("median does not fall")
  was wrong and is retracted in both places.
- That run used an OLD notebook copy: the 5 controls were NOT trained (no control rows, no Fig. 11).
  Controls are still the 3-SEED PREVIEW (finding 17, doc section "Two controls" + Fig 11):
  * weight control: two-stage -> both slopes beat each lambda = 2 variant 3/3 (clean); from scratch J_u-only
    lambda = 2 ~ both (median peak 9.6 vs 10.9) -> claim "both needed" for two-stage only, pending 10 seeds.
  * reset control: Adam restart alone ~halves RMS gain error in 250 steps and drops rel-MSE ~3x; peak lower 2/3 seeds.
    Two-stage vs MSE 3000+1000 shares the restart, so its 10/10 advantage (ratio 0.111) is the slope loss.
- Notebook `claude/level1/level1_jacobian_loss.ipynb` (19 configs, Figs 1-11, kink anatomy section, optional
  RUN_WAVES) passed an end-to-end QUICK test. Code: `claude/level1/level1_core.py`, `level1_report.py`,
  `build_level1_notebook.py` (rebuild the notebook from these after any code change).

## Open items / next steps
1. User uploads the CURRENT project notebook to Kaggle (replacing their old copy) and re-runs in the same output folder:
   only the 5 controls train (waves already done; RUN_WAVES can stay True, it skips). Then update finding 17, doc
   "Two controls" section, Fig 11, Q2 caveat and caveat 3 with 10-seed numbers.
2. Last answered questions (not yet in the doc; offer to add): TRUE labels = both J_x and J_u true (oracle, paired vs
   J_x + J_u 3000, ratio 0.998); Fig 4 colour = model/true gain, blue = below true, wrong sign only inside the black
   0-contour; Fig 5 panels 1-2 x-axis = per-point label error (J_u in units of B; J_x raw), tails = box edges (RMS
   0.093 B within 2 % of edge vs 0.026 B interior; 90 % of |err| > 0.1 B within 5 % of edge); Fig 5 panel 3: k at grid
   cap, variance ~1/k^2, bias 0 on linear plant, U-shape expected on nonlinear plants, kernel/sigma change level
   not trend; seeds -> one model per config per seed; same plant throughout; Fig 6 reading + caution that the
   narrow<->tall correlation is partly built into the theory (MSE hollow points show it too); the right-panel shift
   (scratch slope loss -> ~10x narrower kink from the same init) is the stronger evidence; Level 2 is the causal test.
3. DPC study - FULL RUN DONE 2026-10-22 (A100, float64); results + mechanism in `claude/NOTES_dpc_level1.md` (RESULTS
   section). Headline: LN MSE DPC fails on exactly the 4 narrowest-kink seeds (1, 2, 8, 9: +240 % to divergence), because the
   controller cannot control its own model (wrong resting point / divergence); two-stage ties static norm (P2 holds);
   MSE 12000 fine on all seeds (P4 wrong). Diagnostic (repeat 0): optimisation trap - the kinked surrogate rates the
   reference controller like the plant (~2.5) but DPC's own controller 3-750x worse. EXTENSION built and tested
   (REPS = [0, 1, 2] reusing repeat 0 from MyDrive/dpc_out; diagnostic; Figs D7, D8): user to run on Colab A100 (~1-1.5 h),
   plus the Level 1 notebook for the 10-seed controls. Then: harder condition (nonlinear plant or A = 1.07), Level 2.
   Paper-readiness assessment given 2026-10-22: core solid; gaps = DPC repeats, mechanism (now diagnostic), Level 1 controls,
   one plant/task only, novelty needs Level 2 (Kissel & Diepold 2019 already did estimated-derivative Sobolev). Offered:
   doc section + fable chapter. Original setup notes: notebook `claude/dpc/dpc_through_surrogates.ipynb` ready and tested end to end in
   QUICK mode (5 min CPU, no errors, all 6 figures checked). Design, predictions and engineering notes in
   `claude/NOTES_dpc_level1.md`. User runs it on a Kaggle P100 (float64; DTYPE = "float32" on a T4), uploads
   dpc_out_backup.zip -> interpret against P1/P2/P4, write NOTES findings, add a doc section (offer). Follows the user's
   DPC example (GRU, LQR-type loss, curriculum 5..200, 9000 epochs, AdamW 1e-3) with 4 agreed changes (full-surrogate
   rollout, u = 10 tanh, plain JAX vmapped over seeds, paired init/x0). Regulation to 0, A = 1.03, 7 models.
3b. DPC REPEATS DONE + PHASE STUDY (2026-10-22, see NOTES_dpc_level1.md, last two sections). Repeats: LN MSE fails 13/30
   runs (seeds 1, 2, 8, 9 every repeat), two-stage 0/30; 20 of 21 failures are the optimisation trap. Phase study (user's
   idea): time-ordered (x_k, u_k) over the surrogate's J_u / J_x maps, setpoints r in [-2, 2]. LQR results computed locally:
   setpoint gain error ~ 0.45/d (slope -0.98, d = distance from kink centre in half-widths); with an accurate setpoint gain
   the fixed LQR ignores path colours (fail 1 % at any wrong-sign share) while the rescheduled LQR fails 1 -> 9 -> 27 -> 40 %
   as the path's wrong-sign share grows. Notebook `claude/dpc/dpc_phase_tracking.ipynb` (adds tracking DPC: GRU input
   (x - r, r)) run on Colab A100 2026-10-22 and interpreted (NOTES, last section): tracking-DPC catastrophic failures follow
   the SEED (LN MSE seeds 1, 2, 8, 9; slope-from-0 seeds 2, 3, 9), not the setpoint, and are the optimisation trap again
   (3.5-7200x); two-stage 0 failures. Synthesis: when a controller consults the slopes (setpoint / online / training)
   decides where the kink hurts. New figs P8 (seed x setpoint grid), P6c (catastrophic only), track_diagnostic().
   Doc "Level 1 explained" (rev 69, 2026-10-22): six new sections before "Caveats" - DPC regulation (D7, D8), how to read a
   phase portrait (P1), setpoint study (P3, P4 + tables), path colour (P6), tracking DPC (P8, trap table, P6c, P7, P5),
   synthesis table. Read with sinceRev 69 for later edits.
4. Level 2 (not started): walk through parameters with the user BEFORE coding, then a Kaggle notebook. Plan:
   slope checks concentrated at the closed-form kink ("lens"), lens-share sweep {0, 0.25, 0.5, 0.75, 1.0},
   refresh/warm-up/min-width/M choices, error-guided (RAD-style, Wu et al. 2023) baseline, wrong-centre control,
   3 draw streams per config, linear AND nonlinear plants (eps in {0, 0.5, 1}), keep Adam restart + continuous MSE
   control, add a 12 000-step MSE baseline ("just train longer"); consider extending the k grid on nonlinear plants.
   Literature framing: adaptive/importance sampling (PINN RAD/RAR-D, Nabian et al. 2021, Katharopoulos & Fleuret 2018);
   our novelty = theory-guided placement.

## 2026-10-22 evening: paper-readiness verdict and next steps (latest state - read this before the open items above)
Verdict given to the user: solid workshop / L-CSS-letter material now; NOT yet strong for TMLR / L4DC / main track.
Strengths: closed-form theory confirmed end to end on the 1-state system; paired design + repeats; trap diagnostic;
two-stage fix (0 catastrophic failures for every controller); framing "WHEN a controller uses the slopes decides WHERE
the kink hurts" (setpoint / online / training).
What the 1-state results confirm of the LN math (answer given to the user, with numbers):
1. kink centre at (0,0) from zero-bias init = the regulation target sits in the kink;
2. far field ~1/distance measured through LQR setpoint gain: log-log slope -0.98 over 440 points (d x error 0.3-0.7);
3. petal arms: wrong-sign setpoint gains out to d ~ 4-8; rescheduled LQR fails 9-12 % at r = -1, 0.5, 1;
4. slope-not-value defect: surrogates predict the reference controller's cost to 0.93-1.06x, DPC trapped 3.5-7200x;
   offsets track value error / B in every model;
5. tall + narrow predicts DPC failure: over all 40 LN surrogates, the 7 that break DPC have median peak 28 x B vs 3.1
   (Mann-Whitney p = 1e-5), median 3.5 vs 26 training points inside the kink (p = 0.004); every surrogate below 17 x B
   works (32/32); necessary not sufficient: MSE 12000 seed 2 (35 x B, 4 points) and LN MSE seed 7 (23 x B) work.
   OPEN: what separates those 4 tall-narrow survivors (hypothesis: orientation of the wrong-sign arms vs the u = -3x line).
   [ANSWERED 2026-10-23, see the trap-mechanism section below.]
6. slope loss from step 0 narrows the kink and fails on its narrowest seeds (seed 3, r*_x = 0.037); two-stage keeps it wide.
Not predicted by the math: failure is global (whole policy, every setpoint) not local; broken controllers avoid the kink
(0 % steps inside); fixed LQR ignores path colours; red graded (unstable only above ~29x) vs blue catastrophic.
New related work to cite (found 2026-10-22): Moore, Lee & Chen 2026, "Learning Legged MPC with Smooth Neural Surrogates"
(arXiv 2601.12169: nonsmooth neural surrogates break gradient-based MPC; Lipschitz-bounded smooth surrogates fix it) -
the natural "generic smoothing" baseline; Buehlmaier 2026, "LayerNorm as Implicit Gain Control in Looped Transformers"
(arXiv 2607.10681: LN Lipschitz ~ C/||h||, same 1/sigma mechanism, transformers not dynamics surrogates/control).
Plus Kissel & Diepold 2019 (Sobolev with estimated derivatives) - the slope-loss method itself is not novel.
Reviewer attack list: toy setting (1 state, linear, 1169 params); missing baselines (larger LN eps - widens the kink by
D^2 = ||r_min||^2 + H eps; RMSNorm; generic smoothing / Lipschitz; Jacobian-norm); trap shown not explained
(analyse DPC training gradients) [DONE 2026-10-23]; 1 tracking repeat; MPC only inferred.
Next steps, cheapest first: (1) eps sweep + RMSNorm surrogate in the Level 1 notebook; (2) nonlinear plant (eps_nl, already
in the code) and A = 1.07 through Level 1 + DPC + phase notebooks; (3) tracking-DPC TRACK_REPS = [1, 2] (~1 h A100);
(4) one multi-state system (pendulum / cart-pole) with a standard pre-LN MLP or small transformer world model - decides
the venue; (5) DPC gradient analysis for the trap mechanism [DONE]; (6) Level 2 (lens) for a methods paper.
Venue map: workshop now; IEEE L-CSS (+CDC) with formal statements (kink geometry, 1/d law, MSE invisibility) + eps
baseline; TMLR after (1)-(3); L4DC / main track after (4)-(6).

## Moving to a new project / chat
Data that stays on Google Drive (do not upload; notebooks reuse it): MyDrive/level1_out (Level 1 + waves + labels),
MyDrive/dpc_out (surrogates/, controllers/, trajectories/, track_* and phase outputs). Result zips the user holds:
level1_out_backup (4).zip, dpc_out_backup (2).zip (repeats), dpc_out_phase_backup.zip (phase + tracking).
Living docs (Claude Docs, reachable by link from any chat): "Level 1 explained" https://claude.ai/artifact/M8C52hXNTUbPLAZZs9CJpe
(container a2f7cd28-1d4a-4f53-9d3f-3e62455af2a7, prose node 299ccfed-a56a, rev 72 on 2026-10-23; control sections added before
"Caveats"); fable https://claude.ai/artifact/KbzpkWkrG8n7aA8Ug4asj1.
Drive files can be pulled into a chat's workspace with the Google Drive connector (download_file_content returns base64;
decode to .npz): surrogates/ (95 KB each), controllers/ and track_controllers/ (1 MB each). Done 2026-10-23 for the trap analysis.

## 2026-10-23: topic inventory and advisor-document plan (offered, not yet built)
Inventory given to the user: 5 themes, 22 topics, ~75 sub-questions.
A theory (1 closed-form x-kink, 2 input-channel U1-U5, 3 MSE invisibility, 4 envelope/representability, 5 lens,
6 escape routes); B diagnosis (7 LN-specific & MSE-invisible, 8 anatomy/petals, 9 kink vs approximation error,
10 narrow<->tall / points inside, 11 longer training, 12 kink screen A = 1.03/1.05/1.07); C fixes (13 Direction 1,
14 estimated labels, 15 Level 1 derivative/schedule/labels, 16 controls, 17 SMC E1 baselines/lens preview/noise/lambda);
D control (18 early DPC sweep seed 0, 19 LQR screen, 20 DPC regulation 7x10x3, 21 trap diagnostic, 22 phase/setpoint/
path colour/tracking DPC); E positioning (readiness, novelty, venues).
Proposed advisor document (10-15 pages, separate from "Level 1 explained", which stays the detailed reference):
0 one-page summary; 1 problem; 2 theory + numbered hypotheses H1 kink at closed-form location/width, H2 MSE-invisible,
H3 low-authority amplification (J_u), H4 slope loss with estimated labels removes it, H5 narrow kinks escape uniform
checks (lens needed), H6 control damage depends on how the controller uses slopes; 3 one shared setup section;
4-6 results grouped by hypothesis (diagnosis H1-H3, fixes H4-H5, control H6); 7 hypothesis scorecard incl. refuted
predictions (P4 MSE 12000; "same from large starts"; 3-seed waves preview); 8 limitations/open questions; 9 plan + venues;
appendices (proofs, parameters, paired stats, figure gallery, glossary). Rules: separate exploratory (seed-0, sandbox)
from paper-grade evidence; reconcile cross-phase numbers first (E1 scratch Level-1 peak 4.4 vs Level 1 run 7.9 =
draw-stream dependence, finding 7; E1 lens rows used the older relabelling lens); number each figure once.
Build it as a Claude Doc when the user asks.

## Starting a new chat in THIS project
Everything is in project knowledge: this handoff, NOTES_level1_results.md, NOTES_dpc_level1.md, NOTES_controller_arch.md,
HANDOFF_smc_paper.md, code under claude/level1/ and claude/dpc/ (the DPC notebook embeds the merged modules; QUICK run with
ARCH = "mlp" + RUN_ROBUST verified 2026-10-23 night).
Pending user runs: Level 1 notebook with the 5 controls (10 seeds) -> finding 17 + Fig 11; optional TRACK_REPS = [1, 2];
DPC notebook ARCH = "mlp" / "linear" (E-A1, E-A2) and its RUN_ROBUST block (any ARCH).

## 2026-10-23 evening: controller-architecture study built (NOT yet run at scale)
dpc_core.ARCH in {gru, mlp, linear}; memoryless controllers get policy maps (D9), K_eff table, the linear (K, c) loss
landscape (D10) and a gradient audit of Corollary U5 (D11). Design, predictions A1-A7 and run order E-A1..E-A6 in
`claude/NOTES_controller_arch.md`. Smoke test on seeds 2, 9: MLP reproduces the trap (seed 2 inverted policy
K_eff(0) = -85, diverged; seed 9 gap ~100x), the 2-parameter linear controller does not fail, landscape minimum through
the surrogate within 1 of the true plant's, LN MSE wrong-sign dL/dK share 15-23 % vs 0-5 % kink-free. Next: user runs the
DPC notebook with ARCH = "mlp" then "linear" (REPS = [0,1,2]) in MyDrive/dpc_out; results land in dpc_out_mlp / dpc_out_linear.

## 2026-10-23 (trap chat): trap mechanism found and verified; code merged with the ARCH chat
Two chats edited claude/dpc/ on 2026-10-23. MERGED STATE (written back by the trap chat, ~21:00 local): dpc_core.py = ARCH
(gru/mlp/linear, ctrl_step, arch_dir, linear_landscape, gradient_audit) + robust_grad / aggregate option in
get_dpc_trainer / ROBUST_MODELS (per-start gradient clipping, clip = 3, and 'unit' equal vote); phase_core.py = ctrl_step
dispatch + section 7 (reference_path_colours, cost_interpolation, closed_loop_stability); build_dpc_notebook.py = ARCH
cells + D9-D11 + the "robust per-start gradients" block (RUN_ROBUST); dpc_report.py = D9-D11 appended, fig_phase takes
sur_dir (it loaded surrogates from out_dir, which breaks for ARCH != gru). If you edit these files, project_read them
first: a project_write replaces the whole file.
Results (all in `claude/NOTES_dpc_level1.md`, sections "Trap mechanism", "Verification", "Intervention tests"; Figs P9
figP9_trap_landscape, P10 figP10_warmstart, sent to the user):
- The DPC trap is NOT a loss barrier: along the straight line from the trapped GRU to the reference controller the
  surrogate loss falls monotonically on all 6 broken repeat-0 pairs, and the trapped controller is non-stationary
  (grad norm 29-32 000). It IS an exploding, direction-chaotic gradient: ||grad L_sur|| at the good controller 1e3-1e35
  (healthy pairs 0.03-3.5) while the surrogate's VALUE of that controller is right to 6 %; scales as exp(N lyap) with the
  closed-loop Lyapunov exponent of (reference controller + surrogate), positive only on the broken pairs (0.01-0.42; true
  plant -0.076); ONE start carries 60-100 % of ||grad||^2; the gradient's cosine under a 1e-4 relative weight perturbation
  flips sign. Cause: at the setpoint the surrogate's J_u along the landing path is wrong-sign/tiny -> the surrogate believes
  the good controller is locally unstable -> backprop through 200 steps multiplies by exp(N lyap).
- Interventions: warm-starting AdamW AT the reference controller on a trapped surrogate walks UPHILL on its own loss
  (2.6 -> 3.7 / 2.9 / 3.0 / 6.5 / 5.8) and plateaus; healthy pairs stay. Clipping per-start gradients to norm <= 3 keeps
  every pair at the good controller (Fig P10). Line-spike statistic separates 30/30 LN MSE runs across repeats (failed
  >= 1e3, healthy <= 1e2) and 15/18 slope-from-0 runs (slope seed 9 reps 1-2 and seed 4 once: no spike on the line,
  unexplained). Answers the open "survivors" item: MSE 12000 seed 2 and LN MSE seed 7 have right-signed gain where the
  controller lands (lyap < 0, grad 0.4-0.7): orientation of the arms vs the landing path decides; peak height is necessary.
- Pending A100: the notebook's RUN_ROBUST block (does clipping let DPC FIND the good controller from scratch?); runs for
  whichever ARCH is set. Suggested E-A1b: after E-A1, run phase_core.closed_loop_stability / cost_interpolation on the MLP
  reference controllers (dpc_out_mlp; no training) to test whether the memoryless trap has the same exp(N lyap) signature.
  The linear controller's survival (A2) is consistent with the mechanism: with 2 parameters there is no flexible way to
  park outside the arms, and the (K, c) landscape minimum is right.
- Not done, by judgement: animations of test-time phase portraits (symptom, not cause); doc section for P9/P10 in "Level 1
  explained" and correction of the "local minimum" wording there (offered, not yet written); the paper statement should
  say "flexible policies are trapped by an exploding, chaotic gradient", not "a local minimum".
