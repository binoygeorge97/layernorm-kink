# Plan: the five items that take the slides to a paper (written 2026-11-03, corrected the same day)

Correction log (3 Nov, later the same day):
- Item 2a: the earlier version said the safe band at B = 0.1 is "the SAME band in units of B" and K ~ 1.0. Both were wrong.
  With B = 0.1: K* = 5.2538, P = 10.41, pole 0.505; the band is 0.057 B < J_u_hat < 3.86 B (vs 0.29 B to 19.7 B at B = 0.01);
  a wrong-signed input slope J_u_hat = -B gives m = 1.555 (vs 1.133). The closed loop is MORE sensitive to a given relative
  slope error at B = 0.1. Verified with scipy.linalg.solve_discrete_are.
- Item 2b: the earlier suggested plant A = [[1, 0.1], [0, 1.02]], B = [0, 0.01]^T has u_ss = 0 and x2 = 0 at every position
  setpoint (it contains an integrator), so the setpoint study would not generalise. Replaced by a plant whose equilibrium input
  moves with the setpoint.
- Item 1: memoryless Lambda* maximum is 8.75 (LN MSE 3000, seed 2, repeat 0), quoted earlier as "<= 8.7".
The detailed, authoritative version of this plan for Claude Code is CLAUDE.md in the handoff repository
(also saved to the project as claude/HANDOFF_claude_code_experiments.md).

Order: 1 (architectures), 3 (optimiser baselines), 4 (statistics) can run in one or two Colab sessions and a day of local
post-processing; 2 (second plant) is the long item; 5 (wording fixes) is an afternoon. Pre-register the predictions of 1 and 3
before looking at results: write them into the prereg files, commit, then run.

All data lives in MyDrive/dpc_out (surrogates/*.npz, controllers/*.npz, dpc_results.csv). All notebooks read SUR_DIR from
there, so the 60 surrogates are shared by every experiment below. Compare only within one precision (float64, A100).

------------------------------------------------------------------------------------------------------------------------
## Item 1. Controller architectures: MLP and linear DPC (E-A1, E-A2)

Why: rules out "the trap is an artefact of a recurrent policy", and is the first PROSPECTIVE test of Theorem 2.

Step 1.1 - pre-register (do this first, no GPU needed except the oracle)
- Lambda* of the memoryless linear policy at the LQR gain is already in results/t2_memoryless.csv: maximum 8.75 over all 180
  runs (below the 12.1 threshold). Prediction P-A2: the LINEAR controller is not trapped on any run. Recompute it also from the
  trained linear oracle once that exists.
- For the MLP: the reference MLP controller does not exist yet. Train the MLP through the TRUE plant only (the oracle; DPC
  notebook with ARCH = "mlp", ONLY = ["true plant (oracle)"]), compute its Lambda* through all 180 surrogate/repeat sets with
  a memoryless version of t2_test.py, and record the predicted traps (Lambda* > 12.1, threshold not re-tuned) BEFORE the MLP
  DPC-through-surrogate runs.
- Also predict from NOTES_controller_arch.md: A3 (trapped MLP policies flat or inverted inside the kink band), A4 (linear
  landscape minimum within ~10 % of the true K on every seed), A5 (wrong-sign share of dL/dK larger on LN MSE).

Step 1.2 - run (Colab A100, one session, ~2.5 h)
- dpc_through_surrogates.ipynb (rebuild from build_dpc_notebook.py if the code changed). Settings cell: ARCH = "mlp",
  REPS = [0, 1, 2], DTYPE = "float64", QUICK = False, ONLY = None, RUN_ROBUST = False. Output folder dpc_out_mlp.
- Repeat with ARCH = "linear" -> dpc_out_linear. ~1 h.
- Outputs: dpc_results.csv, the trap diagnostic (own / reference cost on the surrogate), D9 policy maps (K_eff = -d pi/dx at
  0, 0.5, 2), D10 linear landscape, D11 gradient audit.

Step 1.3 - analyse (local, one evening)
- Trap table per architecture: failed (gap > 10 %) and trapped (own > 1.05 x reference on the surrogate) per family, seed, repeat.
- Confusion matrix of P-A1: predicted vs observed traps for the MLP (as for the GRU: 17/20 hits, 0/160 false alarms).
- Policy-map figure: trapped MLP policies over the kink band vs healthy ones; mark the LQR slope -10.29.
- If the MLP is NOT trapped on the GRU's trapped seeds, the claim becomes "recurrent or high-direct-gain policies are trapped;
  memoryless ones with LQR-like gain are not, as Lambda* predicts". Say it plainly.

Reporting in the paper: one table (trap rate by family x architecture), one figure (D9 policy maps), one sentence on P-A1/P-A2.

------------------------------------------------------------------------------------------------------------------------
## Item 2. A second plant (and ordinary authority)

Why: the single most common reviewer objection. Two sub-items in order of cost.

### 2a. Same scalar plant, ordinary authority: B = 0.1 (cheap, ~1 day GPU + 1 day local)
- Plant line: PLANT = L.Plant(A=1.03, B=0.1, eps_nl=0.0). BEFORE running, remove the hard-coded B = 0.01 constants
  (dpc_core.LQR_K = 10.29, phase_report slope maps and path tables, t2core A_TRUE/B_TRUE, t2_memoryless K_STAR,
  level1_core.lqr_screen 3.33, plot labels) and add a plant check to the surrogate cache (it reuses files by name).
- LQR at B = 0.1: K* = 5.2538, P = 10.41, closed-loop pole 0.505. Safe band for the LQR gain:
  (J_x_hat - 1)/K < J_u_hat < (J_x_hat + 1)/K = 0.0057 to 0.386 in absolute units = 0.057 B to 3.86 B
  (at B = 0.01 it is 0.29 B to 19.7 B). A wrong-signed input slope J_u_hat = -B gives m = 1.555 (1.133 at B = 0.01).
  So the band is NARROWER in units of B: the same relative slope error does more damage. Whether traps persist depends on
  whether the surrogate's relative slope errors shrink when the input matters 10x more in the data. Genuinely open.
- Actuator |u| <= 10 now holds |x| <= 33.3, so the data box XMAX = 4 is well inside; X0 range [-3, 3] stays; u_ss = -0.3 x.
- Pre-register after the surrogates and the oracle exist: (i) the fixed/rescheduled LQR study (deterministic, local);
  (ii) DPC traps predicted where Lambda* of the new GRU oracle > 12.1.
- Run order: surrogates + oracle (Colab) -> LQR study and Lambda* (local) -> pre-register -> DPC through surrogates
  (ARCH = "gru", REPS = [0, 1, 2], ~2 h).
- New output folder: dpc_out_B01 (labels inside it too). Do NOT mix with dpc_out.

### 2b. A two-state plant (the real second plant, ~1-2 weeks; confirm the plant with the user first)
Criteria: open-loop unstable; low authority like the scalar case; controllable; an equilibrium input that moves with the
position setpoint (so the setpoint study generalises); the recoverable set under |u| <= 10 inside the data box.
Suggested: A = [[1.02, 0.05], [0, 1.01]], B = [0, 0.01]^T. Eigenvalues 1.02 and 1.01; [B, AB] full rank; equilibria
x2 = -0.4 x1 with u_ss = 0.4 x1. (The earlier suggestion A = [[1, 0.1], [0, 1.02]] has u_ss = 0 at every setpoint.)
Check the recoverable set by simulating the saturated LQR from a grid and choose the data box ~1.2x that set.

Code changes (in level1_core.py / dpc_core.py / phase_core.py; the GRU and MLP already take d_x > 1):
1. Plant class: add Plant2 (A 2x2, B 2x1, f returns a 2-vector; Jx 2x2, Ju 2x1); keep the scalar class.
2. Data: x in a 2-D box, u ~ U[-10, 10]; the surrogate predicts the 2-vector x+ (encoder input z = (x1, x2, u), E is 16 x 3,
   readout 16 x 2). Theorem 1 covers this (n_z = 3): the kink becomes an ellipsoid; z* and D from the same formulas.
3. Slope labels: local-quadratic WLS in 3 inputs for each of the 2 outputs; labels are 2x3 Jacobians.
4. Metrics: cost Q = diag(5, 0.5) (to confirm), R = 0.1, Q_f = 10 Q; "diverged" = ||x|| > 10 or outside the recoverable set;
   gap vs the 2-state oracle.
5. LQR: scipy.linalg.solve_discrete_are on the surrogate's 2x2 / 2x1 Jacobians at the setpoint (fixed) or at (x_k, u_{k-1})
   (rescheduled). Equilibrium input from the surrogate by root finding.
6. Kink maps in 3-D: slices u = u_ss(x1) and x2 = 0 with the ellipsoid sections and the LQR/DPC paths.
7. Theorem 2: the multiplier becomes the 2x2 matrix M_k = Jx_hat - Ju_hat K_k; Lambda for a stretch = log of the operator norm
   of the product over the stretch (not additive; compute running products). Write the definition down before using it.
Run: Level 1 (6 families x 10 seeds), LQR grid, DPC (GRU and MLP), Lambda*. Budget ~6 h A100.
Report: the SAME three headline figures as the scalar case side by side. If the effect is weaker, say by how much.

Decision rule: if 2a shows the traps vanish at B = 0.1, the paper's claim is "low-authority plants", which is still a real
class (thermal, process, large inertia). Then 2b should ALSO be low authority. Do 2a first for this reason.

------------------------------------------------------------------------------------------------------------------------
## Item 3. Optimiser baselines (does a standard trick already fix it?)

All on the same LN MSE 3000 surrogates, GRU controller, 10 seeds x 3 repeats unless stated. One Colab session (~3 h).

3.1 Per-start clipping from scratch (RUN_ROBUST = True in the DPC notebook; ROBUST_MODELS "clip" = 3 and "unit"). ~1 h.
    Prediction: rescues at least 3 of the 4 trapped seeds (1, 2, 8, 9).
3.2 Global clipping (clip the AVERAGED gradient to norm <= 3, as optax.clip_by_global_norm). Add aggregate = "global" to
    robust_grad. Warm-start version first, then from scratch. Prediction: does NOT rescue (direction still set by one start).
3.3 Truncated horizon: curriculum stopping at N = 50 (and N = 100). Prediction: avoids the trap but gives a worse controller on
    the true plant; report both numbers.
3.4 Learning rate 1e-4 (and 3e-3). Prediction: 1e-4 does not rescue.
3.5 Stochastic mini-batches (100 of the 1000 starts per epoch). Outcome uncertain; measure.
3.6 Median-of-means over 10 groups of 100 starts. Prediction: rescues.

Implementation: 3.2 and 3.6 are ~10 lines in robust_grad; 3.5 samples x0 per epoch in get_dpc_trainer (new key offsets);
3.3/3.4 are model options. Unique model names per setting; defaults bit-identical.

Reporting: one table, rows = optimiser settings, columns = failed / trapped out of 30, median gap, cost of the best controller,
wall time. The two-stage surrogate row (0/30 with the plain optimiser) closes the table.

------------------------------------------------------------------------------------------------------------------------
## Item 4. Statistics

4.1 Seed-level bootstrap CIs for every quoted rate (resample the 10 seeds with replacement 10 000 times; a seed carries all its
    runs). Apply to: fixed-LQR failed/catastrophic (26.5 / 10.8 %), rescheduled, each setpoint of fig P3, the path-colour bins,
    DPC trap rates per family, the swap rates.
4.2 Paired comparisons (Wilcoxon over seeds, n = 10) with the per-seed difference and its bootstrap CI.
4.3 Failure rate vs threshold (5, 10, 20, 50, 100 %) per family; threshold-free metrics: median steady offset |x_inf - r| and
    settling time to 5 %.
4.4 Lambda* threshold robustness: leave-one-seed-out choice of the threshold; range of thresholds with 0 false alarms and the
    hit rate across it.
4.5 Holm-adjusted p-values for the ~6 paired hypotheses.
Time: one day local, no GPU.

------------------------------------------------------------------------------------------------------------------------
## Item 5. Known wording fixes

5.1 Slide 5 "2-4 half-widths": farthest reach 1.7-6.8 on 8 seeds, 9.5 (seed 1), 32.6 (seed 2); 56-99 % of the wrong-sign area
    outside the ellipse. Change to "2-7" with that caption in the slides, both companions, the briefing and the handover doc.
5.2 "K scales like 1/B_hat": false in general (B_hat = 18.3 B gives K 3.98, a factor 2.6, not 18). Replace by "an overestimated
    input slope lowers K; it becomes dangerous when K < 3 (true pole >= 1), i.e. B_hat > ~30 B".
5.3 Slide 16 figure titles: "parks at x = 0.6" -> "parks at x = -0.58 or +1.71"; "bang-bang limit cycle" -> "oscillates, then
    parks at x = 1.1; one start keeps cycling". Regenerate with dpc_story.py.
5.4 Slide 17 notes: write the multiplier as Jx_hat - Ju_hat K (u = -K x); both too-large and wrong-signed input slopes leave the
    safe band; gradients as orders of magnitude only.
5.5 Decay exponents: compute Phi(theta) from the weights and show the 1/rho and 1/rho^2 laws, or soften "-0.98" to "about -1
    along the setpoint line, steeper along other rays".

------------------------------------------------------------------------------------------------------------------------
## Suggested calendar
Week 1: Mon Task 0 (reproduction) and pre-registration code; Tue Colab: MLP/linear oracles, then pre-register, then item 1 and
        item 3; Wed-Thu local analysis, item 4, item 5; Fri: item 2a stage A (surrogates + oracle at B = 0.1).
Week 2: item 2a pre-registration and DPC; item 2b plant confirmed with the user, code and smoke test.
Week 3: item 2b full run and analysis. Then assemble the paper (controls venue: Part 1 + DPC trap + Theorem 1; ML venue: add
        Theorem 2 + architectures + 2b).
