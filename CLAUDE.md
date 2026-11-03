# LayerNorm kink: experiments for the paper — handout for Claude Code

Written 2026-11-03 at the end of a long claude.ai working session. This file is the single entry point. Read it fully
before touching code; then read `docs/NOTES_theory.md` and `docs/NOTES_dpc_level1.md` (the science), and the task
section you are working on. Everything else in `docs/` is background you open when a task points to it.

----------------------------------------------------------------------------------------------------------------------
## 0. Who you work for and how

- The user is Binoy George, a final-year PhD student in control systems and neural networks. The goal is a strong
  conference or journal paper on the pre-norm LayerNorm "kink" and how it reaches model-based controllers.
- How he likes to work (keep doing this):
  - Plain language, concrete numbers from the actual runs, every method/parameter/metric explained when asked.
  - Honest caveats. When an earlier statement turns out wrong, say so explicitly and correct it everywhere it appears.
  - Equations do NOT render in his chat client. Anything with real maths goes into a typeset PDF (LaTeX), not chat.
  - Pre-register predictions before looking at results, and report the outcome either way.
  - Simple notebooks with figures he can show his advisor.
- Division of labour:
  - You (Claude Code, local machine): code changes, CPU analyses, smoke tests (`QUICK = True`), building notebooks,
    pre-registration files, post-processing of results, statistics, figures, LaTeX.
  - The user: GPU runs on Google Colab (A100, `DTYPE = "float64"`), using notebooks you build. He uploads result zips
    into `data/`. Give him exact cell settings and the expected run time for every Colab run.
  - The slide deck, briefing and handover doc are claude.ai artifacts (links in §2.2). You cannot edit them. For slide
    changes, write the exact replacement text into `out/slide_fixes.md` for him to apply.
- Ground rules:
  1. `git init` this folder first and commit before every Colab run and every pre-registration. Pre-registrations must
     be committed (hash recorded) BEFORE the results they predict exist.
  2. Never overwrite an earlier result folder. Every experiment gets its own output folder.
  3. Compare numbers only within one precision (float64). Surrogates are always float64.
  4. Default behaviour of existing code must stay bit-identical (same keys, same outputs) unless the task says
     otherwise. Add options; do not change defaults. Prove it with a regression check.
  5. Do not tune anything after seeing a result to make a prediction come true. Report misses.
  6. Log every run in `docs/RESULTS_LOG.md` (append-only): date, task, commit, command/notebook settings, data
     version, key numbers, deviations from plan.
  7. Ask the user before: choosing the two-state plant (Task 2b), deleting or renaming anything on his Google Drive,
     changing a claim in the companions beyond the fixes listed in Task 5.

----------------------------------------------------------------------------------------------------------------------
## 1. The research in one page (what is established)

### 1.1 Setup
- Plant: x+ = A x + B u, A = 1.03, B = 0.01, |u| <= 10 (open-loop unstable, low authority; holdable |x| <= 10 B/(A-1) = 3.33).
  LQR with Q = 5, R = 0.1: K* = 10.2875, P = 110.96, closed-loop pole A - B K* = 0.927.
- Data per seed s: 20 000 transitions, x ~ U[-4, 4], u ~ U[-10, 10], y = f(x, u) + 0.01 N(0, 1); data key 1000 + s,
  init key s, slope-check draws key 10000 + s. Test set: 4000 noiseless points (key 12345).
- Surrogate F_hat(x, u): pre-norm residual block, H = 16: h = E z + b (z = (x, u)), branch G(gamma * LN(h) + beta) with
  16 -> 32 GELU -> 16, skip h, readout w_d. 1169 parameters, LayerNorm eps = 1e-6, encoder bias initialised at 0.
  Code stores E transposed (`p["E"]` is 2 x 16).
- Six surrogate families (10 seeds each; full-batch Adam lr 3e-3, float64):
  | family (doc name) | config name in code | training |
  |---|---|---|
  | LN MSE 3000 | "MSE (3000)" | MSE, 3000 steps |
  | LN MSE 12000 | "MSE (12000)" | MSE, 12 000 steps |
  | LN slope from step 0 | "J_x + J_u (3000)" | MSE + slope loss (estimated labels) from step 0, 3000 steps |
  | LN two-stage | "J_x + J_u (two-stage)" | 3000 MSE, then 1000 with slope loss (Adam restart) |
  | no norm | "no norm, MSE" | MSE, v = h (gamma, beta kept) |
  | static norm | "static norm, MSE" | MSE, centring + frozen scale s0 |
  Slope loss: lambda * mean over M = 2048 uniform draws of [s_x^2 (dF/dx - Jx_lab)^2 + s_u^2 (dF/du - Ju_lab)^2],
  labels from a local-quadratic WLS estimator (k = 1280, Epanechnikov), lambda = 1.

### 1.2 The kink (Theorem 1, `docs/theory/theory.tex`)
- With P = I - 11^T/H, Q_E = P E = [q_x q_u], c = P b: centre z* = -(Q_E^T Q_E)^{-1} Q_E^T c, residual c_perp = Q_E z* + c,
  kink size D = sqrt(||c_perp||^2 + H eps) (0.13 to 0.90 on LN MSE 3000), half-widths r*_x = D/||q_x||, r*_u = D/||q_u||.
- Exact reduction: LN(Ez + b) = V phi(w), w = U^T Q_E (z - z*), phi(w) = sqrt(H) [w; ||c_perp||] / sqrt(||w||^2 + D^2).
  Peak feature gain sqrt(H)/D at z*; far field 1/rho across directions, 1/rho^2 along; image area independent of D
  (values bounded, slope ~ 1/D: the formal reason MSE cannot see the kink). Verified to 1e-15 (`theory/verify_thm1.py`).
- Measured: every LN kink centre within 0.1 of (0, 0); peak input-slope error 6.7 to 201 x B (median 18.7 on LN MSE
  3000); the narrowest, tallest kinks are seeds 1, 2, 8, 9. Value errors a few 1e-3 (up to ~0.05 at the centre).
- Wrong-sign arms reach 1.7 to 6.8 half-widths on 8 seeds, 9.5 (seed 1), 32.6 (seed 2); 56 to 99 % of the wrong-sign
  area lies outside the kink ellipse (`theory/arm_reach.py`).

### 1.3 LQR results (Part 1 of the talk)
- Grid per family: 11 setpoints r in {-2, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 2} x 9 starts linspace(-3, 3, 9)
  x 10 seeds; starts with |x0 - r| < 0.5 (strict; exactly one start per setpoint) excluded from rates -> 880 runs per
  family per controller. 200 steps on the TRUE plant. Controller labels in lqr_study.csv: "LQR fixed", "LQR sched";
  family labels there: "LN MSE 3000", "LN MSE 12000", "LN slope from step 0", "LN two-stage", "no norm", "static norm".
- Metrics: gap = J / J_oracle - 1 (cost around (r, u_ss(r)), u_ss = -3 r); diverged = |x| > 10;
  failed = gap > 10 % or diverged; catastrophic = gap > 100 % or diverged.
- Fixed LQR (linearise the surrogate once at its own equilibrium for r): LN MSE 3000 fails 26.5 % (10.8 % catastrophic)
  vs static norm 0.6 % (0 %). Failure peaks where the setpoint sits on the kink; setpoint input-slope error
  |B_hat/B - 1| ~ 0.49 d^-0.98 (d = setpoint distance from the kink centre in half-widths).
- Rescheduled LQR (relinearise at (x_k, u_{k-1}) each step): with an accurate setpoint gain (within 20 % of B), failures
  rise 1.1 / 9.0 / 27 / 40 % as the path's wrong-sign share grows (0, 0-2 %, 2-10 %, > 10 %); the fixed LQR stays at
  1.0 / 1.0 / 0 / 0 %. Path slopes matter exactly when the controller reads them.
- Swap test (fixed LQR, LN MSE 3000, failed / catastrophic %): both surrogate slopes 26.5 / 10.8; true A 29.8 / 12.2;
  true B 13.5 / 4.1; both true 2.3 / 0. The input slope carries the damage; the state slope matters through A - 1 = 0.03.

### 1.4 DPC results (Part 2)
- DPC: GRU (hidden 64), u = 10 tanh(head(h)); loss (sum_k 5 x_k^2 + 0.1 u_k^2 + 50 x_N^2) / N; curriculum horizons
  5/10/20/50/100/200 with 1000/2000/2000/1000/1000/2000 epochs (9000); AdamW lr 1e-3 (wd 1e-4); full batch of 1000
  starts x0 ~ U[-3, 3]; controller init key 20000 + s + 1000 rep, starts key 30000 + s + 1000 rep; 3 repeats.
  Oracle = same recipe through the true plant (matches analytic saturated LQR to 0.04 %). Test: 1000 typical starts
  (key 12345) + 22 fixed starts, on the TRUE plant.
- Labels: failed = typical-start gap > 10 %; trapped = failed AND DPC's own controller costs > 1.05 x the oracle
  controller, both evaluated through the same surrogate (an optimisation trap, not model mismatch).
- LN MSE 3000: 13/30 failed (seeds 1, 2, 8, 9 in every repeat; seed 5 once by model mismatch), 12 trapped.
  Slope from step 0: 8 trapped (seeds 2, 9 every repeat; 3 and 4 once). Two-stage, MSE 12000, no norm, static: 0/30.
  Two-stage beats LN MSE on 10/10 seeds (Wilcoxon p = 0.002) and ties static norm.
- The surrogate prices the good controller correctly (2.48 to 2.56 vs true 2.50 on seeds 1, 2, 8, 9) and rates DPC's
  own controller 3.8x, 3.4x, 7.5x, 750x worse.
- Mechanism (repeat 0): no loss barrier on the straight line from trapped to good controller; gradient at the good
  controller 1e3 to 1e35 (healthy pairs 0.03 to 3.5); one start carries 61 to 100 % of ||grad||^2; a 1e-4 relative
  weight perturbation rerandomises the gradient direction; warm-started AdamW is expelled from the good controller and
  walks uphill on its own loss; clipping each start's gradient to norm <= 3 keeps it there (diagnosis, warm start only);
  with the true J_u inside backprop the explosion disappears, with the true J_x it does not.
- Gradient sizes are NOT reproducible beyond order of magnitude (e.g. seed 9: 1.2e14 in one computation, 1.9e11 in
  another). Quote them as orders of magnitude only.

### 1.5 Theorem 2 and Lambda* (`docs/NOTES_theory.md`, `docs/theory/theory.tex` §5-6)
- Closed-loop multiplier m_k = J_x_hat(x_k, u_k) - J_u_hat(x_k, u_k) K_k, K_k = -du_k/dx_k. The training gradient is
  built from products of m_k. Safe band (|m| < 1) for K > 0: (J_x_hat - 1)/K < J_u_hat < (J_x_hat + 1)/K; at A, K*:
  0.29 B < J_u_hat < 19.7 B. Both edges matter: on seeds 2 and 8 most expanding steps are input slopes far too LARGE.
- Landing amplification Lambda*: for each training start, from the first step with |x_k| <= 0.2, the largest sum of
  log|m_k| over a contiguous stretch; max over the 1000 starts. Computed from the good (oracle) controller only.
- Result: threshold 12.1 (geometric midpoint of 8.6 and 17.0, fixed on LN MSE 3000) flags 17 of 20 traps and 0 of 160
  other runs (all 6 families x 10 seeds x 3 repeats). Misses: slope-from-0 seed 4 rep 1, seed 9 rep 1, seed 3 rep 2.
  ||grad|| <= 4.7 e^{Lambda*} on all 180 runs. Swapping J_u_hat -> B removes the expansion on 9/12 LN MSE traps.
- Memoryless policy at the LQR gain: Lambda* max 8.75 over all 180 surrogate/start sets (LN MSE 3000 seed 2 rep 0;
  `data/reference_results/theory/t2_memoryless_Sseg.csv`) -> predicts the 2-parameter linear
  DPC controller is never trapped. For the GRU, Lambda* uses the direct gain (memory ignored): a statistic suggested by
  the theorem, not a consequence of it.

### 1.6 What may and may not be claimed today
- May: the kink is structural and closed-form; MSE-accurate LN surrogates break fixed and rescheduled LQR in the
  predicted places; DPC through narrow-kink surrogates is trapped by an exploding, direction-chaotic gradient (not a
  local minimum); two-stage training removes it; Lambda* predicts traps from the good controller.
- May not (yet): "global clipping does not help" (never run); "MLP/linear behave like ..." (2-seed smoke test only);
  anything about other plants or architectures; exact gradient magnitudes.

----------------------------------------------------------------------------------------------------------------------
## 2. Repository map

```
CLAUDE.md                  this handout
README.md                  short pointer + quick commands
requirements.txt
src/                       the experiment code (copied from the claude.ai project, current versions)
  level1_core.py           plant, data, label estimator, surrogate (F_point), training, Level 1 metrics, dare_K
  dpc_core.py              DPC: controllers (gru/mlp/linear via ARCH), rollout, loss, robust_grad, run_dpc, run_diagnostic
  dpc_report.py            DPC figures/tables (D1-D11), gaps, per-start tables
  phase_core.py            Model wrapper (fj = F, dF/dx, dF/du), LQR study (fixed/rescheduled), tracking DPC, section 7
                           (reference_path_colours, cost_interpolation, closed_loop_stability)
  phase_report.py          phase portraits P1-P8
  build_dpc_notebook.py    -> dpc_through_surrogates.ipynb (embeds level1_core, dpc_core, dpc_report)
  build_phase_notebook.py  -> dpc_phase_tracking.ipynb (embeds level1_core, dpc_core, phase_core, phase_report)
  *.ipynb                  built notebooks (rebuild after ANY change to an embedded module)
theory/                    Theorem 1 check and Theorem 2 test (t2core = self-contained re-implementation)
  verify_thm1.py, t2core.py, t2_test.py, t2_memoryless.py, t2_swap.py, lambda_all.py, arm_reach.py,
  make_t2_tables.py, make_t2_fig.py
story/                     scripts behind slides 16-20 figures (story_data -> dpc_story / lqr_story / swap_story;
                           attribution = swap test for LQR and DPC); small result CSVs
companion/                 LaTeX companions to the talk (full/ and part1/, each main.tex + figures/), figure scripts
                           (make_figs.py, make_fig3d.py, run_lqr_study.py), lqr_study.csv (11 880 LQR runs),
                           background/ (background note for slides 17-19 + its figure scripts)
docs/                      notes and handoffs (see 2.1), theory/theory.tex (14 pp: Theorems 1-2 + test)
data/dpc_out/              PARTIAL copy of the Drive folder (see 3.2): all 6 surrogates, some controllers
data/reference_results/    numbers to reproduce: lambda_star_180runs.csv, gradnorm*, interp, warmstart*, path_colours,
                           stability CSVs; figP9 / figP10 PNGs
analysis/, prereg/, out/   empty: your scripts, pre-registrations, outputs
```

### 2.1 docs/
| file | what |
|---|---|
| NOTES_theory.md | Theorem 1/2 status, test numbers, review fixes, predictions for E-A1/E-A2 |
| theory/theory.tex | the full theory note with proofs (tables are generated by theory/make_t2_tables.py) |
| NOTES_dpc_level1.md | DPC design, all DPC results, trap mechanism, interventions, LQR/phase study, swap test |
| NOTES_controller_arch.md | GRU vs MLP vs linear design, predictions A1-A7, experiments E-A1..E-A6 |
| NOTES_level1_results.md | Level 1 (surrogate training) findings 1-17 |
| HANDOFF_level1.md | history and decisions up to 23 Oct, related work, reviewer attack list |
| HANDOFF_presentation_and_theory.md | slide deck structure, notation, corrections, open items as of 28 Oct |
| PLAN_paper_experiments.md | short version of Tasks 1-5, corrected 3 Nov (CLAUDE.md is authoritative) |
| RESULTS_LOG.md | append-only run log (you keep it) |

### 2.2 Living documents outside this repo (claude.ai; read-only for you)
Slide deck https://claude.ai/artifact/K51GsPfQQ1jdU1aw6MaeHy (24 numbered slides + 2 glossary + 8 appendix);
presenter's briefing https://claude.ai/artifact/GkfPr6HhxRKsCKWTfkeZpC; handover doc
https://claude.ai/artifact/Hiy6mvaiXoT7mBMorSb5yT. The reference derivation "The Prenorm LayerNorm Jacobian Anomaly"
(PDF in the claude.ai project) is the source of the notation; ask the user for the PDF if you need it.

----------------------------------------------------------------------------------------------------------------------
## 3. Task 0 — environment, data, paths, reproduction (do first; ~half a day)

### 3.1 Environment
```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
All scripts use float64 (`jax.config.update("jax_enable_x64", True)` or the modules do it). CPU is enough for every
local task; GPU work goes to Colab.

### 3.2 Data
The full results live on the user's Google Drive in `MyDrive/dpc_out` (plus `MyDrive/level1_out` for Level 1). The
bundled `data/dpc_out` is partial:
- present: `surrogates/` all six families (10 seeds each); `controllers/` oracle reps 0-2, LN MSE 3000 reps 0-2,
  slope-from-0 reps 0-2, MSE 12000 / two-stage / static norm rep 0; `track_controllers/` (3 files).
- missing: no-norm controllers; reps 1-2 of MSE 12000 / two-stage / static / no norm; and all tables
  (`dpc_results.csv`, `dpc_diag.csv`, `dpc_curves.csv`, `surrogates.csv`, `phase_lqr.csv`, `track_results.csv`,
  `trajectories/`).
Ask the user to download the whole `MyDrive/dpc_out` folder (tens of MB) and replace `data/dpc_out` with it. Then write
`tools/check_data.py` that prints an inventory and checks: 6 surrogate files with seeds 0-9; 7 models x 3 repeats of
controller files; `dpc_results.csv` has 7 models x 10 seeds x 3 reps rows. File names: `safe(name)` replaces every
non-alphanumeric character by `_` (e.g. "LN, MSE (3000)" -> `LN__MSE__3000`, "true plant (oracle)" ->
`true_plant__oracle`, surrogate "J_x + J_u (two-stage)" -> `J_x___J_u__two_stage`); repeats > 0 add `__rep1`, `__rep2`.

### 3.3 Paths
Scripts written in the analysis sandbox hard-code `/home/claude/...`, including
`sys.path.insert(0, "/home/claude/kinkwork")` (that folder is now `src/`) (story/*.py, companion/*.py,
companion/background/*.py, theory/t2core.py `DATA`, theory/t2_test.py `OUT`, theory/make_t2_fig.py,
theory/make_t2_tables.py). Add one `paths.py` at the repo root (REPO, DATA = env `LK_DATA` or `data/dpc_out`, OUT =
`out/`) and make those scripts use it (or the env vars `T2_DATA`, `T2_OUT` that t2core/t2_test already read). Do not
change any computation.

### 3.4 Reproduction checks (acceptance for Task 0; report every mismatch, do not "fix" numbers)
1. `cd theory && T2_DATA=../data/dpc_out python verify_thm1.py`: identity errors <= 1e-14, singular values 1.7e-6,
   areas exact, block Jacobian 1e-9, trained surrogates 2e-15 (already passes with the bundled data).
2. Theorem 2 test (needs `dpc_results.csv`): `python t2_test.py "LN MSE 3000"` then
   `python t2_test.py "LN MSE 12000,LN slope from 0,LN two-stage,no norm,static norm"`, `python t2_memoryless.py A B`,
   `python t2_swap.py`, `python lambda_all.py`, `python make_t2_tables.py`. Expect: S_seg_max per run equal to
   `data/reference_results/lambda_star_180runs.csv` (3 digits), threshold 12.11, 17/20 traps flagged, 0/160 false,
   memoryless max 8.75 (`reference_results/theory/t2_memoryless_Sseg.csv`), swap column equal to
   `reference_results/theory/t2_swap.csv` (9 of the 12 LN MSE traps fall below 12.1 with J_u_hat -> B). CPU time:
   roughly 1-2 h for all families. Note make_t2_tables.py / make_t2_fig.py expect the second t2_test output under the
   name `t2_runs_LNMSE12000_LNslopefrom0_LNtwo-stage_nonorm_staticnorm.csv` (that is what t2_test writes).
3. `companion/run_lqr_study.py` -> new lqr_study.csv equals `companion/lqr_study.csv` (max abs difference < 1e-9).
   From it: LN MSE 3000 fixed LQR 26.5 % failed / 10.8 % catastrophic (rule in §1.3).
4. Build both notebooks and run the DPC notebook locally in `QUICK = True` mode (CPU, ~10 min) to check the pipeline.
5. Log all of this in `docs/RESULTS_LOG.md`.

----------------------------------------------------------------------------------------------------------------------
## 4. Conventions

- Notation (paper and code comments): Q_E = P E = [q_x q_u], c = P b, z* = (x*, u*), c_perp, s = z - z*,
  delta^2 = ||c_perp||^2/H + eps, D = sqrt(H) delta, r*_x = D/||q_x||. Q is RESERVED for the LQR weight (5). Seeds are
  "seed 2", never s. Multiplier m_k, landing amplification Lambda*, safe band.
- Names: "two-stage training" (never "retrofit"); "slope loss" = Jacobian loss; the six family names in §1.1.
- Metric definitions are exactly those of §1.3 and §1.4; for DPC the typical-start gap uses the oracle of the same
  seed and repeat.
- Random keys: never change the existing key scheme (data 1000 + s, init s, draws 10000 + s, controller 20000 + s +
  1000 rep, starts 30000 + s + 1000 rep, test 12345). New experiments that need new randomness use new offsets
  (e.g. 40000+) and document them.
- Figures: white background; dark red #8B1A1A (bad / trapped / overestimated input slope), dark blue #1B3A6B (good /
  wrong-sign or too-small input slope in the multiplier plots), grey for neutral; slope maps use RdBu_r with
  TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5) for J_u/B (white = correct, red = too large, blue = too small/wrong sign).
- LaTeX for the user: plain style, no bold/italic emphasis, no boxes, booktabs tables, figures with full captions;
  compile with latexmk and fix overfull boxes.
- Statistics: the unit of replication is the SEED (the surrogate). Paired comparisons over seeds with Wilcoxon
  signed-rank; repeats are averaged per seed first (median over repeats) unless stated.

----------------------------------------------------------------------------------------------------------------------
## 5. Known pitfalls (read before Tasks 1-3)

1. Surrogate cache: `dpc_core.train_surrogates` loads `<out>/surrogates/<config>.npz` if it exists and only checks the
   seed list, NOT the plant. With a different plant you must use a fresh output folder for surrogates AND labels, or
   it silently reuses the B = 0.01 models. Add a guard: store (A, B, sigma) in new surrogate files and assert on load
   (old files without the key are (1.03, 0.01, 0.01)).
2. Hard-coded plant constants (must become plant-derived before any new plant): `dpc_core.LQR_K = 10.29` (used by
   `lqr_saturated` and the oracle sanity check, `dpc_report` D9 grey line), `phase_report.slope_maps` (0.01 and 1.03),
   `phase_report.path_table` and `fig_setpoint_distance` (0.01), `phase_core` docstring u_ss = 3 r, `theory/t2core.py`
   (A_TRUE, B_TRUE), `theory/t2_memoryless.py` (K_STAR), `level1_core.lqr_screen` (3.33), plot labels that say
   "1.03 x + 0.01 u" or "u = -3x", colour scales normalised by B = 0.01 in story/ and companion/ scripts.
   grep for `1.03`, `0.01`, `10.29`, `10.2875`, `3.33`, `-3 *`, `u_ss`. Regression: with B = 0.01 every output must stay
   identical.
3. The notebooks embed module sources with `%%writefile`. After changing a module, rebuild the notebook
   (`python src/build_dpc_notebook.py`, `python src/build_phase_notebook.py`) and give the user the new .ipynb.
4. `RUN_ROBUST = True` is the default in the DPC notebook's robust block: it trains the two per-start-aggregation
   models for whatever ARCH is set. Tell the user to set it False unless the task wants it.
5. `ONLY` in the DPC notebook selects models by name; the oracle is "true plant (oracle)".
6. `arch_dir`: GRU writes to the root folder; mlp/linear to `<root>_mlp`, `<root>_linear`; surrogates are read from
   `SUR_DIR` (the root). Keep this for every new experiment.
7. float32 runs (L4/T4) write to `<root>_f32`; never mix with float64 numbers.
8. 12 000-step surrogate training drifts slightly between GPU and CPU; 3000-step training is reproducible to every digit.
9. Gradient norms of order 1e10+ differ between runs and machines (chaotic accumulation). Never compare them beyond
   order of magnitude, and never use them as a regression check.
10. `dpc_results.csv` rows for repeat 0 may lack a `rep` column in old files: `_read_runs` fills 0.

----------------------------------------------------------------------------------------------------------------------
## 6. The tasks

Order: Task 0 -> Task 1 and Task 3 (code + pre-registration, then one Colab session) -> Task 4 and Task 5 (local) ->
Task 2a -> report to the user -> Task 2b only after he confirms the plant. Each task ends with: results log entry,
`docs/RESULTS_<task>.md` (plain language, numbers, figures), and a short message to the user.

### Task 1 — MLP and linear DPC, with a prospective test of Theorem 2

Why: rules out "the trap is an artefact of the recurrent policy"; first PROSPECTIVE test of Lambda*.
Background: `docs/NOTES_controller_arch.md` (A1-A7, E-A1..E-A6), `docs/NOTES_theory.md` (predictions).

1.1 Memoryless Lambda* code. In theory/, add `t2_arch.py`: load a memoryless controller file (MLP tree `L0..Ln` or
    linear `k, b`; see `dpc_core.mlp_step` / `linear_step`, note `LIN_SCALE = 10` and u = 10 tanh(...)), roll it through
    each surrogate from that repeat's 1000 training starts (key 30000 + s + 1000 rep), compute exact multipliers
    (K_k = -du/dx exactly, no memory), landing window |x| <= 0.2, Lambda per start (max_subarray of log|m_k|), Lambda*,
    and also the exact sensitivity check (sum log|m_k| = log|dx_N/dx_0| to ~1e-8, which holds for memoryless policies).
    Unit check: the linear controller at k = -10.2875, b = 0 must reproduce `t2_memoryless.py` part A per run.
1.2 Stage A (Colab, user, ~10-20 min): DPC notebook with `ARCH = "mlp"`, `ONLY = ["true plant (oracle)"]`,
    `REPS = [0, 1, 2]`, `RUN_ROBUST = False`, `DTYPE = "float64"`, Drive root `MyDrive/dpc_out` (results go to
    `MyDrive/dpc_out_mlp`). This trains only the MLP oracle. Same with `ARCH = "linear"` (-> `dpc_out_linear`).
1.3 Pre-registration (you, before Stage B): compute Lambda* of the MLP oracle through all 6 families x 10 seeds x 3
    repeats; write `prereg/PREREG_arch.md` with: the list of runs predicted trapped (Lambda* > 12.1, threshold NOT
    re-tuned), the linear prediction (no run trapped; also compute its Lambda* from the trained linear oracle), and
    predictions A1-A7 copied with any updates. Commit; put the commit hash in the file and in the results log.
1.4 Stage B (Colab, user, ~1.5 h + ~1 h): `ONLY = None`, `ARCH = "mlp"`, `REPS = [0, 1, 2]`, `RUN_ROBUST = False`;
    then the same with `ARCH = "linear"`. The notebook's diagnostic cell writes `dpc_diag.csv` (trap test) and figures
    D9 (policy maps), D10 (linear landscape), D11 (gradient audit). User downloads `dpc_out_mlp`, `dpc_out_linear`.
1.5 Analysis (`analysis/arch_report.py`): per architecture x family: failed / trapped counts out of 30, median gap,
    paired Wilcoxon vs the GRU; confusion matrix of the pre-registered Lambda* prediction vs observed traps (hits,
    misses, false alarms) for the MLP and the linear controller; K_eff table (D9) against the LQR gain; outcomes of
    A1-A7, each marked held / failed / partly with the numbers. Write `docs/RESULTS_arch.md`.
Acceptance: pre-registration committed before Stage B results existed; every prediction reported with its outcome.
If the MLP is NOT trapped on the GRU-trapped seeds, the paper claim narrows to "recurrent / high-direct-gain policies";
say so plainly.

### Task 2a — Same plant with ordinary authority, B = 0.1

Why: the most likely reviewer objection is "low authority makes this a toy". Cheapest test: one line of plant change.
IMPORTANT correction to an earlier plan: the safe band does NOT stay the same in units of B. With B = 0.1:
K* = 5.2538, P = 10.41, pole 0.505; safe band for the LQR gain 0.057 B < J_u_hat < 3.86 B (vs 0.29 B to 19.7 B);
a wrong-signed input slope J_u_hat = -B gives m = 1.56 (vs 1.13 at B = 0.01); holdable |x| <= 33.3; u_ss = -0.3 x.
So the closed loop is MORE sensitive to a given relative slope error; whether traps persist depends on whether the
surrogate's relative slope errors shrink. The outcome is genuinely open.
2a.1 Make every plant constant plant-derived (pitfall 2) and add the surrogate-cache guard (pitfall 1). Add a
     "surrogates only" option to the DPC notebook (train/load the six families and stop) so surrogates and the oracle
     can be trained before any DPC-through-surrogate run. Regression: B = 0.01 outputs unchanged (QUICK run + LQR study).
2a.2 Stage A (Colab): DPC notebook with `PLANT = L.Plant(A=1.03, B=0.1)`, Drive root `MyDrive/dpc_out_B01` (new,
     empty), surrogates only + `ONLY = ["true plant (oracle)"]`, `REPS = [0, 1, 2]`. LABEL_DIR must be inside the new
     root (labels depend on the plant).
2a.3 Pre-registration (you): from the Stage A outputs compute, locally: the kink geometry and peak |J_u_hat/B - 1| per
     seed (Level 1 metrics in `surrogates.csv`/the npz), the full fixed/rescheduled LQR study (deterministic, minutes),
     and Lambda* of the B = 0.1 GRU oracle through the B = 0.1 surrogates. Write `prereg/PREREG_B01.md`: predicted DPC
     traps (Lambda* > 12.1), plus qualitative predictions (relative slope errors smaller than at B = 0.01; catastrophic
     fixed-LQR failures only where the setpoint slope leaves the band). Commit.
2a.4 Stage B (Colab): DPC through the six surrogates, GRU, `REPS = [0, 1, 2]`, same root. Then the phase notebook's
     LQR part is already done locally; tracking DPC optional.
2a.5 Analysis: the same tables and figures as for B = 0.01, side by side (LQR failure rates, setpoint law, path colour,
     DPC traps per family, Lambda* scatter with the pre-registered threshold). `docs/RESULTS_B01.md`.
Decision rule to state in the report: if traps vanish at B = 0.1, the claim becomes "low-authority plants", and the
two-state plant (2b) should also be low authority.

### Task 2b — A two-state plant (start only after the user confirms the plant)

Propose to the user, with the reasoning, then implement. Selection criteria: open-loop unstable; low authority like
the scalar case; controllable; equilibria with u_ss != 0 that move with the setpoint (so the setpoint study
generalises); the recoverable set inside the data box. Example that meets them (verify before proposing):
A = [[1.02, 0.05], [0, 1.01]], B = [[0], [0.01]] -> equilibria x2 = -0.4 x1, u_ss = 0.4 x1; eigenvalues 1.02, 1.01;
[B, AB] full rank. Check the recoverable set under |u| <= 10 by simulating the saturated LQR from a grid, and choose
the data box ~1.2x that set.
Code changes (keep scalar code paths untouched; add n_x-general versions):
1. `level1_core`: Plant2 (A 2x2, B 2x1, f, Jx 2x2, Ju 2x1, u_ss(r) for a position setpoint); data in the 2-D box;
   `init_params(seed, n_in=3, n_out=2)` (E: 3 x 16, Wd: 16 x 2); vector `F_point`; `frozen_sigma`; slope labels from a
   local-quadratic WLS in 3 inputs for each of 2 outputs; slope loss over the 2 x 3 Jacobian with per-input scales;
   Level 1 metrics generalised (peak |J_u_hat - B| relative to |B|, state-slope error as a matrix norm).
2. Kink metrics from Theorem 1 with n_z = 3 (an ellipsoid): z* in R^3, D, half-widths D/||q_i|| per input axis.
3. `dpc_core`: d_x = 2 (GRU and MLP already accept d_x); vector dynamics; loss with Q = diag(q1, q2) (propose
   diag(5, 0.5)), R = 0.1, Q_f = 10 Q; oracle sanity check against scipy.linalg.solve_discrete_are + saturation.
4. LQR study: fixed and rescheduled LQR from the surrogate's 2 x 2 / 2 x 1 Jacobians via DARE; setpoints in position
   (e.g. 11 values), starts on a grid of (x1, x2); same metrics.
5. Theorem 2 for a vector state: M_k = J_x_hat - J_u_hat K_k (2 x 2); Lambda for a stretch = log ||M_{k2-1} ... M_{k1}||
   (operator norm; not additive, compute running products per start, O(N^2) is fine); define it in a short LaTeX
   note before using it.
6. Figures: slices of the 3-D slope field (u = u_ss(x1) plane and x2 = 0 plane) with the kink ellipsoid's sections and
   the LQR/DPC paths; the same three headline figures as for the scalar plant.
Milestones: (i) smoke test of data + one surrogate + kink metrics on CPU; (ii) all families QUICK; (iii) Colab full run
(Level 1 ~1 h, DPC GRU + MLP ~3-4 h); (iv) analysis. Report after each milestone.

### Task 3 — Optimiser baselines (does a standard trick already fix it?)

Background: `dpc_core.robust_grad` (aggregate "mean" | "clip" | "unit"), `ROBUST_MODELS`, story/story_data.py (warm start).
3.1 Code (defaults unchanged; regression: aggregate "mean" bit-identical):
    - aggregate "global": mean gradient, then rescale to norm <= c (as optax.clip_by_global_norm);
    - aggregate "mom": median of means (split the starts into G = 10 groups, mean per group, coordinate-wise median);
    - minibatch: model option "batch" = b starts sampled each epoch (new key offset 40000 + s + 1000 rep, fold_in epoch);
    - model options "lr" (AdamW learning rate) and "curriculum" (e.g. stop at N = 50 or 100).
    Put the settings in a new `OPT_MODELS` list with unique names (e.g. "LN, MSE (3000), global clip 3", "... mom",
    "... batch 100", "... lr 1e-4", "... N <= 50", "... N <= 100"), and a notebook block `RUN_OPT` (default False).
3.2 Warm-start harness (local CPU): generalise the warm start in story/story_data.py to any aggregation: start AdamW at
    the oracle controller, 100 epochs at N = 200, on LN MSE seeds 1, 2, 8, 9, slope-from-0 seeds 2, 9 (broken) and LN MSE
    seeds 3, 7, two-stage seeds 2, 9 (healthy); log true-plant cost and surrogate loss every 5 epochs. Reference results
    for "mean" and "clip 3": `data/reference_results/warmstart*.csv`, figP10.
3.3 Pre-register (prereg/PREREG_opt.md), from the plan: per-start clip from scratch rescues at least 3 of the 4 LN MSE
    trapped seeds; global clip does not rescue (direction still set by one start) - warm start expelled; median of means
    rescues; lr 1e-4 does not rescue; truncated horizon N <= 50 avoids the trap but gives a worse true-plant controller
    (report its gap); mini-batches: no prediction (state "uncertain"). Commit.
3.4 From scratch (Colab, user, ~3 h): DPC notebook, `ONLY = []` for the main models (or skip them), `RUN_ROBUST = True`,
    `RUN_OPT = True`, GRU, REPS = [0, 1, 2], LN MSE 3000 surrogates; separate folder root `MyDrive/dpc_out` is fine
    because names are unique, but confirm no file collisions first.
3.5 Analysis: one table, rows = settings (plain, clip 3, unit, global 3, mom, batch 100, lr 1e-4, N <= 50, N <= 100,
    plus two-stage plain as the reference row), columns = failed / 30, trapped / 30, median gap, gap of the best seed,
    wall time. `docs/RESULTS_opt.md`.

### Task 4 — Statistics (local, ~1 day)

`analysis/stats.py`, outputs in `out/stats/` and `docs/RESULTS_stats.md` (+ a LaTeX table file).
4.1 Seed-level bootstrap (10 000 resamples of the 10 seeds with replacement; a seed carries all its runs, including
    its 3 DPC repeats), 95 % percentile intervals for: fixed and rescheduled LQR failed / catastrophic per family
    (from lqr_study.csv; must reproduce 26.5 / 10.8 for LN MSE fixed); per-setpoint failure rates (fig P3); the
    path-colour bins (the 4 LN families only, |Bh/B - 1| <= 0.2, failed = gap > 10 % or diverged; wrong_sign_share bins 0, (0, 0.02], (0.02, 0.1], > 0.1; must
    reproduce fixed 1.0 / 1.0 / 0 / 0 % on n = 2014, 204, 46, 24 and rescheduled 1.1 / 9.0 / 27 / 40 % on n = 2083,
    111, 44, 50); DPC failed / trapped per family (dpc_results.csv + dpc_diag.csv); LQR swap rates (regenerate per-run
    data with story/attribution.py part 1 and save it).
4.2 Paired claims with per-seed differences, bootstrap CIs, Wilcoxon p and Holm adjustment over the family of
    claims: two-stage vs LN MSE; LN MSE vs static; LN MSE vs no norm; two-stage vs static; two-stage vs no norm;
    MSE 12000 vs MSE 3000 (typical-start DPC gap, per-seed median over repeats).
4.3 Threshold sensitivity: failed rate at gap thresholds 5, 10, 20, 50, 100 % per family and controller; threshold-free
    metrics: median steady offset |x_inf - r| and settling time to a 5 % band (add these columns to the LQR study output
    via an option; do not change the existing columns).
4.4 Lambda* threshold robustness: leave-one-seed-out on LN MSE 3000 (threshold = geometric midpoint from the other 9
    seeds, classify the held-out seed's 3 runs); and the full interval of thresholds with 0 false alarms on the 160
    non-trapped runs together with the hit rate across it.

### Task 5 — Wording and figure fixes (local, an afternoon)

5.1 Companions (`companion/full/main.tex`, `companion/part1/main.tex`), both: (a) wrong-sign arms "2 to 4
    half-widths" -> "2 to 7 half-widths on 8 of 10 seeds (10 and 33 on the two narrowest, seeds 1 and 2); 56 to 99 % of
    the wrong-sign area lies outside the kink ellipse"; (b) "K scales like 1/B_hat" (or equivalent) -> "an
    overestimated input slope lowers K (B_hat = 18.3 B gives K = 3.98, a factor 2.6, not 18); it becomes dangerous
    when K < 3, where the true pole 1.03 - 0.01 K reaches 1, i.e. roughly B_hat > 30 B"; (c) where the slide 16 panels
    are described: seed 1 parks at x = -0.58 (7 of 10 starts) or +1.71 (3 starts); seed 2 parks at x = +1.10 on 9 of 10
    starts after a bang-bang transient, and only the start x0 = -2.5 keeps cycling. grep first, list every hit,
    then edit; recompile both with latexmk, no overfull boxes.
5.2 Regenerate the slide 16 figure with corrected panel titles (`story/dpc_story.py`, PANELS: seed 1 "parks at
    x = -0.58 or +1.71", seed 2 "oscillates, then parks at x = 1.1 (one start keeps cycling)"); needs
    `story/story_data.py` first (closed_loops.npz). Save to `out/figs/`.
5.3 `out/slide_fixes.md`: exact old -> new text for slide 5 (big number "2-7", caption, notes), slide 8 notes (K vs
    B_hat), slide 16 (titles + notes), slide 17 notes (write the multiplier as J_x_hat - J_u_hat K with u = -K x; both
    band edges; gradients as orders of magnitude; e^{N lambda} is an order-of-magnitude rule that fails on seed 1).
5.4 Optional: compute the far-field branch function Phi(theta) from the weights (Corollary 4 in theory.tex) and check
    the decay laws (1/rho across, 1/rho^2 along) on the trained surrogates; otherwise soften "-0.98" to "about -1 along
    the setpoint line".

----------------------------------------------------------------------------------------------------------------------
## 7. Reporting back to the user

- After each task: `docs/RESULTS_<task>.md` in plain language (what was run, numbers with CIs, figures, which
  predictions held), an entry in `docs/RESULTS_LOG.md`, and a short message: what came out, what it means for the
  paper claim, what he needs to run next (exact notebook settings and expected time).
- When a result contradicts a statement in the companions, slides or notes, list every place that needs correcting.
- If you derive anything mathematical, write it as LaTeX and compile it to PDF.

## 8. Paper context (for prioritising)

Venue estimate given 3 Nov: controls (CDC/ACC/ECC) ~50-60 % now, ~70 % with Tasks 1-5; L4DC ~40 % -> ~60 %;
TMLR ~45 % -> ~65 %; NeurIPS/ICML/ICLR main ~15 % -> 35-45 %, where a second plant (Task 2) is the hard requirement.
Novelty that is already defensible: the mechanism (accurate predictions, slope errors break gradient-based control,
proven by intervention), Theorem 1 with corollaries, the "when the controller reads the slopes" synthesis, and the
predictive Lambda* test. Related work to cite is listed in `docs/HANDOFF_level1.md` and `docs/theory/theory.tex`.
