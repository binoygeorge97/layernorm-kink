# Queue status (4 Nov work queue). Newest entries at the bottom.

## Q0 (partly done; A1 pending) - 2026-11-04 17:35 UTC
- A2 timing addendum and oracle hashes: d8fffc2 (all nine hashes match). A3 Run 5 row checks and spot check: f1e7a72.
- B3, the lr 1e-4 convergence rule for Run 4: 84ecb3a (17:19 UTC). Healthy LN MSE seeds 0, 3, 4, 6, 7; each must reach gap <= 10 % in >= 2 of 3 repeats, else P4 "not testable: not converged". Run 4 may start.
- A1 DONE 17:56:34 UTC, commit 91238b4: Lambda*_aug predicts 0 of 180 Run 6 runs trapped (max 9.66, LN MSE 3000 s7 r0); the direct-gain rule (77d8455) predicts 1 (LN MSE 3000 s2 r2). Will be labelled "committed after Run 6 began writing, before any Run 6 file was read"; first Run 6 file in dpc_out_B01: trajectories/LN__MSE__3000.npz, created 2026-11-04T16:50:14.902Z.

## Q1 (done) - 17:35 UTC, commit 180ec7b
- analysis/b01_report.py frozen: LQR rates, setpoint law, path colour, DPC per family, Lambda scatter (direct, aug), confusion matrices and outcomes R1, R2, aug rule, CLAUDE.md decision rule; non-registered model rows excluded and listed.
- End-to-end test on the B = 0.01 data (out/test_b01_report_on_B001): reproduces 26.5 / 10.8 % fixed-LQR LN MSE, path colour 1.0/1.0/0/0 and 1.1/9.0/27/40 %, 12 + 8 traps, direct 17/20 hits 0 FA, aug 15/20 0 FA.
- Scoring convention fixed in the script before results (undefined case): runs predicted trapped but none trapped -> "held (false alarms only)" if false-alarm rate <= 5 %. Setpoint-law fit in the script (log-log over all LN fixed-LQR setpoints) gives 0.39 d^-0.88 at B = 0.01, not the 0.49 d^-0.98 quoted in CLAUDE.md (different subset/fit); reported as is.

## Q2 (done earlier today) - commit f1e7a72
- Run 5 analysis: docs/RESULTS_arch.md; PREREG_arch outcomes appended. MLP trapped 27/30 LN MSE 3000, 12/30 slope from 0, 1/30 MSE 12000; linear 0/180.
- P1 failed (0 of 40 traps predicted; MLP oracle Lambda* <= 7.4), P2' failed (12 false alarms), P2 and A2 held, A1 and A3 partly, A4 failed narrowly (seed 1 -14.9 %), A5 not scored.

## Q3 (done) - 17:36 UTC, commit 218cbbe
- analysis/opt_report.py frozen: Task 3.5 table (failed, trapped, median gap, best-seed gap, wall time s/seed), rescue table, paired vs plain, and score(): P1, P2a, P3a, P4 with the 84ecb3a convergence rule (healthy LN MSE seeds 0, 3, 4, 6, 7), P5a, P5b, U1.
- Tested on the local QUICK Run-4 smoke output (incomplete branches) and on a synthetic complete set (plain B = 0.01 rows copied under each Run 4 name: P1 failed, P2a held, converged, P4 held, P5a failed, P5b held, as expected).

## Q5 (partly done) - 17:37 UTC, commit 24a75cc
- B4: docs/RESULTS_t2_compare.md. Direct gain 17/20 hits, 0 FA; exact price 14/20, 2 FA; anchored scalar 17/20, 0 FA; anchored augmented 15/20, 0 FA. Anchoring costs nothing; memory costs slope-from-0 s9 r0, s9 r2.
- B2(i): docs/RESULTS_mechanism.md. Surrogate and true cost rise together; "tame" (no start > 1e3) and "agree" (cos >= 0.99 of mom with mean and clip-3) hold on 4 of 6 broken pairs only; RESULTS_opt item 3 "norms 3-40" overstated (noted, committed claim not edited).
- B2(ii) 2x2 warm start and B2(iii) step-scale check: scripts committed, run after the B = 0.1 job frees the CPU.

## Q6a (done) - key scheme for seeds 10-29
- data numpy default_rng(1000 + s) 1010-1029; init PRNGKey(s) 10-29; draws PRNGKey(10000 + s); controller 20000 + s + 1000 rep (20010-22029); starts 30000-32029; batch 40000-42029; robustness 50000-52029 (fold_in). No collision among these for s <= 29.
- Pre-existing overlap to know about: tracking-DPC controller init (phase_core) also uses 50000 + s (+1000 rep) as a base; lambda_robust derives fold_in keys from the same integers, so the actual keys differ, but the documented ranges overlap. Test starts reuse numpy rng 12345 (Level 1 test set and DPC test starts), as before.
- Pitfall for seeds 10-29: dpc_core.train_surrogates caches by seed list; a run with SEEDS = 10..29 in data/dpc_out would RETRAIN and overwrite the seeds-0-9 files. Use a new root (data/dpc_out_s10_29 locally, MyDrive/dpc_out_s10_29 on Colab).

## Q6b (done) - 17:38 UTC, commit 23b5835
- prereg/PREREG_heldout.md: H1 rates in 95 % prediction intervals (held if <= 2 of the scored cells outside), H2 LN MSE 3000 > static norm (fixed LQR), H3 all LN kink centres within 0.1.
- Deviation from the request, flagged: the single-stage interval (20 seeds from the fixed 0-9) is NARROWER than the CI of the mean (LN MSE fixed failed [21.7, 31.3] vs CI [19.7, 33.2]); registered the two-stage interval (bootstrap 0-9, then draw 20) as primary: [18.2, 34.4]. Both shown.

## Q7a-c (done) - 17:52 UTC, commits b349d6d, a9b34f2, e221377
- Q7a: docs/theory/rmsnorm/rmsnorm_kink.pdf. RMSNorm = Theorem 1 with P = I; reduction verified to 9e-16 (theory/verify_rms.py), peak gain sqrt(H) sigma_max / D to 4e-9; for the same weights D_RMS >= D_LN (P is a contraction), differing by <= 4 % on the trained LN MSE 3000 encoders.
- Q7b: configs "RMS MSE (3000)", "RMS J_x + J_u (two-stage)", DPC list RMS_MODELS (not in MODELS), t2core families "RMS MSE 3000", "RMS two-stage". tools/regress_norm.py: 4 existing configs bit-identical (weights and metrics), t2core identical; tools/regress_dpc.py PASSED.
- Q7c: prereg/PREREG_rms.md Part 1: M1 centres within 0.1, M2 kink size within factor 2 of LN MSE 3000, M3 fixed-LQR failed in [13.3, 53] %, M4 two-stage lower and <= 13.3 %.

## A1 / Q0 complete - 17:56 UTC, commit 91238b4 (see Q0 above)

## Q8 (code done, milestone (i) passed; (ii) running) - commit f380964
- src/plant2_core.py: Plant2(A, B) parameters, data box, 3-input / 2-output surrogate, 3-input local-quadratic WLS labels (fixed k = 1500; the scalar k-selection is not ported yet), slope loss with per-input scales, Level 1 metrics, ellipsoid kink metrics, DARE LQR study (fixed / rescheduled), GRU / MLP DPC with d_x = 2, vector Lambda.
- docs/theory/vector_lambda/vector_lambda.pdf written before use: Lambda = max over stretches after landing of log ||M_k2 ... M_k1||_2, M_k = J_x_hat - J_u_hat K_k.
- tools/test_plant2.py (milestone i, P1 plant): equilibria exact, labels on noiseless linear data 2.8e-8, vector Lambda = scalar on diagonal / 1x1 cases (9e-16), one surrogate + kink metrics + mini LQR + short DPC: PASSED. Scalar code untouched (separate module; regress_dpc passed).
- Milestone (ii) tools/plant2_quick.py (6 families x 2 seeds, scaled 0.1) running in background. No Colab run, no plant choice.

## Q9 (done) - commit ac9ecc2
- docs/RESULTS_far_field.md: Phi(theta) from the weights; branch -> Phi as 1/rho; branch-gradient exponents along rays -1.84 to -2.01, across -0.75 to -0.99 (medians per seed). Decay laws confirmed, so "-0.98" need not be softened; suggested phrasing "about d^-1 (fit -0.98 on LN MSE 3000)".

## Q4 (done) - 18:02 UTC, commits d8596c5, 8bd07a2
- Run 6 analysed with the frozen b01_report.py, unchanged. B = 0.1 oracle hashes match. 30 rows per family, no robust rows.
- Traps persist: LN MSE 3000 7/30 trapped (s1 r0-r2, s2 r0-r2, s9 r2); all other families 0. B = 0.01 had 12 + 8.
- R1 (direct Lambda*, 77d8455): 1 hit, 6 misses, 0 FA -> not held (script label "partly"); R2 held; aug rule (91238b4) failed (0 predicted).
- Decision rule: claim NOT restricted to low-authority plants. docs/RESULTS_B01.md lists the claims about Lambda* that now need your decision (three prospective failures).

## Q10 (done, slots open) - commit 47fdd6a
- python paper/make_figs.py -> paper/figs (7 figures: LQR rates, kink geometry, swap, DPC traps per architecture/plant, Lambda* scatter retrospective + 3 prospective, optimiser, B = 0.1 side by side); python paper/make_tables.py -> paper/tables (8 booktabs tables; compile-checked, no errors or overfull boxes; needs graphicx).
- Pending slots fill automatically: Run 4 (re-run analysis/opt_report.py), Run 8 (data/run8_s10_29), Run 9 (data/run9_rms), held-out (out/heldout/outcomes.csv), RMS (out/rms/summary.csv).

## Q6d (prepared) - Colab Runs 7 and 8 (held-out seeds 10-29), notebook rebuilt at ca00843
Use the notebook from commit ca00843 (src/dpc_through_surrogates.ipynb; for Run 4 the earlier notebook is fine: the changes only
add RMS options and a seed guard). I will QUICK-test this notebook locally before you start Run 7 (entry below when done).
Run 7 (surrogates + GRU oracles; Drive folder MyDrive/dpc_out_s10_29, new; ~45-60 min on an A100):
```
QUICK = False; ARCH = "gru"; SEEDS = list(range(10, 30)); DTYPE = "float64"
PLANT = L.Plant(A=1.03, B=0.01, eps_nl=0.0); ROOT = "dpc_out_s10_29"; LABEL_DIR = None
SURROGATES_ONLY = True; ONLY = ["true plant (oracle)"]; REPS = [0, 1, 2]
RUN_ROBUST = False; RUN_OPT = False
```
Estimate: six surrogate families x 20 seeds about 2x Run 3 (labels ~16 s per seed), plus 3 x 20 oracle runs (~35 s per seed per repeat).
After Run 7 (me): hash-compare the Colab 3000-step surrogates with the local ones (data/dpc_out_s10_29; MSE 12000 may drift, pitfall 8);
Lambda* of the seed 10-29 oracles through the surrogates (threshold 12.1, unchanged); pre-register Run 8 and commit BEFORE you start it.
Run 8 (DPC; same folder; ~2.5-3 h): as Run 7 but
```
SURROGATES_ONLY = False; ONLY = ["LN, MSE (3000)", "LN, J_x + J_u (3000)", "LN, two-stage", "static norm, MSE"]; REPS = [0, 1, 2]
```
(4 models x 20 seeds x 3 repeats x ~35 s per seed per repeat ~ 2.3 h, plus evaluation and the diagnostic cell; the oracles of Run 7 are reused).

## R1 (done) - 18:16 UTC, commit be5e1ca: does Run 4 (ONLY = []) shrink the dpc_out tables? No.
- Code (notebook at ce89f6f and a42bb3b alike): MODELS = [] when ONLY = []. run_dpc then trains nothing and only APPENDS to
  dpc_results.csv / dpc_curves.csv (append_rows); surrogates.csv is rewritten only if rows were trained, and then MERGED with
  the old rows. run_diagnostic(MODELS = []) produces no rows and does not write dpc_diag.csv (it only reads it). The robust
  and optimiser blocks append their own rows (run_dpc / run_diagnostic per model). Only the derived tables/*.csv
  (dpc_summary, dpc_paired, policy_gains) are rewritten, from the full files. Same as the end-to-end QUICK test of 3 Nov.
- Drive now (MyDrive/dpc_out, 18:15 UTC): dpc_results.csv 116 457 B, dpc_diag.csv 22 047 B, surrogates.csv 11 948 B,
  dpc_curves.csv 372 577 B, all last modified 22 Oct; byte sizes equal my local copies. Backup of all four (210 / 180 /
  60 / 7560 rows) committed in data/pre_run4_backup (MD5SUMS included).
- If a file ever does shrink: in Google Drive (web), right-click the file -> File information -> Manage versions (or
  "Manage versions" in the menu), download the version dated 22 Oct (kept 30 days), and either upload it as a new version
  there or send it to me; nothing needs deleting. The local backup above is a second copy.

## Notebook a42bb3b QUICK-tested (18:3x UTC): ready for Run 7
- tools/run_quick.py, CPU, 745 s, no errors. Against the QUICK run of the earlier notebook: all 16 controller files, dpc_curves.csv
  and dpc_diag.csv bit-identical; dpc_results.csv identical except the wall-time column train_s_per_seed; surrogates.csv now
  keeps all 8 model rows (the earlier run predates the ce89f6f merge fix), the 4 rows in common identical.

## R2 (done, commit 39b1939), R4 (draft, commit 562ba04)
- R2: docs/RESULTS_lambda_posthoc.md. Within LN MSE 3000, Lambda* AUC: GRU B = 0.01 1.00, GRU B = 0.1 0.81, MLP 0.36. Zero-FA
  intervals do not transfer. MLP traps occur with a tame gradient at the MLP oracle; linear false alarms have exploding gradients
  (1e27-1e34, one start 94-100 %) and no trap.
- R4: out/claims_lambda_draft.md: old -> new for CLAIMS.md (T1, T2, T4, T5, T6 -> T6/T7a/T8), NOTES_theory.md, theory.tex
  (summary, item 9, remark after Cor. 6), background note (4 places); companions: no hit; slides 17-19: no Lambda* text (deck read),
  3 proposed slide-fix entries (15-17) for slides 22-23 that Run 5 contradicts. NOT applied.

## R5 (protocol committed, computing) - protocol 51b4b07 + amendment 9e62f0c, script ec... (analysis/trap_predictor.py)
- Candidates C1 (warm-start expulsion on the true cost) and C2 (on the surrogate cost), 720 development runs; selection rule in
  prereg/PROTOCOL_trap_predictor.md, fixed before computing. Linear set running (~4.5 min per (family, repeat) under load);
  MLP chained after it; both GRU sets chained after the R3 arms job. Expected finish: several hours (GRU units are the slow part).
  The freeze commit will come before the Run 8 pre-registration.

## Q8 milestone (ii) (done) - tools/plant2_quick.py, out/plant2_quick
- All six families x 2 seeds ran end to end on the P1 plant (surrogates scaled 0.1: 300 / 1200 steps): Level 1 + ellipsoid kink
  metrics, mini LQR study, short DPC through surrogate and true plant, vector Lambda*. The LQR failure rates (50-100 %) are not
  meaningful with 10x shorter training; tools/plant2_lqr_sanity.py (one full 3000-step static-norm surrogate) checks the LQR code.
