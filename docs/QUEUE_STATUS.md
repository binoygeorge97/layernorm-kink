# Queue status (4 Nov work queue). Newest entries at the bottom.

## Q0 (partly done; A1 pending) - 2026-11-04 17:35 UTC
- A2 timing addendum and oracle hashes: d8fffc2 (all nine hashes match). A3 Run 5 row checks and spot check: f1e7a72.
- B3, the lr 1e-4 convergence rule for Run 4: 84ecb3a (17:19 UTC). Healthy LN MSE seeds 0, 3, 4, 6, 7; each must reach gap <= 10 % in >= 2 of 3 repeats, else P4 "not testable: not converged". Run 4 may start.
- A1 (augmented Lambda* predictions for Run 6): computation running (3 CPU processes, ETA ~17:55 UTC). Will be labelled "committed after Run 6 began writing, before any Run 6 file was read"; first Run 6 file in dpc_out_B01: trajectories/LN__MSE__3000.npz, created 2026-11-04T16:50:14.902Z.

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
