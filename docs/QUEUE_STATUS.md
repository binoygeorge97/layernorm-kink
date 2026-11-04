# Queue status (4 Nov work queue). Newest entries at the bottom.

## Q0 (partly done; A1 pending) - 2026-11-04 17:35 UTC
- A2 timing addendum and oracle hashes: d8fffc2 (all nine hashes match). A3 Run 5 row checks and spot check: f1e7a72.
- B3, the lr 1e-4 convergence rule for Run 4: 84ecb3a (17:19 UTC). Healthy LN MSE seeds 0, 3, 4, 6, 7; each must reach gap <= 10 % in >= 2 of 3 repeats, else P4 "not testable: not converged". Run 4 may start.
- A1 (augmented Lambda* predictions for Run 6): computation running (3 CPU processes, ETA ~17:55 UTC). Will be labelled "committed after Run 6 began writing, before any Run 6 file was read"; first Run 6 file in dpc_out_B01: trajectories/LN__MSE__3000.npz, created 2026-11-04T16:50:14.902Z.

## Q1 (done) - 17:35 UTC, commit 180ec7b
- analysis/b01_report.py frozen: LQR rates, setpoint law, path colour, DPC per family, Lambda scatter (direct, aug), confusion matrices and outcomes R1, R2, aug rule, CLAUDE.md decision rule; non-registered model rows excluded and listed.
- End-to-end test on the B = 0.01 data (out/test_b01_report_on_B001): reproduces 26.5 / 10.8 % fixed-LQR LN MSE, path colour 1.0/1.0/0/0 and 1.1/9.0/27/40 %, 12 + 8 traps, direct 17/20 hits 0 FA, aug 15/20 0 FA.
- Scoring convention fixed in the script before results (undefined case): runs predicted trapped but none trapped -> "held (false alarms only)" if false-alarm rate <= 5 %. Setpoint-law fit in the script (log-log over all LN fixed-LQR setpoints) gives 0.39 d^-0.88 at B = 0.01, not the 0.49 d^-0.98 quoted in CLAUDE.md (different subset/fit); reported as is.
