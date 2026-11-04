# RMSNorm surrogates (Q7)

Written 2026-11-04.
- Theory note: docs/theory/rmsnorm/rmsnorm_kink.pdf (b349d6d).
- Option: a9b34f2; regressions passed.
- Pre-registration: prereg/PREREG_rms.md (Part 1 e221377, Part 2 608dc88).
- Surrogates: trained locally, seeds 0-9, data/dpc_out_rms (34 min of CPU).
- LQR study: companion/run_lqr_study.py with LK_FAMILIES=rms.
- Scoring: analysis/rms_report.py; out/rms/summary.csv.

| family | median r*_x | median r*_u | median peak abs(J_u_hat/B - 1) | fixed LQR failed / catastrophic | rescheduled failed / catastrophic |
|---|---|---|---|---|---|
| RMS MSE 3000 | 0.233 | 0.272 | 16.4 | 29.5 / 15.6 % | 22.2 / 4.1 % |
| RMS two-stage | 0.248 | 0.291 | 1.43 | 4.2 / 1.0 % | 3.0 / 0.1 % |
| LN MSE 3000 (reference) | 0.261 | 0.336 | 18.7 | 26.5 / 10.8 % | 20.5 / 3.4 % |
| LN two-stage (reference) | 0.277 | 0.371 | 1.19 | 5.7 / 0.9 % | 3.5 / 0.1 % |

Outcomes:
- M1 failed narrowly: one centre at 0.109.
- M2, M3, M4 held.

Without centring, the kink and its LQR damage are the same as with LayerNorm, as the note predicts (same structure, and
D_RMS >= D_LN for equal weights). Two-stage training removes the damage again.

Lambda* of the B = 0.01 GRU oracles through the RMS surrogates flags 9 of 30 RMS MSE 3000 runs: seeds 1, 2, 8 in every
repeat, the same narrow-kink seeds as for LN. It flags none of the two-stage runs. These are registered for Run 9.

Run 9 (Colab, after Run 4 has finished; Drive folder MyDrive/dpc_out, new file names only; ~45 min on an A100). Notebook
cd6eb9e, QUICK test of the RMS path pending (see QUEUE_STATUS):
```
QUICK = False; ARCH = "gru"; SEEDS = list(range(10)); DTYPE = "float64"
PLANT = L.Plant(A=1.03, B=0.01, eps_nl=0.0); ROOT = None; LABEL_DIR = None
SURROGATES_ONLY = False; ONLY = ["RMSNorm, MSE (3000)", "RMSNorm, two-stage"]; REPS = [0, 1, 2]
RUN_ROBUST = False; RUN_OPT = False
```
The run trains the two RMS surrogate families on Colab (new files RMS_MSE__3000.npz and RMS_J_x___J_u__two_stage.npz in
surrogates/) and reuses the oracles of repeats 0-2. After it, I will hash-compare the surrogates with the local ones and score
Part 2.
