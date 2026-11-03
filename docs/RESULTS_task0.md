# Task 0 - environment, data, paths, reproduction (2026-11-03)

## Environment
Python 3.13.18 venv (`.venv`, `pip install -r requirements.txt`): jax 0.12.0.1, optax 0.2.8, numpy 2.5.4, scipy 1.18.1,
pandas 3.1.0, matplotlib 3.11.2 (the versions of the analysis sandbox). CPU only (4 cores), float64 everywhere.
LaTeX: TeX Live from Ubuntu (latexmk, pdflatex). Notebooks executed headless with nbclient (`tools/run_quick.py`).

## Data
The bundled `data/dpc_out` was partial. The missing files were read from `MyDrive/dpc_out` with the Google Drive connector
(read-only; nothing on the Drive was changed): `dpc_results.csv`, `dpc_diag.csv`, `surrogates.csv`, and the 9 missing
controller files (no-norm reps 0-2; reps 1-2 of MSE 12000, two-stage, static norm). Not downloaded (not needed for any task
so far): `dpc_curves.csv`, `phase_lqr.csv`, `track_*`, `trajectories/`, `labels/`, `figs/`, `tables/`.
`tools/check_data.py`: ALL CHECKS PASSED - 6 surrogate files with seeds 0-9 (no plant key = (1.03, 0.01, 0.01)); 7 models x 3
repeats of controller files, seeds 0-9; `dpc_results.csv` 210 rows (7 x 10 x 3), `dpc_diag.csv` 180 rows, no duplicates.

## Paths
`paths.py` at the root (REPO, SRC, DATA = $LK_DATA or data/dpc_out, OUT = $LK_OUT or out/, REF). All 14 scripts that
hard-coded `/home/claude/...` now import it (story/, companion/, companion/background/, theory/); only path literals changed.
Re-runs write under `out/`, never into the repository's result files (run_lqr_study.py used to write `lqr_study.csv` into
the current directory; it now writes `out/companion/lqr_study.csv`).

## Reproduction checks (acceptance list of CLAUDE.md 3.4)
| # | check | expected | obtained |
|---|---|---|---|
| 1 | theory/verify_thm1.py | identity <= 1e-14, singular values 1.7e-6, areas exact, block Jacobian 1e-9, trained 2e-15 | identity <= 8.2e-15, singular values 1.7e-6, areas exact (12.566..., 100.53...), block Jacobian 1.0e-9, trained surrogates 2.2e-15 |
| 2a | t2_test.py, all 6 families: Lambda* per run | = lambda_star_180runs.csv (3 digits) | 180 / 180 equal at 3 decimals; trap labels identical (20 traps) |
| 2b | threshold, hits, false alarms | 12.11, 17/20, 0/160 | 12.11, 17/20, 0/160; misses slope-from-0 s4 rep1 (2.49), s9 rep1 (3.37), s3 rep2 (6.45) |
| 2c | t2_memoryless.py A: memoryless Lambda* | max 8.75, = t2_memoryless_Sseg.csv | 180/180 equal at 3 decimals; max 8.746 (LN MSE 3000 seed 2 rep 0); sum log|m| = log|dx_N/dx_0| to 3.2e-8 |
| 2d | t2_swap.py | = t2_swap.csv; 9 of 12 LN MSE traps below 12.1 with J_u_hat -> B | equal at 3 decimals (all 30 runs, 3 columns); 9 of 12; 0 of 12 with J_x_hat -> A |
| 2e | lambda_all.py, gradient bound | Lambda_all - Lambda* <= 3.3 (median 1.9); grad <= 4.7 e^Lambda*, <= 0.61 e^Lambda_all | <= 3.32 (median 1.88); 4.63; 0.61 |
| 2f | t2_memoryless.py B (neighbourhood) | trapped seeds up to 5e43 | max 4.8e43 (LN MSE seed 2 rep 2) |
| 2g | make_t2_tables.py, make_t2_fig.py | tables, figure T1 | `out/theory/tab_*.tex`, `out/theory/figures/t1_theorem2_test.png` |
| 3 | companion/run_lqr_study.py | = companion/lqr_study.csv (< 1e-9); 26.5 / 10.8 % | identical: max abs difference 0 over all 11 880 rows and 20 columns; 26.5 / 10.8 % |
| 4 | both notebooks build; DPC notebook QUICK on CPU | runs end to end | see "QUICK run" below |

Not reproducible beyond the order of magnitude, as the handout warns (pitfall 9): gradient norms at the reference controller.
Re-run vs earlier numbers: LN MSE seed 9 rep 0 1.9e11 (earlier 1.2e14 and 9e11), seed 2 3.9e35 (earlier 1.1e35), seed 8
9.2e13 (3.1e13). Every other number above is reproduced to the digits stored.

Also reproduced on the way (Task 4): fixed-LQR path-colour bins 1.0 / 1.0 / 0 / 0 % (n = 2014, 204, 46, 24) and rescheduled
1.1 / 9.0 / 27 / 40 % (n = 2083, 111, 44, 50); per-setpoint fixed-LQR failure rates of fig P3; the LQR swap rates of
`story/swap_lqr_rates.csv` (all 20 numbers).

## QUICK run of the DPC notebook (CPU)
Both notebooks build (`python src/build_dpc_notebook.py`, `python src/build_phase_notebook.py`). The DPC notebook ran end to
end headless on CPU in QUICK mode (2 seeds, short surrogates, 120 controller epochs, repeats 0-1, robust block on):
the handoff version (commit 38de1d6) in 678 s, the current version in 611 s (other jobs were running), no cell errors.

Regression (default behaviour bit-identical, ground rule 4): every output of the two QUICK runs was compared.
Identical: all 16 controller files, all 16 trajectory files, `dpc_results.csv`, `dpc_diag.csv`, `dpc_curves.csv`,
`surrogates.csv`, `tables/*.csv`, all surrogate weights and Level 1 metrics. The only differences are the intended new keys
(`plant` in the surrogate files, `data_fp` in the label files, Task 2a cache guards) and the wall-clock fields
(`train_s`, `train_s_per_seed`). Also identical: the LQR study (11 880 rows, above), `tools/regress_dpc.py` (GRU and MLP,
mean / clip / unit aggregation, old vs new dpc_core) and `tools/regress_plant.py` (lqr_saturated, lqr_screen, slope maps).
