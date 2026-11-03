# Next Colab session (A100, float64) - exact settings

Upload `src/dpc_through_surrogates.ipynb` from this repository (rebuilt 3 Nov; it contains the new code: optimiser
baselines, plant-derived constants, surrogate cache guard, `PLANT` / `ROOT` / `SURROGATES_ONLY` settings). Do not use an
older copy. Runtime: A100, `DTYPE = "float64"`. Keep the Drive mount cell as it is (root `MyDrive/dpc_out`).

The settings live in the cell that starts with `QUICK  = False`. Only the lines listed change; everything else stays default.
After each run, zip the output folder named below (the notebook's last cell does this) and upload it into `data/` here.

Everything below may start now: the pre-registrations they need are committed (Task 3: `prereg/PREREG_opt.md`, commit
c41967c; Task 2a Part 1: `prereg/PREREG_B01.md`, commit 8ecc750). Runs 1-3 are short and should go first, because their
outputs are needed for the next two pre-registrations.

## Run 1 - Task 1 Stage A: MLP oracle (~10-20 min)
```
ARCH   = "mlp"
ONLY   = ["true plant (oracle)"]
REPS   = [0, 1, 2]
```
In the "Optimiser-side control" cell: `RUN_ROBUST = False`. In "Optimiser baselines": `RUN_OPT = False` (default).
Results go to `MyDrive/dpc_out_mlp`. With only the oracle selected, the notebook zips the folder right after the training
cell (`MyDrive/dpc_out_mlp_stageA.zip`) and stops on purpose with "stage A run finished" (the later cells need
DPC-through-surrogate runs). Download that zip. (Tested locally in QUICK mode, 3 Nov.)

## Run 2 - Task 1 Stage A: linear oracle (~10 min)
Same as Run 1 with `ARCH   = "linear"`. Results: `MyDrive/dpc_out_linear_stageA.zip`.

## Run 3 - Task 2a Stage A: surrogates and GRU oracle at B = 0.1 (~1-1.5 h: 60 surrogates incl. 20 with slope labels, then the oracle)
```
ARCH   = "gru"
ONLY   = ["true plant (oracle)"]
REPS   = [0, 1, 2]
PLANT  = L.Plant(A=1.03, B=0.1, eps_nl=0.0)
ROOT   = "dpc_out_B01"
SURROGATES_ONLY = True
```
`RUN_ROBUST = False`, `RUN_OPT = False`. The run stops on purpose after the training cell with
"stage A run finished" and writes `MyDrive/dpc_out_B01_stageA.zip`. Download that zip.
Safety: the notebook refuses a plant other than (1.03, 0.01) in `dpc_out`, and the surrogate cache refuses files trained on
another plant, so nothing can be mixed with the B = 0.01 study.

## Run 4 - Task 3 from scratch: optimiser baselines (~4-6 h; can be split over two sessions, it resumes)
```
ARCH   = "gru"
ONLY   = []
REPS   = [0, 1, 2]
```
"Optimiser-side control" cell: `RUN_ROBUST = True` and `ROBUST_REPS = [0, 1, 2]`.
"Optimiser baselines" cell: `RUN_OPT = True` (`OPT_REPS = [0, 1, 2]` default).
Root `MyDrive/dpc_out` (tested end to end in QUICK mode on a copy of a finished folder; `surrogates.csv` keeps the main rows. The new model names do not collide with existing files; the oracle controllers of repeats 0-2 are
reused, not retrained). Cost estimate: clip, unit and median-of-means need one backward pass per start (about 3-5x a plain
model); global clip and lr 1e-4 cost one plain model each; batch 100 about 0.2x; N <= 50 and N <= 100 well under one plain
model. A plain model is ~35 s per seed per repeat on the A100, so ~6 min per model and repeat set (30 runs), i.e. roughly
3 x 6 x 4 (clip, unit, mom) + 2 x 6 (global, lr) + 3 x 3 (batch, N <= 50, N <= 100) = ~95 min x the 3-5x factor uncertainty.
Download afterwards: `dpc_results.csv`, `dpc_diag.csv`, `dpc_curves.csv` and `controllers/` of `MyDrive/dpc_out`.

## After Runs 1-3 (local, me)
- `python analysis/make_prereg_arch.py --mlp data/dpc_out_mlp --linear data/dpc_out_linear` -> `prereg/PREREG_arch.md`,
  commit and record the hash.
- B = 0.1: LQR study, kink geometry and Lambda* of the B = 0.1 oracle -> `prereg/PREREG_B01.md` Part 2, commit.
Only then:

## Run 5 - Task 1 Stage B: MLP and linear DPC (~1.5 h + ~1 h)
`ARCH = "mlp"`, `ONLY = None`, `REPS = [0, 1, 2]`, `RUN_ROBUST = False`, `RUN_OPT = False`; then the same with
`ARCH = "linear"`. Download `MyDrive/dpc_out_mlp`, `MyDrive/dpc_out_linear`.

## Run 6 - Task 2a Stage B: DPC through the B = 0.1 surrogates (~2 h)
`PLANT = L.Plant(A=1.03, B=0.1, eps_nl=0.0)`, `ROOT = "dpc_out_B01"`, `SURROGATES_ONLY = False`, `ONLY = None`,
`REPS = [0, 1, 2]`, `ARCH = "gru"`, `RUN_ROBUST = False`, `RUN_OPT = False`. Download `MyDrive/dpc_out_B01`.
