# Data manifest (snapshot 2026-11-03)

## data/dpc_out (PARTIAL copy of MyDrive/dpc_out)
Replace this whole folder with the user's `MyDrive/dpc_out` (download from Google Drive) before Task 0 step 2.

Present here:
- `surrogates/` - one npz per family, all 10 seeds stacked: `MSE__3000.npz` (LN MSE 3000), `MSE__12000.npz`,
  `J_x___J_u__3000.npz` (LN slope from step 0), `J_x___J_u__two_stage.npz` (LN two-stage), `no_norm__MSE.npz`,
  `static_norm__MSE.npz`. Plant (A, B, sigma) = (1.03, 0.01, 0.01); these files do not store it.
- `controllers/` - GRU DPC controllers, all 10 seeds stacked per file:
  `true_plant__oracle` reps 0-2; `LN__MSE__3000` reps 0-2; `LN__J_x___J_u__3000` reps 0-2;
  `LN__MSE__12000`, `LN__two_stage`, `static_norm__MSE` rep 0 only. Repeat r > 0 adds `__rep<r>`.
- `track_controllers/` - 3 tracking-DPC files (phase notebook).

Missing here (on Drive): `no_norm__MSE` controllers; reps 1-2 of MSE 12000 / two-stage / static norm / no norm;
`dpc_results.csv`, `dpc_diag.csv`, `dpc_curves.csv`, `surrogates.csv`, `phase_lqr.csv`, `track_results.csv`, `trajectories/`,
`labels/` (slope labels). `theory/t2_test.py` needs `dpc_results.csv` and every controller file.

Other Drive folders: `MyDrive/level1_out` (Level 1 surrogate study), `MyDrive/dpc_out_quick*` (smoke tests, ignore).
New experiments write to new folders (`dpc_out_mlp`, `dpc_out_linear`, `dpc_out_B01`, ...).

## data/reference_results (numbers to reproduce or reuse)
| file | produced by | content |
|---|---|---|
| lambda_star_180runs.csv | theory/t2_test.py (S_seg_max) + dpc_diag labels | Lambda* and trap label for 6 families x 10 seeds x 3 reps |
| theory/t2_swap.csv | theory/t2_swap.py | LN MSE 3000 Lambda*: surrogate slopes / true J_u / true J_x (rounded 3 dp) |
| theory/t2_memoryless_Sseg.csv | theory/t2_memoryless.py part A | Lambda* of the linear policy at K* = 10.2875, all 180 (rounded 3 dp) |
| gradnorm.csv, gradnorm_line*.csv | story/story_data.py | gradient norms at the own / reference controller and along the straight line |
| interp.csv | phase_core.cost_interpolation | surrogate and true cost along the straight line (21 points) |
| warmstart.csv, warmstart_clip.csv | story/story_data.py | warm start at the good controller, plain and per-start clip 3 |
| path_colours.csv, stability.csv | phase_core section 7 | slope colours along the landing paths; local stability |
| figP9_trap_landscape.png, figP10_warmstart.png | phase_report | the two figures behind slides 18-19 |

Full theory outputs (all columns, including `t2_runs_other_families.csv`, `t2_neighbourhood.csv`, `t2_lambda_all.csv`) are in
the claude.ai project under `claude/theory/results/`; Task 0 regenerates them.

## companion/lqr_study.csv
11 880 rows: 6 families x 10 seeds x 11 setpoints x 9 starts x 2 controllers (fixed, rescheduled), from
companion/run_lqr_study.py. Rates exclude starts within 0.5 of the setpoint (880 runs per family per controller).

- data/run6_B01/controllers_check/ (not in git): true_plant__oracle{,__rep1,__rep2}.npz copied read-only from MyDrive/dpc_out_B01/controllers on 2026-11-04 for the oracle parameter-hash check (docs/RESULTS_B01.md, tools/hash_params.py).
