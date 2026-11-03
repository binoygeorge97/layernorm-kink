# Pre-registration: optimiser baselines for the DPC trap (Task 3)

Date written: 2026-11-03
Commit hash of this file (fill after committing, then commit again): see the line "Registered at commit" at the end
Results that exist at the time of writing:
- the main GRU study (7 models x 10 seeds x 3 repeats, `MyDrive/dpc_out/dpc_results.csv`, `dpc_diag.csv`), including the
  plain LN MSE 3000 runs (13/30 failed, 12/30 trapped; seeds 1, 2, 8, 9 trapped in every repeat);
- warm starts at the good controller, plain AdamW and per-start clip 3, 100-300 epochs (`data/reference_results/warmstart*.csv`,
  figP10): plain expels the controller on LN MSE seeds 2, 8, 9 and slope-from-0 seeds 2, 9 (seed 1 kicked out and back),
  clip 3 keeps it on all of them;
- NO from-scratch run of any robust or optimiser setting (dpc_results.csv contains only the 7 main models), and no warm start
  with global clip, median of means, minibatches or lr 1e-4. The only contact with the new code paths before this file was
  committed: unit tests (`tools/test_opt_units.py`), a regression run of 3 short curriculum stages on seeds 0 and 2
  (`tools/regress_dpc.py`, 50 training epochs, not a test of any prediction), and a 5-epoch compile check of every warm-start
  setting on the healthy pair two-stage seed 2 (no broken pair).

## Question
Does a standard optimiser-side trick (global gradient clipping, median-of-means aggregation, minibatches, a smaller learning
rate, a shorter horizon) already prevent the DPC trap on the LN MSE 3000 surrogates, or is a per-trajectory bound (or a
fix of the model, two-stage training) needed?

## Design (fixed before results)
- Code: commit of this file; `src/dpc_core.py` (`ROBUST_MODELS`, `OPT_MODELS`), `src/dpc_through_surrogates.ipynb`
  (blocks "Optimiser-side control" and "Optimiser baselines"), `analysis/warmstart_harness.py`.
- From scratch (Colab A100, float64): DPC notebook, `ARCH = "gru"`, `DTYPE = "float64"`, `QUICK = False`, Drive root
  `MyDrive/dpc_out` (model names are new, so nothing is overwritten; the oracle controllers of repeats 0-2 are reused),
  `ONLY = []` (main models already done), `REPS = [0, 1, 2]`, `RUN_ROBUST = True` with `ROBUST_REPS = [0, 1, 2]`,
  `RUN_OPT = True` with `OPT_REPS = [0, 1, 2]`.
- Settings (one change each from the plain run; same surrogates, controller inits 20000 + s + 1000 rep, starts
  30000 + s + 1000 rep, curriculum 5/10/20/50/100/200 with 1000/2000/2000/1000/1000/2000 epochs, AdamW 1e-3):
  clip 3 (per-start gradient clipped to norm <= 3, then averaged), unit (per-start unit vectors averaged),
  global clip 3 (mean gradient clipped to norm <= 3), mom (median of the means of 10 contiguous groups of 100 starts),
  batch 100 (100 of the 1000 starts drawn without replacement each epoch, key 40000 + s + 1000 rep, fold_in epoch),
  lr 1e-4, N <= 50 (curriculum stops after the N = 50 stage: 6000 epochs), N <= 100 (stops after N = 100: 7000 epochs).
- Units: 10 seeds x 3 repeats = 30 runs per setting. Unit of replication for paired tests: the seed (median over repeats).
- Definitions (CLAUDE.md 1.4): gap = true-plant cost on the 1000 typical test starts / oracle of the same seed and repeat - 1;
  failed = gap > 10 %; trapped = failed AND (own controller on its surrogate) > 1.05 x (oracle controller on the same
  surrogate), test starts.
- A seed is RESCUED by a setting if it is not failed in at least 2 of its 3 repeats. The four target seeds are LN MSE
  seeds 1, 2, 8, 9 (failed in 3/3 repeats with the plain optimiser).
- Warm start (local CPU, `analysis/warmstart_harness.py`, 100 epochs at N = 200, start = repeat-0 oracle GRU, repeat-0
  training starts): broken pairs LN MSE seeds 1, 2, 8, 9 and slope-from-0 seeds 2, 9; healthy pairs LN MSE seeds 3, 7 and
  two-stage seeds 2, 9. A pair is EXPELLED if its true-plant cost on the training starts exceeds 1.10 x its epoch-0 value at
  any logged epoch from 10 to 100 (every 5 epochs); it STAYS otherwise. (Epochs 0-5 are excluded because healthy pairs show
  an early transient of up to +2 % in the reference run.)
- Thresholds (not to be changed after seeing results): 10 % gap, 1.05 trap ratio, 2-of-3 rescue rule, 1.10 expulsion.

## Predictions
| id | prediction | basis | how it is scored |
|---|---|---|---|
| P1 | per-start clip 3 from scratch rescues at least 3 of the 4 target seeds | clipped warm start stays on every broken pair; one start carries 61-100 % of the gradient | rescue rule, count over seeds 1, 2, 8, 9 |
| P2a | global clip 3 from scratch does NOT rescue: at most 1 of the 4 target seeds rescued | the direction of the mean gradient is still set by the exploding start; clipping only bounds the step | rescue rule |
| P2b | global clip 3 warm start: expelled on at least 4 of the 6 broken pairs | as P2a (each step still points in a random direction) | expulsion rule |
| P3a | median of means from scratch rescues at least 3 of the 4 target seeds | an exploding start corrupts one group mean; the median ignores it | rescue rule |
| P3b | median of means warm start: stays on at least 5 of the 6 broken pairs | as P3a | expulsion rule |
| P4 | lr 1e-4 from scratch does NOT rescue: at most 1 of the 4 target seeds rescued | the direction decorrelates at 1e-4 relative weight changes; a smaller step does not fix the direction | rescue rule |
| P5a | N <= 50 avoids the trap: at most 3 of 30 runs trapped (plain: 12/30) | the landing amplification grows with the horizon; traps appeared at horizons 100-200 | trapped count |
| P5b | N <= 50 gives a worse true-plant controller than the plain optimiser on two-stage surrogates: median typical-start gap over the 30 runs larger than the two-stage plain median over its 30 runs | a 50-step horizon cannot see the cost of slow convergence over 200 steps | medians, both numbers reported |
| P6 | healthy warm-start pairs stay under every setting (all 4 healthy pairs, all 7 settings) | healthy gradients are tame | expulsion rule |
| U1 | minibatch 100: no prediction (uncertain) | a minibatch excludes the exploding start in ~90 % of epochs, but includes it in the rest | reported |
| U2 | unit vote, N <= 100, lr 1e-4 warm start: no prediction | - | reported |

Reproduction checks (known outcomes, not predictions): plain warm start expels LN MSE seeds 2, 8, 9 and slope-from-0 seeds 2, 9;
clip 3 warm start keeps all six broken pairs. Exact trajectories of broken pairs are not expected to reproduce (gradient
norms of 1e10+ differ between machines, CLAUDE.md pitfall 9); healthy pairs should reproduce to many digits.

## Analysis plan
- Table (docs/RESULTS_opt.md): rows = plain, clip 3, unit, global 3, mom, batch 100, lr 1e-4, N <= 50, N <= 100, and the
  two-stage surrogate with the plain optimiser as the reference row; columns = failed / 30, trapped / 30, median gap, gap of
  the best seed (per-seed median over repeats, minimum over seeds), wall time per seed.
- Per target seed: rescued yes / no for every setting.
- Paired comparison with the plain LN MSE runs over the 10 seeds (Wilcoxon signed-rank on the per-seed median gap).
- Warm start: figure of the true-plant cost against epoch, one panel per pair, one line per setting; table of expelled /
  stays.
- Every prediction gets held / failed / partly, with the numbers.

## Outcomes (fill AFTER results; do not edit anything above this line)
| id | outcome (held / failed / partly) | numbers |
|---|---|---|

Registered at commit c41967c7efc27fa9778f575334c01b0d323502e9 (2026-11-03), before any optimiser-baseline result existed.
