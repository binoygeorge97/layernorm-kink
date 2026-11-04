# Pre-registration: augmented-state landing amplification Lambda*_aug for Run 6 (B = 0.1), SECONDARY statistic

The primary, direct-gain predictions for Run 6 are PREREG_B01.md Part 2 (77d8455) and stay as they are.
Definition: docs/theory/exact_lambda_aug/exact_lambda_aug.tex (commit 156e71b, before any value was computed).

## Part 1 (2026-11-04): threshold, fixed from B = 0.01 data only, BEFORE any B = 0.1 value of Lambda*_aug is computed
Retrospective values on the 180 GRU runs at B = 0.01 (theory/t2_aug.py, out/theory/t2_aug_B001.csv):
- LN MSE 3000 separates: largest non-trapped 11.21 (seed 0 rep 1), smallest trapped 13.01 (seed 8 rep 2).
- Threshold by the rule of the definition (geometric midpoint on LN MSE 3000): sqrt(11.21 x 13.01) = 12.08.
- Over all 180 runs: no false alarm for any threshold above 11.21; every threshold in (11.21, 13.01) flags 15 of the 20 traps
  (misses: slope-from-0 s4 r1 2.27, s9 r1 3.84, s3 r2 4.59, s9 r2 7.55, s9 r0 9.38); 0 of 160 false alarms.
  The same anchoring with the scalar multiplier flags 17 of 20 on its zero-false-alarm interval (8.23, 16.10]; the memory, not
  the anchoring, costs the two extra misses.

Run-6 rule (fixed here): a B = 0.1 run is predicted trapped iff Lambda*_aug > 12.08, computed from the B = 0.1 GRU oracle of the
same repeat (theory/t2_aug.py, T2_PLANT = 1.03,0.1, T2_DATA = data/dpc_out_B01). Scored like R1 of PREREG_B01 (hit rate >= 0.75
and false-alarm rate <= 0.05; if no run is predicted trapped, held if at most 1 run is trapped).

## Part 2: run-level predictions (appended after computing the B = 0.1 values, before Run 6)

Label: committed after Run 6 began writing, before any Run 6 file was read.
Written and committed at about 2026-11-04T17:56:34Z (exact commit time in git and docs/RESULTS_LOG.md). First Run 6 file in
MyDrive/dpc_out_B01 (Drive metadata only, no content opened): trajectories/LN__MSE__3000.npz, created / modified
2026-11-04T16:50:14.902Z (10:50:14 CT). From dpc_out_B01 only the Run 3 oracle controllers (repeats 0-2) and the six surrogate files
(fetched 3 Nov) were used; no dpc_results.csv, dpc_diag.csv or non-oracle controller was fetched or read.

Values: out/B01/theory/t2_aug_B01.csv (theory/t2_aug.py, T2_PLANT = 1.03,0.1, T2_DATA = data/dpc_out_B01, the B = 0.1 GRU
oracle repeats 0-2 and the six B = 0.1 surrogates fetched on 3 Nov; nothing else from dpc_out_B01 was fetched or read).
Rule (Part 1): predicted trapped iff Lambda*_aug > 12.08. Predicted trapped: 0 of 180 runs.

| family | runs | predicted trapped | max Lambda*_aug | max anchored scalar |
|---|---|---|---|---|
| LN MSE 3000 | 30 | 0 | 9.66 | 13.95 |
| LN MSE 12000 | 30 | 0 | 1.91 | 0.41 |
| LN slope from 0 | 30 | 0 | 4.64 | 0.42 |
| LN two-stage | 30 | 0 | 1.05 | 0.00 |
| no norm | 30 | 0 | 1.05 | 0.00 |
| static norm | 30 | 0 | 1.05 | 0.00 |

Runs predicted trapped (Lambda*_aug):
- none

Five largest values: LN MSE 3000 s7 r0 9.66; LN MSE 3000 s1 r2 9.12; LN MSE 3000 s7 r1 8.18; LN MSE 3000 s2 r2 7.21; LN MSE 3000 s2 r1 6.60.
Run-level predictions: prereg/PREREG_B01_aug_predicted.csv. For comparison only (not a prediction of this file): the primary
direct-gain rule (PREREG_B01 Part 2, 77d8455) predicts 1 of 180.

Part 2 committed at 91238b4274df2c6880d1c497be2ab17f4743ceb9 (2026-11-04T17:56:34Z), after Run 6 began writing, before any Run 6 file was read.

## Outcome (filled 2026-11-04 from Run 6, frozen analysis/b01_report.py 180ec7b)
Failed: 0 predicted, 7 trapped (rule: held only if at most 1 trapped). Lambda*_aug of the seven trapped runs 2.30 to 9.12.
