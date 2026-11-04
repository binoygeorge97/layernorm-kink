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
