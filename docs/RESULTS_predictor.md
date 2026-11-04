# R5: replacement trap predictor - result (2026-11-04)

Protocol: prereg/PROTOCOL_trap_predictor.md (51b4b07, amendment 9e62f0c). Code: analysis/trap_predictor.py (warm starts),
analysis/select_predictor.py (selection rule). Outcome: NO CANDIDATE IS ELIGIBLE. NOTHING IS FROZEN.

## What was tested
Two candidates, both "warm-start expulsion" from the architecture's own oracle controller (plain AdamW, lr 1e-3, 100 epochs
at N = 200, full batch of the repeat's 1000 training starts):
- C1: true-plant cost rises above 1.10 x its epoch-0 value at any logged epoch 10-100;
- C2: the same on the surrogate cost.
Reference B0 (not eligible by protocol): direct-gain Lambda* > 12.1.
Rule: a candidate is eligible only if its false-alarm rate is <= 5 % in EACH of the four development sets (GRU B = 0.01,
MLP, linear, GRU B = 0.1).

## Data actually computed
390 of the 720 development runs (CPU warm starts; the runs stopped when the session's context was reset and were not
restarted, since the outcome below can no longer change):
| set | runs done / all | families done |
|---|---|---|
| GRU B = 0.01 | 50 / 180 | LN MSE 3000 (30), LN MSE 12000 (20) |
| MLP | 110 / 180 | LN MSE 3000, MSE 12000, slope from 0 (30 each), two-stage (20) |
| linear | 180 / 180 | all |
| GRU B = 0.1 | 50 / 180 | LN MSE 3000 (30), LN MSE 12000 (20) |
All 20 + 40 + 0 + 7 traps of the full sets except 8 GRU B = 0.01 traps (slope from 0) are among the runs done.

## Result: the false-alarm rule each candidate fails, and by how much
| candidate | GRU B = 0.01 FA | MLP FA | linear FA | GRU B = 0.1 FA | fails the 5 % rule in |
|---|---|---|---|---|---|
| C1 | 1 / 38 (2.6 %) | 1 / 70 (1.4 %) | 0 / 180 (0 %) | 9 / 43 (20.9 %) | GRU B = 0.1 |
| C2 | 1 / 38 (2.6 %) | 1 / 70 (1.4 %) | 0 / 180 (0 %) | 9 / 43 (20.9 %) | GRU B = 0.1 |
| B0 (reference) | 0 / 38 | 0 / 70 | 12 / 180 (6.7 %) | 0 / 43 | linear (1.7 points over) |
- C1 and C2 fail at B = 0.1. The observed rate is 20.9 %, four times the limit. The failure is certain whatever the
  unfinished runs give: the full set has 173 untrapped runs, so even with no further false alarm the rate is at least
  9 / 173 = 5.2 % > 5 %.
- Hit rates (for the record; they do not matter once nothing is eligible): C1 12/12, 22/40, -, 6/7 (worst-case 0.55, the
  MLP); C2 10/12, 22/40, -, 5/7.
- The 9 B = 0.1 false alarms are healthy runs: from scratch, DPC reaches gaps of 0.0-1.8 %. The warm start drifts by 10-38 %
  (max ratio 1.11-1.38) and then DPC still finds the good controller from a random start. Five of the nine fire at the
  first check (epoch 10).

## Is there a principled justification beyond the development numbers?
I looked for one and do not find one:
- The trapped B = 0.1 runs mostly have much larger ratios (5.1-6.3 on seed 1, 2.2-3.7 on seed 2) than the false alarms
  (1.11-1.38). But one trap sits at 1.12 (seed 2 rep 0) and one at 1.06 (seed 9 rep 2). A larger ratio threshold would be
  chosen from these numbers, so it would be fitted to the development set, which the protocol forbids.
- Mechanistically, warm-start expulsion measures whether the good controller is stable under the optimiser. A trap from
  scratch also needs the optimiser to land in, and stay in, a bad basin. At B = 0.1 the closed loop is better damped
  (pole 0.505): an expelled controller drifts, but DPC from scratch still converges. So expulsion is necessary-looking,
  not sufficient. This explains the false alarms after the fact; it does not justify a new rule.
- The 1.10 threshold itself was a convention (taken from the warm-start harness), not derived.
Conclusion: no candidate has a principled justification beyond the development numbers. As you planned, the predictor
search stops here. Run 8 becomes a replication of the trap rates and of Part 1 (seeds 10-29), with no predictor test.

What may be said in the paper: Lambda* (direct gain) bounds the gradient at the good controller and separates the GRU traps
at B = 0.01 retrospectively. Neither it nor warm-start expulsion predicts traps across architectures and plants. Both
statements report a negative result honestly.

Files: out/predictor/warm_*.csv (per run), warm_*_curves.csv, selection.csv and selection_partial_390.txt (390 runs);
selection_partial.txt is the earlier 380-run snapshot (MLP FA 1/60 there).
