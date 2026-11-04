# B2: what happens after the first step of the warm start (exploratory, not pre-registered; numbers only)

Written 2026-11-04. No committed claim is changed here. Setting throughout: AdamW lr 1e-3 started AT the repeat-0 oracle GRU,
1000 repeat-0 training starts, N = 200 (analysis/warmstart_harness.py); 6 broken pairs (LN MSE 3000 seeds 1, 2, 8, 9;
slope-from-0 seeds 2, 9) and 4 healthy pairs (LN MSE 3000 seeds 3, 7; two-stage seeds 2, 9).

## (i) Which cost keeps rising, and what "tame" and "agree" meant
Costs: "true cost" = dpc_core.true_lqr_cost of the current controller on the 1000 training starts through the true plant;
"surrogate cost" = the same through the surrogate the controller is trained on (the DPC training loss is this up to the
terminal-weight convention).

Both rise, together. Plain warm start (out/opt/warmstart_plain.csv), epoch 0 -> 20 -> 100:
| pair | true cost | surrogate cost |
|---|---|---|
| LN MSE s1 | 2.556 -> 3.842 -> 2.602 | 2.606 -> 3.912 -> 2.630 |
| LN MSE s2 | 2.638 -> 2.837 -> 2.891 | 2.650 -> 2.759 -> 2.792 |
| LN MSE s8 | 2.609 -> 4.694 -> 5.250 | 2.670 -> 4.546 -> 5.131 |
| LN MSE s9 | 2.639 -> 4.101 -> 4.697 | 2.620 -> 3.655 -> 4.172 |
| slope s2 | 2.638 -> 5.071 -> 5.870 | 2.611 -> 5.078 -> 5.881 |
| slope s9 | 2.639 -> 2.843 -> 4.594 | 2.518 -> 2.751 -> 4.474 |
| healthy (4 pairs), change epoch 0 -> 100 | +0.0 % to +3.8 % | -1.1 % to +2.2 % |
The same holds in the median-of-means run (out/opt/mom_diagnose.csv), e.g. LN MSE s2 true 2.638 -> 3.508, surrogate
2.663 (epoch 1) -> 3.319 (epoch 100). On LN MSE s1 the plain run rises to epoch 20 and falls back by epoch 100 (+1.8 % true,
+0.9 % surrogate). So the optimiser climbs the loss it is minimising; this is not a model-mismatch effect.

Definitions behind "tame, agreeing gradients" in docs/RESULTS_opt.md (item 3), now stated exactly; they refer to the
median-of-means warm start (mom_diagnose.csv, every epoch, epochs 2-100):
- tame at an epoch = no per-start gradient norm above 1e3 (n_starts_gt1e3 = 0). Holds on 92, 99, 97 of 99 epochs for LN MSE
  s1, s2, s8 and 99 of 99 for slope s2, but only 7 of 99 for LN MSE s9 and 44 of 99 for slope s9.
- agree at an epoch = cosine >= 0.99 between the median-of-means direction and (a) the plain mean gradient and (b) the
  per-start clip-3 aggregate, all evaluated at the same controller. On tame epochs: cos(mom, mean) median 1.000 on 5 pairs
  (0.957 on slope s9, minimum -0.92 there); cos(mom, clip 3) minimum 0.974 on the four LN MSE pairs and slope s2.
Correction to RESULTS_opt item 3: "all 1000 starts have norms 3-40" is too strong. The LARGEST per-start norm on tame epochs
has median 15-39 on LN MSE s1, s2, s8 and slope s2 (21 on slope s2), but reaches 290-997 on some tame epochs, and on LN MSE s9
and slope s9 the starts keep exploding on most epochs. "Tame and agreeing after the first step" holds for 4 of the 6 broken
pairs (LN MSE s1, s2, s8, slope s2), not all 6.
