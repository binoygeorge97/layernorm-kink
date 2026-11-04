# Addendum 2026-11-04: robustness of Lambda* to a 1e-12 perturbation of the training starts (before any Run 5 / Run 6 result)

Written and committed before any result file of Run 5 (MLP, linear DPC through the B = 0.01 surrogates) or Run 6 (GRU DPC
through the B = 0.1 surrogates) existed locally or on the Drive. Concerns PREREG_arch (f945a5c, 322e353), PREREG_B01 (77d8455)
and the retrospective GRU set; no prediction, threshold or definition is changed.

Method (analysis/lambda_robust.py): every run with Lambda* > 8 in the four sets (GRU oracle, B = 0.01: 18 runs; MLP oracle: none;
linear oracle: 15; GRU oracle, B = 0.1: 4). The 1000 training starts are perturbed by 1e-12 N(0, 1) with key
fold_in(PRNGKey(50000 + s + 1000 rep), draw), draws 0-4 (new key offset 50000). Lambda* recomputed as registered; for the
maximising start, the landing step and the first and last step of the maximising stretch. Recomputed unperturbed values equal
the registered ones exactly in all 37 runs. Outputs out/theory/lambda_robust_{gru,linear,gru_B01}.csv.

Runs whose spread crosses 12.1: NONE.

Details. Runs close to the threshold do not move at all (spread 0.00): B = 0.1 LN MSE s2 r2 14.75 (stretch steps 6-199),
s2 r1 11.45, s7 r0 10.73, s1 r2 10.12; linear slope-from-0 s2 r0-r2 10.09-11.04; GRU LN MSE s0 r1 8.61, s1 r0 17.03, s1 r1
18.81. Runs whose maximising stretch reaches the end of the horizon on a chaotic loop move by up to 23 units (largest: GRU
slope-from-0 s2 r0 72.1 -> 44.7..67.4; linear LN MSE s2 r2 44.4 -> 26.2..45.9), all far above 12.1. Smallest perturbed value
of any registered trap prediction: 26.2 (linear LN MSE s2 r2).
