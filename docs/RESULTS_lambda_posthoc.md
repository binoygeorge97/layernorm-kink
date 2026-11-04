# R2: Run 5 summary and a post-hoc look at Lambda* and the oracle gradients (POST HOC throughout)

Written 2026-11-04. Nothing here changes a pre-registered outcome. Scripts: analysis/lambda_posthoc.py
(out/theory/lambda_posthoc.csv), theory/arch_gradients.py (out/theory/arch_gradients_{mlp,linear}.csv).

## (a) Run 5: failed / trapped of 30 (registered trap definition), robust rows excluded
| architecture | family | failed | trapped | trapped runs |
|---|---|---|---|---|
| mlp | LN MSE 3000 | 27 | 27 | s0 r0, s0 r1, s0 r2, s1 r0, s1 r1, s1 r2, s2 r0, s2 r1, s2 r2, s3 r0, s3 r1, s3 r2, s5 r0, s5 r1, s5 r2, s6 r0, s6 r1, s6 r2, s7 r0, s7 r1, s7 r2, s8 r0, s8 r1, s8 r2, s9 r0, s9 r1, s9 r2 |
| mlp | LN MSE 12000 | 1 | 1 | s1 r2 |
| mlp | LN slope from 0 | 12 | 12 | s2 r1, s2 r2, s3 r0, s3 r1, s3 r2, s4 r0, s4 r2, s8 r1, s8 r2, s9 r0, s9 r1, s9 r2 |
| mlp | LN two-stage | 0 | 0 | none |
| mlp | no norm | 0 | 0 | none |
| mlp | static norm | 0 | 0 | none |
| linear | LN MSE 3000 | 0 | 0 | none |
| linear | LN MSE 12000 | 0 | 0 | none |
| linear | LN slope from 0 | 0 | 0 | none |
| linear | LN two-stage | 0 | 0 | none |
| linear | no norm | 0 | 0 | none |
| linear | static norm | 0 | 0 | none |
No MLP run of two-stage, no norm or static norm is trapped (0 of 90), and none failed. Every failed MLP run is trapped.

## (b) Lambda* as a ranking and as a thresholded predictor (post hoc)
| test | runs | trapped | ROC AUC, all 180 | ROC AUC within LN MSE 3000 | zero-false-alarm thresholds | hits there |
|---|---|---|---|---|---|---|
| GRU B = 0.01 (retrospective, reference) | 180 | 20 | 0.99 | 1.00 (12 of 30 trapped) | (8.61, 158.1] | 17 / 20 |
| MLP (prospective, f945a5c) | 180 | 40 | 0.92 | 0.36 (27 of 30 trapped) | (5.97, 7.36] | 1 / 40 |
| linear (prospective, 322e353) | 180 | 0 | not defined (no trap) | - | any threshold > 93.4 (all 180 below) | - |
| GRU B = 0.1, direct (prospective, 77d8455) | 180 | 7 | 0.97 | 0.81 (7 of 30) | (10.73, 14.75] | 2 / 7 |
| GRU B = 0.1, augmented (prospective, 91238b4) | 180 | 7 | 0.97 | 0.81 | > 9.66 (no trap above) | 0 / 7 |
Reading: the high all-run AUCs are mostly the family split (kink-free and slope-trained families have Lambda* near 0 and no
traps). Within LN MSE 3000, where the traps are, Lambda* ranks GRU traps perfectly at B = 0.01, moderately at B = 0.1
(0.81), and not at all for the MLP (0.36, worse than chance). No single threshold transfers: the zero-false-alarm interval
moves from (8.6, 158] (GRU) to (6.0, 7.4] (MLP) and (10.7, 14.8] (GRU B = 0.1), and for the linear controller every value
up to 93 is a non-trap.

## (c) MLP: the gradient at the MLP oracle through the surrogate (N = 200, 1000 training starts)
| family | runs | batch gradient norm, median (max) | largest start norm | top-start share of the squared norm, median (max) | starts > 1e3 | Lambda*, median |
|---|---|---|---|---|---|---|
| LN MSE 3000, trapped | 27 | 1.75 (9.8) | 75 | 0.004 (0.79) | 0 | 3.1 |
| LN MSE 3000, healthy | 3 | 1.82 (3.4) | 20 | 0.004 (0.008) | 0 | 4.3 |
| slope from 0, trapped | 12 | 1.17 (2.7) | 13 | 0.003 (0.006) | 0 | 2.5 |
| slope from 0, healthy | 18 | 1.74 (6.8) | 18 | 0.002 (0.10) | 0 | 0.6 |
| two-stage, healthy | 30 | 0.95 (2.6) | 6.5 | 0.002 (0.005) | 0 | 0 |
Answer: no. At the MLP oracle the gradient is tame on the trapped runs: same size as on healthy runs, spread over the
starts (top start 0.4 % of the squared norm), no exploding start. The GRU traps came with 1e3-1e35 gradients carried by one
start; the MLP traps do not. Something else traps the MLP (not tested here: whether the MLP from scratch ever approaches the
oracle's region, i.e. a basin / local-minimum effect rather than expulsion).

## (d) Linear: the same numbers
| family | runs | batch gradient norm, median (max) | largest start norm | top-start share, median (max) | starts > 1e3 (sum) | Lambda*, median |
|---|---|---|---|---|---|---|
| LN MSE 3000, Lambda* > 12.1 (seeds 1, 2, 8, 9), not trapped | 12 | 7.8e27 (2.5e34) | 1.4e37 | 0.94 (1.00) | 3601 | 79 |
| LN MSE 3000, other | 18 | 0.13 (0.35) | 1.1 | 0.004 (0.014) | 0 | 2.5 |
| slope from 0 | 30 | 0.05 (0.46) | 4.5 | 0.003 (0.35) | 0 | 1.7 |
| two-stage | 30 | 0.04 (0.14) | 0.35 | 0.002 (0.004) | 0 | 0 |
Answer: yes. On the 12 false alarms the gradient at the linear oracle explodes exactly as for the trapped GRU (1e27-1e34,
one start carries 94-100 %), and yet training from scratch is never trapped (gaps <= 0.8 %, own/oracle ratio <= 1.0075).

## Together
| | exploding one-start gradient at the oracle | trapped |
|---|---|---|
| GRU, LN MSE 3000 seeds 1, 2, 8, 9 (B = 0.01) | yes | yes |
| linear, same surrogates | yes | no |
| MLP, LN MSE 3000 (9 seeds) and slope from 0 | no | yes |
So at B = 0.01 the explosion at the good controller is neither necessary (MLP) nor sufficient (linear) for a trap. Lambda*
measures the explosion well: ||grad|| <= 4.7 e^Lambda* holds on all 180 GRU runs (max ratio 4.63), on all 90 linear runs
checked here (max 0.41) and on 89 of 90 MLP runs (one at 6.1); it is the link from explosion to trap that fails.
