# Why is the MLP trapped? (2026-11-04; ALL POST HOC, nothing pre-registered, no claim changed)

Context: the MLP DPC controller is trapped on 27/30 LN MSE 3000 and 12/30 slope-from-0 runs (Run 5a), with a tame gradient at
its oracle, so the GRU mechanism (exploding, one-start, direction-chaotic gradient at the good controller) does not explain it.
"Trapped" = typical-start gap > 10 % AND own controller > 1.05 x the oracle controller through the same surrogate.
Scripts: theory/mlp_line.py, theory/mlp_probe.py, theory/mlp_sat_park.py, analysis/mlp_warm_lr.py, analysis/mlp_extend.py,
out/predictor/warm_mlp.csv (R5 warm starts). Data: data/run5_mlp (controllers; slope-from-0 files copied read-only from Drive),
data/run5_mlp_curves/dpc_curves.csv (Run 5a training curves, read-only copy), data/dpc_out_mlp (MLP oracles).

## 1. The GRU checks on the MLP pairs (LN MSE 3000 rep 0; two-stage seeds 2, 9 as healthy controls)
Straight line from the DPC controller (alpha = 0) to the MLP oracle (alpha = 1), DPC loss through the surrogate, N = 200, the
repeat's 1000 training starts; alpha grid 41 points plus 20 near each end.
| pair | label | surrogate loss DPC -> oracle | barrier | grad at DPC | largest grad on line (alpha, top-start share there) | grad at oracle | warm start lr 1e-3 |
|---|---|---|---|---|---|---|---|
| LN MSE s1 | trapped | 11.66 -> 2.59 | none (monotone) | 31.7 | 150 (0.70, 0.003) | 3.85 | expelled, epoch 10 (x2.7) |
| LN MSE s2 | trapped | 12.04 -> 2.66 | none | 53.6 | 245 (0.70; 118 starts > 1e3) | 0.65 | expelled, epoch 10 (x2.6) |
| LN MSE s8 | trapped | 11.78 -> 2.68 | none (monotone) | 23.9 | 3.6e36 (0.96, 0.999) | 1.52 | expelled, epoch 10 (x2.8) |
| LN MSE s9 | trapped | 4.97 -> 2.62 | none (monotone) | 7.5 | 23.6 (0.83, 0.002) | 0.62 | expelled, epoch 15 (x2.4) |
| LN MSE s3 | trapped | 56.2 -> 2.51 | none (monotone) | 1.1e3 (519 starts > 1e3) | at the DPC end | 1.54 | drifts (x1.17 at epoch 70) |
| LN MSE s7 | trapped | 3.00 -> 2.59 | +0.27 at alpha 0.63 | 0.78 | 20 (0.88, 0.001) | 2.73 | stays (x1.04) |
| LN MSE s4 | healthy | 2.55 -> 2.57 | +0.84 at alpha 0.30 | 0.001 | 166 (0.63) | 1.78 | stays (x1.03) |
| two-stage s2 | healthy | 2.87 -> 2.88 | +0.003 | 0.50 | 2.9 | 0.96 | stays |
| two-stage s9 | healthy | 1.99 -> 2.01 | +0.04 | 1.00 | 5.7 | 1.70 | stays |
Answer to "basin with a barrier, spike on the way in, or something else":
- Something else, on most trapped seeds (s1, s2, s9, s3; and s8 apart from its spike). There is no barrier: the surrogate
  loss falls monotonically from the trapped controller to the oracle. The trapped controller is not a stationary point (gradient
  7.5 to 1.1e3, larger than at the oracle), and the surrogate prices it correctly (surrogate loss = true loss within 1 %).
  The optimiser stopped on a slope it could have descended.
- A spike on the way in: seed 8 only (3.6e36 at alpha 0.96, one start carrying 99.9 %; GRU-like but near the oracle, not at it).
  Seed 2 has 118 exploding starts at alpha 0.70, spread (top share 0.02).
- A basin with a barrier: seed 7 (+0.27; small gradient 0.78 at the DPC end; a mild trap, loss 3.00 vs 2.59). Seed 4 (healthy)
  sits in a different basin behind a barrier (+0.84) with a slightly LOWER loss than the oracle: not a trap.

Probes at the MLP oracle (theory/mlp_probe.py), same pairs:
- Gradient direction is stable: cosine between the gradient at c and at c (1 + 1e-4 xi) is 0.997-1.000 everywhere (the GRU's
  flipped). So the MLP gradient is not chaotic in direction.
- AdamW's first step (lr * sign(g)) RAISES the loss on every pair, trapped and healthy alike (+0.12 to +0.31; first-order
  prediction -0.01 to -0.10); at lr 1e-4 it descends on 7 of 9 pairs (rises on LN MSE s3 and s4). Curvature along the sign step
  44-425. So the oracle sits in a stiff valley for every surrogate; this does not separate trapped from healthy.

## 2. M4: warm start from the oracle at lr 1e-3 (100 epochs; expelled = true cost > 1.10 x epoch 0 at any epoch 10-100)
From the R5 warm starts (110 of 180 MLP runs computed: LN MSE 3000 30, MSE 12000 30, slope from 0 30, two-stage 20):
| family | healthy: expelled / runs | trapped: expelled / runs |
|---|---|---|
| LN MSE 3000 | 0 / 3 (max ratio 1.03) | 14 / 27 (median epoch 10) |
| slope from 0 | 1 / 18 (s2 r0, ratio 1.16, epoch 15) | 8 / 12 |
| MSE 12000 | 0 / 29 | 0 / 1 |
| two-stage | 0 / 20 | - |
| total | 1 / 70 | 22 / 40 |
Healthy MLP pairs stay; trapped ones are expelled in about half the runs (within 10-15 epochs). Expulsion therefore marks the
trap-prone surrogates, but does not happen on all trapped runs. At lr 1e-4 (300 epochs, LN MSE s1, 2, 8, 9, 3, 7) the oracle
stays on 5 of 6 (max ratio 1.001-1.011; s2 drifts to 1.065, below the 10 % line): as for the GRU, a smaller step keeps it.

## 3. M1: converged or trapped?
(a) Training curves (Run 5a dpc_curves, logged every 250 epochs; the last 500 epochs are at N = 200), relative drop of the DPC
training loss over epochs 8500 -> 9000:
| family | label | flat (abs <= 1 %) | still falling (> 1 %) | rising (> 1 %) |
|---|---|---|---|---|
| LN MSE 3000 | trapped (27) | 17 | 8 | 2 |
| LN MSE 3000 | healthy (3) | 3 | 0 | 0 |
| slope from 0 | trapped (12) | 8 | 3 | 1 |
| slope from 0 | healthy (18) | 17 | 1 | 0 |
Still falling at the end: LN MSE s0 r0 (-23.5 % over the last 500 epochs; loss 8.4 vs oracle 2.5), s0 r1 (-17.5 %), s1 r0
(-10.5 %), s3 r0 (-22.8 %), s5 r0 (-4.2 %), s5 r1 (-9.2 %), s9 r1 (-3.1 %), s9 r2 (-2.2 %); slope from 0 s2 r2, s3 r2, s9 r0
(-7 to -8 %). Rising: LN MSE s2 r1, s2 r2; slope from 0 s8 r2. So 11 of the 39 trapped runs (28 %) were still descending
clearly when training stopped, 25 were flat at a loss far above the oracle's, 3 were getting worse.
(b) Extended training (+3000 epochs at N = 200 from the Run 5a controllers, fresh AdamW; rep 0; LN MSE 3000 s1, 2, 8, 9 and
healthy s4, two-stage s9 as a second healthy pair since LN MSE 3000 has no second healthy seed): SEE BELOW (running).

## 4. M2: saturation (all 39 trapped runs and their oracles; |pre-tanh| > 2 = saturated)
| family | controller | saturated share of (start, step) pairs, median (IQR) | gradient share from unsaturated steps, median (min) | gradient norm, median |
|---|---|---|---|---|
| LN MSE 3000 | trapped DPC (27) | 0.107 (0.075-0.124) | 0.995 (0.16, the diverged run s8 r1) | 27 |
| LN MSE 3000 | oracle (27) | 0.080 (0.079-0.084) | 0.998 (0.945) | 1.75 |
| slope from 0 | trapped DPC (12) | 0.118 (0.114-0.123) | 0.999 (0.943) | 15 |
| slope from 0 | oracle (12) | 0.081 (0.077-0.083) | 0.997 (0.987) | 1.17 |
(The split g = g_unsat + g_sat is exact to 1e-12.) The hypothesis is not supported: trapped controllers are saturated only
slightly more often than the oracle (11-12 % vs 8 % of steps), their gradient comes almost entirely from unsaturated steps,
and it is LARGER than at the oracle, not vanishing. Except for the one diverged run (s8 r1: 99.95 % saturated), the trapped
MLP is not stuck in bang-bang saturation.

## 5. M3: where the trapped MLP parks
For each trapped run: final state after 200 steps through the surrogate and through the true plant (1000 training starts),
and the fixed points of x -> F_hat(x, pi(x)) and x -> A x + B pi(x).
| outcome | LN MSE 3000 (27) | slope from 0 (12) |
|---|---|---|
| all starts park at one fixed point | 11 | 11 |
| two attracting fixed points (starts split) | 8 | 0 |
| period-2 cycle of the surrogate loop, no stable fixed point | 7 | 0 |
| diverged | 1 (s8 r1) | 1 (s2 r2) |
- Parking points lie at |x| = 0.13 to 2.5 (slope from 0: 0.14 to 1.59). On 29 of the 38 stable surrogate fixed points the TRUE
  closed loop has the same fixed point within 0.01, and the final states agree within 0.01-0.04. The surrogate's input slope at
  the parking point is close to correct (J_u_hat / B = 0.51 to 1.38 on 35 of 38; 3.1, 6.6 and 9.6 on the other three); no
  parking point lies on a wrong-sign arm (J_u_hat / B < 0 on none). Distance from the kink centre: median 13.6 half-widths
  (range 0.3 to 86); only 2 of 38 lie inside the kink ellipse (LN MSE s5 r0: 0.3 and 0.8 half-widths).
- So the MLP parking point is NOT a false equilibrium of the surrogate. It is the controller's own spurious equilibrium: pi
  crosses u_ss(x) = -3 x at a non-zero x (e.g. s1 r0: x = +1.29, pi = -3.9), so the true plant holds there too, and the
  surrogate, accurate there, agrees. The kink is not at the parking point; it acts during training.
- Period-2 cycles (7 LN MSE 3000 runs: s0 r1, s1 r1, s2 r1, s3 r1, s1 r2, s3 r2, s8 r2): the surrogate loop alternates between
  two points at x = 0.09-0.35 (e.g. s0 r1: 0.272 / 0.284; s1 r1: 0.177 / 0.093 with u = -8.9 / +8.1), from every start tried. In
  the (x, u) plane they lie 0.9 to 67 half-widths from the kink centre (large because u swings by up to +-9 on s1 r1, s2 r1,
  s8 r2; 0.9-5 on s0 r1, s3 r1, s3 r2). The true loop typically has a nearby fixed point instead (s0 r1: +0.281). These are
  small-offset, chattering controllers, not parking at a distant point.
- Comparison with the GRU (slide 16): GRU seed 1 parks at -0.58 or +1.71, seed 2 at +1.10. MLP rep 0: seed 1 at +1.29, seed 2 at
  -1.55 or +0.94, seed 8 at -1.30, seed 9 at +0.64. Similar magnitudes (about 1 to 2), different points; both are offsets of the
  policy, held on both plants.

## Reading (so far; M1(b) pending)
- The MLP trap is not the GRU mechanism. At the MLP oracle the gradient is tame and stable in direction. There is no barrier
  between the trapped controller and the oracle on most seeds, and the trapped controller is not a stationary point. The
  trapped policies are bang-bang-like controllers with either a spurious equilibrium at |x| ~ 0.2-2.5 that both plants share
  (30 of 39), a period-2 chatter near x = 0.1-0.35 (7), or divergence (2).
- Two facts point at the optimiser near the good controller rather than at a local minimum: the oracle sits in a stiff valley
  (Adam's first 1e-3 step raises the loss everywhere), and on the trap-prone surrogates training from the oracle at lr 1e-3 is
  expelled in about half the runs, while healthy pairs never are; at lr 1e-4 it stays.
- 28 % of the trapped MLP runs were still descending clearly when training stopped. If extended training brings them to the
  oracle, part of the MLP "trapped" label counts slow convergence (section 3b decides).
