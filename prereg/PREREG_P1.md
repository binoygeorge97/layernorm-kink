# Pre-registration: the two-state plant P1 (Task 2b, P1c)

Written 2026-11-04, BEFORE any P1 surrogate, oracle or DPC result exists (only the CPU QUICK pipeline checks of
tools/plant2_quick.py and the notebook QUICK run, 2 seeds with training shortened 10-20x, which are pipeline checks and are not
used below). Disclosure: those QUICK checks printed kink centres (largest coordinate up to 0.14, slope from step 0 seed 1,
after 300-1200 training steps, seeds 0-1) and LQR / trap numbers of the shortened runs; I have seen them. Commit hash: recorded in docs/RESULTS_LOG.md and in the commit that adds this file.

## Setup (fixed by docs/plant2/DECISION.md and the proposal)
- Plant: A = [[1.02, 0.05], [0, 1.01]], B = (0, 0.01)^T, |u| <= 10. Data box |x1| <= 7.3, |x2| <= 5.6, |u| <= 10;
  20 000 transitions per seed, sigma = 0.01, data key 1000 + s, init key s, slope draws 10000 + s; labels k = 1280 (P1b).
- Six surrogate families x 10 seeds (0-9), the scalar study's recipes (src/plant2_core.CONFIGS), notebook src/plant2_P1.ipynb.
- LQR study (plant2_core.lqr_study): setpoints r in {-4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4} on x1, equilibrium
  (r, -0.4 r), u_ss = 0.4 r; starts = the 25-point grid plant2_core.lqr_grid_starts() (a 5 x 5 grid along the principal axes of
  the recoverable set, inside the recoverable set scaled by 0.9); a start is EXCLUDED from the rates when
  max(|x0 - x_ss(r)|) < 0.5 (the scalar rule). 200 steps on the TRUE plant; gap = J / J_oracle - 1 with the true-plant LQR
  around the true equilibrium; diverged = |x| > 4 x box; failed = gap > 10 % or diverged; catastrophic = gap > 100 % or
  diverged. Fixed LQR = linearised once at the surrogate's own equilibrium for r; rescheduled = relinearised at (x_k, u_{k-1}).
- DPC: GRU (hidden 64) and MLP (2 x 64, tanh), u = 10 tanh(.), loss sum x^T Q x + R u^2 + x_N^T Q_f x_N over N, divided by N,
  Q = diag(5, 0.5), R = 0.1, Q_f = 10 Q; curriculum 5/10/20/50/100/200 with 1000/2000/2000/1000/1000/2000 epochs; AdamW lr
  1e-3, wd 1e-4; full batch of 1000 training starts (uniform in the start parallelogram, key 30000 + s + 1000 rep); controller
  key 20000 + s + 1000 rep; 3 repeats; all six families. Test: 1000 typical starts (key 12345) on the TRUE plant.
  failed = typical-start gap > 10 % against the oracle of the same seed, repeat and architecture; trapped = failed AND own
  controller costs > 1.05 x the oracle controller, both through the same surrogate (the scalar definition).
- Statistics: unit = seed; seed bootstrap with 10 000 resamples, generator seed 20261104.

## Predictions (each with its rule)
| id | prediction | rule: held if |
|---|---|---|
| L1 | Fixed LQR: LN MSE 3000 fails more often than static norm | pooled failed rate LN MSE 3000 > static norm AND the 95 % seed-bootstrap CI of the difference (LN MSE 3000 - static) lies above 0 |
| L2 | Rescheduled LQR: LN MSE 3000 fails more often than static norm | same rule for the rescheduled controller |
| L3 | Fixed LQR: two-stage within the kink-free range | failed rate two-stage <= max(no norm, static norm) + 5 percentage points |
| L4 | Rescheduled LQR: two-stage within the kink-free range | same rule for the rescheduled controller |
| D1 | GRU DPC: LN MSE 3000 has traps | at least 1 of the 30 GRU runs on LN MSE 3000 trapped |
| D2 | MLP DPC: LN MSE 3000 has traps | at least 1 of the 30 MLP runs on LN MSE 3000 trapped |
| D3 | GRU DPC: two-stage, no norm, static norm have no traps | 0 of 30 trapped in each of the three families (3 cells) |
| D4 | MLP DPC: two-stage, no norm, static norm have no traps | 0 of 30 trapped in each of the three families (3 cells) |
| K1 | Every LN kink centre lies close to the origin in (x1, x2, u) | for all 40 LN surrogates (LN MSE 3000, LN MSE 12000, slope from step 0, two-stage x 10 seeds): max(abs(z_x1), abs(z_x2), abs(z_u)) <= 0.3 |
Outcomes are binary (held / failed). For D3, D4 and K1 every cell is reported, and a failure names the cells that broke it.
Reasoning (from the scalar study, not from P1 data): the kink mechanism does not depend on the state dimension; the encoder
bias starts at zero, so the centre stays near the origin (scalar: within 0.15 in units where the box half-widths are 4 and
10; here 0.3 against half-widths 7.3, 5.6, 10); two-stage removed the GRU and MLP traps at B = 0.01; the L3/L4 tolerance of 5
points was chosen from the scalar rates (two-stage 5.7 % vs max kink-free 2.6 % fixed; 3.5 % vs 5.7 % rescheduled).

## Not predicted (reported as diagnostics only)
- The vector landing amplification Lambda* (docs/theory/vector_lambda) of each oracle through each surrogate. It failed as a
  predictor in the scalar study; it is computed and reported, with no threshold and no prediction.
- Trap and failure counts of LN MSE 12000 and slope from step 0; catastrophic rates; the input-slope entries h (x1) and g (x2)
  of each surrogate (P1a); per-setpoint failure rates.
- Whether the P1 result "differs from the scalar result" (the trigger for the optional P2 control) is judged on D1-D4 and
  L1-L4: if any of them fails, P2 is proposed.

## Analysis
Outcomes will be scored by a script (analysis/p1_report.py) written and committed before the DPC results are read. L1-L4 and
K1 can be scored after the surrogates + oracle session; they are scored only after this file is committed.
