# Why the linear DPC oracle lands at K ~ 16.7-17 (B5, 2026-11-04)

Script `analysis/linear_gain.py`; figure `out/figs/fig_linear_gain.png`; scan `out/figs/linear_gain_scan.csv`.

The linear controller is u = 10 tanh(-K x / 10). Near x = 0 it is the line -K x, but it bends over gradually: at K = 10.29 it
gives u = -7.7 at x = 1, where the saturated LQR clip(-10.29 x, +-10) already gives -10 (figure, left, dotted vs grey).
Most training starts (x0 ~ U[-3, 3]) begin in the range where the optimal input is at or near the limit, so DPC steepens the
slope at the origin to push the input to the limit sooner, accepting a too-aggressive response near zero. Scanning the gain
on the true plant (1000 training starts of seed 0 repeat 0, the DPC loss of dpc_core; figure, right): the clipped line is best at
K = 10.25 (the LQR gain), the tanh shape at K = 17.00, the value training converges to (16.70-17.00 over the 30 oracles). Over the
small starts only (|x0| <= 0.3, no saturation, n = 90) the tanh optimum falls back to K = 10.5. The finite horizon is not the
reason: the time-varying LQR with Q_f = 50 over 200 steps has gain 10.29 at its first step. The price of the shape is small:
true-plant test cost 2.5235 for the linear oracle (seed 0, rep 0; best tanh-linear 2.5235 at K = 16.75) against 2.4968 for
the saturated LQR (+1.1 %). The MLP oracle has the freedom to follow the clipped shape and does: slope 10.45 at x = 0,
cost 2.4972.

| policy (seed 0, rep 0) | slope at x = 0 | true-plant test cost |
|---|---|---|
| saturated LQR clip(-10.29 x, +-10) | 10.29 | 2.4968 |
| MLP oracle | 10.45 | 2.4972 |
| linear oracle 10 tanh(-K x / 10) | 16.93 | 2.5235 |

Consequence for Task 1: the linear oracle's steeper gain narrows the multiplier's safe band (upper edge (A + 1)/K = 11.9 B
instead of 19.7 B), which is why the trained linear oracle predicts traps (PREREG_arch Part B, P2') while the LQR-gain policy
did not (P2). Corrections: (1) PREREG_arch Part B (322e353) and NOTES_controller_arch.md A3 attribute the steeper DPC
optimum to "the horizon-200 optimum with the terminal weight"; that explanation is wrong, the cause is the tanh shape
(addendum prereg/ADDENDUM_2026-11-04_linear_gain.md; no prediction changes). (2) The healthy MLP slope is about 10.5, not
"-10 to -17" as A3 expected (A3 is scored after Run 5 as written).
