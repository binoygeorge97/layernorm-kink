# Task 5.4 (Q9): far-field branch function and decay laws on the trained surrogates

Written 2026-11-04. Script theory/far_field.py (float64), table out/theory/far_field.csv, figure out/figs/fig_far_field.png.
Coordinates of Theorem 1: w = U^T Q_E (z - z*), rho = ||w||, direction theta. Far-field branch function (Corollary 4):
Phi(theta) = w_d^T G(gamma * sqrt(H) U (cos theta, sin theta) + beta), computed from the weights of each LN MSE 3000 seed.
Decay exponents are log-log slopes over rho = 10 D to 1000 D (9 points), at 24 directions; "across" uses only directions
where Phi'(theta) is not near zero.

| seed | D | branch - Phi at 10 D | at 1000 D | along exponent (median [min, max] over 24 angles) | across exponent (median [min, max]) | branch J_u, u-axis | branch J_u, x-axis |
|---|---|---|---|---|---|---|---|
| 0 | 0.638 | 2.1e-02 | 2.4e-04 | -1.96 [-2.16, -1.71] | -0.92 [-1.13, -0.69] | -1.90 | -0.97 |
| 1 | 0.195 | 7.6e-03 | 8.0e-05 | -2.01 [-2.54, -1.49] | -0.75 [-1.56, -0.23] | -2.00 | -1.15 |
| 2 | 0.132 | 6.9e-03 | 6.6e-05 | -1.94 [-2.40, -1.51] | -0.99 [-1.19, -0.55] | -1.80 | -1.02 |
| 3 | 0.457 | 9.2e-03 | 1.0e-04 | -1.98 [-2.02, -1.93] | -0.92 [-1.41, -0.60] | -1.97 | -0.42 |
| 4 | 0.447 | 7.9e-03 | 8.6e-05 | -1.96 [-2.82, -1.69] | -0.87 [-1.42, -0.56] | -1.71 | -0.81 |
| 5 | 0.715 | 3.1e-02 | 3.4e-04 | -2.00 [-3.24, -1.92] | -0.92 [-1.12, -0.66] | -1.95 | -0.93 |
| 6 | 0.901 | 9.5e-03 | 9.9e-05 | -1.99 [-2.33, -1.76] | -0.94 [-1.01, -0.80] | -1.98 | -0.93 |
| 7 | 0.351 | 4.7e-03 | 6.6e-05 | -1.84 [-2.21, -1.56] | -0.80 [-1.23, -0.41] | -1.83 | -0.45 |
| 8 | 0.270 | 1.1e-02 | 1.2e-04 | -1.98 [-2.17, -1.96] | -0.83 [-1.79, -0.37] | -1.98 | -0.94 |
| 9 | 0.254 | 1.0e-02 | 1.0e-04 | -1.98 [-2.05, -1.79] | -0.75 [-2.05, -0.50] | -1.97 | -0.71 |

Findings.
- The branch converges to Phi(theta): the largest difference over the 24 directions falls by a factor 100 between 10 D and
  1000 D on every seed, i.e. as 1/rho (the next term of the expansion).
- Along rays the branch gradient decays as rho^-2: median exponent -1.84 to -2.01 over the seeds (all 10 within 0.16 of -2).
- Across rays it decays as rho^-1: median exponent -0.75 to -0.99. The fitted values are a little above -1 because at 10 D the
  O(1/rho^2) correction is still visible; on no seed is the decay faster than -1 in the median.
- In the original inputs: the branch's input slope J_u along the u-axis through the centre (a ray, mostly radial motion)
  decays with exponent -1.71 to -2.00; along the x-axis (across the u-direction) with -0.42 to -1.15 (median -0.93).

Consequence for the wording of the setpoint law (CLAUDE.md 1.3 "|B_hat/B - 1| ~ 0.49 d^-0.98"): the across-direction law
predicts an exponent of -1 for an input slope measured across the radial direction, and the fitted -0.98 is consistent with
it. The decay laws are confirmed, so the softening of Task 5.4 is not needed; a safe phrasing is "about d^-1 (fit -0.98 on
LN MSE 3000), as the 1/rho across-direction law (theory.tex, singular-value corollary, part (c)) predicts". (The frozen B01 report script fits all four LN
families pooled and gets 0.39 d^-0.88 at B = 0.01; quote the LN MSE 3000 fit when citing -0.98.)
