# Addendum 2026-11-04 to PREREG_arch.md (Part A f945a5c, Part B 322e353): the exactness check

Committed pre-registrations are not edited; this file corrects the wording of their notes on the check
sum_k log|m_k| = log|dx_N/dx_0|. No prediction, threshold or definition changes.

How theory/t2_arch.py computed the two sides:
- left: the trajectory (x_k, u_k) of one call of `roll`; m_k = J_x_hat - J_u_hat K_k evaluated at those stored points with
  separate jax.grad calls (T.slopes, grad of pi);
- right: jax.jvp of a SECOND call of `roll` from the same x0; JAX recomputes the primal trajectory inside the jvp, compiled
  together with the tangent, so its rounding differs from the first call.

Recomputed along ONE stored trajectory (analysis/exactness_same_traj.py; linear oracle, LN MSE 3000, the start with the largest
disagreement of each run): one-step jvp of x -> F(x, pi(x)) at each stored x_k vs m_k: max relative difference 1.2e-13 to
3.7e-12; sum of log|one-step jvp| along the stored path vs sum log|m_k|: 5.7e-14 to 1.8e-11 (s9 r0, s1 r0, s2 r1, s8 r2). The
identity therefore holds along one trajectory to rounding, as the chain rule requires.
The second rollout leaves the stored path at the 1e-12 level at steps 29-69 and by more than 1e-6 at steps 51-104 (largest
deviations 1e-3 to 2e-2, while |x| stays below about 0.5): the closed loop through the surrogate amplifies rounding (it is
chaotic on these runs), so the two sides were evaluated on two different trajectories after that point.

Corrected wording (replaces the notes "fails on 15 LN MSE 3000 runs ... chaos ..." in Part B and "holds to <= 1e-6 on 179 of
180 runs ... underflow" in Part A): "The check compares two rollouts. Along one stored trajectory the product of the one-step
derivatives equals prod m_k to rounding (<= 2e-11 in the log) on every run checked. The full-rollout jvp is a second rollout;
where the closed loop through the surrogate amplifies (the trapped LN MSE runs) the two rollouts separate after 30-100 steps,
and where it contracts strongly the jvp underflows (|dx_N/dx_0| < 1e-300). Lambda* uses the multipliers of one rollout and is
not affected."
