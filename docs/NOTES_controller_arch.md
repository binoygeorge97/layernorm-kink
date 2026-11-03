# Controller architecture study: GRU vs MLP vs linear (designed 2026-10-23)

Why: the GRU controller came from the user's DPC example. For a 1-state plant it is more than the task needs and it
hides the analysis: (1) a hidden state means u depends on history, so no policy u = pi(x) can be drawn; (2) backprop
through time through the controller mixes with backprop through the kinked surrogate; (3) there is no gain to compare
with the LQR optimum K = 10.29. Switching to memoryless controllers fixes all three and connects back to the seed-0
sweeps in NOTES_dpc_kink.md, which used a 64x64 tanh MLP.

## Code (claude/dpc/, notebooks rebuilt 2026-10-23)
- `dpc_core.ARCH` in {"gru", "mlp", "linear"}; set before training (notebook setting `ARCH`).
  - gru: unchanged (the original study; same keys, bit-identical).
  - mlp: u = 10 tanh(MLP(x)), MLP_WIDTH = 64, MLP_DEPTH = 2 tanh layers; input (x - r, r) for tracking.
  - linear: u = 10 tanh((K x + c) / 10) ~ sat(K x + c); 2 parameters (per input); stored / 10 so Adam steps fit;
    init K0 = -5 (stabilising, half the LQR gain).
  - `ctrl_step` dispatches on the saved parameter tree, so phase code runs any architecture unchanged.
- Output folders: `arch_dir(out)` = out for gru, out_mlp / out_linear otherwise; surrogates and labels are READ from
  the GRU folder (`sur_dir`), so every architecture controls the same 60 surrogates.
- New analysis (dpc_report): `policy_gain_table` (K_eff = -d pi/dx at x = 0, 0.5, 2), `fig_policy_maps` (D9),
  `fig_linear_landscape` (D10: DPC loss over (K, c) through the surrogate vs the true plant),
  `fig_gradient_audit` (D11: dL/dK through surrogate vs true plant, wrong-sign share; `dpc_core.gradient_audit`).
- Smoke test 2026-10-23 (seeds 2, 9; 260 epochs; CPU 5.5 min for both architectures): no errors, all figures drawn.

## Pre-registered predictions (write the outcome either way)
- A1 The MLP reproduces the GRU result: LN MSE and slope-from-step-0 fail on the same narrow-kink seeds (1, 2, 8, 9 and
  2, 3, 9), two-stage and kink-free on none; the trap test (own / reference on the surrogate) > 1 on the failures.
- A2 The 2-parameter linear controller does NOT fail on those seeds: the kink traps a flexible policy, it does not make
  the problem unsolvable. (Smoke test, 2 seeds, short training: linear LN-MSE cost 3.00 vs oracle 3.00 on seeds 2 and 9,
  while the MLP diverged on seed 2 (K_eff(0) = -85, inverted policy) and failed on seed 9 (gap ~100x, K_eff(0) = 1.4).
  Consistent with A1 and A2; not evidence yet.)
- A3 Policy maps: trapped MLP policies are flat or inverted inside the kink band and recover the LQR slope outside it;
  healthy ones are a saturated line of slope ~ -10 to -17 (the DPC horizon-200 optimum is steeper than the infinite-
  horizon LQR 10.29 because of the terminal weight).
- A4 Landscape (D10): the linear problem's minimum through the surrogate is within ~10 % of the true plant's K on every
  seed (smoke test: -16 vs -17 on LN MSE seed 2, -17 vs -17 on seed 9). So no spurious minimum for 2 parameters.
- A5 Gradient audit (D11): the LN MSE surrogate gives dL/dK the wrong sign on a larger share of the gain axis than the
  kink-free models (smoke test: 15-23 % vs 0-5 %), concentrated at small |K| where the closed loop lingers in the kink.
  Two-stage in between (2-8 %).
- A6 Capacity: the trap rate rises with MLP width / depth (a richer policy has more ways to avoid the kink).
- A7 The four tall-narrow survivors (MSE 12000 seed 2, LN MSE seed 7, two-stage seeds 1/2 ...) have their wrong-sign
  arms pointed away from the equilibrium line u = -3x, so the policy never meets them: check with D9 colour dots.

## Experiments, in order (each resumable; same MyDrive/dpc_out surrogates)
| # | what | run | cost (A100) | answers |
|---|---|---|---|---|
| E-A1 | MLP regulation, 7 models x 10 seeds x 3 repeats | DPC notebook, ARCH = "mlp", REPS = [0, 1, 2] | ~1.5 h | A1, A3 (D9), trap test |
| E-A2 | linear regulation, same grid | ARCH = "linear" | ~1 h | A2; D10 landscape on the failing seeds |
| E-A3 | gradient audit, all 10 seeds x 6 surrogates | cell in the DPC notebook (no training) | minutes | A5 |
| E-A4 | MLP tracking (phase notebook, ARCH = "mlp"), 1 repeat | phase notebook | ~40 min | does P8's seed-row pattern hold for a memoryless policy; D9 for tracking |
| E-A5 | capacity sweep: MLP width {16, 64, 256} x depth {1, 2}, LN MSE + two-stage + static, 10 seeds, 1 repeat | set MLP_WIDTH / MLP_DEPTH in the notebook; one folder per setting | ~2 h total | A6 |
| E-A6 | survivor analysis | post-processing of E-A1 (D9 dots vs the equilibrium line; angle of the wrong-sign arms) | minutes | A7 |
Run E-A1, E-A2, E-A3 first (one Colab session). E-A4-6 after interpreting those.

## Reporting
- GRU results stay as the "original pipeline" row; the paper's main controller becomes the MLP (simpler, standard DPC
  policy class); the linear controller and the landscape are the mechanism figure; the gradient audit is the closed-loop
  test of Corollary U5.
- Compare architectures only within one precision (float64 on A100).
