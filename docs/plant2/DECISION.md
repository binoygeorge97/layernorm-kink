# Task 2b plant decision (2026-11-04)

The user confirmed plant P1 of out/plant2/proposal.pdf (docs/plant2/proposal.tex, 3 Nov), with the proposal's settings:
- A = [[1.02, 0.05], [0, 1.01]], B = (0, 0.01)^T, |u| <= 10.
- Data box |x1| <= 7.3, |x2| <= 5.6, |u| <= 10.
- Setpoints r in {-4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4} on x1; starts on a 5 x 5 grid inside the recoverable set.
- Q = diag(5, 0.5), R = 0.1, Q_f = 10 Q.
- DPC with GRU and MLP, 3 repeats.
P2 stays an optional control, run only if the P1 result differs from the scalar one.

Before any Colab run (user's list): P1a second error channel (addendum to the proposal), P1b label accuracy in three inputs,
P1c pre-registration before the P1 DPC run (Lambda for the vector plant is a diagnostic only), P1d milestones and the Colab
settings for the surrogates + oracle session (Drive folder MyDrive/dpc_out_P1, new).
