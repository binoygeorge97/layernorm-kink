# Addendum 2026-11-04 to PREREG_arch Part B (322e353): the reason for the linear oracle's gain

Part B says the linear oracle's gain K ~ 17 is "not the LQR gain 10.29: the horizon-200 DPC optimum with the terminal weight is
steeper". That reason is wrong (docs/RESULTS_linear_gain.md, analysis/linear_gain.py): with the clipped shape clip(-Kx, +-10)
the DPC loss on the true plant is minimised at K = 10.25, with the shape 10 tanh(-Kx/10) at K = 17.00, and the 200-step LQR with
Q_f = 50 has gain 10.29 at its first step. The steeper gain compensates for the gradual bend of tanh. The registered
predictions (P2, P2', A2), the threshold and Lambda* of the trained oracle are unchanged; only the explanation is corrected.
Written before any Run 5 result existed.
