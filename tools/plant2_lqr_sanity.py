"""Q8 check: the plant-2 LQR study on ONE fully trained surrogate (static norm, MSE, 3000 steps, seed 0, P1 plant) should give
low failure rates if the LQR code is right (scalar study: static norm 0.6 % fixed). Setpoints -1, 0, 1; starts on a 3 x 3 grid."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
import plant2_core as P2
pl = P2.Plant2([[1.02, 0.05], [0.0, 1.01]], [0.0, 0.01])
t0 = time.time()
p, s0, _ = P2.train(P2.CONFIGS["static norm, MSE"], 0, pl)
m = P2.level1_metrics(p, s0, "static", pl, n=11)
print(f"trained in {time.time() - t0:.0f}s:", {k: round(v, 5) for k, v in m.items()})
starts = [(a, b) for a in (-2.0, 0.0, 2.0) for b in (-1.5, 0.0, 1.5)]
T = pd.DataFrame(P2.lqr_study(p, s0, "static", pl, [-1.0, 0.0, 1.0], starts, N=200))
T["dist"] = [np.hypot(r.x0_1 - r.r, r.x0_2 - float(pl.u_ss(r.r)[0][0])) for _, r in T.iterrows()]
T = T[T.dist >= 0.5]; T["failed"] = T.diverged | (T.gap > 0.1)
print(T.groupby("controller").agg(n=("gap", "size"), failed=("failed", "mean"), gap_median=("gap", "median"), gap_max=("gap", "max")))
print("setpoint gains K:", T.groupby("r")[["K1", "K2"]].first().round(3).to_dict("index"), "| true K:", np.round(P2.lqr_K(pl.A, pl.B), 3))
