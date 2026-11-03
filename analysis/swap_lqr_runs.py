"""Per-run data of the LQR swap test (Task 4.1): story/attribution.py part 1, unchanged computation, but every run is saved
(family, design, seed, r, x0, far, failed, catastrophic) so rates can be bootstrapped over seeds.
Check: the pooled rates must equal story/swap_lqr_rates.csv (the numbers on slides 13-14).
Output: out/stats/swap_lqr_runs.csv."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import level1_core as L  # noqa: E402
import phase_core as PC  # noqa: E402

OUT = paths.DATA
plant = L.Plant()
seeds = list(range(10))
A, B = plant.A, plant.B
refs, starts = PC.R_EVAL, PC.X0_TRACK
rr, xx = np.meshgrid(refs, starts, indexing="ij")
r = np.tile(rr.ravel(), (10, 1))
x0 = np.tile(xx.ravel(), (10, 1))
far = np.abs(x0 - r) >= 0.5
truth = PC.Model.true_plant(plant, seeds)
xo, uo, _, _ = PC.run_lqr(truth, truth, r, x0, "fixed")
Jo = PC.track_cost(xo, uo, r, plant)


def sim(uh, K):
    x = x0.copy()
    xs, us = [x], []
    for _ in range(200):
        u = np.clip(uh - K * (x - r), -10, 10)
        x = A * x + B * u
        xs.append(x), us.append(u)
    return np.stack(xs), np.stack(us)


DESIGNS = ["both from surrogate", "true A, surrogate B", "surrogate A, true B", "both true (only u_hat from surrogate)"]
rows = []
for name, sur in [("LN MSE 3000", "MSE (3000)"), ("LN slope from 0", "J_x + J_u (3000)"), ("LN MSE 12000", "MSE (12000)"),
                  ("LN two-stage", "J_x + J_u (two-stage)"), ("static norm", "static norm, MSE")]:
    m = PC.Model.surrogate(OUT, sur, seeds, name)
    uh, nroot, ah, bh = PC.equilibrium(m, r)
    Ks = [L.dare_K(ah, bh), L.dare_K(np.full_like(ah, A), bh), L.dare_K(ah, np.full_like(bh, B)),
          L.dare_K(np.full_like(ah, A), np.full_like(bh, B))]
    for tag, K in zip(DESIGNS, Ks):
        xs, us = sim(uh, K)
        met = PC.track_metrics(xs, us, r, plant, Jo)
        fail = met["diverged"] | (met["gap"] > 0.1)
        cat = met["diverged"] | (met["gap"] > 1.0)
        for i in range(10):
            for j in range(r.shape[1]):
                rows.append(dict(family=name, design=tag, seed=seeds[i], r=r[i, j], x0=x0[i, j], far=bool(far[i, j]),
                                 failed=bool(fail[i, j]), catastrophic=bool(cat[i, j])))
D = pd.DataFrame(rows)
D.to_csv(paths.out("stats", "swap_lqr_runs.csv"), index=False)
f = D[D.far].groupby(["family", "design"], sort=False)[["failed", "catastrophic"]].mean() * 100
print(f.round(2).to_string())
ref = pd.read_csv(os.path.join(paths.REPO, "story", "swap_lqr_rates.csv"))
print("\nreference story/swap_lqr_rates.csv:")
print(ref.round(2).to_string(index=False))
