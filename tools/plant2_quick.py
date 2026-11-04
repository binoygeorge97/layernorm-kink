"""Q8 milestone (ii), CPU QUICK: all six families on the proposal's P1 plant, seeds 0-1, phases scaled by 0.1 (300 / 1200
steps; two-stage 300 + 100), labels k = 1500 (cached in out/plant2_quick/labels2). Per family x seed: Level 1 + kink
metrics; a small LQR study (setpoints -1, 0, 1 x starts (+-2, 0), (0, +-1.5)); one short DPC GRU run through the surrogate
and through the true plant (curriculum (5, 100), (20, 100), (50, 100), 128 starts) and the vector Lambda* of the true-plant
controller through the surrogate (N = 50). Output out/plant2_quick/*.csv. Numbers are pipeline checks, not results."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import plant2_core as P2
OUT = paths.out("plant2_quick")
pl = P2.Plant2([[1.02, 0.05], [0.0, 1.01]], [0.0, 0.01])
X0 = jnp.asarray(np.random.default_rng(30000).uniform([-3, -2], [3, 2], (128, 2)))
cur = [(5, 100), (20, 100), (50, 100)]
c_true, _ = P2.train_dpc(P2.true_dyn(pl), jax.random.PRNGKey(20000), X0, cur)
L1, LQ, DP = [], [], []
for fam, cfg in P2.configs(0.1).items():
    for s in (0, 1):
        t0 = time.time()
        p, s0, _ = P2.train(cfg, s, pl, label_dir=os.path.join(OUT, "labels2"))
        m = P2.level1_metrics(p, s0, cfg["norm"], pl, n=11); L1.append(dict(family=fam, seed=s, **m))
        for r in P2.lqr_study(p, s0, cfg["norm"], pl, [-1.0, 0.0, 1.0], [(2.0, 0.0), (-2.0, 0.0), (0.0, 1.5), (0.0, -1.5)], N=100):
            LQ.append(dict(family=fam, seed=s, **r))
        dyn = P2.sur_dyn(p, s0, cfg["norm"])
        c, hist = P2.train_dpc(dyn, jax.random.PRNGKey(20000 + s), X0, cur)
        xs, us = jax.vmap(lambda x0: P2.rollout(c, P2.true_dyn(pl), x0, 50))(X0)
        lam, kl = P2.vector_lambda(c_true, p, s0, cfg["norm"], X0, N=50)
        DP.append(dict(family=fam, seed=s, dpc_loss_sur=hist[-1][1], true_final_abs_med=float(jnp.median(jnp.abs(xs[:, -1]))),
                       vec_lambda_star=float(lam.max()), landed=int((kl < 50).sum())))
        print(f"{fam:22s} s{s}: rel_mse {m['rel_mse']:.2e} Ju_sup_rel {m['Ju_sup_rel']:.3g} D {m.get('D', np.nan):.3g} | "
              f"vec Lambda* {lam.max():.2f} | {time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(L1).to_csv(os.path.join(OUT, "level1.csv"), index=False)
        pd.DataFrame(LQ).to_csv(os.path.join(OUT, "lqr.csv"), index=False)
        pd.DataFrame(DP).to_csv(os.path.join(OUT, "dpc.csv"), index=False)
T = pd.DataFrame(LQ); T["failed"] = T.diverged | (T.gap > 0.1)
print(T.groupby(["family", "controller"]).failed.mean().unstack().round(2))
print("MILESTONE (ii) finished: all six families ran end to end")
