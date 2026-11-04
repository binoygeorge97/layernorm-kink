"""MLP mechanism (2026-11-04, POST HOC): warm start from the MLP oracle at a smaller AdamW learning rate. Same as the R5 warm
start (analysis/trap_predictor.py, set "mlp": N = 200, full batch of the repeat's 1000 training starts, wd 1e-4, log every 5
epochs, true-plant and surrogate cost of the current controllers), but lr = LR and EPOCHS epochs, on chosen seeds of one
family and repeat. The lr 1e-3 / 100-epoch reference is out/predictor/warm_mlp*.csv.
Usage: python analysis/mlp_warm_lr.py [lr=1e-4] [epochs=300] [seeds=1,2,8,9,3,7] [family="LN MSE 3000"] [rep=0]
-> out/mlp_mech/warm_lr<lr>_<epochs>.csv (curves) and a printed line per seed."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import level1_core as L, dpc_core as DC, phase_core as PC
jax.config.update("jax_enable_x64", True)
dt, tm = jnp.float64, jax.tree_util.tree_map
N, LOG = 200, 5
FAM = {"LN MSE 3000": "MSE (3000)", "LN two-stage": "J_x + J_u (two-stage)"}
a = sys.argv[1:] + [None] * 5
lr = float(a[0] or 1e-4); epochs = int(a[1] or 300)
seeds = [int(s) for s in (a[2] or "1,2,8,9,3,7").split(",")]; fam = a[3] or "LN MSE 3000"; rep = int(a[4] or 0)
plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
sur = FAM[fam]; norm = DC.surrogate_configs()[sur]["norm"]
P, S0, _ = PC._stack_models(paths.DATA, sur, seeds, dt)
dyn = DC.make_dynamics("surrogate", norm, plant)
roll_s, roll_t = DC._roller(dyn, N), DC._roller(DC.make_dynamics("true", plant=plant), N)
opt = optax.adamw(lr, weight_decay=1e-4)
step = DC.get_dpc_trainer(("mlpwarm", lr, norm, tuple(seeds)), dyn, N, LOG, opt, "mean", 3.0)
Co, cs = DC.load_controllers(os.path.join(paths.REPO, "data", "dpc_out_mlp"), "true plant (oracle)", rep)
C = tm(lambda x: jnp.asarray(np.asarray(x)[[cs.index(s) for s in seeds]], dt), Co)
x0 = PC._training_starts(seeds, rep, dt)
O = jax.vmap(opt.init)(C)


def cost(C):
    Xt, Ut = (np.asarray(z, float)[..., 0] for z in roll_t(C, P, S0, x0))
    Xs, Us = (np.asarray(z, float)[..., 0] for z in roll_s(C, P, S0, x0))
    return (np.array([DC.true_lqr_cost(Xt[i], Ut[i]) for i in range(len(seeds))]),
            np.array([DC.true_lqr_cost(Xs[i], Us[i]) for i in range(len(seeds))]))


rows, t0 = [], time.time()
ct, csr = cost(C)
for i, s in enumerate(seeds):
    rows.append(dict(family=fam, rep=rep, seed=s, lr=lr, epoch=0, cost_true=ct[i], cost_sur=csr[i]))
for e in range(LOG, epochs + 1, LOG):
    C, O, _ = step(C, O, P, S0, x0)
    ct, csr = cost(C)
    for i, s in enumerate(seeds):
        rows.append(dict(family=fam, rep=rep, seed=s, lr=lr, epoch=e, cost_true=ct[i], cost_sur=csr[i]))
D = pd.DataFrame(rows)
os.makedirs(paths.out("mlp_mech"), exist_ok=True)
D.to_csv(paths.out("mlp_mech", f"warm_lr{lr:g}_{epochs}.csv"), index=False)
for s in seeds:
    g = D[D.seed == s]; c0 = g.cost_true.iloc[0]
    print(f"{fam} s{s} r{rep} lr {lr:g}: true {c0:.3f} -> {g.cost_true.iloc[-1]:.3f} (max ratio {g[g.epoch >= 10].cost_true.max() / c0:.3f}), "
          f"surrogate {g.cost_sur.iloc[0]:.3f} -> {g.cost_sur.iloc[-1]:.3f}")
print(f"{time.time() - t0:.0f}s")
