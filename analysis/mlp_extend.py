"""MLP mechanism M1 (2026-11-04, POST HOC): converged or trapped? Continue training the Run 5a MLP DPC controllers (rep REP) for
EPOCHS more epochs at N = 200 through their surrogate, full batch of the repeat's 1000 training starts, AdamW (wd 1e-4) at lr,
from a FRESH optimiser state (Run 5a did not save Adam's moments). Every LOG epochs: the DPC loss through the surrogate
(dpc_core.true_lqr_cost of the surrogate rollout = what training descends, as in the warm-start harness) and through the true
plant, and the same for the MLP oracle of the repeat (computed once). Reached = surrogate cost within 5 % of the oracle's.
Usage: python analysis/mlp_extend.py lr epochs seeds family [rep=0]   e.g. 1e-3 3000 1,2,8,9,4 "LN MSE 3000"
-> out/mlp_mech/extend_<family>_lr<lr>_<epochs>.csv"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import level1_core as L, dpc_core as DC, phase_core as PC
jax.config.update("jax_enable_x64", True)
dt, tm = jnp.float64, jax.tree_util.tree_map
N, LOG = 200, 50
FAM = {"LN MSE 3000": ("MSE (3000)", "LN, MSE (3000)"), "LN two-stage": ("J_x + J_u (two-stage)", "LN, two-stage")}
lr, epochs = float(sys.argv[1]), int(sys.argv[2]); seeds = [int(s) for s in sys.argv[3].split(",")]; fam = sys.argv[4]
rep = int(sys.argv[5]) if len(sys.argv) > 5 else 0
sur, model = FAM[fam]
plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0); norm = DC.surrogate_configs()[sur]["norm"]
P, S0, _ = PC._stack_models(paths.DATA, sur, seeds, dt)
dyn = DC.make_dynamics("surrogate", norm, plant)
roll_s, roll_t = DC._roller(dyn, N), DC._roller(DC.make_dynamics("true", plant=plant), N)
opt = optax.adamw(lr, weight_decay=1e-4)
step = DC.get_dpc_trainer(("mlpext", lr, norm, tuple(seeds)), dyn, N, LOG, opt, "mean", 3.0)
pick = lambda C, cs: tm(lambda x: jnp.asarray(np.asarray(x)[[cs.index(s) for s in seeds]], dt), C)
Cd, csd = DC.load_controllers(os.path.join(paths.REPO, "data", "run5_mlp"), model, rep)
Co, cso = DC.load_controllers(os.path.join(paths.REPO, "data", "dpc_out_mlp"), "true plant (oracle)", rep)
C, Corc = pick(Cd, csd), pick(Co, cso)
x0 = PC._training_starts(seeds, rep, dt)
O = jax.vmap(opt.init)(C)


def cost(C):
    Xt, Ut = (np.asarray(z, float)[..., 0] for z in roll_t(C, P, S0, x0))
    Xs, Us = (np.asarray(z, float)[..., 0] for z in roll_s(C, P, S0, x0))
    return (np.array([DC.true_lqr_cost(Xs[i], Us[i]) for i in range(len(seeds))]),
            np.array([DC.true_lqr_cost(Xt[i], Ut[i]) for i in range(len(seeds))]))


orc_s, orc_t = cost(Corc)
rows, t0 = [], time.time()
cs_, ct_ = cost(C)
for i, s in enumerate(seeds):
    rows.append(dict(family=fam, seed=s, rep=rep, lr=lr, epoch=0, cost_sur=cs_[i], cost_true=ct_[i], oracle_sur=orc_s[i], oracle_true=orc_t[i]))
out = paths.out("mlp_mech", f"extend_{fam.replace(' ', '')}_lr{lr:g}_{epochs}.csv")
for e in range(LOG, epochs + 1, LOG):
    C, O, _ = step(C, O, P, S0, x0)
    cs_, ct_ = cost(C)
    for i, s in enumerate(seeds):
        rows.append(dict(family=fam, seed=s, rep=rep, lr=lr, epoch=e, cost_sur=cs_[i], cost_true=ct_[i], oracle_sur=orc_s[i], oracle_true=orc_t[i]))
    if e % 500 == 0:
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"epoch {e} ({time.time() - t0:.0f}s): " + " ".join(f"s{s} {cs_[i] / orc_s[i]:.3f}" for i, s in enumerate(seeds)), flush=True)
D = pd.DataFrame(rows); D.to_csv(out, index=False)
for s in seeds:
    g = D[D.seed == s]; f = g.iloc[-1]
    print(f"{fam} s{s} r{rep} lr {lr:g}: surrogate {g.cost_sur.iloc[0]:.3f} -> {f.cost_sur:.3f} (oracle {f.oracle_sur:.3f}, ratio {f.cost_sur / f.oracle_sur:.3f}, "
          f"reached 5 %: {f.cost_sur <= 1.05 * f.oracle_sur}); true {g.cost_true.iloc[0]:.3f} -> {f.cost_true:.3f} (oracle {f.oracle_true:.3f}); "
          f"last-500 relative drop {(g[g.epoch == epochs - 500].cost_sur.iloc[0] - f.cost_sur) / f.cost_sur:.4f}")
