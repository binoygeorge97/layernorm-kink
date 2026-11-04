"""A4 (Run 5 scoring, 2026-11-04): DPC loss of the linear policy u = 10 tanh(-K x / 10) (b = 0) against K through each
surrogate and through the true plant (N = 200, the repeat-0 training starts of each seed, dpc_core loss), the argmin
through the surrogate vs through the true plant. Same quantity as notebook figure D10, recomputed locally.
Usage: python analysis/linear_landscape.py [families]  -> out/arch/linear_landscape.csv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import level1_core as L, dpc_core as DC, phase_core as PC
plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
FAM = {"LN MSE 3000": "MSE (3000)", "LN two-stage": "J_x + J_u (two-stage)", "static norm": "static norm, MSE"}
fams = sys.argv[1].split(",") if len(sys.argv) > 1 else list(FAM)
Ks = jnp.asarray(np.round(np.arange(6, 30.01, 0.25), 2))
seeds = list(range(10))
X0all = np.asarray(PC._training_starts(seeds, 0))[..., 0]          # [10, 1000]

def loss_curve(F, x0):
    def step(x, _):
        u = 10 * jnp.tanh(-Ks[:, None] * x / 10)
        xn = F(x, u)
        return xn, (5 * x ** 2 + 0.1 * u ** 2).mean(1)
    xN, st = jax.lax.scan(step, jnp.broadcast_to(x0, (len(Ks), len(x0))), None, length=200)
    return (st.sum(0) + (50 * xN ** 2).mean(1)) / 200

true_F = jax.jit(lambda x0: loss_curve(lambda x, u: plant.A * x + plant.B * u, x0))
rows = []
for fam in fams:
    sur = FAM[fam]; norm = DC.surrogate_configs()[sur]["norm"]
    P, S0, _ = PC._stack_models(paths.DATA, sur, seeds)
    for s in seeds:
        p = jax.tree_util.tree_map(lambda a: a[s], P); s0 = S0[s]
        Fb = jax.vmap(jax.vmap(lambda x, u: L.F_point(p, x, u, s0, norm)))
        lt = np.asarray(true_F(jnp.asarray(X0all[s]))); ls = np.asarray(jax.jit(lambda x0: loss_curve(Fb, x0))(jnp.asarray(X0all[s])))
        kt, ks = float(Ks[np.argmin(lt)]), float(Ks[np.nanargmin(np.where(np.isfinite(ls), ls, np.inf))])
        rows.append(dict(family=fam, seed=s, K_true=kt, K_sur=ks, rel=ks / kt - 1, loss_sur_at_Ktrue=float(ls[np.argmin(lt)]),
                         loss_true_min=float(lt.min()), loss_sur_min=float(np.nanmin(ls))))
        print(f"{fam:12s} seed {s}: argmin K true {kt:5.2f} | surrogate {ks:5.2f} ({100 * (ks / kt - 1):+.1f} %)", flush=True)
pd.DataFrame(rows).to_csv(paths.out("arch", "linear_landscape.csv"), index=False)
