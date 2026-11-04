"""MLP mechanism (2026-11-04, POST HOC): the GRU checks of the trap mechanism repeated for the MLP controller.
Along the straight line c(alpha) = (1 - alpha) c_DPC + alpha c_oracle from the MLP DPC controller (alpha = 0) to the MLP oracle
(alpha = 1, trained through the true plant, same repeat), through the surrogate the DPC controller was trained on, N = 200,
the repeat's 1000 training starts: the DPC loss through the surrogate (what training descends), the same loss through the
true plant, the batch-gradient norm, per-start gradient norms (max, median, number > 1e3) and the largest single-start share
of sum_i ||g_i||^2 (as t2_test grad_top_share). alpha grid: 41 uniform points plus 20 extra on [0, 0.05] and [0.95, 1].
Usage (from theory/): python mlp_line.py [pairs] [rep]   pairs e.g. "LN MSE 3000:1,LN two-stage:9"
-> $T2_OUT/mlp_line.csv (one row per pair and alpha) and a printed summary per pair."""
import sys, os, time
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
from t2_arch import pi, load_ref, OUT
import dpc_core as DC
import paths

jax.config.update("jax_enable_x64", True)
DEFAULT = "LN MSE 3000:1,LN MSE 3000:2,LN MSE 3000:8,LN MSE 3000:9,LN MSE 3000:3,LN MSE 3000:7,LN MSE 3000:4,LN two-stage:2,LN two-stage:9"
STEM = {"LN MSE 3000": "LN, MSE (3000)", "LN two-stage": "LN, two-stage"}
ROOT_O = os.path.join(paths.REPO, "data", "dpc_out_mlp")          # MLP oracles
ROOT_D = os.path.join(paths.REPO, "data", "run5_mlp")             # MLP DPC controllers (Run 5a)
ALPHAS = np.unique(np.round(np.concatenate([np.linspace(0, 1, 41), np.linspace(0, 0.05, 11), np.linspace(0.95, 1, 11)]), 6))


def make(norm):
    def roll(c, p, s0, x0, true):
        def step(x, _):
            u = pi(c, x)
            xn = T.A_TRUE * x + T.B_TRUE * u if true else T.F_point(p, x, u, s0, norm)
            return xn, (x, u)
        xN, (xs, us) = jax.lax.scan(step, x0, None, length=T.N_HOR)
        return jnp.append(xs, xN), us

    def one(c, p, s0, x0, true=False):
        xs, us = roll(c, p, s0, x0, true)
        return (jnp.sum(T.Q_X * xs[:-1] ** 2 + T.R_U * us ** 2) + T.Q_F * xs[-1] ** 2) / T.N_HOR
    mean = lambda c, p, s0, X, true=False: jnp.mean(jax.vmap(lambda x0: one(c, p, s0, x0, true))(X))
    gper = jax.jit(jax.vmap(lambda c, p, s0, x0: T.tree_norm(jax.grad(one)(c, p, s0, x0)), in_axes=(None, None, None, 0)))
    gb = jax.jit(jax.grad(mean))
    ls = jax.jit(lambda c, p, s0, X: mean(c, p, s0, X, False))
    lt = jax.jit(lambda c, p, s0, X: mean(c, p, s0, X, True))
    return gper, gb, ls, lt


def load_dpc(fam, rep):
    z = np.load(DC.rep_file(os.path.join(ROOT_D, "controllers"), STEM[fam], rep))
    return DC._unflat({k[2:]: z[k] for k in z.files if k.startswith("c:")}), [int(s) for s in z["seeds"]]


if __name__ == "__main__":
    pairs = [(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).split(",")]
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rows = []
    Co, cso = load_ref(ROOT_O, rep)
    for fam in dict.fromkeys(f for f, _ in pairs):
        P, S0, norm, seeds, met = T.load_surrogate(fam)
        gper, gb, ls, lt = make(norm)
        Cd, csd = load_dpc(fam, rep)
        for f, s in pairs:
            if f != fam:
                continue
            t0 = time.time()
            i = seeds.index(s); p, s0 = T.seed_slice(P, i), S0[i]
            co = jax.tree_util.tree_map(lambda a: jnp.asarray(a[cso.index(s)]), Co)
            cd = jax.tree_util.tree_map(lambda a: jnp.asarray(a[csd.index(s)]), Cd)
            X0 = T.training_starts(s, rep)
            for a in ALPHAS:
                ca = jax.tree_util.tree_map(lambda u, v: (1 - a) * u + a * v, cd, co)
                g = np.asarray(gper(ca, p, s0, X0)); g2 = g ** 2
                rows.append(dict(family=fam, seed=s, rep=rep, alpha=float(a), loss_sur=float(ls(ca, p, s0, X0)),
                                 loss_true=float(lt(ca, p, s0, X0)), grad_batch=float(T.tree_norm(gb(ca, p, s0, X0))),
                                 grad_start_max=float(np.nanmax(g)), grad_start_median=float(np.nanmedian(g)),
                                 top_share=float(np.nanmax(g2) / np.nansum(g2)), n_gt_1e3=int((g > 1e3).sum()),
                                 n_nonfinite=int((~np.isfinite(g)).sum())))
            d = pd.DataFrame([r for r in rows if r["family"] == fam and r["seed"] == s])
            l0, l1, lmax = d.loss_sur.iloc[0], d.loss_sur.iloc[-1], d.loss_sur.max()
            print(f"{fam:12s} s{s} r{rep}: surrogate loss DPC {l0:.3f} -> oracle {l1:.3f}, max on line {lmax:.3f} at alpha "
                  f"{d.alpha[d.loss_sur.idxmax()]:.3f} (barrier {lmax - max(l0, l1):+.3f}); true loss {d.loss_true.iloc[0]:.3f} -> "
                  f"{d.loss_true.iloc[-1]:.3f}; grad max {d.grad_batch.max():.3g} at alpha {d.alpha[d.grad_batch.idxmax()]:.3f}, "
                  f"at DPC {d.grad_batch.iloc[0]:.3g}, at oracle {d.grad_batch.iloc[-1]:.3g}; top share max {d.top_share.max():.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "mlp_line.csv"), index=False)
