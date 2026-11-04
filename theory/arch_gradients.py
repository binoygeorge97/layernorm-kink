"""R2 (c, d) (2026-11-04, post hoc): the DPC gradient at the MEMORYLESS oracle (MLP or linear, trained through the true plant,
same repeat) through each surrogate, N = 200, the 1000 training starts of the repeat: norm of the batch gradient
(mean loss), per-start gradient norms, the largest single-start share of the squared per-start norms (sum over starts of
||g_i||^2 as denominator, as t2_test grad_top_share), number of starts with ||g_i|| > 1e3, and the run's Lambda*
(t2_arch CSV). Usage (from theory/): python arch_gradients.py mlp|linear [families]  -> $T2_OUT/arch_gradients_<arch>.csv"""
import sys, os, time
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
from t2_arch import pi, load_ref, OUT
import paths

arch = sys.argv[1]
fams = sys.argv[2].split(",") if len(sys.argv) > 2 else ["LN MSE 3000", "LN slope from 0", "LN two-stage"]
root = os.path.join(paths.REPO, "data", f"dpc_out_{arch}")


def make(norm):
    def roll(c, p, s0, x0):
        def step(x, _):
            u = pi(c, x)
            return T.F_point(p, x, u, s0, norm), (x, u)
        xN, (xs, us) = jax.lax.scan(step, x0, None, length=T.N_HOR)
        return jnp.append(xs, xN), us

    def one(c, p, s0, x0):
        xs, us = roll(c, p, s0, x0)
        return (jnp.sum(T.Q_X * xs[:-1] ** 2 + T.R_U * us ** 2) + T.Q_F * xs[-1] ** 2) / T.N_HOR
    gper = jax.jit(jax.vmap(lambda c, p, s0, x0: T.tree_norm(jax.grad(one)(c, p, s0, x0)), in_axes=(None, None, None, 0)))
    gb = jax.jit(jax.grad(lambda c, p, s0, X: jnp.mean(jax.vmap(lambda x0: one(c, p, s0, x0))(X))))
    return gper, gb


L = pd.read_csv(os.path.join(OUT, f"t2_arch_{arch}.csv"))
rows = []
for fam in fams:
    P, S0, norm, seeds, met = T.load_surrogate(fam)
    gper, gb = make(norm)
    for rep in (0, 1, 2):
        C, cs = load_ref(root, rep)
        for i, s in enumerate(seeds):
            t0 = time.time()
            p, s0 = T.seed_slice(P, i), S0[i]
            c = jax.tree_util.tree_map(lambda a: jnp.asarray(a[cs.index(s)]), C)
            X0 = T.training_starts(s, rep)
            g = np.asarray(gper(c, p, s0, X0)); gn = float(T.tree_norm(gb(c, p, s0, X0)))
            g2 = g ** 2
            lam = L[(L.family == fam) & (L.seed == s) & (L.rep == rep)].Lambda_star
            rows.append(dict(arch=arch, family=fam, seed=s, rep=rep, grad_batch=gn, grad_start_max=float(np.nanmax(g)),
                             grad_start_median=float(np.nanmedian(g)), top_share=float(np.nanmax(g2) / np.nansum(g2)),
                             n_gt_1e3=int((g > 1e3).sum()), n_gt_3=int((g > 3).sum()), n_nonfinite=int((~np.isfinite(g)).sum()),
                             Lambda_star=float(lam.iloc[0]) if len(lam) else np.nan))
            print(f"{arch} {fam:16s} s{s} r{rep}: |g| {gn:.2e} max start {np.nanmax(g):.2e} top share {rows[-1]['top_share']:.3f} "
                  f"n>1e3 {rows[-1]['n_gt_1e3']} | Lambda* {rows[-1]['Lambda_star']:.2f} | {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, f"arch_gradients_{arch}.csv"), index=False)
