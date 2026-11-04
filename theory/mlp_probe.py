"""MLP mechanism (2026-11-04, POST HOC), probes at the MLP oracle through the surrogate (N = 200, the repeat's 1000 training
starts). Why is the warm-started MLP expelled although its gradient is tame?
(1) Direction: cosine between the batch gradient at c and at c * (1 + 1e-4 xi), xi ~ N(0, 1) per weight (5 draws, key
    50000 + seed), as the GRU check (healthy GRU pairs: 1.000).
(2) First Adam step: AdamW's first update is lr * sign(g) per weight (bias-corrected m / sqrt(v)). Change of the loss
    after that step for lr = 1e-3 and 1e-4, against the first-order prediction -lr * ||g||_1. A rise means the first step
    already overshoots (curvature along the step >> gradient), a stiffness effect rather than a chaotic gradient.
(3) Curvature along the Adam direction d = -sign(g): second difference of the loss at h = 1e-5 / ||d||, i.e. d^T H d / ||d||^2.
Usage (from theory/): python mlp_probe.py [pairs] [rep]   -> $T2_OUT/mlp_probe.csv"""
import sys, os
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
from t2_arch import load_ref, OUT
from mlp_line import make, ROOT_O, DEFAULT
import paths

jax.config.update("jax_enable_x64", True)
tm = jax.tree_util.tree_map
pairs = [(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).split(",")]
rep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
flat = lambda t: jnp.concatenate([jnp.ravel(a) for a in jax.tree_util.tree_leaves(t)])
Co, cso = load_ref(ROOT_O, rep)
rows = []
for fam in dict.fromkeys(f for f, _ in pairs):
    P, S0, norm, seeds, met = T.load_surrogate(fam)
    gper, gb, ls, lt = make(norm)
    for f, s in pairs:
        if f != fam:
            continue
        i = seeds.index(s); p, s0 = T.seed_slice(P, i), S0[i]
        c = tm(lambda a: jnp.asarray(a[cso.index(s)]), Co)
        X0 = T.training_starts(s, rep)
        g = gb(c, p, s0, X0); gf = flat(g); L0 = float(ls(c, p, s0, X0))
        cos = []
        for k in range(5):
            leaves, tdef = jax.tree_util.tree_flatten(c)
            ks = jax.random.split(jax.random.PRNGKey(50_000 + 100 * k + s), len(leaves))
            cp = jax.tree_util.tree_unflatten(tdef, [a * (1 + 1e-4 * jax.random.normal(kk, a.shape, a.dtype)) for a, kk in zip(leaves, ks)])
            gp = flat(gb(cp, p, s0, X0))
            cos.append(float(jnp.dot(gf, gp) / (jnp.linalg.norm(gf) * jnp.linalg.norm(gp))))
        r = dict(family=fam, seed=s, rep=rep, loss=L0, grad=float(jnp.linalg.norm(gf)), grad_l1=float(jnp.sum(jnp.abs(gf))),
                 cos_min=min(cos), cos_med=float(np.median(cos)), n_params=int(gf.size))
        d = tm(lambda a: -jnp.sign(a), g)
        for lr in (1e-3, 1e-4):
            c1 = tm(lambda a, b: a + lr * b, c, d)
            r[f"dL_step_{lr:g}"] = float(ls(c1, p, s0, X0)) - L0
            r[f"dL_pred_{lr:g}"] = -lr * r["grad_l1"]
        h = 1e-5 / float(jnp.linalg.norm(flat(d)))
        lp = float(ls(tm(lambda a, b: a + h * b, c, d), p, s0, X0)); lm = float(ls(tm(lambda a, b: a - h * b, c, d), p, s0, X0))
        r["curv_dir"] = (lp - 2 * L0 + lm) / (h ** 2 * float(jnp.sum(flat(d) ** 2)))
        rows.append(r)
        print(f"{fam:12s} s{s}: loss {L0:.3f} |g| {r['grad']:.3g} cos min/med {r['cos_min']:.4f}/{r['cos_med']:.4f} | first Adam step "
              f"lr 1e-3: dL {r['dL_step_0.001']:+.4f} (1st order {r['dL_pred_0.001']:+.4f}); lr 1e-4: {r['dL_step_0.0001']:+.5f} "
              f"({r['dL_pred_0.0001']:+.5f}) | curvature along sign step {r['curv_dir']:.3g}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "mlp_probe.csv"), index=False)
