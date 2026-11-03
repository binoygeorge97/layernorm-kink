"""EXPLORATORY / POST HOC (not pre-registered): why did median of means fail to keep the good controller (PREREG_opt P3b)?

Re-runs the 'mom' warm start of analysis/warmstart_harness.py (repeat-0 oracle GRU, 1000 training starts of repeat 0,
N = 200, AdamW lr 1e-3, 100 epochs, the update uses the median-of-means gradient exactly as dpc_core.robust_grad 'mom')
on the 6 broken pairs, and logs every epoch:
  group_norm_0..9  norm of each of the 10 group-mean gradients (groups = contiguous blocks of 100 starts)
  n_groups_10x     groups whose norm exceeds 10 x the median group norm
  top_start_norm, top_start_group, n_starts_gt3, n_starts_gt1e3   per-start gradient norms
  cos_mom_mean     cosine between the median-of-means direction and the plain mean gradient
  cos_mom_clean    cosine with the mean over the groups NOT exceeding 10 x the median group norm
  cos_mom_clip3    cosine with the per-start clip-3 aggregate (the setting that keeps the controller)
  cos_clean_clip3, norm_mom, norm_clean, norm_clip3, loss_true (every 5 epochs), loss_sur
Question: did many groups explode at once, or does the coordinate-wise median give a biased direction although only one
group explodes?   Output: out/opt/mom_diagnose.csv"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
import level1_core as L  # noqa: E402
import dpc_core as DC  # noqa: E402
import phase_core as PC  # noqa: E402

jax.config.update("jax_enable_x64", True)
dt, tm = jnp.float64, jax.tree_util.tree_map
N, G, EPOCHS = 200, 10, 100
PAIRS = [("MSE (3000)", 1), ("MSE (3000)", 2), ("MSE (3000)", 8), ("MSE (3000)", 9), ("J_x + J_u (3000)", 2),
         ("J_x + J_u (3000)", 9)]


def flat(tree):
    return jnp.concatenate([l.reshape(l.shape[0], -1) for l in jax.tree_util.tree_leaves(tree)], 1)


def cos(a, b):
    return jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-300)


def main(pairs=PAIRS):
    plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    Co, cs = DC.load_controllers(paths.DATA, "true plant (oracle)", 0)
    roll_t = DC._roller(DC.make_dynamics("true", plant=plant), N)
    rows = []
    for sur, s in pairs:
        t0 = time.time()
        norm = DC.surrogate_configs()[sur]["norm"]
        P, S0, _ = PC._stack_models(paths.DATA, sur, [s], dt)
        x0 = PC._training_starts(list(range(10)), 0, dt)[s:s + 1]            # [1, 1000, 1]
        p1, s01, X0 = tm(lambda a: a[0], P), S0[0], x0[0]
        c = tm(lambda a: jnp.asarray(np.asarray(a)[cs.index(s)], dt), Co)
        dyn = DC.make_dynamics("surrogate", norm, plant)
        roll_s = DC._roller(dyn, N)
        opt = optax.adamw(DC.CTRL_LR)
        o = opt.init(c)
        treedef = jax.tree_util.tree_structure(c)

        @jax.jit
        def step(c, o):
            one = lambda cc, x: DC.dpc_loss(cc, dyn, p1, s01, x[None], N)
            losses, Gt = jax.vmap(jax.value_and_grad(one), in_axes=(None, 0))(c, X0)
            Gf = flat(Gt)                                                     # [B, P]
            B = Gf.shape[0]
            gm = jnp.mean(Gf.reshape(G, B // G, -1), 1)                       # [G, P]
            gn = jnp.linalg.norm(gm, axis=1)
            med = jnp.median(gn)
            big = gn > 10 * med
            mom = jnp.median(gm, 0)
            mean = jnp.mean(Gf, 0)
            w = (~big).astype(Gf.dtype)
            clean = (w[:, None] * gm).sum(0) / jnp.maximum(w.sum(), 1)
            sn = jnp.linalg.norm(Gf, axis=1)
            clip = jnp.mean(Gf * jnp.minimum(1.0, 3.0 / (sn + 1e-30))[:, None], 0)
            # the update: median of means per leaf, exactly as dpc_core.robust_grad('mom')
            gtree = tm(lambda g: jnp.median(jnp.mean(g.reshape((G, B // G) + g.shape[1:]), 1), 0), Gt)
            upd, o2 = opt.update(gtree, o, c)
            stats = dict(gn=gn, n10=big.sum(), top=sn.max(), top_group=jnp.argmax(sn) // (B // G),
                         gt3=(sn > 3).sum(), gt1e3=(sn > 1e3).sum(), cos_mean=cos(mom, mean), cos_clean=cos(mom, clean),
                         cos_clip=cos(mom, clip), cos_clean_clip=cos(clean, clip), n_mom=jnp.linalg.norm(mom),
                         n_clean=jnp.linalg.norm(clean), n_clip=jnp.linalg.norm(clip), loss=jnp.mean(losses))
            return optax.apply_updates(c, upd), o2, stats

        def true_cost(c):
            cb = tm(lambda a: a[None], c)
            Xt, Ut = (np.asarray(a, float)[0, ..., 0] for a in roll_t(cb, P, S0, x0))
            return DC.true_lqr_cost(Xt, Ut)
        c0cost = true_cost(c)
        for e in range(EPOCHS):
            c, o, st = step(c, o)
            st = {k: np.asarray(v) for k, v in st.items()}
            row = dict(sur=sur, seed=s, epoch=e + 1, n_groups_10x=int(st["n10"]), top_start_norm=float(st["top"]),
                       top_start_group=int(st["top_group"]), n_starts_gt3=int(st["gt3"]), n_starts_gt1e3=int(st["gt1e3"]),
                       cos_mom_mean=float(st["cos_mean"]), cos_mom_clean=float(st["cos_clean"]),
                       cos_mom_clip3=float(st["cos_clip"]), cos_clean_clip3=float(st["cos_clean_clip"]),
                       norm_mom=float(st["n_mom"]), norm_clean=float(st["n_clean"]), norm_clip3=float(st["n_clip"]),
                       loss_sur=float(st["loss"]), loss_true=np.nan, loss_true_0=c0cost,
                       **{f"group_norm_{i}": float(v) for i, v in enumerate(st["gn"])})
            if (e + 1) % 5 == 0:
                row["loss_true"] = true_cost(c)
            rows.append(row)
        g = pd.DataFrame([r for r in rows if r["sur"] == sur and r["seed"] == s])
        print(f"{sur:18s} s{s}: true cost {c0cost:.3f} -> {g.loss_true.dropna().iloc[-1]:.3f} | groups >10x median: median "
              f"{g.n_groups_10x.median():.0f}, max {g.n_groups_10x.max()} | starts >1e3: median {g.n_starts_gt1e3.median():.0f} | "
              f"cos(mom, mean) {g.cos_mom_mean.median():+.2f}, (mom, clean) {g.cos_mom_clean.median():+.2f}, "
              f"(mom, clip3) {g.cos_mom_clip3.median():+.2f}, (clean, clip3) {g.cos_clean_clip3.median():+.2f} | "
              f"{time.time() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(paths.out("opt", "mom_diagnose.csv"), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
