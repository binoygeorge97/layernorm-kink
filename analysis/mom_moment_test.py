"""EXPLORATORY / POST HOC: does one exploded first step poison AdamW's moments (median of means warm start)?
Variants, 20 epochs, LN MSE seeds 2 and 8 (repeat-0 oracle, repeat-0 starts, N = 200, AdamW 1e-3, mom 10 groups):
  a  mom as in the harness
  b  mom, AdamW state re-initialised after epoch 1 (the weights keep the epoch-1 step)
  c  mom, the epoch-1 gradient rescaled to norm <= 3 before AdamW sees it
Logged per epoch: true-plant cost (training starts), norm of the mom gradient, cosine between the AdamW update and -gradient.
Output: out/opt/mom_moment_test.csv"""
import os
import sys

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
dt, tm, leaves = jnp.float64, jax.tree_util.tree_map, jax.tree_util.tree_leaves
N, EP = 200, 20


def fl(t):
    return jnp.concatenate([l.ravel() for l in leaves(t)])


def main():
    plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    Co, cs = DC.load_controllers(paths.DATA, "true plant (oracle)", 0)
    roll_t = DC._roller(DC.make_dynamics("true", plant=plant), N)
    rows = []
    for s in (2, 8):
        P, S0, _ = PC._stack_models(paths.DATA, "MSE (3000)", [s], dt)
        x0 = PC._training_starts(list(range(10)), 0, dt)[s:s + 1]
        p1, s01, X0 = tm(lambda a: a[0], P), S0[0], x0[0]
        dyn = DC.make_dynamics("surrogate", "layer", plant)
        grad = jax.jit(lambda c: DC.robust_grad(c, dyn, p1, s01, X0, N, "mom", 3.0, 10)[1])
        opt = optax.adamw(DC.CTRL_LR)
        upd_fn = jax.jit(opt.update)

        def true_cost(c):
            Xt, Ut = (np.asarray(a, float)[0, ..., 0] for a in roll_t(tm(lambda a: a[None], c), P, S0, x0))
            return DC.true_lqr_cost(Xt, Ut)
        for var in ("a", "b", "c"):
            c = tm(lambda a: jnp.asarray(np.asarray(a)[cs.index(s)], dt), Co)
            o = opt.init(c)
            c0 = true_cost(c)
            for e in range(1, EP + 1):
                g = grad(c)
                gn = float(jnp.linalg.norm(fl(g)))
                if var == "c" and e == 1 and gn > 3:
                    g = tm(lambda l: l * (3.0 / gn), g)
                u, o = upd_fn(g, o, c)
                cosu = float(jnp.dot(fl(u), -fl(g)) / (jnp.linalg.norm(fl(u)) * jnp.linalg.norm(fl(g)) + 1e-300))
                c = optax.apply_updates(c, u)
                if var == "b" and e == 1:
                    o = opt.init(c)
                rows.append(dict(seed=s, variant=var, epoch=e, grad_norm=gn, cos_update_negrad=cosu, cost0=c0,
                                 loss_true=true_cost(c) if e % 5 == 0 or e <= 2 else np.nan))
            r = [x for x in rows if x["seed"] == s and x["variant"] == var]
            print(f"LN MSE s{s} variant {var}: true cost {c0:.3f} -> {r[-1]['loss_true']:.3f} | epoch-1 grad {r[0]['grad_norm']:.2e} | "
                  f"cos(update, -grad) epochs 2-20 median {np.median([x['cos_update_negrad'] for x in r[1:]]):+.2f}", flush=True)
    pd.DataFrame(rows).to_csv(paths.out("opt", "mom_moment_test.csv"), index=False)


if __name__ == "__main__":
    main()
