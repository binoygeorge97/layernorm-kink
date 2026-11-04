"""B2 (iii) (2026-11-04, exploratory, not pre-registered): is the AdamW step too large for the surrogate loss surface?
At the good controller (repeat-0 oracle GRU, epoch 0) and at epochs 1, 5, 20 of the PLAIN warm start (AdamW lr 1e-3, mean
gradient, 1000 repeat-0 training starts, N = 200), take the actual AdamW update Delta that the optimiser would apply at
that state (optax.adamw update from the current moments and the current mean gradient g). Along it, evaluate the DPC
training loss through the surrogate at fractions a of the full step, a = 1e-4 ... 1 (17 points, log grid), and compare
the actual change L(theta + a Delta) - L(theta) with the first-order prediction a g . Delta. Report the smallest a where
|actual - predicted| / |predicted| > 0.5 ("parting fraction"; inf if they never part on the grid).
Usage: python analysis/step_scale.py [--pairs ...]  -> out/opt/step_scale.csv"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import level1_core as L, dpc_core as DC, phase_core as PC
from warmstart_harness import PAIRS, BROKEN

jax.config.update("jax_enable_x64", True)
dt, tm = jnp.float64, jax.tree_util.tree_map
N = 200
AT = (0, 1, 5, 20)
FR = np.logspace(-4, 0, 17)


def main(pairs, out):
    plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    seeds = list(range(10))
    Co, cs = DC.load_controllers(paths.DATA, "true plant (oracle)", 0)
    rows = []
    for sur, s in pairs:
        t0 = time.time()
        norm = DC.surrogate_configs()[sur]["norm"]
        P, S0, _ = PC._stack_models(paths.DATA, sur, [s], dt)
        p, s0 = tm(lambda a: a[0], P), S0[0]
        x0 = PC._training_starts(seeds, 0, dt)[s]
        c = tm(lambda a: jnp.asarray(np.asarray(a)[cs.index(s)], dt), Co)
        dyn = DC.make_dynamics("surrogate", norm, plant)
        lossf = jax.jit(lambda c: DC.dpc_loss(c, dyn, p, s0, x0, N))
        vg = jax.jit(jax.value_and_grad(lambda c: DC.dpc_loss(c, dyn, p, s0, x0, N)))
        opt = optax.adamw(DC.CTRL_LR)
        o = opt.init(c)
        upd_f = jax.jit(opt.update)
        e = 0
        for target in AT:
            while e < target:                                    # plain AdamW steps up to the target epoch
                _, g = vg(c); u, o = upd_f(g, o, c); c = optax.apply_updates(c, u); e += 1
            L0, g = vg(c)
            u, _ = upd_f(g, o, c)                                # the update AdamW would apply now (state not advanced)
            gd = float(sum(jnp.vdot(a, b) for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(u))))
            gn = float(jnp.sqrt(sum(jnp.vdot(a, a) for a in jax.tree_util.tree_leaves(g))))
            un = float(jnp.sqrt(sum(jnp.vdot(a, a) for a in jax.tree_util.tree_leaves(u))))
            part = np.inf
            for a in FR:
                La = float(lossf(tm(lambda x, d: x + a * d, c, u)))
                act, pred = La - float(L0), a * gd
                rel = abs(act - pred) / max(abs(pred), 1e-300)
                if rel > 0.5 and not np.isfinite(part):
                    part = a
                rows.append(dict(sur=sur, seed=s, broken=(sur, s) in BROKEN, epoch=target, frac=a, L0=float(L0), L=La,
                                 actual=act, predicted=pred, rel_err=rel, grad_norm=gn, step_norm=un,
                                 cos_step_neggrad=-gd / max(gn * un, 1e-300)))
            print(f"{sur:22s} s{s} epoch {target:2d}: L {float(L0):.4f} |g| {gn:.2e} |step| {un:.2e} cos(step,-g) {-gd / max(gn * un, 1e-300):+.3f} "
                  f"| parting fraction {part:.2e}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"   {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out", default=paths.out("opt", "step_scale.csv"))
    a = ap.parse_args()
    pairs = PAIRS if a.pairs is None else [(q.rsplit(":", 1)[0], int(q.rsplit(":", 1)[1])) for q in a.pairs.split(",")]
    main(pairs, a.out)
