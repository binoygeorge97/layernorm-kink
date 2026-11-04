"""R3 (2026-11-04, exploratory, not pre-registered): warm start at the repeat-0 oracle GRU, 1000 repeat-0 training starts,
N = 200, 100 epochs, on the 10 harness pairs, in three arms (each a separate, unvmapped run per pair):
  plain   AdamW lr 1e-3 (weight decay 1e-4), mean gradient; logs cos(g_e, g_{e-1}) for epochs 1-20 and the step length ||Delta_e||
  b1=0    AdamW lr 1e-3 with beta1 = 0 (no momentum), otherwise identical
  ngd     normalised gradient descent: theta <- theta - l_e g_e / ||g_e||, with l_e = ||Delta_e|| of the plain arm at the same
          epoch and pair (same per-step length as AdamW)
Every 5 epochs: true-plant and surrogate cost on the training starts (dpc_core.true_lqr_cost). Expelled = true cost > 1.10 x
epoch-0 value at any logged epoch 10-100 (PREREG_opt rule). Usage: python analysis/warmstart_arms.py [--pairs ...]
-> out/opt/warmstart_arms.csv (costs), out/opt/warmstart_arms_cos.csv (consecutive-gradient cosines, plain arm)."""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import level1_core as L, dpc_core as DC, phase_core as PC
from warmstart_harness import PAIRS, BROKEN
jax.config.update("jax_enable_x64", True)
dt, tm, leaves = jnp.float64, jax.tree_util.tree_map, jax.tree_util.tree_leaves
N, EPOCHS = 200, 100
nrm = lambda t: float(jnp.sqrt(sum(jnp.vdot(a, a) for a in leaves(t))))
dot = lambda a, b: float(sum(jnp.vdot(x, y) for x, y in zip(leaves(a), leaves(b))))


def main(pairs, out):
    plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    Co, cs = DC.load_controllers(paths.DATA, "true plant (oracle)", 0)
    roll_t = jax.jit(lambda c, x0: DC.rollout(c, DC.make_dynamics("true", plant=plant), None, None, x0, N))
    rows, cosrows = [], []
    for sur, s in pairs:
        norm = DC.surrogate_configs()[sur]["norm"]
        P, S0, _ = PC._stack_models(paths.DATA, sur, [s], dt)
        p, s0 = tm(lambda a: a[0], P), S0[0]
        x0 = PC._training_starts(list(range(10)), 0, dt)[s]
        c0 = tm(lambda a: jnp.asarray(np.asarray(a)[cs.index(s)], dt), Co)
        dyn = DC.make_dynamics("surrogate", norm, plant)
        vg = jax.jit(jax.value_and_grad(lambda c: DC.dpc_loss(c, dyn, p, s0, x0, N)))
        rs = jax.jit(lambda c: DC.rollout(c, dyn, p, s0, x0, N))

        def costs(c):
            Xt, Ut = (np.asarray(a, float)[..., 0] for a in roll_t(c, x0))
            Xs, Us = (np.asarray(a, float)[..., 0] for a in rs(c))
            return DC.true_lqr_cost(Xt, Ut), DC.true_lqr_cost(Xs, Us)
        lens = {}
        for arm in ("plain", "b1=0", "ngd"):
            t0 = time.time(); c = c0
            opt = optax.adamw(DC.CTRL_LR, b1=0.0) if arm == "b1=0" else optax.adamw(DC.CTRL_LR)
            o = opt.init(c); upd = jax.jit(opt.update); gprev = None
            ct, cs_ = costs(c); rows.append(dict(arm=arm, sur=sur, seed=s, broken=(sur, s) in BROKEN, epoch=0, loss_true=ct, loss_sur=cs_))
            for e in range(1, EPOCHS + 1):
                _, g = vg(c)
                if arm == "ngd":
                    gn = nrm(g); step = tm(lambda a: -lens[e] * a / gn, g)
                else:
                    step, o = upd(g, o, c)
                if arm == "plain":
                    lens[e] = nrm(step)
                    if gprev is not None and e <= 21:
                        cosrows.append(dict(sur=sur, seed=s, broken=(sur, s) in BROKEN, epoch=e - 1,
                                            cos_consecutive=dot(g, gprev) / max(nrm(g) * nrm(gprev), 1e-300), grad_norm=nrm(g)))
                    gprev = g
                c = optax.apply_updates(c, step)
                if e % 5 == 0:
                    ct, cs_ = costs(c)
                    rows.append(dict(arm=arm, sur=sur, seed=s, broken=(sur, s) in BROKEN, epoch=e, loss_true=ct, loss_sur=cs_))
            D = pd.DataFrame(rows); q = D[(D.arm == arm) & (D.sur == sur) & (D.seed == s)]
            c0v = q.loss_true.iloc[0]; mx = q[q.epoch >= 10].loss_true.max()
            print(f"{arm:6s} {sur:22s} s{s}: true {c0v:.3f} -> {q.loss_true.iloc[-1]:.3f} (max e10-100 {mx:.3f}) "
                  f"{'EXPELLED' if mx > 1.10 * c0v else 'stays'} | {time.time() - t0:.0f}s", flush=True)
            D.to_csv(out, index=False); pd.DataFrame(cosrows).to_csv(out.replace(".csv", "_cos.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=None)
    ap.add_argument("--out", default=paths.out("opt", "warmstart_arms.csv")); a = ap.parse_args()
    pairs = PAIRS if a.pairs is None else [(q.rsplit(":", 1)[0], int(q.rsplit(":", 1)[1])) for q in a.pairs.split(",")]
    main(pairs, a.out)
