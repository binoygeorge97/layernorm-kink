"""t2_memoryless.py - Theorem 2 with an exactly memoryless good policy, and the neighbourhood of the GRU reference.

Part A (memoryless, exact theory): the policy u = 10 tanh((k x + b) / 10) with k = -K*, K* = 10.2875 (the analytic LQR
gain of the true plant), b = 0, i.e. dpc_core's 'linear' controller at the LQR gain. On the true plant it is a good
policy. Through every surrogate (6 families x 10 seeds) and every repeat's 1000 training starts: multipliers
m_k = J_x_hat - J_u_hat K_k (K_k = -du/dx, exact, no memory), sum log|m_k| vs exact dx_N/dx_0, landing statistic S_seg,
exact costate, and the gradient of the DPC loss with respect to (k, b).
Part B (neighbourhood): for the 30 LN MSE 3000 runs, the GRU reference controller with every weight multiplied by
(1 + 1e-3 xi), xi ~ N(0, 1) (about one AdamW step, lr 1e-3), 8 draws: batch gradient norm of the surrogate DPC loss.
"""
import sys, time, os
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
from t2_test import max_subarray, costate, X_LAND, labels, OUT

K_STAR = T.lqr_gain_true()          # 10.2875 at the reference plant


def lin_u(theta, x):
    return T.U_BOUND * jnp.tanh((theta[0] * x + theta[1]) / T.U_BOUND)


def make_lin(norm):
    def roll(theta, p, s0, x0):
        def step(x, _):
            u = lin_u(theta, x)
            return T.F_point(p, x, u, s0, norm), (x, u)
        xN, (xs, us) = jax.lax.scan(step, x0, None, length=T.N_HOR)
        return jnp.append(xs, xN), us

    def per_start(theta, p, s0, x0):
        xs, us = roll(theta, p, s0, x0)
        _, jx, ju = jax.vmap(T.slopes(p, s0, norm))(xs[:-1], us)
        K = -jax.vmap(jax.grad(lambda x: lin_u(theta, x)))(xs[:-1])
        sens = jax.jvp(lambda a: roll(theta, p, s0, a)[0][-1], (x0,), (jnp.ones_like(x0),))[1]
        return xs, us, jx, ju, K, jx - ju * K, sens

    def loss(theta, p, s0, X0):
        def one(x0):
            xs, us = roll(theta, p, s0, x0)
            return (jnp.sum(T.Q_X * xs[:-1] ** 2 + T.R_U * us ** 2) + T.Q_F * xs[-1] ** 2) / T.N_HOR
        return jnp.mean(jax.vmap(one)(X0))
    return jax.jit(jax.vmap(per_start, in_axes=(None, None, None, 0))), jax.jit(jax.grad(loss))


def part_a():
    R = labels()
    rows = []
    theta = jnp.array([-K_STAR, 0.0])
    for fam in T.FAMILIES:
        P, S0, norm, seeds, met = T.load_surrogate(fam)
        ps, g = make_lin(norm)
        for rep in (0, 1, 2):
            for i, s in enumerate(seeds):
                p, s0, X0 = T.seed_slice(P, i), S0[i], T.training_starts(s, rep)
                xs, us, jx, ju, K, m, sens = (np.asarray(a) for a in ps(theta, p, s0, X0))
                logm = np.log(np.abs(m) + 1e-300)
                S_seg, S_all, lam = [], [], []
                for b in range(len(X0)):
                    near = np.abs(xs[b, :-1]) <= X_LAND
                    kl = int(np.argmax(near)) if near.any() else T.N_HOR
                    S_seg.append(max_subarray(logm[b, kl:]))
                    S_all.append(logm[b].sum())
                    lam.append(np.max(np.abs(costate(xs[b], us[b], K[b], m[b]))))
                S_all = np.array(S_all)
                exact = np.log(np.abs(sens) + 1e-300)
                gn = float(np.linalg.norm(np.asarray(g(theta, p, s0, X0))))
                r = R[(R.model == T.RESULTS_NAME[fam]) & (R.seed == s) & (R.rep == rep)].iloc[0]
                rows.append(dict(family=fam, rep=rep, seed=s, gap=r.gap, own_on_model=r.cost_test_own,
                                 S_seg_max=float(np.max(S_seg)), S_all_max=float(S_all.max()),
                                 factorisation_err=float(np.max(np.abs(S_all - exact))),
                                 lam_max=float(np.max(lam)), grad_lin=gn,
                                 diverged=float((np.abs(xs).max(1) > T.DIVERGE if hasattr(T, "DIVERGE") else np.abs(xs).max(1) > 10).mean())))
                print(f"A {fam:16s} rep {rep} seed {s}: gap {100*r.gap:9.3g}% | S_seg {rows[-1]['S_seg_max']:7.2f} | "
                      f"sum log|m| - log|dx_N/dx_0| max {rows[-1]['factorisation_err']:.1e} | lam {rows[-1]['lam_max']:9.3g} | "
                      f"grad(k,b) {gn:9.3g}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/t2_memoryless.csv", index=False)


def part_b(n_draw=8, rel=1e-3):
    from t2_test import make_fns
    P, S0, norm, seeds, met = T.load_surrogate("LN MSE 3000")
    ps, gb, gs, lv = make_fns(norm)
    rows = []
    for rep in (0, 1, 2):
        Co, cs = T.load_controllers("true_plant__oracle", rep)
        for i, s in enumerate(seeds):
            t0 = time.time()
            p, s0, c = T.seed_slice(P, i), S0[i], T.seed_slice(Co, cs.index(s))
            X0 = T.training_starts(s, rep)
            g0 = float(T.tree_norm(gb(c, p, s0, X0)))
            gn = []
            for d in range(n_draw):
                leaves, tdef = jax.tree_util.tree_flatten(c)
                keys = jax.random.split(jax.random.PRNGKey(777 + 100 * rep + 10 * s + d), len(leaves))
                pert = [l * (1 + rel * jax.random.normal(k, l.shape, l.dtype)) for l, k in zip(leaves, keys)]
                cp = jax.tree_util.tree_unflatten(tdef, pert)
                gn.append(float(T.tree_norm(gb(cp, p, s0, X0))))
            rows.append(dict(rep=rep, seed=s, grad_ref=g0, grad_nbhd_max=float(np.max(gn)),
                             grad_nbhd_med=float(np.median(gn)), grad_nbhd_min=float(np.min(gn))))
            print(f"B rep {rep} seed {s}: grad at reference {g0:9.3g} | neighbourhood (rel {rel:g}, {n_draw} draws) "
                  f"min {np.min(gn):9.3g} median {np.median(gn):9.3g} max {np.max(gn):9.3g} | {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/t2_neighbourhood.csv", index=False)


if __name__ == "__main__":
    if "A" in sys.argv[1:]:
        part_a()
    if "B" in sys.argv[1:]:
        part_b()
