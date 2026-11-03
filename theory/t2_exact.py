"""EXPLORATORY (not pre-registered): exact-memory landing amplification Lambda*_mem (docs/theory/exact_lambda/exact_lambda.tex).

For every (family, repeat, seed): the GRU reference controller of that repeat through the surrogate, 1000 training starts.
Exact price p_k = d loss / d eps_k with x_k -> x_k + eps_k at every step (one reverse pass per start, all memory paths);
direct sources c_k = (2/N)(Q x_k - R u_k K_k) (K_k = direct gain), c_N = 2 Q_f x_N / N; landing window W from k_land
(first |x_k| <= 0.2). Lambda_mem(x0) = log(max_W |p_k| / max_W |c_k|), Lambda*_mem = max over starts. Lambda_scal: same with
the scalar recursion p_k = c_k + m_k p_{k+1} (t2_test.costate). Also p_0 of the start with the largest Lambda_mem, both ways.
Output: $T2_OUT/t2_exact.csv, t2_exact_starts.npz.   Usage: python3 t2_exact.py [families comma-separated]"""
import sys
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import t2core as T
from t2_test import make_fns, costate, X_LAND, OUT, REPS


def make_price(norm):
    def loss_eps(c, p, s0, x0, eps):
        def step(carry, e):
            h, x = carry
            xe = x + e
            h2, u = T.gru_step(c, h, xe)
            return (h2, T.F_point(p, xe, u, s0, norm)), T.Q_X * xe ** 2 + T.R_U * u ** 2
        (h, xN), stage = jax.lax.scan(step, (jnp.zeros(T.HIDDEN), x0), eps[:-1])
        return (jnp.sum(stage) + T.Q_F * (xN + eps[-1]) ** 2) / T.N_HOR
    g = jax.grad(loss_eps, argnums=4)
    return jax.jit(jax.vmap(lambda c, p, s0, x0: g(c, p, s0, x0, jnp.zeros(T.N_HOR + 1)), in_axes=(None, None, None, 0)))


def stats(xs, us, K, m, P):
    B, N = us.shape
    out = dict(L_mem=np.zeros(B), L_scal=np.zeros(B), p0_mem=np.zeros(B), p0_scal=np.zeros(B))
    for b in range(B):
        lam = costate(xs[b], us[b], K[b], m[b])                       # scalar recursion, p_N = c_N
        c = np.append((2 * T.Q_X * xs[b, :-1] - 2 * T.R_U * us[b] * K[b]) / N, 2 * T.Q_F * xs[b, -1] / N)
        out["p0_mem"][b], out["p0_scal"][b] = P[b, 0], lam[0]
        near = np.abs(xs[b, :-1]) <= X_LAND
        if not near.any():
            continue
        kl = int(np.argmax(near))
        s = np.max(np.abs(c[kl:]))
        out["L_mem"][b] = np.log(np.max(np.abs(P[b, kl:])) / s)
        out["L_scal"][b] = np.log(np.max(np.abs(lam[kl:])) / s)
    return out


def run(families):
    rows, per = [], {}
    for fam in families:
        Pm, S0, norm, seeds, met = T.load_surrogate(fam)
        ps = make_fns(norm)[0]
        price = make_price(norm)
        for rep in REPS:
            Co, cs = T.load_controllers("true_plant__oracle", rep)
            for i, s in enumerate(seeds):
                t0 = time.time()
                p, s0, c = T.seed_slice(Pm, i), S0[i], T.seed_slice(Co, cs.index(s))
                X0 = T.training_starts(s, rep)
                xs, us, jx, ju, K, m, sens = (np.asarray(a) for a in ps(c, p, s0, X0))
                P = np.asarray(price(c, p, s0, X0))
                st = stats(xs, us, K, m, P)
                b = int(np.argmax(st["L_mem"]))
                rows.append(dict(family=fam, rep=rep, seed=int(s), L_mem_star=float(st["L_mem"].max()),
                                 L_scal_star=float(st["L_scal"].max()), top_start=float(X0[b]),
                                 p0_mem_top=float(st["p0_mem"][b]), p0_scal_top=float(st["p0_scal"][b]),
                                 max_abs_p0_mem=float(np.abs(st["p0_mem"]).max()),
                                 max_abs_p0_scal=float(np.abs(st["p0_scal"]).max())))
                per[f"{fam}|{rep}|{s}"] = np.stack([st["L_mem"], st["L_scal"], st["p0_mem"], st["p0_scal"]])
                r = rows[-1]
                print(f"{fam:16s} rep {rep} seed {s}: L*_mem {r['L_mem_star']:8.2f} L*_scal {r['L_scal_star']:8.2f} | top start "
                      f"{r['top_start']:+.2f}: p0 exact {r['p0_mem_top']:.2e} scalar {r['p0_scal_top']:.2e} | {time.time() - t0:.0f}s",
                      flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/t2_exact.csv", index=False)
        np.savez_compressed(f"{OUT}/t2_exact_starts.npz", **per)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run(sys.argv[1].split(",") if len(sys.argv) > 1 else list(T.FAMILIES))
