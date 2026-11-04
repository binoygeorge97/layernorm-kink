"""B4 (retrospective): augmented-state landing amplification Lambda*_aug of the GRU reference controller
(definition docs/theory/exact_lambda_aug/exact_lambda_aug.tex, commit before this computation).
z_k = (x_k, h_k); Phi_k = d z_{k+1} / d z_k by jax.jacfwd of the closed-loop step; running product reset to I at k_land
(first |x_k| <= 0.2); Lambda_aug(x0) = max over k > k_land of log ||Phi(k, k_land)||_2 (exact SVD), >= 0;
Lambda_scal_land = the same anchoring with the scalar direct-gain multiplier. Lambda* = max over the 1000 training starts.
Usage: python t2_aug.py [families] ; env T2_DATA / T2_PLANT / T2_OUT / T2_REPS as for t2_test. Output $T2_OUT/t2_aug.csv"""
import sys, time
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
from t2_test import X_LAND, OUT, REPS


def make(norm):
    def step(c, p, s0, z):
        x, h = z[0], z[1:]
        h2, u = T.gru_step(c, h, x)
        return jnp.concatenate([jnp.atleast_1d(T.F_point(p, x, u, s0, norm)), h2])

    def one(c, p, s0, x0):
        z0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.zeros(T.HIDDEN)])
        def body(carry, k):
            z, M, landed, best, bests = carry
            x = z[0]
            land_now = (~landed) & (jnp.abs(x) <= X_LAND)
            M = jnp.where(land_now, jnp.eye(65), M)
            landed = landed | land_now
            J = jax.jacfwd(lambda zz: step(c, p, s0, zz))(z)
            # scalar direct-gain multiplier at this step (x-x entry with h fixed)
            m = J[0, 0]
            M2 = J @ M
            nrm = jnp.linalg.norm(M2, 2)
            best = jnp.where(landed, jnp.maximum(best, jnp.log(nrm)), best)
            bests_new = jnp.where(landed, bests[0] + jnp.log(jnp.abs(m) + 1e-300), 0.0)
            bests = jnp.array([bests_new, jnp.where(landed, jnp.maximum(bests[1], bests_new), bests[1])])
            M2 = jnp.where(landed, M2, M)
            return (step(c, p, s0, z), M2, landed, best, bests), None
        init = (z0, jnp.eye(65), jnp.array(False), jnp.array(0.0), jnp.zeros(2))
        (z, M, landed, best, bests), _ = jax.lax.scan(body, init, jnp.arange(T.N_HOR))
        return best, bests[1]
    return jax.jit(jax.vmap(one, in_axes=(None, None, None, 0)))


def run(families):
    rows = []
    for fam in families:
        P, S0, norm, seeds, met = T.load_surrogate(fam)
        f = make(norm)
        for rep in REPS:
            Co, cs = T.load_controllers("true_plant__oracle", rep)
            for i, s in enumerate(seeds):
                t0 = time.time()
                c, p, s0 = T.seed_slice(Co, cs.index(s)), T.seed_slice(P, i), S0[i]
                La, Ls = (np.asarray(a) for a in f(c, p, s0, T.training_starts(s, rep)))
                rows.append(dict(family=fam, rep=rep, seed=int(s), L_aug_star=float(La.max()), L_scal_land_star=float(Ls.max()),
                                 top_start=int(np.argmax(La))))
                print(f"{fam:16s} rep {rep} seed {s}: L*_aug {La.max():8.2f} | L*_scal,land {Ls.max():8.2f} | {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/t2_aug.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run(sys.argv[1].split(",") if len(sys.argv) > 1 else list(T.FAMILIES))
