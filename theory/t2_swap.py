"""Swap test for the landing statistic: Lambda* recomputed along the same reference paths with J_u_hat -> B or J_x_hat -> A."""
import numpy as np, pandas as pd
import t2core as T
from t2_test import make_fns, max_subarray, X_LAND, OUT
P, S0, norm, seeds, met = T.load_surrogate("LN MSE 3000")
ps = make_fns(norm)[0]
rows = []
for rep in (0, 1, 2):
    Co, cs = T.load_controllers("true_plant__oracle", rep)
    for s in seeds:
        xs, us, jx, ju, K, m, sens = (np.asarray(a) for a in ps(T.seed_slice(Co, cs.index(s)), T.seed_slice(P, s), S0[s], T.training_starts(s, rep)))
        out = {}
        for tag, mm in (("surrogate", m), ("true_Ju", jx - T.B_TRUE * K), ("true_Jx", T.A_TRUE - ju * K)):
            lm = np.log(np.abs(mm) + 1e-300)
            best = 0.0
            for b in range(len(xs)):
                near = np.abs(xs[b, :-1]) <= X_LAND
                kl = int(np.argmax(near)) if near.any() else T.N_HOR
                best = max(best, max_subarray(lm[b, kl:]))
            out[tag] = best
        rows.append(dict(rep=rep, seed=s, **out))
        print(rep, s, {k: round(v, 2) for k, v in out.items()}, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/t2_swap.csv", index=False)
