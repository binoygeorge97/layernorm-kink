import numpy as np, pandas as pd, t2core as T
from t2_test import make_fns, max_subarray, OUT
rows = []
for fam in T.FAMILIES:
    P, S0, norm, seeds, met = T.load_surrogate(fam); ps = make_fns(norm)[0]
    for rep in (0, 1, 2):
        Co, cs = T.load_controllers("true_plant__oracle", rep)
        for s in seeds:
            xs, us, jx, ju, K, m, sens = (np.asarray(a) for a in ps(T.seed_slice(Co, cs.index(s)), T.seed_slice(P, s), S0[s], T.training_starts(s, rep)))
            lm = np.log(np.abs(m) + 1e-300)
            rows.append(dict(family=fam, rep=rep, seed=s, L_all=max(max_subarray(r) for r in lm)))
    print(fam, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/t2_lambda_all.csv", index=False)
