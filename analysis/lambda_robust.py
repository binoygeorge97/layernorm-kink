"""B3 (2026-11-04): robustness of Lambda* to a 1e-12 perturbation of the training starts, for every run with Lambda* > 8.
Perturbation: x0 -> x0 + 1e-12 * N(0, 1), key jax.random.fold_in(PRNGKey(50000 + s + 1000 rep), draw), draws 0-4 (new key
offset 50000, documented in RESULTS_LOG). For the unperturbed starts and each draw: Lambda* (max over starts of the max
contiguous sum of log|m_k| after landing, the same statistic as t2_test / t2_arch) and, for the maximising start, k_land and
the first and last step of the maximising stretch.
Sets: gru (GRU oracle, t2_test multipliers with the direct gain), mlp / linear (memoryless oracles, t2_arch), gru_B01 (needs
T2_PLANT=1.03,0.1 T2_DATA=<dpc_out_B01>, set by the caller).
Usage (from the repo root): python analysis/lambda_robust.py SET RUNS_CSV [CONTROLLER_ROOT]   -> out/theory/lambda_robust_<SET>.csv"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory"))
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import t2core as T
from t2_test import X_LAND, OUT

EPS, NDRAW, KEY = 1e-12, 5, 50_000


def max_subarray_idx(a):
    """(best sum, i, j): the segment a[i:j] with the largest sum (empty segment -> 0, -1, -1)."""
    best, bi, bj, cur, ci = 0.0, -1, -1, 0.0, 0
    for k, v in enumerate(a):
        if cur <= 0:
            cur, ci = v, k
        else:
            cur += v
        if cur > best:
            best, bi, bj = cur, ci, k + 1
    return best, bi, bj


def stat(xs, m):
    logm = np.log(np.abs(m) + 1e-300)
    best = (-1.0, None, None, None, None)
    for b in range(xs.shape[0]):
        near = np.abs(xs[b, :-1]) <= X_LAND
        kl = int(np.argmax(near)) if near.any() else T.N_HOR
        v, i, j = max_subarray_idx(logm[b, kl:])
        if v > best[0]:
            best = (v, b, kl, kl + i if i >= 0 else None, kl + j if j >= 0 else None)
    return best


def main(which, runs_csv, root=None):
    R = pd.read_csv(runs_csv)
    col = "S_seg_max" if "S_seg_max" in R else "Lambda_star"
    R = R[R[col] > 8].sort_values(["family", "rep", "seed"])
    rows, cache = [], {}
    for _, r in R.iterrows():
        fam, s, rep = r.family, int(r.seed), int(r.rep)
        if fam not in cache:
            P, S0, norm, seeds, met = T.load_surrogate(fam)
            if which in ("gru", "gru_B01"):
                from t2_test import make_fns
                ps = make_fns(norm)[0]
            else:
                import t2_arch as A
                ps = A.make_fns(norm)[0]
            cache[fam] = (P, S0, ps)
        P, S0, ps = cache[fam]
        if which in ("gru", "gru_B01"):
            C, cs = T.load_controllers("true_plant__oracle", rep)
        else:
            import t2_arch as A
            C, cs = A.load_ref(root, rep)
        c, p, s0 = T.seed_slice(C, cs.index(s)), T.seed_slice(P, s), S0[s]
        X0 = T.training_starts(s, rep)
        out = dict(set=which, family=fam, seed=s, rep=rep, Lambda_registered=float(r[col]))
        vals = []
        for d in range(-1, NDRAW):
            if d < 0:
                X = X0
            else:
                k = jax.random.fold_in(jax.random.PRNGKey(KEY + s + 1000 * rep), d)
                X = X0 + EPS * jax.random.normal(k, X0.shape, X0.dtype)
            res = ps(c, p, s0, X)
            xs, m = np.asarray(res[0]), np.asarray(res[5])
            v, b, kl, k1, k2 = stat(xs, m)
            tag = "orig" if d < 0 else f"d{d}"
            out.update({f"L_{tag}": v, f"start_{tag}": b, f"kland_{tag}": kl, f"k1_{tag}": k1, f"k2_{tag}": k2})
            if d >= 0:
                vals.append(v)
        out.update(L_min=min(vals), L_max=max(vals), spread=max(vals) - min(vals),
                   crosses_12_1=bool((min(vals + [out["L_orig"]]) <= 12.1) and (max(vals + [out["L_orig"]]) > 12.1)))
        rows.append(out)
        print(f"{which:8s} {fam:16s} s{s} r{rep}: registered {out['Lambda_registered']:7.2f} | orig {out['L_orig']:7.2f} "
              f"(start {out['start_orig']}, land {out['kland_orig']}, stretch {out['k1_orig']}-{out['k2_orig']}) | perturbed "
              f"{out['L_min']:7.2f} .. {out['L_max']:7.2f} | {'CROSSES 12.1' if out['crosses_12_1'] else ''}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(f"{OUT}/lambda_robust_{which}.csv", index=False)
    return D


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
