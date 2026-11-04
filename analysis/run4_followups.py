"""S2 (2026-11-04), POST HOC follow-ups to Run 4 (written after the Run 4 results were read; nothing here was pre-registered).
(a) per setting: the failed / trapped (seed, repeat) runs, and whether healthy LN MSE seeds (0, 3-7) are trapped;
(c) paired per-seed comparisons (median gap over the 3 repeats, Wilcoxon signed-rank over the 10 seeds, Holm over the
    8 comparisons of this family): two-stage plain vs unit, clip 3, global clip 3, lr 1e-4; plain vs the same four;
(d) wall time per setting (median train_s_per_seed, s per seed);
(e) the per-start-clip failures: surrogate cost of the own controller vs the oracle controller (ratio <= 1.05 = not trapped).
Usage: python analysis/run4_followups.py [root = data/run4]   -> out/run4/followups/*.csv and a printed report."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from common import runs
R_ = lambda *a: os.path.join(paths.REPO, *a)
root = sys.argv[1] if len(sys.argv) > 1 else R_("data", "run4")
OUT = R_("out", "run4", "followups"); os.makedirs(OUT, exist_ok=True)
S = {"plain": "LN, MSE (3000)", "clip 3": "LN, MSE (3000), clipped grads", "unit": "LN, MSE (3000), unit grads",
     "global clip 3": "LN, MSE (3000), global clip 3", "mom": "LN, MSE (3000), mom", "batch 100": "LN, MSE (3000), batch 100",
     "lr 1e-4": "LN, MSE (3000), lr 1e-4", "N <= 50": "LN, MSE (3000), N <= 50", "N <= 100": "LN, MSE (3000), N <= 100",
     "two-stage plain": "LN, two-stage"}
HEALTHY = [0, 3, 4, 5, 6, 7]
R = runs(root)
pd.set_option("display.width", 250)

# (a) + (d)
rows, lab = [], []
for s, m in S.items():
    g = R[R.model == m].sort_values(["seed", "rep"])
    f = g[g.failed]; t = g[g.trapped]
    for _, r in f.iterrows():
        lab.append(dict(setting=s, seed=r.seed, rep=r.rep, gap_pct=100 * r.gap, trapped=bool(r.trapped), ratio_own_oracle_sur=r.ratio))
    ht = t[t.seed.isin(HEALTHY)]
    rows.append(dict(setting=s, runs=len(g), failed=len(f), trapped=len(t),
                     failed_runs=" ".join(f"s{r.seed}r{r.rep}" for _, r in f.iterrows()),
                     trapped_runs=" ".join(f"s{r.seed}r{r.rep}" for _, r in t.iterrows()),
                     healthy_trapped=len(ht), healthy_trapped_runs=" ".join(f"s{r.seed}r{r.rep}" for _, r in ht.iterrows()),
                     wall_s_per_seed=g.train_s_per_seed.median()))
A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, "labels_by_setting.csv"), index=False)
pd.DataFrame(lab).to_csv(os.path.join(OUT, "failed_runs.csv"), index=False)
print("(a)/(d) failed and trapped runs per setting\n", A.to_string(index=False))

# (c)
def med(m):
    return R[R.model == m].groupby("seed").gap.median()
P = []
for ref in ("two-stage plain", "plain"):
    for o in ("unit", "clip 3", "global clip 3", "lr 1e-4"):
        a, b = med(S[ref]), med(S[o]); j = pd.concat([a, b], axis=1, keys=["ref", "other"]).dropna()
        d = j.other - j.ref
        p = wilcoxon(j.ref, j.other).pvalue if (d != 0).any() else 1.0
        P.append(dict(comparison=f"{ref} vs {o}", n_seeds=len(j), ref_median_pct=100 * j.ref.median(), other_median_pct=100 * j.other.median(),
                      seeds_ref_better=int((d > 0).sum()), seeds_other_better=int((d < 0).sum()), ties=int((d == 0).sum()),
                      median_diff_pct=100 * d.median(), p=p))
P = pd.DataFrame(P)
o = np.argsort(P.p.values); k = len(P); adj = np.empty(k); run = 0
for i, idx in enumerate(o):
    run = max(run, min(1.0, (k - i) * P.p.values[idx])); adj[idx] = run
P["p_holm"] = adj
P.to_csv(os.path.join(OUT, "paired_holm.csv"), index=False)
print("\n(c) paired per-seed comparisons (median gap over repeats; Wilcoxon; Holm over 8)\n", P.round(4).to_string(index=False))

# (e)
D = pd.read_csv(os.path.join(root, "dpc_diag.csv")); D["rep"] = D.rep.fillna(0).astype(int)
c = R[(R.model == S["clip 3"]) & R.failed][["seed", "rep", "gap"]].merge(D[D.model == S["clip 3"]], on=["seed", "rep"])
c["own_over_oracle_on_sur"] = c.own_on_model / c.oracle_on_model
c["own_over_oracle_on_true"] = c.own_on_true / c.oracle_on_true
c.to_csv(os.path.join(OUT, "clip_failures.csv"), index=False)
print("\n(e) per-start clip failures\n", c[["seed", "rep", "gap", "own_on_model", "oracle_on_model", "own_over_oracle_on_sur",
                                           "own_on_true", "oracle_on_true"]].round(4).to_string(index=False))
