"""R2 (b) (2026-11-04), POST HOC: for each prospective Lambda* test, ROC AUC of Lambda* for trapped vs not trapped and the
interval of thresholds with zero false alarms, with its hit rate. Also the GRU B = 0.01 retrospective set for reference.
AUC = P(Lambda*(trapped) > Lambda*(not trapped)) + 0.5 P(equal) (Mann-Whitney). Output out/theory/lambda_posthoc.csv."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
from common import runs, FAMILY
R_ = lambda *a: os.path.join(paths.REPO, *a)
specs = [("GRU B = 0.01 (retrospective)", "data/dpc_out", "out/theory/t2_runs_all_B001.csv", "S_seg_max", None),
         ("MLP B = 0.01 (prospective, f945a5c)", "data/run5_mlp", "prereg/PREREG_arch_predicted.csv", "Lambda_star", "mlp"),
         ("linear B = 0.01 (prospective, 322e353)", "data/run5_linear", "prereg/PREREG_arch_predicted.csv", "Lambda_star", "linear"),
         ("GRU B = 0.1 direct (prospective, 77d8455)", "data/run6_B01", "prereg/PREREG_B01_predicted.csv", "S_seg_max", None),
         ("GRU B = 0.1 augmented (prospective, 91238b4)", "data/run6_B01", "prereg/PREREG_B01_aug_predicted.csv", "L_aug_star", None)]
rows = []
for name, root, lam, col, arch in specs:
    Rr = runs(R_(root)); Rr = Rr[Rr.model.isin(FAMILY)].assign(family=lambda d: d.model.map(FAMILY))
    L = pd.read_csv(R_(lam)); L = L[L.arch == arch] if arch else L
    J = Rr.merge(L[["family", "seed", "rep", col]], on=["family", "seed", "rep"])
    t, v = J.trapped.values, J[col].values
    pos, neg = v[t], v[~t]
    auc = np.nan if not len(pos) or not len(neg) else (np.mean(pos[:, None] > neg[None]) + 0.5 * np.mean(pos[:, None] == neg[None]))
    lo = neg.max() if len(neg) else -np.inf
    hits = int((pos > lo).sum()) if len(pos) else 0
    hi = pos.max() if hits else np.nan
    m = (J.family == "LN MSE 3000").values; p3, n3 = v[m & t], v[m & ~t]
    auc3 = np.nan if not len(p3) or not len(n3) else (np.mean(p3[:, None] > n3[None]) + 0.5 * np.mean(p3[:, None] == n3[None]))
    rows.append(dict(test=name, runs=len(J), trapped=int(t.sum()), auc=auc, auc_LNMSE3000_only=auc3,
                     LNMSE3000_trapped_of_30=int((m & t).sum()), zero_fa_interval=f"({lo:.2f}, {hi:.2f}]" if hits else f"> {lo:.2f} (no trap above)",
                     hits_in_interval=hits, hit_rate=hits / len(pos) if len(pos) else np.nan,
                     trapped_values=" ".join(f"{x:.2f}" for x in np.sort(pos)), max_not_trapped=lo))
D = pd.DataFrame(rows); D.to_csv(paths.out("theory", "lambda_posthoc.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print(D.drop(columns=["trapped_values"]).to_string(index=False))
for _, r in D.iterrows():
    print(r.test, "| trapped Lambda*:", r.trapped_values)
