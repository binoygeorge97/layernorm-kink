"""R5: apply the selection rule of prereg/PROTOCOL_trap_predictor.md (51b4b07 + amendment 9e62f0c) to the development set.
Reads out/predictor/warm_<set>.csv (C1, C2 per run) and the trap labels of each set; B0 = registered direct-gain Lambda* > 12.1.
Rule: eligible iff false-alarm rate <= 5 % in EACH of the four sets; choose the largest worst-case hit rate over the sets with
traps (GRU B = 0.01, MLP, GRU B = 0.1); ties: larger pooled hit rate, then lower pooled false-alarm rate, then C2 < C1.
If none is eligible: report and freeze nothing. Usage: python analysis/select_predictor.py [--allow-partial]
-> out/predictor/selection.csv and a printed table."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
from common import runs, FAMILY
R_ = lambda *a: os.path.join(paths.REPO, *a)
SETS = {"gru_B001": ("data/dpc_out", "out/theory/t2_runs_all_B001.csv", "S_seg_max", None),
        "mlp": ("data/run5_mlp", "prereg/PREREG_arch_predicted.csv", "Lambda_star", "mlp"),
        "linear": ("data/run5_linear", "prereg/PREREG_arch_predicted.csv", "Lambda_star", "linear"),
        "gru_B01": ("data/run6_B01", "prereg/PREREG_B01_predicted.csv", "S_seg_max", None)}
partial = "--allow-partial" in sys.argv
J = []
for s, (root, lam, col, arch) in SETS.items():
    Rr = runs(R_(root)); Rr = Rr[Rr.model.isin(FAMILY)].assign(family=lambda d: d.model.map(FAMILY))
    L = pd.read_csv(R_(lam)); L = L[L.arch == arch] if arch else L
    W = pd.read_csv(paths.out("predictor", f"warm_{s}.csv"))
    j = Rr[["family", "seed", "rep", "trapped"]].merge(L[["family", "seed", "rep", col]].rename(columns={col: "lam"}), on=["family", "seed", "rep"])
    j = j.merge(W[["family", "seed", "rep", "c1", "c2"]], on=["family", "seed", "rep"], how="left")
    j["set"] = s; j["B0"] = j.lam > 12.1
    J.append(j)
J = pd.concat(J)
done = J.c1.notna()
print(f"runs with warm-start result: {int(done.sum())} of {len(J)}")
if not done.all() and not partial:
    sys.exit("incomplete development set; rerun when all four sets are finished (or --allow-partial for a preview)")
J = J[done].copy(); J["C1"] = J.c1.astype(bool); J["C2"] = J.c2.astype(bool)
rows = []
for cand in ("C1", "C2", "B0"):
    r = dict(candidate=cand)
    for s in SETS:
        g = J[J.set == s]; t, p = g.trapped.values, g[cand].values
        r[f"{s}_hits"] = int((p & t).sum()); r[f"{s}_traps"] = int(t.sum())
        r[f"{s}_fa"] = int((p & ~t).sum()); r[f"{s}_neg"] = int((~t).sum())
        r[f"{s}_hit_rate"] = (p & t).sum() / t.sum() if t.sum() else np.nan
        r[f"{s}_fa_rate"] = (p & ~t).sum() / (~t).sum()
    r["eligible"] = all(r[f"{s}_fa_rate"] <= 0.05 for s in SETS) and cand != "B0"
    r["worst_hit_rate"] = min(r[f"{s}_hit_rate"] for s in SETS if r[f"{s}_traps"])
    r["pooled_hit_rate"] = sum(r[f"{s}_hits"] for s in SETS) / sum(r[f"{s}_traps"] for s in SETS)
    r["pooled_fa_rate"] = sum(r[f"{s}_fa"] for s in SETS) / sum(r[f"{s}_neg"] for s in SETS)
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(paths.out("predictor", "selection.csv"), index=False)
pd.set_option("display.width", 250)
cols = ["candidate", "eligible", "worst_hit_rate", "pooled_hit_rate", "pooled_fa_rate"] + [f"{s}_{k}" for s in SETS for k in ("hits", "traps", "fa", "neg")]
print(D[cols].round(3).to_string(index=False))
E = D[D.eligible].copy()
if not len(E):
    print("\nNO CANDIDATE MET THE CRITERION (false-alarm rate <= 5 % in every set). Nothing frozen; decision needed.")
else:
    E["order"] = E.candidate.map({"C2": 0, "C1": 1})
    E = E.sort_values(["worst_hit_rate", "pooled_hit_rate", "pooled_fa_rate", "order"], ascending=[False, False, True, True])
    print(f"\nSELECTED: {E.candidate.iloc[0]} (worst-case hit rate {E.worst_hit_rate.iloc[0]:.3f})")
