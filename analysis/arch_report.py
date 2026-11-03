"""Task 1.5: MLP / linear DPC results vs the GRU, scored against prereg/PREREG_arch.md.

Per architecture x family: failed / trapped of 30, median gap; paired Wilcoxon vs the GRU (per-seed median over repeats);
confusion matrix of the pre-registered Lambda* prediction (prereg/PREREG_arch_predicted.csv) vs observed traps;
K_eff = -d pi / dx at x = 0, 0.5, 2 of every learned memoryless controller (median over repeats) vs the LQR gain.
Usage: python analysis/arch_report.py --mlp data/dpc_out_mlp --linear data/dpc_out_linear [--gru data/dpc_out]
Outputs out/arch/*.csv and a printed summary for docs/RESULTS_arch.md."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import runs, summary, paired, FAMILY, ORACLE  # noqa: E402

FAMS = list(FAMILY)


def keff(root):
    import dpc_core as DC
    rows = []
    for m in [ORACLE] + FAMS:
        for rep in (0, 1, 2):
            C, cs = DC.load_controllers(root, m, rep)
            if C is None or "Wz" in C:
                continue
            for i, s in enumerate(cs):
                c = {k: (np.asarray(v)[i] if not isinstance(v, dict) else {kk: np.asarray(vv)[i] for kk, vv in v.items()})
                     for k, v in C.items()}
                g = DC.policy_gain(c, [0.0, 0.5, 2.0])
                rows.append(dict(model=m, rep=rep, seed=int(s), K0=g[0], K05=g[1], K2=g[2]))
    return pd.DataFrame(rows)


def confusion(R, arch):
    fn = os.path.join(paths.REPO, "prereg", "PREREG_arch_predicted.csv")
    if not os.path.exists(fn):
        return None
    P = pd.read_csv(fn)
    P = P[P.arch == arch]
    R = R[R.model.isin(FAMS)].assign(family=lambda d: d.model.map(FAMILY))
    J = R.merge(P, on=["family", "seed", "rep"], how="inner")
    tp = int((J.predicted & J.trapped).sum()); fn_ = int((~J.predicted & J.trapped).sum())
    fp = int((J.predicted & ~J.trapped).sum()); tn = int((~J.predicted & ~J.trapped).sum())
    return dict(arch=arch, runs=len(J), hits=tp, misses=fn_, false_alarms=fp, correct_rejections=tn,
                hit_rate=tp / max(tp + fn_, 1), false_alarm_rate=fp / max(fp + tn, 1),
                missed_runs="; ".join(f"{r.family} s{r.seed} r{r.rep} (L*={r.Lambda_star:.1f})" for _, r in J[~J.predicted & J.trapped].iterrows()),
                false_alarm_runs="; ".join(f"{r.family} s{r.seed} r{r.rep} (L*={r.Lambda_star:.1f})" for _, r in J[J.predicted & ~J.trapped].iterrows()))


def main(roots, gru_root):
    pd.set_option("display.width", 220)
    G = runs(gru_root)
    out = {}
    for arch, root in roots.items():
        if not root or not os.path.exists(os.path.join(root, "dpc_results.csv")):
            print(f"{arch}: no results in {root}")
            continue
        R = runs(root)
        S = summary(R, [ORACLE] + FAMS)
        S["family"] = S.model.map(FAMILY).fillna("oracle")
        S["gru_failed"] = [int(G[G.model == m].failed.sum()) for m in S.model]
        S["gru_trapped"] = [int(G[G.model == m].trapped.sum()) for m in S.model]
        pw = pd.DataFrame([dict(model=m, **paired(R[R.model == m], G[G.model == m])) for m in FAMS])
        per_seed = (R[R.model.isin(FAMS)].groupby(["model", "seed"]).failed.sum().unstack(1))
        cm = confusion(R, arch)
        K = keff(root)
        for d, n in ((S, "summary"), (pw, "paired_vs_gru"), (per_seed, "failed_per_seed"), (K, "keff")):
            d.to_csv(paths.out("arch", f"{arch}_{n}.csv"), index=n == "failed_per_seed")
        print(f"\n===== {arch.upper()} ({root})\n", S.round(3).to_string(index=False))
        print("\npaired vs GRU (A = this architecture; a_better = seeds where it has the smaller gap)\n", pw.round(4).to_string(index=False))
        print("\nfailed repeats per seed (of 3)\n", per_seed.to_string())
        if cm:
            print("\nconfusion vs pre-registered Lambda* prediction:", cm)
        if len(K):
            print("\nK_eff = -dpi/dx, median over repeats (LQR gain 10.29):\n",
                  K.groupby(["model", "seed"])[["K0", "K05", "K2"]].median().unstack(1)["K0"].round(1).to_string())
        out[arch] = dict(S=S, pw=pw, cm=cm, K=K)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp"), ap.add_argument("--linear"), ap.add_argument("--gru", default=paths.DATA)
    a = ap.parse_args()
    main({"mlp": a.mlp, "linear": a.linear}, a.gru)
