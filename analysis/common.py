"""Shared helpers for the analysis scripts: run tables with the CLAUDE.md 1.4 labels, per-seed paired tests."""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

FAMILY = {"LN, MSE (3000)": "LN MSE 3000", "LN, MSE (12000)": "LN MSE 12000", "LN, J_x + J_u (3000)": "LN slope from 0",
          "LN, two-stage": "LN two-stage", "no norm, MSE": "no norm", "static norm, MSE": "static norm"}
ORACLE = "true plant (oracle)"


def runs(root):
    """dpc_results.csv of a folder with gap (vs the oracle of the same seed and repeat), failed, trapped (needs dpc_diag.csv)."""
    R = pd.read_csv(os.path.join(root, "dpc_results.csv"))
    if "rep" not in R:
        R["rep"] = 0
    R["rep"] = R.rep.fillna(0).astype(int)
    o = R[R.model == ORACLE][["seed", "rep", "cost_test_true"]].rename(columns={"cost_test_true": "oracle"})
    R = R.merge(o, on=["seed", "rep"], how="left")
    R["gap"] = R.cost_test_true / R.oracle - 1
    fn = os.path.join(root, "dpc_diag.csv")
    if os.path.exists(fn):
        Dg = pd.read_csv(fn)
        Dg["rep"] = Dg.rep.fillna(0).astype(int)
        R = R.merge(Dg[["model", "seed", "rep", "oracle_on_model"]], on=["model", "seed", "rep"], how="left")
    else:
        R["oracle_on_model"] = np.nan
    R["failed"] = R.gap > 0.10
    R["ratio"] = R.cost_test_own / R.oracle_on_model
    R["trapped"] = R.failed & (R.ratio > 1.05)
    return R


def summary(R, models):
    rows = []
    for m in models:
        g = R[R.model == m]
        if not len(g):
            rows.append(dict(model=m, runs=0))
            continue
        per_seed = g.groupby("seed").gap.median()
        rows.append(dict(model=m, runs=len(g), failed=int(g.failed.sum()), trapped=int(g.trapped.sum()),
                         diag_missing=int(g.oracle_on_model.isna().sum()), median_gap_pct=100 * g.gap.median(),
                         best_seed_gap_pct=100 * per_seed.min(),
                         s_per_seed=g.train_s_per_seed.median() if "train_s_per_seed" in g else np.nan))
    return pd.DataFrame(rows)


def paired(a, b):
    """per-seed median over repeats of the gap; Wilcoxon signed-rank a vs b over common seeds. a, b: run tables."""
    ma, mb = a.groupby("seed").gap.median(), b.groupby("seed").gap.median()
    s = ma.index.intersection(mb.index)
    if len(s) < 2:
        return dict(n=len(s))
    d = ma[s] - mb[s]
    p = wilcoxon(ma[s], mb[s]).pvalue if (d != 0).any() else 1.0
    return dict(n=len(s), a_better=int((d < 0).sum()), median_diff_pp=100 * float(np.median(d)), wilcoxon_p=float(p))


def md(df, floatfmt="{:.3g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)
