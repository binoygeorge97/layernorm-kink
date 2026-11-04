"""Q7d (2026-11-04): RMSNorm surrogates, scored against prereg/PREREG_rms.md Part 1 (e221377).
Inputs: data/dpc_out_rms/surrogates.csv (kink metrics, Theorem 1 with P = I), the RMS LQR study
  LK_DATA=data/dpc_out_rms LK_FAMILIES=rms LK_OUT=out/rms python companion/run_lqr_study.py   -> out/rms/companion/lqr_study.csv
and the LN references (data/dpc_out/surrogates.csv, companion/lqr_study.csv). Writes out/rms/summary.csv (paper table) and prints
M1-M4. Rule CLAUDE.md 1.3."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
R_ = lambda *a: os.path.join(paths.REPO, *a)
S = pd.read_csv(R_("data", "dpc_out_rms", "surrogates.csv"))
SL = pd.read_csv(R_("data", "dpc_out", "surrogates.csv"))


def lqr(fn):
    T = pd.read_csv(fn); T = T[(T.x0 - T.r).abs() >= 0.5].copy()
    T["failed"] = T.diverged | (T.gap > 0.1); T["catastrophic"] = T.diverged | (T.gap > 1.0)
    return T
T, TL = lqr(R_("out", "rms", "companion", "lqr_study.csv")), lqr(R_("companion", "lqr_study.csv"))
rows = []
for fam, model, lq, SS in [("RMS MSE 3000", "RMSNorm, MSE (3000)", "RMS MSE 3000", S), ("RMS two-stage", "RMSNorm, two-stage", "RMS two-stage", S),
                           ("LN MSE 3000 (reference)", "LN, MSE (3000)", "LN MSE 3000", SL), ("LN two-stage (reference)", "LN, two-stage", "LN two-stage", SL)]:
    g = SS[SS.model == model]; TT = TL if "reference" in fam else T
    f = TT[(TT.model == lq) & (TT.controller == "LQR fixed")]; s = TT[(TT.model == lq) & (TT.controller == "LQR sched")]
    rows.append(dict(family=fam, seeds=len(g), rstar_x=g.rstar_x.median(), rstar_u=g.rstar_u.median(), Ju_sup_rel=g.Ju_sup_rel.median(),
                     max_centre_offset=g[["zmin_x", "zmin_u"]].abs().max(axis=1).max(), rel_mse=g.rel_mse.median(),
                     fixed_failed_pct=100 * f.failed.mean(), fixed_catastrophic_pct=100 * f.catastrophic.mean(),
                     sched_failed_pct=100 * s.failed.mean(), sched_catastrophic_pct=100 * s.catastrophic.mean(), n_lqr=len(f)))
D = pd.DataFrame(rows); os.makedirs(R_("out", "rms"), exist_ok=True); D.to_csv(R_("out", "rms", "summary.csv"), index=False)
pd.set_option("display.width", 250); print(D.round(4).to_string(index=False))
r = D.set_index("family")
off = S[["zmin_x", "zmin_u"]].abs().max(axis=1)
m1 = "held" if (off <= 0.1).all() else "failed"
m2 = "held" if (0.13 <= r.loc["RMS MSE 3000", "rstar_x"] <= 0.52 and 0.17 <= r.loc["RMS MSE 3000", "rstar_u"] <= 0.67) else "failed"
m3 = "held" if 13.3 <= r.loc["RMS MSE 3000", "fixed_failed_pct"] <= 53 else "failed"
m4 = "held" if (r.loc["RMS two-stage", "fixed_failed_pct"] < r.loc["RMS MSE 3000", "fixed_failed_pct"] and r.loc["RMS two-stage", "fixed_failed_pct"] <= 13.3) else "failed"
print(f"\nM1 {m1}: {int((off > 0.1).sum())} of {len(off)} centres outside 0.1 (max {off.max():.3f})")
print(f"M2 {m2}: RMS MSE 3000 median r*_x {r.loc['RMS MSE 3000', 'rstar_x']:.3f} (band [0.13, 0.52]), r*_u {r.loc['RMS MSE 3000', 'rstar_u']:.3f} ([0.17, 0.67])")
print(f"M3 {m3}: RMS MSE 3000 fixed failed {r.loc['RMS MSE 3000', 'fixed_failed_pct']:.1f} % (band [13.3, 53])")
print(f"M4 {m4}: RMS two-stage fixed failed {r.loc['RMS two-stage', 'fixed_failed_pct']:.1f} % vs RMS MSE 3000 {r.loc['RMS MSE 3000', 'fixed_failed_pct']:.1f} % (<= 13.3 required)")
