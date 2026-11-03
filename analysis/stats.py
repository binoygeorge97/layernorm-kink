"""Task 4: statistics (seed-level bootstrap, paired tests with Holm, threshold sensitivity, Lambda* threshold robustness).

Unit of replication = the SEED (the surrogate). Bootstrap: 10 000 resamples of the 10 seeds with replacement; a resampled
seed carries all its runs (all setpoints and starts, all 3 DPC repeats); the statistic is recomputed on the pooled runs;
95 % percentile interval. Fixed generator seed (np.random.default_rng(20261103)) so intervals are reproducible.

Inputs (first existing file wins):
  LQR study     out/companion/lqr_study_extra.csv (has settle_time), out/companion/lqr_study.csv, companion/lqr_study.csv
  DPC           <DATA>/dpc_results.csv, <DATA>/dpc_diag.csv
  LQR swap      out/stats/swap_lqr_runs.csv (analysis/swap_lqr_runs.py)
  Lambda*       out/theory/t2_runs_*.csv (theory/t2_test.py), else data/reference_results/lambda_star_180runs.csv
Outputs: out/stats/*.csv, out/stats/tab_stats.tex, printed summary (docs/RESULTS_stats.md is written from it by hand)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

NB = 10_000
RNG_SEED = 20261103
SEEDS = np.arange(10)
FAM_LQR = ["LN MSE 3000", "LN MSE 12000", "LN slope from step 0", "LN two-stage", "no norm", "static norm"]
FAM_DPC = {"LN, MSE (3000)": "LN MSE 3000", "LN, MSE (12000)": "LN MSE 12000", "LN, J_x + J_u (3000)": "LN slope from step 0",
           "LN, two-stage": "LN two-stage", "no norm, MSE": "no norm", "static norm, MSE": "static norm"}
OS = lambda *p: paths.out("stats", *p)


def first(*cands):
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(cands)


def boot_idx(n_boot=NB, seed=RNG_SEED):
    """[n_boot, 10] resampled seed indices (the same draws for every statistic, so intervals are comparable)."""
    return np.random.default_rng(seed).integers(0, 10, (n_boot, 10))


IDX = boot_idx()


def seed_boot(df, num, den=None):
    """Point estimate and 95 % percentile CI of sum(num) / sum(den) (den = count of rows if None) over seed resamples.
    df has a 'seed' column; num/den are column names (bool or float)."""
    g = df.groupby("seed")
    n = g[num].sum().reindex(SEEDS, fill_value=0).to_numpy(float)
    d = (g.size() if den is None else g[den].sum()).reindex(SEEDS, fill_value=0).to_numpy(float)
    est = n.sum() / d.sum() if d.sum() else np.nan
    nb, db = n[IDX].sum(1), d[IDX].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = nb / db
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return est, lo, hi, int(d.sum())


def rate_row(df, col, **keys):
    est, lo, hi, n = seed_boot(df, col)
    return dict(**keys, metric=col, pct=100 * est, ci_lo=100 * lo, ci_hi=100 * hi, n=n)


# ------------------------------------------------------------------ 4.1 LQR
def lqr_table():
    fn = first(paths.out("companion", "lqr_study_extra.csv"), paths.out("companion", "lqr_study.csv"),
               os.path.join(paths.REPO, "companion", "lqr_study.csv"))
    T = pd.read_csv(fn)
    T["far"] = (T.x0 - T.r).abs() >= 0.5
    T["failed"] = T.diverged | (T.gap > 0.1)
    T["catastrophic"] = T.diverged | (T.gap > 1.0)
    print(f"LQR study: {fn} ({len(T)} rows)")
    return T[T.far].copy(), fn


def lqr_rates(T):
    rows = []
    for fam in FAM_LQR:
        for ctl in ("LQR fixed", "LQR sched"):
            g = T[(T.model == fam) & (T.controller == ctl)]
            for col in ("failed", "catastrophic"):
                rows.append(rate_row(g, col, family=fam, controller=ctl))
    return pd.DataFrame(rows)


def setpoint_rates(T):
    rows = []
    for fam in FAM_LQR:
        for ctl in ("LQR fixed", "LQR sched"):
            for r, g in T[(T.model == fam) & (T.controller == ctl)].groupby("r"):
                rows.append(rate_row(g, "failed", family=fam, controller=ctl, r=r))
    return pd.DataFrame(rows)


WS_BINS, WS_LABELS = [-0.01, 0.0, 0.02, 0.1, 1.0], ["0", "0-2 %", "2-10 %", "> 10 %"]


def path_colour(T, B=0.01, good=0.2):
    """phase_report.path_table rule: LN families, |B_hat/B - 1| < 0.2, bins of wrong_sign_share (pd.cut, right-closed)."""
    g = T[T.model.str.startswith("LN")].copy()
    g["ws_bin"] = pd.cut(g.wrong_sign_share, WS_BINS, labels=WS_LABELS)
    g = g[np.abs(g.Bh / B - 1) < good]
    rows = []
    for ctl in ("LQR fixed", "LQR sched"):
        for b in WS_LABELS:
            h = g[(g.controller == ctl) & (g.ws_bin == b)]
            rows.append(rate_row(h, "failed", controller=ctl, ws_bin=b))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4.1 DPC
def dpc_table():
    R = pd.read_csv(os.path.join(paths.DATA, "dpc_results.csv"))
    R["rep"] = R.rep.fillna(0).astype(int)
    o = R[R.model == "true plant (oracle)"][["seed", "rep", "cost_test_true"]].rename(columns={"cost_test_true": "oracle"})
    R = R.merge(o, on=["seed", "rep"])
    R["gap"] = R.cost_test_true / R.oracle - 1
    Dg = pd.read_csv(os.path.join(paths.DATA, "dpc_diag.csv"))
    Dg["rep"] = Dg.rep.fillna(0).astype(int)
    R = R.merge(Dg[["model", "seed", "rep", "oracle_on_model"]], on=["model", "seed", "rep"], how="left")
    R["failed"] = R.gap > 0.10
    R["trapped"] = R.failed & (R.cost_test_own / R.oracle_on_model > 1.05)
    R["family"] = R.model.map(FAM_DPC)
    return R[R.family.notna()].copy()


def dpc_rates(R):
    rows = []
    for fam in FAM_LQR:
        g = R[R.family == fam]
        for col in ("failed", "trapped"):
            est, lo, hi, n = seed_boot(g, col)
            rows.append(dict(family=fam, metric=col, count=int(g[col].sum()), runs=len(g), pct=100 * est,
                             ci_lo=100 * lo, ci_hi=100 * hi))
    return pd.DataFrame(rows)


def swap_rates():
    fn = paths.out("stats", "swap_lqr_runs.csv")
    if not os.path.exists(fn):
        print("swap: run analysis/swap_lqr_runs.py first - skipped")
        return pd.DataFrame()
    S = pd.read_csv(fn)
    S = S[S.far]
    rows = []
    for (fam, des), g in S.groupby(["family", "design"], sort=False):
        for col in ("failed", "catastrophic"):
            rows.append(rate_row(g, col, family=fam, design=des))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4.2 paired claims
CLAIMS = [("LN two-stage", "LN MSE 3000"), ("LN MSE 3000", "static norm"), ("LN MSE 3000", "no norm"),
          ("LN two-stage", "static norm"), ("LN two-stage", "no norm"), ("LN MSE 12000", "LN MSE 3000")]


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired(R):
    """per-seed median over repeats of the typical-start gap; difference A - B per seed (negative = A better)."""
    med = R.groupby(["family", "seed"]).gap.median().unstack(0)
    rows = []
    for a, b in CLAIMS:
        d = (med[a] - med[b]).to_numpy()
        st = wilcoxon(med[a], med[b], alternative="two-sided")
        bm = np.median(d[IDX], 1)
        rows.append(dict(A=a, B=b, A_better=int((d < 0).sum()), ties=int((d == 0).sum()), n=len(d),
                         median_diff_pp=100 * np.median(d), ci_lo_pp=100 * np.percentile(bm, 2.5),
                         ci_hi_pp=100 * np.percentile(bm, 97.5), median_gap_A_pct=100 * med[a].median(),
                         median_gap_B_pct=100 * med[b].median(), wilcoxon_p=st.pvalue))
    P = pd.DataFrame(rows)
    P["holm_p"] = holm(P.wilcoxon_p)
    return P, med


# ------------------------------------------------------------------ 4.3 threshold sensitivity
THR = [0.05, 0.10, 0.20, 0.50, 1.00]


def thresholds(T, R):
    rows = []
    for fam in FAM_LQR:
        for ctl in ("LQR fixed", "LQR sched"):
            g = T[(T.model == fam) & (T.controller == ctl)].copy()
            row = dict(family=fam, controller=ctl, runs=len(g))
            for t in THR:
                g["f"] = g.diverged | (g.gap > t)
                row[f"failed_gt_{int(100 * t)}pct"] = 100 * g.f.mean()
            ok = g[~g.diverged]
            row["median_offset"] = ok.offset.median()
            if "settle_time" in g:
                row["median_settle_time"] = g.settle_time.median()       # NaN (never settles / diverged) excluded
                row["never_settled_pct"] = 100 * g.settle_time.isna().mean()
            rows.append(row)
        g = R[R.family == fam]
        row = dict(family=fam, controller="DPC (GRU, typical starts)", runs=len(g))
        for t in THR:
            row[f"failed_gt_{int(100 * t)}pct"] = 100 * (g.gap > t).mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4.4 Lambda* threshold robustness
def lambda_table():
    a = paths.out("theory", "t2_runs_LNMSE3000.csv")
    b = paths.out("theory", "t2_runs_LNMSE12000_LNslopefrom0_LNtwo-stage_nonorm_staticnorm.csv")
    if os.path.exists(a) and os.path.exists(b):
        G = pd.concat([pd.read_csv(a), pd.read_csv(b)], ignore_index=True)
        G["trap"] = (G.gap > 0.1) & (G.own_on_model / G.ref_on_model > 1.05)
        src = "out/theory/t2_runs_*.csv (re-run)"
    else:
        G = pd.read_csv(os.path.join(paths.REF, "lambda_star_180runs.csv"))
        G["trap"] = G.trap.astype(bool)
        src = "data/reference_results/lambda_star_180runs.csv"
    print(f"Lambda*: {src} ({len(G)} runs, {int(G.trap.sum())} trapped)")
    return G[["family", "rep", "seed", "trap", "S_seg_max"]].copy()


def lambda_robustness(G):
    ln = G[G.family == "LN MSE 3000"]
    rows = []
    for s in SEEDS:
        tr = ln[ln.seed != s]
        thr = np.sqrt(tr[~tr.trap].S_seg_max.max() * tr[tr.trap].S_seg_max.min())
        te = ln[ln.seed == s]
        flag = te.S_seg_max > thr
        rows.append(dict(held_out_seed=s, threshold=thr, traps=int(te.trap.sum()), hits=int((flag & te.trap).sum()),
                         false_alarms=int((flag & ~te.trap).sum()), misses=int((~flag & te.trap).sum())))
    loso = pd.DataFrame(rows)
    non, trp = G[~G.trap].S_seg_max, G[G.trap].S_seg_max
    lo = non.max()                                           # any threshold >= lo gives 0 false alarms on the non-trapped
    grid = np.unique(np.concatenate([[lo], np.sort(trp.to_numpy())]))
    sweep = pd.DataFrame([dict(threshold=t, hits=int((trp > t).sum()), traps=len(trp), false_alarms=int((non > t).sum()),
                               non_trapped=len(non)) for t in grid])
    ln_lo, ln_hi = ln[~ln.trap].S_seg_max.max(), ln[ln.trap].S_seg_max.min()
    return loso, sweep, dict(zero_fa_from=lo, n_non_trapped=len(non), n_traps=len(trp),
                             hits_just_above=int((trp > lo).sum()), lnmse_gap=(ln_lo, ln_hi),
                             threshold_12_1_hits=int((trp > 12.1).sum()), threshold_12_1_fa=int((non > 12.1).sum()))


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 220)
    T, lqr_src = lqr_table()
    L1 = lqr_rates(T)
    L1.to_csv(OS("lqr_rates_boot.csv"), index=False)
    print("\n4.1 LQR failure rates (% of runs with |x0 - r| >= 0.5), seed bootstrap 95 % CI")
    print(L1.round(1).to_string(index=False))
    SP = setpoint_rates(T)
    SP.to_csv(OS("lqr_setpoint_rates_boot.csv"), index=False)
    PC_ = path_colour(T)
    PC_.to_csv(OS("path_colour_boot.csv"), index=False)
    print("\n4.1 path colour (4 LN families, |B_hat/B - 1| < 0.2)")
    print(PC_.round(1).to_string(index=False))
    R = dpc_table()
    D1 = dpc_rates(R)
    D1.to_csv(OS("dpc_rates_boot.csv"), index=False)
    print("\n4.1 DPC failed / trapped per family (30 runs each)")
    print(D1.round(1).to_string(index=False))
    SW = swap_rates()
    if len(SW):
        SW.to_csv(OS("swap_rates_boot.csv"), index=False)
        print("\n4.1 LQR swap test")
        print(SW.round(1).to_string(index=False))
    P, med = paired(R)
    P.to_csv(OS("paired_claims.csv"), index=False)
    med.to_csv(OS("dpc_gap_per_seed_median.csv"))
    print("\n4.2 paired claims (typical-start DPC gap, per-seed median over repeats; diff = A - B in percentage points)")
    print(P.round(4).to_string(index=False))
    TH = thresholds(T, R)
    TH.to_csv(OS("threshold_sensitivity.csv"), index=False)
    print("\n4.3 failure rate (%) vs gap threshold; threshold-free metrics")
    print(TH.round(2).to_string(index=False))
    G = lambda_table()
    loso, sweep, info = lambda_robustness(G)
    loso.to_csv(OS("lambda_loso.csv"), index=False)
    sweep.to_csv(OS("lambda_threshold_sweep.csv"), index=False)
    print("\n4.4 Lambda* leave-one-seed-out (LN MSE 3000)")
    print(loso.round(2).to_string(index=False))
    print("totals: hits", loso.hits.sum(), "of", loso.traps.sum(), "| false alarms", loso.false_alarms.sum(),
          "| thresholds", round(loso.threshold.min(), 2), "to", round(loso.threshold.max(), 2))
    print("4.4 zero-false-alarm interval and hit rate:", {k: (np.round(v, 3) if not isinstance(v, tuple) else tuple(np.round(v, 3)))
                                                          for k, v in info.items()})
    print(sweep.round(3).to_string(index=False))
    write_tex(L1, D1, P)
    return dict(L1=L1, PC=PC_, D1=D1, SW=SW, P=P, TH=TH, loso=loso, sweep=sweep, info=info, lqr_src=lqr_src)


def write_tex(L1, D1, P):
    def ci(r):
        return f"{r.pct:.1f} [{r.ci_lo:.1f}, {r.ci_hi:.1f}]"
    lines = [r"\begin{tabular}{lllll}", r"\toprule",
             r"family & fixed LQR failed & fixed LQR catastrophic & DPC failed & DPC trapped \\", r"\midrule"]
    for fam in FAM_LQR:
        f = L1[(L1.family == fam) & (L1.controller == "LQR fixed")]
        d = D1[D1.family == fam]
        lines.append(f"{fam} & {ci(f[f.metric == 'failed'].iloc[0])} & {ci(f[f.metric == 'catastrophic'].iloc[0])} & "
                     f"{ci(d[d.metric == 'failed'].iloc[0])} & {ci(d[d.metric == 'trapped'].iloc[0])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\begin{tabular}{llrrrr}", r"\toprule",
              r"A & B & A better (of 10) & median difference, pp [95\,\% CI] & Wilcoxon $p$ & Holm $p$ \\", r"\midrule"]
    for _, r in P.iterrows():
        lines.append(f"{r.A} & {r.B} & {r.A_better} & {r.median_diff_pp:.2f} [{r.ci_lo_pp:.2f}, {r.ci_hi_pp:.2f}] & "
                     f"{r.wilcoxon_p:.3f} & {r.holm_p:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(OS("tab_stats.tex"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
