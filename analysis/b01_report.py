"""Q1 (frozen 2026-11-04, before any Run 6 result was read): CLAUDE.md Task 2a.5 tables and figures for the B = 0.1 plant
side by side with B = 0.01, and the scoring of PREREG_B01 Part 2 (R1, R2; direct-gain Lambda*, 77d8455) and of
PREREG_B01_aug Part 2 (augmented Lambda*_aug).

Inputs (defaults = the B = 0.1 analysis; every one can be pointed at the B = 0.01 data for the end-to-end test):
  --root        DPC folder of the plant under test (dpc_results.csv, dpc_diag.csv)           default data/dpc_out_B01
  --lqr         its LQR study (lqr_study[_extra].csv)                                       default out/B01/companion/lqr_study_extra.csv
  --B           its input gain                                                               default 0.1
  --pred-direct run-level direct-gain predictions (family, seed, rep, S_seg_max, predicted)   default prereg/PREREG_B01_predicted.csv
  --pred-aug    run-level augmented predictions (family, seed, rep, L_aug_star, predicted)   default prereg/PREREG_B01_aug_predicted.csv
  --ref-root, --ref-lqr, --ref-B, --ref-lam   the B = 0.01 reference (data/dpc_out, companion/lqr_study.csv, 0.01,
                out/theory/t2_runs_all_B001.csv)
  --out         output folder                                                                default out/B01/report
A prediction file without a 'predicted' column is thresholded at --thr-direct (12.1) / --thr-aug (12.08).

Rules (fixed here, from CLAUDE.md 1.3 / 1.4 and the pre-registrations):
  LQR: runs with |x0 - r| >= 0.5; failed = diverged or gap > 10 %; catastrophic = diverged or gap > 100 %.
  DPC: the six registered families only (rows of any other model name, e.g. robust-aggregation models, are EXCLUDED and
       listed separately as exploratory); gap vs the oracle of the same seed and repeat; failed = gap > 10 %;
       trapped = failed and own controller > 1.05 x the oracle controller through the same surrogate (dpc_diag.csv).
  R1 / aug rule: held if hit rate >= 0.75 and false-alarm rate <= 0.05; if no run is predicted trapped: held if at most 1
       run is trapped. Undefined case (convention fixed now, before results): if runs are predicted trapped but NO run is
       trapped, the hit rate is undefined; the outcome is reported as "held (false alarms only)" if the false-alarm rate
       is <= 0.05, else "failed".
  R2: two-stage, no norm and static norm have no trapped run.
  CLAUDE.md decision rule: if no run of any LN family is trapped at B = 0.1, the claim becomes "low-authority plants".
Outputs: <out>/*.csv, <out>/fig_*.png, <out>/report.md (tables only, no prose)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import FAMILY, ORACLE, runs  # noqa: E402

RED, BLUE, GREY = "#8B1A1A", "#1B3A6B", "#888888"
FAMS = ["LN MSE 3000", "LN MSE 12000", "LN slope from 0", "LN two-stage", "no norm", "static norm"]
LQR_FAM = {"LN MSE 3000": "LN MSE 3000", "LN MSE 12000": "LN MSE 12000", "LN slope from 0": "LN slope from step 0",
           "LN two-stage": "LN two-stage", "no norm": "no norm", "static norm": "static norm"}
WS_BINS, WS_LABELS = [-0.01, 0.0, 0.02, 0.1, 1.0], ["0", "0-2 %", "2-10 %", "> 10 %"]
R2_FAMS = ["LN two-stage", "no norm", "static norm"]


def md(df, fmt="{:.3g}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------- LQR
def lqr(fn):
    T = pd.read_csv(fn)
    T = T[(T.x0 - T.r).abs() >= 0.5].copy()
    T["failed"] = T.diverged | (T.gap > 0.1)
    T["catastrophic"] = T.diverged | (T.gap > 1.0)
    return T


def lqr_rates(T, tag):
    rows = []
    for f in FAMS:
        for c in ("LQR fixed", "LQR sched"):
            g = T[(T.model == LQR_FAM[f]) & (T.controller == c)]
            rows.append(dict(plant=tag, family=f, controller=c, n=len(g), failed_pct=100 * g.failed.mean(),
                             catastrophic_pct=100 * g.catastrophic.mean()))
    return pd.DataFrame(rows)


def setpoint_law(T, B, tag):
    """|B_hat/B - 1| against the setpoint distance d from the kink centre (half-widths), fixed-LQR rows of the four LN
    families (one row per seed x setpoint); log-log least squares over d > 0, finite errors > 0."""
    g = T[(T.controller == "LQR fixed") & T.model.str.startswith("LN")].drop_duplicates(["model", "seed", "r"])
    e, d = np.abs(g.Bh / B - 1).values, g.setpoint_kink_dist.values
    ok = np.isfinite(e) & np.isfinite(d) & (e > 0) & (d > 0)
    slope, icpt = np.polyfit(np.log(d[ok]), np.log(e[ok]), 1)
    return dict(plant=tag, n=int(ok.sum()), coef=float(np.exp(icpt)), exponent=float(slope)), g.assign(err=e, d=d)


def path_colour(T, B, tag, good=0.2):
    g = T[T.model.str.startswith("LN")].copy()
    g["ws_bin"] = pd.cut(g.wrong_sign_share, WS_BINS, labels=WS_LABELS)
    g = g[np.abs(g.Bh / B - 1) < good]
    rows = []
    for c in ("LQR fixed", "LQR sched"):
        for b in WS_LABELS:
            h = g[(g.controller == c) & (g.ws_bin == b)]
            rows.append(dict(plant=tag, controller=c, wrong_sign_share=b, n=len(h),
                             failed_pct=100 * h.failed.mean() if len(h) else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- DPC
def dpc(root):
    R = runs(root)
    extra = R[~R.model.isin(list(FAMILY) + [ORACLE])].copy()
    R = R[R.model.isin(FAMILY)].copy()
    R["family"] = R.model.map(FAMILY)
    return R, extra


def check_rows(R):
    """exactly one row per family x seed x repeat (10 x 3) and a diag value for each."""
    c = R.groupby("family").size().reindex(FAMS)
    dup = int(R.duplicated(["family", "seed", "rep"]).sum())
    return dict(rows_per_family=c.to_dict(), duplicates=dup, diag_missing=int(R.oracle_on_model.isna().sum()))


def dpc_counts(R, tag):
    rows = []
    for f in FAMS:
        g = R[R.family == f]
        ps = g.groupby("seed").gap.median()
        rows.append(dict(plant=tag, family=f, runs=len(g), failed=int(g.failed.sum()), trapped=int(g.trapped.sum()),
                         median_gap_pct=100 * g.gap.median() if len(g) else np.nan,
                         best_seed_gap_pct=100 * ps.min() if len(ps) else np.nan,
                         trapped_seeds=" ".join(f"s{s}r{r}" for s, r in zip(g[g.trapped].seed, g[g.trapped].rep))))
    return pd.DataFrame(rows)


def load_pred(fn, stat, thr):
    P = pd.read_csv(fn)
    if stat not in P:
        for alt in ("S_seg_max", "Lambda_star", "L_aug_star"):
            if alt in P:
                P = P.rename(columns={alt: stat})
                break
    if "predicted" not in P:
        P["predicted"] = P[stat] > thr
    P["predicted"] = P.predicted.astype(bool)
    return P[["family", "seed", "rep", stat, "predicted"]]


def confusion(R, P, stat, name):
    J = R.merge(P, on=["family", "seed", "rep"], how="inner")
    tp = int((J.predicted & J.trapped).sum()); fn = int((~J.predicted & J.trapped).sum())
    fp = int((J.predicted & ~J.trapped).sum()); tn = int((~J.predicted & ~J.trapped).sum())
    hit = tp / (tp + fn) if tp + fn else np.nan
    fa = fp / (fp + tn) if fp + tn else np.nan
    npred, ntrap = tp + fp, tp + fn
    if npred == 0:
        outcome = "held" if ntrap <= 1 else "failed"
    elif ntrap == 0:
        outcome = "held (false alarms only)" if fa <= 0.05 else "failed"
    else:
        ok_h, ok_f = hit >= 0.75, fa <= 0.05
        outcome = "held" if ok_h and ok_f else ("partly" if ok_h or ok_f else "failed")
    lst = lambda m: "; ".join(f"{r.family} s{r.seed} r{r.rep} ({stat} {r[stat]:.2f})" for _, r in J[m].iterrows()) or "none"
    return dict(statistic=name, runs=len(J), predicted=npred, trapped=ntrap, hits=tp, misses=fn, false_alarms=fp,
                correct_rejections=tn, hit_rate=hit, false_alarm_rate=fa, outcome=outcome,
                missed=lst(~J.predicted & J.trapped), false_alarm_runs=lst(J.predicted & ~J.trapped)), J


# ----------------------------------------------------------------------------------------------- figures
def fig_lqr(L, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for a, c in zip(ax, ("LQR fixed", "LQR sched")):
        x = np.arange(len(FAMS)); w = 0.2
        for i, (tag, col) in enumerate(zip(sorted(L.plant.unique()), (BLUE, RED))):
            g = L[(L.plant == tag) & (L.controller == c)].set_index("family").reindex(FAMS)
            a.bar(x + (2 * i - 1) * w, g.failed_pct, w * 0.95, color=col, alpha=0.45, label=f"{tag}: failed")
            a.bar(x + (2 * i - 1) * w, g.catastrophic_pct, w * 0.95, color=col, label=f"{tag}: catastrophic")
        a.set_xticks(x); a.set_xticklabels(FAMS, rotation=25, ha="right", fontsize=8); a.set_title(c, fontsize=10)
        a.grid(alpha=0.3, axis="y")
    ax[0].set_ylabel("% of runs with |x0 - r| >= 0.5"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_lqr_rates.png"), dpi=150, facecolor="white"); plt.close(fig)


def fig_setpoint(pts, laws, out):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for (tag, g), col in zip(sorted(pts.items()), (BLUE, RED)):
        ok = (g.err > 0) & (g.d > 0)
        ax.loglog(g.d[ok], g.err[ok], ".", ms=3, color=col, alpha=0.5)
        L = laws[tag]; dd = np.logspace(np.log10(g.d[ok].min()), np.log10(g.d[ok].max()), 50)
        ax.loglog(dd, L["coef"] * dd ** L["exponent"], color=col, label=f"{tag}: {L['coef']:.2f} d^{L['exponent']:.2f}")
    ax.set_xlabel("setpoint distance from the kink centre d (half-widths)"); ax.set_ylabel("|B_hat / B - 1| at the setpoint")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_setpoint_law.png"), dpi=150, facecolor="white"); plt.close(fig)


def fig_traps(D, out):
    fig, ax = plt.subplots(figsize=(7, 3.4)); x = np.arange(len(FAMS)); w = 0.35
    for i, (tag, col) in enumerate(zip(sorted(D.plant.unique()), (BLUE, RED))):
        g = D[D.plant == tag].set_index("family").reindex(FAMS)
        ax.bar(x + (i - 0.5) * w, g.trapped, w * 0.95, color=col, label=f"{tag}: trapped")
        ax.bar(x + (i - 0.5) * w, g.failed - g.trapped, w * 0.95, bottom=g.trapped, color=col, alpha=0.35,
               label=f"{tag}: failed, not trapped")
    ax.set_xticks(x); ax.set_xticklabels(FAMS, rotation=25, ha="right", fontsize=8); ax.set_ylabel("runs (of 30)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_dpc_traps.png"), dpi=150, facecolor="white"); plt.close(fig)


def fig_lambda(sets, out):
    """Lambda statistic per run against the typical-start gap; trapped red, other runs grey; threshold line."""
    fig, ax = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.8), squeeze=False)
    for a, (title, J, stat, thr) in zip(ax[0], sets):
        g = np.clip(100 * J.gap.values, 1e-3, None)
        a.semilogy(J[stat][~J.trapped], g[~J.trapped.values], "o", ms=3, color=GREY, label="not trapped")
        a.semilogy(J[stat][J.trapped], g[J.trapped.values], "o", ms=4, color=RED, label="trapped")
        a.axvline(thr, color="k", lw=0.8, ls="--"); a.axhline(10, color="k", lw=0.5, ls=":")
        a.set_xlabel(stat); a.set_title(title, fontsize=9); a.grid(alpha=0.3)
    ax[0][0].set_ylabel("typical-start gap (%)"); ax[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_lambda_scatter.png"), dpi=150, facecolor="white"); plt.close(fig)


# ----------------------------------------------------------------------------------------------- main
def main(a):
    os.makedirs(a.out, exist_ok=True)
    tag, rtag = f"B = {a.B:g}", f"B = {a.ref_B:g}"
    if tag == rtag:                                   # end-to-end test on the reference data itself
        tag += " (under test)"
    T, Tr = lqr(a.lqr), lqr(a.ref_lqr)
    L = pd.concat([lqr_rates(Tr, rtag), lqr_rates(T, tag)])
    law, pts = setpoint_law(T, a.B, tag); lawr, ptsr = setpoint_law(Tr, a.ref_B, rtag)
    PCo = pd.concat([path_colour(Tr, a.ref_B, rtag), path_colour(T, a.B, tag)])
    R, extra = dpc(a.root); Rr, _ = dpc(a.ref_root)
    chk = check_rows(R)
    D = pd.concat([dpc_counts(Rr, rtag), dpc_counts(R, tag)])
    Pd, Pa = load_pred(a.pred_direct, "Lambda_direct", a.thr_direct), load_pred(a.pred_aug, "Lambda_aug", a.thr_aug)
    cd, Jd = confusion(R, Pd, "Lambda_direct", "direct-gain Lambda* > 12.1 (R1, primary)")
    ca, Ja = confusion(R, Pa, "Lambda_aug", "augmented Lambda*_aug > 12.08 (secondary)")
    r2 = R[R.family.isin(R2_FAMS)]
    r2_out = dict(trapped=int(r2.trapped.sum()), runs=len(r2), outcome="held" if r2.trapped.sum() == 0 else "failed",
                  runs_trapped="; ".join(f"{r.family} s{r.seed} r{r.rep}" for _, r in r2[r2.trapped].iterrows()) or "none")
    ln_trapped = int(R[R.family.str.startswith("LN")].trapped.sum())
    decision = ("traps vanish at this plant: the claim becomes 'low-authority plants' (CLAUDE.md decision rule)" if ln_trapped == 0
                else f"traps persist ({ln_trapped} trapped LN runs): the claim is not restricted to low-authority plants")
    Lr = pd.read_csv(a.ref_lam); Lr = Lr.rename(columns={"S_seg_max": "Lambda_direct"})
    Jr = Rr.merge(Lr[["family", "seed", "rep", "Lambda_direct"]], on=["family", "seed", "rep"])
    fig_lqr(L, a.out); fig_setpoint({rtag: ptsr, tag: pts}, {rtag: lawr, tag: law}, a.out); fig_traps(D, a.out)
    fig_lambda([(f"{rtag}: direct-gain Lambda*", Jr, "Lambda_direct", 12.1), (f"{tag}: direct-gain Lambda*", Jd, "Lambda_direct", a.thr_direct),
                (f"{tag}: augmented Lambda*_aug", Ja, "Lambda_aug", a.thr_aug)], a.out)
    for df, n in ((L, "lqr_rates"), (pd.DataFrame([lawr, law]), "setpoint_law"), (PCo, "path_colour"), (D, "dpc_counts"),
                  (pd.DataFrame([cd, ca]), "confusion"), (Jd, "runs_direct"), (Ja, "runs_aug"), (extra, "excluded_rows")):
        df.to_csv(os.path.join(a.out, f"{n}.csv"), index=False)
    C = pd.DataFrame([cd, ca])
    txt = [f"# B01 report (analysis/b01_report.py), root {a.root}", "", f"Row check: {chk}", "",
           f"Excluded rows (not a registered family, exploratory): {len(extra)} ({', '.join(sorted(extra.model.unique())) or 'none'})", "",
           "## LQR failure rates", md(L), "", "## Setpoint law", md(pd.DataFrame([lawr, law])), "", "## Path colour",
           md(PCo), "", "## DPC per family", md(D), "", "## Confusion matrices",
           md(C[["statistic", "runs", "predicted", "trapped", "hits", "misses", "false_alarms", "correct_rejections",
                 "hit_rate", "false_alarm_rate", "outcome"]]), "",
           f"Misses (direct): {cd['missed']}", f"False alarms (direct): {cd['false_alarm_runs']}",
           f"Misses (aug): {ca['missed']}", f"False alarms (aug): {ca['false_alarm_runs']}", "",
           "## Outcomes", "| id | outcome | numbers |", "|---|---|---|",
           f"| R1 | {cd['outcome']} | {cd['hits']} hits, {cd['misses']} misses, {cd['false_alarms']} false alarms of {cd['runs']} runs |",
           f"| R2 | {r2_out['outcome']} | {r2_out['trapped']} of {r2_out['runs']} two-stage / no norm / static norm runs trapped ({r2_out['runs_trapped']}) |",
           f"| aug rule | {ca['outcome']} | {ca['hits']} hits, {ca['misses']} misses, {ca['false_alarms']} false alarms |", "",
           f"Decision rule: {decision}"]
    open(os.path.join(a.out, "report.md"), "w").write("\n".join(txt))
    print("\n".join(txt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(paths.REPO, "data", "dpc_out_B01"))
    ap.add_argument("--lqr", default=paths.out("B01", "companion", "lqr_study_extra.csv"))
    ap.add_argument("--B", type=float, default=0.1)
    ap.add_argument("--pred-direct", default=os.path.join(paths.REPO, "prereg", "PREREG_B01_predicted.csv"))
    ap.add_argument("--pred-aug", default=os.path.join(paths.REPO, "prereg", "PREREG_B01_aug_predicted.csv"))
    ap.add_argument("--thr-direct", type=float, default=12.1)
    ap.add_argument("--thr-aug", type=float, default=12.08)
    ap.add_argument("--ref-root", default=os.path.join(paths.REPO, "data", "dpc_out"))
    ap.add_argument("--ref-lqr", default=os.path.join(paths.REPO, "companion", "lqr_study.csv"))
    ap.add_argument("--ref-B", type=float, default=0.01)
    ap.add_argument("--ref-lam", default=paths.out("theory", "t2_runs_all_B001.csv"))
    ap.add_argument("--out", default=paths.out("B01", "report"))
    main(ap.parse_args())
