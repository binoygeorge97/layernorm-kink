"""Q10: regenerate every paper figure from the CSVs (no computation beyond reading and plotting).
python paper/make_figs.py  -> paper/figs/*.png (+ .pdf). A figure whose input is missing is drawn as a labelled placeholder
("pending: <run>"), so the command always succeeds and shows which runs are still needed.
House style (CLAUDE.md 4): white background; dark red #8B1A1A bad / trapped / overestimated input slope; dark blue #1B3A6B
good / wrong-sign or too-small input slope; grey neutral; slope maps RdBu_r with TwoSlopeNorm(-3, 1, 5) for J_u/B.
Figures
  F1 fig_lqr_rates          Part 1: fixed / rescheduled LQR failed and catastrophic per family, seed-bootstrap 95 % CI (out/stats)
  F2 fig_kink_geometry      half-widths r*_x, r*_u and peak |J_u_hat/B - 1| per seed, LN families (data/dpc_out/surrogates.csv)
  F3 fig_swap               LQR swap test: failed / catastrophic with surrogate vs true A, B (out/stats/swap_rates_boot.csv)
  F4 fig_dpc_traps          DPC failed / trapped per family and architecture: GRU, MLP (Run 5a), linear (Run 5b), GRU at B = 0.1 (Run 6),
                            held-out seeds (Run 8, pending), RMSNorm (Run 9, pending)
  F5 fig_lambda_scatter     direct-gain Lambda* vs typical-start gap, threshold 12.1: GRU B = 0.01, MLP, linear, GRU B = 0.1
  F6 fig_optimiser          Task 3.5 from scratch (Run 4, pending until out/opt/from_scratch.csv is complete) + warm start (local)
  F7 fig_b01_side_by_side   B = 0.01 vs B = 0.1: LQR rates and DPC traps (out/B01/report)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RED, BLUE, GREY = "#8B1A1A", "#1B3A6B", "#888888"
OUT = os.path.join(HERE, "figs")
FAM = ["LN MSE 3000", "LN MSE 12000", "LN slope from 0", "LN two-stage", "no norm", "static norm"]
LQRF = {"LN slope from 0": "LN slope from step 0"}
R = lambda *a: os.path.join(paths.REPO, *a)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 9})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def placeholder(ax, text):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color=GREY, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GREY); s.set_linestyle(":")


def exists(*fs):
    return all(os.path.exists(f) for f in fs)


def f1():
    fn = R("out", "stats", "lqr_rates_boot.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    if not exists(fn):
        for a in ax: placeholder(a, "pending: analysis/stats.py")
        return save(fig, "fig_lqr_rates")
    T = pd.read_csv(fn)
    x = np.arange(len(FAM))
    for a, c in zip(ax, ("LQR fixed", "LQR sched")):
        for k, (m, col, off) in enumerate((("failed", RED, -0.18), ("catastrophic", BLUE, 0.18))):
            g = T[(T.controller == c) & (T.metric == m)].set_index("family").reindex([LQRF.get(f, f) for f in FAM])
            a.bar(x + off, g.pct, 0.34, color=col, label=m)
            a.errorbar(x + off, g.pct, yerr=[g.pct - g.ci_lo, g.ci_hi - g.pct], fmt="none", ecolor="k", lw=0.8, capsize=2)
        a.set_xticks(x); a.set_xticklabels(FAM, rotation=30, ha="right"); a.set_title(c); a.grid(alpha=0.3, axis="y")
    ax[0].set_ylabel("% of runs (|x0 - r| >= 0.5), 95 % seed-bootstrap CI"); ax[0].legend()
    save(fig, "fig_lqr_rates")


def f2():
    fn = R("data", "dpc_out", "surrogates.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    if not exists(fn):
        for a in ax: placeholder(a, "pending: surrogates.csv")
        return save(fig, "fig_kink_geometry")
    S = pd.read_csv(fn)
    lab = {"LN, MSE (3000)": "LN MSE 3000", "LN, MSE (12000)": "LN MSE 12000", "LN, J_x + J_u (3000)": "LN slope from 0",
           "LN, two-stage": "LN two-stage"}
    for i, (m, l) in enumerate(lab.items()):
        g = S[S.model == m].sort_values("seed")
        ax[0].plot(g.seed + 0.15 * (i - 1.5), g.rstar_x, "o", ms=4, color=[RED, GREY, BLUE, "k"][i], label=l)
        ax[1].semilogy(g.seed + 0.15 * (i - 1.5), g.Ju_sup_rel, "o", ms=4, color=[RED, GREY, BLUE, "k"][i], label=l)
    ax[0].set_ylabel("kink half-width r*_x"); ax[1].set_ylabel("peak |J_u_hat / B - 1|")
    for a in ax: a.set_xlabel("seed"); a.set_xticks(range(10)); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    save(fig, "fig_kink_geometry")


def f3():
    fn = R("out", "stats", "swap_rates_boot.csv")
    fig, ax = plt.subplots(figsize=(5, 3.3))
    if not exists(fn):
        placeholder(ax, "pending: swap test (analysis/stats.py)"); return save(fig, "fig_swap")
    T = pd.read_csv(fn); T = T[T.family == "LN MSE 3000"]
    des = list(dict.fromkeys(T.design)); x = np.arange(len(des))
    for m, col, off in (("failed", RED, -0.18), ("catastrophic", BLUE, 0.18)):
        g = T[T.metric == m].set_index("design").reindex(des)
        ax.bar(x + off, g.pct, 0.34, color=col, label=m)
        ax.errorbar(x + off, g.pct, yerr=[g.pct - g.ci_lo, g.ci_hi - g.pct], fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(des, rotation=20, ha="right"); ax.set_ylabel("% of runs, LN MSE 3000, fixed LQR")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig_swap")


SETS = [("GRU, B = 0.01", R("data", "dpc_out"), None), ("MLP (Run 5a)", R("data", "run5_mlp"), None),
        ("linear (Run 5b)", R("data", "run5_linear"), None), ("GRU, B = 0.1 (Run 6)", R("data", "run6_B01"), None),
        ("GRU, held-out seeds 10-29 (Run 8)", R("data", "run8_s10_29"), "pending: Run 8"),
        ("GRU, RMSNorm (Run 9)", R("data", "run9_rms"), "pending: Run 9")]


def dpc_counts(root):
    from common import runs, FAMILY
    Rr = runs(root); Rr = Rr[Rr.model.isin(FAMILY)].assign(family=lambda d: d.model.map(FAMILY))
    return Rr


def f4():
    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for a, (title, root, pend) in zip(ax.flat, SETS):
        if not exists(os.path.join(root, "dpc_results.csv"), os.path.join(root, "dpc_diag.csv")):
            placeholder(a, pend or "pending"); a.set_title(title); continue
        Rr = dpc_counts(root)
        fams = [f for f in FAM if f in set(Rr.family)] + sorted(set(Rr.family) - set(FAM))
        g = Rr.groupby("family").agg(failed=("failed", "sum"), trapped=("trapped", "sum")).reindex(fams)
        x = np.arange(len(fams))
        a.bar(x, g.trapped, 0.6, color=RED, label="trapped")
        a.bar(x, g.failed - g.trapped, 0.6, bottom=g.trapped, color=GREY, label="failed, not trapped")
        a.set_xticks(x); a.set_xticklabels(fams, rotation=30, ha="right", fontsize=7); a.set_title(title); a.grid(alpha=0.3, axis="y")
    ax[0, 0].set_ylabel("runs of 30"); ax[1, 0].set_ylabel("runs of 30"); ax[0, 0].legend(fontsize=7)
    save(fig, "fig_dpc_traps")


def f5():
    from common import runs, FAMILY
    specs = [("GRU, B = 0.01 (retrospective)", R("data", "dpc_out"), R("out", "theory", "t2_runs_all_B001.csv"), "S_seg_max", None),
             ("MLP (prospective, f945a5c)", R("data", "run5_mlp"), R("prereg", "PREREG_arch_predicted.csv"), "Lambda_star", "mlp"),
             ("linear (prospective, 322e353)", R("data", "run5_linear"), R("prereg", "PREREG_arch_predicted.csv"), "Lambda_star", "linear"),
             ("GRU, B = 0.1 (prospective, 77d8455)", R("data", "run6_B01"), R("prereg", "PREREG_B01_predicted.csv"), "S_seg_max", None)]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for a, (title, root, lam, col, arch) in zip(ax, specs):
        if not exists(os.path.join(root, "dpc_results.csv"), lam):
            placeholder(a, "pending"); a.set_title(title, fontsize=9); continue
        Rr = runs(root); Rr = Rr[Rr.model.isin(FAMILY)].assign(family=lambda d: d.model.map(FAMILY))
        L = pd.read_csv(lam)
        if arch:
            L = L[L.arch == arch]
        J = Rr.merge(L[["family", "seed", "rep", col]], on=["family", "seed", "rep"])
        g = np.clip(100 * J.gap.values, 1e-3, None); t = J.trapped.values
        a.semilogy(J[col][~t], g[~t], "o", ms=3, color=GREY, label="not trapped")
        a.semilogy(J[col][t], g[t], "o", ms=4, color=RED, label="trapped")
        a.axvline(12.1, color="k", ls="--", lw=0.8); a.axhline(10, color="k", ls=":", lw=0.5)
        a.set_title(title, fontsize=9); a.set_xlabel("Lambda* of the oracle"); a.grid(alpha=0.3)
    ax[0].set_ylabel("typical-start gap (%)"); ax[0].legend(fontsize=7)
    save(fig, "fig_lambda_scatter")


def f6():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    fn = R("out", "opt", "from_scratch.csv")
    T = pd.read_csv(fn) if exists(fn) else None
    if T is None or (T.runs.fillna(0) < 30).any():
        placeholder(ax[0], "pending: Run 4 (optimiser baselines from scratch)\nre-run analysis/opt_report.py first")
    else:
        x = np.arange(len(T))
        ax[0].bar(x, T.trapped, 0.6, color=RED, label="trapped"); ax[0].bar(x, T.failed - T.trapped, 0.6, bottom=T.trapped, color=GREY, label="failed, not trapped")
        ax[0].set_xticks(x); ax[0].set_xticklabels(T.setting, rotation=30, ha="right"); ax[0].set_ylabel("runs of 30"); ax[0].legend(fontsize=7)
    ax[0].set_title("From scratch, LN MSE 3000 (GRU)")
    fw = R("out", "opt", "warmstart_table.csv")
    if exists(fw):
        W = pd.read_csv(fw)
        b = W[W.broken].groupby("setting").expelled.sum(); h = W[~W.broken].groupby("setting").expelled.sum()
        order = [s for s in ["plain", "clip3", "unit", "global3", "mom", "batch100", "lr1e-4"] if s in b.index]
        x = np.arange(len(order))
        ax[1].bar(x - 0.18, b.reindex(order), 0.34, color=RED, label="broken pairs expelled (of 6)")
        ax[1].bar(x + 0.18, h.reindex(order), 0.34, color=BLUE, label="healthy pairs expelled (of 4)")
        ax[1].set_xticks(x); ax[1].set_xticklabels(order); ax[1].legend(fontsize=7); ax[1].set_ylabel("pairs")
    else:
        placeholder(ax[1], "pending: warm-start harness")
    ax[1].set_title("Warm start at the oracle, 100 epochs")
    for a in ax: a.grid(alpha=0.3, axis="y")
    save(fig, "fig_optimiser")


def f7():
    d = R("out", "B01", "report")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    if not exists(os.path.join(d, "lqr_rates.csv"), os.path.join(d, "dpc_counts.csv")):
        for a in ax: placeholder(a, "pending: Run 6 report")
        return save(fig, "fig_b01_side_by_side")
    L, D = pd.read_csv(os.path.join(d, "lqr_rates.csv")), pd.read_csv(os.path.join(d, "dpc_counts.csv"))
    x = np.arange(len(FAM))
    for i, (pl, col) in enumerate(zip(["B = 0.01", "B = 0.1"], (BLUE, RED))):
        g = L[(L.plant == pl) & (L.controller == "LQR fixed")].set_index("family").reindex(FAM)
        ax[0].bar(x + (i - 0.5) * 0.38, g.failed_pct, 0.36, color=col, label=pl)
        h = D[D.plant == pl].set_index("family").reindex(FAM)
        ax[1].bar(x + (i - 0.5) * 0.38, h.trapped, 0.36, color=col, label=pl)
    ax[0].set_ylabel("fixed LQR failed (%)"); ax[1].set_ylabel("DPC trapped (of 30)")
    for a in ax: a.set_xticks(x); a.set_xticklabels(FAM, rotation=30, ha="right"); a.grid(alpha=0.3, axis="y"); a.legend(fontsize=7)
    save(fig, "fig_b01_side_by_side")


if __name__ == "__main__":
    for f in (f1, f2, f3, f4, f5, f6, f7):
        f()
