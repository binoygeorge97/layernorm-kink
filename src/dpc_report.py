"""dpc_report.py - tables, paired statistics and figures for the DPC-through-the-surrogate experiment."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse
from scipy.stats import wilcoxon, spearmanr
import dpc_core as D

# One colour per model, fixed (validated order: adjacent pairs separable in colour-vision deficiency). LayerNorm + MSE
# is the neutral grey baseline (the problem), the oracle is black. Identity is also in every tick label / legend.
COL = {"true plant (oracle)": "#0b0b0b", "LN, MSE (3000)": "#6f6e69", "LN, MSE (12000)": "#eda100",
       "LN, J_x + J_u (3000)": "#1baf7a", "LN, two-stage": "#6250d6", "no norm, MSE": "#eb6834",
       "static norm, MSE": "#2a78d6"}
LS = {"true plant (oracle)": "-", "LN, MSE (3000)": "-", "LN, MSE (12000)": "--", "LN, J_x + J_u (3000)": "-.",
      "LN, two-stage": "-", "no norm, MSE": ":", "static norm, MSE": "--"}
ORDER = [m["name"] for m in D.MODELS]
SHORT = {m["name"]: m["short"] for m in D.MODELS}
KINKFREE = ("static norm, MSE", "no norm, MSE")
SMALL, LARGE = 0.2, 1.0          # |x0| <= 0.2 = small starts (begin inside or next to the kink); |x0| >= 1 = large

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.titlesize": 9, "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 160})


def _save(fig, out_dir, name):
    d = os.path.join(out_dir, "figs")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")


def _rep(df):
    if "rep" not in df:
        df["rep"] = 0
    df["rep"] = df["rep"].fillna(0).astype(int)
    return df


FAIL = 10.0     # a run "fails" when its true-plant cost on the typical starts is more than 10 % above the oracle's


def load(out_dir):
    R = _rep(pd.read_csv(os.path.join(out_dir, "dpc_results.csv"))).drop_duplicates(["model", "seed", "rep"], keep="last")
    C = _rep(pd.read_csv(os.path.join(out_dir, "dpc_curves.csv"))).drop_duplicates(["model", "seed", "rep", "epoch"], keep="last")
    sp = os.path.join(out_dir, "surrogates.csv")
    S = pd.read_csv(sp) if os.path.exists(sp) else pd.DataFrame()
    return R, C, S


def traj(out_dir, model, rep=0):
    fn = D.rep_file(os.path.join(out_dir, "trajectories"), model, rep)
    return np.load(fn) if os.path.exists(fn) else None


def reps_of(out_dir):
    """Repeats that have trajectories on disk (0 = the original run)."""
    import glob, re
    found = {0}
    for f in glob.glob(os.path.join(out_dir, "trajectories", "*__rep*.npz")):
        found.add(int(re.search(r"__rep(\d+)\.npz$", f).group(1)))
    return sorted(found)


def per_start(out_dir, models=None, tail=20):
    """One row per (model, seed, start x0) from the saved true-plant trajectories:
      J        total cost  sum_k (Q x_k^2 + R u_k^2) + Q_f x_N^2   (what the true plant pays)
      offset   x_inf = mean of x over the last `tail` steps (a steady error left by a biased model)
      J_dyn    the same cost about the controller's own resting point (x - x_inf, u - u_inf): the transient only
      ratio, ratio_dyn = J / J_oracle on the same seed and start (1 = as good as the oracle)."""
    rows = []
    for rep in reps_of(out_dir):
        for m in models or ORDER:
            T = traj(out_dir, m, rep)
            if T is None:
                continue
            XE, UE = T["XE"].astype(float), T["UE"].astype(float)
            for i, s in enumerate(T["seeds"]):
                x, u = XE[i], UE[i]
                xi, ui = x[-tail:].mean(0), u[-tail:].mean(0)
                J = D.per_start_cost(x, u)
                Jd = D.per_start_cost(x - xi, u - ui)
                for j, x0 in enumerate(T["x0"]):
                    rows.append(dict(model=m, seed=int(s), rep=rep, x0=float(x0), J=J[j], J_dyn=Jd[j], offset=xi[j],
                                     final=x[-1, j]))
    P = pd.DataFrame(rows)
    o = P[P.model == "true plant (oracle)"].set_index(["seed", "rep", "x0"])
    key = list(zip(P.seed, P.rep, P.x0))
    P["ratio"] = P.J.values / o.loc[key, "J"].values
    P["ratio_dyn"] = P.J_dyn.values / o.loc[key, "J_dyn"].values
    return P


def add_kink_share(R, out_dir):
    """Share of the 22 evaluation trajectories' steps (x_k, u_k) inside each LayerNorm surrogate's closed-form kink ellipse,
    computed from the saved trajectories (runs before the column fix lost it from dpc_results.csv)."""
    sp = os.path.join(out_dir, "surrogates.csv")
    if not os.path.exists(sp):
        return R
    S = pd.read_csv(sp)
    rows = []
    for rep in reps_of(out_dir):
        for m in [m for m in ORDER if m in set(S.model)]:
            T = traj(out_dir, m, rep)
            if T is None:
                continue
            for i, s in enumerate(T["seeds"]):
                g = S[(S.model == m) & (S.seed == s)]
                if not len(g) or not np.isfinite(g.rstar_x.iloc[0]):
                    continue
                g = g.iloc[0]
                ins = ((T["XE"][i, :-1] - g.zmin_x) / g.rstar_x) ** 2 + ((T["UE"][i] - g.zmin_u) / g.rstar_u) ** 2 <= 1
                rows.append(dict(model=m, seed=int(s), rep=rep, kink_share=float(ins.mean()),
                                 kink_share_small=float(ins[:, np.abs(T["x0"]) <= SMALL].mean())))
    if not rows:
        return R
    R = R.drop(columns=[c for c in ("kink_share", "kink_share_small") if c in R])
    return R.merge(pd.DataFrame(rows), on=["model", "seed", "rep"], how="left")


def add_gaps(R, P):
    """Per (model, seed): cost gaps vs the oracle controller of the same seed, in %.
      gap_test   true-plant cost on the 1000 typical starts (the example's headline number)
      gap_small  median over small starts (|x0| <= 0.2) of J / J_oracle - 1     (offset included)
      gap_small_dyn  the same for the transient only (offset removed)
      gap_large  median over large starts (|x0| >= 1)
      gap_worst  worst start, transient only (offset removed; the offset has its own column)
      offset     median |x_inf| over the 22 starts (steady error)"""
    R = R.copy()
    o = R[R.model == "true plant (oracle)"].set_index(["seed", "rep"]).cost_test_true
    R["gap_test"] = 100 * (R.cost_test_true / o.loc[list(zip(R.seed, R.rep))].values - 1)
    a = np.abs(P.x0)
    g = lambda mask, col, f: (P[mask].groupby(["model", "seed", "rep"])[col].agg(f) - 1) * 100
    add = pd.DataFrame({"gap_small": g(a <= SMALL, "ratio", "median"), "gap_small_dyn": g(a <= SMALL, "ratio_dyn", "median"),
                        "gap_large": g(a >= LARGE, "ratio", "median"), "gap_worst": g(a > 0, "ratio_dyn", "max"),
                        "offset": P.assign(o=P.offset.abs()).groupby(["model", "seed", "rep"]).o.median()})
    return R.merge(add.reset_index(), on=["model", "seed", "rep"], how="left")


def lqr_reference(plant, out_dir=None):
    """Analytic LQR with saturation on the same 1000 typical starts: the true-plant cost a perfect linear design gets."""
    x0 = np.random.default_rng(12345).uniform(-D.X0_RANGE, D.X0_RANGE, D.N_TEST)
    xs, us = D.lqr_saturated(plant, x0)
    return D.true_lqr_cost(xs, us)


def _q(s):
    s = pd.Series(s).dropna()
    return f"{np.median(s):.3g} [{np.percentile(s, 25):.2g}, {np.percentile(s, 75):.2g}]" if len(s) else "-"


def summary_table(R, out_dir):
    rows = []
    for m in [m for m in ORDER if m in set(R.model)]:
        g = R[R.model == m]
        row = {"model": m, "runs (seeds x repeats)": f"{g.seed.nunique()} x {g.rep.nunique()}",
               "true-plant cost (typical starts)": _q(g.cost_test_true),
               "cost on its own model": _q(g.cost_test_own)}
        if m != "true plant (oracle)":
            for k, lab in (("gap_test", "gap typical %"), ("gap_small", "gap small starts %"),
                           ("gap_small_dyn", "gap small, transient only %"), ("gap_large", "gap large starts %"),
                           ("gap_worst", "gap worst start, transient only %")):
                row[lab] = _q(g[k])
                if k == "gap_worst":
                    row["gap worst start: worst run %"] = f"{g[k].max():.3g}"
            f = g.gap_test > FAIL
            per_seed = f.groupby(g.seed).agg(["sum", "count"])
            row[f"runs failed (gap > {FAIL:g} %)"] = f"{int(f.sum())}/{len(g)}"
            row["seeds failing in some / every repeat"] = (f"{int((per_seed['sum'] > 0).sum())} / "
                                                           f"{int((per_seed['sum'] == per_seed['count']).sum())}  "
                                                           f"(seeds {', '.join(str(s) for s in per_seed.index[per_seed['sum'] > 0])})")
        row["steady offset |x_inf| (median)"] = _q(g.offset)
        row["diverged / not settled (of 22 starts x seeds)"] = f"{int(g.n_diverged.sum())} / {int(g.n_not_settled.sum())}"
        if "kink_share" in g and g.kink_share.notna().any():
            row["share of steps inside the kink (all / small starts)"] = f"{g.kink_share.median():.2f} / {g.kink_share_small.median():.2f}"
        rows.append(row)
    T = pd.DataFrame(rows)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    T.to_csv(os.path.join(out_dir, "tables", "dpc_summary.csv"), index=False)
    return T


COMPARISONS = [
    ("Does the kink hurt?", "LN, MSE (3000)", "static norm, MSE"),
    ("Does the kink hurt?", "LN, MSE (3000)", "no norm, MSE"),
    ("Does the slope loss help?", "LN, two-stage", "LN, MSE (3000)"),
    ("Does the slope loss help?", "LN, J_x + J_u (3000)", "LN, MSE (3000)"),
    ("Does it reach kink-free?", "LN, two-stage", "static norm, MSE"),
    ("Does it reach kink-free?", "LN, two-stage", "no norm, MSE"),
    ("Just train longer?", "LN, MSE (12000)", "LN, MSE (3000)"),
    ("Just train longer?", "LN, two-stage", "LN, MSE (12000)"),
]


def paired_table(R, out_dir, metrics=("gap_test", "gap_small_dyn", "gap_large", "gap_worst", "offset")):
    """Per seed (median over its repeats), A and B share the surrogate's data/init and the controllers' inits/initial states.
    Median of A - B
    (gaps are in %, so the difference is in percentage points), seeds where A is lower (better), Wilcoxon p."""
    rows = []
    Rs = R.groupby(["model", "seed"])[list(metrics)].median().reset_index()       # one value per seed: median over repeats
    for q, a, b in COMPARISONS:
        if a not in set(Rs.model) or b not in set(Rs.model):
            continue
        A, B = Rs[Rs.model == a].set_index("seed"), Rs[Rs.model == b].set_index("seed")
        s = sorted(set(A.index) & set(B.index))
        row = {"question": q, "A": a, "B": b, "n": len(s)}
        for m in metrics:
            x, y = A.loc[s, m].values, B.loc[s, m].values
            try:
                p = wilcoxon(x, y).pvalue if len(s) >= 5 and np.any(x != y) else np.nan
            except ValueError:
                p = np.nan
            row[f"{m}: median A-B"] = f"{np.median(x - y):.3g}"
            row[f"{m}: A better"] = f"{int(np.sum(x < y))}/{len(s)}"
            row[f"{m}: p"] = f"{p:.3g}" if np.isfinite(p) else "-"
        rows.append(row)
    T = pd.DataFrame(rows)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    T.to_csv(os.path.join(out_dir, "tables", "dpc_paired.csv"), index=False)
    return T


def _strip(ax, R, col, order, scale="symlog", band=True, f=1.0):
    rng = np.random.default_rng(0)
    ref = R[R.model.isin(KINKFREE)][col] * f
    if band and len(ref):
        ax.axhspan(ref.min(), ref.max(), color="#b3b2ab", alpha=0.25, lw=0, label="range of kink-free models")
    for i, m in enumerate(order):
        v = R[R.model == m][col].values * f
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=24, color=COL[m], edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#0b0b0b", lw=1.6, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[m] for m in order], fontsize=7)
    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=1)
        lo = np.nanmin(R[R.model.isin(order)][col].values * f)
        ax.set_ylim(bottom=-1.5 if lo > -1 else 1.5 * lo)
    elif scale == "log":
        ax.set_yscale("log")


def fig_headline(R, out_dir, lqr_cost=None):
    """Figure D1: one dot per seed, bar = median, grey band = range of the two kink-free models."""
    order = [m for m in ORDER if m in set(R.model) and m != "true plant (oracle)"]
    panels = [("gap_test", "true-plant cost gap vs oracle, typical starts (%)", "symlog"),
              ("gap_large", "cost gap, large starts |x0| >= 1 (median, %)", "symlog"),
              ("gap_small_dyn", "cost gap, small starts |x0| <= 0.2, transient only (median, %)", "symlog"),
              ("gap_worst", "cost gap, worst of the 22 starts, transient only (%)", "symlog"),
              ("offset", "steady offset |x_inf| (median over starts)", "log"),
              ("cost_ratio_own", "cost on its own model / on the true plant (typical starts)", "log")]
    R = R.assign(cost_ratio_own=R.cost_test_own / R.cost_test_true)
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6))
    for ax, (col, title, sc) in zip(axes.ravel(), panels):
        _strip(ax, R, col, order, sc)
        ax.set_title(title, fontsize=8.5)
        if col.startswith("gap"):
            ax.axhline(0, color="#8a8984", lw=0.8)
        if col == "gap_test" and lqr_cost is not None:
            o = R[R.model == "true plant (oracle)"].cost_test_true.median()
            yl = 100 * (lqr_cost / o - 1)
            ax.axhline(yl, color="#0b0b0b", ls="--", lw=1, label="analytic LQR with saturation")
            ax.set_ylim(bottom=min(ax.get_ylim()[0], 1.5 * yl if yl < 0 else ax.get_ylim()[0]))
        if col == "cost_ratio_own":
            ax.axhline(1, color="#8a8984", lw=0.8)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("DPC trained through each model, tested on the true plant (one dot per run = seed x repeat, bar = median; "
                 "gap = cost / cost of the same DPC trained through the true plant - 1). Axes: linear within +-1 %, log beyond.",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "figD1_headline")
    return fig


def worst_seed(R, model="LN, MSE (3000)", col="gap_small_dyn"):
    g = R[R.model == model].groupby("seed")[col].median()
    return int(g.idxmax()) if len(g) else 0


def fig_closed_loop(out_dir, seed, starts=(-2.0, -0.5, -0.05, 0.05, 0.5, 2.0), models=None):
    """Figure D2 (the example's plot_closed_loop, one column per model): x(t), |x(t)| (log) and u(t) on the TRUE plant.
    Thin black = the oracle controller from the same start."""
    models = [m for m in (models or ORDER) if m != "true plant (oracle)" and traj(out_dir, m) is not None]
    O = traj(out_dir, "true plant (oracle)")
    fig, axes = plt.subplots(3, len(models), figsize=(3.2 * len(models), 8.0), sharex=True, sharey="row", squeeze=False)
    cmap = plt.get_cmap("coolwarm")
    for j, m in enumerate(models):
        T = traj(out_dir, m)
        i = list(T["seeds"]).index(seed)
        io = list(O["seeds"]).index(seed)
        for k, x0 in enumerate(starts):
            jj = int(np.argmin(np.abs(T["x0"] - x0)))
            c = cmap(0.1 + 0.8 * k / max(1, len(starts) - 1))
            t = np.arange(T["XE"].shape[1])
            for r_, (y, yo) in enumerate(((T["XE"][i, :, jj], O["XE"][io, :, jj]), (np.abs(T["XE"][i, :, jj]) + 1e-9,
                                         np.abs(O["XE"][io, :, jj]) + 1e-9))):
                axes[r_, j].plot(t, yo, color="#0b0b0b", lw=0.7, alpha=0.6)
                axes[r_, j].plot(t, y, color=c, lw=1.5, label=f"x0 = {x0:g}" if j == 0 and r_ == 0 else None)
            axes[2, j].step(t[:-1], T["UE"][i, :, jj], where="post", color=c, lw=1.3)
        axes[0, j].set_title(m, fontsize=8.5, color=COL[m])
        axes[1, j].set_yscale("log")
        axes[2, j].set_xlabel("step k")
    axes[0, 0].set_ylabel("state x on the true plant")
    axes[1, 0].set_ylabel("|x| (log): offsets and slow decay")
    axes[2, 0].set_ylabel("control u")
    axes[0, 0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle(f"Seed {seed}: DPC controllers trained through each model, run on the TRUE plant "
                 "(thin black = DPC trained through the true plant, same start)", fontsize=9.5, y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, f"figD2_closed_loop_seed{seed}")
    return fig


def fig_norm(out_dir, models=None):
    """Figure D3 (the example's plot_norm_comparison): median over seeds of the mean |x(t)| on the true plant, separately
    for small starts (inside / next to the kink) and large starts."""
    models = [m for m in (models or ORDER) if traj(out_dir, m) is not None]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
    for ax, (mask_f, title) in zip(axes, ((lambda a: a <= SMALL, f"small starts, |x0| <= {SMALL}"),
                                          (lambda a: a >= LARGE, f"large starts, |x0| >= {LARGE}"))):
        for m in models:
            Ts = [t for t in (traj(out_dir, m, r) for r in reps_of(out_dir)) if t is not None]
            sel = mask_f(np.abs(Ts[0]["x0"]))
            n = np.concatenate([np.abs(t["XE"][:, :, sel]).mean(2) for t in Ts], 0)     # [runs, N+1]
            t = np.arange(n.shape[1])
            ax.semilogy(t, np.median(n, 0) + 1e-9, color=COL[m], ls=LS[m], lw=2 if m != "true plant (oracle)" else 1.2,
                        label=m)
            ax.fill_between(t, np.percentile(n, 25, 0) + 1e-9, np.percentile(n, 75, 0) + 1e-9, color=COL[m], alpha=0.08, lw=0)
        ax.set_title(title)
        ax.set_xlabel("step k")
        ax.set_ylabel("mean |x| over starts (median over runs, band = IQR)")
    h, l = axes[0].get_legend_handles_labels()
    fig.suptitle("Closed-loop regulation on the TRUE plant", fontsize=9.5, y=1.02)
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7.5)
    _save(fig, out_dir, "figD3_state_norm")
    return fig


def fig_phase(out_dir, seed, plant, models=("LN, MSE (3000)", "LN, two-stage", "static norm, MSE"), wx=0.6, wu=6.0, n=161,
              sur_dir=None):
    """Figure D4: where the closed loop runs. True-plant trajectories (x_k, u_k) from the small and medium starts, drawn
    over each surrogate's input-gain ratio (model dF/du / true B; blue = wrong sign) with its kink ellipse."""
    import jax.numpy as jnp
    import level1_core as L
    cfgs = {m["name"]: m for m in D.MODELS}
    models = [m for m in models if traj(out_dir, m) is not None]
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models) + 1, 4.4), squeeze=False, constrained_layout=True)
    norm_ = TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5)
    im = None
    for j, m in enumerate(models):
        ax = axes[0, j]
        P, S0, met, seeds = D.load_surrogate(sur_dir or out_dir, cfgs[m]["surrogate"])
        i = seeds.index(seed)
        sur = D.surrogate_configs()[cfgs[m]["surrogate"]]
        p = L.unstack_tree(P, i)
        X, U = np.meshgrid(np.linspace(-wx, wx, n), np.linspace(-wu, wu, n))
        _, _, ju = L.get_evaluator(sur["norm"])(L.stack_trees([p]), jnp.asarray(X.ravel()[None]), jnp.asarray(U.ravel()[None]),
                                                jnp.asarray([S0[i]]))
        G = np.asarray(ju)[0].reshape(X.shape) / plant.B
        im = ax.pcolormesh(X, U, np.clip(G, -3, 5), cmap="RdBu_r", norm=norm_, shading="auto")
        if np.any(G < 0):
            ax.contour(X, U, G, levels=[0], colors="#0b0b0b", linewidths=0.7)
        if np.isfinite(met["rstar_x"][i]):
            ax.add_patch(Ellipse((met["zmin_x"][i], met["zmin_u"][i]), 2 * met["rstar_x"][i], 2 * met["rstar_u"][i],
                                 fill=False, color="#0b0b0b", lw=1.0, ls="--"))
        T = traj(out_dir, m)
        k = list(T["seeds"]).index(seed)
        for jj, x0 in enumerate(T["x0"]):
            if abs(x0) <= 0.5:
                ax.plot(T["XE"][k, :-1, jj], T["UE"][k, :, jj], "-", color="#0b0b0b", lw=0.8, alpha=0.8)
                ax.plot(T["XE"][k, 0, jj], T["UE"][k, 0, jj], "o", ms=3, color="#0b0b0b")
        ax.set_xlim(-wx, wx); ax.set_ylim(-wu, wu)
        ax.set_title(f"{m}, seed {seed}", fontsize=8.5)
        ax.set_xlabel("x"); ax.grid(False)
        if j == 0:
            ax.set_ylabel("u")
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("model / true input gain (1 = correct, < 0 = wrong sign)", fontsize=8)
    fig.suptitle("Where the closed loop runs: true-plant trajectories (x_k, u_k) from starts |x0| <= 0.5 (dots = start), over the "
                 "surrogate's input gain; dashed ellipse = closed-form kink", fontsize=9)
    _save(fig, out_dir, f"figD4_phase_seed{seed}")
    return fig


def fig_curves(C, out_dir):
    """Figure D5: DPC loss on the controller's own model during training (median over seeds, band = IQR)."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for m in [m for m in ORDER if m in set(C.model)]:
        st = C[C.model == m].groupby("epoch").loss
        x = st.median().index.values
        ax.semilogy(x, st.median().values, color=COL[m], ls=LS[m], lw=1.8, label=m)
        ax.fill_between(x, st.quantile(0.25).values, st.quantile(0.75).values, color=COL[m], alpha=0.1, lw=0)
    e = 0
    for N, n in D.CURRICULUM[:-1]:
        e += n
        ax.axvline(e, color="#8a8984", lw=0.7, ls=":")
    ax.set_xlabel("epoch (dotted = horizon increases: 5, 10, 20, 50, 100, 200)")
    ax.set_ylabel("DPC loss on its own model")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("DPC training curves (the loss is divided by the horizon, so it drops at each horizon increase)")
    fig.tight_layout()
    _save(fig, out_dir, "figD5_training_curves")
    return fig


def fig_kink_vs_gap(R, S, out_dir, col="gap_small_dyn"):
    """Figure D6: does the size of the surrogate's kink predict the closed-loop damage? x = peak input-gain error of the
    surrogate (Level 1 metric), y = cost gap; one dot per (model, seed)."""
    if not len(S):
        return None
    M = R.merge(S[["model", "seed", "Ju_sup_rel"]], on=["model", "seed"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in [m for m in ORDER if m in set(M.model)]:
        g = M[M.model == m]
        ax.scatter(g.Ju_sup_rel, g[col], s=30, color=COL[m], edgecolor="white", linewidth=0.6, label=m, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1)
    lo = np.nanmin(M[col].values)
    ax.set_ylim(bottom=-1.5 if lo > -1 else 1.5 * lo)
    ax.axhline(0, color="#8a8984", lw=0.8)
    rho = spearmanr(M.Ju_sup_rel, M[col])[0] if len(M) > 3 else np.nan
    ax.set_xlabel("surrogate's peak input-gain error  sup |dF/du - B| / B")
    ax.set_ylabel(f"closed-loop {col} (%)")
    ax.set_title(f"Kink size vs closed-loop damage (Spearman rho = {rho:.2f} over all {len(M)} model-seed pairs)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir, "figD6_kink_vs_gap")
    return fig


def fig_robustness(R, out_dir, col="gap_test"):
    """Figure D7: is a failure a property of the surrogate (fails in every repeat) or of one unlucky DPC training (fails in
    one repeat)? Rows = seeds, columns = models; colour = median cost gap over the repeats (log); text = failed repeats."""
    from matplotlib.colors import LogNorm
    models = [m for m in ORDER if m in set(R.model) and m != "true plant (oracle)"]
    seeds = sorted(R.seed.unique())
    G = np.full((len(seeds), len(models)), np.nan)
    T = [["" for _ in models] for _ in seeds]
    for j, m in enumerate(models):
        for i, s in enumerate(seeds):
            g = R[(R.model == m) & (R.seed == s)][col]
            if len(g):
                G[i, j] = np.median(g)
                T[i][j] = f"{int((g > FAIL).sum())}/{len(g)}"
    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 2.5, 0.42 * len(seeds) + 1.8))
    im = ax.imshow(np.clip(np.abs(G), 1e-2, 1e5), cmap="Greys", norm=LogNorm(1e-2, 1e5), aspect="auto")
    for i in range(len(seeds)):
        for j in range(len(models)):
            if T[i][j]:
                dark = np.abs(G[i, j]) > 30
                ax.text(j, i, f"{G[i, j]:.2g} %\n{T[i][j]} failed", ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "#0b0b0b", fontweight="bold" if T[i][j][0] != "0" else "normal")
    ax.set_xticks(range(len(models))); ax.set_xticklabels([SHORT[m] for m in models], fontsize=7.5)
    ax.set_yticks(range(len(seeds))); ax.set_yticklabels([f"seed {s}" for s in seeds], fontsize=7.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("median true-plant cost gap over repeats (%, log)", fontsize=8)
    ax.set_title(f"Robustness to the controller's training: cost gap per seed (median over repeats) and how many repeats "
                 f"failed (gap > {FAIL:g} %)", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "figD7_robustness")
    return fig


def fig_diagnostic(Dg, R, out_dir):
    """Figure D8: the reference controller run through each surrogate.
    Left: its cost on the surrogate / on the true plant (1 = the surrogate treats a good controller like the plant does).
    Right: cost on the surrogate of the controller DPC found through it / of the reference controller. Below 1: the
    surrogate rates its own (possibly bad) controller better, so the MODEL misled DPC. Above 1: a better controller existed
    for this surrogate and DPC missed it (an OPTIMISATION trap). Crosses = runs that failed on the true plant."""
    if not len(Dg):
        return None
    Dg = Dg.merge(R[["model", "seed", "rep", "gap_test"]], on=["model", "seed", "rep"], how="left")
    order = [m for m in ORDER if m in set(Dg.model)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    rng = np.random.default_rng(0)
    for ax, (y, title) in zip(axes, ((Dg.oracle_on_model / Dg.oracle_on_true,
                                      "reference controller: cost on the surrogate / on the true plant"),
                                     (Dg.own_on_model / Dg.oracle_on_model,
                                      "on the surrogate: cost of DPC's own controller / of the reference controller"))):
        Dg = Dg.assign(y=y.values)
        for i, m in enumerate(order):
            g = Dg[Dg.model == m]
            f = g.gap_test > FAIL
            xj = i + rng.uniform(-0.18, 0.18, len(g))
            ax.scatter(xj[~f.values], g.y[~f], s=22, color=COL[m], edgecolor="white", linewidth=0.5, zorder=3)
            ax.scatter(xj[f.values], g.y[f], s=46, color=COL[m], marker="X", edgecolor="#0b0b0b", linewidth=0.6, zorder=4,
                       label="failed on the true plant" if i == 0 else None)
        ax.axhline(1, color="#8a8984", lw=0.9)
        ax.set_yscale("log")
        ax.set_xticks(range(len(order))); ax.set_xticklabels([SHORT[m] for m in order], fontsize=7)
        ax.set_title(title, fontsize=8.5)
    axes[1].text(0.02, 0.97, "above 1: a better controller existed for the model and DPC missed it (optimisation trap)\n"
                             "below 1: the model prefers DPC's controller (the model misleads)",
                 transform=axes[1].transAxes, fontsize=7, color="#52514e", va="top")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Diagnostic: the controller trained through the TRUE plant, run through each surrogate (one dot per run)",
                 fontsize=9.5, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "figD8_diagnostic")
    return fig


# ------------------------------------------------------------------------------------------------ #
# Memoryless controllers: policy maps, effective gains, and the linear controller's loss landscape  #
# ------------------------------------------------------------------------------------------------ #
def _policy(c, xs, r=None):
    import jax.numpy as jnp
    inp = xs[:, None] if r is None else np.column_stack([xs - r, np.full_like(xs, r)])
    return np.asarray(D.ctrl_step(c, jnp.zeros((len(xs), 0)), jnp.asarray(inp))[1])[:, 0]


def policy_gain_table(out_dir, seeds, models, rep=0, xs=(0.0, 0.5, 2.0)):
    """K_eff = -d pi/dx at a few states, per model and seed (memoryless controllers). True saturated LQR: 10.29 for
    |x| < 0.97, 0 beyond (saturated)."""
    rows = []
    for m in models:
        C, cs = D.load_controllers(out_dir, m["name"], rep)
        if C is None or "Wz" in C:
            continue
        for s in seeds:
            c1 = D.tmap(lambda a: a[cs.index(s)], C)
            g = D.policy_gain(c1, list(xs))
            rows.append(dict(model=m["name"], seed=s, **{f"K_eff(x={x:g})": float(v) for x, v in zip(xs, g)},
                             u_at_0=float(_policy(c1, np.array([0.0]))[0])))
    T = pd.DataFrame(rows)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    T.to_csv(os.path.join(out_dir, "tables", "policy_gains.csv"), index=False)
    return T.round(3)


def fig_policy_maps(out_dir, sur_dir, seeds, plant, models, rep=0, xlim=2.0, name="figD9_policy_maps"):
    """Rows = models, columns = seeds: the learned policy u = pi(x) (black) over the surrogate's input-gain ratio along it
    (coloured dots: white = right, red = too large, blue = wrong sign), the saturated LQR (grey dashed) and the kink
    half-width in x (grey band). A healthy policy is a straight line of slope -10.3 clipped at +-10."""
    import jax.numpy as jnp
    import level1_core as L
    models = [m for m in models if m.get("kind") != "true" and D.load_controllers(out_dir, m["name"], rep)[0] is not None]
    xs = np.linspace(-xlim, xlim, 801)
    fig, axes = plt.subplots(len(models), len(seeds), figsize=(max(2.6 * len(seeds), 7.5), 2.4 * len(models)), squeeze=False,
                             sharex=True, sharey=True)
    norm_ = TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5)
    sc = None
    for a, m in enumerate(models):
        C, cs = D.load_controllers(out_dir, m["name"], rep)
        if "Wz" in C:
            continue
        P, S0, met, ss = D.load_surrogate(sur_dir, m["surrogate"])
        sur = D.surrogate_configs()[m["surrogate"]]
        ev = L.get_evaluator(sur["norm"])
        for j, s in enumerate(seeds):
            ax = axes[a, j]
            c1 = D.tmap(lambda a_: a_[cs.index(s)], C)
            u = _policy(c1, xs)
            i = ss.index(s)
            _, _, ju = ev(L.stack_trees([L.unstack_tree(P, i)]), jnp.asarray(xs[None]), jnp.asarray(u[None]), jnp.asarray([S0[i]]))
            g = np.asarray(ju)[0] / plant.B
            ax.plot(xs, np.clip(-D.lqr_gain(plant) * xs, -10, 10), color="#8a8984", ls="--", lw=1)
            ax.plot(xs, u, color="#0b0b0b", lw=1.4, zorder=2)
            bad = np.abs(g - 1) > 0.5                                   # colour only where the surrogate is off by > 50 %
            sc = ax.scatter(xs[bad], u[bad], c=np.clip(g[bad], -3, 5), cmap="RdBu_r", norm=norm_, s=14, zorder=3,
                            edgecolor="none")
            if met is not None and np.isfinite(met["rstar_x"][i]):
                ax.axvspan(met["zmin_x"][i] - met["rstar_x"][i], met["zmin_x"][i] + met["rstar_x"][i], color="#f0efec", zorder=0)
            k0 = float(D.policy_gain(c1, [0.0])[0])
            ax.set_title(f"{m['short'].replace(chr(10), ' ')}, seed {s}\nK_eff(0) = {k0:.1f}", fontsize=7.5)
            ax.set_ylim(-11, 11)
            ax.tick_params(labelsize=6.5)
            if a == len(models) - 1:
                ax.set_xlabel("x", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("u = pi(x)", fontsize=7.5)
    fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01).set_label("surrogate / true input gain along the policy", fontsize=8)
    fig.suptitle(f"Learned regulation policies (black) vs the saturated LQR (grey dashed, gain {D.lqr_gain(plant):.4g}). Coloured dots = points of the "
                 "policy where the surrogate's input gain is off by > 50 % (red too large, blue wrong sign); grey band = kink half-width",
                 fontsize=9, y=1.0 + 0.3 / fig.get_figheight())
    _save(fig, out_dir, name)
    return fig


def fig_linear_landscape(out_dir, sur_dir, plant, seeds, models=("LN, MSE (3000)", "LN, two-stage"), N=200, n_x0=200,
                         ks=np.linspace(-30, 2, 97), cs=np.linspace(-3, 3, 25), name="figD10_linear_landscape"):
    """DPC loss of the 2-parameter controller u ~ K x + c on a (K, c) grid, through the surrogate (left of each pair) and
    through the true plant (right), horizon N, starts x0 evenly spaced in [-3, 3]. Star = minimum; dot = where DPC's linear
    controller ended (if one was trained in out_dir). The true plant's optimum is K ~ -14 to -17 (saturated LQR)."""
    import jax.numpy as jnp
    import level1_core as L
    from matplotlib.colors import LogNorm
    cfgs = {m["name"]: m for m in D.MODELS}
    models = [m for m in models if m in cfgs]
    x0 = jnp.linspace(-3, 3, n_x0)[:, None]
    dynT = D.make_dynamics("true", plant=plant)
    fig, axes = plt.subplots(len(seeds), 2 * len(models), figsize=(3.4 * 2 * len(models), 3.4 * len(seeds)), squeeze=False)
    im = None
    for a, s in enumerate(seeds):
        for b, m in enumerate(models):
            spec = cfgs[m]
            P, S0, met, ss = D.load_surrogate(sur_dir, spec["surrogate"])
            i = ss.index(s)
            p1 = L.unstack_tree(P, i)
            dynS = D.make_dynamics("surrogate", D.surrogate_configs()[spec["surrogate"]]["norm"], plant)
            LS = D.linear_landscape(dynS, p1, jnp.asarray(S0[i]), x0, N, ks, cs)
            LT = D.linear_landscape(dynT, p1, jnp.asarray(S0[i]), x0, N, ks, cs)
            lo = np.nanmin([LS.min(), LT.min()])
            hi = lo * 30
            C, cseeds = D.load_controllers(out_dir, m, 0)
            for q, (Lm, ttl) in enumerate(((LS, "through the surrogate"), (LT, "through the true plant"))):
                ax = axes[a, 2 * b + q]
                im = ax.pcolormesh(ks, cs, np.clip(Lm, lo, hi).T, cmap="viridis_r", norm=LogNorm(lo, hi), shading="auto")
                ax.contour(ks, cs, Lm.T, levels=[lo * 1.01, lo * 1.05, lo * 1.2], colors="white", linewidths=0.6)
                ki, ci = np.unravel_index(np.nanargmin(Lm), Lm.shape)
                ax.plot(ks[ki], cs[ci], "*", ms=12, mfc="white", mec="#0b0b0b")
                if C is not None and "k" in C and s in cseeds:
                    c1 = D.tmap(lambda a_: a_[cseeds.index(s)], C)
                    ax.plot(float(c1["k"][0, 0]) * D.LIN_SCALE, float(c1["b"][0]) * D.LIN_SCALE, "o", ms=7, color="#eb6834", mec="white")
                ax.set_title(f"{m}, seed {s}: {ttl}\nmin at K = {ks[ki]:.1f}, c = {cs[ci]:.1f}, loss {Lm[ki, ci]:.3g}", fontsize=7.5)
                ax.tick_params(labelsize=6.5)
                ax.set_xlabel("gain K  (u ~ K x + c)", fontsize=7.5)
                if 2 * b + q == 0:
                    ax.set_ylabel("offset c", fontsize=7.5)
                ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01).set_label("DPC loss (log; clipped at 30 x the minimum)", fontsize=8)
    fig.suptitle(f"The whole DPC problem for a linear controller, horizon {N}: loss over (K, c) through the surrogate vs the true "
                 "plant. White contours = 1 %, 5 %, 20 % above the minimum; star = minimum; orange dot = where DPC ended.",
                 fontsize=9, y=1.0 + 0.3 / fig.get_figheight())
    fig.subplots_adjust(hspace=0.55, wspace=0.3, top=1 - 0.9 / fig.get_figheight())
    _save(fig, out_dir, name)
    return fig


def fig_gradient_audit(sur_dir, plant, seeds, models=("LN, MSE (3000)", "LN, two-stage", "static norm, MSE"), N=200, n_x0=200,
                       ks=np.linspace(-30, 2, 161), name="figD11_gradient_audit"):
    """Corollary U5 in closed loop: dL/dK of the linear controller through the surrogate vs through the true plant, as a
    function of the gain K (b = 0). Where the curves have opposite signs (shaded), gradient descent through the surrogate
    moves K the wrong way. Grey dashed = the true plant's optimum K; saturated LQR sits near -14 to -17 at horizon 200."""
    import jax.numpy as jnp
    import level1_core as L
    cfgs = {m["name"]: m for m in D.MODELS}
    models = [m for m in models if m in cfgs]
    x0 = jnp.linspace(-3, 3, n_x0)[:, None]
    dynT = D.make_dynamics("true", plant=plant)
    fig, axes = plt.subplots(len(seeds), len(models), figsize=(4.2 * len(models), 2.8 * len(seeds)), squeeze=False, sharex=True)
    rows = []
    for a, s in enumerate(seeds):
        for b, m in enumerate(models):
            spec = cfgs[m]
            P, S0, met, ss = D.load_surrogate(sur_dir, spec["surrogate"])
            i = ss.index(s)
            p1 = L.unstack_tree(P, i)
            dynS = D.make_dynamics("surrogate", D.surrogate_configs()[spec["surrogate"]]["norm"], plant)
            G = D.gradient_audit(dynS, dynT, p1, jnp.asarray(S0[i]), x0, N, ks)
            ax = axes[a, b]
            ax.plot(G["k"], G["g_true"], color="#0b0b0b", lw=1.6, label="through the true plant")
            ax.plot(G["k"], G["g_sur"], color=COL.get(m, "#d03b3b"), lw=1.6, ls="--", label="through the surrogate")
            wrong = np.sign(G["g_sur"]) != np.sign(G["g_true"])
            ax.fill_between(G["k"], ax.get_ylim()[0] if False else -1e9, 1e9, where=wrong, color="#fbebeb", zorder=0)
            kopt = G["k"][np.argmin(G["L_true"])]
            ksur = G["k"][np.argmin(G["L_sur"])]
            ax.axvline(kopt, color="#8a8984", ls="--", lw=1)
            ax.axvline(ksur, color=COL.get(m, "#d03b3b"), ls=":", lw=1)
            ax.axhline(0, color="#8a8984", lw=0.8)
            ax.set_yscale("symlog", linthresh=1e-2)
            lim = np.nanmax(np.abs(np.concatenate([G["g_true"], G["g_sur"]])))
            ax.set_ylim(-lim * 1.2, lim * 1.2)
            ax.set_title(f"{m}, seed {s}\nwrong-sign gradient on {100 * wrong.mean():.0f} % of K; argmin true {kopt:.1f}, surrogate {ksur:.1f}", fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            if a == len(seeds) - 1:
                ax.set_xlabel("linear gain K (u ~ K x, saturating at +-10)", fontsize=7.5)
            if b == 0:
                ax.set_ylabel("dL/dK (symlog)", fontsize=7.5)
            rows.append(dict(model=m, seed=s, wrong_sign_share=float(wrong.mean()), k_opt_true=float(kopt), k_opt_sur=float(ksur)))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Gradient audit (Corollary U5): dL/dK through the surrogate vs the true plant for the linear controller; "
                 "red shading = opposite signs", fontsize=9, y=1.0 + 0.25 / fig.get_figheight())
    fig.tight_layout()
    _save(fig, "/".join(sur_dir.split("/")), name) if False else None
    d = os.path.join(sur_dir, "figs"); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), bbox_inches="tight"); fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")
    return fig, pd.DataFrame(rows)
