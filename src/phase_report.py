"""phase_report.py - time-ordered (x_k, u_k) phase plots over the surrogate's slope maps, and setpoint studies."""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
import level1_core as L
import phase_core as PC

TRAJ_COL = ["#1baf7a", "#eda100", "#6250d6", "#0b0b0b"]       # starts (validated categorical slots + ink)
MODEL_COL = {"LN, MSE (3000)": "#6f6e69", "LN, MSE (12000)": "#eda100", "LN, J_x + J_u (3000)": "#1baf7a",
             "LN, two-stage": "#6250d6", "no norm, MSE": "#eb6834", "static norm, MSE": "#2a78d6",
             "true plant (oracle)": "#0b0b0b"}
MODEL_ORDER = ["LN, MSE (3000)", "LN, MSE (12000)", "LN, J_x + J_u (3000)", "LN, two-stage", "no norm, MSE", "static norm, MSE"]
TIME_MARKS = (1, 2, 5, 10, 20, 50)
LIN_X, LIN_U = 0.25, 1.0          # symlog axes: linear within |x| <= 0.25, |u| <= 1


def _save(fig, out_dir, name):
    d = os.path.join(out_dir, "figs")
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name + ".png"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(d, name + ".pdf"), bbox_inches="tight")


def _symgrid(lo, hi, t, n):
    """n points between lo and hi, uniform on a symlog axis with linear range +-t."""
    f = lambda v: np.sign(v) * np.log1p(np.abs(v) / t)
    g = lambda w: np.sign(w) * t * np.expm1(np.abs(w))
    return g(np.linspace(f(lo), f(hi), n))


PLANT_A, PLANT_B = 1.03, 0.01     # true-plant slopes used to normalise maps and setpoint gains; set_plant(PLANT) for another plant


def set_plant(plant):
    """Normalise slope maps and setpoint-gain errors by this plant's A and B (default: the reference plant 1.03, 0.01)."""
    global PLANT_A, PLANT_B
    PLANT_A, PLANT_B = float(plant.A), float(plant.B)


def slope_maps(model1, xlim, ulim, n=301, sym=True):
    """Grids and fields for one seed (model1 = Model.seed_view): J_u / B, J_x - A, F(x, u) - x."""
    xs = _symgrid(*xlim, LIN_X, n) if sym else np.linspace(*xlim, n)
    us = _symgrid(*ulim, LIN_U, n) if sym else np.linspace(*ulim, n)
    X, U = np.meshgrid(xs, us)
    F, jx, ju = model1.fj(X.ravel()[None], U.ravel()[None])
    sh = X.shape
    return dict(X=X, U=U, Ju=(ju[0] / PLANT_B).reshape(sh), Jx=(jx[0] - PLANT_A).reshape(sh), G=(F[0] - X.ravel()).reshape(sh))


def phase_panel(ax, maps, trajs, plant, r=None, uh=None, field="Ju", kink=None, sym=True, own=None, legend=False,
                label_time=False):
    """Background: the surrogate's input gain / true (field 'Ju') or state-gain error (field 'Jx'). Overlays: true
    equilibrium line, the surrogate's equilibrium curve F(x, u) = x, the setpoint, the kink ellipse, and the time-ordered
    trajectories [(xs, us), ...] (solid = true plant; `own` = the same controller run through the surrogate, dashed)."""
    X, U = maps["X"], maps["U"]
    if field == "Ju":
        im = ax.pcolormesh(X, U, np.clip(maps["Ju"], -3, 5), cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5),
                           shading="auto", rasterized=True)
        if np.any(maps["Ju"] < 0):
            ax.contour(X, U, maps["Ju"], levels=[0], colors="#0b0b0b", linewidths=0.6)
    else:
        im = ax.pcolormesh(X, U, np.clip(maps["Jx"], -1, 1), cmap="PuOr_r", norm=SymLogNorm(1e-2, vmin=-1, vmax=1, base=10),
                           shading="auto", rasterized=True)
    ax.contour(X, U, maps["G"], levels=[0], colors="#0b0b0b", linewidths=1.0, linestyles="--")
    xl = np.array([X.min(), X.max()])
    ax.plot(xl, plant.u_ss(xl), color="#52514e", lw=1.0, ls=":")
    if kink is not None and np.isfinite(kink[2]):
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(kink[0] + kink[2] * np.cos(t), kink[1] + kink[3] * np.sin(t), color="#0b0b0b", lw=0.8, ls="-.")
    for k, (xs, us) in enumerate(trajs):
        col = TRAJ_COL[k % len(TRAJ_COL)]
        xk, uk = xs[:-1], us
        ok = np.isfinite(xk) & (np.abs(xk) < 50)
        xk, uk = xk[ok], uk[ok]
        ax.plot(xk, uk, color="white", lw=3.2, zorder=4, solid_capstyle="round")
        ax.plot(xk, uk, color=col, lw=1.6, zorder=5)
        if own is not None:
            xo, uo = own[k][0][:-1], own[k][1]
            ok2 = np.isfinite(xo) & (np.abs(xo) < 50)
            ax.plot(xo[ok2], uo[ok2], color=col, lw=1.0, ls=(0, (3, 2)), zorder=5)
        ax.plot(xk[0], uk[0], "o", ms=7, mfc="white", mec=col, mew=2, zorder=6)
        for j, km in enumerate(TIME_MARKS):
            if km < len(xk):
                ax.plot(xk[km], uk[km], "o", ms=2.5 + 0.7 * j, color=col, mec="white", mew=0.6, zorder=6)
                if label_time and k == 0 and km in (1, 5, 20):
                    ax.annotate(f"k={km}", (xk[km], uk[km]), xytext=(4, 4), textcoords="offset points", fontsize=6.5,
                                color="#0b0b0b", zorder=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        ax.plot(xk[-1], uk[-1], "X", ms=7, color=col, mec="white", mew=0.8, zorder=6)
    if r is not None:
        ax.plot(r, plant.u_ss(r), "*", ms=14, mfc="white", mec="#0b0b0b", mew=1.2, zorder=7)
        if uh is not None:
            ax.plot(r, uh, "D", ms=5, mfc="#0b0b0b", mec="white", mew=0.6, zorder=7)
    if sym:
        ax.set_xscale("symlog", linthresh=LIN_X)
        ax.set_yscale("symlog", linthresh=LIN_U)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(U.min(), U.max())
    ax.grid(False)
    if legend:
        ax.legend(handles=legend_handles(own is not None), fontsize=6.5, loc="upper right", framealpha=0.9, frameon=True)
    return im


def legend_handles(own=False, starts=None):
    from matplotlib.lines import Line2D
    if True:
        h = [Line2D([], [], color="#52514e", ls=":", label="true equilibria u = u_ss(x)"),
             Line2D([], [], color="#0b0b0b", ls="--", label="surrogate's equilibria F(x, u) = x"),
             Line2D([], [], color="#0b0b0b", ls="-.", lw=0.8, label="closed-form kink ellipse"),
             Line2D([], [], marker="*", ls="", ms=11, mfc="white", mec="#0b0b0b", label="setpoint (r, u_ss(r))"),
             Line2D([], [], marker="D", ls="", ms=5, mfc="#0b0b0b", mec="white", label="surrogate's u_hat(r)"),
             Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec="#52514e", mew=2, label="start (k = 0)"),
             Line2D([], [], marker="X", ls="", ms=7, color="#52514e", label="k = 199"),
             Line2D([], [], color="#52514e", lw=1.6, marker="o", ms=3, label="true plant; dots at k = 1, 2, 5, 10, 20, 50")]
        if own:
            h.append(Line2D([], [], color="#52514e", lw=1.0, ls=(0, (3, 2)), label="same controller through the surrogate"))
        if starts is not None:
            h += [Line2D([], [], color=TRAJ_COL[k % len(TRAJ_COL)], lw=2.2, label=f"start x0 = {x:g}") for k, x in enumerate(starts)]
        return h


def fig_legend(fig, own=False, starts=None, bbox=(0.905, 0.55)):
    fig.legend(handles=legend_handles(own, starts), fontsize=7, loc="upper left", bbox_to_anchor=bbox, frameon=False)


def strip_panels(ax_x, ax_g, trajs, seen, r, plant, cols=TRAJ_COL):
    """Below a phase panel: tracking error x_k - r, and the surrogate input gain the path visits (J_u_hat / B)."""
    for k, (xs, us) in enumerate(trajs):
        kk = np.arange(len(us)) + 1
        ax_x.plot(np.arange(len(xs)) + 1, xs - r, color=cols[k % len(cols)], lw=1.4)
        ax_g.plot(kk, seen[k], color=cols[k % len(cols)], lw=1.4)
    for a in (ax_x, ax_g):
        a.set_xscale("log")
        a.axhline(0, color="#8a8984", lw=0.8)
    ax_x.set_yscale("symlog", linthresh=0.01)
    lo, hi = ax_x.get_ylim()
    ax_x.set_ylim(max(lo, -10), min(hi, 10))
    ax_x.set_ylabel("x_k - r\n(lin. within +-0.01)", fontsize=7)
    ax_g.set_yscale("symlog", linthresh=1)
    ax_g.axhline(1, color="#8a8984", lw=0.8, ls=":")
    ax_g.axhspan(-1e3, 0, color="#f3d9d9", zorder=0, lw=0)
    ax_g.set_ylim(min(-3, ax_g.get_ylim()[0]), max(5, ax_g.get_ylim()[1]))
    ax_g.set_ylabel("J_u_hat / B seen\n(red band = wrong sign)", fontsize=7)
    ax_g.set_xlabel("step k + 1", fontsize=7)
    for a in (ax_x, ax_g):
        a.tick_params(labelsize=6.5)


def _window(trajs, r, plant, pad=1.25):
    xs = np.concatenate([t[0][np.isfinite(t[0]) & (np.abs(t[0]) < 50)] for t in trajs] + [np.array([r, -0.3, 0.3])])
    us = np.concatenate([t[1][np.isfinite(t[1])] for t in trajs] + [np.array([plant.u_ss(r), -1.2, 1.2])])
    xl = (max(-4.0, xs.min() * pad - 0.05), min(4.0, xs.max() * pad + 0.05))
    ul = (max(-10.5, us.min() * pad - 0.2), min(10.5, us.max() * pad + 0.2))
    return xl, ul


def _gap_text(J, J0, x0, r, xs):
    """Per-start cost gap (only for starts at least 0.5 from the setpoint: nearer ones have almost no transient, so any
    steady offset looks like a huge relative gap) and the steady offset |x - r| averaged over the last 20 steps."""
    J, J0, x0 = np.ravel(J), np.ravel(J0), np.ravel(x0)
    parts = []
    for j in range(len(J)):
        if abs(x0[j] - r) < 0.5 or J0[j] < 1e-9:
            parts.append("-")
        elif not np.isfinite(J[j]) or J[j] / J0[j] > 1e3:
            parts.append("diverged" if not np.isfinite(J[j]) or J[j] / J0[j] > 1e6 else f"{J[j] / J0[j] - 1:.0f}x")
        else:
            parts.append(f"{100 * (J[j] / J0[j] - 1):.2g}%")
    fin = np.abs(np.asarray(xs)[-20:].reshape(20, -1).mean(0) - r)
    fin = fin[np.isfinite(fin) & (fin < 10)]
    off = f"{np.median(fin):.2g}" if len(fin) else "n/a"
    return "cost gap per start: " + ", ".join(parts) + f"\nsteady offset |x - r| (median over starts) {off}"


def fig_phase_lqr(model, seed, plant, out_dir, refs=(0.0, 0.25, 1.0), starts=(-2.5, -0.5, 0.5, 2.5), kinds=("fixed", "sched"),
                  name=None):
    """One surrogate, one seed. Columns = setpoints; for each LQR design: phase panel + two strips."""
    i = model.seeds.index(seed)
    m1 = model.seed_view(i)
    truth = PC.Model.true_plant(plant, [seed])
    kink = tuple(v[0] for v in m1.kink())
    nk = len(kinds)
    fig = plt.figure(figsize=(5.0 * len(refs), 6.3 * nk))
    gs = fig.add_gridspec(4 * nk, len(refs), height_ratios=[3.2, 1, 1, 0.45] * nk, hspace=0.5, wspace=0.28, top=0.91, right=0.86)
    im = None
    for a, kind in enumerate(kinds):
        for j, r in enumerate(refs):
            rr = np.full((1, len(starts)), r)
            x0 = np.array(starts, float)[None]
            xs, us, Ks, info = PC.run_lqr(m1, truth, rr, x0, kind)
            trajs = [(xs[:, 0, k], us[:, 0, k]) for k in range(len(starts))]
            _, ju = PC.seen_gain(m1, xs, us)
            seen = [ju[:, 0, k] / plant.B for k in range(len(starts))]
            xl, ul = _window(trajs, r, plant)
            maps = slope_maps(m1, xl, ul)
            ax = fig.add_subplot(gs[4 * a, j])
            im = phase_panel(ax, maps, trajs, plant, r, info["uh"][0, 0], kink=kink, label_time=False)
            J0 = PC.track_cost(*PC.run_lqr(truth, truth, rr, x0, "fixed")[:2], rr, plant)
            J = PC.track_cost(xs, us, rr, plant)
            gap = np.where(J0 > 1e-9, J / np.maximum(J0, 1e-12) - 1, np.nan)[0]
            ttl = (f"LQR {'designed at the setpoint' if kind == 'fixed' else 'rescheduled every step'}, r = {r:g}\n"
                   f"B_hat/B at setpoint {info['bh'][0, 0] / plant.B:.2g}, K = {info['K_fixed'][0, 0]:.3g} (true 10.3); "
                   f"{info['nroot'][0, 0]} surrogate equilibri{'um' if info['nroot'][0, 0] == 1 else 'a'}\n" + _gap_text(J, J0, x0, r, xs))
            ax.set_title(ttl, fontsize=7.5)
            ax.set_xlabel("x (symlog)", fontsize=7.5)
            ax.set_ylabel("u (symlog)", fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            strip_panels(fig.add_subplot(gs[4 * a + 1, j]), fig.add_subplot(gs[4 * a + 2, j]), trajs, seen, r, plant)
    cax = fig.add_axes([0.88, 0.76, 0.012, 0.17])
    fig.colorbar(im, cax=cax).set_label("surrogate / true input gain\n(1 = correct, < 0 = wrong sign)", fontsize=7.5)
    fig_legend(fig, starts=starts, bbox=(0.875, 0.72))
    fig.suptitle(f"{model.name}, seed {seed}: LQR from the surrogate, run on the true plant. Starts x0 = "
                 + ", ".join(f"{s:g}" for s in starts) + " (colours). Axes are symmetric-log so the kink and the far setpoints "
                 "fit in one view.", fontsize=9, y=0.985)
    _save(fig, out_dir, name or f"figP1_lqr_phase_{PC.DC.safe(model.name)}_seed{seed}")
    return fig


def fig_phase_atlas(models, seed, plant, out_dir, controller="fixed", refs=(-1.0, 0.0, 0.25, 1.0),
                    starts=(-2.5, -0.5, 0.5, 2.5), name=None):
    """Rows = surrogates, columns = setpoints, one seed and one LQR design: where the closed loop runs over each slope map."""
    fig, axes = plt.subplots(len(models), len(refs), figsize=(4.2 * len(refs), 3.7 * len(models)), squeeze=False)
    im = None
    for a, model in enumerate(models):
        i = model.seeds.index(seed)
        m1 = model.seed_view(i)
        truth = PC.Model.true_plant(plant, [seed])
        kink = tuple(v[0] for v in m1.kink())
        for j, r in enumerate(refs):
            rr = np.full((1, len(starts)), r)
            x0 = np.array(starts, float)[None]
            xs, us, Ks, info = PC.run_lqr(m1, truth, rr, x0, controller)
            trajs = [(xs[:, 0, k], us[:, 0, k]) for k in range(len(starts))]
            maps = slope_maps(m1, (-3.2, 3.2), (-10.5, 10.5), n=241)
            ax = axes[a, j]
            im = phase_panel(ax, maps, trajs, plant, r, info["uh"][0, 0], kink=kink, label_time=False)
            J0 = PC.track_cost(*PC.run_lqr(truth, truth, rr, x0, "fixed")[:2], rr, plant)
            J = PC.track_cost(xs, us, rr, plant)
            ok = J0[0] > 1e-9
            ax.set_title(f"{model.name}, r = {r:g}\nB_hat/B at setpoint {info['bh'][0, 0] / plant.B:.2g}\n"
                         + _gap_text(J, J0, x0, r, xs).replace("cost gap per start", "gap").replace("steady offset |x - r| (median over starts)", "offset"), fontsize=7)
            ax.tick_params(labelsize=6.5)
            if j == 0:
                ax.set_ylabel("u (symlog)", fontsize=7.5)
            if a == len(models) - 1:
                ax.set_xlabel("x (symlog)", fontsize=7.5)
    fig.subplots_adjust(top=1 - 1.15 / fig.get_figheight(), hspace=0.5, wspace=0.22)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01).set_label("surrogate / true input gain", fontsize=8)
    what = "designed once at the setpoint" if controller == "fixed" else "rescheduled every step"
    fig.suptitle(f"Seed {seed}: LQR {what}, on the true plant, for four setpoints (star). Dashed = surrogate's equilibria, "
                 "dotted = true equilibria; starts " + ", ".join(f"{s:g}" for s in starts), fontsize=9, y=1 - 0.2 / fig.get_figheight())
    _save(fig, out_dir, name or f"figP2_atlas_lqr_{controller}_seed{seed}")
    return fig


def _gap_table(T, min_dist=0.5):
    """Gaps only for starts at least min_dist from the setpoint: with x0 ~ r the reference cost is ~ 0 and a steady offset
    of 0.01 alone looks like a huge relative gap."""
    T = T[(np.abs(T.x0 - T.r) >= min_dist)].copy()
    T["failed"] = (T.gap > 0.10) | T.diverged
    return T


def fig_gap_vs_setpoint(T, out_dir, controllers=("LQR fixed", "LQR sched"), name="figP3_gap_vs_setpoint", kink_halfwidth=None,
                        title_extra=""):
    """Cost gap and steady offset against the setpoint, per surrogate (median over seeds and starts, IQR band over seeds)."""
    T = _gap_table(T)
    fig, axes = plt.subplots(3, len(controllers), figsize=(6.2 * len(controllers), 9.2), squeeze=False, sharex=True)
    for j, c in enumerate(controllers):
        g = T[T.controller == c]
        for mname in [m for m in MODEL_ORDER if m in set(g.model)]:
            h = g[g.model == mname]
            per_seed = h.groupby(["r", "seed"]).agg(gap=("gap", "median"), off=("offset", "median"), fail=("failed", "mean"))
            for row, (col, f, lab) in enumerate((("gap", 100, "cost gap, median over starts (%)"),
                                                ("fail", 100, "starts failed: gap > 10 % or diverged (%)"),
                                                ("off", 1, "steady offset |x - r| (median)"))):
                st = per_seed[col].groupby("r")
                x = st.median().index.values
                ax = axes[row, j]
                ax.plot(x, f * st.median().values, "o-", ms=3.5, color=MODEL_COL[mname], lw=1.8, label=mname)
                ax.fill_between(x, f * st.quantile(0.25).values, f * st.quantile(0.75).values, color=MODEL_COL[mname],
                                alpha=0.12, lw=0)
                ax.set_ylabel(lab, fontsize=8)
        axes[0, j].set_title(c.replace("LQR fixed", "LQR designed at the setpoint").replace("LQR sched", "LQR rescheduled every step")
                             .replace("DPC", "tracking DPC"), fontsize=9)
        axes[0, j].set_yscale("log")
        axes[2, j].set_yscale("log")
        axes[2, j].set_xlabel("setpoint r (symlog, linear within +-0.1)")
        for a in axes[:, j]:
            a.set_xscale("symlog", linthresh=0.1)
            if kink_halfwidth is not None:
                for v in (-kink_halfwidth, kink_halfwidth):
                    a.axvline(v, color="#0b0b0b", lw=0.8, ls="-.")
    axes[0, 0].legend(fontsize=7)
    axes[2, 0].axhline(PC.SETTLED, color="#d03b3b", lw=0.8, ls=":")
    fig.suptitle("Where the setpoint sits relative to the kink: true-plant cost gap and steady offset against the setpoint r "
                 "(line = median over seeds, band = inter-quartile range over seeds; dash-dot lines = +- median LN-MSE kink half-width; "
                 "starts within 0.5 of r excluded from the gap)" + title_extra, fontsize=8.5, y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, name)
    return fig


def fig_setpoint_distance(T, out_dir, controller="LQR fixed", name="figP4_setpoint_distance"):
    """Per (seed, setpoint) of the LayerNorm surrogates: distance of the setpoint from the kink centre (in kink half-widths,
    measured in the (x, u) plane at the setpoint's true equilibrium input) against (left) the error of the surrogate's input
    gain there, which is what an LQR designed at the setpoint inherits, and (right) the resulting true-plant cost gap."""
    T = _gap_table(T)
    g = T[(T.controller == controller) & T.model.str.startswith("LN")]
    P = g.groupby(["model", "seed", "r"]).agg(d=("setpoint_kink_dist", "first"), bh=("Bh", "first"), gap=("gap", "median"),
                                              div=("diverged", "mean")).reset_index()
    P["e"] = np.abs(P.bh / PLANT_B - 1)
    far = P[P.d > 1]
    slope, icpt = np.polyfit(np.log(far.d), np.log(far.e + 1e-6), 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for mname in [m for m in MODEL_ORDER if m in set(P.model)]:
        h = P[P.model == mname]
        ok, bad = h[h.bh > 0], h[h.bh <= 0]
        ax[0].scatter(ok.d, ok.e, s=16, color=MODEL_COL[mname], label=mname, edgecolor="white", linewidth=0.4)
        ax[0].scatter(bad.d, bad.e, s=34, marker="x", color=MODEL_COL[mname], linewidth=1.4)
        hv = h[h["div"] > 0]
        hn = h[h["div"] == 0]
        ax[1].scatter(hn.d, 100 * np.maximum(hn.gap, 1e-4), s=16, color=MODEL_COL[mname], edgecolor="white", linewidth=0.4)
        ax[1].scatter(hv.d, np.full(len(hv), 3e4), s=34, marker="x", color=MODEL_COL[mname], linewidth=1.4)
    dd = np.geomspace(0.05, 150, 50)
    ax[0].plot(dd, np.exp(icpt) * dd ** slope, "k--", lw=1, label=f"fit beyond 1 half-width: slope {slope:.2f}")
    ax[0].axhline(1, color="#d03b3b", lw=0.8, ls=":")
    ax[0].text(160, 1.25, "above: can have the wrong sign", fontsize=7, color="#d03b3b", ha="right")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("setpoint distance from the kink centre (kink half-widths)")
    ax[0].set_ylabel("|surrogate input gain at the setpoint / B - 1|")
    ax[0].set_title("The gain an LQR designed at the setpoint inherits: error falls ~ 1 / distance\n"
                    "(x = wrong sign; one dot per LayerNorm surrogate, seed and setpoint)", fontsize=9)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("setpoint distance from the kink centre (kink half-widths)")
    ax[1].set_ylabel("cost gap, median over starts (%)")
    ax[1].set_title(f"{controller.replace('LQR fixed', 'LQR designed at the setpoint')}: true-plant cost gap\n"
                    "(x at the top = diverged on at least one start)", fontsize=9)
    ax[1].axhline(10, color="#d03b3b", lw=0.8, ls=":")
    for a in ax:
        a.axvline(1, color="#8a8984", lw=0.8, ls="--")
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    _save(fig, out_dir, name)
    return fig, slope


def fig_phase_dpc(models, seed, plant, out_dir, C_by_model, starts=(-2.5, -0.5, 0.5, 2.5), r=0.0, track=False, name=None,
                  fields=("Ju", "Jx")):
    """DPC controllers (regulation, or tracking at setpoint r): rows = surrogates, per field a phase panel with the true-plant
    path (solid) and the same controller run through its surrogate (dashed), plus the two strips."""
    fig = plt.figure(figsize=(4.6 * len(models), 3.9 * len(fields) + 3.0))
    gs = fig.add_gridspec(len(fields) + 2, len(models), height_ratios=[3.2] * len(fields) + [1, 1], hspace=0.4, wspace=0.3,
                          top=0.92, right=0.85)
    ims = {}
    for j, model in enumerate(models):
        i = model.seeds.index(seed)
        m1 = model.seed_view(i)
        truth = PC.Model.true_plant(plant, [seed])
        C, cseeds = C_by_model[model.name]
        C1 = PC.tmap(lambda a: a[cseeds.index(seed):cseeds.index(seed) + 1], C)
        x0 = np.array(starts, float)[None]
        rr = np.full_like(x0, r)
        xs, us = PC.run_dpc(C1, truth, x0, rr, track=track)
        xo, uo = PC.run_dpc(C1, m1, x0, rr, track=track)
        trajs = [(xs[:, 0, k], us[:, 0, k]) for k in range(len(starts))]
        own = [(xo[:, 0, k], uo[:, 0, k]) for k in range(len(starts))]
        _, ju = PC.seen_gain(m1, xs, us)
        seen = [ju[:, 0, k] / plant.B for k in range(len(starts))]
        xl, ul = _window(trajs + own, r, plant)
        maps = slope_maps(m1, xl, ul)
        kink = tuple(v[0] for v in m1.kink())
        uh = PC.equilibrium(m1, np.array([r]))[0][0, 0]
        J0 = PC.track_cost(*PC.run_lqr(truth, truth, rr, x0, "fixed")[:2], rr, plant)
        J = PC.track_cost(xs, us, rr, plant)
        for f_i, field in enumerate(fields):
            ax = fig.add_subplot(gs[f_i, j])
            ims[field] = phase_panel(ax, maps, trajs, plant, r, uh, field=field, kink=kink, own=own,
                                     label_time=False)
            ttl = f"{model.name}, seed {seed}" + ("\n" + _gap_text(J, J0, x0, r, xs).replace("cost gap", "cost gap vs true-plant LQR") if f_i == 0 else "")
            ax.set_title(ttl, fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            ax.set_xlabel("x (symlog)", fontsize=7)
            if j == 0:
                ax.set_ylabel(("input gain map" if field == "Ju" else "state gain map") + "\nu (symlog)", fontsize=7.5)
        strip_panels(fig.add_subplot(gs[len(fields), j]), fig.add_subplot(gs[len(fields) + 1, j]), trajs, seen, r, plant)
    if "Ju" in ims:
        cb = fig.colorbar(ims["Ju"], cax=fig.add_axes([0.87, 0.80, 0.01, 0.12]))
        cb.set_label("J_u_hat / B\n(< 0 = wrong sign)", fontsize=7)
    if "Jx" in ims:
        cb = fig.colorbar(ims["Jx"], cax=fig.add_axes([0.87, 0.62, 0.01, 0.12]))
        cb.set_label("J_x_hat - A\n(state gain error)", fontsize=7)
    fig_legend(fig, own=True, starts=starts, bbox=(0.86, 0.58))
    what = f"tracking DPC, setpoint r = {r:g}" if track else "regulation DPC (setpoint 0)"
    fig.suptitle(f"{what}: time-ordered (x_k, u_k) on the true plant (solid) and through the controller's own surrogate (dashed). "
                 "Top: surrogate / true input gain; middle: state-gain error (purple = A_hat too small, orange = too large); "
                 "strips: tracking error and the input gain the true path visits", fontsize=8.5, y=0.985)
    _save(fig, out_dir, name or f"figP5_dpc_phase_seed{seed}{'_r' + str(r) if track else ''}")
    return fig


CTRL_STYLE = {"LQR fixed": ("#0b0b0b", "-", "o", "LQR designed at the setpoint"),
              "LQR sched": ("#d03b3b", "--", "s", "LQR rescheduled every step"),
              "DPC": ("#2a78d6", "-.", "^", "tracking DPC")}
WS_BINS = [-0.01, 0.0, 0.02, 0.1, 1.0]
WS_LABELS = ["0", "0-2 %", "2-10 %", "> 10 %"]


def path_table(T, good=0.2):
    """Per run (LayerNorm surrogates only, starts >= 0.5 from the setpoint): share of the true path's steps where the surrogate
    says the input gain has the wrong sign, and whether the gain at the setpoint is accurate (|B_hat/B - 1| < good)."""
    T = _gap_table(T)
    T = T[T.model.str.startswith("LN")].copy()
    T["ws_bin"] = pd.cut(T.wrong_sign_share, WS_BINS, labels=WS_LABELS)
    T["setpoint_ok"] = np.abs(T.Bh / PLANT_B - 1) < good
    return T


def fig_path_vs_setpoint(T, out_dir, name="figP6_path_vs_setpoint", good=0.2, catastrophic=False, drop_broken=False):
    """Does it matter where the closed loop goes, or only where it ends? Failure rate against the share of the path spent where
    the surrogate's input gain has the wrong sign, for each controller; right: only runs whose setpoint gain is accurate.
    catastrophic=True counts only gap > 100 % or divergence (not the 10-100 % gaps that steady offsets cause);
    drop_broken=True leaves out (controller, model, seed) combinations that fail catastrophically on more than half their runs
    (a controller broken everywhere, e.g. a DPC policy trapped in training, says nothing about the path)."""
    P = path_table(T, good)
    P["cat"] = (P.gap > 1.0) | P.diverged
    if drop_broken:
        P = P[P.groupby(["controller", "model", "seed"]).cat.transform("mean") <= 0.5]
    if catastrophic:
        P["failed"] = P["cat"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    rows = []
    for a, (sub, ttl) in enumerate(((P, "all runs"), (P[P.setpoint_ok], f"only runs whose setpoint gain is within {100 * good:.0f} % of B"))):
        for c in [c for c in CTRL_STYLE if c in set(sub.controller)]:
            col, ls, mk, lab = CTRL_STYLE[c]
            g = sub[sub.controller == c].groupby("ws_bin", observed=False).agg(fail=("failed", "mean"), n=("failed", "size"))
            ax[a].plot(range(len(g)), 100 * g.fail.values, color=col, ls=ls, marker=mk, lw=1.8, ms=6, label=lab)
            for b, row in g.iterrows():
                rows.append(dict(subset=ttl, controller=c, wrong_sign_share=b, failed_pct=100 * row.fail, n=int(row.n)))
        ax[a].set_xticks(range(len(WS_LABELS)))
        ax[a].set_xticklabels(WS_LABELS)
        ax[a].set_xlabel("share of the true path's steps where the surrogate says J_u has the wrong sign")
        ax[a].set_title(ttl + " (run counts per bin: see the table)", fontsize=9)
    ax[0].set_ylabel("runs failed: " + ("gap > 100 % or diverged" if catastrophic else "gap > 10 % or diverged") + " (%)")
    ax[0].legend(fontsize=7.5)
    fig.suptitle("Does the path's colour matter, or only the setpoint's? LayerNorm surrogates, setpoints and starts (>= 0.5 from "
                 "the setpoint)" + ("; catastrophic failures only" if catastrophic else "")
                 + ("; broken controllers (seeds failing everywhere) left out" if drop_broken else ""), fontsize=9.5)
    fig.tight_layout()
    _save(fig, out_dir, name)
    return fig, pd.DataFrame(rows)


def fig_seed_setpoint_grid(T, out_dir, controllers=("LQR fixed", "LQR sched", "DPC"),
                           models=("LN, MSE (3000)", "LN, J_x + J_u (3000)", "LN, two-stage", "static norm, MSE"),
                           name="figP8_seed_setpoint_grid"):
    """Rows = seeds, columns = setpoints; colour = share of starts that fail (gap > 10 % or diverged). Separates failures
    that follow the SETPOINT (a vertical band at the kink) from failures that follow the SEED (whole rows: the controller
    itself is broken, whatever the setpoint)."""
    from matplotlib.colors import ListedColormap
    G = _gap_table(T)
    controllers = [c for c in controllers if c in set(G.controller)]
    models = [m for m in models if m in set(G.model)]
    fig, axes = plt.subplots(len(controllers), len(models), figsize=(3.3 * len(models) + 1.2, 3.0 * len(controllers)),
                             squeeze=False, constrained_layout=True)
    cmap = matplotlib.colormaps["Reds"]
    im = None
    for i, c in enumerate(controllers):
        for j, m in enumerate(models):
            ax = axes[i, j]
            g = G[(G.controller == c) & (G.model == m)]
            M = 100 * g.groupby(["seed", "r"]).failed.mean().unstack()
            im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(M.shape[1]))
            ax.set_xticklabels([f"{v:g}" for v in M.columns], fontsize=6, rotation=90)
            ax.set_yticks(range(M.shape[0]))
            ax.set_yticklabels(M.index, fontsize=6.5)
            ax.grid(False)
            n_bad_rows = int((M.mean(1) > 50).sum())
            ax.set_title(f"{c.replace('LQR fixed', 'LQR at setpoint').replace('LQR sched', 'LQR rescheduled').replace('DPC', 'tracking DPC')}"
                         f"\n{m}: {M.values.mean():.0f} % failed; seeds failing > half: {n_bad_rows}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("seed", fontsize=7.5)
            if i == len(controllers) - 1:
                ax.set_xlabel("setpoint r", fontsize=7.5)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01).set_label("starts failed (%)", fontsize=8)
    fig.suptitle("Is a failure tied to the setpoint (vertical band near r = 0) or to the seed (whole rows)? "
                 "Share of starts (>= 0.5 from r) with gap > 10 % or divergence", fontsize=9)
    _save(fig, out_dir, name)
    return fig
