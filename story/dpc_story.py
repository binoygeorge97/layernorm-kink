"""Failure-story figures (2026-10-24). Needs <OUT>/surrogates and the repeat-0 controllers (GRU) of dpc_out, plus story_data.py's outputs.
Run: story_data.py (8 min CPU) -> lqr_story.py -> dpc_story.py. Figures S1-S4, animations A1-A3 in the story folder."""
import sys; import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "paths.py").exists()))); import paths
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from matplotlib.colors import TwoSlopeNorm; from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import level1_core as L, dpc_core as DC, phase_core as PC
OUT = paths.DATA; ST = paths.out("story"); plant = L.Plant(); seeds = list(range(10))
CL = np.load(f"{ST}/closed_loops.npz"); starts = CL["starts"]
SENS = np.load(f"{ST}/sens.npz"); WK = np.load(f"{ST}/warm_ckpt.npz"); WC = pd.read_csv(f"{ST}/warm_ckpt.csv")
LN = PC.Model.surrogate(OUT, "MSE (3000)", seeds, "LN MSE"); SN = PC.Model.surrogate(OUT, "static norm, MSE", seeds, "static")
norm_ = TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5)
def bg(ax, m, s, win=(-1.5, 1.5, -10, 10), n=161):
    X, U = np.meshgrid(np.linspace(win[0], win[1], n), np.linspace(win[2], win[3], n))
    if m is None: G = np.ones_like(X)
    else: _, _, jg = m.seed_view(seeds.index(s)).fj(X.ravel()[None], U.ravel()[None]); G = jg[0].reshape(X.shape) / plant.B
    im = ax.pcolormesh(X, U, np.clip(G, -3, 5), cmap="RdBu_r", norm=norm_, shading="auto", rasterized=True)
    if (G < 0).any(): ax.contour(X, U, G, levels=[0], colors="k", linewidths=0.6)
    if m is not None:
        zx, zu, rx, ru = m.kink(); i = seeds.index(s)
        if np.isfinite(rx[i]): ax.add_patch(Ellipse((zx[i], zu[i]), 2 * rx[i], 2 * ru[i], fill=False, ls="--", lw=1, color="k"))
    xx = np.linspace(win[0], win[1], 50); ax.plot(xx, plant.u_ss(xx), ":", color="k", lw=0.8); ax.plot(0, 0, "*", ms=12, color="gold", mec="k", zorder=5)
    ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.grid(False); return im
# ================= S2 / A2: how the DPC controllers fail (true plant), ideal vs kink =================
PANELS = [("oracle (DPC through the true plant), seed 2", "true plant (oracle)", None, 2),
          ("IDEAL: DPC through static-norm surrogate, seed 2", "static norm, MSE", SN, 2),
          ("KINK: LN MSE seed 1 - parks at a false rest x = 0.6", "LN, MSE (3000)", LN, 1),
          ("KINK: LN MSE seed 2 - bang-bang limit cycle", "LN, MSE (3000)", LN, 2),
          ("KINK: LN MSE seed 9 - pushed the wrong way, diverges", "LN, MSE (3000)", LN, 9),
          ("FIX: two-stage surrogate, seed 2", "LN, two-stage", PC.Model.surrogate(OUT, "J_x + J_u (two-stage)", seeds, "two-stage"), 2)]
cmap = plt.get_cmap("coolwarm")
fig, ax = plt.subplots(2, 6, figsize=(24, 8), gridspec_kw=dict(height_ratios=[1.5, 1]))
for c, (title, name, m, s) in enumerate(PANELS):
    a = ax[0, c]; im = bg(a, m, s); X, U = CL[f"{name}|X_true"][s], CL[f"{name}|U_true"][s]
    for j, x0 in enumerate(starts):
        col = cmap(j / (len(starts) - 1)); a.plot(np.clip(X[:-1, j], -50, 50), U[:, j], "-", color=col, lw=1.1, alpha=0.9); a.plot(X[0, j], U[0, j], "o", ms=4, color=col, mec="k")
    a.set_title(title, fontsize=9); a.set_xlabel("x");  a.set_ylabel("u") if c == 0 else None
    b = ax[1, c]; t = np.arange(201)
    for j, x0 in enumerate(starts): b.plot(t, np.clip(X[:, j], -12, 12), color=cmap(j / (len(starts) - 1)), lw=1.1)
    b.axhline(0, color="k", lw=0.5); b.set_ylim(-3.2, 3.2); b.set_xlabel("step k"); b.grid(alpha=0.3); b.set_ylabel("x_k on the true plant") if c == 0 else None
    b.text(0.02, 0.95, f"cost {DC.true_lqr_cost(X, U):.2f}", transform=b.transAxes, fontsize=8, va="top", bbox=dict(fc="white", lw=0, alpha=0.8))
fig.colorbar(im, ax=ax[0, :], fraction=0.01, pad=0.005).set_label("surrogate / true input gain")
fig.suptitle("How DPC controllers fail: the saved repeat-0 controllers run on the TRUE plant from 10 starts (dots), over their own surrogate's input gain. "
             "Top: phase plane (x_k, u_k); bottom: x_k. The oracle ends at (0, 0); the kinked ones park off-target, limit-cycle or diverge, always OUTSIDE the kink.", fontsize=10)
fig.savefig(f"{ST}/S2_dpc_how_it_fails.png", dpi=120, bbox_inches="tight")
# animation A2: same panels, marching dots
fig, ax = plt.subplots(1, 6, figsize=(24, 4.6)); arts = []
for c, (title, name, m, s) in enumerate(PANELS):
    a = ax[c]; bg(a, m, s); a.set_title(title, fontsize=8.5); a.set_xlabel("x")
    X, U = CL[f"{name}|X_true"][s], CL[f"{name}|U_true"][s]
    lines = [a.plot([], [], "-", color=cmap(j / (len(starts) - 1)), lw=1.0, alpha=0.8)[0] for j in range(len(starts))]
    dots = a.scatter(X[0], U[0], c=[cmap(j / (len(starts) - 1)) for j in range(len(starts))], s=50, edgecolor="k", zorder=6)
    arts.append((X, U, lines, dots))
tx = fig.text(0.5, 0.01, "", ha="center", fontsize=10)
def upd(k):
    out = []
    for X, U, lines, dots in arts:
        for j, ln in enumerate(lines): ln.set_data(np.clip(X[:k + 1, j], -50, 50), U[:k + 1, j])
        dots.set_offsets(np.column_stack([np.clip(X[k], -50, 50), U[k]])); out += lines + [dots]
    tx.set_text(f"step k = {k}"); return out + [tx]
FuncAnimation(fig, upd, frames=list(range(0, 60)) + list(range(60, 200, 4)), blit=True).save(f"{ST}/A2_dpc_how_it_fails.gif", writer=PillowWriter(fps=12), dpi=60)
# ================= S3: why - one trajectory's sensitivity through surrogate vs true plant =================
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for c, (sur, s, ttl) in enumerate([("static norm, MSE", 2, "IDEAL: static-norm surrogate, seed 2"), ("MSE (3000)", 2, "KINK: LN MSE seed 2 (dominating start)"), ("MSE (3000)", 8, "KINK: LN MSE seed 8 (dominating start)")]):
    a = ax[c]; k = np.arange(201); x0 = SENS[f"{sur}|{s}|x0"][0]
    a.plot(k, SENS[f"{sur}|{s}|true|s"], color="k", lw=1.5, label="through the TRUE plant: |d x_k / d x_0|")
    a.plot(k, SENS[f"{sur}|{s}|sur|s"], color="#D55E00" if c else "#009E73", lw=1.5, label="through the SURROGATE")
    a.set_yscale("log"); a.set_ylim(1e-8, 1e40 if c else 1e2); a.set_xlabel("step k"); a.grid(alpha=0.3)
    a.set_title(f"{ttl}\nreference controller from x0 = {x0:+.2f}", fontsize=9)
    a2 = a.twinx(); a2.plot(k, SENS[f"{sur}|{s}|sur|x"], color="#7570b3", lw=1, ls="--", label="x_k through the surrogate"); a2.set_ylim(-0.3, 0.3); a2.set_ylabel("x_k (dashed)")
    if c == 0: a.set_ylabel("closed-loop sensitivity (log)")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
fig.suptitle("WHY DPC fails: run the GOOD controller through the surrogate. The state settles near 0 either way (dashed), but the surrogate's slopes make the closed loop locally "
             "unstable, so the sensitivity the gradient is built from grows like e^{lambda k} instead of decaying: the loss value is right, its gradient is 1e13-1e35 too large.", fontsize=9.5)
fig.savefig(f"{ST}/S3_dpc_why_sensitivity.png", dpi=140, bbox_inches="tight")
# ================= A3: warm start animation: policy expelled from the good basin + per-start gradient panel =================
pairs = [("static norm, MSE", 2, "IDEAL: static-norm surrogate, seed 2"), ("MSE (3000)", 2, "KINK: LN MSE seed 2"), ("MSE (3000)", 9, "KINK: LN MSE seed 9")]
mod = {"static norm, MSE": SN, "MSE (3000)": LN}
epochs = sorted({int(k.split("|")[2]) for k in WK.files if k.count("|") == 3})
CK = [e for e in epochs if f"MSE (3000)|2|{e}|gnorm" in WK.files]
x0all = WK["x0"]
fig = plt.figure(figsize=(18, 8.4)); gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])
arts = []
for c, (sur, s, ttl) in enumerate(pairs):
    a = fig.add_subplot(gs[0, c]); bg(a, mod[sur], s, win=(-1.2, 1.2, -6, 6)); a.set_title(ttl, fontsize=9.5); a.set_xlabel("x"); a.set_ylabel("u") if c == 0 else None
    X0 = WK[f"{sur}|{s}|0|X_true"]; sel = np.arange(0, 1000, 25)
    lines = [a.plot([], [], "-", color="#111", lw=0.7, alpha=0.7)[0] for _ in sel]
    b = fig.add_subplot(gs[1, c]); b.set_yscale("log"); b.set_ylim(1e-2, 1e40); b.set_xlim(-3, 3); b.grid(alpha=0.3); b.set_xlabel("training start x0"); b.set_ylabel("per-start gradient norm (log)") if c == 0 else None
    b.axhline(3, color="k", lw=0.6, ls=":"); gsc = b.scatter(x0all[s], np.full(1000, np.nan), s=6, color="#D55E00" if c else "#009E73")
    tx = a.text(0.02, 0.97, "", transform=a.transAxes, fontsize=8.5, va="top", bbox=dict(fc="white", alpha=0.85, lw=0))
    arts.append((sur, s, sel, lines, gsc, tx))
fig.suptitle("WHY, watched: AdamW started AT the good controller, trained on each surrogate's own loss (horizon 200). Top: closed loops on the TRUE plant (25 of the 1000 training starts). "
             "Bottom: norm of each start's gradient (dotted = 3, the healthy maximum). At epoch 0 ONE start's gradient is 1e15-1e38 on the kinked surrogates (100 % of the batch); the first steps follow it and the policy leaves the target.", fontsize=9.5)
def upd(fi):
    e = epochs[fi]; out = []
    for sur, s, sel, lines, gsc, tx in arts:
        X, U = WK[f"{sur}|{s}|{e}|X_true"], WK[f"{sur}|{s}|{e}|U_true"]
        for j, ln in zip(sel, lines): ln.set_data(np.clip(X[:-1, j], -50, 50), U[:, j])
        ck = max([c for c in CK if c <= e]); g = WK[f"{sur}|{s}|{ck}|gnorm"]; gsc.set_offsets(np.column_stack([x0all[s], np.clip(g, 1e-2, 1e40)]))
        row = WC[(WC.sur == sur) & (WC.seed == s) & (WC.epoch == e)].iloc[0]
        tx.set_text(f"epoch {e}\ntrue-plant cost {row.loss_true:.2f}\nsurrogate loss {row.loss_sur:.2f}\nmax per-start grad {g.max():.2g} (at epoch {ck})")
        out += lines + [gsc, tx]
    return out
FuncAnimation(fig, upd, frames=range(len(epochs)), blit=True).save(f"{ST}/A3_dpc_why_warmstart.gif", writer=PillowWriter(fps=2), dpi=65)
# static version: epochs 0, 10, 50, 100 of the kink seed 2 and ideal
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
for r_, (sur, s, ttl) in enumerate([pairs[0], pairs[2]]):
    for c, e in enumerate([0, 10, 50, 100]):
        a = ax[r_, c]; bg(a, mod[sur], s, win=(-1.2, 1.2, -6, 6)); X, U = WK[f"{sur}|{s}|{e}|X_true"], WK[f"{sur}|{s}|{e}|U_true"]
        for j in range(0, 1000, 25): a.plot(np.clip(X[:-1, j], -50, 50), U[:, j], "-", color="#111", lw=0.7, alpha=0.7)
        row = WC[(WC.sur == sur) & (WC.seed == s) & (WC.epoch == e)].iloc[0]
        a.set_title(f"{ttl}\nepoch {e}: true cost {row.loss_true:.2f}, surrogate loss {row.loss_sur:.2f}", fontsize=8.5); a.set_xlabel("x"); a.set_ylabel("u") if c == 0 else None
fig.suptitle("Warm start at the good controller: ideal surrogate keeps it (top); on the kinked LN MSE seed 9 one start's 1e15 gradient kicks it off the target within 5 epochs and it settles at x = 0.7 (bottom). Window zoomed to |x| < 1.2.", fontsize=10)
fig.savefig(f"{ST}/S4_dpc_why_warmstart.png", dpi=120, bbox_inches="tight")
print("dpc story done")
