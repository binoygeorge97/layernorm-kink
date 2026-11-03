"""LQR failure story (rebuilt 2026-10-24): same plant, same start x0 = 1, same setpoint r = 0 in every column."""
import sys; sys.path.insert(0, "/home/claude/kinkwork")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from matplotlib.colors import TwoSlopeNorm; from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, PillowWriter
import level1_core as L, phase_core as PC
OUT = "/home/claude/data/dpc_out"; ST = "/home/claude/story"; plant = L.Plant(); seeds = list(range(10))
LN = PC.Model.surrogate(OUT, "MSE (3000)", seeds, "LN MSE"); SN = PC.Model.surrogate(OUT, "static norm, MSE", seeds, "static")
truth = PC.Model.true_plant(plant, seeds)
R0, X0 = 0.0, 1.0; WIN = (-0.6, 2.4, -11, 11); KLIM = 40
CASES = [("IDEAL", "static-norm surrogate (no kink), seed 9", SN, 9, "fixed"),
         ("KINK, wrong-sign gain", "LayerNorm MSE surrogate, seed 9", LN, 9, "fixed"),
         ("KINK, overestimated gain", "LayerNorm MSE surrogate, seed 8", LN, 8, "fixed"),
         ("KINK, chattering", "LayerNorm MSE surrogate, seed 5, RESCHEDULED", LN, 5, "sched")]
runs = []
for tag, sub, m, s, kind in CASES:
    i = seeds.index(s); r = np.full((10, 1), R0); x0 = np.full((10, 1), X0)
    xs, us, Ks, info = PC.run_lqr(m, truth, r, x0, kind, 200); jx, ju = PC.seen_gain(m, xs, us)
    xo, uo, _, _ = PC.run_lqr(truth, truth, r, x0, "fixed", 200)
    X, U = np.meshgrid(np.linspace(WIN[0], WIN[1], 181), np.linspace(WIN[2], WIN[3], 181))
    _, _, jg = m.seed_view(i).fj(X.ravel()[None], U.ravel()[None]); zx, zu, rx, ru = m.kink()
    bh, ah, Kf = info["bh"][i, 0] / plant.B, info["ah"][i, 0], info["K_fixed"][i, 0]
    cap = (f"{tag}\n{sub}\nslopes at the setpoint: B^/B = {bh:+.1f}, A^ = {ah:.2f}\n-> K = {Kf:+.1f}   (true slopes give K = 10.3)"
           if kind == "fixed" else f"{tag}\n{sub}\nK recomputed every step from the slopes at (x_k, u_k-1)\n{int((np.sign(Ks[1:, i, 0]) != np.sign(Ks[:-1, i, 0])).sum())} sign flips of K in 200 steps")
    x = xs[:, i, 0]; out = np.where((x[:-1] < WIN[0]) | (x[:-1] > WIN[1]))[0]
    uh_i = info["uh"][i, 0]
    if kind == "sched":   # K_k was computed from the slopes at (x_k, u_{k-1}), with u_{-1} = u_hat
        ex, eu = x[:-1], np.concatenate([[uh_i], us[:-1, i, 0]])
    else:                 # K was computed once, from the slopes at the setpoint (r, u_hat)
        ex, eu = np.full(200, R0), np.full(200, uh_i)
    _, ja, jb = m.seed_view(i).fj(np.clip(ex, -50, 50)[None], eu[None]); ae, be = ja[0], jb[0]
    Kchk = L.dare_K(ae, be); assert np.allclose(Kchk, Ks[:, i, 0], rtol=1e-6, atol=1e-6), (tag, np.abs(Kchk - Ks[:, i, 0]).max())
    runs.append(dict(cap=cap, kind=kind, ex=ex, eu=eu, ae=ae, ge=be / plant.B, x=x, u=us[:, i, 0], K=Ks[:, i, 0], g=ju[:, i, 0] / plant.B, xo=xo[:, i, 0], uo=uo[:, i, 0],
                     X=X, U=U, G=jg[0].reshape(X.shape) / plant.B, kink=(zx[i], zu[i], rx[i], ru[i]), leave=int(out[0]) if len(out) else None))
norm_ = TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5)
def bg(ax, d):
    im = ax.pcolormesh(d["X"], d["U"], np.clip(d["G"], -3, 5), cmap="RdBu_r", norm=norm_, shading="auto", rasterized=True)
    if (d["G"] < 0).any(): ax.contour(d["X"], d["U"], d["G"], levels=[0], colors="k", linewidths=0.7)
    zx, zu, rx, ru = d["kink"]
    if np.isfinite(rx): ax.add_patch(Ellipse((zx, zu), 2 * rx, 2 * ru, fill=False, ls="--", lw=1, color="k"))
    xx = np.linspace(WIN[0], WIN[1], 50); ax.plot(xx, plant.u_ss(xx), ":", color="k", lw=0.8)
    ax.plot(R0, plant.u_ss(R0), "*", ms=14, color="gold", mec="k", zorder=5)
    ax.set_xlim(WIN[0], WIN[1]); ax.set_ylim(WIN[2], WIN[3]); ax.set_xlabel("state x"); ax.grid(False); return im
def kaxis(b2):
    b2.set_ylim(-KLIM, KLIM); b2.axhline(10.29, color="#7570b3", lw=0.6, alpha=0.6); b2.axhline(0, color="#7570b3", lw=0.4, alpha=0.4)
def fmt_k(K): return np.clip(K, -KLIM, KLIM)
t = np.arange(200)
# ---------------- static figure ----------------
fig, ax = plt.subplots(2, 4, figsize=(19, 9.4), gridspec_kw=dict(height_ratios=[1.6, 1]))
for c, d in enumerate(runs):
    a = ax[0, c]; im = bg(a, d)
    a.plot(d["xo"][:-1], d["uo"], color="k", lw=1.2, alpha=0.55, label="ideal path (LQR from the true plant)")
    a.plot(np.clip(d["x"][:-1], -99, 99), d["u"], "-o", ms=2.2, lw=1.1, color="#111", label="this controller, on the true plant")
    a.plot(d["x"][0], d["u"][0], "s", ms=9, color="white", mec="k", zorder=6, label=f"start: x0 = {X0:g}, u0 = {d['u'][0]:+.1f}")
    if d["kind"] == "sched":
        a.scatter(np.clip(d["ex"], -99, 99), d["eu"], marker="x", s=22, c=np.where(d["ge"] < 0, "#2166ac", "#b2182b"), lw=1.0, zorder=5,
                  label="where K_k was computed: (x_k, u_k-1)\n(blue x = wrong-sign slope there -> K_k < 0)")
    if d["leave"] is not None: a.annotate(f"leaves the window\nat step {d['leave']}", xy=(WIN[1], d["u"][d["leave"]]), xytext=(1.2, 6), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    a.set_title(d["cap"], fontsize=8.5); a.legend(fontsize=7, loc="upper right" if c != 1 else "lower right"); a.set_ylabel("input u") if c == 0 else None
    b = ax[1, c]; b.plot(t, d["xo"][:-1] - R0, color="k", lw=1, alpha=0.55, label="x - r, ideal")
    b.plot(t, np.clip(d["x"][:-1] - R0, -5, 5), color="#009E73" if c == 0 else "#D55E00", lw=1.5, label="x - r, this controller")
    b.set_ylim(-1.6, 1.6); b.set_xlabel("step k"); b.grid(alpha=0.3); b.set_ylabel("tracking error x - r (solid)") if c == 0 else None
    b2 = b.twinx(); kaxis(b2); b2.plot(t, fmt_k(d["K"]), color="#7570b3", lw=1, ls="--", label=f"gain K_k used (dashed, clipped at +-{KLIM})")
    if c == 3: b2.set_ylabel("gain K_k (dashed; thin line = true LQR gain 10.3)")
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); b.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
fig.colorbar(im, ax=ax[0, :], fraction=0.012, pad=0.01).set_label("surrogate's input slope / true B  (white = right, red = too large, blue = wrong sign)")
fig.suptitle(f"LQR designed from a neural surrogate of x+ = 1.03 x + 0.01 u, run on the TRUE plant. Every column: same start x0 = {X0:g}, same setpoint r = {R0:g} (star), same Q = 5, R = 0.1.\n"
             "Top: the path (x_k, u_k) over the surrogate's input-slope map; dotted = true equilibria u = -3x; dashed ellipse = closed-form kink. Bottom: tracking error (solid, left axis) and the gain used (dashed, right axis).", fontsize=9.5)
fig.savefig(f"{ST}/S1_lqr_failures.png", dpi=140, bbox_inches="tight")
# ---------------- animation ----------------
fig, ax = plt.subplots(2, 4, figsize=(19, 8.2), gridspec_kw=dict(height_ratios=[1.6, 1])); arts = []
for c, d in enumerate(runs):
    a = ax[0, c]; bg(a, d); a.set_title(d["cap"], fontsize=8.5); a.plot(d["xo"][:-1], d["uo"], color="k", lw=1, alpha=0.4)
    a.plot(d["x"][0], d["u"][0], "s", ms=9, color="white", mec="k", zorder=6)
    ln, = a.plot([], [], "-", lw=1.2, color="#111"); pt, = a.plot([], [], "o", ms=11, color="yellow", mec="k", zorder=7)
    ev, = a.plot([], [], "x", ms=12, mew=2.5, color="magenta", zorder=8)
    if c == 0: a.plot([], [], "x", ms=10, mew=2, color="magenta", label="where the slopes for K are read"); a.plot([], [], "o", ms=9, color="yellow", mec="k", label="current (x_k, u_k)"); a.legend(fontsize=7, loc="upper right")
    b = ax[1, c]; b.set_xlim(0, 200); b.set_ylim(-1.6, 1.6); b.grid(alpha=0.3); b.set_xlabel("step k"); b.axhline(0, color="k", lw=0.5)
    e1, = b.plot([], [], color="#009E73" if c == 0 else "#D55E00", lw=1.5); b2 = b.twinx(); kaxis(b2); k1, = b2.plot([], [], color="#7570b3", lw=1, ls="--")
    tx = a.text(0.02, 0.97, "", transform=a.transAxes, fontsize=8, va="top", bbox=dict(fc="white", alpha=0.85, lw=0)); arts.append((ln, pt, e1, k1, tx, ev))
ax[1, 0].set_ylabel("x - r (solid)"); ax[0, 0].set_ylabel("input u")
fig.suptitle(f"LQR from the surrogate on the TRUE plant, x0 = {X0:g}, r = {R0:g}: the ball is (x_k, u_k); magenta x = the point whose surrogate slopes produced this step's K; box = those slopes and K", fontsize=9.5)
def upd(k):
    out = []
    for d, (ln, pt, e1, k1, tx, ev) in zip(runs, arts):
        xk = np.clip(d["x"], -99, 99); ln.set_data(xk[:k + 1], d["u"][:k + 1]); pt.set_data([xk[k]], [d["u"][k]])
        ev.set_data([np.clip(d["ex"][k], -99, 99)], [d["eu"][k]])
        e1.set_data(t[:k + 1], np.clip(d["x"][:k + 1] - R0, -5, 5)); k1.set_data(t[:k + 1], fmt_k(d["K"][:k + 1]))
        where = "at (x_k, u_k-1)" if d["kind"] == "sched" else "once, at the setpoint"
        tx.set_text(f"k = {k}\nx = {d['x'][k]:+.3f}, u = {d['u'][k]:+.2f}\nslopes used for K ({where}):\n  B^/B = {d['ge'][k]:+.2f}, A^ = {d['ae'][k]:.3f}\nK used: {d['K'][k]:+.1f}"); out += [ln, pt, e1, k1, tx, ev]
    return out
FuncAnimation(fig, upd, frames=range(0, 150), blit=True).save(f"{ST}/A1_lqr_failures.gif", writer=PillowWriter(fps=15), dpi=70)
print("done")

# ---------------- zoom on the rescheduled case: where each K_k was read ----------------
d = runs[3]; m, s = CASES[3][2], CASES[3][3]; i = seeds.index(s)
ZW = (-0.15, 0.45, -3.2, 3.2)
X, U = np.meshgrid(np.linspace(ZW[0], ZW[1], 241), np.linspace(ZW[2], ZW[3], 241))
_, _, jg = m.seed_view(i).fj(X.ravel()[None], U.ravel()[None]); G = jg[0].reshape(X.shape) / plant.B
fig = plt.figure(figsize=(17, 6.4)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
a = fig.add_subplot(gs[:, 0])
a.pcolormesh(X, U, np.clip(G, -3, 5), cmap="RdBu_r", norm=norm_, shading="auto", rasterized=True); a.contour(X, U, G, levels=[0], colors="k", linewidths=0.8)
K0, K1 = 35, 75                                    # a window of steps inside the chattering
sel = np.arange(K0, K1)
a.plot(d["x"][sel], d["u"][sel], "-", color="k", lw=0.8, alpha=0.6)
a.scatter(d["x"][sel], d["u"][sel], s=26, c="yellow", edgecolor="k", zorder=5, label="(x_k, u_k): the applied step")
neg = d["ge"][sel] < 0
a.scatter(d["ex"][sel][~neg], d["eu"][sel][~neg], marker="x", s=55, lw=2, c="#b2182b", zorder=6, label="(x_k, u_k-1) with B^ > 0  ->  K_k > 0")
a.scatter(d["ex"][sel][neg], d["eu"][sel][neg], marker="x", s=55, lw=2, c="#2166ac", zorder=6, label="(x_k, u_k-1) with B^ < 0  ->  K_k < 0")
for k in sel[:8]:                                   # connect a few evaluation points to the step they produced
    a.annotate("", xy=(d["x"][k], d["u"][k]), xytext=(d["ex"][k], d["eu"][k]), arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"))
a.plot(R0, plant.u_ss(R0), "*", ms=15, color="gold", mec="k", zorder=7)
a.set_xlim(ZW[0], ZW[1]); a.set_ylim(ZW[2], ZW[3]); a.set_xlabel("state x"); a.set_ylabel("input u"); a.grid(False); a.legend(fontsize=7.5, loc="lower right")
a.set_title(f"Zoom on the rescheduled LQR (LN MSE seed {s}), steps {K0}-{K1 - 1}.\nEach cross is where the controller read the surrogate's slopes; the arrow goes to the step it produced.", fontsize=9)
b = fig.add_subplot(gs[0, 1]); t = np.arange(200)
b.scatter(t, np.clip(d["ge"], -5, 8), c=np.where(d["ge"] < 0, "#2166ac", "#b2182b"), s=12); b.axhline(1, color="k", lw=0.8); b.axhline(0, color="k", lw=0.5, ls=":")
b.axvspan(K0, K1, color="#eeeeee", zorder=0); b.set_ylim(-5.5, 8.5); b.grid(alpha=0.3)
b.set_ylabel("B^_k / B used for K_k\n(clipped to [-5, 8])", fontsize=8.5); b.set_title("the input slope the controller read at (x_k, u_k-1), step by step (1 = true value)", fontsize=9)
c_ = fig.add_subplot(gs[1, 1], sharex=b)
c_.plot(t, np.clip(d["K"], -40, 40), color="#7570b3", lw=1.2, label="K_k (clipped at +-40)"); c_.axhline(10.29, color="k", lw=0.7, ls="--", label="true LQR gain 10.29"); c_.axhline(0, color="k", lw=0.5)
c_.axvspan(K0, K1, color="#eeeeee", zorder=0); c_.set_ylim(-42, 42); c_.set_xlabel("step k"); c_.grid(alpha=0.3); c_.legend(fontsize=7.5, loc="upper right")
c_.set_ylabel("gain K_k = DARE(A^_k, B^_k)", fontsize=8.5); c_.set_title("the resulting gain: negative exactly when the slope read was negative; huge when it was near zero", fontsize=9)
fig.suptitle("Rescheduled LQR: the gain at step k comes from the surrogate's slopes at (x_k, u_k-1) - the current state with the PREVIOUS input. "
             "The resting point sits on the edge of a blue band, so the evaluation point keeps falling in and out of it.", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{ST}/S1b_rescheduled_zoom.png", dpi=140, bbox_inches="tight")
print("zoom done, sign agreement K vs B^:", np.mean(np.sign(d["K"]) == np.sign(d["ge"])))
