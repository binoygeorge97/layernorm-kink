"""Figures for the companion document (Part 1 of the deck, slides 1-13). Reads the saved surrogates and lqr_study.csv (run_lqr_study.py)."""
import sys, json; import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "paths.py").exists()))); import paths
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Ellipse, FancyBboxPatch
import level1_core as L, phase_core as PC

OUT = paths.DATA; FD = paths.out("companion", "figures"); plant = L.Plant(); A, B = plant.A, plant.B
seeds = list(range(10))
LN = PC.Model.surrogate(OUT, "MSE (3000)", seeds, "LN MSE 3000")
SN = PC.Model.surrogate(OUT, "static norm, MSE", seeds, "static norm")
truth = PC.Model.true_plant(plant, seeds)
zx, zu, rx, ru = LN.kink()
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "savefig.dpi": 200})
NU = TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5); NX = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
facts = {}


def maps(m, i, X, U):
    F, jx, ju = m.seed_view(i).fj(X.ravel()[None], U.ravel()[None])
    return F[0].reshape(X.shape), jx[0].reshape(X.shape), ju[0].reshape(X.shape)


def draw_u(ax, X, U, G, i=None, cbar=False):
    im = ax.pcolormesh(X, U, np.clip(G, -3, 5), cmap="RdBu_r", norm=NU, shading="auto", rasterized=True)
    if (G < 0).any():
        ax.contour(X, U, G, levels=[0], colors="k", linewidths=0.7)
    if i is not None and np.isfinite(rx[i]):
        ax.add_patch(Ellipse((zx[i], zu[i]), 2 * rx[i], 2 * ru[i], fill=False, ls="--", lw=1, color="k"))
    return im


def draw_x(ax, X, U, G, i=None):
    im = ax.pcolormesh(X, U, np.clip(G, -0.5, 0.5), cmap="PuOr_r", norm=NX, shading="auto", rasterized=True)
    if i is not None and np.isfinite(rx[i]):
        ax.add_patch(Ellipse((zx[i], zu[i]), 2 * rx[i], 2 * ru[i], fill=False, ls="--", lw=1, color="k"))
    return im


# ---------------------------------------------------------------- F1 data and kink locations
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
d = L.make_data(plant, 0.01, 1000 + 9)
sel = np.random.default_rng(0).choice(L.N_DATA, 3000, replace=False)
ax[0].scatter(d["x"][sel], d["u"][sel], s=1, color="0.55", rasterized=True, label="training inputs (3000 of 20000 shown)")
xx = np.linspace(-4, 4, 50)
ax[0].plot(xx, plant.u_ss(xx), "k:", lw=1, label="true equilibria u = -3x")
ax[0].add_patch(plt.Rectangle((-1, -2.5), 2, 5, fill=False, color="C3", lw=1.2))
ax[0].text(1.1, 2.6, "zoom (right)", color="C3", fontsize=8)
ax[0].set_xlim(-4.2, 4.2); ax[0].set_ylim(-10.5, 10.5); ax[0].set_xlabel("state x"); ax[0].set_ylabel("input u")
ax[0].set_title("Data box: x ~ U[-4, 4], u ~ U[-10, 10], y = 1.03x + 0.01u + 0.01 N(0,1)")
ax[0].legend(fontsize=7, loc="upper right", markerscale=5)
inside_counts = []
for i in seeds:
    di = L.make_data(plant, 0.01, 1000 + i)
    inside = ((di["x"] - zx[i]) / rx[i]) ** 2 + ((di["u"] - zu[i]) / ru[i]) ** 2 <= 1
    inside_counts.append(int(inside.sum()))
    ax[1].add_patch(Ellipse((zx[i], zu[i]), 2 * rx[i], 2 * ru[i], fill=False, lw=1.1, color=plt.cm.tab10(i), label=f"seed {i}: {inside.sum()} pts"))
m = (np.abs(d["x"]) < 1) & (np.abs(d["u"]) < 2.5)
ax[1].scatter(d["x"][m], d["u"][m], s=2, color="0.6", rasterized=True)
ax[1].plot(xx, plant.u_ss(xx), "k:", lw=1)
ax[1].set_xlim(-1, 1); ax[1].set_ylim(-2.5, 2.5); ax[1].set_xlabel("state x")
ax[1].set_title("Kink ellipses of the 10 LN MSE 3000 surrogates\n(grey = all seed-9 training inputs in this window; legend = training points inside)")
ax[1].legend(fontsize=6.5, loc="upper right", ncol=2)
fig.tight_layout(); fig.savefig(f"{FD}/f01_data_and_kinks.png"); plt.close(fig)
facts["inside_counts"] = inside_counts

# ---------------------------------------------------------------- F2 architecture schematic
fig, ax = plt.subplots(figsize=(10.5, 2.6)); ax.axis("off"); ax.set_xlim(0, 10.3); ax.set_ylim(0.3, 3.0)
def box(x, y, w, h, t, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="k", lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
def arr(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9))
box(0.05, 1.2, 0.8, 0.6, "input\nz = (x, u)")
box(1.15, 1.2, 1.35, 0.6, "encoder\nh = E^T z + b\n(2 -> 16)")
box(2.95, 2.05, 1.25, 0.6, "LayerNorm\n(h - mean) / std", fc="#fde0dd")
box(4.45, 2.05, 1.0, 0.6, "gamma, beta\n(16)")
box(5.7, 2.05, 1.5, 0.6, "W1 (16 -> 32), GELU\nW2 (32 -> 16)")
box(7.45, 1.2, 0.55, 0.6, "+")
box(8.3, 1.15, 1.95, 0.7, "readout (16 -> 1)\nx+ = w_d . (h + g) + b_d")
arr(0.85, 1.5, 1.15, 1.5); arr(2.5, 1.6, 2.95, 2.3); arr(4.2, 2.35, 4.45, 2.35); arr(5.45, 2.35, 5.7, 2.35)
arr(7.2, 2.35, 7.6, 1.8); arr(2.5, 1.35, 7.45, 1.35); arr(8.0, 1.5, 8.3, 1.5)
ax.text(4.9, 1.0, "residual (skip) path: h passes unchanged, so the output is LINEAR in (x, u) plus the branch g", ha="center", fontsize=7.5)
ax.text(6.45, 2.8, "branch g = W2 GELU(W1 (gamma * LN(h) + beta) + b1) + b2", ha="center", fontsize=7.5)
ax.text(5.0, 0.55, "1169 parameters: E 32, b 16, gamma 16, beta 16, W1 512, b1 32, W2 512, b2 16, w_d 16, b_d 1", ha="center", fontsize=7.5)
fig.savefig(f"{FD}/f02_architecture.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- F3 the fixed-u slice (slide 2)
i9 = seeds.index(9)
xs_ = np.linspace(-1.5, 1.5, 3001)
F, jx, ju = LN.seed_view(i9).fj(xs_[None], np.zeros_like(xs_)[None])
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
ax[0].plot(xs_, 1e3 * (F[0] - plant.f(xs_, 0 * xs_)), color="k", lw=1)
ax[0].set_title("value error F^(x, 0) - f(x, 0)  [x 1e-3]"); ax[0].set_xlabel("x"); ax[0].grid(alpha=0.3)
ax[1].plot(xs_, jx[0] - A, color="#b35806", lw=1); ax[1].set_title("state slope error J^_x(x, 0) - A"); ax[1].set_xlabel("x"); ax[1].grid(alpha=0.3)
ax[2].plot(xs_, ju[0] / B, color="#2166ac", lw=1); ax[2].axhline(1, color="k", lw=0.6); ax[2].axhline(0, color="k", lw=0.4, ls=":")
ax[2].set_title("input slope ratio J^_u(x, 0) / B  (1 = right)"); ax[2].set_xlabel("x"); ax[2].grid(alpha=0.3)
for a in ax[:3]:
    a.axvspan(zx[i9] - rx[i9], zx[i9] + rx[i9], color="0.85", zorder=0)
X, U = np.meshgrid(np.linspace(-1.5, 1.5, 241), np.linspace(-11, 11, 241))
_, GX, GU = maps(LN, i9, X, U)
draw_u(ax[3], X, U, GU / B, i9)
ax[3].axhline(0, color="C2", lw=1.5, label="the slice u = 0")
xo, uo, _, _ = PC.run_lqr(truth, truth, np.zeros((10, 1)), np.ones((10, 1)), "fixed")
ax[3].plot(xo[:-1, 0, 0], uo[:, 0, 0], "-", color="k", lw=1.2, label="ideal LQR path from x0 = 1")
ax[3].set_title("J^_u / B over the plane, seed 9"); ax[3].set_xlabel("x"); ax[3].set_ylabel("u"); ax[3].legend(fontsize=7, loc="lower left")
fig.suptitle("LN MSE 3000, seed 9, along the line u = 0 (grey band = kink half-width r*_x)", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{FD}/f03_fixed_u_slice.png"); plt.close(fig)
facts["slice_value_err_max"] = float(np.abs(F[0] - plant.f(xs_, 0 * xs_)).max())

# ---------------------------------------------------------------- F4 full-box maps, LN seed 9 vs static seed 9
X, U = np.meshgrid(np.linspace(-4, 4, 321), np.linspace(-10, 10, 321))
fig, ax = plt.subplots(2, 3, figsize=(13, 7.4))
for row, (m, nm) in enumerate(((LN, "LayerNorm MSE 3000"), (SN, "static norm (kink-free)"))):
    Fv, GX, GU = maps(m, i9, X, U)
    im0 = ax[row, 0].pcolormesh(X, U, 1e3 * np.abs(Fv - plant.f(X, U)), cmap="Greys", vmin=0, vmax=10, shading="auto", rasterized=True)
    imx = draw_x(ax[row, 1], X, U, GX - A, i9 if m is LN else None)
    imu = draw_u(ax[row, 2], X, U, GU / B, i9 if m is LN else None)
    ax[row, 0].set_ylabel(f"{nm}, seed 9\ninput u")
    for a in ax[row]:
        a.set_xlabel("state x")
    facts[f"value_err_rms_{'LN' if m is LN else 'SN'}_s9"] = float(np.sqrt(np.mean((Fv - plant.f(X, U)) ** 2)))
ax[0, 0].set_title("|value error| F^ - f  [x 1e-3]"); ax[0, 1].set_title("state slope error J^_x - A"); ax[0, 2].set_title("input slope ratio J^_u / B")
fig.colorbar(im0, ax=ax[:, 0], fraction=0.04, pad=0.09, orientation="horizontal").set_label("x 1e-3")
fig.colorbar(imx, ax=ax[:, 1], fraction=0.04, pad=0.09, orientation="horizontal").set_label("J^_x - A (0 = right; margin A - 1 = 0.03)")
fig.colorbar(imu, ax=ax[:, 2], fraction=0.04, pad=0.09, orientation="horizontal").set_label("J^_u / B (white 1 = right; red > 1; blue < 1; black contour = 0)")
fig.suptitle("Whole data box, seed 9. LN: value error a few 1e-3 except a small spot at the kink (up to 5e-2, saturated here); slope errors concentrated around (0, 0) and along thin arms.\nStatic norm: value errors of the same size spread smoothly over the box (largest at the edges), slopes right everywhere.", fontsize=9.5)
fig.savefig(f"{FD}/f04_maps_full_box.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- F5 zoomed maps for three seeds
X, U = np.meshgrid(np.linspace(-1, 1, 281), np.linspace(-2.5, 2.5, 281))
fig, ax = plt.subplots(2, 3, figsize=(13, 7.6))
for c, s in enumerate((9, 2, 6)):
    i = seeds.index(s); _, GX, GU = maps(LN, i, X, U)
    imx = draw_x(ax[0, c], X, U, GX - A, i); imu = draw_u(ax[1, c], X, U, GU / B, i)
    for a in ax[:, c]:
        a.plot(X[0], plant.u_ss(X[0]), "k:", lw=0.8); a.set_ylim(-2.5, 2.5); a.set_xlim(-1, 1)
    ax[0, c].set_title(f"seed {s}: r*_x = {rx[i]:.3f}, r*_u = {ru[i]:.3f}\nmax |J^_x - A| = {LN.met['Jx_sup'][i]:.2f}")
    ax[1, c].set_title(f"seed {s}: max |J^_u / B - 1| = {LN.met['Ju_sup_rel'][i]:.1f}")
    ax[1, c].set_xlabel("state x")
ax[0, 0].set_ylabel("J^_x - A\ninput u"); ax[1, 0].set_ylabel("J^_u / B\ninput u")
fig.colorbar(imx, ax=ax[0, :], fraction=0.015, pad=0.01).set_label("J^_x - A")
fig.colorbar(imu, ax=ax[1, :], fraction=0.015, pad=0.01).set_label("J^_u / B")
fig.suptitle("LN MSE 3000, window |x| <= 1, |u| <= 2.5. Dashed = closed-form kink ellipse; black solid = J^_u = 0 (wrong sign inside); dotted = true equilibria u = -3x.", fontsize=9.5)
fig.savefig(f"{FD}/f05_maps_zoom.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- F6 petals: angular structure and 1/d decay
th = np.linspace(-np.pi, np.pi, 721); dgrid = np.geomspace(0.5, 40, 60)
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
i = i9
for dval, col in zip((2, 4, 8, 16), ("C0", "C1", "C2", "C3")):
    xq, uq = zx[i] + dval * rx[i] * np.cos(th), zu[i] + dval * ru[i] * np.sin(th)
    _, jxq, juq = LN.seed_view(i).fj(xq[None], uq[None])
    ax[0].plot(np.degrees(th), juq[0] / B - 1, color=col, lw=1, label=f"d = {dval}")
ax[0].set_yscale("symlog", linthresh=0.1); ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("direction theta (deg), 0 = +x, 90 = +u")
ax[0].set_ylabel("J^_u / B - 1  (symlog)"); ax[0].set_title("seed 9: input-slope error on ellipses of radius d\n(sign alternates with direction = petals; shrinks with d)")
ax[0].legend(fontsize=7); ax[0].set_xticks([-180, -90, 0, 90, 180]); ax[0].grid(alpha=0.3)
slopes = {}
for k, (q, lab) in enumerate((("u", "max over theta of |J^_u / B - 1|"), ("x", "max over theta of |J^_x - A|"))):
    a = ax[1 + k]; curves = []
    for i in seeds:
        mx = []
        for dval in dgrid:
            xq, uq = zx[i] + dval * rx[i] * np.cos(th), zu[i] + dval * ru[i] * np.sin(th)
            ok = (np.abs(xq) <= 4) & (np.abs(uq) <= 10)
            _, jxq, juq = LN.seed_view(i).fj(xq[None], uq[None])
            e = np.abs(juq[0] / B - 1) if q == "u" else np.abs(jxq[0] - A)
            mx.append(e[ok].max() if ok.sum() > 50 else np.nan)
        curves.append(mx); a.loglog(dgrid, mx, lw=0.8, color=plt.cm.tab10(i), alpha=0.8)
    C = np.array(curves); med = np.nanmedian(C, 0); a.loglog(dgrid, med, "k", lw=2, label="median over seeds")
    sel = (dgrid >= 2) & (dgrid <= 20) & np.isfinite(med)
    p = np.polyfit(np.log(dgrid[sel]), np.log(med[sel]), 1); slopes[q] = p[0]
    a.loglog(dgrid, np.exp(p[1]) * dgrid ** p[0], "--", color="C3", lw=1, label=f"fit d = 2-20: slope {p[0]:.2f}")
    m2 = med[np.argmin(np.abs(dgrid - 2))]
    a.loglog(dgrid, m2 * (dgrid / 2) ** -1.0, ":", color="0.3", lw=1, label="reference slope -1")
    a.loglog(dgrid, m2 * (dgrid / 2) ** -2.0, "-.", color="0.3", lw=0.8, label="reference slope -2")
    a.set_xlabel("d = distance from kink centre in half-widths"); a.set_title(lab + "\n(10 seeds, thin; points inside the data box)")
    a.legend(fontsize=7); a.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(f"{FD}/f06_petals_decay.png"); plt.close(fig)
facts["petal_slopes"] = slopes

# ---------------------------------------------------------------- F8 time-ordered version of the four LQR cases
CASES = [("ideal: static norm, seed 9, fixed", SN, 9, "fixed"), ("LN seed 9, fixed (wrong-sign gain)", LN, 9, "fixed"),
         ("LN seed 8, fixed (overestimated gain)", LN, 8, "fixed"), ("LN seed 5, rescheduled (chattering)", LN, 5, "sched")]
fig, ax = plt.subplots(4, 4, figsize=(13, 10), sharex=True)
r = np.zeros((10, 1)); x0 = np.ones((10, 1)); t = np.arange(201)
case_facts = []
for row, (nm, m, s, kind) in enumerate(CASES):
    i = seeds.index(s); xs, us, Ks, info = PC.run_lqr(m, truth, r, x0, kind)
    if kind == "sched":
        ex, eu = xs[:-1, i, 0], np.concatenate([[info["uh"][i, 0]], us[:-1, i, 0]])
    else:
        ex, eu = np.zeros(200), np.full(200, info["uh"][i, 0])
    _, ae, be = m.seed_view(i).fj(np.clip(ex, -50, 50)[None], eu[None])
    ax[row, 0].plot(t, xo[:, 0, 0], color="0.5", lw=1, label="ideal"); ax[row, 0].plot(t, np.clip(xs[:, i, 0], -3, 3), "k", lw=1.2, label="this controller")
    ax[row, 0].set_ylim(-1.5, 3.1); ax[row, 0].set_ylabel(nm.replace(", ", "\n", 1), fontsize=8)
    ax[row, 1].plot(t[:-1], uo[:, 0, 0], color="0.5", lw=1); ax[row, 1].plot(t[:-1], us[:, i, 0], "k", lw=1); ax[row, 1].set_ylim(-11, 11)
    g = be[0] / B; ax[row, 2].plot(t[:-1], np.clip(g, -5, 8), ".", ms=3, color="#2166ac"); ax[row, 2].axhline(1, color="k", lw=0.6); ax[row, 2].axhline(0, color="k", lw=0.4, ls=":")
    ax[row, 2].set_ylim(-5.5, 8.5)
    ax[row, 3].plot(t[:-1], np.clip(Ks[:, i, 0], -40, 40), color="#7570b3", lw=1); ax[row, 3].axhline(L.dare_K(A, B), color="k", lw=0.6, ls="--")
    ax[row, 3].set_ylim(-42, 42)
    if kind == "fixed" and (g[0] < -5 or g[0] > 8):
        ax[row, 2].text(100, 4.5 if g[0] < 0 else -3, f"B^/B = {g[0]:+.1f} at every step\n(off the axis, drawn at the edge)", ha="center", fontsize=8)
    if np.abs(xs[:, i, 0]).max() > 3:
        k10 = int(np.argmax(np.abs(xs[:, i, 0]) > 10))
        ax[row, 0].text(100, 1.8, f"|x| > 10 (diverged) at step {k10}\nx_200 = {xs[-1, i, 0]:.0f}", ha="center", fontsize=8)
    for a in ax[row]:
        a.grid(alpha=0.3)
    case_facts.append(dict(case=nm, Bh_over_B=float(info["bh"][i, 0] / B), Ah=float(info["ah"][i, 0]), uh=float(info["uh"][i, 0]),
                           K_fixed=float(info["K_fixed"][i, 0]), u0=float(us[0, i, 0]), x_end=float(xs[-1, i, 0]),
                           sign_flips=int((np.sign(Ks[1:, i, 0]) != np.sign(Ks[:-1, i, 0])).sum())))
ax[0, 0].set_title("state x_k on the true plant (grey = ideal)"); ax[0, 1].set_title("input u_k (grey = ideal)")
ax[0, 2].set_title("B^/B read for the gain at step k\n(fixed: at the setpoint; rescheduled: at (x_k, u_k-1))"); ax[0, 3].set_title("gain K_k used (dashed = true 10.29)")
ax[0, 0].legend(fontsize=7)
for a in ax[-1]:
    a.set_xlabel("step k")
fig.suptitle("The four LQR cases in time order: x0 = 1, r = 0, 200 steps on the true plant (values clipped to the axis ranges)", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{FD}/f08_lqr_timeline.png"); plt.close(fig)
facts["cases"] = case_facts

# ---------------------------------------------------------------- study-table figures
df = pd.read_csv(paths.find("lqr_study.csv", paths.out("companion"), paths.REPO + "/companion")); df["far"] = (df.x0 - df.r).abs() >= 0.5
df["fail"] = df.diverged | (df.gap > 0.1); df["cat"] = df.diverged | (df.gap > 1.0)
f = df[df.far]
FAMS = ["LN MSE 3000", "LN MSE 12000", "LN slope from step 0", "LN two-stage", "no norm", "static norm"]
COL = dict(zip(FAMS, ["#d73027", "#fc8d59", "#984ea3", "#1a9850", "0.45", "#4575b4"]))
REFS = sorted(df.r.unique())

# F10 path-colour test
fig, ax = plt.subplots(figsize=(6.5, 3.6)); labs = ["0", "0-2 %", "2-10 %", "> 10 %"]; pc = {}
for k, (ctl, col) in enumerate((("LQR fixed", "0.4"), ("LQR sched", "#d73027"))):
    g = f[(f.controller == ctl) & f.model.str.startswith("LN") & (np.abs(f.Bh / B - 1) <= 0.2)]
    b = pd.cut(g.wrong_sign_share, [-1e-9, 1e-12, 0.02, 0.10, 1.0], labels=labs)
    rate = g.groupby(b, observed=False).fail.mean() * 100; n = g.groupby(b, observed=False).size()
    pc[ctl] = dict(rate=rate.round(1).tolist(), n=n.tolist())
    xpos = np.arange(4) + (k - 0.5) * 0.38
    ax.bar(xpos, rate.values, 0.38, color=col, label="fixed LQR" if k == 0 else "rescheduled LQR")
    for xp, v, nn in zip(xpos, rate.values, n.values):
        ax.text(xp, v + 0.8, f"{v:.1f}\nn={nn}", ha="center", fontsize=6.5)
ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_xlabel("share of steps on the TRUE path where the surrogate's J^_u <= 0")
ax.set_ylabel("runs failed (%)"); ax.set_ylim(0, 50); ax.legend(fontsize=8)
ax.set_title("LN surrogates, only runs whose setpoint gain is within 20 % of B", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FD}/f10_path_colour_test.png"); plt.close(fig)
facts["path_colour"] = pc

# F11 failure vs setpoint
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for k, ctl in enumerate(("LQR fixed", "LQR sched")):
    for fam in FAMS:
        g = f[(f.controller == ctl) & (f.model == fam)]
        rate = g.groupby("r").fail.mean().reindex(REFS) * 100
        ax[k].plot(range(len(REFS)), rate.values, "-o", ms=3.5, color=COL[fam], lw=1.4 if fam == "LN MSE 3000" else 1, label=fam)
    ax[k].set_xticks(range(len(REFS))); ax[k].set_xticklabels([f"{v:g}" for v in REFS]); ax[k].set_xlabel("setpoint r")
    ax[k].set_title(("fixed LQR (slopes read once at the setpoint)" if k == 0 else "rescheduled LQR (slopes read every step)") + "\n% of the 80 runs per setpoint that fail (gap > 10 % or diverged)")
    ax[k].grid(alpha=0.3)
ax[0].set_ylabel("runs failed (%)"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FD}/f11_failure_vs_setpoint.png"); plt.close(fig)

# F12 the setpoint law
sp = df[(df.controller == "LQR fixed") & df.model.str.startswith("LN")].drop_duplicates(["model", "seed", "r"]).copy()
sp["err"] = np.abs(sp.Bh / B - 1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for fam in FAMS[:4]:
    g = sp[sp.model == fam]
    ax[0].scatter(g.setpoint_kink_dist, g.err, s=10, color=COL[fam], alpha=0.75, label=fam, edgecolor="none")
w = sp[sp.Bh <= 0]; ax[0].scatter(w.setpoint_kink_dist, w.err, s=26, facecolor="none", edgecolor="k", lw=0.8, label="wrong sign (B^ <= 0)")
s1 = sp.setpoint_kink_dist > 1; p = np.polyfit(np.log(sp.setpoint_kink_dist[s1]), np.log(sp.err[s1]), 1)
dd = np.geomspace(1, 300, 50); ax[0].loglog(dd, np.exp(p[1]) * dd ** p[0], "k--", lw=1.2, label=f"fit d > 1: {np.exp(p[1]):.2f} d^{p[0]:.2f}")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("d = distance of (r, u_ss(r)) from the kink centre, in half-widths")
ax[0].set_ylabel("|B^/B - 1| at the setpoint"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")
ax[0].set_title("Input-slope error the fixed LQR is designed with (440 setpoint designs)")
facts["setpoint_law"] = dict(slope=float(p[0]), coef=float(np.exp(p[1])))
runs = f[(f.controller == "LQR fixed") & f.model.str.startswith("LN")]
bins = [0, 1, 2, 4, 8, 32, 1e9]; lb = ["<1", "1-2", "2-4", "4-8", "8-32", ">32"]
bb = pd.cut(runs.setpoint_kink_dist, bins, labels=lb, right=False)
rate = runs.groupby(bb, observed=False).fail.mean() * 100; cat = runs.groupby(bb, observed=False).cat.mean() * 100; n = runs.groupby(bb, observed=False).size()
ax[1].bar(np.arange(6) - 0.2, rate.values, 0.4, color="#d73027", label="failed (> 10 %)"); ax[1].bar(np.arange(6) + 0.2, cat.values, 0.4, color="k", label="catastrophic (> 100 % or diverged)")
for k2, (v, nn) in enumerate(zip(rate.values, n.values)):
    ax[1].text(k2 - 0.2, v + 1, f"n={nn}", ha="center", fontsize=6.5)
ax[1].set_xticks(range(6)); ax[1].set_xticklabels(lb); ax[1].set_xlabel("d (setpoint distance from the kink, half-widths)")
ax[1].set_ylabel("fixed-LQR runs (%)"); ax[1].legend(fontsize=7); ax[1].set_title("Fixed-LQR outcome vs setpoint distance (4 LN families)")
fig.tight_layout(); fig.savefig(f"{FD}/f12_setpoint_law.png"); plt.close(fig)
facts["fail_vs_d"] = dict(fail=rate.round(1).tolist(), cat=cat.round(1).tolist(), n=n.tolist())

# F15 swap test rates (left panel of W2, redrawn)
S = pd.read_csv(paths.find("swap_lqr_rates.csv", paths.out("story"), paths.REPO + "/story"))
designs = list(dict.fromkeys(S.design)); fams = list(dict.fromkeys(S.surrogate))
short = ["1: surrogate A^, B^", "2: true A, surrogate B^", "3: surrogate A^, true B", "4: true A, B (only u^ from surrogate)"]
cols = ["#333333", "#7570b3", "#D55E00", "#009E73"]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
for k, met in enumerate(("failed", "catastrophic")):
    for j, dsg in enumerate(designs):
        v = [S[(S.surrogate == fm) & (S.design == dsg)][met].iloc[0] for fm in fams]
        ax[k].bar(np.arange(len(fams)) + (j - 1.5) * 0.2, v, 0.2, color=cols[j], label=short[j])
    ax[k].set_xticks(range(len(fams))); ax[k].set_xticklabels(fams, fontsize=8); ax[k].grid(alpha=0.3, axis="y")
    ax[k].set_ylabel(f"fixed-LQR runs {met} (%)"); ax[k].set_title(("failed: gap > 10 % or diverged" if k == 0 else "catastrophic: gap > 100 % or diverged") + ", 880 runs per surrogate family")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{FD}/f15_swap_rates.png"); plt.close(fig)

# swap numbers for seed 8 (r = 0, x0 = 1)
uh, nroot, ah, bh = PC.equilibrium(LN, np.zeros((10, 1)))
sw = {}
for s in (9, 8):
    i = seeds.index(s); out = []
    for nm, (a_, b_) in zip(short, ((ah[i, 0], bh[i, 0]), (A, bh[i, 0]), (ah[i, 0], B), (A, B))):
        K = float(L.dare_K(a_, b_)); x = 1.0; mx = 0
        for _ in range(200):
            u = np.clip(uh[i, 0] - K * x, -10, 10); x = A * x + B * u; mx = max(mx, abs(x))
        out.append(dict(design=nm, Ah=float(a_), Bh_over_B=float(b_ / B), K=K, x_end=x, diverged=mx > 10))
    sw[s] = out
facts["swap_r0"] = sw
facts["Ju_sup_rel_LN3000"] = LN.met["Ju_sup_rel"].round(2).tolist(); facts["Jx_sup_LN3000"] = LN.met["Jx_sup"].round(3).tolist()
facts["kink"] = dict(zx=np.round(zx, 3).tolist(), zu=np.round(zu, 3).tolist(), rx=np.round(rx, 3).tolist(), ru=np.round(ru, 3).tolist())
json.dump(facts, open(paths.out("companion", "facts.json"), "w"), indent=1, default=float)
print(json.dumps(facts, indent=1, default=float))
