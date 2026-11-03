"""Figure f16: the slope maps of slide 5 drawn as 3-D surfaces, so the kink shows as crests and troughs.
Reads the saved surrogates (same models and seeds as make_figs.py)."""
import sys; sys.path.insert(0, "/home/claude/kinkwork")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import level1_core as L, phase_core as PC

OUT = "/home/claude/data/dpc_out"; FD = "/home/claude/companion/figures"; plant = L.Plant(); A, B = plant.A, plant.B
seeds = list(range(10))
LN = PC.Model.surrogate(OUT, "MSE (3000)", seeds, "LN MSE 3000")
SN = PC.Model.surrogate(OUT, "static norm, MSE", seeds, "static norm")
zx, zu, rx, ru = LN.kink()
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "savefig.dpi": 200})

S = int(sys.argv[1]) if len(sys.argv) > 1 else 9
XW, UW, NG = 1.0, 2.5, 201
xg, ug = np.linspace(-XW, XW, NG), np.linspace(-UW, UW, NG)
X, U = np.meshgrid(xg, ug)


def slopes(m, s):
    _, jx, ju = m.seed_view(seeds.index(s)).fj(X.ravel()[None], U.ravel()[None])
    return jx[0].reshape(X.shape) - A, ju[0].reshape(X.shape) / B


ex_ln, gu_ln = slopes(LN, S)
ex_sn, gu_sn = slopes(SN, S)
UC = (-45.0, 45.0)          # clip for J_u / B (seed 9 spans -43 to +41, so nothing is clipped)
XC = (-0.6, 0.8)            # clip for J_x - A
print(f"seed {S}: J_u/B LN [{gu_ln.min():.1f}, {gu_ln.max():.1f}]  static [{gu_sn.min():.2f}, {gu_sn.max():.2f}];"
      f"  J_x-A LN [{ex_ln.min():.2f}, {ex_ln.max():.2f}]  static [{ex_sn.min():.3f}, {ex_sn.max():.3f}]")

i = seeds.index(S)
th = np.linspace(0, 2 * np.pi, 200)
ex_, eu_ = zx[i] + rx[i] * np.cos(th), zu[i] + ru[i] * np.sin(th)


def surf(ax, Z, lo, hi, cmap, norm, title, zlabel, floor_level, ref):
    Zc = np.clip(Z, lo, hi)
    ax.plot_surface(X, U, Zc, cmap=cmap, norm=norm, rstride=2, cstride=2, linewidth=0, antialiased=True, rasterized=True)
    ax.plot(ex_, eu_, np.full_like(th, lo), "k--", lw=1.1)                  # kink ellipse on the floor
    xe = xg[np.abs(plant.u_ss(xg)) <= UW]
    ax.plot(xe, plant.u_ss(xe), np.full_like(xe, lo), "k:", lw=1)   # true equilibria on the floor
    ax.set_zlim(lo, hi); ax.set_xlim(-XW, XW); ax.set_ylim(-UW, UW)
    ax.set_xlabel("state x", labelpad=2); ax.set_ylabel("input u", labelpad=2); ax.set_zlabel(zlabel, labelpad=4)
    ax.set_title(title, pad=0)
    ax.view_init(elev=28, azim=-58)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.tick_params(labelsize=7, pad=0)


fig = plt.figure(figsize=(15, 4.9))
nu = TwoSlopeNorm(vmin=UC[0], vcenter=1, vmax=UC[1]); nx = TwoSlopeNorm(vmin=XC[0], vcenter=0, vmax=XC[1])
ax = fig.add_subplot(1, 3, 1, projection="3d")
surf(ax, gu_ln, *UC, "RdBu_r", nu, f"LayerNorm (LN MSE 3000), seed {S}: input slope", "J^_u / B  (true = 1)", UC[0], 1)
ax = fig.add_subplot(1, 3, 2, projection="3d")
surf(ax, ex_ln, *XC, "PuOr_r", nx, f"LayerNorm (LN MSE 3000), seed {S}: state slope", "J^_x - A  (true = 0)", XC[0], 0)
ax = fig.add_subplot(1, 3, 3, projection="3d")
surf(ax, gu_sn, *UC, "RdBu_r", nu, f"static norm, seed {S}: input slope (same scale)", "J^_u / B  (true = 1)", UC[0], 1)
fig.suptitle(f"The kink as crests and troughs: surrogate slopes over |x| <= {XW:g}, |u| <= {UW:g} (same window as the 2-D maps). "
             f"Dashed on the floor = kink ellipse; dotted = true equilibria u = -3x. Heights clipped to the axis range where needed.", fontsize=9.5)
fig.subplots_adjust(left=0.0, right=0.98, top=0.88, bottom=0.02, wspace=0.02)
fn = f"{FD}/f16_slopes_3d.png" if S == 9 else f"{FD}/f16_slopes_3d_seed{S}.png"
fig.savefig(fn, bbox_inches="tight"); print("saved", fn)
