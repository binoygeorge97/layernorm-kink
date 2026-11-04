"""Task 2b P1a (2026-11-04): the second error channel of the two-state plant P1. A surrogate with the correct state slope
(A_hat = A) but input slope J_u_hat = 0.01 (h, g)^T (true: h = 0, g = 1) under the true-plant LQR gain K (DARE, Q = diag(5, 0.5),
R = 0.1) gives the closed-loop multiplier M(h, g) = A - J_u_hat K. Map its spectral radius over (h, g), with the stable region
rho < 1, and the safe h-interval at g = 1 (bisection on the grid's sign changes). Also the safe g-band at h = 0 (the
proposal's band, as a check) and the safe h-interval at a few other g.
Usage: python analysis/plant2_band.py -> out/plant2/band2.csv, docs/plant2/fig_band2.png, printed intervals."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are
from scipy.optimize import brentq

A = np.array([[1.02, 0.05], [0.0, 1.01]]); B = np.array([[0.0], [0.01]])
Q, R = np.diag([5.0, 0.5]), np.array([[0.1]])
P = solve_discrete_are(A, B, Q, R); K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)      # 1 x 2
rho = lambda h, g: max(abs(np.linalg.eigvals(A - 0.01 * np.array([[h], [g]]) @ K)))


def interval(f, lo, hi, n=4001):
    """sub-intervals of [lo, hi] where f < 1, endpoints refined by brentq"""
    xs = np.linspace(lo, hi, n); v = np.array([f(x) for x in xs]) - 1
    out, a = [], (xs[0] if v[0] < 0 else None)
    for i in range(1, n):
        if v[i - 1] >= 0 > v[i]:
            a = brentq(lambda x: f(x) - 1, xs[i - 1], xs[i])
        elif v[i - 1] < 0 <= v[i]:
            out.append((a, brentq(lambda x: f(x) - 1, xs[i - 1], xs[i]))); a = None
    if a is not None:
        out.append((a, xs[-1]))
    return out


hs, gs = np.linspace(-3, 4, 281), np.linspace(-2, 22, 241)
Rg = np.array([[rho(h, g) for h in hs] for g in gs])
os.makedirs(paths.out("plant2"), exist_ok=True)
pd.DataFrame([(h, g, Rg[j, i]) for j, g in enumerate(gs) for i, h in enumerate(hs)], columns=["h", "g", "rho"]).to_csv(
    paths.out("plant2", "band2.csv"), index=False)
print("K =", K.round(4), "| true closed-loop rho", round(rho(0, 1), 4))
print("g-band at h = 0:", [(round(a, 3), round(b, 3)) for a, b in interval(lambda g: rho(0, g), -2, 40)])
res = {}
for g in (0.29, 0.5, 1.0, 2.0, 5.0, 10.0, 17.0):
    res[g] = interval(lambda h: rho(h, g), -20, 20)
    print(f"h-interval at g = {g}:", [(round(a, 4), round(b, 4)) for a, b in res[g]])
fig, ax = plt.subplots(figsize=(7.2, 4.6))
cs = ax.pcolormesh(hs, gs, np.clip(Rg, 0.85, 1.3), cmap="RdBu_r", norm=matplotlib.colors.TwoSlopeNorm(vmin=0.85, vcenter=1.0, vmax=1.3), shading="auto")
ax.contour(hs, gs, Rg, levels=[1.0], colors="k", linewidths=1.4)
ax.plot([0], [1], "o", color="#1B3A6B", ms=6, label="true plant (h = 0, g = 1)")
(a, b), = res[1.0][:1]
ax.plot([a, b], [1, 1], "-", color="#8B1A1A", lw=2.5, label=f"safe h at g = 1: ({a:.3f}, {b:.3f})")
ax.set_xlabel("h = J_u_hat[x1] / 0.01 (true 0)"); ax.set_ylabel("g = J_u_hat[x2] / 0.01 (true 1)")
ax.set_title("P1, LQR gain K: spectral radius of A - J_u_hat K (black: rho = 1)", fontsize=10)
fig.colorbar(cs, ax=ax, label="spectral radius (clipped to [0.85, 1.3])"); ax.legend(loc="upper right", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(paths.REPO, "docs", "plant2", "fig_band2.png"), dpi=200, facecolor="white")
