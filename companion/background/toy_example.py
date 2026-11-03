"""Toy example: four runs of 200 steps with made-up multipliers. Where the kink sits relative to the run decides."""
import sys, pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "paths.py").exists()))); import paths
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
N = 200
def run(kind):
    m = np.full(N, 0.9)                                   # safe everywhere by default
    if kind == "A": m[100:103] = -3.0                     # passes through the kink: 3 bad steps
    if kind == "C": m[50:] = 0.8                          # lands in the kink, on a safe spot
    if kind == "D": m[50::25] = -10.0                     # lands on a bad spot, 1 bad step in 25
    if kind == "E": m[50::4] = -10.0                      # lands on a bad spot, 1 bad step in 4
    return m
def worst_stretch(m):
    cur = best = 0.0
    for v in np.log(np.abs(m)): cur = max(0.0, cur + v); best = max(best, cur)
    return best
names = {"A": "A: passes through the kink (3 steps at -3)", "C": "B: lands in the kink on a safe spot",
         "D": "C: lands on a bad spot, 1 step in 25 at -10", "E": "D: lands on a bad spot, 1 step in 4 at -10"}
cols = {"A": "#555555", "C": "#1B3A6B", "D": "#C07A30", "E": "#8B1A1A"}
fig, ax = plt.subplots(figsize=(9, 4.2))
for k in "ACDE":
    m = run(k); P = np.concatenate([[1.0], np.cumprod(np.abs(m))])
    L = worst_stretch(m)
    print(k, "running product at 50,100,103,150,200:", ["%.1e" % P[i] for i in (50, 100, 103, 150, 200)],
          " worst stretch %.1f -> x%.1e" % (L, np.exp(L)), " safe share after 50: %.0f%%" % (100 * np.mean(np.abs(m[50:]) < 1)))
    ax.plot(P, color=cols[k], lw=1.6, label=names[k])
ax.axvline(50, color="k", lw=0.6, ls=":"); ax.text(52, 1e-14, "lands at the target\n(inside the kink)", fontsize=8)
ax.set_yscale("log"); ax.set_xlabel("step k"); ax.set_ylabel("running product of |m|"); ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
fig.savefig(paths.out("companion", "background", "figures", "fig_toy.png"), dpi=150)
print("steps of 0.9 needed to cancel one x10:", np.log(10) / -np.log(0.9))
print("block of 4 (one -10, three 0.9): growth per step", (np.log(10) + 3 * np.log(0.9)) / 4)
print("block of 25 (one -10, 24 x 0.9): growth per step", (np.log(10) + 24 * np.log(0.9)) / 25)
