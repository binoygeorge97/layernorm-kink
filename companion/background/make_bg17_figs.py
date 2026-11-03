"""Two figures for the slide 17 background note:
fig_zones.png  - per-step input slope and scalar multiplier along the three runs of slide 17
fig_lambda.png - landing amplification Lambda* for all 180 runs (theory results)."""
import sys; import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "paths.py").exists()))); import paths
import numpy as np, pandas as pd, jax, jax.numpy as jnp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import level1_core as L, dpc_core as DC, phase_core as PC
RED, BLUE, GREY = "#8B1A1A", "#1B3A6B", "#9a9a9a"
OUT = paths.DATA; plant = L.Plant(); B = plant.B
FIG = paths.out("companion", "background", "figures"); tm = jax.tree_util.tree_map
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------- per-step multipliers along the three runs ----------
Co, _ = DC.load_controllers(OUT, "true plant (oracle)", 0)
norm_of = {"MSE (3000)": "layer", "static norm, MSE": "static"}
S = np.load(paths.out("story", "sens.npz"))
runs = [("static norm, MSE", 2, "Static norm, seed 2"), ("MSE (3000)", 2, "LN MSE 3000, seed 2"), ("MSE (3000)", 8, "LN MSE 3000, seed 8")]
fig, ax = plt.subplots(3, 2, figsize=(11, 9.5), sharex=True)
for r, (sur, s, ttl) in enumerate(runs):
    x0 = float(S[f"{sur}|{s}|x0"][0])
    P, S0, _ = PC._stack_models(OUT, sur, [s]); P1 = tm(lambda a: a[0], P); S01 = S0[0]
    C1 = tm(lambda a: jnp.asarray(np.asarray(a)[s], jnp.float64), Co)
    F = lambda x, u: L.F_point(P1, x, u, S01, norm_of[sur])
    Jx = jax.jit(jax.grad(F, 0)); Ju = jax.jit(jax.grad(F, 1)); Fj = jax.jit(F)
    ustep = jax.jit(lambda h, x: (lambda o: (o[0][0], o[1][0, 0]))(DC.ctrl_step(C1, h[None], jnp.array([[x]]))))
    dudx = jax.jit(jax.grad(lambda x, h: ustep(h, x)[1]))
    h = jnp.zeros(DC.HIDDEN); x = x0; rows = []
    for k in range(200):
        K = -float(dudx(x, h)); h2, u = ustep(h, x); u = float(u)
        jx, ju = float(Jx(x, u)), float(Ju(x, u)); rows.append((k, ju / B, jx - ju * K))
        x = float(Fj(x, u)); h = h2
    R = np.array(rows); k, g, m = R[:, 0], R[:, 1], R[:, 2]
    col = np.where(m > 1, BLUE, np.where(m < -1, RED, GREY))
    a = ax[r, 0]; a.scatter(k, g, c=col, s=9); a.axhline(1, color="k", lw=0.8); a.axhline(0, color="k", lw=0.4, ls=":")
    a.set_yscale("symlog", linthresh=1); a.set_ylim(-5, 150); a.set_ylabel("surrogate input slope / B")
    a.set_title(f"{ttl}, start x0 = {x0:+.2f}: input slope at each step", fontsize=10)
    a = ax[r, 1]; a.scatter(k, m, c=col, s=9); [a.axhline(v, color="k", lw=0.8, ls="--") for v in (1, -1)]
    a.set_yscale("symlog", linthresh=1); a.set_ylim(-30, 5); a.set_ylabel("multiplier m_k")
    a.set_title(f"{ttl}: multiplier at each step", fontsize=10)
for a in ax[2]: a.set_xlabel("step k")
from matplotlib.lines import Line2D
h_ = [Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
      ((GREY, "safe: |m| < 1"), (BLUE, "m > 1: grows (slope too small or wrong sign)"), (RED, "m < -1: grows, flipping (slope too large)"))]
fig.legend(handles=h_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{FIG}/fig_zones.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- Lambda* for all 180 runs ----------
d = pd.read_csv(paths.find("lambda_runs.csv", paths.out("companion", "background"), paths.REPO + "/companion/background")); d["trap"] = d.trap.astype(bool)
fams = ["LN MSE 3000", "LN slope from 0", "LN MSE 12000", "LN two-stage", "no norm", "static norm"]
names = ["LN MSE\n3000", "LN slope\nfrom step 0", "LN MSE\n12000", "LN\ntwo-stage", "no norm", "static\nnorm"]
fig, a = plt.subplots(figsize=(9.5, 4.6)); rng = np.random.default_rng(0)
for i, f in enumerate(fams):
    e = d[d.family == f]; jit = rng.uniform(-0.18, 0.18, len(e))
    ok = ~e.trap.values
    a.scatter(i + jit[ok], e.S_seg_max.values[ok], s=22, facecolors="none", edgecolors=BLUE, label="not trapped" if i == 0 else None)
    a.scatter(i + jit[~ok], e.S_seg_max.values[~ok], s=26, color=RED, label="trapped" if i == 0 else None)
a.axhline(12.1, color="k", ls="--", lw=1); a.text(5.45, 13.5, "threshold 12.1", ha="right", fontsize=9)
a.set_yscale("symlog", linthresh=1); a.set_ylim(-0.2, 300); a.set_xticks(range(6)); a.set_xticklabels(names)
a.set_ylabel("landing amplification $\\Lambda^\\star$"); a.legend(frameon=False, loc="upper right")
a.set_title("Each dot = one DPC run (family x seed x repeat); $\\Lambda^\\star$ computed for the good controller, before any DPC training", fontsize=9.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_lambda.png", dpi=150); plt.close(fig)
print(d.groupby("family").agg(runs=("trap", "size"), traps=("trap", "sum")))
print("caught", ((d.S_seg_max > 12.1) & d.trap).sum(), "false", ((d.S_seg_max > 12.1) & ~d.trap).sum())
print("missed", d[d.trap & (d.S_seg_max <= 12.1)][["family", "rep", "seed", "S_seg_max"]].to_string())
print("max healthy", d[~d.trap].S_seg_max.max(), "min trapped LN MSE", d[d.trap & (d.family == "LN MSE 3000")].S_seg_max.min())
