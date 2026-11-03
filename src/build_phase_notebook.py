"""Builds dpc_phase_tracking.ipynb (self-contained: embeds level1_core.py, dpc_core.py, phase_core.py, phase_report.py)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {n: open(os.path.join(HERE, n)).read() for n in ("level1_core.py", "dpc_core.py", "phase_core.py", "phase_report.py")}

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Phase portraits over the slope maps: where the closed loop goes, and where the setpoint sits

**Question.** The LayerNorm kink is a small region around (x, u) ~ (0, 0) where the surrogate's slopes are wrong, with red
(gain too large) and blue (wrong-sign) arms radiating from it. Does it matter which coloured regions the closed loop passes
through on its way, or only where it ends up? And how does the placement of the setpoint relative to the kink change that?

**Time-ordered phase plots.** Each trajectory (x_k, u_k) on the TRUE plant is drawn in order (open circle = start, dots at
k = 1, 2, 5, 10, 20, 50, cross = k = 199) over the surrogate's input-gain map dF/du / B (white = correct, red = too large,
blue = wrong sign, black contour = sign flip) or its state-gain error map. Also drawn: the true equilibria u = u_ss(x) (dotted),
the surrogate's own equilibria F(x, u) = x (dashed: where a controller trained on the surrogate thinks it can rest), the
setpoint (star), the surrogate's steady-state input for it (diamond) and the closed-form kink ellipse. Under each panel:
the tracking error over time and the input gain the surrogate reports along the true path (red band = wrong sign).
The axes are symmetric-log (linear within |x| <= 0.25, |u| <= 1) so the kink and the far setpoints fit in one view.

**Three controllers, all designed from a surrogate and run on the true plant** `x+ = 1.03 x + 0.01 u`, cost
`sum Q (x - r)^2 + R (u - u_ss(r))^2 + Q_f (x_N - r)^2` (Q = 5, R = 0.1, Q_f = 50, N = 200), |u| <= 10:
* **LQR designed at the setpoint**: the surrogate is linearised once at its own equilibrium for r. Only the slopes AT THE
  SETPOINT enter, so the colours along the path should not matter: the negative control.
* **LQR rescheduled every step**: relinearised at (x_k, u_{k-1}) each step. The slopes ALONG THE PATH enter directly.
* **DPC**: (a) the regulation controllers already trained by the DPC notebook (setpoint 0), and (b) new **tracking**
  controllers: the GRU gets (x - r, r) and is trained through the surrogate on 1000 pairs x0 ~ U[-3, 3], r ~ U[-2, 2],
  with the loss measured around the surrogate's own equilibrium input (DPC only knows its model). Reference = the same
  tracking controller trained through the true plant.

**Setpoints** r in {-2, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, 2}, 9 starts x0 in [-3, 3] each, all 10 seeds and all
six surrogates. The kink centre of every LayerNorm surrogate sits within 0.1 of (0, 0) (the encoder bias starts at zero), so
r = 0 puts the setpoint inside the kink and |r| >= 1 puts it many kink widths away.

**What was found already from the saved surrogates (LQR, no training needed)** - rerun here, plus the tracking DPC:
1. The input gain an LQR designed at the setpoint inherits is off by ~ 0.45 / d, d = distance of the setpoint from the kink
   centre in kink half-widths (log-log slope -0.98): the 1/distance fall-off of the petals. Wrong signs occur out to ~ 4-8
   half-widths (the petal arms).
2. With an accurate setpoint gain (within 20 %), the fixed LQR fails on ~ 1 % of runs whatever the path crosses; the
   rescheduled LQR fails on 1 % -> 9 % -> 27 % -> 40 % as the path's share of wrong-sign steps grows (0, 0-2, 2-10, > 10 %).
   So the colours along the path matter exactly when the controller uses the slopes along the path.

**Needs** the output folder of the DPC notebook (`MyDrive/dpc_out` on Colab: it holds `surrogates/` and `controllers/`).
Nothing in it is overwritten; this notebook adds `phase_lqr.csv`, `track_results.csv`, `track_controllers/`,
`track_trajectories/` and `figs/figP*`. **GPU:** A100 with `DTYPE = "float64"` (as the DPC run). The LQR part and the
figures take a few minutes; tracking DPC trains 7 models x 10 seeds, about 30-45 min per repeat on an A100. Re-running
skips what is done. `QUICK = True` checks everything in ~5 min with a tiny training (results in a separate folder).""")

code("""import importlib.util, subprocess, sys
for pkg in ("jax", "optax"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import jax
print("jax", jax.__version__, "| devices:", jax.devices())""")

for n, s in SRC.items():
    code(f"%%writefile {n}\n" + s)

code("""import os, sys, time, shutil
try:                                   # Colab: the DPC run's folder on Google Drive
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["DPC_OUT_ROOT"] = "/content/drive/MyDrive/dpc_out"
except ImportError:
    pass
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
sys.path.insert(0, os.getcwd())
import importlib, level1_core as L, dpc_core as DC, phase_core as PC, phase_report as PR
for m in (L, DC, PC, PR):
    importlib.reload(m)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
OUT = DC.out_root()
need = [os.path.join(OUT, "surrogates", DC.safe(m["surrogate"]) + ".npz") for m in DC.MODELS if m.get("surrogate")]
missing = [f for f in need if not os.path.exists(f)]
if missing:
    raise FileNotFoundError("This notebook needs the DPC run's folder (surrogates/ and controllers/). Missing:\\n" + "\\n".join(missing)
                            + f"\\nPoint DPC_OUT_ROOT at it, or unzip dpc_out_backup.zip into {OUT}.")
print("surrogates and controllers from", OUT)""")

md("""## Settings""")
code("""QUICK  = False                 # True: tiny tracking-DPC training (~5 min) into a separate folder, to check the setup
ARCH   = "gru"                 # tracking controller: "gru" (original), "mlp" (memoryless u = pi(x - r, r)), "linear"
SEEDS  = list(range(10))
DTYPE  = "float64"             # "float64" on an A100 (or CPU); "float32" on L4 / G4 / T4
PHASE_SEEDS = [9, 2, 6, 0]     # seeds drawn in the phase figures: 9, 2 = narrow kink (LN MSE fails), 6 = widest kink, 0
PHASE_REFS  = (0.0, 0.25, 1.0) # setpoints in the per-seed figures: inside, at the edge of, and far from the kink
STARTS      = (-2.5, -0.5, 0.5, 2.5)
RUN_TRACK   = True             # train the tracking DPC controllers
TRACK_REPS  = [0]              # tracking-DPC repeats (add 1, 2 later for robustness, like the regulation run)

PLANT = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
SUR_DIR = OUT                  # surrogates and the GRU regulation controllers are read from here
DC.ARCH = ARCH
OUT = DC.arch_dir(OUT, ARCH)   # "gru" writes into the DPC folder; other architectures into <folder>_<arch>
CURRICULUM, N_PAIRS = DC.CURRICULUM, DC.N_X0
if QUICK:
    CURRICULUM, N_PAIRS, TRACK_REPS = [(5, 20), (10, 20), (20, 20), (50, 20), (100, 10), (200, 10)], 128, [0]
    OUT = OUT.rstrip("/") + "_phase_quick"
if DTYPE == "float32":
    jax.config.update("jax_default_matmul_precision", "highest")
    OUT = OUT.rstrip("/") + "_f32"
os.makedirs(OUT, exist_ok=True)
FDTYPE = jnp.float64 if DTYPE == "float64" else jnp.float32
SURR = {m["name"]: PC.Model.surrogate(SUR_DIR, m["surrogate"], SEEDS, m["name"]) for m in DC.MODELS if m.get("kind") != "true"}
print("phase results go to", OUT)""")

md("""## 1 - LQR from every surrogate, every seed, setpoint and start (no training; ~1-2 min)
`gap` = true-plant cost / cost of the analytic LQR of the true plant from the same start - 1. Starts within 0.5 of the
setpoint are left out of the gap (they have almost no transient, so any steady offset looks like a huge relative gap).""")
code("""fn = os.path.join(OUT, "phase_lqr.csv")
if os.path.exists(fn):
    TL = pd.read_csv(fn)
else:
    TL, _, _ = PC.lqr_study(list(SURR.values()), PLANT)
    TL.to_csv(fn, index=False)
G = PR._gap_table(TL)
for c in ("LQR fixed", "LQR sched"):
    g = G[G.controller == c]
    print(f"{c}: runs failed (gap > 10 % or diverged), % of the runs at each setpoint (10 seeds x the 8 starts at least 0.5 from r)")
    display((100 * g.groupby(["model", "r"]).failed.mean().unstack()).round(0).loc[[m for m in PR.MODEL_ORDER if m in set(g.model)]])""")

md("""## Figure P3 - Where the setpoint sits relative to the kink
Cost gap, failure rate and steady offset against the setpoint (median over seeds, band = IQR over seeds). Dash-dot lines =
+- the median kink half-width of LayerNorm MSE. Expectation: the LayerNorm surrogates fail near r = 0 and recover once the
setpoint is a few kink widths away; the kink-free models are flat.""")
code("""HW = float(np.median(SURR["LN, MSE (3000)"].met["rstar_x"]))
PR.fig_gap_vs_setpoint(TL, OUT, kink_halfwidth=HW); plt.show()""")

md("""## Figure P4 - The gain an LQR designed at the setpoint inherits, against the setpoint's distance from the kink
One dot per LayerNorm surrogate, seed and setpoint. Left: |B_hat / B - 1| at the setpoint (x = wrong sign), with a power-law
fit beyond one half-width: the petal theory predicts slope -1. Right: the resulting cost gap (x at the top = diverged).""")
code("""fig, slope = PR.fig_setpoint_distance(TL, OUT); plt.show()
print(f"fitted log-log slope beyond one kink half-width: {slope:.2f} (petal theory: -1)")""")

md("""## Figure P6 - Does the path's colour matter, or only the setpoint's?
Failure rate against the share of the true path's steps where the surrogate says the input gain has the wrong sign. Left: all
runs; right: only runs whose setpoint gain is accurate (within 20 % of B), which removes the setpoint effect. Expectation: the
fixed LQR is flat on the right (it never uses the path's slopes); the rescheduled LQR rises. The tracking-DPC line is added
after section 4.""")
code("""fig, tab = PR.fig_path_vs_setpoint(TL, OUT); plt.show()
display(tab.round(1))""")

md("""## Figure P1 - Per seed: LQR phase portraits at three setpoints
Columns = setpoints (inside, at the edge of, far from the kink); top block = LQR designed at the setpoint, bottom block =
rescheduled every step. Title: the surrogate's gain and LQR gain K at the setpoint (true K = 10.3), the number of surrogate
equilibria there (3 = the dashed curve folds: the kink makes F non-monotone in u), per-start cost gap and steady offset.""")
code("""for s in PHASE_SEEDS:
    for mname in ("LN, MSE (3000)", "LN, two-stage"):
        PR.fig_phase_lqr(SURR[mname], s, PLANT, OUT, refs=PHASE_REFS, starts=STARTS); plt.show()""")

md("""## Figure P2 - Atlas: all six surrogates x four setpoints, one seed, one LQR design""")
code("""ms = [SURR[k] for k in PR.MODEL_ORDER]
for kind in ("fixed", "sched"):
    PR.fig_phase_atlas(ms, PHASE_SEEDS[0], PLANT, OUT, controller=kind, refs=(-1.0, 0.0, 0.25, 1.0), starts=STARTS); plt.show()""")

md("""## Figure P5 - Regulation DPC (the controllers of the DPC notebook, repeat 0), setpoint 0
Solid = the controller on the true plant; dashed = the same controller run through its own surrogate (what DPC was optimising).
Top row: input-gain map; middle row: state-gain error map (purple = A_hat too small, orange = too large).""")
code("""REG = ["LN, MSE (3000)", "LN, J_x + J_u (3000)", "LN, two-stage", "static norm, MSE"]
C_REG = {k: DC.load_controllers(SUR_DIR, k, 0) for k in REG}
for s in PHASE_SEEDS:
    PR.fig_phase_dpc([SURR[k] for k in REG], s, PLANT, OUT, C_REG, starts=STARTS); plt.show()""")

md("""## 4 - Tracking DPC: train through every model, test on the true plant at every setpoint
One line per horizon of the curriculum (the loss on the controller's own model), then a summary line per model.
Resumable: a model and repeat already in `track_results.csv` is skipped.""")
code("""if RUN_TRACK:
    t0 = time.time()
    TT = PC.run_track_study(DC.MODELS, SEEDS, PLANT, OUT, reps=TRACK_REPS, curriculum=CURRICULUM, n_pairs=N_PAIRS,
                            dtype=FDTYPE, sur_dir=SUR_DIR)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
TT = pd.read_csv(os.path.join(OUT, "track_results.csv"))
TT = PC.add_track_gaps(TT)
GT = PR._gap_table(TT[TT.model != "true plant (oracle)"])
print("tracking DPC: runs failed (gap > 10 % or diverged), % of seeds x starts (x repeats) at each setpoint")
display((100 * GT.groupby(["model", "r"]).failed.mean().unstack()).round(0).loc[[m for m in PR.MODEL_ORDER if m in set(GT.model)]])
print("median steady offset |x - r| by setpoint")
display(GT.groupby(["model", "r"]).offset.median().unstack().round(3).loc[[m for m in PR.MODEL_ORDER if m in set(GT.model)]])""")

md("""## Figure P3 with tracking DPC, and Figure P6 with tracking DPC
The tracking-DPC gap is measured against the tracking controller trained through the true plant (same seed, repeat,
setpoint and start); the LQR gaps against the true plant's LQR.""")
code("""ALL = pd.concat([TL, TT[TT.model != "true plant (oracle)"]], ignore_index=True)
PR.fig_gap_vs_setpoint(ALL, OUT, controllers=("LQR fixed", "LQR sched", "DPC"), kink_halfwidth=HW,
                       name="figP3b_gap_vs_setpoint_with_dpc"); plt.show()
fig, tab = PR.fig_path_vs_setpoint(ALL, OUT, name="figP6b_path_vs_setpoint_with_dpc"); plt.show()
display(tab.round(1))""")

md("""## Figure P8 - Is a failure tied to the setpoint or to the seed?
Rows = seeds, columns = setpoints, colour = share of starts that fail. LQR failures form a vertical band at the kink (they
follow the setpoint); a tracking-DPC failure, if any, fills whole rows (the trained policy itself is broken, whatever the
setpoint).""")
code("""PR.fig_seed_setpoint_grid(ALL, OUT); plt.show()""")

md("""## Figure P6c - Path colours, catastrophic failures only
Counts only failures with gap > 100 % or divergence (the 10-100 % gaps are mostly steady offsets: a value error of the
surrogate amplified by 1/B, not the kink) and leaves out controllers broken on more than half of their runs.""")
code("""fig, tab = PR.fig_path_vs_setpoint(ALL, OUT, name="figP6c_path_vs_setpoint_catastrophic", catastrophic=True, drop_broken=True)
plt.show(); display(tab.round(1))""")

md("""## Trap test for tracking DPC
The tracking controller trained through the TRUE plant, run through each surrogate. `ref_model_vs_true` ~ 1: the surrogate
judges a good controller correctly. `trap` = DPC's own controller / the reference controller, both on the surrogate: > 1 means
a better controller existed for this surrogate and DPC missed it.""")
code("""TD = PC.track_diagnostic(OUT, SUR_DIR, SEEDS, PLANT, rep=TRACK_REPS[0])
display(TD[TD.gap_true > 0.1].round(3))
display(TD.groupby("model")[["trap", "ref_model_vs_true", "gap_true"]].median().round(3))""")

md("""## Figure P7 - Tracking DPC phase portraits: one figure per seed and setpoint""")
code("""C_TR = {m["name"]: PC.load_track_controllers(OUT, m["name"], TRACK_REPS[0]) for m in DC.MODELS if m.get("kind") != "true"}
TR = ["LN, MSE (3000)", "LN, J_x + J_u (3000)", "LN, two-stage", "static norm, MSE"]
for s in PHASE_SEEDS:
    for r in PHASE_REFS:
        PR.fig_phase_dpc([SURR[k] for k in TR], s, PLANT, OUT, {k: C_TR[k] for k in TR}, starts=STARTS, r=r, track=True,
                         name=f"figP7_track_dpc_seed{s}_r{r:g}"); plt.show()""")

md("""## Save
Zips only this notebook's outputs (tables, tracking controllers and trajectories, figP* figures), not the whole DPC folder.""")
code("""import zipfile
z = OUT.rstrip("/") + "_phase_backup.zip"
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as f:
    for root, _, files in os.walk(OUT):
        for n in files:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, OUT)
            if (n.startswith(("phase_", "track_")) or rel.startswith(("track_", os.path.join("figs", "figP")))):
                f.write(p, rel)
print("zipped to", z, f"({os.path.getsize(z) / 1e6:.1f} MB)")
try:
    from google.colab import files
    files.download(z)
except ImportError:
    pass""")

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
nb.cells = cells
out = os.path.join(HERE, "dpc_phase_tracking.ipynb")
nbf.write(nb, out)
print("wrote", out)
