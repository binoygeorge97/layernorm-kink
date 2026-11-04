"""Builds dpc_through_surrogates.ipynb (self-contained: embeds level1_core.py, dpc_core.py and dpc_report.py)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {n: open(os.path.join(HERE, n)).read() for n in ("level1_core.py", "dpc_core.py", "dpc_report.py")}

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# DPC through the LayerNorm surrogates: does the kink hurt control, and does the slope loss fix it?

**Question.** A controller trained by Differentiable Predictive Control (DPC) learns from the model's gradients. Does the
LayerNorm kink make that controller worse on the TRUE plant, and does Level 1 training (the Jacobian / slope loss) bring
it back to the level of the kink-free models?

**Pipeline** (your DPC example, applied to this study):
1. *Identify the plant*: retrain the Level 1 surrogates exactly as in the Level 1 notebook (same data, initial weights,
   labels and slope-check draws; the 3000-step models match the Level 1 run to every printed digit). Plant
   `x+ = 1.03 x + 0.01 u`, 20 000 noisy transitions, x in [-4, 4], u in [-10, 10].
2. *Train a GRU controller by unrolling it through the learned model* `x_{k+1} = F(x_k, u_k)` - the whole network,
   so the controller sees the kink. Loss = your LQR-type loss (Q_x = 5, R_u = 0.1, Q_f = 50, divided by N), horizon
   curriculum 5 -> 10 -> 20 -> 50 -> 100 -> 200 (1000 / 2000 / 2000 / 1000 / 1000 / 2000 epochs), AdamW 1e-3,
   1000 initial states x0 ~ U[-3, 3], full batch.
3. *Test on the TRUE plant.* Reference = the same controller trained through the true plant ("oracle"). Headline =
   the true-plant cost gap.

**Changes from the example, each needed here:** (a) rollout through the whole surrogate, not W and V after LayerNorm
(the example's own warning: that is not the model, and the kink would never be seen); (b) `u = 10 tanh(head)`: the
surrogates only saw |u| <= 10 and the actuator is limited there (unbounded u would test extrapolation, not the kink);
(c) plain JAX instead of flax.nnx so the 10 seeds train in parallel (same GRU gates, `optax.adamw`); (d) per seed,
every model's controller starts from the same weights and trains on the same 1000 initial states: paired comparisons.

**Models** (one controller per model per seed, 10 seeds): true plant (oracle); LayerNorm MSE 3000 steps (the problem);
LayerNorm MSE 12 000 steps ("just train longer"); LayerNorm J_x + J_u from step 0; LayerNorm two-stage (3000 MSE, then
1000 MSE + slope loss); no norm and static norm (kink-free yardsticks).

**Task: regulation to x = 0.** Every trajectory ends at (x, u) = (0, 0), within about 0.1 of every model's kink centre:
the most kink-exposed task for this plant.

**Predictions, written before the run (report the outcome either way):**
* **P1** LayerNorm MSE does worse than the kink-free models, mainly from small starts (which begin inside the kink);
  about the same from large starts.
* **P2** Two-stage falls within the kink-free range.
* **P4** MSE 12 000 lands in between, failing mainly on the narrow-kink seeds (1, 2, 8, 9).
(P3 of the earlier plan concerned sampling-based MPC, which is not run here.)

**Extension (2026-10-22): repeats and a diagnostic.** (1) `REPS = [0, 1, 2]`: every controller is trained three times,
each repeat with a fresh controller initialisation and a fresh draw of its 1000 training starts. Repeat 0 is exactly the
first full run, so **point the notebook at the same output folder** (on Colab: `MyDrive/dpc_out`, the default) and only
repeats 1 and 2 train (about 1-1.5 h on an A100); the surrogates and labels are reused too. Question: is a failure a
property of the surrogate (it fails in every repeat) or of one unlucky DPC training? (2) A diagnostic that needs no
training: the controller trained through the TRUE plant is run through each surrogate. If the surrogate rates that good
controller about as the plant does, but rates the controller DPC found through it much worse, a good controller existed
for the surrogate and DPC missed it: an optimisation trap caused by the wrong slopes DPC learns from, not a model whose
predictions mislead.

**Extension (2026-10-23): controller architecture.** `ARCH` selects the controller: `"gru"` (the original study, unchanged),
`"mlp"` (memoryless u = 10 tanh(MLP(x)), 2 x 64 tanh) or `"linear"` (u ~ sat(K x + c), 2 parameters). Non-GRU results go to
`dpc_out_mlp` / `dpc_out_linear` and reuse the surrogates in `dpc_out`. See `NOTES_controller_arch.md` (predictions A1-A7).

**Extension (2026-10-23): robust per-start gradients** (last section before "Save everything"): the optimiser-side control
suggested by the trap mechanism. With the full run already in the output folder, only the two robust models train.

**Where to run.** The notebook works on Colab and Kaggle. On Colab it mounts Google Drive and writes to `MyDrive/dpc_out`, so
a disconnect loses nothing: re-run and it continues from the last finished model.

| GPU | Setting | Why |
|---|---|---|
| **A100** (Colab) - recommended | `DTYPE = "float64"` | fast 64-bit arithmetic: same precision as Level 1 |
| L4, G4 (Colab), T4 (Kaggle/Colab) | `DTYPE = "float32"` | these GPUs are slow in 64-bit; 32-bit runs at full precision (see below) |

The surrogates are always trained in float64 (as in Level 1); `DTYPE` only sets the precision of the controller training and
the closed-loop tests. With float32 the notebook forces full-precision matrix products (`jax_default_matmul_precision =
"highest"`): by default, recent NVIDIA GPUs multiply float32 matrices in TF32 (about 3 significant digits), too coarse near the
kink, where LayerNorm divides by about 0.03.

Run once with `QUICK = True` (5-10 min) to check the GPU and the time per model, then `QUICK = False` and run all. Measured
cost: ~6.7 s per epoch at horizon 200 on a 2-core CPU (float64), i.e. ~6 h per model on CPU; a GPU should take 1-3 h for all
7 models (the quick run's per-model times tell you).""")

code("""import importlib.util, subprocess, sys
for pkg in ("jax", "optax"):
    if importlib.util.find_spec(pkg) is None:      # Kaggle images usually ship both; otherwise needs Internet ON
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import jax
print("jax", jax.__version__, "| devices:", jax.devices())""")

for n, s in SRC.items():
    code(f"%%writefile {n}\n" + s)

code("""import os, sys, time, shutil
try:                                   # Colab: save to Google Drive so a disconnect loses nothing
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ["DPC_OUT_ROOT"] = "/content/drive/MyDrive/dpc_out"
except ImportError:
    pass
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
sys.path.insert(0, os.getcwd())
import importlib, level1_core as L, dpc_core as D, dpc_report as RP
importlib.reload(L); importlib.reload(D); importlib.reload(RP)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 50)
OUT = D.out_root()
os.makedirs(OUT, exist_ok=True)
print("results go to", OUT)""")

md("""## Settings
`LABEL_DIR`: the slope labels take ~16 s per seed to compute. If you attach the Level 1 output as a Kaggle dataset, point
`LABEL_DIR` at the folder with `labels_seed*_sigma0.01_epan.npz` to reuse them; otherwise they are computed and cached in
the output folder.""")

code("""QUICK  = False                 # True: 2 seeds, short surrogates, 120 controller epochs - a ~10 minute check
ARCH   = "gru"                 # controller: "gru" (the original study), "mlp" (u = 10 tanh(MLP(x)), memoryless), "linear" (u ~ K x + c)
SEEDS  = list(range(10))       # same seeds as Level 1
DTYPE  = "float64"             # "float64" on an A100 (or CPU); "float32" on L4 / G4 / T4. Surrogates are float64 either way
SIGMA, KERNEL = 0.01, "epan"   # Level 1 data noise and label kernel
LABEL_DIR = None               # e.g. "/kaggle/input/level1-out/level1_out" to reuse the Level 1 labels
ONLY   = None                  # e.g. ["true plant (oracle)", "LN, MSE (3000)"] to run a subset
REPS   = [0, 1, 2]             # DPC training repeats; 0 = the original run (reused if already in OUT)
PLANT  = L.Plant(A=1.03, B=0.01, eps_nl=0.0)   # Task 2a: L.Plant(A=1.03, B=0.1, eps_nl=0.0) together with ROOT = "dpc_out_B01"
ROOT   = None                  # None = MyDrive/dpc_out; another name = a sibling folder (labels and surrogates go inside it)
SURROGATES_ONLY = False        # True: train/load the six surrogate families (+ the models in ONLY), then stop

if ROOT:
    OUT = os.path.join(os.path.dirname(OUT.rstrip("/")), ROOT)
if (PLANT.A, PLANT.B) != (1.03, 0.01) and os.path.basename(OUT.rstrip("/")) == "dpc_out":
    raise ValueError("a plant other than (1.03, 0.01) needs its own ROOT (e.g. 'dpc_out_B01'): never mix with dpc_out")
if list(SEEDS) != list(range(10)) and not QUICK and os.path.basename(OUT.rstrip("/")) == "dpc_out":
    raise ValueError("seeds other than 0-9 need their own ROOT (e.g. 'dpc_out_s10_29'): the surrogate cache in dpc_out would be retrained")
CURRICULUM, N_X0, SURR = D.CURRICULUM, D.N_X0, D.surrogate_configs(3000, 1000, 12000)
if QUICK:
    SEEDS = [0, 1]
    CURRICULUM = [(5, 20), (10, 20), (20, 20), (50, 20), (100, 20), (200, 20)]
    N_X0, SURR = 200, D.surrogate_configs(300, 100, 1200)
    REPS = [0, 1]
    OUT = OUT.rstrip("/") + "_quick"
if DTYPE == "float32":                                  # no TF32 shortcut: full float32 precision in every matrix product
    jax.config.update("jax_default_matmul_precision", "highest")
    OUT = OUT.rstrip("/") + "_f32"                      # keep float32 results apart from float64 ones
D.ARCH = ARCH
SUR_DIR = OUT                                           # surrogates and labels: shared by every architecture
OUT = D.arch_dir(OUT, ARCH)                             # "gru" keeps the original folder; others get <folder>_<arch>
os.makedirs(OUT, exist_ok=True)
MODELS = [m for m in D.MODELS if ONLY is None or m["name"] in ONLY]
print("plant: A =", PLANT.A, "B =", PLANT.B, "| LQR gain", round(D.lqr_gain(PLANT), 4), "| controller:", ARCH,
      "| results go to", OUT, "| surrogates from", SUR_DIR)
pd.DataFrame([dict(model=m["name"], surrogate=m.get("surrogate", "- (true plant)"),
                   surrogate_training=" -> ".join(f"{k} x {n}" for k, wx, wu, n in SURR[m["surrogate"]]["phases"]) if m.get("surrogate") else "-")
              for m in MODELS])""")

md("""## Train: surrogates, then one DPC controller per model and seed
For each model: retrain (or load) its 10 surrogates, train 10 GRU controllers through them together (one line per horizon
of the curriculum: the DPC loss on the controller's own model), then test them on the true plant. One summary line per
model: median true-plant cost on 1000 typical starts, the same controller's cost on its own model, and failure counts.""")

code("""t0 = time.time()
if SURROGATES_ONLY:                                      # Task 2a stage A: surrogates (all six families) + ONLY models
    os.makedirs(SUR_DIR, exist_ok=True)
    display(D.train_all_surrogates(D.MODELS, SEEDS, PLANT, SIGMA, KERNEL, SUR_DIR, SURR, label_dir=LABEL_DIR)
            .groupby("model")[["rel_mse", "Ju_sup_rel", "rstar_x", "rstar_u"]].median())
D.run_dpc(MODELS, SEEDS, PLANT, SIGMA, KERNEL, OUT, SURR, label_dir=LABEL_DIR, curriculum=CURRICULUM, n_x0=N_X0,
          dtype=jnp.float64 if DTYPE == "float64" else jnp.float32, reps=REPS, sur_dir=SUR_DIR)
print(f"done in {(time.time() - t0) / 60:.1f} min")
if SURROGATES_ONLY or (MODELS and all(m.get("kind") == "true" for m in MODELS)):   # stage A: surrogates and/or oracle only
    z = shutil.make_archive(OUT.rstrip("/") + "_stageA", "zip", OUT)
    print("Stage A complete (surrogates and/or oracle only), zipped to", z,
          "- the rest of the notebook needs DPC-through-surrogate runs and is skipped (this stop is intended).")
    raise SystemExit("stage A run finished")""")

md("""## Check 1 - the surrogates are the Level 1 models
Medians over seeds of the Level 1 metrics of the surrogates used here. With the full settings they should match the
Level 1 run: peak input-gain error 18.7 (LN MSE), 2.3 (MSE 12 000), 7.9 (J_x + J_u from step 0), 1.15 (two-stage),
0.83 (no norm), 0.47 (static norm) x B.""")

code("""R, CUR, SUR = RP.load(OUT)
if len(SUR):
    display(SUR.groupby("model")[["Ju_sup_rel", "Ju_rms_rel", "rel_mse", "lqr_unstable", "rstar_x", "rstar_u"]].median()
            .loc[[m for m in RP.ORDER if m in set(SUR.model)]])""")

md("""## Check 2 - is the oracle controller good?
The oracle is the GRU trained through the true plant. Its true-plant cost should be close to analytic LQR with saturation
(`u = clip(-10.29 x, +-10)`) on the same 1000 starts. If it is much worse, the controller training itself (not the model)
limits every result below.""")

code("""LQR_COST = RP.lqr_reference(PLANT)
o = R[R.model == "true plant (oracle)"].cost_test_true
print(f"analytic LQR with saturation: {LQR_COST:.4f}   |   oracle DPC, median over seeds: {o.median():.4f} "
      f"(range {o.min():.4f} - {o.max():.4f}, i.e. {100 * (o.median() / LQR_COST - 1):+.1f} % vs LQR)")""")

md("""## Summary table
Median [inter-quartile range] over seeds. **gap** = cost of the controller trained through the model / cost of the oracle
controller on the same seed and start - 1, in %, all measured on the TRUE plant (0 = as good as the oracle).
* *typical*: the 1000 starts x0 ~ U[-3, 3] (the example's headline number).
* *small starts* (|x0| <= 0.2: they begin inside or next to the kink) and *large starts* (|x0| >= 1): median over the
  evaluation starts +-{0.01, 0.02, 0.05, 0.1, 0.2} and +-{1, 1.5, 2, 2.5, 3}.
* *transient only*: the same cost measured about the controller's own resting point, so a steady offset does not count.
  The offset is its own column: |x_inf| = mean |x| over the last 20 steps, left by a model whose prediction is biased.
* *diverged*: |x| > 10 anywhere; *not settled*: |x_200| > 0.05.
* *runs failed*: runs (seed x repeat) whose typical-start gap exceeds 10 %; *seeds failing in some / every repeat*.
* *share of steps inside the kink*: fraction of the (x_k, u_k) of the 22 evaluation trajectories inside the surrogate's
  closed-form kink ellipse (LayerNorm models only).""")

code("""P_START = RP.per_start(OUT)
R = RP.add_gaps(R, P_START)
R = RP.add_kink_share(R, OUT)
display(RP.summary_table(R, OUT))""")

md("""## Paired comparisons
For each seed, both controllers share the surrogate's data and initial weights and, within each repeat, the controller's
initial weights and training starts. Each seed's value is the median over its repeats, so the test has 10 pairs (one per
seed): median of A - B in percentage points (negative = A better), seeds where A is better, Wilcoxon p (10/10 gives
p = 0.002).""")

code("""display(RP.paired_table(R, OUT))""")

md("""## Figure D7 - Robustness: does a failure belong to the surrogate or to one unlucky training?
Rows = seeds, columns = models. Colour and number = the median true-plant cost gap over the repeats; second line = how many
repeats failed (gap > 10 %). A seed that fails in every repeat has a surrogate DPC cannot handle; a seed that fails once
had an unlucky training.""")
code("""RP.fig_robustness(R, OUT); plt.show()""")

md("""## Figure D8 - Diagnostic: the reference controller run through each surrogate
Left: the controller trained through the TRUE plant, run through each surrogate; its cost there / on the true plant
(1 = the surrogate judges a good controller as the plant does). Right: on each surrogate, the cost of the controller DPC
found through it / of the reference controller. Above 1 = a better controller existed for this surrogate and DPC missed it
(an optimisation trap); below 1 = the surrogate prefers DPC's controller (the model misleads). Crosses = runs that failed
on the true plant.""")
code("""DIAG = D.run_diagnostic(MODELS, SEEDS, PLANT, OUT, reps=REPS, dtype=jnp.float64 if DTYPE == "float64" else jnp.float32, sur_dir=SUR_DIR)
RP.fig_diagnostic(DIAG, R, OUT); plt.show()
d = DIAG.merge(R[["model", "seed", "rep", "gap_test"]], on=["model", "seed", "rep"])
d["own / reference on the model"] = d.own_on_model / d.oracle_on_model
d["reference: model / true plant"] = d.oracle_on_model / d.oracle_on_true
display(d[d.gap_test > RP.FAIL][["model", "seed", "rep", "gap_test", "reference: model / true plant",
                                  "own / reference on the model", "oracle_on_model_diverged"]].round(3))""")

md("""## Memoryless controllers only: the learned policy and the DPC loss landscape
Skipped for the GRU (its output depends on the history, so there is no function u = pi(x) to draw).
**Policy map (fig D9):** u = pi(x) of every seed's controller over the surrogate's input-gain profile along that policy,
with the saturated LQR u = clip(-10.29 x) for comparison; K_eff = -d pi/dx at x = 0 per seed and model.
**Loss landscape (fig D10, "linear" only, also drawn for "mlp" as a reference):** the DPC loss of the 2-parameter
controller u ~ K x + c on a (K, c) grid, through the surrogate and through the true plant, for the seeds that fail.
If DPC on this surrogate ends at a wrong K while the landscape's minimum is right, the trap is an optimisation effect.""")
code("""if ARCH != "gru":
    RP.fig_policy_maps(OUT, SUR_DIR, SEEDS, PLANT, MODELS); plt.show()
    display(RP.policy_gain_table(OUT, SEEDS, MODELS))
    RP.fig_linear_landscape(OUT, SUR_DIR, PLANT, seeds=sorted(set([RP.worst_seed(R), SEEDS[0]]))); plt.show()""")

md("""## Gradient audit (any ARCH; needs only the surrogates)
Corollary U5 says DPC's gradient through a kinked surrogate can point the wrong way. Test: for the 2-parameter controller
u ~ K x, compute dL/dK through the surrogate and through the true plant at the same gains and starts. Red shading = the two
disagree in sign, i.e. gradient descent through the surrogate moves K away from the true optimum. Figure D11.""")
code("""fig, GA = RP.fig_gradient_audit(SUR_DIR, PLANT, seeds=sorted(set([RP.worst_seed(R), SEEDS[0]]))); plt.show()
display(GA.round(3))""")

md("""## Figure D1 - Headline
One dot per seed, black bar = median, grey band = range of the two kink-free models. Dashed line (top left) = analytic
LQR with saturation relative to the oracle. Bottom right: cost the controller expects on its own model / what it gets on
the true plant (1 = the model told the truth along the closed loop).""")
code("""RP.fig_headline(R, OUT, LQR_COST); plt.show()""")

md("""## Figure D2 - Closed loops on the true plant (your `plot_closed_loop`, one column per model)
Seed where LayerNorm MSE does worst on small starts, and seed 0. Rows: x(t), |x(t)| on a log axis (shows offsets and slow
decay near zero) and u(t). Thin black = the oracle controller from the same start.""")
code("""for s in dict.fromkeys([RP.worst_seed(R), SEEDS[0]]):
    RP.fig_closed_loop(OUT, s); plt.show()""")

md("""## Figure D3 - Convergence on the true plant (your `plot_norm_comparison`)
Median over seeds of the mean |x(t)| over the starts, small starts and large starts separately.""")
code("""RP.fig_norm(OUT); plt.show()""")

md("""## Figure D4 - Where the closed loop runs, and what the model says there
True-plant trajectories (x_k, u_k) from the starts |x0| <= 0.5, drawn over the surrogate's input gain (model dF/du / true B:
white = correct, red = too large, blue = wrong sign, black contour = sign flip) and its closed-form kink (dashed ellipse).
If P1 holds, the LayerNorm MSE controller's trajectories linger or circle inside the kink.""")
code("""for s in dict.fromkeys([RP.worst_seed(R), SEEDS[0]]):
    RP.fig_phase(OUT, s, PLANT, sur_dir=SUR_DIR); plt.show()
    RP.fig_phase(OUT, s, PLANT, wx=1.5, wu=8.0, sur_dir=SUR_DIR); plt.show()      # wider window: wrong resting points away from (0, 0)""")

md("""## Figure D5 - DPC training curves
The DPC loss on the controller's own model (median over seeds, band = IQR). The loss is divided by the horizon, so it drops
at each horizon increase (dotted lines). A curve that rises at long horizons means the controller could not stabilise its
own model.""")
code("""RP.fig_curves(CUR, OUT); plt.show()""")

md("""## Figure D6 - Does the kink's size predict the closed-loop damage?
x = the surrogate's peak input-gain error (the Level 1 headline metric), y = small-start transient cost gap; one dot per
model and seed.""")
code("""RP.fig_kink_vs_gap(R, SUR, OUT); plt.show()""")

ROBUST_MD = """## Optimiser-side control (2026-10-23): robust per-start gradients
The trap diagnostic (local analysis, `NOTES_dpc_level1.md`, "Trap mechanism") found that on the failing LayerNorm MSE
seeds the surrogate's loss gradient at a GOOD controller is 1e13-1e35 times too large and chaotic in direction, and that a
SINGLE training start (one whose rollout runs through a region where the surrogate's slopes make the closed loop locally
unstable) carries 60-100 % of the batch gradient. The healthy surrogates' per-start gradients never exceed ~3.

**Control.** The same LayerNorm MSE surrogate, same controller inits and starts, but the per-start gradients are combined
with a bounded vote before AdamW sees them: `clip` = each start's gradient clipped to norm <= 3 (just above the healthy
maximum), `unit` = each normalised to unit norm (equal vote, no threshold). If this rescues seeds 1, 2, 8, 9 the failure is
the exploding per-trajectory gradient, and this is the "generic fix the optimiser" baseline a reviewer will ask for. It
costs one backward pass per start (`jax.vmap` over the 1000 starts), roughly 3-5x the time of a plain model on the A100.
Runs for whichever `ARCH` is set (the robust models are trained with the same controller class). Set `RUN_ROBUST = False` to skip.
`ROBUST_REPS` (default `[0]`, as before) sets the repeats; Task 3 uses `[0, 1, 2]`."""

ROBUST_CODE = """RUN_ROBUST = True
ROBUST_REPS = [0]              # Task 3: [0, 1, 2]
if RUN_ROBUST:
    t0 = time.time()
    D.run_dpc(D.ROBUST_MODELS, SEEDS, PLANT, SIGMA, KERNEL, OUT, SURR, label_dir=LABEL_DIR, curriculum=CURRICULUM, n_x0=N_X0,
              dtype=jnp.float64 if DTYPE == "float64" else jnp.float32, reps=ROBUST_REPS, sur_dir=SUR_DIR)
    print(f"robust models done in {(time.time() - t0) / 60:.1f} min")
    R2, _, _ = RP.load(OUT)
    R2 = RP.add_gaps(R2, RP.per_start(OUT, models=[m["name"] for m in D.MODELS + D.ROBUST_MODELS]))
    names = ["LN, MSE (3000)"] + [m["name"] for m in D.ROBUST_MODELS] + ["LN, two-stage", "static norm, MSE"]
    T = R2[(R2.rep == 0) & R2.model.isin(names)].pivot_table(index="seed", columns="model", values="gap_test")[names]
    print("true-plant cost gap vs the oracle (%), repeat 0, per seed:")
    display(T.round(2))
    print("runs failed (gap > 10 %):", {m: int((T[m] > RP.FAIL).sum()) for m in names})"""

md(ROBUST_MD)
code(ROBUST_CODE)

OPT_MD = """## Optimiser baselines (Task 3, 2026-11-03): does a standard optimiser trick already fix the trap?
Same LN MSE (3000) surrogates, same controller inits and training starts, one setting changed per model (`D.OPT_MODELS`):
`global clip 3` (the AVERAGED gradient clipped to norm <= 3, as `optax.clip_by_global_norm`), `mom` (median of means over
10 groups of 100 starts), `batch 100` (100 of the 1000 starts drawn per epoch, key 40000 + s + 1000 rep), `lr 1e-4`,
`N <= 50` and `N <= 100` (curriculum stopped at that horizon; the test is still 200 steps on the true plant).
Then the trap diagnostic (reference controller through the same surrogate) for every robust and optimiser model, and one
table: failed (typical-start gap > 10 %) and trapped (failed AND own controller > 1.05 x the reference on the surrogate)
out of len(SEEDS) x len(OPT_REPS) (the robust models are included when `RUN_ROBUST = True`). `RUN_OPT = False`
(default) skips the block."""

OPT_CODE = """RUN_OPT = False
OPT_REPS = [0, 1, 2]
if RUN_OPT:
    t0 = time.time()
    D.run_dpc(D.OPT_MODELS, SEEDS, PLANT, SIGMA, KERNEL, OUT, SURR, label_dir=LABEL_DIR, curriculum=CURRICULUM, n_x0=N_X0,
              dtype=jnp.float64 if DTYPE == "float64" else jnp.float32, reps=OPT_REPS, sur_dir=SUR_DIR)
    print(f"optimiser baselines done in {(time.time() - t0) / 60:.1f} min")
    extra = (D.ROBUST_MODELS if RUN_ROBUST else []) + D.OPT_MODELS
    done = D.done_runs(os.path.join(OUT, "dpc_results.csv"), SEEDS)
    for m in extra:                                   # only finished (model, repeat) pairs, so no empty rows are cached
        reps_done = [r for r in sorted(set(OPT_REPS) | set(ROBUST_REPS)) if (m["name"], r) in done]
        D.run_diagnostic([m], SEEDS, PLANT, OUT, reps=reps_done, dtype=jnp.float64 if DTYPE == "float64" else jnp.float32,
                         sur_dir=SUR_DIR)
    R3 = D._read_runs(os.path.join(OUT, "dpc_results.csv"))
    orc = R3[R3.model == "true plant (oracle)"][["seed", "rep", "cost_test_true"]].rename(columns={"cost_test_true": "oracle"})
    R3 = R3.merge(orc, on=["seed", "rep"])
    R3["gap"] = R3.cost_test_true / R3.oracle - 1
    DG = pd.read_csv(os.path.join(OUT, "dpc_diag.csv"))
    DG["rep"] = DG.rep.fillna(0).astype(int)
    R3 = R3.merge(DG[["model", "seed", "rep", "oracle_on_model"]], on=["model", "seed", "rep"], how="left")
    R3["failed"] = R3.gap > 0.10
    R3["trapped"] = R3.failed & (R3.cost_test_own / R3.oracle_on_model > 1.05)
    names = ["LN, MSE (3000)"] + [m["name"] for m in extra] + ["LN, two-stage"]
    T3 = (R3[R3.model.isin(names)].groupby("model")
          .agg(runs=("gap", "size"), failed=("failed", "sum"), trapped=("trapped", "sum"), median_gap_pct=("gap", "median"),
               best_seed_gap_pct=("gap", "min"), s_per_seed=("train_s_per_seed", "median"))
          .reindex(names))
    T3[["median_gap_pct", "best_seed_gap_pct"]] *= 100
    display(T3.round(3))"""

md(OPT_MD)
code(OPT_CODE)

md("""## Save everything
Download `dpc_out_backup.zip` (Kaggle: Output tab of the committed version). It holds `dpc_results.csv`, `dpc_curves.csv`,
`surrogates.csv`, the tables, figures, trajectories, and the trained surrogates and controllers.""")
code("""z = shutil.make_archive(OUT.rstrip("/") + "_backup", "zip", OUT)   # for ARCH != "gru" this zip has no surrogates/ (shared)
print("zipped to", z)
try:
    from google.colab import files
    files.download(z)
except ImportError:
    pass""")

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
nb.cells = cells
out = os.path.join(HERE, "dpc_through_surrogates.ipynb")
nbf.write(nb, out)
print("wrote", out)
