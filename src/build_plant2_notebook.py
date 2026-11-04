"""Builds plant2_P1.ipynb (Task 2b, plant P1; self-contained: embeds plant2_core.py and plant2_session.py)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {n: open(os.path.join(HERE, n)).read() for n in ("plant2_core.py", "plant2_session.py")}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Task 2b: the two-state plant P1 - surrogates and oracle controllers

Plant P1 (confirmed 4 Nov): `x+ = A x + B u`, `A = [[1.02, 0.05], [0, 1.01]]`, `B = (0, 0.01)^T`, `|u| <= 10`. Data box
`|x1| <= 7.3, |x2| <= 5.6, |u| <= 10`, 20 000 noisy transitions (sigma 0.01) per seed. The six surrogate families of the
scalar study with a 3-input / 2-output pre-norm block (H = 16); slope labels from a local-quadratic fit in 3 inputs, k = 1280.

**This session (STAGE = "surrogates+oracle")**: train the 6 families x 10 seeds and the oracle controllers (DPC through the
TRUE plant; GRU and MLP; 3 repeats), save everything to Google Drive (`MyDrive/<ROOT>`), print the Level 1 / kink table and
the oracle sanity check against the saturated LQR. No DPC through a surrogate happens here: that is the next session, after
the pre-registration is committed.

Set `QUICK = True` for a ~10 minute pipeline check (2 seeds, short training, 1 repeat, results in `<ROOT>_quick`).""")

code("""import importlib, subprocess, sys
for pkg in ("jax", "optax", "scipy"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
import jax
print("jax", jax.__version__, "| devices:", jax.devices())""")

for n, s in SRC.items():
    code(f"%%writefile {n}\n" + s)

code("""QUICK  = False                 # True: 2 seeds, phases x 0.1, oracle curriculum x 0.05, 1 repeat, 200 starts
SEEDS  = list(range(10))
REPS   = [0, 1, 2]
ARCHS  = ["gru", "mlp"]
ROOT   = "dpc_out_P1"           # Drive folder MyDrive/<ROOT> (new); QUICK appends _quick
STAGE  = "surrogates+oracle\"""")

code("""import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import plant2_core as P2, plant2_session as S
importlib.reload(P2); importlib.reload(S)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    base = "/content/drive/MyDrive"
except ImportError:
    base = os.environ.get("P2_OUT_BASE", os.getcwd())
OUT = os.path.join(base, ROOT + ("_quick" if QUICK else ""))
os.makedirs(OUT, exist_ok=True)
if QUICK:
    SEEDS, REPS = [0, 1], [0]
scale = 0.1 if QUICK else 1.0
cur = [(N, max(1, E // 20)) for N, E in P2.CURRICULUM] if QUICK else P2.CURRICULUM
plant = S.plant_p1()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("results go to", OUT, "| seeds", SEEDS, "| reps", REPS)""")

code("""t0 = time.time()
L1 = S.stage_surrogates(OUT, SEEDS, plant, scale=scale)
print(f"surrogates done in {(time.time() - t0) / 60:.1f} min")
cols = ["rel_mse", "Ju_sup_rel", "Ju1_sup_rel", "Ju2_sup_rel", "Jx_sup", "flip_area", "D", "z_x1", "z_x2", "z_u", "r_x1", "r_x2", "r_u"]
display(L1.groupby("family")[[c for c in cols if c in L1]].median().round(4))""")

code("""t0 = time.time()
OR = S.stage_oracle(OUT, SEEDS, plant, reps=REPS, archs=ARCHS, curriculum=cur, n_x0=200 if QUICK else P2.N_X0)
print(f"oracles done in {(time.time() - t0) / 60:.1f} min")
display(OR.groupby(["arch", "rep"])[["cost_test_true", "gap_vs_sat_lqr"]].describe().round(4))""")

code("""import shutil
print("files in", OUT)
for r, _, fs in os.walk(OUT):
    for f in sorted(fs):
        print(" ", os.path.relpath(os.path.join(r, f), OUT), os.path.getsize(os.path.join(r, f)))
shutil.make_archive(OUT, "zip", OUT)
print("zip:", OUT + ".zip  (download it, or tell Claude the Drive folder)")""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, os.path.join(HERE, "plant2_P1.ipynb"))
print("wrote", os.path.join(HERE, "plant2_P1.ipynb"))
