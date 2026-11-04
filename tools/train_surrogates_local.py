"""Train (or load) surrogates locally on CPU, float64, exactly as the notebook's 'surrogates only' stage
(dpc_core.train_all_surrogates): python tools/train_surrogates_local.py ROOT SEEDS MODEL_NAMES
  ROOT         output folder (surrogates/, labels in ROOT/labels, surrogates.csv); must be NEW for new seeds or norms
  SEEDS        e.g. 10-29 or 0-9 or 0,1,2
  MODEL_NAMES  ';'-separated DPC model names whose surrogates to train (dpc_core.MODELS / RMS_MODELS), or 'all' (MODELS)
Plant x+ = 1.03 x + 0.01 u (B = 0.01), sigma 0.01, kernel epan. surrogates.csv is merged with existing rows."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import pandas as pd, jax
jax.config.update("jax_enable_x64", True)
import level1_core as L, dpc_core as DC
root, sd, names = sys.argv[1], sys.argv[2], sys.argv[3]
seeds = list(range(int(sd.split("-")[0]), int(sd.split("-")[1]) + 1)) if "-" in sd else [int(s) for s in sd.split(",")]
allm = DC.MODELS + DC.RMS_MODELS
models = DC.MODELS if names == "all" else [m for m in allm if m["name"] in names.split(";")]
assert models, names
os.makedirs(os.path.join(root, "labels"), exist_ok=True)
plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
fn = os.path.join(root, "surrogates.csv")
old = pd.read_csv(fn) if os.path.exists(fn) else None
for m in models:
    if not m.get("surrogate"):
        continue
    t0 = time.time()
    S = DC.train_all_surrogates([m], seeds, plant, 0.01, "epan", root, DC.surrogate_configs(), label_dir=os.path.join(root, "labels"))
    old = S if old is None else pd.concat([old[old.model != m["name"]], S])
    old.to_csv(fn, index=False)
    print(f"{m['name']}: {time.time() - t0:.0f}s", flush=True)
