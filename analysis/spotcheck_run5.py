"""A3 (2026-11-04): re-evaluate two Run 5 main controllers on the true plant (and through their own surrogate) and
compare with their dpc_results.csv rows. Usage: python analysis/spotcheck_run5.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax
jax.config.update("jax_enable_x64", True)
import level1_core as L, dpc_core as DC, phase_core as PC
plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
seeds = list(range(10))
for arch, rep in [("mlp", 0), ("linear", 1)]:
    DC.ARCH = arch
    root = os.path.join(paths.REPO, "data", f"run5_{arch}")
    m = [x for x in DC.MODELS if x["name"] == "LN, MSE (3000)"][0]
    m = dict(m, norm=DC.surrogate_configs()[m["surrogate"]]["norm"])
    C, cs = DC.load_controllers(root, m["name"], rep)
    assert cs == seeds
    P, S0, _ = PC._stack_models(paths.DATA, m["surrogate"], seeds)
    rows, _ = DC.evaluate_controllers(m, C, P, S0, seeds, plant, reps=(rep,))
    new = pd.DataFrame(rows)
    R = pd.read_csv(os.path.join(root, "dpc_results.csv"))
    old = R[(R.model == m["name"]) & (R.rep == rep)].sort_values("seed").reset_index(drop=True)
    for c in ["cost_test_true", "cost_test_own", "n_diverged", "J[-3]", "J[+0.5]"]:
        d = np.abs(new[c].values - old[c].values) / np.maximum(1e-12, np.abs(old[c].values))
        print(f"{arch:6s} rep {rep} {c:15s} max rel diff over 10 seeds {np.nanmax(d):.2e}")
    print(pd.DataFrame(dict(seed=seeds, csv=old.cost_test_true, recomputed=new.cost_test_true)).to_string(index=False))
