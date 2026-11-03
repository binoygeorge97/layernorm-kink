"""Task 2a regression: plant-derived constants leave the reference plant (1.03, 0.01) bit-identical, and the caches
refuse a different plant. Usage: python tools/regress_plant.py [rev]  (rev = commit before the 2a changes)."""
import os, sys, subprocess, tempfile, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import dpc_core as D, level1_core as L, phase_report as PR, phase_core as PC
REV = sys.argv[1] if len(sys.argv) > 1 else "HEAD~1"
ok = True
def check(c, m):
    global ok; print(("ok   " if c else "FAIL ") + m); ok &= bool(c)
# old modules
d = tempfile.mkdtemp()
for f in ("dpc_core.py", "level1_core.py", "phase_core.py", "phase_report.py"):
    open(os.path.join(d, f), "w").write(subprocess.run(["git", "show", f"{REV}:src/{f}"], cwd=paths.REPO, capture_output=True, text=True, check=True).stdout)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
saved = {k: sys.modules[k] for k in ("level1_core", "dpc_core", "phase_core", "phase_report")}
sys.path.insert(0, d)
for k in saved: del sys.modules[k]
Lo = load("level1_core", f"{d}/level1_core.py"); sys.modules["level1_core"] = Lo
Do = load("dpc_core", f"{d}/dpc_core.py"); sys.modules["dpc_core"] = Do
PCo = load("phase_core", f"{d}/phase_core.py"); sys.modules["phase_core"] = PCo
PRo = load("phase_report", f"{d}/phase_report.py")
sys.path.remove(d); sys.modules.update(saved)
plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
x0 = np.random.default_rng(12345).uniform(-3, 3, 1000)
a, b = Do.lqr_saturated(plant, x0), D.lqr_saturated(plant, x0)
check(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "lqr_saturated (oracle sanity reference) bit-identical at B = 0.01")
check(D.lqr_gain(plant) == 10.29 and abs(D.lqr_gain(L.Plant(B=0.1)) - 5.2538) < 1e-4, f"lqr_gain: 10.29 at B = 0.01, {D.lqr_gain(L.Plant(B=0.1)):.4f} at B = 0.1")
# lqr_screen (Level 1 metric) old vs new on random model slopes
r, us = L.eq_line(plant); rng = np.random.default_rng(0)
ah, bh = 1.03 + 0.01 * rng.standard_normal(len(r)), 0.01 * (1 + 0.5 * rng.standard_normal(len(r)))
check(Lo.lqr_screen(ah, bh, plant, r, us) == L.lqr_screen(ah, bh, plant, r, us), "level1_core.lqr_screen identical at B = 0.01")
# phase_report normalisation
m = PC.Model.surrogate(paths.DATA, "MSE (3000)", list(range(10)), "LN MSE")
mo = PCo.Model.surrogate(paths.DATA, "MSE (3000)", list(range(10)), "LN MSE")
A1, A2 = PRo.slope_maps(mo.seed_view(2), (-1, 1), (-2, 2), n=61), PR.slope_maps(m.seed_view(2), (-1, 1), (-2, 2), n=61)
check(all(np.array_equal(A1[k], A2[k]) for k in A1), "phase_report.slope_maps identical by default")
PR.set_plant(plant)
A3 = PR.slope_maps(m.seed_view(2), (-1, 1), (-2, 2), n=61)
check(all(np.array_equal(A1[k], A3[k]) for k in A1), "phase_report.slope_maps identical after set_plant(1.03, 0.01)")
# surrogate cache guard
cfg = D.surrogate_configs()["MSE (3000)"]
P, S0, met = D.train_surrogates(cfg, list(range(10)), plant, 0.01, "epan", paths.DATA)
check(np.isfinite(met["Ju_sup_rel"]).all(), "legacy surrogate file (no plant key) loads for (1.03, 0.01, 0.01)")
try:
    D.train_surrogates(cfg, list(range(10)), L.Plant(A=1.03, B=0.1), 0.01, "epan", paths.DATA); check(False, "guard")
except ValueError as e:
    check("plant" in str(e), "legacy surrogate file refused for B = 0.1: " + str(e)[:90] + "...")
# label cache guard
tmp = tempfile.mkdtemp()
dd = L.make_data(plant, 0.01, 1000)
fp = np.asarray(dd["y"][:16], float)
np.savez_compressed(os.path.join(tmp, "labels_seed0_sigma0.01_epan.npz"), data_fp=fp, Jx=np.zeros(3), Ju=np.zeros(3))
check(set(L.get_labels(dd, 0, 0.01, "epan", tmp)) == {"Jx", "Ju"}, "label cache: matching data loads, fingerprint key hidden")
try:
    L.get_labels(L.make_data(L.Plant(B=0.1), 0.01, 1000), 0, 0.01, "epan", tmp); check(False, "label guard")
except ValueError:
    check(True, "label cache refuses labels computed from another plant's data")
print("PLANT REGRESSION PASSED" if ok else "PLANT REGRESSION FAILED"); sys.exit(0 if ok else 1)
