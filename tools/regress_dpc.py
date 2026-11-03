"""Regression check for dpc_core changes: the module at a git revision (default HEAD) vs the working tree.

Trains short DPC runs (2 seeds, 200 starts, a 3-stage mini curriculum, CPU float64) through the LN MSE 3000 surrogates and
the true plant with both versions and requires bit-identical controllers and loss curves for every model that exists in
the old version (default behaviour must not change). New models are smoke-tested (finite results).
Usage: python tools/regress_dpc.py [rev] [--new MODEL_LIST_ATTR ...]"""
import importlib.util
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
REV = next((a for a in sys.argv[1:] if not a.startswith("--")), "HEAD")
NEW_LISTS = [a[6:] for a in sys.argv[1:] if a.startswith("--new=")] or ["OPT_MODELS"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def old_module():
    d = tempfile.mkdtemp()
    for f in ("dpc_core.py", "level1_core.py"):
        src = subprocess.run(["git", "show", f"{REV}:src/{f}"], cwd=paths.REPO, capture_output=True, text=True, check=True).stdout
        open(os.path.join(d, f), "w").write(src)
    sys.path.insert(0, d)
    L_old = load_module("level1_core", os.path.join(d, "level1_core.py"))
    sys.modules["level1_core"] = L_old
    D_old = load_module("dpc_core_old", os.path.join(d, "dpc_core.py"))
    sys.path.remove(d)
    del sys.modules["level1_core"]
    return D_old


def train(D, model, seeds=(0, 2), reps=(0, 1)):
    import level1_core as L  # noqa: F401
    plant = D.L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    if model.get("kind") == "true":
        P, S0 = D.L.stack_trees([D.L.init_params(s) for s in seeds]), np.ones(len(seeds))
        m = dict(model, norm=None)
    else:
        P, S0, met, ss = D.load_surrogate(paths.DATA, model["surrogate"])
        idx = [ss.index(s) for s in seeds]
        P = jax.tree_util.tree_map(lambda a: jnp.asarray(np.asarray(a)[idx]), P)
        S0 = np.asarray(S0)[idx]
        m = dict(model, norm=D.surrogate_configs()[model["surrogate"]]["norm"])
    cur = [(5, 20), (20, 20), (50, 10)]
    C, curves = D.train_controllers(m, P, S0, list(seeds), plant, cur, n_x0=200, log_every=20, verbose=False, reps=reps)
    return C, np.array([c["loss"] for c in curves])


def same(a, b):
    la, lb = jax.tree_util.tree_leaves(a[0]), jax.tree_util.tree_leaves(b[0])
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)) and np.array_equal(a[1], b[1])


def main():
    import dpc_core as D_new
    D_old = old_module()
    ok = True
    old_models = D_old.MODELS[:2] + D_old.ROBUST_MODELS
    for arch in ("gru", "mlp"):
        D_old.ARCH = D_new.ARCH = arch
        for m in old_models:
            r_old, r_new = train(D_old, m), train(D_new, m)
            eq = same(r_old, r_new)
            ok &= eq
            print(f"[{arch}] {m['name']:34s} old vs new bit-identical: {eq}")
    D_new.ARCH = "gru"
    for lst in NEW_LISTS:
        for m in getattr(D_new, lst):
            C, losses = train(D_new, m)
            fin = bool(np.isfinite(losses).all() and all(np.isfinite(np.asarray(l)).all() for l in jax.tree_util.tree_leaves(C)))
            ok &= fin
            print(f"[gru] new {m['name']:34s} finite: {fin} | final losses {np.round(losses[-4:], 4)}")
    print("REGRESSION PASSED" if ok else "REGRESSION FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
