"""Regression check for the Q7 RMSNorm option: short surrogate trainings (2 seeds, T = 60, two-stage extra 20, 201-grid
Level 1 metrics) of existing configurations with the module at a git revision (default HEAD) and the working tree must be
bit-identical (weights and every metric); the new RMS configurations are smoke-tested (finite metrics, kink centre).
Also checks t2core.F_point old vs new on the trained LN MSE 3000 surrogates. Usage: python tools/regress_norm.py [rev]"""
import importlib.util, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
REV = sys.argv[1] if len(sys.argv) > 1 else "HEAD"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def modules(rev):
    d = tempfile.mkdtemp()
    for f in ("level1_core.py", "dpc_core.py"):
        open(os.path.join(d, f), "w").write(subprocess.run(["git", "show", f"{rev}:src/{f}"], cwd=paths.REPO, capture_output=True,
                                                           text=True, check=True).stdout)
    sys.modules.pop("level1_core", None)
    Lo = load("level1_core", os.path.join(d, "level1_core.py")); sys.modules["level1_core"] = Lo
    Do = load("dpc_core_old", os.path.join(d, "dpc_core.py"))
    sys.modules.pop("level1_core", None)
    Ln = load("level1_core", os.path.join(paths.SRC, "level1_core.py")); sys.modules["level1_core"] = Ln
    Dn = load("dpc_core_new", os.path.join(paths.SRC, "dpc_core.py"))
    return Do, Dn


def run(D, name, out):
    cfg = D.surrogate_configs(T=60, T_extra=20, T_long=60)[name]
    plant = D.L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    P, S0, met = D.train_surrogates(cfg, [0, 1], plant, 0.01, "epan", out, label_dir=LAB)
    return P, S0, met


Do, Dn = modules(REV)
LAB = tempfile.mkdtemp()                                   # labels computed once (old run), reused by the new one
bad = 0
for name in ["MSE (3000)", "J_x + J_u (two-stage)", "static norm, MSE", "no norm, MSE"]:
    a, b = run(Do, name, tempfile.mkdtemp()), run(Dn, name, tempfile.mkdtemp())
    la, lb = jax.tree_util.tree_leaves(a[0]), jax.tree_util.tree_leaves(b[0])
    w_ok = all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))
    m_ok = all(np.array_equal(np.asarray(a[2][k]), np.asarray(b[2][k]), equal_nan=True) for k in a[2])
    bad += not (w_ok and m_ok)
    print(f"{name:24s}: weights identical {w_ok}, all {len(a[2])} metrics identical {m_ok}")
for name in ["RMS MSE (3000)", "RMS J_x + J_u (two-stage)"]:
    P, S0, met = run(Dn, name, tempfile.mkdtemp())
    fin = all(np.all(np.isfinite(np.asarray(met[k], float))) for k in ("rel_mse", "Ju_sup_rel", "zmin_x", "zmin_u", "rstar_x", "rstar_u"))
    print(f"{name:24s}: smoke test finite {fin}; zmin {np.round(met['zmin_x'], 3)}, {np.round(met['zmin_u'], 3)}; rstar_x {np.round(met['rstar_x'], 3)}")
    bad += not fin
sys.path.insert(0, os.path.join(paths.REPO, "theory"))
src = subprocess.run(["git", "show", f"{REV}:theory/t2core.py"], cwd=paths.REPO, capture_output=True, text=True, check=True).stdout
d = os.path.join(paths.REPO, "theory"); open(os.path.join(d, "_t2core_old_tmp.py"), "w").write(src)
To, Tn = load("t2core_old", os.path.join(d, "_t2core_old_tmp.py")), load("t2core_new", os.path.join(paths.REPO, "theory", "t2core.py"))
P, S0, *_ = Tn.load_surrogate("LN MSE 3000")
p = Tn.seed_slice(P, 2); xs = jnp.linspace(-3, 3, 7)
f_old = jax.vmap(lambda x: To.F_point(p, x, 0.3, S0[2], "layer"))(xs); f_new = jax.vmap(lambda x: Tn.F_point(p, x, 0.3, S0[2], "layer"))(xs)
print("t2core F_point (LN MSE 3000 seed 2) identical:", bool(jnp.array_equal(f_old, f_new)))
bad += not bool(jnp.array_equal(f_old, f_new))
os.remove(os.path.join(d, "_t2core_old_tmp.py"))
print("REGRESSION", "PASSED" if bad == 0 else f"FAILED ({bad})")
