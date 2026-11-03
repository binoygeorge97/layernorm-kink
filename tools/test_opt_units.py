"""Unit tests of the Task 3 optimiser options in dpc_core (CPU, seconds).
1 'global' = optax.clip_by_global_norm applied to the mean gradient.
2 'mom' = coordinate-wise median of 10 group means of the per-start gradients (manual computation).
3 minibatch draws: deterministic per (seed, rep, epoch), different across epochs, without replacement.
4 max_horizon drops the curriculum stages above it (curve rows only at horizons <= max_horizon)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402,F401
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
import dpc_core as D  # noqa: E402
import level1_core as L  # noqa: E402

jax.config.update("jax_enable_x64", True)
tm, leaves = jax.tree_util.tree_map, jax.tree_util.tree_leaves
ok = True


def check(c, msg):
    global ok
    print(("ok   " if c else "FAIL ") + msg)
    ok &= bool(c)


plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
P, S0, _, ss = D.load_surrogate(paths.DATA, "MSE (3000)")
p, s0 = tm(lambda a: jnp.asarray(np.asarray(a)[2]), P), float(np.asarray(S0)[2])
dyn = D.make_dynamics("surrogate", "layer", plant)
c = D.init_controller(2)
x0 = jax.random.uniform(jax.random.PRNGKey(30002), (100, 1), jnp.float64, -3, 3)
N = 50

# 1 global
l_m, g_m = D.robust_grad(c, dyn, p, s0, x0, N, "mean")
l_g, g_g = D.robust_grad(c, dyn, p, s0, x0, N, "global", 0.01)
ref, _ = optax.clip_by_global_norm(0.01).update(g_m, optax.clip_by_global_norm(0.01).init(g_m))
err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves(g_g), leaves(ref)))
nrm = float(jnp.sqrt(sum(jnp.sum(a ** 2) for a in leaves(g_g))))
check(err < 1e-15 and abs(nrm - 0.01) < 1e-12 and l_g == l_m, f"global clip = optax.clip_by_global_norm (max diff {err:.1e}, norm {nrm:.6f})")
_, g_big = D.robust_grad(c, dyn, p, s0, x0, N, "global", 1e30)
check(all(np.array_equal(a, b) for a, b in zip(leaves(g_big), leaves(g_m))), "global clip with a huge threshold = mean gradient")

# 2 median of means
_, g_mom = D.robust_grad(c, dyn, p, s0, x0, N, "mom", groups=10)
one = lambda cc, x: D.dpc_loss(cc, dyn, p, s0, x[None], N)
G = [jax.grad(one)(c, x0[i]) for i in range(100)]
manual = tm(lambda *gs: jnp.median(jnp.stack([jnp.mean(jnp.stack(gs[10 * k:10 * k + 10]), 0) for k in range(10)]), 0), *G)
err = max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(leaves(g_mom), leaves(manual)))
check(err < 1e-12, f"median of means = manual computation (max diff {err:.1e})")

# 3 minibatch draws
bkey = jax.random.PRNGKey(D.BATCH_KEY + 2 + 1000 * 1)
draw = lambda e: np.asarray(jax.random.choice(jax.random.fold_in(bkey, e), 1000, (100,), replace=False))
d0, d0b, d1 = draw(0), draw(0), draw(1)
check(np.array_equal(d0, d0b) and not np.array_equal(d0, d1) and len(set(d0)) == 100,
      f"minibatch draws deterministic, change with the epoch, no repeats (overlap of epochs 0/1: {len(set(d0) & set(d1))} of 100)")

# 4 max_horizon
m = dict(D.OPT_MODELS[4], norm="layer")
_, curves = D.train_controllers(m, tm(lambda a: a[None], p), np.array([s0]), [2], plant,
                                [(5, 2), (10, 2), (50, 2), (100, 2), (200, 2)], n_x0=50, log_every=2, verbose=False)
check(sorted({r["horizon"] for r in curves}) == [5, 10, 50], f"max_horizon 50 keeps horizons {sorted({r['horizon'] for r in curves})}")
print("ALL UNIT TESTS PASSED" if ok else "SOME UNIT TESTS FAILED")
sys.exit(0 if ok else 1)
