"""Q8 milestone (i), CPU: unit checks of src/plant2_core.py on the proposal's P1 plant (A = [[1.02, 0.05], [0, 1.01]],
B = [0, 0.01]). 1. data shapes, plant equilibria; 2. 3-input label estimator on noiseless data of the linear plant returns
(A, B) to 1e-7; 3. lambda_from: diagonal multipliers = max of the two scalar max-subarray statistics; 1x1 = scalar;
4. one LN MSE surrogate (short training), Level 1 + kink metrics (ellipsoid) and a mini LQR study; 5. a short DPC run
through the true plant vs the saturated LQR cost. Usage: python tools/test_plant2.py [--steps 300]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, jax, jax.numpy as jnp
import plant2_core as P2
steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 300
A, B = np.array([[1.02, 0.05], [0.0, 1.01]]), np.array([0.0, 0.01])
pl = P2.Plant2(A, B)
ok = True
d = P2.make_data(pl, 0.01, 1000)
x2, us = pl.u_ss([1.0, -2.0])
res = [(A - np.eye(2)) @ np.array([r, a]) + B * b for r, a, b in zip([1.0, -2.0], x2, us)]
print(f"1. data {d['x'].shape} {d['u'].shape} {d['y'].shape}; equilibria x1=1: x2 {x2[0]:.3f} u {us[0]:.3f}; residual {np.abs(res).max():.1e}")
ok &= np.abs(res).max() < 1e-12
d0 = P2.make_data(pl, 0.0, 1000, n=4000)
J = P2.LabelEstimator3(np.column_stack([d0["x"], d0["u"]]), d0["y"], k=200)(np.column_stack([d0["x"], d0["u"]])[:50])
e = np.abs(J - np.concatenate([A, B[:, None]], 1)[None]).max()
print(f"2. labels on noiseless linear data: max error {e:.1e}"); ok &= e < 1e-7                                                     # ridge 1e-8 in the WLS
rng = np.random.default_rng(0)
S, N = 3, 60
m = np.exp(rng.normal(0, 0.5, (S, N, 2)))
Ms = np.zeros((S, N, 2, 2)); Ms[..., 0, 0], Ms[..., 1, 1] = m[..., 0], m[..., 1]
xs = np.zeros((S, N, 2)); xs[:, :5] = 1.0                          # landing at step 5


def maxsub(a):
    best = cur = 0.0
    for v in a:
        cur = max(v, cur + v); best = max(best, cur)
    return best
lv, kl = P2.lambda_from(xs, Ms)
ref = np.array([max(maxsub(np.log(m[s, 5:, 0])), maxsub(np.log(m[s, 5:, 1]))) for s in range(S)])
print(f"3. diagonal case: max |vector - scalar| {np.abs(lv - ref).max():.1e}; landing {kl}")
ok &= np.abs(lv - ref).max() < 1e-10 and (kl == 5).all()
l1, _ = P2.lambda_from(xs[..., :1], Ms[..., :1, :1])
ref1 = np.array([maxsub(np.log(m[s, 5:, 0])) for s in range(S)])
ok &= np.abs(l1 - ref1).max() < 1e-10
print(f"   1x1 case: max |vector - scalar| {np.abs(l1 - ref1).max():.1e}")
t0 = time.time()
cfg = P2.configs(steps / 3000)["MSE (3000)"]
p, s0, _ = P2.train(cfg, 0, pl, verbose=True)
met = P2.level1_metrics(p, s0, "layer", pl, n=11)
print(f"4. LN MSE seed 0, {steps} steps ({time.time() - t0:.0f}s): " + ", ".join(f"{k} {v:.3g}" for k, v in met.items()))
ok &= all(np.isfinite(v) for v in met.values())
rows = P2.lqr_study(p, s0, "layer", pl, [0.0, 1.0], [(2.0, 0.0), (-2.0, 1.0)], N=100)
print("   mini LQR study:", [(r["controller"], r["r"], round(r["gap"], 3), r["diverged"]) for r in rows])
X0 = jnp.asarray(np.random.default_rng(1).uniform([-3, -2], [3, 2], (64, 2)))
c, hist = P2.train_dpc(P2.true_dyn(pl), jax.random.PRNGKey(20000), X0, [(5, 100), (20, 100), (50, 100)])
xsd, usd = jax.vmap(lambda x0: P2.rollout(c, P2.true_dyn(pl), x0, 50))(X0)
print(f"5. short DPC through the true plant: loss by stage {[round(h[1], 3) for h in hist]}; final |x| median {float(jnp.median(jnp.abs(xsd[:, -1]))):.3f}")
lam, kl = P2.vector_lambda(c, p, s0, "layer", X0[:16], N=50)
print(f"   vector Lambda through the surrogate (16 starts, N = 50): max {lam.max():.2f}, landed {(kl < 50).sum()} of 16")
print("MILESTONE (i)", "PASSED" if ok else "FAILED")
