"""Which slope error hurts control: J_x (A_hat) or J_u (B_hat)? Swap one for the true value and re-measure."""
import sys; import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "paths.py").exists()))); import paths
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import level1_core as L, dpc_core as DC, phase_core as PC
OUT = paths.DATA; plant = L.Plant(); seeds = list(range(10)); dt = jnp.float64; tm = jax.tree_util.tree_map
A, B = plant.A, plant.B
# ------------------ 1. LQR fixed at the setpoint ------------------
print("=== LQR fixed: runs failed (gap > 10 % or diverged), % of runs with |x0 - r| >= 0.5 ===")
refs, starts = PC.R_EVAL, PC.X0_TRACK
rr, xx = np.meshgrid(refs, starts, indexing="ij"); r = np.tile(rr.ravel(), (10, 1)); x0 = np.tile(xx.ravel(), (10, 1))
far = np.abs(x0 - r) >= 0.5
truth = PC.Model.true_plant(plant, seeds)
xo, uo, _, _ = PC.run_lqr(truth, truth, r, x0, "fixed"); Jo = PC.track_cost(xo, uo, r, plant)
def sim(uh, K):
    x = x0.copy(); xs, us = [x], []
    for _ in range(200):
        u = np.clip(uh - K * (x - r), -10, 10); x = A * x + B * u; xs.append(x); us.append(u)
    return np.stack(xs), np.stack(us)
rows = []
for name, sur in [("LN MSE 3000", "MSE (3000)"), ("LN slope from 0", "J_x + J_u (3000)"), ("LN MSE 12000", "MSE (12000)"), ("LN two-stage", "J_x + J_u (two-stage)"), ("static norm", "static norm, MSE")]:
    m = PC.Model.surrogate(OUT, sur, seeds, name)
    uh, nroot, ah, bh = PC.equilibrium(m, r)
    for tag, K in [("both from surrogate", L.dare_K(ah, bh)), ("true A, surrogate B", L.dare_K(np.full_like(ah, A), bh)),
                   ("surrogate A, true B", L.dare_K(ah, np.full_like(bh, B))), ("both true (only u_hat from surrogate)", L.dare_K(np.full_like(ah, A), np.full_like(bh, B)))]:
        xs, us = sim(uh, K); met = PC.track_metrics(xs, us, r, plant, Jo)
        fail = (met["diverged"] | (met["gap"] > 0.1))[far]; cat = (met["diverged"] | (met["gap"] > 1.0))[far]
        rows.append(dict(surrogate=name, design=tag, failed_pct=100 * fail.mean(), catastrophic_pct=100 * cat.mean()))
    rows.append(dict(surrogate=name, design="|A_hat - A| at setpoints: median / max", failed_pct=np.median(np.abs(ah - A)), catastrophic_pct=np.abs(ah - A).max()))
    rows.append(dict(surrogate=name, design="|B_hat/B - 1| at setpoints: median / max", failed_pct=np.median(np.abs(bh / B - 1)), catastrophic_pct=np.abs(bh / B - 1).max()))
pd.set_option("display.width", 200); print(pd.DataFrame(rows).round(3).to_string(index=False))
# ------------------ 2. DPC: gradient at the reference controller with one Jacobian made true ------------------
print("\n=== DPC (GRU reference controller, 1000 training starts, horizon 200): ||grad L_sur|| and largest closed-loop Lyapunov exponent ===")
sg = jax.lax.stop_gradient
def variants(norm):
    F = DC.make_dynamics("surrogate", norm, plant)
    return {"surrogate J_x and J_u": F,
            "true J_x, surrogate J_u": lambda p, s0, x, u: F(p, s0, sg(x), u) + A * (x - sg(x)),
            "surrogate J_x, true J_u": lambda p, s0, x, u: F(p, s0, x, sg(u)) + B * (u - sg(u))}
Co, cs = DC.load_controllers(OUT, "true plant (oracle)", 0)
x0t = PC._training_starts(seeds, 0, dt)
pairs = [("MSE (3000)", s) for s in (1, 2, 8, 9, 0, 3, 7)] + [("J_x + J_u (3000)", 2), ("J_x + J_u (3000)", 9), ("MSE (12000)", 2), ("J_x + J_u (two-stage)", 2)]
rows = []
for sur, s in pairs:
    P, S0, met = PC._stack_models(OUT, sur, [s], dt); P1 = tm(lambda a: a[0], P); S01 = S0[0]
    C1 = tm(lambda a: jnp.asarray(np.asarray(a)[s], dt), Co); X0 = x0t[s]
    row = dict(surrogate=sur, seed=s, broke=(sur, s) in [("MSE (3000)", 1), ("MSE (3000)", 2), ("MSE (3000)", 8), ("MSE (3000)", 9), ("J_x + J_u (3000)", 2), ("J_x + J_u (3000)", 9)])
    for tag, dyn in variants(DC.surrogate_configs()[sur]["norm"]).items():
        g = jax.grad(lambda c: DC.dpc_loss(c, dyn, P1, S01, X0, 200))(C1)
        gn = float(jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree_util.tree_leaves(g))))
        fin = lambda c, x: DC.rollout(c, dyn, P1, S01, x[None, None], 200)[0][-1, 0, 0]
        d = jax.vmap(jax.grad(fin, argnums=1), in_axes=(None, 0))(C1, X0[:64, 0])
        row[f"grad | {tag}"] = gn; row[f"lyap | {tag}"] = float(np.max(np.log(np.abs(np.asarray(d)) + 1e-300)) / 200)
    rows.append(row); print(sur, s, "done", flush=True)
D = pd.DataFrame(rows); D.to_csv(paths.out("story", "attribution_dpc.csv"), index=False)
g = D[["surrogate", "seed", "broke"] + [c for c in D if c.startswith("grad")]]; l = D[["surrogate", "seed", "broke"] + [c for c in D if c.startswith("lyap")]]
print(g.to_string(index=False, float_format=lambda v: f"{v:.2g}")); print(l.round(3).to_string(index=False))
