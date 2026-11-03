"""Failure-story figures (2026-10-24). Needs <OUT>/surrogates and the repeat-0 controllers (GRU) of dpc_out, plus story_data.py's outputs.
Run: story_data.py (8 min CPU) -> lqr_story.py -> dpc_story.py. Figures S1-S4, animations A1-A3 in the story folder."""
"""Data for the failure-story figures: (1) warm start with checkpoints + per-start gradient norms, (2) closed-loop
sensitivity along one rollout, (3) saved closed loops of the repeat-0 controllers on the true plant."""
import sys, time, os; sys.path.insert(0, "/home/claude/kinkwork")
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import level1_core as L, dpc_core as DC, phase_core as PC
OUT = "/home/claude/data/dpc_out"; ST = "/home/claude/story"; plant = L.Plant(); seeds = list(range(10)); dt = jnp.float64
tm = jax.tree_util.tree_map; leaves = jax.tree_util.tree_leaves
t0 = time.time()
pairs = [("MSE (3000)", 2), ("MSE (3000)", 9), ("static norm, MSE", 2)]
norm_of = {"MSE (3000)": "layer", "static norm, MSE": "static"}
Co, _ = DC.load_controllers(OUT, "true plant (oracle)", 0)
dyn_t = DC.make_dynamics("true", plant=plant)
roll_t = DC._roller(dyn_t, 200)

# ---------- (1) warm start with checkpoints (each pair alone: different norms) ----------
N, EPOCHS, LOG = 200, 100, 5
CK = (0, 5, 10, 25, 50, 100)                          # per-start gradient norms at these epochs
rows, ckpt = [], {}
for sur, s in pairs:
    P, S0, met = PC._stack_models(OUT, sur, [s], dt); x0 = PC._training_starts(seeds, 0, dt)[s:s + 1]
    C = tm(lambda a: jnp.asarray(np.asarray(a)[s:s + 1], dt), Co)
    dyn_s = DC.make_dynamics("surrogate", norm_of[sur], plant); roll_s = DC._roller(dyn_s, N)
    opt = optax.adamw(DC.CTRL_LR); O = jax.vmap(opt.init)(C)
    step = DC.get_dpc_trainer(("story", norm_of[sur]), dyn_s, N, LOG, opt)
    one = lambda c, p, s0, x: DC.dpc_loss(c, dyn_s, p, s0, x[None], N)
    gps = jax.jit(jax.vmap(lambda c, p, s0, X: jax.vmap(lambda x: jnp.sqrt(sum(jnp.sum(l ** 2) for l in leaves(jax.grad(one)(c, p, s0, x)))))(X)))
    def record(e, C):
        Xt, Ut = (np.asarray(a, float)[0, ..., 0] for a in roll_t(C, P, S0, x0))
        Xs, Us = (np.asarray(a, float)[0, ..., 0] for a in roll_s(C, P, S0, x0))
        d = dict(X_true=Xt.astype(np.float32), U_true=Ut.astype(np.float32), X_sur=Xs.astype(np.float32), U_sur=Us.astype(np.float32))
        if e in CK:
            d["gnorm"] = np.asarray(gps(C, P, S0, x0))[0]
        ckpt[(sur, s, e)] = d
        rows.append(dict(sur=sur, seed=s, epoch=e, loss_true=DC.true_lqr_cost(Xt, Ut), loss_sur=DC.true_lqr_cost(Xs, Us)))
    record(0, C)
    for e in range(0, EPOCHS, LOG):
        C, O, losses = step(C, O, P, S0, x0); record(e + LOG, C)
    print(sur, s, f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"{ST}/warm_ckpt.npz", **{f"{sur}|{s}|{e}|{k}": v for (sur, s, e), d in ckpt.items() for k, v in d.items()},
                    x0=np.asarray(PC._training_starts(seeds, 0, dt))[:, :, 0])
pd.DataFrame(rows).to_csv(f"{ST}/warm_ckpt.csv", index=False)

# ---------- (2) closed-loop sensitivity along one rollout: reference controller through surrogate vs true plant ----------
def sens_path(C1, dyn, P1, S01, x0s):
    """x_k and ||d(h_k, x_k)/d x_0|| for k = 0..N along the rollout of one controller from one start."""
    def stepf(hx):
        h, x = hx[:-1][None], hx[-1:][None]
        h2, u = DC.ctrl_step(C1, h, x); x2 = dyn(P1, S01, x, u)
        return jnp.concatenate([h2[0], x2[0]])
    J = jax.jit(jax.jacfwd(stepf)); f = jax.jit(stepf)
    hx = jnp.concatenate([jnp.zeros(DC.HIDDEN, dt), jnp.array([x0s], dt)])
    v = np.zeros(DC.HIDDEN + 1); v[-1] = 1.0; v = jnp.asarray(v)
    xs, ss = [float(hx[-1])], [1.0]
    for k in range(200):
        v = J(hx) @ v; hx = f(hx)
        xs.append(float(hx[-1])); ss.append(float(jnp.linalg.norm(v)))
    return np.array(xs), np.array(ss)
sens = {}
for sur, s in [("MSE (3000)", 2), ("MSE (3000)", 8), ("static norm, MSE", 2)]:
    P, S0, met = PC._stack_models(OUT, sur, [s], dt); P1 = tm(lambda a: a[0], P); S01 = S0[0]
    C1 = tm(lambda a: jnp.asarray(np.asarray(a)[s], dt), Co)
    dyn_s = DC.make_dynamics("surrogate", norm_of[sur], plant)
    g = ckpt[(sur, s, 0)]["gnorm"] if (sur, s, 0) in ckpt else None
    x0s = np.asarray(PC._training_starts(seeds, 0, dt))[s, :, 0]
    top = int(np.argmax(g)) if g is not None else 0
    for tag, dyn in (("sur", dyn_s), ("true", dyn_t)):
        xs, ss = sens_path(C1, dyn, P1, S01, float(x0s[top]))
        sens[f"{sur}|{s}|{tag}|x"] = xs; sens[f"{sur}|{s}|{tag}|s"] = ss
    sens[f"{sur}|{s}|x0"] = np.array([x0s[top]])
    print("sens", sur, s, "start", x0s[top], f"{time.time()-t0:.0f}s", flush=True)
np.savez_compressed(f"{ST}/sens.npz", **sens)

# ---------- (3) closed loops of the saved repeat-0 controllers on the true plant (and through their own surrogate) ----------
starts = np.array([-2.5, -1.5, -0.8, -0.4, -0.15, 0.15, 0.4, 0.8, 1.5, 2.5])
cl = {}
for name, sur in [("true plant (oracle)", None), ("static norm, MSE", "static norm, MSE"), ("LN, MSE (3000)", "MSE (3000)"), ("LN, two-stage", "J_x + J_u (two-stage)")]:
    Cm, cs = DC.load_controllers(OUT, name, 0); Cm = tm(lambda a: jnp.asarray(np.asarray(a), dt), Cm)
    if sur is None:
        P = L.stack_trees([L.init_params(s) for s in seeds]); S0 = jnp.ones(10, dt); dyn_s = dyn_t
    else:
        P, S0, met = PC._stack_models(OUT, sur, seeds, dt); dyn_s = DC.make_dynamics("surrogate", DC.surrogate_configs()[sur]["norm"], plant)
    xe = jnp.asarray(np.repeat(starts[None, :, None], 10, 0), dt)
    Xt, Ut = (np.asarray(a, float)[..., 0] for a in roll_t(Cm, P, S0, xe))
    Xs, Us = (np.asarray(a, float)[..., 0] for a in DC._roller(dyn_s, 200)(Cm, P, S0, xe))
    cl[f"{name}|X_true"], cl[f"{name}|U_true"], cl[f"{name}|X_sur"], cl[f"{name}|U_sur"] = Xt, Ut, Xs, Us
np.savez_compressed(f"{ST}/closed_loops.npz", starts=starts, **cl)
print("done", f"{time.time()-t0:.0f}s")
