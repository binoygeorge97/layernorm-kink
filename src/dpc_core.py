"""dpc_core.py - Differentiable Predictive Control (DPC) THROUGH the Level 1 surrogates.

Pipeline (the user's DPC example, applied to the LayerNorm-kink study):
  1. Identify the plant: retrain the Level 1 surrogates F(x, u) exactly as in level1_core (same data, init, labels,
     draws), 10 seeds per model.
  2. Train a GRU controller by unrolling it through the FULL surrogate x_{k+1} = F(x_k, u_k) (no linear extraction,
     so the controller sees the kink), with the LQR-type loss and the horizon curriculum of the example.
  3. Test every controller on the TRUE plant x+ = 1.03 x + 0.01 u. Reference = the same controller trained through the
     true plant ("oracle"); the headline number is the true-plant cost gap.

Changes from the example, each needed for this study: rollout through the whole network (not W, V after LayerNorm);
input bounded to |u| <= 10 by u = 10 tanh(head) (the surrogates only saw |u| <= 10 and the plant's actuator is limited
there); plain JAX instead of flax.nnx so the 10 seeds train together (same GRU equations, optax.adamw); per seed every
model's controller starts from the same weights and trains on the same 1000 initial states (paired comparisons).
"""
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import level1_core as L

tmap = jax.tree_util.tree_map

# ------------------------------------------------------------------------------------------------ #
# 0. Settings (the example's values)                                                               #
# ------------------------------------------------------------------------------------------------ #
CONFIG = dict(N=200, Q_x=5.0, R_u=0.1, Q_f=50.0)          # LQR-type loss and test horizon
CURRICULUM = [(5, 1000), (10, 2000), (20, 2000), (50, 1000), (100, 1000), (200, 2000)]   # (horizon, epochs)
HIDDEN = 64                                               # GRU width
CTRL_LR = 1e-3                                            # optax.adamw (default weight decay 1e-4, as in the example)
N_X0 = 1000                                               # training initial states per controller
X0_RANGE = 3.0                                            # x0 ~ U[-3, 3] (controllable range with |u| <= 10: |x| < 3.33)
U_BOUND = 10.0                                            # u = 10 tanh(head)
X0_EVAL = np.array(sorted({s * v for v in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0) for s in (-1, 1)}))
N_TEST = 1000                                             # "typical" test starts, x0 ~ U[-3, 3], same for every seed
DIVERGE = 10.0                                            # |x| above this anywhere = diverged
SETTLED = 0.05                                            # |x_N| above this at the end = not regulated
LQR_K = 10.29                                             # analytic LQR gain for (1.03, 0.01, Q=5, R=0.1)

# The models compared. 'surrogate' names a Level 1 configuration (level1_configs or wave_configs).
MODELS = [dict(name="true plant (oracle)", short="oracle", kind="true", group="oracle"),
          dict(name="LN, MSE (3000)", short="LN MSE", surrogate="MSE (3000)", group="MSE"),
          dict(name="LN, MSE (12000)", short="LN MSE\n12000", surrogate="MSE (12000)", group="MSE long"),
          dict(name="LN, J_x + J_u (3000)", short="LN slope\nfrom step 0", surrogate="J_x + J_u (3000)", group="J_x + J_u"),
          dict(name="LN, two-stage", short="LN\ntwo-stage", surrogate="J_x + J_u (two-stage)", group="two-stage"),
          dict(name="no norm, MSE", short="no\nnorm", surrogate="no norm, MSE", group="kink-free"),
          dict(name="static norm, MSE", short="static\nnorm", surrogate="static norm, MSE", group="kink-free")]

# Optimiser-side control (2026-10-23): the same LN MSE surrogate, but DPC aggregates per-start gradients with a bounded vote
# (robust_grad 'clip'). If this rescues the trapped seeds, the failure is the exploding per-trajectory gradient, and this is
# the generic "fix the optimiser" baseline. Not in MODELS by default: add it in the notebook (ROBUST_MODELS) for the A100 run.
ROBUST_MODELS = [dict(name="LN, MSE (3000), clipped grads", short="LN MSE\nclipped", surrogate="MSE (3000)", group="MSE robust",
                      aggregate="clip", clip=3.0),
                 dict(name="LN, MSE (3000), unit grads", short="LN MSE\nunit vote", surrogate="MSE (3000)", group="MSE robust",
                      aggregate="unit")]


def surrogate_configs(T=3000, T_extra=1000, T_long=12000):
    """name -> Level 1 configuration, with the step counts scaled for QUICK runs."""
    C = {c["name"]: c for c in L.level1_configs(T, T_extra)}
    long = dict(name="MSE (12000)", group="MSE", loss="MSE", schedule=str(T_long), norm="layer", labels=None,
                phases=[("mse", 0, 0, T_long)])
    C["MSE (12000)"] = long
    return C


def safe(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def out_root(name="dpc_out"):
    """DPC_OUT_ROOT (Colab: MyDrive/dpc_out) or a local folder. arch_dir() appends the controller architecture."""
    if os.environ.get("DPC_OUT_ROOT"):
        return os.environ["DPC_OUT_ROOT"]
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    return os.path.join(base, name)


# ------------------------------------------------------------------------------------------------ #
# 1. Surrogates: retrain all seeds of one Level 1 configuration together, save the weights          #
# ------------------------------------------------------------------------------------------------ #
def _flat(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "/"))
        else:
            out[prefix + k] = np.asarray(v)
    return out


def _unflat(d):
    out = {}
    for k, v in d.items():
        parts = k.split("/")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = jnp.asarray(v)
    return out


def train_surrogates(cfg, seeds, plant, sigma, kernel, out_dir, label_dir=None, M=2048, chunk=250, lam=1.0):
    """Stacked weights P [S, ...] and static-norm scales S0 [S] for one Level 1 configuration, trained exactly as
    level1_core.run_config does (same data, init, labels, slope-check draws, Adam restart at each phase switch).
    Cached in out_dir/surrogates/<name>.npz; also returns the Level 1 metrics of the trained models."""
    os.makedirs(os.path.join(out_dir, "surrogates"), exist_ok=True)
    fn = os.path.join(out_dir, "surrogates", safe(cfg["name"]) + ".npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=True)
        if list(z["seeds"]) == list(seeds):
            P = _unflat({k[2:]: z[k] for k in z.files if k.startswith("p:")})
            return P, z["S0"], {k[2:]: z[k] for k in z.files if k.startswith("m:")}
    label_dir = label_dir or out_dir
    t0 = time.time()
    datas, P0, S0, Jx_lab, Ju_lab = [], [], [], [], []
    for s in seeds:
        d = L.make_data(plant, sigma, 1000 + s)
        p = L.init_params(s)
        datas.append(d), P0.append(p)
        S0.append(L.frozen_sigma(p, d["x"], d["u"]) if cfg["norm"] == "static" else 1.0)
        if cfg.get("labels") == "est":
            lab = L.get_labels(d, s, sigma, kernel, label_dir)
            Jx_lab.append(lab["Jx"]), Ju_lab.append(lab["Ju"])
        elif cfg.get("labels") == "true":
            Jx_lab.append(plant.Jx(d["x"], d["u"])), Ju_lab.append(plant.Ju(d["x"], d["u"]))
        else:
            Jx_lab.append(np.zeros(L.N_DATA)), Ju_lab.append(np.zeros(L.N_DATA))
    D = tmap(jnp.asarray, dict(x=np.stack([d["x"] for d in datas]), u=np.stack([d["u"] for d in datas]),
                               y=np.stack([d["y"] for d in datas]), s0=np.array(S0),
                               sx=np.array([d["x"].std() for d in datas]), su=np.array([d["u"].std() for d in datas]),
                               cdf=np.tile(np.arange(1, L.N_DATA + 1) / L.N_DATA, (len(seeds), 1)),
                               Jx_lab=np.stack(Jx_lab), Ju_lab=np.stack(Ju_lab)))
    P = L.stack_trees(P0)
    KEYS = jnp.stack([jax.random.PRNGKey(10_000 + s) for s in seeds])
    lam = float(cfg.get("lam", lam))
    for kind, wx, wu, nsteps in cfg["phases"]:
        run = L.get_trainer(cfg["norm"], kind, M, chunk)
        Mo, Vo = tmap(jnp.zeros_like, P), tmap(jnp.zeros_like, P)
        for ci in range(max(1, nsteps // chunk)):
            P, Mo, Vo, KEYS = run(P, Mo, Vo, float(ci * chunk), KEYS, D, float(wx), float(wu), lam)
    jax.block_until_ready(P)
    met, _ = L.evaluate_batch(P, S0, cfg["norm"], plant, L.make_test(plant), 201, full=True)
    keep = ("rel_mse", "Ju_sup_rel", "Ju_rms_rel", "Jx_sup", "flip_area_interior", "lqr_unstable",
            "zmin_x", "zmin_u", "rstar_x", "rstar_u", "crest")
    met = {k: np.asarray(met[k], float) for k in keep if k in met}
    np.savez_compressed(fn, seeds=np.array(seeds), S0=np.array(S0), train_s=np.array(time.time() - t0),
                        **{"p:" + k: v for k, v in _flat(P).items()}, **{"m:" + k: v for k, v in met.items()})
    return P, np.array(S0), met


# ------------------------------------------------------------------------------------------------ #
# 2. Controllers. ARCH selects one; every controller has init(seed, d_x) -> c, step(c, h, x) -> (h, u), and a   #
#    hidden size (0 = memoryless: h is a dummy [B, 0] array, so rollout / phase code needs no special case).     #
#    'gru'    the example's StandaloneGRUController in plain JAX (hidden 64): the original study.               #
#    'mlp'    u = 10 tanh(MLP(x)), tanh hidden layers, memoryless: a plain policy u = pi(x), so phase plots show #
#             a function and K_eff = -d pi/dx is comparable to the LQR gain 10.29.                               #
#    'linear' u = 10 tanh((k x + b) / 10) ~ sat(-K x + c), 2 parameters per input: the whole DPC problem is a    #
#             2-D landscape that can be plotted exhaustively.                                                   #
# ------------------------------------------------------------------------------------------------ #
ARCH = "gru"                                              # default = the original study; set before training
MLP_WIDTH, MLP_DEPTH = 64, 2                              # 'mlp': hidden layers (tanh)
LIN_K0 = -5.0                                             # 'linear': initial gain u ~ LIN_K0 x (stabilising, half the LQR gain)
LIN_SCALE = 10.0                                          # 'linear' parameters are stored / LIN_SCALE so Adam's unit steps fit


def _linear_init(key, n_in, n_out, dtype):
    """flax.nnx.Linear default: lecun_normal kernel, zero bias."""
    return dict(W=jax.nn.initializers.lecun_normal()(key, (n_in, n_out), dtype), b=jnp.zeros(n_out, dtype))


def hidden_size(arch=None):
    return HIDDEN if (arch or ARCH) == "gru" else 0


def init_controller(seed, d_x=1, d_u=1, hidden=HIDDEN, dtype=jnp.float64, arch=None):
    """Controller weights from key 20000 + seed (run_keys passes seed + 1000 * repeat for repeats > 0).
    The same key gives the same GRU as before; the MLP and linear controllers draw from it too."""
    arch = arch or ARCH
    key = jax.random.PRNGKey(20_000 + int(seed))
    if arch == "gru":
        k = jax.random.split(key, 4)
        return dict(Wz=_linear_init(k[0], d_x + hidden, hidden, dtype), Wr=_linear_init(k[1], d_x + hidden, hidden, dtype),
                    Wh=_linear_init(k[2], d_x + hidden, hidden, dtype), head=_linear_init(k[3], hidden, d_u, dtype))
    if arch == "mlp":
        k = jax.random.split(key, MLP_DEPTH + 1)
        sizes = [d_x] + [MLP_WIDTH] * MLP_DEPTH + [d_u]
        return {f"L{i}": _linear_init(k[i], sizes[i], sizes[i + 1], dtype) for i in range(MLP_DEPTH + 1)}
    if arch == "linear":                                  # gain on the FIRST input (x, or x - r for tracking), 0 on the rest
        k = jnp.zeros((d_x, d_u), dtype).at[0, 0].set(LIN_K0 / LIN_SCALE)
        return dict(k=k, b=jnp.zeros(d_u, dtype))
    raise ValueError(arch)


def gru_step(c, h_prev, x):
    """h_new, u for a batch: x [B, d_x], h [B, hidden]. Same gates as the example; u = 10 tanh(head(h))."""
    lin = lambda l, a: a @ l["W"] + l["b"]
    hx = jnp.concatenate([x, h_prev], -1)
    z = jax.nn.sigmoid(lin(c["Wz"], hx))
    r = jax.nn.sigmoid(lin(c["Wr"], hx))
    h_tilde = jnp.tanh(lin(c["Wh"], jnp.concatenate([x, r * h_prev], -1)))
    h_new = (1 - z) * h_prev + z * h_tilde
    return h_new, U_BOUND * jnp.tanh(lin(c["head"], h_new))


def mlp_step(c, h, x):
    names = sorted(k for k in c if k.startswith("L"))      # L0 ... Ln
    a = x
    for nm in names[:-1]:
        a = jnp.tanh(a @ c[nm]["W"] + c[nm]["b"])
    l = c[names[-1]]
    return h, U_BOUND * jnp.tanh(a @ l["W"] + l["b"])


def linear_step(c, h, x):
    """u = 10 tanh((x K + B) / 10) with K = LIN_SCALE k, B = LIN_SCALE b: linear with gain K (u ~ K x + B) until it
    saturates at +-10. Stored parameters are k = K / 10, so an Adam step of 1e-3 moves the gain by 0.01."""
    return h, U_BOUND * jnp.tanh((x @ c["k"] + c["b"]) * LIN_SCALE / U_BOUND)


def ctrl_step(c, h, x):
    """Dispatch on the parameter tree, so a saved controller of any architecture runs with the same code."""
    if "Wz" in c:
        return gru_step(c, h, x)
    if "L0" in c:
        return mlp_step(c, h, x)
    return linear_step(c, h, x)


def policy_gain(c, x, r=None):
    """K_eff = -du/dx of a memoryless controller at x (scalar or array; r for tracking controllers)."""
    def u_of(xx):
        inp = jnp.array([xx]) if r is None else jnp.array([xx - r, r])
        return ctrl_step(c, jnp.zeros((1, 0)), inp[None])[1][0, 0]
    return -np.asarray(jax.vmap(jax.grad(u_of))(jnp.atleast_1d(jnp.asarray(x, float))))


# ------------------------------------------------------------------------------------------------ #
# 3. Dynamics the controller is trained through, and the DPC loss                                  #
# ------------------------------------------------------------------------------------------------ #
def make_dynamics(kind, norm=None, plant=None):
    """kind 'true': x+ = A x + B u. kind 'surrogate': x+ = F(x, u), the whole network (norm = layer/static/none)."""
    if kind == "true":
        A, B = float(plant.A), float(plant.B)
        return lambda p, s0, x, u: A * x + B * u
    Fb = jax.vmap(lambda p, x, u, s0: L.F_point(p, x, u, s0, norm), in_axes=(None, 0, 0, None))
    return lambda p, s0, x, u: Fb(p, x[:, 0], u[:, 0], s0)[:, None]


def rollout(c, dyn, p, s0, x0, N, hidden=None):
    """Closed loop of N steps through dyn. Returns xs [N+1, B, 1], us [N, B, 1]."""
    h0 = jnp.zeros((x0.shape[0], HIDDEN if "Wz" in c else 0), x0.dtype)

    def step(carry, _):
        h, x = carry
        h, u = ctrl_step(c, h, x)
        xn = dyn(p, s0, x, u)
        return (h, xn), (x, u)

    # Gradient checkpointing: keep only the carried (h, x) per step and recompute the step's internals in the backward
    # pass. Without it a full-size step (200 steps x 1000 starts x 10 seeds) needs ~13 GB for backpropagation.
    (h, xN), (xs, us) = jax.lax.scan(jax.checkpoint(step), (h0, x0), None, length=N)
    return jnp.concatenate([xs, xN[None]], 0), us


def dpc_loss(c, dyn, p, s0, x0, N, cfg=CONFIG):
    """The example's compute_lqr_loss: (sum_k mean(Q x_k^2 + R u_k^2) + mean(Q_f x_N^2)) / N."""
    xs, us = rollout(c, dyn, p, s0, x0, N)
    stage = jnp.sum(jnp.mean(cfg["Q_x"] * xs[:-1, :, 0] ** 2 + cfg["R_u"] * us[:, :, 0] ** 2, axis=1))
    return (stage + jnp.mean(cfg["Q_f"] * xs[-1, :, 0] ** 2)) / N


_DPC_STEPS = {}


def robust_grad(c, dyn, p, s0, x0, N, aggregate="mean", clip=1.0):
    """Loss and gradient over the batch of starts x0 [B, 1] with a choice of aggregation over starts.
    'mean'  : the ordinary full-batch gradient (gradient of the mean loss).
    'clip'  : per-start gradients, each clipped to norm <= clip, then averaged (a bounded vote per trajectory: one start
              whose rollout runs through a locally unstable region of the surrogate cannot dominate the batch).
    'unit'  : per-start gradients normalised to unit norm, then averaged (equal vote, no threshold).
    Per-start gradients cost one backward pass per start (vmap over starts), ~B x the memory of the mean gradient."""
    if aggregate == "mean":
        return jax.value_and_grad(dpc_loss)(c, dyn, p, s0, x0, N)
    one = lambda c, x: dpc_loss(c, dyn, p, s0, x[None], N)
    losses, G = jax.vmap(jax.value_and_grad(one), in_axes=(None, 0))(c, x0)          # [B], tree of [B, ...]
    nrm = jnp.sqrt(sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, 1) for g in jax.tree_util.tree_leaves(G)))   # [B]
    scale = jnp.minimum(1.0, clip / (nrm + 1e-30)) if aggregate == "clip" else 1.0 / (nrm + 1e-30)
    G = tmap(lambda g: jnp.mean(g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), 0), G)
    return jnp.mean(losses), G


def get_dpc_trainer(key, dyn, N, n_epochs, opt, aggregate="mean", clip=1.0):
    """n_epochs full-batch AdamW steps at horizon N, vmapped over seeds. Returns (C, O, losses[S, n_epochs]).
    aggregate: how per-start gradients are combined (see robust_grad)."""
    k = (key, N, n_epochs, aggregate, clip)
    if k in _DPC_STEPS:
        return _DPC_STEPS[k]

    def one(c, o, p, s0, x0):
        def st(carry, _):
            c, o = carry
            loss, g = robust_grad(c, dyn, p, s0, x0, N, aggregate, clip)
            upd, o = opt.update(g, o, c)
            return (optax.apply_updates(c, upd), o), loss
        (c, o), losses = jax.lax.scan(st, (c, o), None, length=n_epochs)
        return c, o, losses

    f = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0, 0)))
    _DPC_STEPS[k] = f
    return f


def run_keys(seed, rep):
    """Random keys of one DPC training run. rep 0 = the original run (controller init 20000 + s, starts 30000 + s);
    rep r > 0 = a fresh controller initialisation AND a fresh draw of the 1000 training starts."""
    return 20_000 + int(seed) + 1000 * int(rep), 30_000 + int(seed) + 1000 * int(rep)


def train_controllers(model, P, S0, seeds, plant, curriculum=CURRICULUM, n_x0=N_X0, log_every=250, dtype=jnp.float64,
                      verbose=True, reps=(0,)):
    """Train one controller (architecture ARCH) per (seed, repeat) through this model's dynamics, all together (vmapped).
    P, S0 are per seed; they are tiled over the repeats. Returns (C [len(reps) * S, ...], curve rows); the run order is
    repeat-major: index = k * S + i for repeat reps[k] and seed seeds[i]."""
    norm = model.get("norm")
    dyn = make_dynamics(model["kind"] if model.get("kind") == "true" else "surrogate", norm, plant)
    cast = lambda a: jnp.asarray(a, dtype) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a
    pairs = [(s, r) for r in reps for s in seeds]
    P = tmap(lambda a: jnp.concatenate([cast(a)] * len(reps), 0), P)
    S0 = jnp.asarray(np.concatenate([np.asarray(S0)] * len(reps)), dtype)
    X0 = jnp.stack([jax.random.uniform(jax.random.PRNGKey(run_keys(s, r)[1]), (n_x0, 1), dtype, -X0_RANGE, X0_RANGE)
                    for s, r in pairs])
    C = L.stack_trees([init_controller(run_keys(s, r)[0] - 20_000, dtype=dtype) for s, r in pairs])
    opt = optax.adamw(CTRL_LR)
    O = jax.vmap(opt.init)(C)
    curves, epoch, t0 = [], 0, time.time()
    for N, n_ep in curriculum:
        done = 0
        while done < n_ep:
            n = min(log_every, n_ep - done)
            step = get_dpc_trainer((model["name"], str(dtype), ARCH), dyn, N, n, opt, model.get("aggregate", "mean"), model.get("clip", 3.0))
            C, O, losses = step(C, O, P, S0, X0)
            losses = np.asarray(losses)
            done += n
            epoch += n
            for i, (s, r) in enumerate(pairs):
                curves.append(dict(model=model["name"], seed=s, rep=r, epoch=epoch, horizon=N, loss=float(losses[i, -1])))
        if verbose:
            last = np.array([r["loss"] for r in curves[-len(pairs):]])
            print(f"    horizon {N:3d}: {n_ep:5d} epochs | loss on its model: median {np.median(last):.4g}, "
                  f"max {np.max(last):.4g} | {time.time() - t0:.0f}s", flush=True)
    return C, curves


# ------------------------------------------------------------------------------------------------ #
# 4. Test on the TRUE plant                                                                        #
# ------------------------------------------------------------------------------------------------ #
def true_lqr_cost(xs, us, cfg=CONFIG):
    """The example's true_lqr_cost: (sum over steps of Q x^2 + R u^2, + Q_f x_N^2), mean over starts, / (N+1)."""
    stage = cfg["Q_x"] * xs[:-1] ** 2 + cfg["R_u"] * us ** 2
    return float((stage.sum(0) + cfg["Q_f"] * xs[-1] ** 2).mean() / xs.shape[0])


def per_start_cost(xs, us, cfg=CONFIG):
    """Unnormalised cost of each trajectory: sum_k (Q x_k^2 + R u_k^2) + Q_f x_N^2. xs [N+1, B], us [N, B]."""
    return (cfg["Q_x"] * xs[:-1] ** 2 + cfg["R_u"] * us ** 2).sum(0) + cfg["Q_f"] * xs[-1] ** 2


def lqr_saturated(plant, x0, N=CONFIG["N"], K=LQR_K):
    """Reference: analytic LQR u = clip(-K x, +-10) on the true plant."""
    xs, us = [np.asarray(x0, float)], []
    for _ in range(N):
        u = np.clip(-K * xs[-1], -U_BOUND, U_BOUND)
        us.append(u)
        xs.append(plant.A * xs[-1] + plant.B * u)
    return np.stack(xs), np.stack(us)


def _starts(n_runs, dtype):
    xe = jnp.asarray(np.repeat(X0_EVAL[None, :, None], n_runs, 0), dtype)
    xt = jnp.asarray(np.repeat(np.random.default_rng(12345).uniform(-X0_RANGE, X0_RANGE, (1, N_TEST, 1)), n_runs, 0), dtype)
    return xe, xt


def _roller(dyn, N=CONFIG["N"]):
    return jax.jit(jax.vmap(lambda c, p, s0, x0: rollout(c, dyn, p, s0, x0, N), in_axes=(0, 0, 0, 0)))


def evaluate_controllers(model, C, P, S0, seeds, plant, dtype=jnp.float64, kink=None, reps=(0,)):
    """Closed loops on the TRUE plant (and, for comparison, through the controller's own model) for the 22 evaluation
    starts and the 1000 typical starts, for every (seed, repeat) run in C (repeat-major order, as train_controllers).
    Returns (rows, {rep: trajectories})."""
    norm = model.get("norm")
    true_dyn = make_dynamics("true", plant=plant)
    own_dyn = true_dyn if model.get("kind") == "true" else make_dynamics("surrogate", norm, plant)
    cast = lambda a: jnp.asarray(a, dtype) if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a
    pairs = [(s, r) for r in reps for s in seeds]
    P = tmap(lambda a: jnp.concatenate([cast(a)] * len(reps), 0), P)
    S0 = jnp.asarray(np.concatenate([np.asarray(S0)] * len(reps)), dtype)
    xe, xt = _starts(len(pairs), dtype)
    rt, ro = _roller(true_dyn), _roller(own_dyn)
    XE, UE = (np.asarray(a, float)[..., 0] for a in rt(C, P, S0, xe))          # [runs, N+1, 22], [runs, N, 22]
    XT, UT = (np.asarray(a, float)[..., 0] for a in rt(C, P, S0, xt))
    XO, UO = (np.asarray(a, float)[..., 0] for a in ro(C, P, S0, xt))          # same controller, its own model
    rows = []
    for i, (s, rep) in enumerate(pairs):
        J = per_start_cost(XE[i], UE[i])
        fin = np.abs(XE[i, -1])
        div = ~np.isfinite(XE[i]).all(0) | (np.nanmax(np.abs(XE[i]), 0) > DIVERGE)
        row = dict(model=model["name"], group=model["group"], seed=s, rep=rep,
                   cost_test_true=true_lqr_cost(XT[i], UT[i]), cost_test_own=true_lqr_cost(XO[i], UO[i]),
                   n_diverged=int(div.sum()), n_not_settled=int(((fin > SETTLED) & ~div).sum()),
                   final_abs_max=float(np.nanmax(fin)), final_abs_median=float(np.nanmedian(fin)),
                   test_not_settled=float(np.mean(np.abs(XT[i, -1]) > SETTLED)))
        for j, x0 in enumerate(X0_EVAL):
            row[f"J[{x0:+g}]"] = float(J[j])
        k = seeds.index(s)
        if kink is not None and np.isfinite(kink["rstar_x"][k]):
            zx, zu, rx, ru = (kink[q][k] for q in ("zmin_x", "zmin_u", "rstar_x", "rstar_u"))
            inside = ((XE[i, :-1] - zx) / rx) ** 2 + ((UE[i] - zu) / ru) ** 2 <= 1.0
            row["kink_share"] = float(inside.mean())
            row["kink_share_small"] = float(inside[:, np.abs(X0_EVAL) <= 0.2].mean())
        rows.append(row)
    trajs = {}
    for r in reps:
        idx = [i for i, (s, rr) in enumerate(pairs) if rr == r]
        trajs[r] = dict(XE=XE[idx].astype(np.float32), UE=UE[idx].astype(np.float32), seeds=np.array(seeds), x0=X0_EVAL)
    return rows, trajs


# ------------------------------------------------------------------------------------------------ #
# 5. The experiment, resumable per (model, repeat)                                                 #
# ------------------------------------------------------------------------------------------------ #
def append_rows(path, rows):
    """Append rows; if they bring columns the file lacks (e.g. kink_share after the oracle's rows), widen the file."""
    if rows and os.path.exists(path):
        import pandas as pd
        old = pd.read_csv(path)
        if set(rows[0]) - set(old.columns):
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            return
    L.append_csv(path, rows)


def _read_runs(path):
    import pandas as pd
    df = pd.read_csv(path)
    if "rep" not in df:
        df["rep"] = 0                                   # files from before repeats existed = repeat 0
    df["rep"] = df["rep"].fillna(0).astype(int)
    return df


def done_runs(path, seeds):
    """{(model, rep)} with every seed finished."""
    if not os.path.exists(path):
        return set()
    df = _read_runs(path)
    return {(m, int(r)) for (m, r), g in df.groupby(["model", "rep"]) if set(seeds) <= set(g.seed)}


def rep_file(folder, name, rep):
    """rep 0 keeps the original file name, so a run made before repeats existed is reused as repeat 0."""
    return os.path.join(folder, safe(name) + ("" if rep == 0 else f"__rep{rep}") + ".npz")


def load_controllers(out_dir, name, rep):
    fn = rep_file(os.path.join(out_dir, "controllers"), name, rep)
    if not os.path.exists(fn):
        return None, None
    z = np.load(fn)
    return _unflat({k[2:]: z[k] for k in z.files if k.startswith("c:")}), list(z["seeds"])


def run_dpc(models, seeds, plant, sigma, kernel, out_dir, surr_cfgs, label_dir=None, curriculum=CURRICULUM,
            n_x0=N_X0, dtype=jnp.float64, M=2048, chunk=250, lam=1.0, reps=(0,), sur_dir=None):
    """sur_dir: folder whose surrogates/ (and labels) are reused; default out_dir. With ARCH != 'gru' pass the GRU study's
    folder as sur_dir and its arch_dir() as out_dir, so every architecture controls the SAME surrogates."""
    sur_dir = sur_dir or out_dir
    os.makedirs(os.path.join(out_dir, "controllers"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "trajectories"), exist_ok=True)
    res, cur = os.path.join(out_dir, "dpc_results.csv"), os.path.join(out_dir, "dpc_curves.csv")
    done = done_runs(res, seeds)
    todo = [(m["name"], r) for m in models for r in reps if (m["name"], r) not in done]
    print(f"{len(models)} models x {len(seeds)} seeds x repeats {list(reps)} -> {out_dir}  "
          f"({len(models) * len(reps) - len(todo)} model-repeats already done, skipped)")
    surr_rows = []
    for m in models:
        t0 = time.time()
        if m.get("kind") == "true":
            P, S0, met = L.stack_trees([L.init_params(s) for s in seeds]), np.ones(len(seeds)), None   # placeholder, unused
            m = dict(m, norm=None)
        else:
            cfg = surr_cfgs[m["surrogate"]]
            P, S0, met = train_surrogates(cfg, seeds, plant, sigma, kernel, sur_dir, label_dir, M, chunk, lam)
            m = dict(m, norm=cfg["norm"])
            for i, s in enumerate(seeds):
                surr_rows.append(dict(model=m["name"], seed=s, **{k: float(v[i]) for k, v in met.items()}))
        missing = [r for r in reps if (m["name"], r) not in done]
        if not missing:
            continue
        print(f"  DPC through '{m['name']}', repeats {missing}", flush=True)
        C, curves = train_controllers(m, P, S0, seeds, plant, curriculum, n_x0, dtype=dtype, reps=missing)
        rows, trajs = evaluate_controllers(m, C, P, S0, seeds, plant, dtype, met if m.get("norm") == "layer" else None, missing)
        for r in rows:
            r["train_s_per_seed"] = (time.time() - t0) / (len(seeds) * len(missing))
        append_rows(res, rows)
        append_rows(cur, curves)
        for k, rep in enumerate(missing):
            np.savez_compressed(rep_file(os.path.join(out_dir, "trajectories"), m["name"], rep), **trajs[rep])
            Ck = tmap(lambda a: a[k * len(seeds):(k + 1) * len(seeds)], C)
            np.savez_compressed(rep_file(os.path.join(out_dir, "controllers"), m["name"], rep), seeds=np.array(seeds),
                                **{"c:" + kk: v for kk, v in _flat(Ck).items()})
        med = lambda k: np.median([r[k] for r in rows])
        print(f"  -> {m['name']:24s} | true-plant test cost {med('cost_test_true'):.4g} (own model {med('cost_test_own'):.4g}) "
              f"| diverged {sum(r['n_diverged'] for r in rows)} | not settled {sum(r['n_not_settled'] for r in rows)} "
              f"| {time.time() - t0:.0f}s", flush=True)
    if surr_rows:
        import pandas as pd
        pd.DataFrame(surr_rows).to_csv(os.path.join(out_dir, "surrogates.csv"), index=False)


# ------------------------------------------------------------------------------------------------ #
# 6. Diagnostic: the reference controller run through each surrogate                              #
# ------------------------------------------------------------------------------------------------ #
def run_diagnostic(models, seeds, plant, out_dir, reps=(0,), dtype=jnp.float64, sur_dir=None):
    """For every surrogate model and repeat: take the controller trained through the TRUE plant (same seed and repeat)
    and run it through the surrogate. Two questions:
      1. Does the surrogate misbehave even under a good controller?  (oracle_on_model_* vs oracle_on_true)
      2. Which controller does the surrogate rate better: the good one, or the one DPC found through it?
         own_on_model < oracle_on_model -> the model prefers its own (possibly bad) controller: the MODEL misleads DPC.
         own_on_model > oracle_on_model -> a better controller existed for this model and DPC did not find it: an
         OPTIMISATION trap.
    Rows go to dpc_diag.csv (resumable per model and repeat)."""
    import pandas as pd
    path = os.path.join(out_dir, "dpc_diag.csv")
    done = set()
    if os.path.exists(path):
        d = pd.read_csv(path)
        done = {(m, int(r)) for (m, r), g in d.groupby(["model", "rep"]) if set(seeds) <= set(g.seed)}
    R = _read_runs(os.path.join(out_dir, "dpc_results.csv"))
    rows = []
    for m in models:
        if m.get("kind") == "true":
            continue
        cfg = surrogate_configs()[m["surrogate"]]
        P, S0, met, sseeds = load_surrogate(sur_dir or out_dir, m["surrogate"])
        idx = [sseeds.index(s) for s in seeds]
        P = tmap(lambda a: jnp.asarray(np.asarray(a)[idx], dtype), P)
        S0 = jnp.asarray(np.asarray(S0)[idx], dtype)
        roll = _roller(make_dynamics("surrogate", cfg["norm"], plant))
        for rep in reps:
            if (m["name"], rep) in done:
                continue
            Co, cs = load_controllers(out_dir, "true plant (oracle)", rep)
            if Co is None:
                continue
            Co = tmap(lambda a: jnp.asarray(np.asarray(a)[[cs.index(s) for s in seeds]], dtype), Co)
            xe, xt = _starts(len(seeds), dtype)
            XT, UT = (np.asarray(a, float)[..., 0] for a in roll(Co, P, S0, xt))
            XE, UE = (np.asarray(a, float)[..., 0] for a in roll(Co, P, S0, xe))
            for i, s in enumerate(seeds):
                g = R[(R.model == m["name"]) & (R.rep == rep) & (R.seed == s)]
                o = R[(R.model == "true plant (oracle)") & (R.rep == rep) & (R.seed == s)]
                fin = np.abs(XE[i, -1])
                div = ~np.isfinite(XE[i]).all(0) | (np.nanmax(np.abs(XE[i]), 0) > DIVERGE)
                rows.append(dict(model=m["name"], seed=s, rep=rep,
                                 oracle_on_model=true_lqr_cost(XT[i], UT[i]),
                                 oracle_on_true=float(o.cost_test_true.iloc[0]) if len(o) else np.nan,
                                 own_on_model=float(g.cost_test_own.iloc[0]) if len(g) else np.nan,
                                 own_on_true=float(g.cost_test_true.iloc[0]) if len(g) else np.nan,
                                 oracle_on_model_diverged=int(div.sum()),
                                 oracle_on_model_not_settled=int(((fin > SETTLED) & ~div).sum()),
                                 oracle_on_model_offset=float(np.nanmedian(np.abs(XE[i, -20:]).mean(0)))))
    if rows:
        append_rows(path, rows)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def load_surrogate(out_dir, name):
    """(P, S0, metrics) of a saved surrogate configuration."""
    z = np.load(os.path.join(out_dir, "surrogates", safe(name) + ".npz"), allow_pickle=True)
    return (_unflat({k[2:]: z[k] for k in z.files if k.startswith("p:")}), z["S0"],
            {k[2:]: z[k] for k in z.files if k.startswith("m:")}, list(z["seeds"]))


def arch_dir(out_dir, arch=None):
    """The GRU study keeps its original folder; other architectures get a sibling folder (<out>_mlp, <out>_linear)
    so results never mix. Surrogates and labels are shared: pass sur_dir=out_dir to the training functions."""
    arch = arch or ARCH
    return out_dir if arch == "gru" else out_dir.rstrip("/") + "_" + arch


def linear_landscape(dyn, p, s0, x0, N, ks, bs, cfg=CONFIG):
    """DPC loss of the LINEAR controller u ~ k x + b (gain k in plant units, saturating at +-10) on a (k, b) grid, through `dyn` (a surrogate or the true
    plant), for one seed: the whole optimisation problem DPC faces, as a 2-D map. Returns loss [len(ks), len(bs)]."""
    ks, bs = jnp.asarray(ks, x0.dtype), jnp.asarray(bs, x0.dtype)

    def one(k, b):
        c = dict(k=jnp.full((1, 1), k / LIN_SCALE, x0.dtype), b=jnp.full((1,), b / LIN_SCALE, x0.dtype))
        return dpc_loss(c, dyn, p, s0, x0, N, cfg)
    f = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None)))
    return np.asarray(f(ks, bs))


def gradient_audit(dyn_sur, dyn_true, p, s0, x0, N, ks, b=0.0, cfg=CONFIG):
    """Test of Corollary U5 on the linear controller u ~ k x + b: dL/dk through the SURROGATE and through the TRUE plant
    at the same gains k, same starts, same horizon. Where the two have opposite signs, DPC through the surrogate is pushed
    away from the direction that improves the true cost. Returns dict(k, g_sur, g_true, L_sur, L_true)."""
    ks = jnp.asarray(ks, x0.dtype)

    def loss(k, dyn):
        c = dict(k=jnp.full((1, 1), k / LIN_SCALE, x0.dtype), b=jnp.full((1,), b / LIN_SCALE, x0.dtype))
        return dpc_loss(c, dyn, p, s0, x0, N, cfg)
    vg = lambda dyn: jax.jit(jax.vmap(jax.value_and_grad(lambda k: loss(k, dyn))))
    Ls, gs = vg(dyn_sur)(ks)
    Lt, gt = vg(dyn_true)(ks)
    return dict(k=np.asarray(ks), g_sur=np.asarray(gs) / LIN_SCALE, g_true=np.asarray(gt) / LIN_SCALE,
                L_sur=np.asarray(Ls), L_true=np.asarray(Lt))
