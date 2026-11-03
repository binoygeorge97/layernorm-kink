"""t2core.py - minimal, self-contained re-implementation of the pieces of level1_core / dpc_core needed to test
Theorem 1 on the trained surrogates and Theorem 2 on the saved DPC controllers. Equations copied from
level1_core.F_point / _norm and dpc_core.gru_step / rollout / dpc_loss (float64 throughout)."""
import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

H, EPS = 16, 1e-6
A_TRUE, B_TRUE = 1.03, 0.01
Q_X, R_U, Q_F, N_HOR = 5.0, 0.1, 50.0, 200
U_BOUND, HIDDEN, N_X0, X0_RANGE = 10.0, 64, 1000, 3.0
DATA = os.environ.get("T2_DATA", "/home/claude/data")

FAMILIES = {  # display name -> (surrogate file, norm, controller file stem or None)
    "LN MSE 3000": ("MSE__3000", "layer", "LN__MSE__3000"),
    "LN MSE 12000": ("MSE__12000", "layer", None),
    "LN slope from 0": ("J_x___J_u__3000", "layer", None),
    "LN two-stage": ("J_x___J_u__two_stage", "layer", None),
    "no norm": ("no_norm__MSE", "none", None),
    "static norm": ("static_norm__MSE", "static", None),
}
RESULTS_NAME = {"LN MSE 3000": "LN, MSE (3000)", "LN MSE 12000": "LN, MSE (12000)",
                "LN slope from 0": "LN, J_x + J_u (3000)", "LN two-stage": "LN, two-stage",
                "no norm": "no norm, MSE", "static norm": "static norm, MSE"}


def _unflat(d, prefix):
    out = {}
    for k, v in d.items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].split("/")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = jnp.asarray(v)
    return out


def load_surrogate(family):
    fn, norm, _ = FAMILIES[family]
    z = np.load(os.path.join(DATA, "surrogates", fn + ".npz"), allow_pickle=False)
    P = _unflat({k: z[k] for k in z.files}, "p:")
    met = {k[2:]: np.asarray(z[k]) for k in z.files if k.startswith("m:")}
    return P, jnp.asarray(z["S0"], jnp.float64), norm, list(z["seeds"]), met


def load_controllers(stem, rep):
    fn = os.path.join(DATA, "controllers", stem + ("" if rep == 0 else f"__rep{rep}") + ".npz")
    z = np.load(fn, allow_pickle=False)
    return _unflat({k: z[k] for k in z.files}, "c:"), list(z["seeds"])


def seed_slice(tree, i):
    return jax.tree_util.tree_map(lambda a: a[i], tree)


# ------------------------------------------------------------------ surrogate (one seed, scalar x, u)
def _norm(kind, h, s0):
    if kind == "none":
        return h
    r = h - jnp.mean(h)
    if kind == "layer":
        return r / jnp.sqrt(jnp.mean(r ** 2) + EPS)
    return r / s0


def F_point(p, x, u, s0, norm):
    h = x * p["E"][0] + u * p["E"][1] + p["b"]
    v = _norm(norm, h, s0)
    a = jax.nn.gelu((p["gamma"] * v + p["beta"]) @ p["br"]["W1"] + p["br"]["b1"], approximate=False)
    return h @ p["Wd"][:, 0] + (a @ p["br"]["W2"] + p["br"]["b2"]) @ p["Wd"][:, 0] + p["bd"][0]


def slopes(p, s0, norm):
    """(x, u) -> (F, dF/dx, dF/du), scalar."""
    def f(x, u):
        return F_point(p, x, u, s0, norm)
    return lambda x, u: (f(x, u), jax.grad(f, 0)(x, u), jax.grad(f, 1)(x, u))


# ------------------------------------------------------------------ GRU controller (dpc_core.gru_step)
def gru_step(c, h_prev, x):
    """one start: h_prev [64], x scalar -> (h_new [64], u scalar)."""
    lin = lambda l, a: a @ l["W"] + l["b"]
    xv = jnp.atleast_1d(x)
    hx = jnp.concatenate([xv, h_prev])
    zg = jax.nn.sigmoid(lin(c["Wz"], hx))
    rg = jax.nn.sigmoid(lin(c["Wr"], hx))
    ht = jnp.tanh(lin(c["Wh"], jnp.concatenate([xv, rg * h_prev])))
    h_new = (1 - zg) * h_prev + zg * ht
    return h_new, U_BOUND * jnp.tanh(lin(c["head"], h_new))[0]


def rollout_one(c, p, s0, norm, x0, N=N_HOR):
    """xs [N+1], us [N], hs [N+1] (hs[k] = hidden state BEFORE step k, hs[0] = 0)."""
    def step(carry, _):
        h, x = carry
        h2, u = gru_step(c, h, x)
        xn = F_point(p, x, u, s0, norm)
        return (h2, xn), (x, u, h)
    (hN, xN), (xs, us, hs) = jax.lax.scan(step, (jnp.zeros(HIDDEN), x0), None, length=N)
    return jnp.append(xs, xN), us, jnp.concatenate([hs, hN[None]], 0)


def loss_one(c, p, s0, norm, x0, N=N_HOR):
    xs, us, _ = rollout_one(c, p, s0, norm, x0, N)
    return (jnp.sum(Q_X * xs[:-1] ** 2 + R_U * us ** 2) + Q_F * xs[-1] ** 2) / N


def dpc_loss(c, p, s0, norm, X0, N=N_HOR):
    """dpc_core.dpc_loss: mean over starts of the per-start loss (same thing, since all sums are linear)."""
    return jnp.mean(jax.vmap(lambda x0: loss_one(c, p, s0, norm, x0, N))(X0))


def training_starts(seed, rep):
    key = 30_000 + int(seed) + 1000 * int(rep)
    return jax.random.uniform(jax.random.PRNGKey(key), (N_X0, 1), jnp.float64, -X0_RANGE, X0_RANGE)[:, 0]


def test_starts():
    return jnp.asarray(np.random.default_rng(12345).uniform(-X0_RANGE, X0_RANGE, (1, 1000, 1))[0, :, 0])


def true_lqr_cost(xs, us):
    """dpc_core.true_lqr_cost on one batch: xs [N+1, B], us [N, B]."""
    stage = Q_X * xs[:-1] ** 2 + R_U * us ** 2
    return float((stage.sum(0) + Q_F * xs[-1] ** 2).mean() / xs.shape[0])


def tree_norm(g):
    return jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree_util.tree_leaves(g)))
