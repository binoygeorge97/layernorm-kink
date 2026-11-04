"""Two-state plant study (CLAUDE.md Task 2b, Q8): n_x-general versions of the scalar pieces, in a SEPARATE module so that
level1_core / dpc_core / phase_core stay untouched (scalar code paths bit-identical by construction).
The plant (A, B) is a parameter everywhere; nothing here picks the plant. Test case: proposal P1 (docs/plant2/proposal.tex).

Contents
  Plant2                 x+ = A x + B u (A 2x2, B 2x1), |u| <= UMAX; Jx, Ju; u_ss(r) for a position setpoint x1 = r
  make_data              n transitions uniform in the data box, y = f + sigma N(0, I); numpy rng(data_seed)
  init_params(seed)      pre-norm residual block with n_in = 3 inputs (x1, x2, u), n_out = 2 (E: 3 x H, Wd: H x 2)
  F_point, jac           surrogate F_hat(x, u) in R^2 and its 2 x 3 Jacobian [J_x_hat (2x2) | J_u_hat (2x1)]
  frozen_sigma           static-norm scale at initialisation
  LabelEstimator3        local-quadratic WLS in 3 inputs (Epanechnikov, k nearest in std units), per output
  train                  full-batch Adam; phases ("mse", ., ., T) or ("sob", wx, wu, T) with per-input scales
  level1_metrics         rel. test MSE, peak / rms |J_u_hat - B| / |B|, state-slope error (spectral norm), flip area
  kink3                  Theorem 1 with n_z = 3: centre z* in R^3, D, half-widths D / ||q_i|| per input axis (ellipsoid)
  lqr_K, dare            LQR gain from (A_hat, B_hat) via scipy DARE
  lqr_study              fixed (setpoint linearisation) and rescheduled (relinearise at (x_k, u_{k-1})) LQR on the TRUE plant
  Controllers / DPC      GRU (hidden 64) and MLP with d_x = 2 inputs, u = UMAX tanh(.), loss sum x^T Q x + R u^2 + x_N^T Q_f x_N
  vector_lambda          the matrix landing amplification of docs/theory/vector_lambda (operator norm of products of
                         M_k = J_x_hat - J_u_hat K_k, K_k = -du_k/dx_k (1 x 2), running products restarted at landing)"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.linalg import solve_discrete_are
from scipy.spatial import cKDTree

jax.config.update("jax_enable_x64", True)
H, WB, EPS = 16, 32, 1e-6
UMAX = 10.0
Q2, R2 = np.diag([5.0, 0.5]), 0.1
N_DATA, N_TEST = 20000, 4000


# ----------------------------------------------------------------------------------------------- plant
class Plant2:
    def __init__(self, A, B, box=(7.3, 5.6, 10.0)):
        self.A, self.B = np.asarray(A, float).reshape(2, 2), np.asarray(B, float).reshape(2, 1)
        self.box = tuple(float(b) for b in box)                      # data box |x1|, |x2|, |u|

    def f(self, x, u):
        """x [..., 2], u [...] -> [..., 2]"""
        return x @ self.A.T + np.asarray(u)[..., None] * self.B[:, 0]

    def Jx(self, x, u):
        return np.broadcast_to(self.A, np.shape(x)[:-1] + (2, 2))

    def Ju(self, x, u):
        return np.broadcast_to(self.B[:, 0], np.shape(x)[:-1] + (2,))

    def u_ss(self, r):
        """equilibrium (x2, u) with x1 = r: (A - I) [r, x2] + B u = 0 (least squares; exact for P1-type plants)."""
        M = np.column_stack([(self.A - np.eye(2))[:, 1], self.B[:, 0]])
        r = np.atleast_1d(np.asarray(r, float))
        sol = np.linalg.lstsq(M, -np.outer((self.A - np.eye(2))[:, 0], r), rcond=None)[0]
        return sol[0], sol[1]                                         # x2_ss(r), u_ss(r)


def make_data(plant, sigma, data_seed, n=N_DATA):
    rng = np.random.default_rng(data_seed)
    b1, b2, bu = plant.box
    x = np.column_stack([rng.uniform(-b1, b1, n), rng.uniform(-b2, b2, n)])
    u = rng.uniform(-bu, bu, n)
    y = plant.f(x, u) + sigma * rng.standard_normal((n, 2))
    return dict(x=x, u=u, y=y)


def make_test(plant):
    return make_data(plant, 0.0, 12345, N_TEST)


# ----------------------------------------------------------------------------------------------- surrogate
def _glorot(key, shape):
    lim = np.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim, dtype=jnp.float64)


def init_params(seed, n_in=3, n_out=2):
    k = jax.random.split(jax.random.PRNGKey(int(seed)), 4)
    kb = jax.random.split(k[2], 8)
    return dict(E=_glorot(k[0], (n_in, H)), b=jnp.zeros(H), gamma=jnp.ones(H), beta=jnp.zeros(H),
                Wd=_glorot(k[1], (H, n_out)), bd=jnp.zeros(n_out),
                br=dict(W1=_glorot(kb[0], (H, WB)), b1=jnp.zeros(WB), W2=_glorot(kb[1], (WB, H)), b2=jnp.zeros(H)))


def _norm(kind, h, s0):
    if kind == "none":
        return h
    if kind == "rms":
        return h / jnp.sqrt(jnp.mean(h ** 2) + EPS)
    r = h - jnp.mean(h)
    if kind == "layer":
        return r / jnp.sqrt(jnp.mean(r ** 2) + EPS)
    if kind == "static":
        return r / s0
    raise ValueError(kind)


def F_point(p, z, s0, norm):
    """z = (x1, x2, u) [3] -> F_hat [2]. Same block as level1_core.F_point with a 3-vector input and 2 outputs."""
    h = z @ p["E"] + p["b"]
    v = _norm(norm, h, s0)
    a = jax.nn.gelu((p["gamma"] * v + p["beta"]) @ p["br"]["W1"] + p["br"]["b1"], approximate=False)
    return h @ p["Wd"] + (a @ p["br"]["W2"] + p["br"]["b2"]) @ p["Wd"] + p["bd"]


def jac(p, z, s0, norm):
    """2 x 3 Jacobian [J_x_hat | J_u_hat] at z."""
    return jax.jacfwd(lambda zz: F_point(p, zz, s0, norm))(z)


def frozen_sigma(p, x, u):
    z = np.column_stack([x, u])
    h = z @ np.asarray(p["E"]) + np.asarray(p["b"])
    r = h - h.mean(-1, keepdims=True)
    return float(np.sqrt((r ** 2).mean(-1) + EPS).mean())


# ----------------------------------------------------------------------------------------------- labels
def _design3(t):
    """local quadratic in 3 inputs: 1, t1, t2, t3, and the 6 quadratic terms -> [m, k, 10]"""
    t1, t2, t3 = t[..., 0], t[..., 1], t[..., 2]
    one = np.ones_like(t1)
    return np.stack([one, t1, t2, t3, t1 * t1, t2 * t2, t3 * t3, t1 * t2, t1 * t3, t2 * t3], -1)


class LabelEstimator3:
    """Local-quadratic weighted least squares around each query z0 = (x1, x2, u), Epanechnikov weights on the k nearest
    neighbours in data-std units (the 3-input version of level1_core.LabelEstimator). Returns J [n, 2, 3]."""

    def __init__(self, z, y, k=1500, ridge=1e-8):
        self.z, self.y = np.asarray(z, float), np.asarray(y, float)
        self.k, self.ridge = int(k), float(ridge)
        self.s = self.z.std(0)
        self.tree = cKDTree(self.z / self.s)

    def __call__(self, zq):
        zq = np.atleast_2d(np.asarray(zq, float))
        out = np.empty((len(zq), 2, 3))
        chunk = max(16, 200_000 // self.k)
        P = np.eye(10); P[0, 0] = 0.0
        for s in range(0, len(zq), chunk):
            q = zq[s:s + chunk]
            d, idx = self.tree.query(q / self.s, k=self.k)
            R = np.maximum(d[:, -1:], 1e-12)
            w = np.maximum(1 - (d / R) ** 2, 0.0)
            t = (self.z[idx] - q[:, None, :]) / (self.s * R[..., None])
            Phi = _design3(t)
            PhiW = Phi * w[..., None]
            G = np.einsum("mki,mkj->mij", PhiW, Phi) + self.ridge * P
            beta = np.linalg.solve(G, np.einsum("mki,mkj->mij", PhiW, self.y[idx]))      # [m, 10, 2]
            out[s:s + chunk] = np.transpose(beta[:, 1:4, :], (0, 2, 1)) / (self.s[None, None, :] * R[:, :, None])
        return out


def get_labels(d, seed, k, cache_dir):
    fn = os.path.join(cache_dir, f"labels2_seed{seed}_k{k}.npz")
    fp = np.asarray(d["y"][:8], float)
    if os.path.exists(fn):
        z = np.load(fn)
        if not np.array_equal(z["data_fp"], fp):
            raise ValueError(f"{fn} belongs to other data")
        return z["J"]
    J = LabelEstimator3(np.column_stack([d["x"], d["u"]]), d["y"], k)(np.column_stack([d["x"], d["u"]]))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(fn, J=J, data_fp=fp)
    return J


# ----------------------------------------------------------------------------------------------- training
CONFIGS = {  # same six families as the scalar study
    "MSE (3000)": dict(norm="layer", labels=None, phases=[("mse", 0, 0, 3000)]),
    "MSE (12000)": dict(norm="layer", labels=None, phases=[("mse", 0, 0, 12000)]),
    "J_x + J_u (3000)": dict(norm="layer", labels="est", phases=[("sob", 1.0, 1.0, 3000)]),
    "J_x + J_u (two-stage)": dict(norm="layer", labels="est", phases=[("mse", 0, 0, 3000), ("sob", 1.0, 1.0, 1000)]),
    "no norm, MSE": dict(norm="none", labels=None, phases=[("mse", 0, 0, 3000)]),
    "static norm, MSE": dict(norm="static", labels=None, phases=[("mse", 0, 0, 3000)]),
}


def configs(scale=1.0):
    """CONFIGS with every phase length multiplied by `scale` (QUICK runs)."""
    out = {}
    for n, c in CONFIGS.items():
        out[n] = dict(c, phases=[(k, wx, wu, max(1, int(round(T * scale)))) for k, wx, wu, T in c["phases"]])
    return out


def train(cfg, seed, plant, sigma=0.01, lr=3e-3, M=2048, lam=1.0, label_dir=None, label_k=1500, verbose=False):
    """One seed: data key 1000 + s, init key s, slope-loss draws key 10000 + s (as the scalar study). Returns p, s0, data."""
    d = make_data(plant, sigma, 1000 + seed)
    p = init_params(seed)
    norm = cfg["norm"]
    s0 = frozen_sigma(p, d["x"], d["u"]) if norm == "static" else 1.0
    Z = jnp.asarray(np.column_stack([d["x"], d["u"]])); Y = jnp.asarray(d["y"])
    sz = np.column_stack([d["x"], d["u"]]).std(0)
    if cfg["labels"] == "est":
        Jlab = jnp.asarray(get_labels(d, seed, label_k, label_dir or "labels2"))
    else:
        Jlab = jnp.zeros((len(d["u"]), 2, 3))
    Fb = jax.vmap(lambda p, z: F_point(p, z, s0, norm), in_axes=(None, 0))
    Jb = jax.vmap(lambda p, z: jac(p, z, s0, norm), in_axes=(None, 0))
    key = jax.random.PRNGKey(10_000 + seed)

    def loss(p, kind, wx, wu, k):
        mse = jnp.mean((Fb(p, Z) - Y) ** 2)
        if kind == "mse":
            return mse
        i = jax.random.choice(k, Z.shape[0], (M,), replace=True)
        J = Jb(p, Z[i]); E = J - Jlab[i]                               # [M, 2, 3]
        w = jnp.array([wx * sz[0] ** 2, wx * sz[1] ** 2, wu * sz[2] ** 2])
        return mse + lam * jnp.mean(jnp.sum(E ** 2 * w[None, None, :], axis=(1, 2)))

    for kind, wx, wu, T in cfg["phases"]:
        opt = optax.adam(lr); o = opt.init(p)                          # fresh Adam at each phase switch

        @jax.jit
        def step(p, o, k):
            l, g = jax.value_and_grad(loss)(p, kind, wx, wu, k)
            u, o = opt.update(g, o, p)
            return optax.apply_updates(p, u), o, l
        for t in range(T):
            key, k = jax.random.split(key)
            p, o, l = step(p, o, k)
        if verbose:
            print(f"    seed {seed} {kind} {T} steps: loss {float(l):.3e}", flush=True)
    return p, s0, d


# ----------------------------------------------------------------------------------------------- Level 1 + kink metrics
def grid3(plant, n=21):
    b1, b2, bu = plant.box
    g = np.meshgrid(np.linspace(-b1, b1, n), np.linspace(-b2, b2, n), np.linspace(-bu, bu, n), indexing="ij")
    return np.stack([a.ravel() for a in g], -1)


def kink3(p, norm="layer"):
    """Theorem 1 with n_z = 3: z* = argmin ||P(E^T z + b)||, D, half-widths D / ||q_i||."""
    E, b = np.asarray(p["E"], float), np.asarray(p["b"], float)          # E: 3 x H
    PM = np.eye(H) if norm == "rms" else np.eye(H) - np.ones((H, H)) / H
    A = PM @ E.T; c0 = PM @ b
    z = -np.linalg.lstsq(A, c0, rcond=None)[0]
    r = A @ z + c0
    D = float(np.sqrt(r @ r + H * EPS))
    return dict(z_x1=z[0], z_x2=z[1], z_u=z[2], D=D, r_x1=D / np.linalg.norm(A[:, 0]), r_x2=D / np.linalg.norm(A[:, 1]),
                r_u=D / np.linalg.norm(A[:, 2]))


def level1_metrics(p, s0, norm, plant, n=21):
    te = make_test(plant)
    Zt = jnp.asarray(np.column_stack([te["x"], te["u"]]))
    F = np.asarray(jax.vmap(lambda z: F_point(p, z, s0, norm))(Zt))
    rel_mse = float(np.mean((F - te["y"]) ** 2) / np.mean(np.var(te["y"], 0)))
    G = grid3(plant, n)
    J = np.asarray(jax.vmap(lambda z: jac(p, z, s0, norm))(jnp.asarray(G)))
    Bn = np.linalg.norm(plant.B)
    eu = np.linalg.norm(J[:, :, 2] - plant.B[:, 0][None], axis=1) / Bn
    ex = np.linalg.norm(J[:, :, :2] - plant.A[None], ord=2, axis=(1, 2))
    flip = np.mean(J[:, :, 2] @ plant.B[:, 0] <= 0)
    out = dict(rel_mse=rel_mse, Ju_sup_rel=float(eu.max()), Ju_rms_rel=float(np.sqrt((eu ** 2).mean())),
               Jx_sup=float(ex.max()), flip_area=float(flip))
    if norm in ("layer", "rms"):
        out.update(kink3(p, norm))
    return out


# ----------------------------------------------------------------------------------------------- LQR
def lqr_K(A, B, Q=Q2, R=R2):
    A, B = np.asarray(A, float).reshape(2, 2), np.asarray(B, float).reshape(2, 1)
    P = solve_discrete_are(A, B, Q, np.array([[R]]))
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)               # 1 x 2


def surrogate_equilibrium(p, s0, norm, r, x2_0, u_0, iters=50):
    """Solve F_hat((r, x2), u) = (r, x2) for (x2, u) by Newton from the true equilibrium."""
    v = np.array([x2_0, u_0], float)
    for _ in range(iters):
        z = jnp.array([r, v[0], v[1]])
        res = np.asarray(F_point(p, z, s0, norm)) - np.array([r, v[0]])
        J = np.asarray(jac(p, z, s0, norm))
        Jv = np.column_stack([J[:, 1] - np.array([0.0, 1.0]), J[:, 2]])
        step = np.linalg.lstsq(Jv, -res, rcond=None)[0]
        v = v + step
        if np.abs(step).max() < 1e-12:
            break
    return v


def lqr_study(p, s0, norm, plant, setpoints, starts, N=200, sched=True):
    """Fixed and (optionally) rescheduled LQR on the TRUE plant. Cost J = sum (x - x_ss)^T Q (x - x_ss) + R (u - u_ss)^2
    over N steps; gap = J / J_oracle - 1 with the true-plant LQR around the true equilibrium; diverged = |x| > 4 x box."""
    Ktrue = lqr_K(plant.A, plant.B)
    rows = []
    for r in setpoints:
        x2s, uss = (float(a[0]) for a in plant.u_ss(r))
        xs = np.array([r, x2s])
        x2h, uh = surrogate_equilibrium(p, s0, norm, r, x2s, uss)
        J = np.asarray(jac(p, jnp.array([r, x2h, uh]), s0, norm))
        try:
            Kf = lqr_K(J[:, :2], J[:, 2])
        except Exception:
            Kf = np.full((1, 2), np.nan)
        for x0 in starts:
            for ctl in (("LQR fixed", "LQR sched") if sched else ("LQR fixed",)):
                x = np.array(x0, float); u_prev = uh; cost = 0.0; div = False
                xo = np.array(x0, float); cost_o = 0.0
                for k in range(N):
                    if ctl == "LQR fixed":
                        u = uh - float((Kf @ (x - np.array([r, x2h])))[0])
                    else:
                        Jk = np.asarray(jac(p, jnp.array([x[0], x[1], u_prev]), s0, norm))
                        try:
                            Kk = lqr_K(Jk[:, :2], Jk[:, 2])
                        except Exception:
                            Kk = Kf
                        u = uh - float((Kk @ (x - np.array([r, x2h])))[0])
                    u = float(np.clip(u, -UMAX, UMAX))
                    uo = float(np.clip(uss - (Ktrue @ (xo - xs))[0], -UMAX, UMAX))
                    cost += (x - xs) @ Q2 @ (x - xs) + R2 * (u - uss) ** 2
                    cost_o += (xo - xs) @ Q2 @ (xo - xs) + R2 * (uo - uss) ** 2
                    x = plant.A @ x + plant.B[:, 0] * u; xo = plant.A @ xo + plant.B[:, 0] * uo; u_prev = u
                    if not np.all(np.isfinite(x)) or np.abs(x).max() > 4 * max(plant.box[:2]):
                        div = True
                        break
                rows.append(dict(controller=ctl, r=r, x0_1=x0[0], x0_2=x0[1], cost=cost, ref_cost=cost_o,
                                 gap=cost / max(cost_o, 1e-300) - 1, diverged=div, K1=Kf[0, 0], K2=Kf[0, 1],
                                 Bh1=J[0, 2], Bh2=J[1, 2]))
    return rows


# ----------------------------------------------------------------------------------------------- controllers and DPC
def init_ctrl(key, arch="gru", d_x=2, hidden=64, width=64, depth=2):
    k = jax.random.split(key, 8)
    if arch == "gru":
        g = lambda kk, s: _glorot(kk, s)
        return dict(Wz=g(k[0], (d_x + hidden, hidden)), Wr=g(k[1], (d_x + hidden, hidden)), Wh=g(k[2], (d_x + hidden, hidden)),
                    bz=jnp.zeros(hidden), br=jnp.zeros(hidden), bh=jnp.zeros(hidden), Wo=g(k[3], (hidden, 1)) * 0.1, bo=jnp.zeros(1))
    sizes = [d_x] + [width] * depth + [1]
    return {f"L{i}": dict(W=_glorot(k[i], (sizes[i], sizes[i + 1])), b=jnp.zeros(sizes[i + 1])) for i in range(len(sizes) - 1)}


def ctrl_step(c, h, x):
    """x [d_x] -> (h', u). GRU if 'Wz' in c, else MLP (memoryless, h unused)."""
    if "Wz" in c:
        xh = jnp.concatenate([x, h])
        z = jax.nn.sigmoid(xh @ c["Wz"] + c["bz"]); r = jax.nn.sigmoid(xh @ c["Wr"] + c["br"])
        n = jnp.tanh(jnp.concatenate([x, r * h]) @ c["Wh"] + c["bh"])
        h = (1 - z) * h + z * n
        return h, UMAX * jnp.tanh(h @ c["Wo"] + c["bo"])[0]
    a = x
    n = len(c)
    for i in range(n):
        a = a @ c[f"L{i}"]["W"] + c[f"L{i}"]["b"]
        if i < n - 1:
            a = jnp.tanh(a)
    return h, UMAX * jnp.tanh(a)[0]


def rollout(c, dyn, x0, N, hidden=64):
    h0 = jnp.zeros(hidden if "Wz" in c else 1)

    def st(carry, _):
        h, x = carry
        h, u = ctrl_step(c, h, x)
        xn = dyn(x, u)
        return (h, xn), (x, u)
    (h, xN), (xs, us) = jax.lax.scan(st, (h0, x0), None, length=N)
    return jnp.concatenate([xs, xN[None]], 0), us


def dpc_loss(c, dyn, X0, N, Q=Q2, R=R2, Qf=None):
    Qf = 10 * Q if Qf is None else Qf
    Qj, Qfj = jnp.asarray(Q), jnp.asarray(Qf)

    def one(x0):
        xs, us = rollout(c, dyn, x0, N)
        return jnp.sum(jnp.einsum("ki,ij,kj->k", xs[:-1], Qj, xs[:-1]) + R * us ** 2) + xs[-1] @ Qfj @ xs[-1]
    return jnp.mean(jax.vmap(one)(X0)) / N


def true_dyn(plant):
    A, B = jnp.asarray(plant.A), jnp.asarray(plant.B[:, 0])
    return lambda x, u: A @ x + B * u


def sur_dyn(p, s0, norm):
    return lambda x, u: F_point(p, jnp.concatenate([x, jnp.atleast_1d(u)]), s0, norm)


def train_dpc(dyn, key_ctrl, X0, curriculum, arch="gru", lr=1e-3, wd=1e-4):
    c = init_ctrl(key_ctrl, arch)
    opt = optax.adamw(lr, weight_decay=wd); o = opt.init(c)
    hist = []
    for N, E in curriculum:
        @jax.jit
        def step(c, o):
            l, g = jax.value_and_grad(dpc_loss)(c, dyn, X0, N)
            u, o = opt.update(g, o, c)
            return optax.apply_updates(c, u), o, l
        for e in range(E):
            c, o, l = step(c, o)
        hist.append((N, float(l)))
    return c, hist


def saturated_lqr_cost(plant, X0, N=200):
    K = lqr_K(plant.A, plant.B)
    x = np.asarray(X0, float).copy(); J = np.zeros(len(x))
    for k in range(N):
        u = np.clip(-(x @ K.T)[:, 0], -UMAX, UMAX)
        J += np.einsum("ni,ij,nj->n", x, Q2, x) + R2 * u ** 2
        x = x @ plant.A.T + u[:, None] * plant.B[:, 0][None]
    J += np.einsum("ni,ij,nj->n", x, 10 * Q2, x)
    return float(J.mean() / N)


# ----------------------------------------------------------------------------------------------- vector Lambda
def vector_lambda(c, p, s0, norm, X0, N=200, land=0.2, hidden=64):
    """docs/theory/vector_lambda: along the closed loop through the surrogate, M_k = J_x_hat(x_k, u_k) - J_u_hat(x_k, u_k) K_k
    with K_k = -du_k/dx_k (1 x 2, at the current hidden state: direct gain). Landing step k_land = first k with
    ||x_k||_inf <= land. Lambda(start) = max over k_land <= k1 < k2 <= N of log ||M_{k2-1} ... M_{k1}||_2; Lambda* = max over
    starts. Returns (Lambda per start, k_land per start). O(N^2) running products per start."""
    def per_start(x0):
        h0 = jnp.zeros(hidden if "Wz" in c else 1)

        def st(carry, _):
            h, x = carry
            hn, u = ctrl_step(c, h, x)
            Kk = -jax.grad(lambda xx: ctrl_step(c, h, xx)[1])(x)
            J = jac(p, jnp.concatenate([x, jnp.atleast_1d(u)]), s0, norm)
            M = J[:, :2] - jnp.outer(J[:, 2], Kk)
            xn = F_point(p, jnp.concatenate([x, jnp.atleast_1d(u)]), s0, norm)
            return (hn, xn), (x, M)
        _, (xs, Ms) = jax.lax.scan(st, (h0, x0), None, length=N)
        return xs, Ms
    xs, Ms = jax.jit(jax.vmap(per_start))(jnp.asarray(X0))
    return lambda_from(np.asarray(xs), np.asarray(Ms), land)


def lambda_from(xs, Ms, land=0.2):
    """The statistic from states xs [S, N, n] and multipliers Ms [S, N, n, n] (see vector_lambda)."""
    S, N = Ms.shape[:2]; n = Ms.shape[-1]
    near = np.abs(xs).max(2) <= land
    kl = np.where(near.any(1), np.argmax(near, 1), N)
    # Pm[s, k1] = normalised running product M_{k2} ... M_{k1} (identity until k2 reaches k1); L[s, k1] = its log norm
    Pm = np.broadcast_to(np.eye(n), (S, N, n, n)).copy(); L = np.zeros((S, N)); best = np.zeros(S)
    k1s = np.arange(N)
    for k2 in range(N):
        act = (k1s[None, :] <= k2) & (k1s[None, :] >= kl[:, None])                   # stretches [k1, k2] after landing
        Pm = np.where(act[..., None, None], np.einsum("sij,skjl->skil", Ms[:, k2], Pm), Pm)
        nrm = np.linalg.norm(Pm, ord=2, axis=(2, 3))
        ok = act & np.isfinite(nrm) & (nrm > 0)
        L = np.where(ok, L + np.log(np.where(ok, nrm, 1.0)), L)
        Pm = np.where(ok[..., None, None], Pm / np.where(ok, nrm, 1.0)[..., None, None], Pm)
        best = np.maximum(best, np.where(act, L, 0.0).max(1))
    return best, kl
