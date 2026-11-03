"""level1_core.py - Level 1 of the LayerNorm-kink method.

Level 1 = MSE + lambda x (Jacobian loss with labels estimated from the same transitions), the
slope-check points drawn uniformly from the training inputs. No lens, no trust weights, no other
tricks beyond the schedules being compared.

Everything is identical to smc_core.py (same plant, data seeds, model, initialisation, optimiser,
label estimator, k selection, sampling of slope-check points), so the numbers are comparable with
the E1 runs. The only differences: no label-trust weights (Level 1 has none; on box data they are
all 1 anyway), the Jacobian loss can use J_x only, J_u only or both, and the closed-form kink-line
evaluation points are clipped into the box (fixed count, so nothing recompiles).
"""
import os
import csv
import time
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.spatial import cKDTree

tmap = jax.tree_util.tree_map

# ------------------------------------------------------------------------------------------------ #
# 0. Problem setup (Level 0, inherited from the TMLR theory paper)                                 #
# ------------------------------------------------------------------------------------------------ #
XMAX, UMAX = 4.0, 10.0          # data box: x ~ U[-4, 4], u ~ U[-10, 10]
N_DATA, N_TEST = 20000, 4000    # training transitions, noiseless test points
H, WB, EPS = 16, 32, 1e-6       # encoder width, MLP width, LayerNorm epsilon
LR = 3e-3                       # Adam learning rate, full batch
LQR_Q, LQR_R = 5.0, 0.1         # LQR weights of the closed-loop screen
K_GRID = (40, 80, 160, 320, 640, 1280)


class Plant:
    """x+ = A x + B u + eps_nl (a_sin sin x + c_xu x u). eps_nl = 0 is the linear plant."""

    def __init__(self, A=1.03, B=0.01, eps_nl=0.0, a_sin=0.05, c_xu=0.002):
        self.A, self.B, self.eps = float(A), float(B), float(eps_nl)
        self.a_sin, self.c_xu = float(a_sin), float(c_xu)

    def f(self, x, u):
        return self.A * x + self.B * u + self.eps * (self.a_sin * np.sin(x) + self.c_xu * x * u)

    def Jx(self, x, u):
        return self.A + self.eps * (self.a_sin * np.cos(x) + self.c_xu * u)

    def Ju(self, x, u):
        return self.B + self.eps * self.c_xu * x + 0.0 * u

    def u_ss(self, r):
        r = np.asarray(r, float)
        den = self.B + self.eps * self.c_xu * r
        with np.errstate(divide="ignore", invalid="ignore"):
            return (r - self.A * r - self.eps * self.a_sin * np.sin(r)) / den

    def r_max(self):
        """Setpoints the actuator can hold with a 10 % margin."""
        return min(0.9 * self.B * UMAX / max(self.A - 1.0, 1e-6), XMAX)


def make_data(plant, sigma, data_seed, n=N_DATA):
    rng = np.random.default_rng(data_seed)
    x = rng.uniform(-XMAX, XMAX, n)
    u = rng.uniform(-UMAX, UMAX, n)
    y = plant.f(x, u) + sigma * rng.standard_normal(n)
    return dict(x=x, u=u, y=y)


def make_test(plant):
    return make_data(plant, 0.0, 12345, N_TEST)


# ------------------------------------------------------------------------------------------------ #
# 1. Jacobian labels from the transitions (local quadratic, k nearest in std-scaled distance)      #
# ------------------------------------------------------------------------------------------------ #
def kernel_weights(t, kernel="epan"):
    """t = d/R in [0,1]. 'epan': 1 - t^2 (current code). 'gauss': exp(-2 t^2) (the note's rule)."""
    t = np.clip(t, 0.0, 1.0)
    if kernel == "epan":
        w = 1.0 - t ** 2
    elif kernel == "gauss":
        w = np.exp(-2.0 * t ** 2)
    else:
        raise ValueError(kernel)
    return np.maximum(w, 1e-12)


def _design(tx, tu):
    one = np.ones_like(tx)
    return np.stack([one, tx, tu, 0.5 * tx ** 2, tx * tu, 0.5 * tu ** 2], -1)


class LabelEstimator:
    """Local-quadratic weighted least squares around each query (x0, u0).
    d_i = sqrt(((x_i-x0)/s_x)^2 + ((u_i-u0)/s_u)^2), k nearest, R = d_(k), w_i = kernel(d_i/R),
    t = offsets / (s R), J_x = beta_1/(s_x R), J_u = beta_2/(s_u R). s_x, s_u = data std.
    Also returns v = Var(label)/sigma^2 from the sandwich formula."""

    def __init__(self, x, u, y, k, kernel="epan", ridge=1e-8):
        self.x, self.u, self.y = (np.asarray(a, float) for a in (x, u, y))
        self.k, self.kernel, self.ridge = int(k), kernel, float(ridge)
        self.sx, self.su = float(self.x.std()), float(self.u.std())
        self.tree = cKDTree(np.column_stack([self.x / self.sx, self.u / self.su]))

    def __call__(self, xq, uq, want_var=True):
        xq, uq = np.atleast_1d(np.asarray(xq, float)), np.atleast_1d(np.asarray(uq, float))
        chunk = max(32, 400_000 // self.k)
        n = len(xq)
        out = {k: np.empty(n) for k in ("Jx", "Ju", "vx", "vu", "R")}
        P = np.eye(6)
        P[0, 0] = 0.0
        for s in range(0, n, chunk):
            xs, us = xq[s:s + chunk], uq[s:s + chunk]
            d, idx = self.tree.query(np.column_stack([xs / self.sx, us / self.su]), k=self.k)
            R = np.maximum(d[:, -1:], 1e-12)
            w = kernel_weights(d / R, self.kernel)
            tx = (self.x[idx] - xs[:, None]) / (self.sx * R)
            tu = (self.u[idx] - us[:, None]) / (self.su * R)
            Phi = _design(tx, tu)
            PhiW = Phi * w[..., None]
            G = np.einsum("mki,mkj->mij", PhiW, Phi) + self.ridge * P
            Ginv = np.linalg.inv(G)
            beta = np.einsum("mij,mj->mi", Ginv, np.einsum("mki,mk->mi", PhiW, self.y[idx]))
            Rs = R[:, 0]
            out["Jx"][s:s + chunk] = beta[:, 1] / (self.sx * Rs)
            out["Ju"][s:s + chunk] = beta[:, 2] / (self.su * Rs)
            out["R"][s:s + chunk] = Rs
            if want_var:
                Cov = Ginv @ np.einsum("mki,mkj->mij", PhiW * w[..., None], Phi) @ Ginv
                out["vx"][s:s + chunk] = Cov[:, 1, 1] / (self.sx * Rs) ** 2
                out["vu"][s:s + chunk] = Cov[:, 2, 2] / (self.su * Rs) ** 2
        return out

    def noise_var(self, n_probe=2000, seed=0):
        """Pooled residual variance of unweighted local quadratic fits: an estimate of sigma^2."""
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(self.x), size=min(n_probe, len(self.x)), replace=False)
        d, idx = self.tree.query(np.column_stack([self.x[sel] / self.sx, self.u[sel] / self.su]), k=self.k)
        R = np.maximum(d[:, -1:], 1e-12)
        Phi = _design((self.x[idx] - self.x[sel, None]) / (self.sx * R), (self.u[idx] - self.u[sel, None]) / (self.su * R))
        Yn = self.y[idx]
        G = np.einsum("mki,mkj->mij", Phi, Phi) + 1e-12 * np.eye(6)
        beta = np.linalg.solve(G, np.einsum("mki,mk->mi", Phi, Yn)[..., None])[..., 0]
        res = Yn - np.einsum("mki,mi->mk", Phi, beta)
        return float((res ** 2).sum() / (res.size - 6 * len(sel)))


def select_k(x, u, y, ks=K_GRID, kernel="epan", n_query=2000, seed=0):
    """k minimising the estimated label MSE (bias^2 + variance) in loss units, with no true Jacobian:
    variance = sigma_hat^2 x sandwich factor; bias^2(k) ~ mean[(J(2k) - J(k))^2] - sigma_hat^2 (v_k - v_2k)
    (doubling trick: the slope bias grows ~ linearly in k in 2-D); score = Var(x) MSE_x + Var(u) MSE_u.
    Returns (k, table, sigma_hat^2)."""
    x, u, y = (np.asarray(a, float) for a in (x, u, y))
    rng = np.random.default_rng(seed)
    q = rng.choice(len(x), min(n_query, len(x)), replace=False)
    xq, uq = x[q], u[q]
    ks = sorted(set(int(k) for k in ks))
    allk = [k for k in sorted(set(ks + [2 * k for k in ks])) if k < len(x)]
    labs = {k: LabelEstimator(x, u, y, k, kernel)(xq, uq) for k in allk}
    s2 = LabelEstimator(x, u, y, ks[0], kernel).noise_var()
    rows = []
    for k in ks:
        if 2 * k not in labs:
            continue
        a, b = labs[k], labs[2 * k]
        row = dict(k=k)
        for J, v in (("Jx", "vx"), ("Ju", "vu")):
            var_k = s2 * np.mean(a[v])
            bias2 = max(np.mean((b[J] - a[J]) ** 2) - s2 * (np.mean(a[v]) - np.mean(b[v])), 0.0)
            row[f"bias2_{J}"], row[f"var_{J}"], row[f"mse_{J}"] = bias2, var_k, bias2 + var_k
        row["score"] = np.var(x) * row["mse_Jx"] + np.var(u) * row["mse_Ju"]
        rows.append(row)
    best = min(rows, key=lambda r: r["score"])["k"]
    return best, rows, s2


def get_labels(d, seed, sigma, kernel, cache_dir):
    """Labels at every training input, computed once per dataset and cached to disk."""
    fn = os.path.join(cache_dir, f"labels_seed{seed}_sigma{sigma:g}_{kernel}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return {k: z[k] for k in z.files}
    t0 = time.time()
    k, table, s2 = select_k(d["x"], d["u"], d["y"], kernel=kernel)
    lab = LabelEstimator(d["x"], d["u"], d["y"], k, kernel)(d["x"], d["u"])
    out = dict(Jx=lab["Jx"], Ju=lab["Ju"], vx=lab["vx"], vu=lab["vu"], k=np.array(k), sigma_hat=np.array(np.sqrt(s2)),
               ktab_k=np.array([r["k"] for r in table]), ktab_score=np.array([r["score"] for r in table]),
               ktab_bias2_Ju=np.array([r["bias2_Ju"] for r in table]), ktab_var_Ju=np.array([r["var_Ju"] for r in table]),
               ktab_bias2_Jx=np.array([r["bias2_Jx"] for r in table]), ktab_var_Jx=np.array([r["var_Jx"] for r in table]),
               time_s=np.array(time.time() - t0))
    np.savez_compressed(fn, **out)
    return out


# ------------------------------------------------------------------------------------------------ #
# 2. The surrogate (identical to kink_core / smc_core)                                             #
# ------------------------------------------------------------------------------------------------ #
def _glorot(key, shape):
    lim = np.sqrt(6.0 / (shape[0] + shape[1]))
    return jax.random.uniform(key, shape, minval=-lim, maxval=lim, dtype=jnp.float64)


def init_params(seed):
    k = jax.random.split(jax.random.PRNGKey(int(seed)), 4)
    kb = jax.random.split(k[2], 8)
    return dict(E=_glorot(k[0], (2, H)), b=jnp.zeros(H), gamma=jnp.ones(H), beta=jnp.zeros(H),
                Wd=_glorot(k[1], (H, 1)), bd=jnp.zeros(1),
                br=dict(W1=_glorot(kb[0], (H, WB)), b1=jnp.zeros(WB), W2=_glorot(kb[1], (WB, H)), b2=jnp.zeros(H)))


def stack_trees(trees):
    return tmap(lambda *a: jnp.stack(a), *trees)


def unstack_tree(tree, i):
    return tmap(lambda a: a[i], tree)


def _norm(kind, h, s0):
    if kind == "none":
        return h
    r = h - jnp.mean(h, -1, keepdims=True)
    if kind == "layer":
        return r / jnp.sqrt(jnp.mean(r ** 2, -1, keepdims=True) + EPS)
    if kind == "static":
        return r / s0
    raise ValueError(kind)


def F_point(p, x, u, s0, norm):
    h = x * p["E"][0] + u * p["E"][1] + p["b"]
    v = _norm(norm, h, s0)
    a = jax.nn.gelu((p["gamma"] * v + p["beta"]) @ p["br"]["W1"] + p["br"]["b1"], approximate=False)
    return h @ p["Wd"][:, 0] + (a @ p["br"]["W2"] + p["br"]["b2"]) @ p["Wd"][:, 0] + p["bd"][0]


def frozen_sigma(p, x, u):
    """Scale of the 'static' normalisation: the LayerNorm scale at initialisation, averaged over the data."""
    h = np.asarray(x)[:, None] * np.asarray(p["E"][0]) + np.asarray(u)[:, None] * np.asarray(p["E"][1]) + np.asarray(p["b"])
    r = h - h.mean(-1, keepdims=True)
    return float(np.sqrt((r ** 2).mean(-1) + EPS).mean())


def _Fb(norm):
    return jax.vmap(lambda p, x, u, s0: F_point(p, x, u, s0, norm), in_axes=(None, 0, 0, None))


def _derivs(Fb, p, x, u, s0):
    jx = jax.jvp(lambda xx: Fb(p, xx, u, s0), (x,), (jnp.ones_like(x),))[1]
    ju = jax.jvp(lambda uu: Fb(p, x, uu, s0), (u,), (jnp.ones_like(u),))[1]
    return jx, ju


# ------------------------------------------------------------------------------------------------ #
# 3. Loss and trainer (vmapped over seeds)                                                         #
# ------------------------------------------------------------------------------------------------ #
def make_loss(norm, kind, M):
    """kind 'mse':  mean (F - y)^2
       kind 'sob':  mean (F - y)^2 + lam * mean_j [ wx s_x^2 (dF/dx - Jx_lab)^2 + wu s_u^2 (dF/du - Ju_lab)^2 ]
    over M slope-check points drawn uniformly (with replacement) from the training inputs each step.
    wx, wu in {0, 1} select J_x only / J_u only / both. s_x, s_u = data std (unit-invariant loss)."""
    Fb = _Fb(norm)

    def loss(p, key, D, wx, wu, lam):
        mse = jnp.mean((Fb(p, D["x"], D["u"], D["s0"]) - D["y"]) ** 2)
        if kind == "mse":
            return mse
        n = D["x"].shape[0]
        idx = jnp.clip(jnp.searchsorted(D["cdf"], jax.random.uniform(key, (M,), dtype=jnp.float64)), 0, n - 1)
        jx, ju = _derivs(Fb, p, D["x"][idx], D["u"][idx], D["s0"])
        reg = jnp.mean(wx * D["sx"] ** 2 * (jx - D["Jx_lab"][idx]) ** 2 + wu * D["su"] ** 2 * (ju - D["Ju_lab"][idx]) ** 2)
        return mse + lam * reg

    return loss


_TRAINERS = {}


def get_trainer(norm, kind, M, chunk):
    key = (norm, kind, M, chunk)
    if key in _TRAINERS:
        return _TRAINERS[key]
    loss = make_loss(norm, kind, M)
    b1, b2, ae = 0.9, 0.999, 1e-8

    def one_chunk(p, m, v, t0, key, D, wx, wu, lam):
        def st(c, i):
            p, m, v, key = c
            key, sub = jax.random.split(key)
            t = t0 + i + 1.0
            g = jax.grad(loss)(p, sub, D, wx, wu, lam)
            m = tmap(lambda a, b: b1 * a + (1 - b1) * b, m, g)
            v = tmap(lambda a, b: b2 * a + (1 - b2) * b * b, v, g)
            p = tmap(lambda a, mm, vv: a - LR * (mm / (1 - b1 ** t)) / (jnp.sqrt(vv / (1 - b2 ** t)) + ae), p, m, v)
            return (p, m, v, key), 0.0
        (p, m, v, key), _ = jax.lax.scan(st, (p, m, v, key), jnp.arange(chunk, dtype=jnp.float64))
        return p, m, v, key

    run = jax.jit(jax.vmap(one_chunk, in_axes=(0, 0, 0, None, 0, 0, None, None, None)))
    _TRAINERS[key] = run
    return run


_EVALS = {}


def get_evaluator(norm):
    """Batched over seeds: (P, X[S,n], U[S,n], s0[S]) -> F, dF/dx, dF/du, each [S, n]."""
    if norm in _EVALS:
        return _EVALS[norm]
    Fb = _Fb(norm)

    def one(p, x, u, s0):
        jx, ju = _derivs(Fb, p, x, u, s0)
        return Fb(p, x, u, s0), jx, ju

    ev = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0)))
    _EVALS[norm] = ev
    return ev


# ------------------------------------------------------------------------------------------------ #
# 4. Kink geometry (closed form, from the encoder weights) and evaluation                          #
# ------------------------------------------------------------------------------------------------ #
_PM = np.eye(H) - np.ones((H, H)) / H


def kink_center(p):
    """z_min = argmin ||P(E^T z + b)|| and the kink half-widths r*_x, r*_u (Theorem U2 of the theory paper)."""
    E, b = np.asarray(p["E"], float), np.asarray(p["b"], float)
    A = _PM @ E.T
    c0 = _PM @ b
    z = -np.linalg.lstsq(A, c0, rcond=None)[0]
    r = A @ z + c0
    D = np.sqrt(r @ r + H * EPS)
    return z, D / np.linalg.norm(A[:, 0]), D / np.linalg.norm(A[:, 1])


def kink_line_points(p, n_lines=41, n_pts=61, tmax=8.0):
    """Points on the closed-form kink lines (x-sweeps at fixed u and u-sweeps at fixed x), each dense around
    its own closest-approach point. Clipped into the box so the count is fixed (2 n_lines n_pts)."""
    E, b = np.asarray(p["E"], float), np.asarray(p["b"], float)
    th = np.tan(np.linspace(-np.arctan(tmax), np.arctan(tmax), n_pts))
    out = []
    for sweep, fixed, lim in ((0, np.linspace(-UMAX, UMAX, n_lines), XMAX), (1, np.linspace(-XMAX, XMAX, n_lines), UMAX)):
        q = _PM @ E[sweep]
        nq2 = float(q @ q)
        C = fixed[:, None] * (_PM @ E[1 - sweep])[None, :] + (_PM @ b)[None, :]
        s_star = -(C @ q) / nq2
        cp = C + s_star[:, None] * q[None, :]
        rstar = np.sqrt((cp ** 2).sum(1) + H * EPS) / np.sqrt(nq2)
        s = np.clip(s_star[:, None] + rstar[:, None] * th[None, :], -lim, lim)
        f0 = np.repeat(fixed[:, None], n_pts, 1)
        xy = np.stack([s, f0], -1) if sweep == 0 else np.stack([f0, s], -1)
        out.append(xy.reshape(-1, 2))
    return np.concatenate(out, 0)


def grid(n):
    xg, ug = np.meshgrid(np.linspace(-XMAX, XMAX, n), np.linspace(-UMAX, UMAX, n))
    return xg.ravel(), ug.ravel()


def dare_K(a, b, Q=LQR_Q, R=LQR_R):
    a, b = np.asarray(a, float), np.asarray(b, float)
    b2 = b * b
    coef = R - Q * b2 - a * a * R
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(b2 > 1e-14, (-coef + np.sqrt(coef ** 2 + 4 * b2 * Q * R)) / (2 * b2), np.nan)
        return np.where(b2 > 1e-14, a * b * P / (R + b2 * P), 0.0)


def eq_line(plant, n=801):
    rm = plant.r_max()
    r = np.linspace(-rm, rm, n)
    us = plant.u_ss(r)
    ok = np.isfinite(us) & (np.abs(us) <= UMAX)
    return r[ok], us[ok]


def lqr_screen(a_h, b_h, plant, r, us):
    """LQR (Q, R) designed from the MODEL's local (A^, B^) at each setpoint on the equilibrium line, applied
    to the TRUE local plant. 'local' uses the model Jacobian at the setpoint; 'win' averages it over +-W."""
    a_t, b_t = plant.Jx(r, us), plant.Ju(r, us)
    pole = a_t - b_t * dare_K(a_h, b_h)
    pole_o = a_t - b_t * dare_K(a_t, b_t)
    W = 0.25 * plant.B * UMAX / max(plant.A - 1.0, 1e-6) / 3.33
    k = max(1, int(round(W / max(r[1] - r[0], 1e-12))))
    ker = np.ones(2 * k + 1)
    nrm = np.convolve(np.ones_like(a_h), ker, "same")
    a_w, b_w = np.convolve(a_h, ker, "same") / nrm, np.convolve(b_h, ker, "same") / nrm
    pole_w = a_t - b_t * dare_K(a_w, b_w)
    return dict(lqr_unstable=float(np.mean(np.abs(pole) >= 1.0)), lqr_unstable_win=float(np.mean(np.abs(pole_w) >= 1.0)),
                lqr_pole_err_max=float(np.max(np.abs(pole - pole_o))), eq_gain_min=float(np.min(b_h / b_t)),
                eq_gain_max=float(np.max(b_h / b_t)))


CENTRE_WIN = (0.5, 1.5)    # |x| <= 0.5, |u| <= 1.5: 1.9 % of the box around (0, 0), where the LayerNorm kink sits


def kink_tests(E, X, U, sx=XMAX / np.sqrt(3.0), su=UMAX / np.sqrt(3.0)):
    """Four measurable signatures that separate a localised kink from smooth approximation error, computed from the
    input-gain error E = (dF/du - B)/B on the evaluation grid (E: [S, n, n], rows = u, columns = x)."""
    E = np.asarray(E, float)
    A = np.abs(E)
    S = E.shape[0]
    rms = np.sqrt((E ** 2).reshape(S, -1).mean(1))
    peak = A.reshape(S, -1).max(1)
    win = (np.abs(X) <= CENTRE_WIN[0]) & (np.abs(U) <= CENTRE_WIN[1])
    centre = (E ** 2)[:, win].sum(1) / np.maximum((E ** 2).reshape(S, -1).sum(1), 1e-300)
    dx, du = X[0, 1] - X[0, 0], U[1, 0] - U[0, 0]
    gU, gX = np.gradient(E, du, dx, axis=(1, 2))
    steep = np.sqrt((gX * sx) ** 2 + (gU * su) ** 2).reshape(S, -1).max(1)
    k = A.reshape(S, -1).argmax(1)
    wx, wu = X.ravel()[k], U.ravel()[k]
    xmax, umax = np.abs(X).max(), np.abs(U).max()
    where = np.where((np.abs(wx) <= 2 * CENTRE_WIN[0]) & (np.abs(wu) <= 2 * CENTRE_WIN[1]), "centre",
                     np.where((np.abs(wx) >= 0.95 * xmax) | (np.abs(wu) >= 0.95 * umax), "edge", "interior"))
    return dict(crest=peak / np.maximum(rms, 1e-300), centre_energy=centre, steep=steep, worst_x=wx, worst_u=wu, worst_where=where)


def evaluate_batch(P, S0, norm, plant, test, ngrid, full=False):
    """Metrics for every seed of a stacked model P. Never touches training data."""
    ev = get_evaluator(norm)
    S = len(S0)
    s0 = jnp.asarray(S0)
    tile = lambda a: jnp.asarray(np.repeat(np.asarray(a)[None], S, 0))
    Ft, _, _ = ev(P, tile(test["x"]), tile(test["u"]), s0)
    tmse = np.mean((np.asarray(Ft) - test["y"][None]) ** 2, 1)
    xg, ug = grid(ngrid)
    _, jxg, jug = ev(P, tile(xg), tile(ug), s0)
    jxg, jug = np.asarray(jxg), np.asarray(jug)
    ex, eu = jxg - plant.Jx(xg, ug)[None], (jug - plant.Ju(xg, ug)[None]) / plant.B
    out = dict(rel_mse=tmse / np.var(test["y"]), Jx_rms=np.sqrt((ex ** 2).mean(1)), Ju_rms_rel=np.sqrt((eu ** 2).mean(1)),
               Jx_sup=np.abs(ex).max(1), Ju_sup_rel=np.abs(eu).max(1),
               flip_area=np.mean(np.sign(jug) != np.sign(plant.Ju(xg, ug))[None], 1))
    inner = (np.abs(xg) <= 0.95 * XMAX) & (np.abs(ug) <= 0.95 * UMAX)        # box without its outer 5 % (edge effects)
    out["flip_area_interior"] = np.mean((np.sign(jug) != np.sign(plant.Ju(xg, ug))[None])[:, inner], 1)
    if norm == "layer":
        KL = np.stack([kink_line_points(unstack_tree(P, i)) for i in range(S)])
        _, jxk, juk = ev(P, jnp.asarray(KL[..., 0]), jnp.asarray(KL[..., 1]), s0)
        ekx = np.abs(np.asarray(jxk) - plant.Jx(KL[..., 0], KL[..., 1])).max(1)
        eku = np.abs((np.asarray(juk) - plant.Ju(KL[..., 0], KL[..., 1])) / plant.B).max(1)
        out["Jx_sup"] = np.maximum(out["Jx_sup"], ekx)
        out["Ju_sup_rel"] = np.maximum(out["Ju_sup_rel"], eku)
    if not full:
        return out, None
    out["gain_bad_area"] = np.mean(np.abs(jug / plant.Ju(xg, ug)[None] - 1.0) > 1.0, 1)
    out.update(kink_tests(eu.reshape(S, ngrid, ngrid), xg.reshape(ngrid, ngrid), ug.reshape(ngrid, ngrid)))
    r, us = eq_line(plant)
    _, jxe, jue = ev(P, tile(r), tile(us), s0)
    jxe, jue = np.asarray(jxe), np.asarray(jue)
    scr = [lqr_screen(jxe[i], jue[i], plant, r, us) for i in range(S)]
    for key in scr[0]:
        out[key] = np.array([s[key] for s in scr])
    geo = [kink_center(unstack_tree(P, i)) for i in range(S)] if norm == "layer" else [(np.full(2, np.nan), np.nan, np.nan)] * S
    out["zmin_x"] = np.array([g[0][0] for g in geo])
    out["zmin_u"] = np.array([g[0][1] for g in geo])
    out["rstar_x"] = np.array([g[1] for g in geo])
    out["rstar_u"] = np.array([g[2] for g in geo])
    zoom = []
    for i in range(S):
        cx = np.clip(out["zmin_x"][i] if np.isfinite(out["zmin_x"][i]) else 0.0, -XMAX + 1.5, XMAX - 1.5)
        cu = np.clip(out["zmin_u"][i] if np.isfinite(out["zmin_u"][i]) else 0.0, -UMAX + 4.0, UMAX - 4.0)
        zx, zu = np.meshgrid(np.linspace(cx - 1.5, cx + 1.5, 121), np.linspace(cu - 4.0, cu + 4.0, 121))
        zoom.append((zx.ravel(), zu.ravel()))
    ZX, ZU = np.stack([z[0] for z in zoom]), np.stack([z[1] for z in zoom])
    _, _, juz = ev(P, jnp.asarray(ZX), jnp.asarray(ZU), s0)
    prof = dict(eq_r=r, eq_u=us, eq_Ju=jue, eq_Jx=jxe, eq_Ju_true=plant.Ju(r, us), eq_Jx_true=plant.Jx(r, us),
                map_x=xg.reshape(ngrid, ngrid), map_u=ug.reshape(ngrid, ngrid),
                map_Ju=jug.reshape(S, ngrid, ngrid).astype(np.float32), map_Ju_true=plant.Ju(xg, ug).reshape(ngrid, ngrid),
                zoom_x=ZX.reshape(S, 121, 121), zoom_u=ZU.reshape(S, 121, 121),
                zoom_Ju=np.asarray(juz).reshape(S, 121, 121).astype(np.float32))
    return out, prof


# ------------------------------------------------------------------------------------------------ #
# 6. The experiment: configurations, resumable I/O                                                 #
# ------------------------------------------------------------------------------------------------ #
LOSSES = {"J_x only": (1.0, 0.0), "J_u only": (0.0, 1.0), "J_x + J_u": (1.0, 1.0)}


def level1_configs(T=3000, T_extra=1000):
    """14 configurations. T = main training length, T_extra = the extra phase of the 4000-step schedules.
    two-stage = MSE for T steps, then MSE + Jacobian loss for T_extra steps (fresh Adam moments at the switch)."""
    C = [dict(name="MSE (3000)", group="MSE", loss="MSE", schedule="3000", norm="layer", labels=None,
              phases=[("mse", 0, 0, T)]),
         dict(name="MSE (3000+1000)", group="MSE", loss="MSE", schedule="3000+1000", norm="layer", labels=None,
              phases=[("mse", 0, 0, T), ("mse", 0, 0, T_extra)]),
         dict(name="static norm, MSE", group="kink-free", loss="MSE", schedule="3000", norm="static", labels=None,
              phases=[("mse", 0, 0, T)]),
         dict(name="no norm, MSE", group="kink-free", loss="MSE", schedule="3000", norm="none", labels=None,
              phases=[("mse", 0, 0, T)])]
    for lname, (wx, wu) in LOSSES.items():
        C += [dict(name=f"{lname} (3000)", group=lname, loss=lname, schedule="3000", norm="layer", labels="est",
                   phases=[("sob", wx, wu, T)]),
              dict(name=f"{lname} (4000)", group=lname, loss=lname, schedule="4000", norm="layer", labels="est",
                   phases=[("sob", wx, wu, T + T_extra)]),
              dict(name=f"{lname} (two-stage)", group=lname, loss=lname, schedule="two-stage", norm="layer", labels="est",
                   phases=[("mse", 0, 0, T), ("sob", wx, wu, T_extra)])]
    C.append(dict(name="J_x + J_u, TRUE labels (3000)", group="true labels", loss="J_x + J_u", schedule="3000",
                  norm="layer", labels="true", phases=[("sob", 1.0, 1.0, T)]))
    return C


def out_root(name="level1_out"):
    if os.environ.get("L1_OUT_ROOT"):
        return os.environ["L1_OUT_ROOT"]
    base = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
    return os.path.join(base, name)


def append_csv(path, rows):
    if not rows:
        return
    new = not os.path.exists(path)
    cols = list(rows[0].keys())
    if not new:
        with open(path) as f:
            cols = next(csv.reader(f))
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def safe(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
