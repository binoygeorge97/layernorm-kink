"""phase_core.py - closed loops with a reference (setpoint), and what the controller sees along the way.

Three kinds of controller, all designed or trained from a surrogate F(x, u) and all run on the TRUE plant
x+ = 1.03 x + 0.01 u (and, for comparison, on the surrogate itself):

  lqr_fixed   LQR designed once at the setpoint: linearise the surrogate at its own equilibrium (r, u_hat(r)) where
              F(r, u_hat) = r, K = DARE(A_hat, B_hat), then u = clip(u_hat - K (x - r), +-10). Only the surrogate's
              slopes AT THE SETPOINT (and its equilibrium input u_hat) enter. Negative control for "the path matters".
  lqr_sched   Gain-scheduled LQR: at every step relinearise the surrogate at (x_k, u_{k-1}) and recompute K_k. The
              slopes ALONG THE PATH enter directly: a wrong-sign B_hat gives a wrong-sign gain.
  dpc         GRU controller trained through the surrogate (regulation controllers from dpc_core, or the
              reference-tracking controllers trained here: input (x - r, r), loss around the surrogate's equilibrium).

The reference controller is the same design on the TRUE plant (analytic LQR, or DPC trained through the true plant).
Everything uses the cost of dpc_core: sum_k Q (x_k - r)^2 + R (u_k - u_ss(r))^2 + Q_f (x_N - r)^2, u_ss(r) = 3 r.
"""
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import level1_core as L
import dpc_core as DC

tmap = jax.tree_util.tree_map
Q, R, QF, N = DC.CONFIG["Q_x"], DC.CONFIG["R_u"], DC.CONFIG["Q_f"], DC.CONFIG["N"]
UB = DC.U_BOUND
R_EVAL = np.array([-2.0, -1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0, 2.0])   # setpoints evaluated
X0_TRACK = np.linspace(-3.0, 3.0, 9)                                              # starts per setpoint (metrics)
R_TRAIN = 2.0                                                                     # tracking DPC: r ~ U[-2, 2]
SETTLED = DC.SETTLED


# ------------------------------------------------------------------------------------------------ #
# 1. Models with one interface: F and its slopes on [S, n] arrays (S = seeds)                       #
# ------------------------------------------------------------------------------------------------ #
class Model:
    """A stacked surrogate (one per seed) or the true plant, evaluated on arrays X, U of shape [S, n]."""

    def __init__(self, P=None, S0=None, norm=None, seeds=None, plant=None, name=""):
        self.P, self.norm, self.seeds, self.plant, self.name = P, norm, list(seeds), plant, name
        self.true = P is None
        if not self.true:
            self.S0 = jnp.asarray(np.asarray(S0, float))
            self.ev = L.get_evaluator(norm)

    @classmethod
    def surrogate(cls, out_dir, name, seeds, display=""):
        P, S0, met, sseeds = DC.load_surrogate(out_dir, name)
        idx = [sseeds.index(s) for s in seeds]
        P = tmap(lambda a: jnp.asarray(np.asarray(a)[idx]), P)
        cfg = DC.surrogate_configs()[name]
        m = cls(P, np.asarray(S0)[idx], cfg["norm"], seeds, name=display or name)
        m.met = {k: np.asarray(v)[idx] for k, v in met.items()}
        return m

    @classmethod
    def true_plant(cls, plant, seeds):
        m = cls(seeds=seeds, plant=plant, name="true plant")
        m.met = {}
        return m

    def seed_view(self, i):
        """The same model restricted to seed index i (still stacked, S = 1)."""
        if self.true:
            return Model.true_plant(self.plant, [self.seeds[i]])
        m = Model(tmap(lambda a: a[i:i + 1], self.P), np.asarray(self.S0)[i:i + 1], self.norm, [self.seeds[i]], name=self.name)
        m.met = {k: v[i:i + 1] for k, v in self.met.items()}
        return m

    def fj(self, X, U):
        """F, dF/dx, dF/du on [S, n] arrays (numpy in, numpy out)."""
        X, U = np.asarray(X, float), np.asarray(U, float)
        if self.true:
            return self.plant.f(X, U), self.plant.Jx(X, U) + 0 * X, self.plant.Ju(X, U) + 0 * X
        out = [], [], []
        n = X.shape[1]
        step = max(1, 400_000 // max(1, X.shape[0]))
        for s in range(0, n, step):
            F, jx, ju = self.ev(self.P, jnp.asarray(X[:, s:s + step]), jnp.asarray(U[:, s:s + step]), self.S0)
            for o, v in zip(out, (F, jx, ju)):
                o.append(np.asarray(v))
        return tuple(np.concatenate(o, 1) for o in out)

    def kink(self):
        """(zmin_x, zmin_u, rstar_x, rstar_u) per seed, NaN for kink-free models."""
        if self.true or self.norm != "layer":
            return tuple(np.full(len(self.seeds), np.nan) for _ in range(4))
        return tuple(self.met[k] for k in ("zmin_x", "zmin_u", "rstar_x", "rstar_u"))


# ------------------------------------------------------------------------------------------------ #
# 2. The surrogate's own equilibrium: u_hat(r) with F(r, u_hat) = r                                 #
# ------------------------------------------------------------------------------------------------ #
def linear_fit(model, n=81):
    """Least-squares linear model F ~ a x + b u + c of each seed's surrogate over the data box (a data-only guess)."""
    xg, ug = L.grid(n)
    S = len(model.seeds)
    F, _, _ = model.fj(np.tile(xg, (S, 1)), np.tile(ug, (S, 1)))
    Phi = np.column_stack([xg, ug, np.ones_like(xg)])
    coef = np.linalg.lstsq(Phi, F.T, rcond=None)[0]          # [3, S]
    return coef[0], coef[1], coef[2]


def equilibrium(model, r, n_u=801):
    """u_hat [S, nr]: the surrogate's steady-state input for setpoints r (array [nr] or [S, nr]).
    F(r, .) - r can cross zero more than once near the kink (a wrong-sign gain makes F non-monotone in u): we take the
    root closest to the global linear fit's guess (r - a r - c) / b, which needs no true plant. Also returns the number
    of roots (> 1 = the surrogate has several equilibria at this setpoint) and the surrogate's slopes at (r, u_hat)."""
    S = len(model.seeds)
    r = np.broadcast_to(np.asarray(r, float), (S,) + np.shape(r)[-1:]).copy()
    nr = r.shape[1]
    if model.true:
        A, B = model.plant.A, model.plant.B
        uh = (r - A * r) / B
        return uh, np.ones_like(uh, int), np.full_like(uh, A), np.full_like(uh, B)
    a, b, c = linear_fit(model)
    guess = (r - a[:, None] * r - c[:, None]) / b[:, None]
    ug = np.linspace(-UB, UB, n_u)
    X = np.repeat(r, n_u, 1)
    U = np.tile(ug, (S, nr))
    G = model.fj(X, U)[0].reshape(S, nr, n_u) - r[..., None]
    uh = np.empty((S, nr))
    nroot = np.zeros((S, nr), int)
    for s in range(S):
        for j in range(nr):
            g = G[s, j]
            idx = np.where(np.sign(g[:-1]) * np.sign(g[1:]) <= 0)[0]
            if len(idx) == 0:
                uh[s, j] = ug[np.argmin(np.abs(g))]
                continue
            roots = ug[idx] - g[idx] * (ug[idx + 1] - ug[idx]) / np.where(g[idx + 1] != g[idx], g[idx + 1] - g[idx], 1.0)
            nroot[s, j] = len(roots)
            uh[s, j] = roots[np.argmin(np.abs(roots - guess[s, j]))]
    # refine with a few Newton steps on the chosen branch (stays on it: the bracket is one grid cell wide)
    for _ in range(3):
        F, _, ju = model.fj(r, uh)
        step = (F - r) / np.where(np.abs(ju) > 1e-9, ju, 1e-9)
        uh = np.clip(uh - np.clip(step, -0.03, 0.03), -UB, UB)
    _, ah, bh = model.fj(r, uh)
    return uh, nroot, ah, bh


# ------------------------------------------------------------------------------------------------ #
# 3. LQR closed loops (fixed at the setpoint, or rescheduled every step)                            #
# ------------------------------------------------------------------------------------------------ #
def run_lqr(design, plant_model, r, x0, kind, N=N):
    """design = the model the controller is designed from; plant_model = the model it runs on (true plant normally).
    r, x0: [S, n]. kind in {'fixed', 'sched'}. Returns xs [N+1, S, n], us [N, S, n], K [N, S, n]."""
    uh, nroot, ah, bh = equilibrium(design, r)
    Kf = L.dare_K(ah, bh)
    x, u_prev = np.asarray(x0, float).copy(), uh.copy()
    xs, us, Ks = [x], [], []
    for _ in range(N):
        if kind == "fixed":
            K = Kf
        else:
            _, a, b = design.fj(x, u_prev)
            K = L.dare_K(a, b)
        u = np.clip(uh - K * (x - r), -UB, UB)
        x = plant_model.fj(np.clip(x, -1e3, 1e3), u)[0] if not plant_model.true else plant_model.plant.f(x, u)
        xs.append(x), us.append(u), Ks.append(K)
        u_prev = u
    return np.stack(xs), np.stack(us), np.stack(Ks), dict(uh=uh, nroot=nroot, ah=ah, bh=bh, K_fixed=Kf)


# ------------------------------------------------------------------------------------------------ #
# 4. DPC closed loops (regulation controllers from dpc_core, or tracking controllers from below)    #
# ------------------------------------------------------------------------------------------------ #
def _gru_batch(C, h, inp):
    return jax.vmap(DC.ctrl_step)(C, h, inp)


def run_dpc(C, plant_model, x0, r=None, uh=None, N=N, track=False):
    """C stacked over seeds [S]. x0 (and r) [S, n]. Regulation controllers take x; tracking ones take (x - r, r).
    Returns xs [N+1, S, n], us [N, S, n]."""
    S, n = x0.shape
    x = np.asarray(x0, float).copy()
    h = jnp.zeros((S, n, DC.HIDDEN if "Wz" in C else 0))
    step = jax.jit(_gru_batch)
    xs, us = [x], []
    for _ in range(N):
        inp = np.stack([x - r, r], -1) if track else x[..., None]
        h, u = step(C, h, jnp.asarray(inp))
        u = np.asarray(u)[..., 0]
        x = plant_model.plant.f(x, u) if plant_model.true else plant_model.fj(np.clip(x, -1e3, 1e3), u)[0]
        xs.append(x), us.append(u)
    return np.stack(xs), np.stack(us)


# ------------------------------------------------------------------------------------------------ #
# 5. Metrics and what the controller sees along the path                                            #
# ------------------------------------------------------------------------------------------------ #
def track_cost(xs, us, r, plant):
    """True cost of each trajectory, measured around the TRUE equilibrium (r, u_ss(r))."""
    uss = plant.u_ss(r)
    return (Q * (xs[:-1] - r) ** 2 + R * (us - uss) ** 2).sum(0) + QF * (xs[-1] - r) ** 2


def track_metrics(xs, us, r, plant, ref_cost=None):
    J = track_cost(xs, us, r, plant)
    fin = np.abs(xs[-20:].mean(0) - r)
    div = ~np.isfinite(xs).all(0) | (np.nanmax(np.abs(xs), 0) > DC.DIVERGE)
    out = dict(cost=J, offset=fin, diverged=div, not_settled=(fin > SETTLED) & ~div)
    if ref_cost is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["gap"] = np.where(ref_cost > 1e-9, J / np.maximum(ref_cost, 1e-300) - 1.0, np.nan)
    return out


def seen_gain(model, xs, us):
    """The surrogate's slopes at the visited points (x_k, u_k): Jx_hat - A and Ju_hat / B. xs [N+1, S, n], us [N, S, n]."""
    Np, S, n = us.shape
    X = np.moveaxis(xs[:-1], 0, 1).reshape(S, -1)
    U = np.moveaxis(us, 0, 1).reshape(S, -1)
    _, jx, ju = model.fj(np.clip(X, -50, 50), U)
    back = lambda a: np.moveaxis(a.reshape(S, Np, n), 1, 0)
    return back(jx), back(ju)


def kink_visits(model, xs, us):
    """Share of steps inside each seed's closed-form kink ellipse."""
    zx, zu, rx, ru = model.kink()
    if not np.isfinite(rx).any():
        return np.full(xs.shape[1:], np.nan)
    inside = ((xs[:-1] - zx[None, :, None]) / rx[None, :, None]) ** 2 + ((us - zu[None, :, None]) / ru[None, :, None]) ** 2 <= 1
    return inside.mean(0)


def lqr_study(models, plant, refs=R_EVAL, starts=X0_TRACK, N=N):
    """Fixed and scheduled LQR from every surrogate, every seed, every (setpoint, start), on the TRUE plant.
    Returns a long table (one row per model, seed, controller, setpoint, start) and the trajectories."""
    import pandas as pd
    truth = Model.true_plant(plant, models[0].seeds)
    S = len(truth.seeds)
    rr, xx = np.meshgrid(refs, starts, indexing="ij")
    r = np.tile(rr.ravel(), (S, 1))
    x0 = np.tile(xx.ravel(), (S, 1))
    xs_o, us_o, _, _ = run_lqr(truth, truth, r, x0, "fixed", N)
    J_o = track_cost(xs_o, us_o, r, plant)
    rows, trajs = [], {}
    for m in models:
        zx, zu, rx, ru = m.kink()
        for kind in ("fixed", "sched"):
            t0 = time.time()
            xs, us, Ks, info = run_lqr(m, truth, r, x0, kind, N)
            met = track_metrics(xs, us, r, plant, J_o)
            share = kink_visits(m, xs, us)
            jx, ju = seen_gain(m, xs, us)
            wrong = (ju <= 0).mean(0)
            trajs[(m.name, kind)] = dict(xs=xs.astype(np.float32), us=us.astype(np.float32), r=r, x0=x0, info=info)
            for i, s in enumerate(m.seeds):
                d_eq = np.hypot((rr.ravel() - zx[i]) / rx[i], (plant.u_ss(rr.ravel()) - zu[i]) / ru[i]) if np.isfinite(rx[i]) else np.nan + 0 * rr.ravel()
                for j in range(r.shape[1]):
                    rows.append(dict(model=m.name, seed=s, controller=f"LQR {kind}", r=r[i, j], x0=x0[i, j],
                                     cost=met["cost"][i, j], ref_cost=J_o[i, j], gap=met["gap"][i, j],
                                     offset=met["offset"][i, j], diverged=bool(met["diverged"][i, j]),
                                     not_settled=bool(met["not_settled"][i, j]), kink_share=share[i, j],
                                     wrong_sign_share=wrong[i, j], uh=info["uh"][i, j], u_ss=plant.u_ss(r[i, j]),
                                     n_eq=info["nroot"][i, j], Ah=info["ah"][i, j], Bh=info["bh"][i, j],
                                     K_fixed=info["K_fixed"][i, j], setpoint_kink_dist=d_eq[j]))
            print(f"  {m.name:22s} LQR {kind:5s}: median gap {100 * np.nanmedian(met['gap']):8.3g} %, diverged "
                  f"{int(met['diverged'].sum())}/{met['diverged'].size} | {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows), trajs, dict(xs=xs_o, us=us_o, r=r, x0=x0, cost=J_o)


# ------------------------------------------------------------------------------------------------ #
# 6. Reference-tracking DPC: GRU input (x - r, r); loss around the surrogate's own equilibrium      #
# ------------------------------------------------------------------------------------------------ #
def track_keys(seed, rep):
    """controller init 50000 + s, training pairs 60000 + s (+ 1000 x repeat): separate from the regulation runs."""
    return 50_000 + int(seed) + 1000 * int(rep), 60_000 + int(seed) + 1000 * int(rep)


def rollout_track(c, dyn, p, s0, x0, r, N, hidden=None):
    h0 = jnp.zeros((x0.shape[0], DC.HIDDEN if "Wz" in c else 0), x0.dtype)

    def step(carry, _):
        h, x = carry
        h, u = DC.ctrl_step(c, h, jnp.concatenate([x - r, r], -1))
        return (h, dyn(p, s0, x, u)), (x, u)

    (h, xN), (xs, us) = jax.lax.scan(jax.checkpoint(step), (h0, x0), None, length=N)
    return jnp.concatenate([xs, xN[None]], 0), us


def track_loss(c, dyn, p, s0, x0, r, uh, N):
    """(sum_k mean(Q (x_k - r)^2 + R (u_k - u_hat(r))^2) + mean(Q_f (x_N - r)^2)) / N. u_hat = the SURROGATE's
    equilibrium input (the true plant's 3 r for the oracle): DPC only knows its model."""
    xs, us = rollout_track(c, dyn, p, s0, x0, r, N)
    stage = jnp.sum(jnp.mean(Q * (xs[:-1, :, 0] - r[:, 0]) ** 2 + R * (us[:, :, 0] - uh[:, 0]) ** 2, axis=1))
    return (stage + jnp.mean(QF * (xs[-1, :, 0] - r[:, 0]) ** 2)) / N


_TRACK_STEPS = {}


def _track_trainer(key, dyn, N, n_epochs, opt):
    k = (key, N, n_epochs)
    if k in _TRACK_STEPS:
        return _TRACK_STEPS[k]

    def one(c, o, p, s0, x0, r, uh):
        def st(carry, _):
            c, o = carry
            loss, g = jax.value_and_grad(track_loss)(c, dyn, p, s0, x0, r, uh, N)
            upd, o = opt.update(g, o, c)
            return (optax.apply_updates(c, upd), o), loss
        (c, o), losses = jax.lax.scan(st, (c, o), None, length=n_epochs)
        return c, o, losses

    f = jax.jit(jax.vmap(one))
    _TRACK_STEPS[k] = f
    return f


def train_track(model_spec, model, seeds, plant, curriculum=DC.CURRICULUM, n_pairs=DC.N_X0, dtype=jnp.float64,
                reps=(0,), log_every=250, verbose=True):
    """One tracking controller per (seed, repeat), trained through `model` (a Model; the true plant for the oracle).
    Training pairs: x0 ~ U[-3, 3], r ~ U[-2, 2]; u_hat(r) from the model's own equilibrium."""
    pairs = [(s, rp) for rp in reps for s in seeds]
    X0 = np.stack([np.asarray(jax.random.uniform(jax.random.PRNGKey(track_keys(s, rp)[1]), (n_pairs, 1), jnp.float64,
                                                 -DC.X0_RANGE, DC.X0_RANGE)) for s, rp in pairs])
    Rr = np.stack([np.asarray(jax.random.uniform(jax.random.PRNGKey(track_keys(s, rp)[1] + 7), (n_pairs, 1), jnp.float64,
                                                 -R_TRAIN, R_TRAIN)) for s, rp in pairs])
    t0 = time.time()
    UH = []
    for k, (s, rp) in enumerate(pairs):
        i = seeds.index(s)
        uh, nroot, _, _ = equilibrium(model.seed_view(i), Rr[k, :, 0][None], n_u=401)
        UH.append(uh[0][:, None])
    UH = np.stack(UH)
    if verbose:
        print(f"    surrogate equilibria for {len(pairs)} x {n_pairs} training setpoints: {time.time() - t0:.0f}s", flush=True)
    if model.true:
        dyn = DC.make_dynamics("true", plant=plant)
        P = L.stack_trees([L.init_params(s) for s, _ in pairs])
        S0 = jnp.ones(len(pairs), dtype)
    else:
        dyn = DC.make_dynamics("surrogate", model.norm, plant)
        idx = [seeds.index(s) for s, _ in pairs]
        P = tmap(lambda a: jnp.asarray(np.asarray(a)[idx], dtype), model.P)
        S0 = jnp.asarray(np.asarray(model.S0)[idx], dtype)
    C = L.stack_trees([DC.init_controller(track_keys(s, rp)[0] - 20_000, d_x=2, dtype=dtype) for s, rp in pairs])
    opt = optax.adamw(DC.CTRL_LR)
    O = jax.vmap(opt.init)(C)
    X0, Rr, UH = (jnp.asarray(a, dtype) for a in (X0, Rr, UH))
    curves, epoch = [], 0
    for Nh, n_ep in curriculum:
        done = 0
        while done < n_ep:
            n = min(log_every, n_ep - done)
            C, O, losses = _track_trainer((model_spec["name"], str(dtype), DC.ARCH), dyn, Nh, n, opt)(C, O, P, S0, X0, Rr, UH)
            losses = np.asarray(losses)
            done += n
            epoch += n
            for i, (s, rp) in enumerate(pairs):
                curves.append(dict(model=model_spec["name"], seed=s, rep=rp, epoch=epoch, horizon=Nh, loss=float(losses[i, -1])))
        if verbose:
            last = np.array([c["loss"] for c in curves[-len(pairs):]])
            print(f"    horizon {Nh:3d}: {n_ep:5d} epochs | loss on its model: median {np.median(last):.4g}, max {np.max(last):.4g}"
                  f" | {time.time() - t0:.0f}s", flush=True)
    return C, curves, pairs


def run_track_study(models, seeds, plant, out_dir, reps=(0,), curriculum=DC.CURRICULUM, n_pairs=DC.N_X0,
                    dtype=jnp.float64, refs=R_EVAL, starts=X0_TRACK, sur_dir=None):
    """Train tracking DPC through every model (resumable per model and repeat), evaluate on the TRUE plant for every
    (setpoint, start), and save rows (track_results.csv), controllers and trajectories (track_*/) in out_dir. The surrogates
    are read from sur_dir (default out_dir: the DPC run's folder, which holds surrogates/)."""
    import pandas as pd
    for d in ("track_controllers", "track_trajectories"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)
    res = os.path.join(out_dir, "track_results.csv")
    done = set()
    if os.path.exists(res):
        df = pd.read_csv(res)
        done = {(m, int(r)) for (m, r), g in df.groupby(["model", "rep"]) if set(seeds) <= set(g.seed)}
    truth = Model.true_plant(plant, seeds)
    S = len(seeds)
    rr, xx = np.meshgrid(refs, starts, indexing="ij")
    r_e, x_e = np.tile(rr.ravel(), (S, 1)), np.tile(xx.ravel(), (S, 1))
    for spec in models:
        todo = [rp for rp in reps if (spec["name"], rp) not in done]
        if not todo:
            continue
        t0 = time.time()
        model = truth if spec.get("kind") == "true" else Model.surrogate(sur_dir or out_dir, spec["surrogate"], seeds, spec["name"])
        print(f"  tracking DPC through '{spec['name']}', repeats {todo}", flush=True)
        C, curves, pairs = train_track(spec, model, seeds, plant, curriculum, n_pairs, dtype, todo)
        rows = []
        for k, rp in enumerate(todo):
            Ck = tmap(lambda a: a[k * S:(k + 1) * S], C)
            xs, us = run_dpc(Ck, truth, x_e, r_e, track=True)
            xo, uo = (run_dpc(Ck, model, x_e, r_e, track=True) if not model.true else (xs, us))
            met = track_metrics(xs, us, r_e, plant)
            share = kink_visits(model, xs, us) if not model.true else np.full(xs.shape[1:], np.nan)
            if model.true:
                wrong, bh = np.zeros(xs.shape[1:]), np.full(xs.shape[1:], plant.B)
            else:                                   # what the surrogate says along the true path and at the setpoint
                _, ju = seen_gain(model, xs, us)
                wrong = (ju <= 0).mean(0)
                bh = equilibrium(model, r_e)[3]
            Jown = track_cost(xo, uo, r_e, plant)
            for i, s in enumerate(seeds):
                for j in range(r_e.shape[1]):
                    rows.append(dict(model=spec["name"], seed=s, rep=rp, r=r_e[i, j], x0=x_e[i, j], cost=met["cost"][i, j],
                                     cost_own=Jown[i, j], offset=met["offset"][i, j], diverged=bool(met["diverged"][i, j]),
                                     not_settled=bool(met["not_settled"][i, j]), kink_share=share[i, j],
                                     wrong_sign_share=wrong[i, j], Bh=bh[i, j]))
            np.savez_compressed(DC.rep_file(os.path.join(out_dir, "track_trajectories"), spec["name"], rp),
                                xs=xs.astype(np.float32), us=us.astype(np.float32), xo=xo.astype(np.float32),
                                uo=uo.astype(np.float32), r=r_e, x0=x_e, seeds=np.array(seeds))
            np.savez_compressed(DC.rep_file(os.path.join(out_dir, "track_controllers"), spec["name"], rp),
                                seeds=np.array(seeds), **{"c:" + kk: v for kk, v in DC._flat(Ck).items()})
        DC.append_rows(res, rows)
        DC.append_rows(os.path.join(out_dir, "track_curves.csv"), curves)
        g = pd.DataFrame(rows)
        print(f"  -> {spec['name']:24s} | median true cost {g.cost.median():.4g} | diverged {int(g.diverged.sum())} | "
              f"not settled {int(g.not_settled.sum())} of {len(g)} | {time.time() - t0:.0f}s", flush=True)
    return pd.read_csv(res) if os.path.exists(res) else None


def add_track_gaps(T):
    """gap = cost / cost of the oracle (same seed, repeat, setpoint, start) - 1; controller = 'DPC' (for the joint figures)."""
    o = T[T.model == "true plant (oracle)"][["seed", "rep", "r", "x0", "cost"]].rename(columns={"cost": "ref_cost"})
    T = T.drop(columns=[c for c in ("ref_cost", "gap") if c in T]).merge(o, on=["seed", "rep", "r", "x0"], how="left")
    T["gap"] = T.cost / T.ref_cost - 1.0
    T["controller"] = "DPC"
    return T


def load_track_controllers(out_dir, name, rep=0):
    fn = DC.rep_file(os.path.join(out_dir, "track_controllers"), name, rep)
    if not os.path.exists(fn):
        return None, None
    z = np.load(fn)
    return DC._unflat({k[2:]: z[k] for k in z.files if k.startswith("c:")}), list(z["seeds"])


def track_diagnostic(out_dir, sur_dir, seeds, plant, refs=R_EVAL, starts=X0_TRACK, rep=0):
    """Trap test for tracking DPC (like dpc_core.run_diagnostic): the tracking controller trained through the TRUE plant is run
    through each surrogate. Per model and seed (medians over setpoint x start pairs with |x0 - r| >= 0.5):
    ref_model_vs_true = its cost on the surrogate / on the true plant (~1: the surrogate judges a good controller correctly);
    trap = cost of DPC's own controller on the surrogate / the reference controller's (> 1: a better controller existed for
    this surrogate and DPC missed it)."""
    import pandas as pd
    Co, cs = load_track_controllers(out_dir, "true plant (oracle)", rep)
    idx = [cs.index(s) for s in seeds]
    Co = tmap(lambda a: a[np.array(idx)], Co)
    rr, xx = np.meshgrid(refs, starts, indexing="ij")
    r_e, x_e = np.tile(rr.ravel(), (len(seeds), 1)), np.tile(xx.ravel(), (len(seeds), 1))
    T = add_track_gaps(pd.read_csv(os.path.join(out_dir, "track_results.csv")))
    T = T[T.rep == rep]
    rows = []
    for m in DC.MODELS:
        if m.get("kind") == "true":
            continue
        model = Model.surrogate(sur_dir, m["surrogate"], seeds, m["name"])
        xs, us = run_dpc(Co, model, x_e, r_e, track=True)
        J = track_cost(xs, us, r_e, plant)
        for i, s in enumerate(seeds):
            g = T[(T.model == m["name"]) & (T.seed == s)].set_index(["r", "x0"]).loc[list(zip(r_e[i], x_e[i]))]
            far = np.abs(x_e[i] - r_e[i]) >= 0.5
            rows.append(dict(model=m["name"], seed=s, rep=rep, oracle_on_model=np.median(J[i, far]),
                             oracle_on_true=np.median(g.ref_cost.values[far]), own_on_model=np.median(g.cost_own.values[far]),
                             own_on_true=np.median(g.cost.values[far])))
    D = pd.DataFrame(rows)
    D["trap"] = D.own_on_model / D.oracle_on_model
    D["ref_model_vs_true"] = D.oracle_on_model / D.oracle_on_true
    D["gap_true"] = D.own_on_true / D.oracle_on_true - 1
    D.to_csv(os.path.join(out_dir, "track_diag.csv"), index=False)
    return D


# ------------------------------------------------------------------------------------------------ #
# 7. Training-time view of the trap: what DPC's gradients saw, and the landscape between the two   #
#    controllers (added 2026-10-23). Regulation task, 1000 training starts of repeat `rep`.          #
# ------------------------------------------------------------------------------------------------ #
def _training_starts(seeds, rep, dtype=jnp.float64):
    """The 1000 initial states each DPC run was trained on (key 30000 + s + 1000 rep), [S, 1000, 1]."""
    return jnp.stack([jax.random.uniform(jax.random.PRNGKey(DC.run_keys(s, rep)[1]), (DC.N_X0, 1), dtype,
                                         -DC.X0_RANGE, DC.X0_RANGE) for s in seeds])


def _stack_models(out_dir, name, seeds, dtype=jnp.float64):
    P, S0, met, ss = DC.load_surrogate(out_dir, name)
    idx = [ss.index(s) for s in seeds]
    P = tmap(lambda a: jnp.asarray(np.asarray(a)[idx], dtype), P)
    return P, jnp.asarray(np.asarray(S0)[idx], dtype), {k: np.asarray(v)[idx] for k, v in met.items()}


def adjoint_gradients(xs, us, jx, ju, plant, cfg=DC.CONFIG):
    """Per-step gradient of the regulation cost with respect to each input u_k along a GIVEN path, as backprop
    through the model would compute it: lam_N = 2 Q_f x_N, lam_k = 2 Q x_k + J_x(k) lam_{k+1}, g_k = 2 R u_k + J_u(k) lam_{k+1}.
    xs [N+1, B], us [N, B], jx, ju [N, B] = the slopes the model reports at (x_k, u_k). Returns g_model, g_true [N, B]
    (the same recursion with the true A, B), and lam_model [N+1, B]."""
    Q, R, QF = cfg["Q_x"], cfg["R_u"], cfg["Q_f"]
    N = us.shape[0]
    def run(JX, JU):
        lam = np.zeros_like(xs); g = np.zeros_like(us)
        lam[N] = 2 * QF * xs[N]
        for k in range(N - 1, -1, -1):
            g[k] = 2 * R * us[k] + JU[k] * lam[k + 1]
            lam[k] = 2 * Q * xs[k] + JX[k] * lam[k + 1]
        return g, lam
    gm, lm = run(jx, ju)
    gt, _ = run(np.full_like(jx, plant.A), np.full_like(ju, plant.B))
    return gm, gt, lm


def reference_path_colours(out_dir, seeds, plant, models=None, rep=0, dtype=jnp.float64, N=N):
    """What the surrogate's slopes look like along the REFERENCE controller's path (the controller trained through the true
    plant, run through each surrogate on the 1000 training starts): the slopes DPC had to learn from to reach a good policy.
    Per (model, seed): share of steps with wrong-sign J_u, with |J_u/B - 1| > 1, inside the kink ellipse; max |J_x|; share
    of steps with |J_x| > A + 0.5; and the adjoint per-step input gradients g_k through the model vs through the true plant
    along the same path: share of steps whose sign disagrees, and the share of the model's gradient energy sum g_k^2 that
    comes from steps inside the kink. Also the same statistics along DPC's OWN controller's path (what it ended up seeing)."""
    import pandas as pd
    models = models or [m for m in DC.MODELS if m.get("kind") != "true"]
    Co, cs = DC.load_controllers(out_dir, "true plant (oracle)", rep)
    Co = tmap(lambda a: jnp.asarray(np.asarray(a)[[cs.index(s) for s in seeds]], dtype), Co)
    x0 = _training_starts(seeds, rep, dtype)
    rows = []
    for m in models:
        cfg = DC.surrogate_configs()[m["surrogate"]]
        P, S0, met = _stack_models(out_dir, m["surrogate"], seeds, dtype)
        model = Model(P, np.asarray(S0), cfg["norm"], seeds, name=m["name"]); model.met = met
        roll = DC._roller(DC.make_dynamics("surrogate", cfg["norm"], plant), N)
        Cm, cm = DC.load_controllers(out_dir, m["name"], rep)
        Cm = tmap(lambda a: jnp.asarray(np.asarray(a)[[cm.index(s) for s in seeds]], dtype), Cm)
        zx, zu, rx, ru = model.kink()
        for who, C in (("reference", Co), ("own", Cm)):
            X, U = (np.asarray(a, float)[..., 0] for a in roll(C, P, S0, x0))     # [S, N+1, B], [S, N, B]
            xs, us = np.moveaxis(X, 0, 1), np.moveaxis(U, 0, 1)                      # [N+1, S, B]
            jx, ju = seen_gain(model, xs, us)
            for i, s in enumerate(seeds):
                g_m, g_t, lam = adjoint_gradients(xs[:, i], us[:, i], jx[:, i], ju[:, i], plant)
                if np.isfinite(rx[i]):
                    inside = ((xs[:-1, i] - zx[i]) / rx[i]) ** 2 + ((us[:, i] - zu[i]) / ru[i]) ** 2 <= 1
                else:
                    inside = np.zeros_like(us[:, i], bool)
                gain = ju[:, i] / plant.B
                e = g_m ** 2
                # the final approach: steps with |x| <= 0.2 (where regulation must end)
                near = np.abs(xs[:-1, i]) <= 0.2
                rows.append(dict(model=m["name"], seed=s, rep=rep, path=who,
                                 peak_Ju=float(met["Ju_sup_rel"][i]), rstar_x=rx[i], rstar_u=ru[i],
                                 wrong_sign=float((gain <= 0).mean()), bad_gain=float((np.abs(gain - 1) > 1).mean()),
                                 wrong_sign_near=float((gain[near] <= 0).mean()) if near.any() else np.nan,
                                 kink_share=float(inside.mean()), Jx_max=float(np.abs(jx[:, i]).max()),
                                 Jx_big=float((np.abs(jx[:, i]) > plant.A + 0.5).mean()),
                                 grad_sign_flip=float((np.sign(g_m) != np.sign(g_t)).mean()),
                                 grad_sign_flip_near=float((np.sign(g_m) != np.sign(g_t))[near].mean()) if near.any() else np.nan,
                                 grad_energy_kink=float(e[inside].sum() / max(e.sum(), 1e-300)),
                                 grad_energy_wrong=float(e[gain <= 0].sum() / max(e.sum(), 1e-300)),
                                 grad_rms_ratio=float(np.sqrt(e.mean() / max((g_t ** 2).mean(), 1e-300))),
                                 cost_model=float(DC.true_lqr_cost(X[i], U[i])),
                                 diverged=float((np.abs(X[i]).max(0) > DC.DIVERGE).mean())))
    return pd.DataFrame(rows)


def _interp(Ca, Cb, a):
    return tmap(lambda u, v: (1 - a) * u + a * v, Ca, Cb)


def cost_interpolation(out_dir, seeds, plant, model_spec, rep=0, alphas=np.linspace(0, 1, 21), dtype=jnp.float64, N=N):
    """The regulation loss along the straight line in weight space from DPC's own controller (alpha = 0) to the reference
    controller (alpha = 1), evaluated through the surrogate (what DPC minimised) and through the true plant, on the 1000
    training starts. A barrier in the surrogate loss with none in the true loss is the optimisation trap in cost units.
    The surrogate loss is also split into the stage cost accrued inside / outside the kink ellipse. Also returns the norm
    of the surrogate-loss gradient at both ends (is the reference controller a stationary point of the surrogate loss?)."""
    import pandas as pd
    cfg = DC.surrogate_configs()[model_spec["surrogate"]]
    P, S0, met = _stack_models(out_dir, model_spec["surrogate"], seeds, dtype)
    Co, cs = DC.load_controllers(out_dir, "true plant (oracle)", rep)
    Co = tmap(lambda a: jnp.asarray(np.asarray(a)[[cs.index(s) for s in seeds]], dtype), Co)
    Cm, cm = DC.load_controllers(out_dir, model_spec["name"], rep)
    Cm = tmap(lambda a: jnp.asarray(np.asarray(a)[[cm.index(s) for s in seeds]], dtype), Cm)
    x0 = _training_starts(seeds, rep, dtype)
    dyn_s, dyn_t = DC.make_dynamics("surrogate", cfg["norm"], plant), DC.make_dynamics("true", plant=plant)
    roll_s, roll_t = DC._roller(dyn_s, N), DC._roller(dyn_t, N)
    gS = jax.jit(jax.vmap(jax.grad(lambda c, p, s0, x: DC.dpc_loss(c, dyn_s, p, s0, x, N))))
    gnorm = lambda C: np.asarray(jax.vmap(lambda g: jnp.sqrt(sum(jnp.sum(l ** 2) for l in jax.tree_util.tree_leaves(g))))(gS(C, P, S0, x0)))
    zx, zu, rx, ru = (met[k] for k in ("zmin_x", "zmin_u", "rstar_x", "rstar_u"))
    Q, R, QF = DC.CONFIG["Q_x"], DC.CONFIG["R_u"], DC.CONFIG["Q_f"]
    rows = []
    for a in alphas:
        C = _interp(Cm, Co, float(a))
        Xs, Us = (np.asarray(v, float)[..., 0] for v in roll_s(C, P, S0, x0))
        Xt, Ut = (np.asarray(v, float)[..., 0] for v in roll_t(C, P, S0, x0))
        for i, s in enumerate(seeds):
            stage = Q * Xs[i, :-1] ** 2 + R * Us[i] ** 2                       # [N, B]
            if np.isfinite(rx[i]):
                inside = ((Xs[i, :-1] - zx[i]) / rx[i]) ** 2 + ((Us[i] - zu[i]) / ru[i]) ** 2 <= 1
            else:
                inside = np.zeros_like(stage, bool)
            rows.append(dict(model=model_spec["name"], seed=s, rep=rep, alpha=float(a),
                             loss_model=DC.true_lqr_cost(Xs[i], Us[i]), loss_true=DC.true_lqr_cost(Xt[i], Ut[i]),
                             loss_model_inside=float((stage * inside).sum(0).mean() / Xs.shape[1]),
                             loss_model_outside=float((stage * ~inside).sum(0).mean() / Xs.shape[1]),
                             kink_share_model=float(inside.mean()),
                             offset_model=float(np.median(np.abs(Xs[i, -20:]).mean(0))),
                             diverged_model=float((np.abs(Xs[i]).max(0) > DC.DIVERGE).mean())))
    D = pd.DataFrame(rows)
    g0, g1 = gnorm(Cm), gnorm(Co)
    G = pd.DataFrame(dict(model=model_spec["name"], seed=seeds, rep=rep, gradnorm_own=g0, gradnorm_reference=g1))
    return D, G


def closed_loop_stability(out_dir, seeds, plant, models=None, rep=0, dtype=jnp.float64, N=N, n_starts=64):
    """Is the GOOD controller closed-loop stable according to the surrogate? For the reference controller (trained through
    the true plant) run through each surrogate: the surrogate's slopes at its resting point (median over the last 20 steps:
    J_u_hat / B and J_x_hat), and the Lyapunov exponent of the closed loop (controller + model) along each rollout,
    (1/N) log || d(h_N, x_N) / d x_0 ||, averaged over n_starts training starts. Through the true plant the oracle gives
    log(0.927) = -0.076 per step. A positive exponent through the surrogate means the model's slopes make the good
    controller locally unstable at the setpoint, so backpropagating the loss through N steps multiplies the gradient by
    exp(N lyap): the loss VALUE can stay right while its GRADIENT explodes. Also reported for DPC's own controller."""
    import pandas as pd
    models = models or [m for m in DC.MODELS if m.get("kind") != "true"]
    Co, cs = DC.load_controllers(out_dir, "true plant (oracle)", rep)
    Co = tmap(lambda a: jnp.asarray(np.asarray(a)[[cs.index(s) for s in seeds]], dtype), Co)
    x0 = _training_starts(seeds, rep, dtype)[:, :n_starts]

    def make_lyap(dyn):
        def final(c, p, s0, x0_scalar):
            xs, us = DC.rollout(c, dyn, p, s0, x0_scalar[None, None], N)
            return xs[-1, 0, 0]
        def one(c, p, s0, x0s):            # x0s [n]
            # sensitivity of the final state to the initial state (true plant: 0.927^N); log / N
            d = jax.vmap(jax.grad(final, argnums=3), in_axes=(None, None, None, 0))(c, p, s0, x0s)
            return jnp.log(jnp.abs(d) + 1e-300) / N
        return jax.jit(jax.vmap(one))

    rows = []
    for m in models:
        cfg = DC.surrogate_configs()[m["surrogate"]]
        P, S0, met = _stack_models(out_dir, m["surrogate"], seeds, dtype)
        model = Model(P, np.asarray(S0), cfg["norm"], seeds, name=m["name"]); model.met = met
        dyn = DC.make_dynamics("surrogate", cfg["norm"], plant)
        roll = DC._roller(dyn, N)
        lyap = make_lyap(dyn)
        Cm, cm = DC.load_controllers(out_dir, m["name"], rep)
        Cm = tmap(lambda a: jnp.asarray(np.asarray(a)[[cm.index(s) for s in seeds]], dtype), Cm)
        for who, C in (("reference", Co), ("own", Cm)):
            X, U = (np.asarray(a, float)[..., 0] for a in roll(C, P, S0, x0))
            ly = np.asarray(lyap(C, P, S0, x0[..., 0]))                                   # [S, n]
            xs, us = np.moveaxis(X, 0, 1), np.moveaxis(U, 0, 1)
            jx, ju = seen_gain(model, xs, us)
            for i, s in enumerate(seeds):
                rows.append(dict(model=m["name"], seed=s, rep=rep, path=who, peak_Ju=float(met["Ju_sup_rel"][i]),
                                 rest_x=float(np.median(np.abs(X[i, -20:]))), rest_gain=float(np.median(ju[-20:, i] / plant.B)),
                                 rest_Jx=float(np.median(jx[-20:, i])), rest_gain_min=float(np.min(ju[-20:, i] / plant.B)),
                                 lyap_mean=float(np.mean(ly[i])), lyap_max=float(np.max(ly[i])),
                                 lyap_pos_share=float(np.mean(ly[i] > 0))))
    return pd.DataFrame(rows)
