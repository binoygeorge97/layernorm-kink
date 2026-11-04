"""MLP mechanism M2 + M3 (2026-11-04, POST HOC). For every trapped MLP run (Run 5a, LN MSE 3000 and LN slope from 0; controllers
data/run5_mlp/controllers) and for the MLP oracle of the same repeat, through the run's surrogate, N = 200, the repeat's 1000
training starts:
M2 saturation: pre = the MLP output before u = 10 tanh(pre). sat_frac = share of (start, step) pairs with |pre| > 2 (tanh slope
   sech^2(2) = 0.07). Gradient split: with u_k = pi(x_k) + delta_k, the adjoint a_{i,k} = dL/d delta_{i,k} (L = mean DPC loss);
   then grad L = sum_{i,k} a_{i,k} d pi(x_{i,k}) / d theta exactly, and g_unsat / g_sat are the sums over unsaturated / saturated
   pairs (checked: g_unsat + g_sat = grad L). Reported: ||g_unsat||, ||g_sat||, share_unsat = ||g_unsat|| / (||g_unsat|| + ||g_sat||).
M3 parking: final state x_200 through the surrogate and through the true plant (median, and the share of starts within 0.05 of
   the median); fixed points of the closed loops x -> F_hat(x, pi(x)) and x -> A x + B pi(x) on [-4, 4] (step 1e-3, sign changes
   refined by bisection) with their stability |d/dx| < 1; for each surrogate fixed point its distance from the kink centre in
   half-widths, d = sqrt(((x - x*) / r*_x)^2 + ((u - u*) / r*_u)^2), and the surrogate input slope there (J_u_hat / B < 0 = on a
   wrong-sign arm).
Usage (from theory/): python mlp_sat_park.py -> $T2_OUT/mlp_saturation.csv, mlp_parking.csv"""
import os
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
from t2_arch import load_ref, OUT
import dpc_core as DC
import paths

jax.config.update("jax_enable_x64", True)
ROOT_O = os.path.join(paths.REPO, "data", "dpc_out_mlp"); ROOT_D = os.path.join(paths.REPO, "data", "run5_mlp")
FAMS = {"LN MSE 3000": ("LN, MSE (3000)", "MSE (3000)"), "LN slope from 0": ("LN, J_x + J_u (3000)", "J_x + J_u (3000)")}
S = pd.read_csv(os.path.join(paths.REPO, "data", "dpc_out", "surrogates.csv"))
import sys; sys.path.insert(0, os.path.join(paths.REPO, "analysis"))
from common import runs
R = runs(ROOT_D)


def pre_fn(c, x):
    names = sorted(k for k in c if k.startswith("L")); a = jnp.reshape(x, (1,))
    for nm in names[:-1]:
        a = jnp.tanh(a @ c[nm]["W"] + c[nm]["b"])
    return (a @ c[names[-1]]["W"] + c[names[-1]]["b"])[0]


pi = lambda c, x: 10.0 * jnp.tanh(pre_fn(c, x))


def make(p, s0, norm):
    def roll(c, x0, delta, true=False):
        def st(x, d):
            u = pi(c, x) + d
            xn = T.A_TRUE * x + T.B_TRUE * u if true else T.F_point(p, x, u, s0, norm)
            return xn, (x, u)
        xN, (xs, us) = jax.lax.scan(st, x0, delta)
        return xs, us, xN

    def one(c, x0, delta):
        xs, us, xN = roll(c, x0, delta)
        return (jnp.sum(T.Q_X * xs ** 2 + T.R_U * us ** 2) + T.Q_F * xN ** 2) / T.N_HOR
    return roll, one


def load_dpc(stem, rep):
    z = np.load(DC.rep_file(os.path.join(ROOT_D, "controllers"), stem, rep))
    return DC._unflat({k[2:]: z[k] for k in z.files if k.startswith("c:")}), [int(s) for s in z["seeds"]]


def fixed_points(g, lo=-4.0, hi=4.0, h=1e-3):
    xs = np.arange(lo, hi + h, h); v = np.asarray(jax.vmap(g)(jnp.asarray(xs))) - xs
    out = []
    for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        a, b = xs[i], xs[i + 1]
        for _ in range(50):
            m = 0.5 * (a + b); fm = float(g(m)) - m
            a, b = (m, b) if np.sign(fm) == np.sign(float(g(a)) - a) else (a, m)
        x = 0.5 * (a + b); out.append((x, float(jax.grad(g)(x))))
    return out


if __name__ == "__main__":
    sat_rows, park_rows = [], []
    for fam, (model, sur) in FAMS.items():
        P, S0, norm, seeds, met = T.load_surrogate(fam)
        trapped = R[(R.model == model) & R.trapped][["seed", "rep"]].values.tolist()
        for rep in sorted({r for _, r in trapped}):
            Cd, csd = load_dpc(model, rep); Co, cso = load_ref(ROOT_O, rep)
            for s, r in [t for t in trapped if t[1] == rep]:
                i = seeds.index(s); p, s0 = T.seed_slice(P, i), S0[i]
                roll, one = make(p, s0, norm)
                X0 = T.training_starts(s, rep); Z = jnp.zeros((len(X0), T.N_HOR))
                k = S[(S.model == model) & (S.seed == s)].iloc[0]
                for who, C, cs in (("DPC (trapped)", Cd, csd), ("oracle", Co, cso)):
                    c = jax.tree_util.tree_map(lambda a: jnp.asarray(a[cs.index(s)]), C)
                    Lf = lambda cc, dd: jnp.mean(jax.vmap(lambda x0, d: one(cc, x0, d))(X0, dd))
                    gtot, adj = jax.grad(Lf, argnums=(0, 1))(c, Z)
                    xs, us, xN = jax.vmap(lambda x0: roll(c, x0, jnp.zeros(T.N_HOR)))(X0)
                    pre = jax.vmap(jax.vmap(lambda x: pre_fn(c, x)))(xs)
                    mask = (jnp.abs(pre) > 2.0)
                    _, vjp = jax.vjp(lambda cc: jax.vmap(jax.vmap(lambda x: pi(cc, x)))(xs), c)
                    g_sat, = vjp(adj * mask); g_un, = vjp(adj * (~mask))
                    fl = lambda t: jnp.concatenate([jnp.ravel(a) for a in jax.tree_util.tree_leaves(t)])
                    ns, nu, nt = float(jnp.linalg.norm(fl(g_sat))), float(jnp.linalg.norm(fl(g_un))), float(jnp.linalg.norm(fl(gtot)))
                    chk = float(jnp.linalg.norm(fl(g_sat) + fl(g_un) - fl(gtot)) / max(nt, 1e-300))
                    sat_rows.append(dict(family=fam, seed=s, rep=rep, controller=who, sat_frac=float(mask.mean()), grad=nt,
                                         g_unsat=nu, g_sat=ns, share_unsat=nu / (nu + ns), split_check=chk))
                    if who != "DPC (trapped)":
                        continue
                    _, _, xNt = jax.vmap(lambda x0: roll(c, x0, jnp.zeros(T.N_HOR), True))(X0)
                    xN, xNt = np.asarray(xN), np.asarray(xNt)
                    fs = fixed_points(lambda x: T.F_point(p, x, pi(c, x), s0, norm))
                    ft = fixed_points(lambda x: T.A_TRUE * x + T.B_TRUE * pi(c, x))
                    slope = T.slopes(p, s0, norm)
                    desc = []
                    for x, dg in fs:
                        u = float(pi(c, x)); ju = float(slope(x, u)[2]) / T.B_TRUE
                        d = float(np.hypot((x - k.zmin_x) / k.rstar_x, (u - k.zmin_u) / k.rstar_u))
                        tm = min((abs(x - xt) for xt, _ in ft), default=np.inf)
                        desc.append(dict(x=x, u=u, stable=abs(dg) < 1, d_half=d, Ju_over_B=ju, true_fp_within_0p01=tm < 0.01))
                    st = [q for q in desc if q["stable"]]
                    med, medt = float(np.median(xN)), float(np.median(xNt))
                    park_rows.append(dict(family=fam, seed=s, rep=rep, park_sur_median=med, share_at_median=float(np.mean(np.abs(xN - med) < 0.05)),
                                          park_true_median=medt, share_true_at_median=float(np.mean(np.abs(xNt - medt) < 0.05)),
                                          n_fp_sur=len(fs), n_fp_true=len(ft),
                                          stable_fp_sur=" ".join(f"{q['x']:+.3f}" for q in st),
                                          stable_fp_true=" ".join(f"{x:+.3f}" for x, dg in ft if abs(dg) < 1),
                                          fp_sur_d_half=" ".join(f"{q['d_half']:.1f}" for q in st),
                                          fp_sur_Ju_over_B=" ".join(f"{q['Ju_over_B']:.2f}" for q in st),
                                          fp_sur_matches_true=" ".join(str(q["true_fp_within_0p01"]) for q in st),
                                          kink_x=k.zmin_x, kink_u=k.zmin_u, rstar_x=k.rstar_x, rstar_u=k.rstar_u))
                    q = park_rows[-1]
                    print(f"{fam:15s} s{s} r{rep}: park sur {q['park_sur_median']:+.3f} ({q['share_at_median']:.2f}) true {q['park_true_median']:+.3f} | "
                          f"stable fp sur [{q['stable_fp_sur']}] d [{q['fp_sur_d_half']}] Ju/B [{q['fp_sur_Ju_over_B']}] true fp [{q['stable_fp_true']}] | "
                          f"sat {sat_rows[-1]['sat_frac']:.3f} share_unsat {sat_rows[-1]['share_unsat']:.3f} (split check {sat_rows[-1]['split_check']:.1e})", flush=True)
                pd.DataFrame(sat_rows).to_csv(os.path.join(OUT, "mlp_saturation.csv"), index=False)
                pd.DataFrame(park_rows).to_csv(os.path.join(OUT, "mlp_parking.csv"), index=False)
