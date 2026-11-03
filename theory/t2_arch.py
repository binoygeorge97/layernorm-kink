"""t2_arch.py - landing amplification Lambda* of a MEMORYLESS reference controller (MLP or linear), Task 1.1.

For every surrogate family x seed x repeat: the reference controller u = pi(x) (trained through the TRUE plant, same
repeat; dpc_core 'mlp' tree L0..Ln or 'linear' tree k, b, run with dpc_core.ctrl_step itself) is rolled through the
surrogate from that repeat's 1000 training starts (key 30000 + s + 1000 rep), N = 200. Along each rollout
    m_k = J_x_hat(x_k, u_k) - J_u_hat(x_k, u_k) K_k,   K_k = -d pi / dx (x_k), exact (no memory).
Landing window: k >= first k with |x_k| <= 0.2. Lambda(x0) = max contiguous-segment sum of log|m_k| on it (max_subarray),
Lambda* = max over the 1000 starts (the same statistic as t2_test.S_seg_max and t2_memoryless part A).
Exact check (memoryless policies only): sum_k log|m_k| = log|dx_N/dx_0| (forward-mode sensitivity), error ~1e-8.
Also: gradient norm of the surrogate DPC loss w.r.t. the controller parameters, and the policy gain at the landing.
Prediction rule (pre-registered threshold, NOT re-tuned): trapped iff Lambda* > 12.1.

Usage (from theory/):
  python t2_arch.py <controller root> [--tag NAME] [--families "LN MSE 3000,..."] [--reps 0,1,2]
      <controller root> = folder with controllers/true_plant__oracle[__repN].npz (e.g. ../data/dpc_out_mlp)
  python t2_arch.py --unit      linear controller at k = -K* = -10.2875, b = 0 (built here, no file): must reproduce
                                t2_memoryless.py part A run by run (out/theory/t2_memoryless.csv or the reference CSV).
Surrogates are read from T2_DATA (the GRU root). Output: $T2_OUT/t2_arch_<tag>.csv and t2_arch_<tag>_starts.npz."""
import sys, os, time, argparse
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
import paths
from t2_test import max_subarray, costate, X_LAND, OUT
import dpc_core as DC

THRESHOLD = 12.1          # fixed on LN MSE 3000 GRU runs (geometric midpoint of 8.6 and 17.0); do not re-tune
K_STAR = T.lqr_gain_true()          # 10.2875 at the reference plant


def pi(c, x):
    """scalar x -> scalar u, exactly dpc_core.ctrl_step (mlp_step / linear_step) on a batch of one."""
    return DC.ctrl_step(c, jnp.zeros((1, 0), jnp.asarray(x).dtype), jnp.reshape(x, (1, 1)))[1][0, 0]


def make_fns(norm):
    def roll(c, p, s0, x0):
        def step(x, _):
            u = pi(c, x)
            return T.F_point(p, x, u, s0, norm), (x, u)
        xN, (xs, us) = jax.lax.scan(step, x0, None, length=T.N_HOR)
        return jnp.append(xs, xN), us

    def per_start(c, p, s0, x0):
        xs, us = roll(c, p, s0, x0)
        _, jx, ju = jax.vmap(T.slopes(p, s0, norm))(xs[:-1], us)
        K = -jax.vmap(jax.grad(lambda x: pi(c, x)))(xs[:-1])
        sens = jax.jvp(lambda a: roll(c, p, s0, a)[0][-1], (x0,), (jnp.ones_like(x0),))[1]
        return xs, us, jx, ju, K, jx - ju * K, sens

    def loss(c, p, s0, X0):
        def one(x0):
            xs, us = roll(c, p, s0, x0)
            return (jnp.sum(T.Q_X * xs[:-1] ** 2 + T.R_U * us ** 2) + T.Q_F * xs[-1] ** 2) / T.N_HOR
        return jnp.mean(jax.vmap(one)(X0))
    return jax.jit(jax.vmap(per_start, in_axes=(None, None, None, 0))), jax.jit(jax.grad(loss))


def run_stats(xs, us, jx, ju, K, m, sens):
    B = xs.shape[0]
    logm = np.log(np.abs(m) + 1e-300)
    S_seg, S_all, lam, kl_all, K_land = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B), np.full(B, np.nan)
    for b in range(B):
        near = np.abs(xs[b, :-1]) <= X_LAND
        kl = int(np.argmax(near)) if near.any() else T.N_HOR
        kl_all[b] = kl
        S_seg[b] = max_subarray(logm[b, kl:])
        S_all[b] = logm[b].sum()
        lam[b] = np.max(np.abs(costate(xs[b], us[b], K[b], m[b])))
        if kl < T.N_HOR:
            K_land[b] = np.median(K[b, kl:])
    exact = np.log(np.abs(sens) + 1e-300)
    return dict(S_seg=S_seg, S_all=S_all, lam=lam, k_land=kl_all, K_land=K_land, fact_err=np.abs(S_all - exact))


def load_ref(root, rep):
    fn = DC.rep_file(os.path.join(root, "controllers"), "true plant (oracle)", rep)
    z = np.load(fn)
    c = DC._unflat({k[2:]: z[k] for k in z.files if k.startswith("c:")})
    return c, [int(s) for s in z["seeds"]]


def main(root, tag, families, reps, unit=False):
    rows, per = [], {}
    for fam in families:
        P, S0, norm, seeds, met = T.load_surrogate(fam)
        ps, g = make_fns(norm)
        for rep in reps:
            if unit:
                C, cs = None, seeds
                c_lin = dict(k=jnp.full((1, 1), -K_STAR / DC.LIN_SCALE), b=jnp.zeros(1))
            else:
                C, cs = load_ref(root, rep)
            for i, s in enumerate(seeds):
                t0 = time.time()
                c = c_lin if unit else T.seed_slice(C, cs.index(int(s)))
                p, s0, X0 = T.seed_slice(P, i), S0[i], T.training_starts(s, rep)
                arr = [np.asarray(a) for a in ps(c, p, s0, X0)]
                st = run_stats(*arr)
                gn = float(T.tree_norm(g(c, p, s0, X0)))
                xs = arr[0]
                row = dict(family=fam, rep=rep, seed=int(s), Lambda_star=float(st["S_seg"].max()),
                           S_all_max=float(st["S_all"].max()), factorisation_err=float(st["fact_err"].max()),
                           lam_max=float(st["lam"].max()), grad_norm=gn,
                           K_land_med=float(np.nanmedian(st["K_land"])), never_landed=int((st["k_land"] >= T.N_HOR).sum()),
                           diverged_share=float((np.abs(xs).max(1) > 10).mean()),
                           peak_Ju=float(met["Ju_sup_rel"][i]), predicted_trap=bool(st["S_seg"].max() > THRESHOLD))
                rows.append(row)
                per[f"{fam}|{rep}|{s}"] = np.stack([st["S_seg"], st["S_all"], st["k_land"], st["K_land"]])
                print(f"{fam:16s} rep {rep} seed {s}: Lambda* {row['Lambda_star']:7.2f} "
                      f"{'TRAP' if row['predicted_trap'] else '    '} | fact err {row['factorisation_err']:.1e} | "
                      f"grad {gn:9.3g} | K_land {row['K_land_med']:6.2f} | {time.time() - t0:.1f}s", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(f"{OUT}/t2_arch_{tag}.csv", index=False)
    np.savez_compressed(f"{OUT}/t2_arch_{tag}_starts.npz", **per)
    print(f"-> {OUT}/t2_arch_{tag}.csv | predicted trapped: {int(D.predicted_trap.sum())} of {len(D)} | "
          f"max Lambda* {D.Lambda_star.max():.2f} | max factorisation error {D.factorisation_err.max():.1e}")
    return D


def unit_check(D):
    """compare with t2_memoryless part A (same policy, same starts): per-run S_seg_max."""
    cand = [os.path.join(OUT, "t2_memoryless.csv"), os.path.join(paths.REF, "theory", "t2_memoryless_Sseg.csv")]
    for fn in cand:
        if not os.path.exists(fn):
            continue
        M = pd.read_csv(fn)
        col = "S_seg_max" if "S_seg_max" in M else "S_seg_max_linear_LQR"
        J = D.merge(M[["family", "rep", "seed", col]], on=["family", "rep", "seed"])
        err = (J.Lambda_star - J[col]).abs()
        # 1e-8: dpc_core stores k = -K*/10 and evaluates (x k) 10 / 10, t2_memoryless evaluates -K* x / 10; the policies differ
        # by <= 5e-15 per step, which moves Lambda* by <= 2.3e-10 relative (2.0e-9 absolute, 1 run of 180, 2026-11-03).
        # The reference CSV is rounded to 3 decimals.
        tol = 1e-8 if col == "S_seg_max" else 5e-4
        print(f"unit check vs {fn}: {len(J)} runs, max |diff| {err.max():.2e} (tolerance {tol:g}) -> "
              + ("PASS" if len(J) == len(D) and err.max() <= tol else "FAIL"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--tag")
    ap.add_argument("--families", default=",".join(T.FAMILIES))
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--unit", action="store_true")
    a = ap.parse_args()
    fams = a.families.split(",")
    reps = [int(r) for r in a.reps.split(",")]
    if a.unit:
        unit_check(main(None, a.tag or "unit_linear_Kstar", fams, reps, unit=True))
    else:
        assert a.root, "controller root required (folder with controllers/true_plant__oracle*.npz)"
        main(a.root, a.tag or os.path.basename(os.path.normpath(a.root)), fams, reps)
