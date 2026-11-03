"""t2_test.py - test of Theorem 2 (closed-loop multiplier condition for the DPC trap).

For every (family, repeat, seed): the reference controller (GRU trained through the TRUE plant, same repeat) is rolled out
through the surrogate from that repeat's 1000 training starts, N = 200. Along each rollout:
  m_k = J_x_hat(x_k, u_k) - J_u_hat(x_k, u_k) K_k,   K_k = -du_k/dx_k at fixed hidden state (the GRU's direct gain),
the exact sensitivity dx_N/dx_0 (forward mode through the full GRU + surrogate loop), the per-start costate of the
scalar recursion, and the per-start and batch gradient norms of the DPC loss through the surrogate.
Landing window of a start: steps k >= k_land, k_land = first k with |x_k| <= 0.2.
Outputs: t2_runs.csv (one row per run) and t2_starts.npz (per-start statistics).
Usage: python3 t2_test.py [families comma-separated] [--line]   (--line adds the straight line from DPC's own
controller to the reference controller, 21 points, LN MSE 3000 only)."""
import sys, time, os
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import t2core as T
import paths

OUT = os.environ.get("T2_OUT", paths.out("theory"))
os.makedirs(OUT, exist_ok=True)
X_LAND = 0.2
REPS = (0, 1, 2)


def make_fns(norm):
    def per_start(c, p, s0, x0):
        xs, us, hs = T.rollout_one(c, p, s0, norm, x0)
        sl = T.slopes(p, s0, norm)
        _, jx, ju = jax.vmap(sl)(xs[:-1], us)
        K = -jax.vmap(lambda h, x: jax.grad(lambda xx: T.gru_step(c, h, xx)[1])(x))(hs[:-1], xs[:-1])
        m = jx - ju * K
        sens = jax.jvp(lambda a: T.rollout_one(c, p, s0, norm, a)[0][-1], (x0,), (jnp.ones_like(x0),))[1]
        return xs, us, jx, ju, K, m, sens
    ps = jax.jit(jax.vmap(per_start, in_axes=(None, None, None, 0)))
    gb = jax.jit(jax.grad(lambda c, p, s0, X: T.dpc_loss(c, p, s0, norm, X)))
    gs = jax.jit(jax.vmap(lambda c, p, s0, x0: T.tree_norm(jax.grad(T.loss_one)(c, p, s0, norm, x0)),
                          in_axes=(None, None, None, 0)))
    lv = jax.jit(lambda c, p, s0, X: T.dpc_loss(c, p, s0, norm, X))
    return ps, gb, gs, lv


def max_subarray(a):
    """largest sum over contiguous segments (>= 0 when the empty segment is allowed)."""
    best = cur = 0.0
    for v in a:
        cur = max(0.0, cur + v)
        best = max(best, cur)
    return best


def costate(xs, us, K, m):
    """scalar costate lam_j = dl/dx_j of the per-start loss with memoryless multipliers m_k (Theorem 2a)."""
    N = len(us)
    lam = np.zeros(N + 1)
    lam[N] = 2 * T.Q_F * xs[N] / N
    for k in range(N - 1, -1, -1):
        lam[k] = (2 * T.Q_X * xs[k] - 2 * T.R_U * us[k] * K[k]) / N + m[k] * lam[k + 1]
    return lam


def start_stats(xs, us, jx, ju, K, m, sens):
    """per-start statistics, numpy arrays [B, ...]."""
    B, N = us.shape
    logm = np.log(np.abs(m) + 1e-300)
    out = {k: np.full(B, np.nan) for k in ("k_land", "S_seg", "S_land", "S_all", "share_exp", "lyap_exact",
                                           "gain_land_med", "jx_land_med", "K_land_med", "band_lo", "band_hi",
                                           "lam_max", "x_land_rms")}
    for b in range(B):
        near = np.abs(xs[b, :-1]) <= X_LAND
        kl = int(np.argmax(near)) if near.any() else N
        out["k_land"][b] = kl
        w = logm[b, kl:]
        out["S_seg"][b] = max_subarray(w) if len(w) else 0.0
        out["S_land"][b] = w.sum() if len(w) else 0.0
        out["S_all"][b] = logm[b].sum()
        out["share_exp"][b] = (np.abs(m[b, kl:]) > 1).mean() if len(w) else np.nan
        out["lyap_exact"][b] = np.log(np.abs(sens[b]) + 1e-300) / N
        if len(w):
            g, a, kk = ju[b, kl:] / T.B_TRUE, jx[b, kl:], K[b, kl:]
            out["gain_land_med"][b] = np.median(g)
            out["jx_land_med"][b] = np.median(a)
            out["K_land_med"][b] = np.median(kk)
            out["band_lo"][b] = np.median((a - 1) / (T.B_TRUE * kk))
            out["band_hi"][b] = np.median((a + 1) / (T.B_TRUE * kk))
            out["x_land_rms"][b] = np.sqrt(np.mean(xs[b, kl:] ** 2))
        out["lam_max"][b] = np.max(np.abs(costate(xs[b], us[b], K[b], m[b])))
    return out


def labels():
    R = pd.read_csv(T.DATA + "/dpc_results.csv")
    R["rep"] = R.rep.fillna(0).astype(int)
    o = R[R.model == "true plant (oracle)"][["seed", "rep", "cost_test_true"]].rename(columns={"cost_test_true": "oracle_true"})
    R = R.merge(o, on=["seed", "rep"])
    R["gap"] = R.cost_test_true / R.oracle_true - 1
    return R


def run(families, line=False):
    R = labels()
    rows, per_start = [], {}
    xt = T.test_starts()
    for fam in families:
        P, S0, norm, seeds, met = T.load_surrogate(fam)
        ps, gb, gs, lv = make_fns(norm)
        roll_test = jax.jit(jax.vmap(lambda c, p, s0, x0: T.rollout_one(c, p, s0, norm, x0)[:2], in_axes=(None, None, None, 0)))
        for rep in REPS:
            Co, cs = T.load_controllers("true_plant__oracle", rep)
            stem = T.FAMILIES[fam][2]
            Cm = T.load_controllers(stem, rep)[0] if stem else None
            for i, s in enumerate(seeds):
                t0 = time.time()
                p, s0, c = T.seed_slice(P, i), S0[i], T.seed_slice(Co, cs.index(s))
                X0 = T.training_starts(s, rep)
                arr = [np.asarray(a) for a in ps(c, p, s0, X0)]
                st = start_stats(*arr)
                gnorm = float(T.tree_norm(gb(c, p, s0, X0)))
                gper = np.asarray(gs(c, p, s0, X0))
                xs, us = roll_test(c, p, s0, xt)
                ref_on_model = T.true_lqr_cost(np.asarray(xs).T, np.asarray(us).T)
                r = R[(R.model == T.RESULTS_NAME[fam]) & (R.seed == s) & (R.rep == rep)].iloc[0]
                row = dict(family=fam, rep=rep, seed=s, gap=r.gap, own_on_model=r.cost_test_own,
                           ref_on_model=ref_on_model, peak_Ju=float(met["Ju_sup_rel"][i]),
                           rstar_x=float(met["rstar_x"][i]) if "rstar_x" in met else np.nan,
                           grad_batch=gnorm, grad_start_max=float(np.nanmax(gper)),
                           grad_top_share=float(np.nanmax(gper ** 2) / np.nansum(gper ** 2)),
                           S_seg_max=float(np.nanmax(st["S_seg"])), S_land_max=float(np.nanmax(st["S_land"])),
                           S_all_max=float(np.nanmax(st["S_all"])), lyap_exact_max=float(np.nanmax(st["lyap_exact"])),
                           lyap_scalar_max=float(np.nanmax(st["S_all"]) / T.N_HOR),
                           lam_max=float(np.nanmax(st["lam_max"])), share_exp_max=float(np.nanmax(st["share_exp"])),
                           share_exp_med=float(np.nanmedian(st["share_exp"])),
                           gain_land_med=float(np.nanmedian(st["gain_land_med"])),
                           gain_land_min=float(np.nanmin(st["gain_land_med"])),
                           jx_land_med=float(np.nanmedian(st["jx_land_med"])), K_land_med=float(np.nanmedian(st["K_land_med"])),
                           band_lo=float(np.nanmedian(st["band_lo"])), band_hi=float(np.nanmedian(st["band_hi"])),
                           x_land_rms=float(np.nanmedian(st["x_land_rms"])),
                           corr_lyap=float(np.corrcoef(st["S_all"] / T.N_HOR, st["lyap_exact"])[0, 1]))
                if line and Cm is not None:
                    cm = T.seed_slice(Cm, i)
                    seg, gn, ls = [], [], []
                    for a in np.linspace(0, 1, 21):
                        ca = jax.tree_util.tree_map(lambda u, v: (1 - a) * u + a * v, cm, c)
                        sa = start_stats(*[np.asarray(z) for z in ps(ca, p, s0, X0)])
                        seg.append(np.nanmax(sa["S_seg"])); gn.append(float(T.tree_norm(gb(ca, p, s0, X0))))
                        ls.append(float(lv(ca, p, s0, X0)))
                    row.update(line_S_seg_max=float(np.max(seg)), line_grad_max=float(np.max(gn)),
                               line_alpha_Smax=float(np.linspace(0, 1, 21)[int(np.argmax(seg))]),
                               line_alpha_gmax=float(np.linspace(0, 1, 21)[int(np.argmax(gn))]),
                               line_loss_monotone=bool(np.all(np.diff(ls) <= 1e-9)))
                    per_start[f"line|{fam}|{rep}|{s}"] = np.array([seg, gn, ls])
                rows.append(row)
                per_start[f"{fam}|{rep}|{s}"] = np.stack([st[k] for k in ("S_seg", "S_land", "S_all", "lyap_exact", "lam_max", "share_exp", "gain_land_med")] + [gper])
                print(f"{fam:16s} rep {rep} seed {s}: gap {100*r.gap:9.3g}% | S_seg {row['S_seg_max']:7.2f} S_all {row['S_all_max']:8.2f} "
                      f"| lyap exact {row['lyap_exact_max']:+.3f} scalar {row['lyap_scalar_max']:+.3f} | grad {gnorm:9.3g} lam {row['lam_max']:9.3g} "
                      f"| land gain {row['gain_land_med']:+7.2f} band [{row['band_lo']:.2f},{row['band_hi']:.1f}]"
                      + (f" | line S {row['line_S_seg_max']:.1f} g {row['line_grad_max']:.3g}" if 'line_grad_max' in row else "")
                      + f" | {time.time()-t0:.0f}s", flush=True)
        tag = "_".join(f.replace(" ", "") for f in families) + ("_line" if line else "")
        pd.DataFrame(rows).to_csv(f"{OUT}/t2_runs_{tag}.csv", index=False)
        np.savez_compressed(f"{OUT}/t2_starts_{tag}.npz", **per_start)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fams = sys.argv[1].split(",") if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else ["LN MSE 3000"]
    run(fams, line="--line" in sys.argv)
