"""Q6c (2026-11-04): Part 1 replication on held-out seeds 10-29 scored against prereg/PREREG_heldout.md (23b5835).
Inputs: data/dpc_out_s10_29/surrogates.csv (kink metrics), the held-out LQR study
  LK_DATA=data/dpc_out_s10_29 LK_SEEDS=10,...,29 LK_FAMILIES="LN MSE 3000;LN two-stage;static norm" LK_OUT=out/heldout \
      python companion/run_lqr_study.py        -> out/heldout/companion/lqr_study.csv
and prereg/PREREG_heldout_intervals.csv. Rule CLAUDE.md 1.3. Families not trained: "not trained".
H1: rate inside its two-stage 95 % PI (held if at most 2 scored cells outside); H2: LN MSE 3000 fixed failed > static norm
and its cell inside; H3: every LN centre within 0.1 of (0, 0).
Also: seed-bootstrap 95 % CI of each held-out rate (10 000 resamples of the 20 seeds, rng 20261104).
Outputs out/heldout/outcomes.csv (for paper/make_tables.py) and a printed report."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
R_ = lambda *a: os.path.join(paths.REPO, *a)
LQR = R_("out", "heldout", "companion", "lqr_study.csv")
SUR = R_("data", "dpc_out_s10_29", "surrogates.csv")
PI = pd.read_csv(R_("prereg", "PREREG_heldout_intervals.csv"))
SURNAME = {"LN MSE 3000": "LN, MSE (3000)", "LN MSE 12000": "LN, MSE (12000)", "LN slope from step 0": "LN, J_x + J_u (3000)",
           "LN two-stage": "LN, two-stage", "no norm": "no norm, MSE", "static norm": "static norm, MSE"}

T = pd.read_csv(LQR)
T = T[(T.x0 - T.r).abs() >= 0.5].copy()
T["failed"] = T.diverged | (T.gap > 0.1); T["catastrophic"] = T.diverged | (T.gap > 1.0)
seeds = sorted(T.seed.unique())
rng = np.random.default_rng(20261104); idx = rng.integers(0, len(seeds), (10_000, len(seeds)))
rows = []
for _, p in PI.iterrows():
    g = T[(T.model == p.family) & (T.controller == p.controller)]
    if not len(g):
        rows.append(dict(family=p.family, controller=p.controller, metric=p.metric, trained=False)); continue
    k = g.groupby("seed")[p.metric].sum().reindex(seeds).values; n = g.groupby("seed").size().reindex(seeds).values
    rate = 100 * k.sum() / n.sum(); b = 100 * k[idx].sum(1) / n[idx].sum(1)
    rows.append(dict(family=p.family, controller=p.controller, metric=p.metric, trained=True, n=int(n.sum()), pct=rate,
                     ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), seeds_0_9_pct=p.seeds_0_9_pct,
                     pi_lo_pct=p.pi_lo_pct, pi_hi_pct=p.pi_hi_pct,
                     inside=bool(p.pi_lo_pct - 1e-9 <= rate <= p.pi_hi_pct + 1e-9)))
O = pd.DataFrame(rows)
os.makedirs(R_("out", "heldout"), exist_ok=True)
O[O.trained].to_csv(R_("out", "heldout", "outcomes.csv"), index=False)
sc = O[O.trained].copy(); sc["inside"] = sc.inside.astype(bool)
n_out = int((~sc.inside).sum())
h1 = "held" if n_out <= 2 else "failed"
lnf = sc[(sc.family == "LN MSE 3000") & (sc.controller == "LQR fixed") & (sc.metric == "failed")]
stf = sc[(sc.family == "static norm") & (sc.controller == "LQR fixed") & (sc.metric == "failed")]
h2 = ("held" if (len(lnf) and len(stf) and lnf.pct.iloc[0] > stf.pct.iloc[0] and bool(lnf.inside.iloc[0])) else
      "failed" if len(lnf) and len(stf) else "not testable")
S = pd.read_csv(SUR); S = S[S.model.str.startswith("LN")]
off = S[["zmin_x", "zmin_u"]].abs().max(axis=1)
h3 = "held" if (off <= 0.1).all() else "failed"
pd.set_option("display.width", 220)
print(sc[["family", "controller", "metric", "n", "pct", "ci_lo", "ci_hi", "seeds_0_9_pct", "pi_lo_pct", "pi_hi_pct", "inside"]].round(2).to_string(index=False))
print(f"\nH1: {h1} ({n_out} of {len(sc)} scored cells outside: "
      + "; ".join(f"{r.family} {r.controller} {r.metric} {r.pct:.1f} vs [{r.pi_lo_pct:.1f}, {r.pi_hi_pct:.1f}]" for _, r in sc[~sc.inside].iterrows()) + ")")
print(f"H2: {h2} (LN MSE 3000 {lnf.pct.iloc[0] if len(lnf) else np.nan:.1f} % vs static norm {stf.pct.iloc[0] if len(stf) else np.nan:.1f} %)")
print(f"H3: {h3} ({int((off > 0.1).sum())} of {len(off)} LN centres outside 0.1: "
      + "; ".join(f"{m} seed {s} ({x:+.3f}, {u:+.3f})" for m, s, x, u in S[off > 0.1][["model", "seed", "zmin_x", "zmin_u"]].values) + ")")
print("\nkink geometry (median over seeds):\n", S.groupby("model")[["rstar_x", "rstar_u", "Ju_sup_rel"]].median().round(3).to_string())
print("not trained:", sorted(set(O[~O.trained].family)))
