"""Q6b (2026-11-04): pre-registration of the Part 1 (LQR) replication on held-out seeds 10-29, written BEFORE any surrogate
of seeds 10-29 exists. Prediction interval for a 20-seed sample: resample 20 seeds with replacement from seeds 0-9 (a seed
carries all its 88 runs with |x0 - r| >= 0.5), pooled rate over the resampled runs, 10 000 resamples, rng 20261104;
95 % percentile interval. Source: companion/lqr_study.csv (B = 0.01, rule CLAUDE.md 1.3).
Two-stage version (primary): bootstrap seeds 0-9, then draw the 20 seeds from that resample (adds the uncertainty of the
10-seed rates; the single-stage draw alone is narrower than the CI of the mean). Prints the table for
prereg/PREREG_heldout.md (Part 1) and writes prereg/PREREG_heldout_intervals.csv."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
T = pd.read_csv(os.path.join(paths.REPO, "companion", "lqr_study.csv"))
T = T[(T.x0 - T.r).abs() >= 0.5].copy()
T["failed"] = T.diverged | (T.gap > 0.1); T["catastrophic"] = T.diverged | (T.gap > 1.0)
rng = np.random.default_rng(20261104); NB, NS = 10_000, 20
idx1 = rng.integers(0, 10, (NB, NS))                        # single stage: 20 seeds drawn from the fixed seeds 0-9
boot = rng.integers(0, 10, (NB, 10))                        # two stage: bootstrap seeds 0-9 (population uncertainty),
idx2 = np.take_along_axis(boot, rng.integers(0, 10, (NB, NS)), 1)   # then draw the 20 new seeds from that resample
FAM = ["LN MSE 3000", "LN MSE 12000", "LN slope from step 0", "LN two-stage", "no norm", "static norm"]
rows = []
for f in FAM:
    for c in ("LQR fixed", "LQR sched"):
        g = T[(T.model == f) & (T.controller == c)]
        for m in ("failed", "catastrophic"):
            k = g.groupby("seed")[m].sum().reindex(range(10)).values; n = g.groupby("seed").size().reindex(range(10)).values
            r1, r2 = (k[i].sum(1) / n[i].sum(1) for i in (idx1, idx2))
            rb = k[boot].sum(1) / n[boot].sum(1)
            rows.append(dict(family=f, controller=c, metric=m, seeds_0_9_pct=100 * k.sum() / n.sum(),
                             pi_lo_pct=100 * np.percentile(r2, 2.5), pi_hi_pct=100 * np.percentile(r2, 97.5),
                             single_lo_pct=100 * np.percentile(r1, 2.5), single_hi_pct=100 * np.percentile(r1, 97.5),
                             ci_mean_lo_pct=100 * np.percentile(rb, 2.5), ci_mean_hi_pct=100 * np.percentile(rb, 97.5)))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(paths.REPO, "prereg", "PREREG_heldout_intervals.csv"), index=False)
lines = ["| family | controller | metric | seeds 0-9 (%) | 95 % PI, two-stage (PRIMARY) | single-stage (20 from fixed 0-9) | 95 % CI of the mean (10 seeds) |", "|---|---|---|---|---|---|---|"]
lines += [f"| {r.family} | {r.controller} | {r.metric} | {r.seeds_0_9_pct:.1f} | [{r.pi_lo_pct:.1f}, {r.pi_hi_pct:.1f}] | [{r.single_lo_pct:.1f}, {r.single_hi_pct:.1f}] | [{r.ci_mean_lo_pct:.1f}, {r.ci_mean_hi_pct:.1f}] |" for _, r in P.iterrows()]
print("\n".join(lines))
