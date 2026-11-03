"""Swap test figures: which slope error (J_x or J_u) drives the LQR and DPC failures."""
import sys; sys.path.insert(0, "/home/claude/kinkwork")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm; from matplotlib.patches import Ellipse
import level1_core as L, phase_core as PC
OUT = "/home/claude/data/dpc_out"; ST = "/home/claude/story"; plant = L.Plant(); seeds = list(range(10)); A, B = plant.A, plant.B
LN = PC.Model.surrogate(OUT, "MSE (3000)", seeds, "LN MSE"); truth = PC.Model.true_plant(plant, seeds)
R0, X0 = 0.0, 1.0; WIN = (-0.6, 2.4, -11, 11)
VARIANTS = [("surrogate A^ and B^ (what LQR normally gets)", lambda ah, bh: (ah, bh)), ("TRUE A, surrogate B^", lambda ah, bh: (np.full_like(ah, A), bh)),
            ("surrogate A^, TRUE B", lambda ah, bh: (ah, np.full_like(bh, B))), ("TRUE A and B (only u^ from the surrogate)", lambda ah, bh: (np.full_like(ah, A), np.full_like(bh, B)))]
def sim(uh, K, r, x0):
    x = x0.copy(); xs, us = [x], []
    for _ in range(200):
        u = np.clip(uh - K * (x - r), -10, 10); x = A * x + B * u; xs.append(x); us.append(u)
    return np.stack(xs), np.stack(us)
# ---------- Figure W1: one seed, four designs, J_u map (top) and J_x map (middle), error vs time (bottom) ----------
for s in (9, 8):
    i = seeds.index(s); r = np.full((10, 1), R0); x0 = np.full((10, 1), X0)
    uh, nroot, ah, bh = PC.equilibrium(LN, r)
    X, U = np.meshgrid(np.linspace(*WIN[:2], 161), np.linspace(*WIN[2:], 161))
    _, jx, ju = LN.seed_view(i).fj(X.ravel()[None], U.ravel()[None]); GU = ju[0].reshape(X.shape) / B; GX = jx[0].reshape(X.shape) - A
    zx, zu, rx, ru = LN.kink()
    fig, ax = plt.subplots(3, 4, figsize=(19, 11), gridspec_kw=dict(height_ratios=[1.3, 1.3, 0.9]))
    nu = TwoSlopeNorm(vmin=-3, vcenter=1, vmax=5); nx = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
    for c, (name, f) in enumerate(VARIANTS):
        a_, b_ = f(ah, bh); K = L.dare_K(a_, b_); xs, us = sim(uh, K, r, x0)
        xo, uo = sim(np.full_like(uh, 0.0), np.full_like(K, L.dare_K(A, B)), r, x0)
        for row, (G, nrm, cm, lab) in enumerate(((GU, nu, "RdBu_r", "input slope B^/B (1 = right, blue = wrong sign)"), (GX, nx, "PuOr_r", "state slope A^ - A (0 = right; the stability margin is A - 1 = 0.03)"))):
            a = ax[row, c]; im = a.pcolormesh(X, U, np.clip(G, nrm.vmin, nrm.vmax), cmap=cm, norm=nrm, shading="auto", rasterized=True)
            if row == 0 and (G < 0).any(): a.contour(X, U, G, levels=[0], colors="k", linewidths=0.6)
            if np.isfinite(rx[i]): a.add_patch(Ellipse((zx[i], zu[i]), 2 * rx[i], 2 * ru[i], fill=False, ls="--", lw=1, color="k"))
            a.plot(xo[:-1, i, 0], uo[:, i, 0], color="grey", lw=1.2); a.plot(np.clip(xs[:-1, i, 0], -99, 99), us[:, i, 0], "-o", ms=2, lw=1.1, color="k")
            a.plot(X0, us[0, i, 0], "s", ms=8, color="white", mec="k", zorder=6); a.plot(R0, 0, "*", ms=13, color="gold", mec="k", zorder=6)
            a.set_xlim(*WIN[:2]); a.set_ylim(*WIN[2:]); a.grid(False); a.set_xlabel("x") if row == 1 else None
            if c == 0: a.set_ylabel("u   |   " + ("over the J_u map" if row == 0 else "over the J_x map"), fontsize=8.5)
            if c == 3: fig.colorbar(im, ax=ax[row, :], fraction=0.012, pad=0.01).set_label(lab, fontsize=8)
        used = "A^ = %.3f, B^/B = %+.1f" % (a_[i, 0], b_[i, 0] / B)
        ax[0, c].set_title(f"design {c + 1}: {name}\nslopes fed to the Riccati equation: {used}\n-> K = {K[i, 0]:+.2f}   (true slopes give {L.dare_K(A, B):.2f})", fontsize=8.5)
        e = ax[2, c]; t = np.arange(200); e.plot(t, xo[:-1, i, 0], color="grey", lw=1.2, label="ideal"); e.plot(t, np.clip(xs[:-1, i, 0], -5, 5), color="#D55E00", lw=1.6, label="this design")
        e.axhline(0, color="k", lw=0.5); e.set_ylim(-1.5, 1.5); e.set_xlabel("step k"); e.grid(alpha=0.3); e.legend(fontsize=7)
        fin = abs(xs[-20:, i, 0]).mean(); e.set_title(f"x_k on the true plant; |x| at the end = {fin:.3g}" + ("  (DIVERGED)" if abs(xs[:, i, 0]).max() > 10 else ""), fontsize=8.5)
        if c == 0: e.set_ylabel("x_k")
    fig.suptitle(f"SWAP TEST on one surrogate (LayerNorm MSE, seed {s}), x0 = {X0:g}, r = {R0:g}: the same fixed LQR recipe, but one or both of the surrogate's slopes at the setpoint\n"
                 "replaced by the true value before computing K. Grey = ideal path. Top row shows the surrogate's input-slope map, middle row its state-slope map (same path drawn on both).", fontsize=10)
    fig.savefig(f"{ST}/W1_swap_lqr_seed{s}.png", dpi=120, bbox_inches="tight")
# ---------- Figure W2: failure rates per surrogate x design, and DPC gradient norms per variant ----------
rows = []
refs, starts = PC.R_EVAL, PC.X0_TRACK; rr, xx = np.meshgrid(refs, starts, indexing="ij"); r = np.tile(rr.ravel(), (10, 1)); x0 = np.tile(xx.ravel(), (10, 1)); far = np.abs(x0 - r) >= 0.5
xo, uo, _, _ = PC.run_lqr(truth, truth, r, x0, "fixed"); Jo = PC.track_cost(xo, uo, r, plant)
SUR = [("LN MSE 3000", "MSE (3000)"), ("LN slope from 0", "J_x + J_u (3000)"), ("LN MSE 12000", "MSE (12000)"), ("LN two-stage", "J_x + J_u (two-stage)"), ("static norm", "static norm, MSE")]
for name, sur in SUR:
    m = PC.Model.surrogate(OUT, sur, seeds, name); uh, nroot, ah, bh = PC.equilibrium(m, r)
    for vn, f in VARIANTS:
        a_, b_ = f(ah, bh); xs, us = sim(uh, L.dare_K(a_, b_), r, x0); met = PC.track_metrics(xs, us, r, plant, Jo)
        rows.append(dict(surrogate=name, design=vn, failed=100 * (met["diverged"] | (met["gap"] > 0.1))[far].mean(), catastrophic=100 * (met["diverged"] | (met["gap"] > 1.0))[far].mean()))
F = pd.DataFrame(rows); F.to_csv(f"{ST}/swap_lqr_rates.csv", index=False)
D = pd.read_csv(f"{ST}/attribution_dpc.csv")
fig, ax = plt.subplots(1, 2, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1.1, 1]))
a = ax[0]; w = 0.2; cols = ["#333333", "#7570b3", "#D55E00", "#009E73"]
for j, (vn, _) in enumerate(VARIANTS):
    g = F[F.design == vn]; a.bar(np.arange(len(SUR)) + (j - 1.5) * w, g.failed.values, w, color=cols[j], label=vn)
a.set_xticks(range(len(SUR))); a.set_xticklabels([n for n, _ in SUR], fontsize=8.5); a.set_ylabel("% of LQR runs failed (gap > 10 % or diverged)")
a.set_title("Fixed LQR, 11 setpoints x 9 starts x 10 seeds per surrogate: which slope must be right?", fontsize=9.5); a.legend(fontsize=7.5); a.grid(axis="y", alpha=0.3)
b = ax[1]; labels = [f"{'LN MSE' if s_ == 'MSE (3000)' else 'slope0' if 'J_u (3000)' in s_ else 'MSE12k' if '12000' in s_ else 'two-st'} s{sd}" for s_, sd in zip(D.surrogate, D.seed)]
xi = np.arange(len(D))
for j, (col, lab, c_) in enumerate([("grad | surrogate J_x and J_u", "surrogate J_x and J_u", "#333333"), ("grad | true J_x, surrogate J_u", "TRUE J_x, surrogate J_u", "#7570b3"), ("grad | surrogate J_x, true J_u", "surrogate J_x, TRUE J_u", "#D55E00")]):
    b.bar(xi + (j - 1) * 0.27, D[col].values, 0.27, color=c_, label=lab)
b.set_yscale("log"); b.set_ylim(1e-2, 1e37); b.axhline(4, color="k", lw=0.7, ls=":"); b.text(-0.4, 6, "healthy level (< 4)", fontsize=7.5)
b.set_xticks(xi); b.set_xticklabels(labels, rotation=45, ha="right", fontsize=8); b.set_ylabel("||gradient of the DPC loss|| at the GOOD controller (log)")
for k_, br in enumerate(D.broke):
    if br: b.axvspan(k_ - 0.5, k_ + 0.5, color="#fde0dd", zorder=0)
b.set_title("DPC: the training gradient at the good controller, with one slope made true inside backprop\n(pink = seeds whose DPC run failed)", fontsize=9.5); b.legend(fontsize=7.5, loc="upper right"); b.grid(axis="y", alpha=0.3)
fig.suptitle("SWAP TEST summary. Left: LQR fails because of B^ (fixing A^ alone changes nothing; fixing B^ alone halves the failures). Right: the DPC gradient explosion is entirely the J_u slope.", fontsize=10)
fig.tight_layout(); fig.savefig(f"{ST}/W2_swap_summary.png", dpi=130, bbox_inches="tight"); print("ok")
