"""Figure T1 for the theory notes: Theorem 2 tested on the saved DPC runs.
Panel 1: landing amplification Lambda* of the GRU reference controller, every run of the six families (30 each), trapped
runs filled dark red, others open dark blue. Panel 2: gradient norm of the surrogate DPC loss at the reference controller
and the largest over 8 neighbours (relative weight perturbation 1e-3) against Lambda*, LN MSE 3000. Panel 3: Lambda* of the
memoryless LQR-gain policy against Lambda* of the GRU reference, same surrogate and starts."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, GREY = "#8B1A1A", "#1B3A6B", "#888888"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
O = "/home/claude/work/out"
G = pd.concat([pd.read_csv(f"{O}/t2_runs_LNMSE3000.csv"),
               pd.read_csv(f"{O}/t2_runs_LNMSE12000_LNslopefrom0_LNtwo-stage_nonorm_staticnorm.csv")], ignore_index=True)
G["trap"] = (G.gap > 0.1) & (G.own_on_model / G.ref_on_model > 1.05)
G["failed"] = G.gap > 0.1
Bn = pd.read_csv(f"{O}/t2_neighbourhood.csv")
Mm = pd.read_csv(f"{O}/t2_memoryless.csv")
fams = ["LN MSE 3000", "LN MSE 12000", "LN slope from 0", "LN two-stage", "no norm", "static norm"]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0), gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
a = ax[0]
rng = np.random.default_rng(0)
for j, f in enumerate(fams):
    g = G[G.family == f]
    xj = j + rng.uniform(-0.22, 0.22, len(g))
    t = g.trap.values
    mild = (g.failed & ~g.trap).values
    a.scatter(xj[~t & ~mild], g.S_seg_max[~t & ~mild], s=22, facecolors="none", edgecolors=BLUE, linewidths=0.9)
    a.scatter(xj[mild], g.S_seg_max[mild], s=26, marker="s", facecolors="none", edgecolors=GREY, linewidths=0.9)
    a.scatter(xj[t], g.S_seg_max[t], s=26, color=RED)
a.set_xticks(range(len(fams)), [f.replace("LN ", "LN\n").replace(" from", "\nfrom") for f in fams])
a.set_ylabel(r"landing amplification $\Lambda^\star$ (natural log)")
a.set_yscale("symlog", linthresh=10)
a.set_ylim(-0.5, 400)
lnm = G[G.family == "LN MSE 3000"]
lo, hi = lnm[~lnm.trap].S_seg_max.max(), lnm[lnm.trap].S_seg_max.min()
a.axhspan(lo, hi, color=GREY, alpha=0.15, lw=0)
thr = np.sqrt(lo * hi)
a.axhline(thr, color=GREY, lw=0.9, ls="--")
a.text(5.45, hi * 1.08, f"threshold {thr:.1f}, fixed on LN MSE 3000\n(grey band: gap between its trapped and other runs)",
       ha="right", va="bottom", fontsize=7.2, color="#444444")
a.scatter([], [], color=RED, s=26, label="trapped (failed, own/ref on model > 1.05)")
a.scatter([], [], s=26, marker="s", facecolors="none", edgecolors=GREY, label="failed, not a trap")
a.scatter([], [], s=22, facecolors="none", edgecolors=BLUE, label="not failed")
a.legend(loc="upper right", frameon=False, fontsize=7.5)
a.set_title("GRU reference controller through each surrogate\n6 families x 10 seeds x 3 repeats", fontsize=9.5)

a = ax[1]
L = lnm.merge(Bn, on=["rep", "seed"])
for col, mk, lab in [("grad_batch", "o", "at the reference"), ("grad_nbhd_max", "^", "largest of 8 neighbours")]:
    t = L.trap.values
    a.scatter(L.S_seg_max[~t], L[col][~t], marker=mk, s=24, facecolors="none", edgecolors=BLUE, linewidths=0.9)
    a.scatter(L.S_seg_max[t], L[col][t], marker=mk, s=26, color=RED, label=lab)
xx = np.linspace(0, 170, 50)
a.plot(xx, np.exp(xx), color=GREY, lw=0.8, ls="--")
a.text(100, np.exp(100) * 1e-9, r"$e^{\Lambda^\star}$", color="#444444", fontsize=9, ha="left", va="top")
a.set_yscale("log"); a.set_ylim(3e-2, 1e60)
a.set_xlabel(r"$\Lambda^\star$"); a.set_ylabel(r"$\|\nabla_\theta L\|$ through the surrogate")
a.legend(loc="lower right", frameon=False, fontsize=7.5)
a.set_title("LN MSE 3000: gradient at and near the reference", fontsize=9.5)

a = ax[2]
M = G.merge(Mm[["family", "rep", "seed", "S_seg_max"]].rename(columns={"S_seg_max": "S_lin"}), on=["family", "rep", "seed"])
t = M.trap.values
a.scatter(M.S_seg_max[~t], M.S_lin[~t], s=20, facecolors="none", edgecolors=BLUE, linewidths=0.8)
a.scatter(M.S_seg_max[t], M.S_lin[t], s=24, color=RED)
a.plot([0, 170], [0, 170], color=GREY, lw=0.8, ls="--")
a.set_xscale("symlog", linthresh=10); a.set_yscale("symlog", linthresh=10)
a.set_xlim(-0.5, 300); a.set_ylim(-0.5, 300)
a.set_xlabel(r"$\Lambda^\star$, GRU reference (direct gain about 17 to 22)")
a.set_ylabel(r"$\Lambda^\star$, memoryless policy at the LQR gain 10.29")
a.set_title("Same surrogates and starts, two good policies", fontsize=9.5)
fig.tight_layout()
fig.savefig("/home/claude/work/theory/figures/t1_theorem2_test.png", dpi=200)
print("saved")
