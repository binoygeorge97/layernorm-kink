"""LaTeX tables for the Theorem 2 test, generated from the result files (no hand transcription)."""
import numpy as np, pandas as pd
O = "out"
G = pd.concat([pd.read_csv(f"{O}/t2_runs_LNMSE3000.csv"), pd.read_csv(f"{O}/t2_runs_LNMSE12000_LNslopefrom0_LNtwo-stage_nonorm_staticnorm.csv")], ignore_index=True)
G["ratio"] = G.own_on_model / G.ref_on_model
G["failed"] = G.gap > 0.1
G["trap"] = G.failed & (G.ratio > 1.05)
lnm = G[G.family == "LN MSE 3000"]
thr = np.sqrt(lnm[~lnm.trap].S_seg_max.max() * lnm[lnm.trap].S_seg_max.min())
SW = pd.read_csv(f"{O}/t2_swap.csv"); NB = pd.read_csv(f"{O}/t2_neighbourhood.csv")
M = pd.read_csv(f"{O}/t2_memoryless.csv")
L = lnm.merge(SW, on=["rep", "seed"]).merge(NB, on=["rep", "seed"]).merge(
    M[M.family == "LN MSE 3000"][["rep", "seed", "S_seg_max"]].rename(columns={"S_seg_max": "S_lin"}), on=["rep", "seed"])

def sci(v):
    if v < 1e3:
        return f"{v:.2g}" if v < 10 else f"{v:.0f}"
    e = int(np.floor(np.log10(v))); m = v / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"

def gap(v):
    return f"{100*v:.1f}" if v < 10 else f"{100*v:.0f}"

lines = []
for _, r in L.sort_values(["rep", "seed"]).iterrows():
    tr = "yes" if r.trap else ("mism." if r.failed else "no")
    lines.append(f"{r.rep} & {r.seed} & {gap(r.gap)} & {r.ratio:.2f} & {tr} & {r.S_seg_max:.1f} & {r.true_Ju:.1f} & {r.true_Jx:.1f} & "
                 f"{r.S_lin:.1f} & {sci(r.grad_batch)} & {sci(r.grad_nbhd_max)} & {r.lyap_exact_max:+.3f} \\\\")
open("theory/tab_lnmse.tex", "w").write("\n".join(lines) + "\n")

rows = []
for f in ["LN MSE 3000", "LN MSE 12000", "LN slope from 0", "LN two-stage", "no norm", "static norm"]:
    g = G[G.family == f]; m = M[M.family == f]
    flag = g.S_seg_max > thr
    tp, fp, fn = (flag & g.trap).sum(), (flag & ~g.trap).sum(), (~flag & g.trap).sum()
    gtr = (g.grad_batch > 1e3)
    rows.append(f"{f} & {g.failed.sum()} & {g.trap.sum()} & {tp} & {fn} & {fp} & {g[~g.trap].S_seg_max.max():.1f} & "
                f"{(gtr & g.trap).sum()} & {(gtr & ~g.trap).sum()} & {m.S_seg_max.max():.1f} \\\\")
open("theory/tab_families.tex", "w").write("\n".join(rows) + "\n")
print(f"threshold {thr:.2f}")
print(open("theory/tab_families.tex").read())
print(open("theory/tab_lnmse.tex").read())
