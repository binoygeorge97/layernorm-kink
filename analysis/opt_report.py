"""Task 3.5: optimiser baselines - from-scratch table (Colab results in dpc_out) and warm-start table (local harness),
scored against prereg/PREREG_opt.md.

From scratch: rows plain (LN MSE 3000), clip 3, unit, global clip 3, mom, batch 100, lr 1e-4, N <= 50, N <= 100, and the
two-stage surrogate with the plain optimiser (reference row); columns failed / trapped of 30, median gap, best-seed gap,
wall time; per target seed (1, 2, 8, 9) rescued = not failed in >= 2 of 3 repeats; paired Wilcoxon vs plain.
Warm start: out/opt/warmstart_<setting>.csv; expelled = true-plant cost > 1.10 x epoch-0 value at any logged epoch 10-100.
Frozen 2026-11-04 (Q3) before any Run 4 output was read: score() gives the outcomes of PREREG_opt P1-P5 and U1 (c41967c)
and applies the convergence condition for lr 1e-4 (prereg/ADDENDUM_2026-11-04c_run4_convergence.md, 84ecb3a): "LN, MSE
(3000), lr 1e-4" is converged only if each healthy LN MSE seed 0, 3, 4, 6, 7 has gap <= 10 % in >= 2 of 3 repeats; if not,
P4 is "not testable: not converged". Wall time = median train_s_per_seed over the 30 runs (s per seed, as logged).
Usage: python analysis/opt_report.py [dpc_out root]   -> out/opt/*.csv, figure out/opt/fig_warmstart.png, printed scoring"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from common import runs, summary, paired, md  # noqa: E402

ROWS = [("plain", "LN, MSE (3000)"), ("clip 3", "LN, MSE (3000), clipped grads"), ("unit", "LN, MSE (3000), unit grads"),
        ("global clip 3", "LN, MSE (3000), global clip 3"), ("mom", "LN, MSE (3000), mom"),
        ("batch 100", "LN, MSE (3000), batch 100"), ("lr 1e-4", "LN, MSE (3000), lr 1e-4"),
        ("N <= 50", "LN, MSE (3000), N <= 50"), ("N <= 100", "LN, MSE (3000), N <= 100"),
        ("two-stage, plain (reference)", "LN, two-stage")]
TARGET = [1, 2, 8, 9]
SETTINGS = ["plain", "clip3", "unit", "global3", "mom", "batch100", "lr1e-4"]
BROKEN = {("MSE (3000)", 1), ("MSE (3000)", 2), ("MSE (3000)", 8), ("MSE (3000)", 9), ("J_x + J_u (3000)", 2),
          ("J_x + J_u (3000)", 9)}


def from_scratch(root):
    R = runs(root)
    T = summary(R, [m for _, m in ROWS])
    T.insert(0, "setting", [s for s, _ in ROWS])
    plain = R[R.model == "LN, MSE (3000)"]
    resc, pw = [], []
    for s, m in ROWS:
        g = R[R.model == m]
        if not len(g):
            resc.append(dict(setting=s)), pw.append(dict(setting=s))
            continue
        nf = g[g.seed.isin(TARGET)].groupby("seed").failed.sum()
        nr = g[g.seed.isin(TARGET)].groupby("seed").size()
        resc.append(dict(setting=s, **{f"seed {t}": ("rescued" if nf.get(t, 0) <= nr.get(t, 0) - 2 else "not")
                                       if nr.get(t, 0) == 3 else f"{nr.get(t, 0)} reps" for t in TARGET},
                         rescued=int(sum((nr.get(t, 0) == 3) and (nf.get(t, 0) <= 1) for t in TARGET))))
        pw.append(dict(setting=s, **paired(g, plain)))
    return T, pd.DataFrame(resc), pd.DataFrame(pw)


HEALTHY_LN = [0, 3, 4, 6, 7]


def score(root):
    """PREREG_opt P1-P5, U1 and the lr 1e-4 convergence rule. Returns a DataFrame (id, outcome, numbers)."""
    R = runs(root)
    def resc(model):
        g = R[(R.model == model) & R.seed.isin(TARGET)]
        n = g.groupby("seed").size(); f = g.groupby("seed").failed.sum()
        complete = all(n.get(s, 0) == 3 for s in TARGET)
        return int(sum((n.get(s, 0) == 3) and (f.get(s, 0) <= 1) for s in TARGET)), complete, \
            ", ".join(f"s{s}: {int(f.get(s, 0))}/{int(n.get(s, 0))} failed" for s in TARGET)
    out = []
    def add(i, model, cond, desc):
        k, complete, det = resc(model)
        if not complete:
            out.append(dict(id=i, outcome="not available (incomplete: needs 3 repeats of seeds 1, 2, 8, 9)", numbers=det)); return
        out.append(dict(id=i, outcome="held" if cond(k) else "failed", numbers=f"{k} of 4 target seeds rescued ({desc}); {det}"))
    add("P1", "LN, MSE (3000), clipped grads", lambda k: k >= 3, "predicted >= 3")
    add("P2a", "LN, MSE (3000), global clip 3", lambda k: k <= 1, "predicted <= 1")
    add("P3a", "LN, MSE (3000), mom", lambda k: k >= 3, "predicted >= 3")
    g = R[(R.model == "LN, MSE (3000), lr 1e-4") & R.seed.isin(HEALTHY_LN)]
    n = g.groupby("seed").size(); ok = g.groupby("seed").apply(lambda h: int((h.gap <= 0.10).sum()))
    conv_complete = all(n.get(s, 0) == 3 for s in HEALTHY_LN)
    converged = conv_complete and all(ok.get(s, 0) >= 2 for s in HEALTHY_LN)
    cdet = (", ".join(f"s{s}: {int(ok.get(s, 0))}/{int(n.get(s, 0))} <= 10 %" for s in HEALTHY_LN)
            + f"; healthy runs <= 10 %: {int((g.gap <= 0.10).sum())} of {len(g)}; gaps % "
            + " ".join(f"s{r.seed}r{r.rep} {100 * r.gap:.1f}" for _, r in g.sort_values(['seed', 'rep']).iterrows()))
    out.append(dict(id="P4 convergence (84ecb3a)", outcome=("converged" if converged else
               ("not converged" if conv_complete else "not available (incomplete)")), numbers=cdet))
    if converged:
        add("P4", "LN, MSE (3000), lr 1e-4", lambda k: k <= 1, "predicted <= 1")
    else:
        k, complete, det = resc("LN, MSE (3000), lr 1e-4")
        t4 = R[(R.model == "LN, MSE (3000), lr 1e-4") & R.seed.isin(TARGET)].sort_values(["seed", "rep"])
        out.append(dict(id="P4", outcome="not testable: not converged" if conv_complete else "not available (incomplete)",
                        numbers=f"{det}; target gaps % " + " ".join(f"s{r.seed}r{r.rep} {100 * r.gap:.1f}" for _, r in t4.iterrows())))
    g50 = R[R.model == "LN, MSE (3000), N <= 50"]; ts = R[R.model == "LN, two-stage"]
    if len(g50) == 30:
        out.append(dict(id="P5a", outcome="held" if g50.trapped.sum() <= 3 else "failed",
                        numbers=f"{int(g50.trapped.sum())} of 30 trapped (predicted <= 3; plain 12/30)"))
        m50, mts = g50.gap.median(), ts.gap.median()
        out.append(dict(id="P5b", outcome="held" if m50 > mts else "failed",
                        numbers=f"median gap N <= 50 {100 * m50:.2f} % vs two-stage plain {100 * mts:.2f} % (n = {len(g50)}, {len(ts)})"))
    else:
        out += [dict(id="P5a", outcome="not available (incomplete)", numbers=f"{len(g50)} runs"),
                dict(id="P5b", outcome="not available (incomplete)", numbers=f"{len(g50)} runs")]
    gb = R[R.model == "LN, MSE (3000), batch 100"]
    out.append(dict(id="U1", outcome="reported (no prediction)",
                    numbers=f"batch 100: {int(gb.failed.sum())} failed, {int(gb.trapped.sum())} trapped of {len(gb)}; "
                            f"target seeds rescued {resc('LN, MSE (3000), batch 100')[0]} of 4"))
    return pd.DataFrame(out)


def warm_start():
    rows, frames = [], []
    for st in SETTINGS:
        fn = paths.out("opt", f"warmstart_{st}.csv")
        if not os.path.exists(fn):
            continue
        W = pd.read_csv(fn)
        frames.append(W)
        for (sur, s), g in W.groupby(["sur", "seed"], sort=False):
            c0 = g[g.epoch == 0].loss_true.iloc[0]
            late = g[(g.epoch >= 10) & (g.epoch <= 100)]
            rows.append(dict(setting=st, sur=sur, seed=s, broken=(sur, s) in BROKEN, cost0=c0, cost_max=late.loss_true.max(),
                             cost_end=g[g.epoch == g.epoch.max()].loss_true.iloc[0],
                             expelled=bool((late.loss_true > 1.10 * c0).any())))
    return pd.DataFrame(rows), (pd.concat(frames) if frames else pd.DataFrame())


def fig_warm(W):
    pairs = list(dict.fromkeys(zip(W.sur, W.seed)))
    nc = 5
    fig, ax = plt.subplots(int(np.ceil(len(pairs) / nc)), nc, figsize=(4 * nc, 3.2 * np.ceil(len(pairs) / nc)), squeeze=False)
    cols = {"plain": "#8B1A1A", "clip3": "#1B3A6B", "unit": "#4C78A8", "global3": "#E45756", "mom": "#54A24B",
            "batch100": "#B279A2", "lr1e-4": "#888888"}
    for k, (sur, s) in enumerate(pairs):
        a = ax[k // nc, k % nc]
        for st, g in W[(W.sur == sur) & (W.seed == s)].groupby("setting"):
            a.plot(g.epoch, g.loss_true, color=cols.get(st, "k"), lw=1.3, label=st)
        lab = {"MSE (3000)": "LN MSE", "J_x + J_u (3000)": "slope from 0", "J_x + J_u (two-stage)": "two-stage"}[sur]
        a.set_title(f"{lab}, seed {s}" + (" (broken)" if (sur, s) in BROKEN else " (healthy)"), fontsize=9)
        a.set_yscale("log"), a.grid(alpha=0.3), a.set_xlabel("epoch")
        if k % nc == 0:
            a.set_ylabel("true-plant cost")
    ax[0, 0].legend(fontsize=7)
    for k in range(len(pairs), ax.size):
        ax.flat[k].axis("off")
    fig.suptitle("Warm start at the good (oracle) GRU controller, 100 AdamW epochs at N = 200 through the surrogate: "
                 "true-plant cost on the 1000 training starts, per optimiser setting", fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.out("opt", "fig_warmstart.png"), dpi=130, facecolor="white")


def main(root):
    pd.set_option("display.width", 220)
    T, Rz, Pw = from_scratch(root)
    for d, n in ((T, "from_scratch"), (Rz, "rescued"), (Pw, "paired_vs_plain")):
        d.to_csv(paths.out("opt", f"{n}.csv"), index=False)
    print("FROM SCRATCH (", root, ")\n", T.round(3).to_string(index=False), "\n\n", Rz.to_string(index=False), "\n\n",
          Pw.round(4).to_string(index=False))
    S = score(root)
    S.to_csv(paths.out("opt", "prereg_outcomes.csv"), index=False)
    print("\nPRE-REGISTERED OUTCOMES (from scratch)\n" + md(S))
    WS, W = warm_start()
    if len(WS):
        WS.to_csv(paths.out("opt", "warmstart_table.csv"), index=False)
        fig_warm(W)
        piv = WS.pivot_table(index=["sur", "seed"], columns="setting", values="expelled", sort=False)
        print("\nWARM START: expelled (true cost > 1.10 x start at any epoch 10-100)\n", piv.to_string())
        b = WS[WS.broken].groupby("setting").expelled.agg(["sum", "size"])
        h = WS[~WS.broken].groupby("setting").expelled.agg(["sum", "size"])
        print("\nbroken pairs expelled:\n", b.to_string(), "\nhealthy pairs expelled:\n", h.to_string())
        print("\nPre-registered warm-start predictions:")
        if "global3" in b.index:
            print(f"  P2b global clip 3 expelled on >= 4 of 6 broken: {b.loc['global3', 'sum']} of {b.loc['global3', 'size']} -> "
                  + ("held" if b.loc["global3", "sum"] >= 4 else "failed"))
        if "mom" in b.index:
            st = b.loc["mom", "size"] - b.loc["mom", "sum"]
            print(f"  P3b median of means stays on >= 5 of 6 broken: {st} of {b.loc['mom', 'size']} -> " + ("held" if st >= 5 else "failed"))
        print(f"  P6 healthy pairs stay under every setting: {int(h['sum'].sum())} expulsions of {int(h['size'].sum())} -> "
              + ("held" if h["sum"].sum() == 0 else "failed"))
        for st in ("plain", "clip3"):
            if st in b.index:
                print(f"  reproduction {st}: broken expelled {b.loc[st, 'sum']} of {b.loc[st, 'size']}")
    return T, Rz, Pw, WS


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else paths.DATA)
