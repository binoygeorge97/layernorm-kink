"""Task 1.3: pre-registration of the MLP / linear DPC runs from the Stage A oracles (no DPC-through-surrogate result needed).

1. Lambda* of the MLP oracle and of the linear oracle (trained through the TRUE plant, repeats 0-2) through all 6 families
   x 10 seeds x 3 repeats (theory/t2_arch.py; memoryless, exact multipliers).
2. Run-level prediction: trapped iff Lambda* > 12.1 (threshold fixed on the GRU LN MSE 3000 runs, NOT re-tuned).
3. Writes prereg/PREREG_arch.md and prereg/PREREG_arch_predicted.csv. Commit both BEFORE Stage B results exist, then
   record the hash (CLAUDE.md ground rule 1).

Usage: python analysis/make_prereg_arch.py --mlp data/dpc_out_mlp --linear data/dpc_out_linear
       (folders containing controllers/true_plant__oracle[__repN].npz; surrogates are read from data/dpc_out)"""
import argparse
import datetime
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import pandas as pd  # noqa: E402

THR = 12.1


def lam(root, tag):
    out = paths.out("theory")
    fn = os.path.join(out, f"t2_arch_{tag}.csv")
    if not os.path.exists(fn):
        env = dict(os.environ, T2_DATA=paths.DATA, T2_OUT=out)
        subprocess.run([sys.executable, "t2_arch.py", os.path.abspath(root), "--tag", tag], cwd=os.path.join(paths.REPO, "theory"),
                       env=env, check=True)
    return pd.read_csv(fn)


def table(D):
    g = D.groupby("family", sort=False).agg(runs=("Lambda_star", "size"), predicted=("predicted_trap", "sum"),
                                            max_Lambda=("Lambda_star", "max"), med_K_land=("K_land_med", "median"))
    return g.reset_index()


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main(mlp_root, lin_root):
    M, Lr = lam(mlp_root, "mlp"), lam(lin_root, "linear")
    M["arch"], Lr["arch"] = "mlp", "linear"
    P = pd.concat([M, Lr])[["arch", "family", "seed", "rep", "Lambda_star", "predicted_trap", "K_land_med", "factorisation_err"]]
    P.rename(columns={"predicted_trap": "predicted"}).to_csv(os.path.join(paths.REPO, "prereg", "PREREG_arch_predicted.csv"),
                                                              index=False)
    lst = M[M.predicted_trap].sort_values(["family", "rep", "seed"])
    runs = "\n".join(f"- {r.family}, seed {r.seed}, repeat {r.rep}: Lambda* = {r.Lambda_star:.1f}" for _, r in lst.iterrows()) or "- none"
    Kstar = pd.read_csv(paths.out("theory", "t2_arch_unit_linear_Kstar.csv")) if os.path.exists(
        paths.out("theory", "t2_arch_unit_linear_Kstar.csv")) else None
    today = datetime.date.today().isoformat()
    text = f"""# Pre-registration: MLP and linear DPC controllers, prospective test of Theorem 2 (Task 1)

Date written: {today}
Commit hash of this file (fill after committing, then commit again): see "Registered at commit" at the end
Results that exist at the time of writing: the GRU study (dpc_out, 7 models x 10 seeds x 3 repeats); the MLP and linear
ORACLE controllers (Stage A: trained through the true plant only, `{mlp_root}`, `{lin_root}`). No MLP or linear controller
trained through any surrogate exists.

## Question
Is the DPC trap an artefact of the recurrent (GRU) policy, and does the landing amplification Lambda*, computed from the good
(oracle) controller alone, predict which MLP and linear DPC runs are trapped?

## Design (fixed before results)
- Stage B (Colab A100): DPC notebook, `ARCH = "mlp"` then `ARCH = "linear"`, `ONLY = None`, `REPS = [0, 1, 2]`,
  `RUN_ROBUST = False`, `RUN_OPT = False`, `DTYPE = "float64"`, `QUICK = False`, Drive root `MyDrive/dpc_out` (results go to
  `dpc_out_mlp`, `dpc_out_linear`; surrogates read from `dpc_out`). MLP: width 64, depth 2, tanh, u = 10 tanh(.).
  Linear: u = 10 tanh((k x + b) / 10), k, b stored / 10, init gain -5.
- Units: 6 surrogate families x 10 seeds x 3 repeats = 180 runs per architecture.
- Primary outcome: trapped = failed (typical-start gap > 10 %) AND own controller on its surrogate > 1.05 x the oracle
  controller of the same architecture on the same surrogate (dpc_diag.csv of the architecture's folder).
- Predictor: Lambda* of the architecture's oracle (same repeat) through the surrogate, 1000 training starts of that repeat,
  landing window |x| <= 0.2 (theory/t2_arch.py, commit of this file). Threshold 12.1, NOT re-tuned.

## Predictions
| id | prediction | basis | how it is scored |
|---|---|---|---|
| P1 | MLP: trapped iff Lambda* > 12.1; predicted trapped: {int(M.predicted_trap.sum())} of 180 runs (list below) | Theorem 2, GRU test 17/20 hits, 0/160 false alarms | confusion matrix; held if hit rate >= 0.75 and false-alarm rate <= 0.05 (if no run is predicted trapped: held if at most 1 of 180 is trapped); partly if one of the two holds |
| P2 | Linear: no run trapped (0 of 180). Lambda* of the trained linear oracle: max {Lr.Lambda_star.max():.2f}, {int(Lr.predicted_trap.sum())} run(s) above 12.1{'' if Kstar is None else f'; linear policy at K* = 10.2875: max {Kstar.Lambda_star.max():.2f}'} | memoryless policy with LQR-like gain stays in the safe band | held if 0 trapped; else failed, with the runs listed |
| A1 | (copied, 23 Oct) The MLP reproduces the GRU result: LN MSE fails on seeds 1, 2, 8, 9 and slope-from-0 on 2, 3, 9; two-stage and kink-free on none; trap test > 1 on the failures | GRU result | per-seed failures (majority of repeats) vs the listed seeds |
| A2 | (copied) The linear controller does NOT fail on those seeds | as P2 | failed runs on those seeds |
| A3 | (copied) Trapped MLP policies are flat or inverted inside the kink band, recover the LQR slope outside; healthy ones are a saturated line of slope -10 to -17 | - | D9 policy maps, K_eff at x = 0, 0.5, 2 |
| A4 | (copied) The linear DPC loss minimum through the surrogate is within ~10 % of the true plant's K on every seed | - | D10 |
| A5 | (copied) dL/dK through LN MSE has the wrong sign on a larger share of the gain axis than through kink-free models | - | D11 |
| A6, A7 | (copied) capacity sweep, survivor orientation: not run in this stage | - | not scored here |

Note on P1 vs A1 (written before results): A1 was written on 23 Oct from the GRU result. P1 is the theory's run-level
prediction from the MLP oracle's own closed loop. Where they disagree on a seed (A1 says failed, P1 says Lambda* <= 12.1, or
vice versa), the outcome decides between "the trap belongs to the surrogate" (A1) and "the trap needs a policy whose good
solution amplifies through the kink" (P1). Both are scored.

Per-family summary of the MLP predictor (oracle Lambda*):
{md_table(table(M))}

Per-family summary of the linear predictor (trained linear oracle Lambda*):
{md_table(table(Lr))}

Runs predicted trapped (MLP):
{runs}

Run-level predictions for both architectures: `prereg/PREREG_arch_predicted.csv`.

## Analysis plan (analysis/arch_report.py, docs/RESULTS_arch.md)
Per architecture x family: failed / trapped of 30, median gap; paired Wilcoxon vs the GRU over seeds (per-seed median over
repeats); confusion matrices of P1 and P2; K_eff table (D9) against the LQR gain 10.29; A1-A5 marked held / failed / partly
with the numbers. If the MLP is not trapped on the GRU-trapped seeds, the paper claim narrows to "recurrent /
high-direct-gain policies", stated plainly.

## Outcomes (fill AFTER results; do not edit anything above this line)
| id | outcome (held / failed / partly) | numbers |
|---|---|---|
"""
    fn = os.path.join(paths.REPO, "prereg", "PREREG_arch.md")
    open(fn, "w").write(text)
    print(f"wrote {fn}: MLP predicted trapped {int(M.predicted_trap.sum())}/180, linear {int(Lr.predicted_trap.sum())}/180. "
          "Commit it now, record the hash, BEFORE Stage B.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp", required=True)
    ap.add_argument("--linear", required=True)
    a = ap.parse_args()
    main(a.mlp, a.linear)
