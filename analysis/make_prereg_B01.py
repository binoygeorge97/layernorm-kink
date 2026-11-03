"""Task 2a.3: from the Stage A outputs at B = 0.1 (surrogates + GRU oracle), compute locally
  1. Level 1 metrics per seed (kink geometry, peak |J_u_hat/B - 1|) from <root>/surrogates.csv, against B = 0.01;
  2. the fixed / rescheduled LQR study at B = 0.1 (companion/run_lqr_study.py with LK_PLANT=1.03,0.1);
  3. Lambda* of the B = 0.1 GRU oracle through the B = 0.1 surrogates (theory/t2_test.py, T2_PLANT=1.03,0.1);
then score PREREG_B01 Part 1 (Q1-Q3) and APPEND Part 2 (run-level DPC trap predictions, Lambda* > 12.1) to
prereg/PREREG_B01.md. Commit the file BEFORE Stage B (DPC through the B = 0.1 surrogates).

Usage: python analysis/make_prereg_B01.py data/dpc_out_B01 [--seeds 0,...,9] [--reps 0,1,2] [--out out/B01] [--dry]
--dry prints Part 2 instead of appending it (for tests on QUICK data)."""
import argparse
import datetime
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import md  # noqa: E402

A, B = 1.03, 0.1
THR = 12.1
FAMS = ["LN MSE 3000", "LN MSE 12000", "LN slope from 0", "LN two-stage", "no norm", "static norm"]
LQR_FAM = {"LN MSE 3000": "LN MSE 3000", "LN MSE 12000": "LN MSE 12000", "LN slope from step 0": "LN slope from 0",
           "LN two-stage": "LN two-stage", "no norm": "no norm", "static norm": "static norm"}


def dare_K(a, b, Q=5.0, R=0.1):
    b2 = b * b
    coef = R - Q * b2 - a * a * R
    with np.errstate(divide="ignore", invalid="ignore"):
        P = (-coef + np.sqrt(coef ** 2 + 4 * b2 * Q * R)) / (2 * b2)
        return np.where(b2 > 1e-14, a * b * P / (R + b2 * P), 0.0)


def main(root, seeds, reps, out, dry):
    root, out = os.path.abspath(root), os.path.abspath(out)
    os.makedirs(os.path.join(out, "theory"), exist_ok=True)
    env = dict(os.environ, LK_DATA=root, LK_OUT=out, LK_PLANT=f"{A},{B}", LK_SEEDS=",".join(map(str, seeds)),
               T2_DATA=root, T2_OUT=os.path.join(out, "theory"), T2_PLANT=f"{A},{B}", T2_REPS=",".join(map(str, reps)))
    py = sys.executable
    # 1. Level 1 metrics
    S1 = pd.read_csv(os.path.join(root, "surrogates.csv"))
    S0 = pd.read_csv(os.path.join(paths.REPO, "data", "dpc_out", "surrogates.csv"))
    med = pd.DataFrame({"B = 0.01": S0.groupby("model").Ju_sup_rel.median(), "B = 0.1": S1.groupby("model").Ju_sup_rel.median()})
    geo = S1[S1.model == "LN, MSE (3000)"][["seed", "Ju_sup_rel", "zmin_x", "zmin_u", "rstar_x", "rstar_u"]]
    # 2. LQR study
    fn_lqr = os.path.join(out, "companion", "lqr_study_extra.csv")
    if not os.path.exists(fn_lqr):
        subprocess.run([py, os.path.join(paths.REPO, "companion", "run_lqr_study.py"), "--extra"], env=env, check=True)
    T = pd.read_csv(fn_lqr)
    T = T[(T.x0 - T.r).abs() >= 0.5].copy()
    T["failed"] = T.diverged | (T.gap > 0.1)
    T["cat"] = T.diverged | (T.gap > 1.0)
    rates = (T.groupby(["model", "controller"])[["failed", "cat"]].mean() * 100).round(1).unstack()
    rates.columns = [f"{'failed' if m == 'failed' else 'catastrophic'}, {c}" for m, c in rates.columns]
    F = T[T.model.str.startswith("LN") & (T.controller == "LQR fixed")].copy()
    F["pole"] = A - B * dare_K(F.Ah.to_numpy(), F.Bh.to_numpy())
    out_band = F.pole.abs() >= 1
    q2 = 100 * F[~out_band].cat.mean()
    q3 = 100 * F[out_band].cat.mean() if out_band.any() else np.nan
    # 3. Lambda*
    th = os.path.join(paths.REPO, "theory")
    for fams in (["LN MSE 3000"], FAMS[1:]):
        tag = "_".join(f.replace(" ", "") for f in fams)
        if not os.path.exists(os.path.join(out, "theory", f"t2_runs_{tag}.csv")):
            subprocess.run([py, "t2_test.py", ",".join(fams)], cwd=th, env=env, check=True)
    G = pd.concat([pd.read_csv(os.path.join(out, "theory", f"t2_runs_{'_'.join(f.replace(' ', '') for f in fams)}.csv"))
                   for fams in (["LN MSE 3000"], FAMS[1:])], ignore_index=True)
    G["predicted"] = G.S_seg_max > THR
    G[["family", "seed", "rep", "S_seg_max", "predicted", "gain_land_med", "K_land_med", "lyap_exact_max"]].to_csv(
        os.path.join(paths.REPO, "prereg", "PREREG_B01_predicted.csv") if not dry else os.path.join(out, "PREREG_B01_predicted.csv"),
        index=False)
    summ = G.groupby("family", sort=False).agg(runs=("S_seg_max", "size"), predicted_trapped=("predicted", "sum"),
                                               max_Lambda=("S_seg_max", "max")).reset_index()
    lst = "\n".join(f"- {r.family}, seed {r.seed}, repeat {r.rep}: Lambda* = {r.S_seg_max:.1f}"
                    for _, r in G[G.predicted].sort_values(["family", "rep", "seed"]).iterrows()) or "- none"
    q1_ln = med.loc["LN, MSE (3000)", "B = 0.1"] < 18.7
    lnf = ["LN, MSE (3000)", "LN, MSE (12000)", "LN, J_x + J_u (3000)", "LN, two-stage"]
    q1_n = int((med.loc[lnf, "B = 0.1"] < med.loc[lnf, "B = 0.01"]).sum())
    text = f"""
## Outcomes Part 1 (computed {datetime.date.today().isoformat()} from Stage A, `{os.path.relpath(root, paths.REPO)}`)
| id | outcome | numbers |
|---|---|---|
| Q1 | {"held" if (q1_ln and q1_n >= 3) else ("partly" if (q1_ln or q1_n >= 3) else "failed")} | median peak abs(J_u_hat/B - 1), B = 0.01 -> 0.1: {"; ".join(f"{m} {med.loc[m, 'B = 0.01']:.2f} -> {med.loc[m, 'B = 0.1']:.2f}" for m in med.index)}; LN families smaller: {q1_n} of 4 |
| Q2 | {"held" if q2 <= 2 else "failed"} | P(catastrophic given pole inside the band) = {q2:.2f} % (n = {int((~out_band).sum())}) |
| Q3 | {"not testable" if not out_band.any() else ("held" if q3 == 100 else "failed")} | designs outside the band: {int(out_band.sum())}, catastrophic {q3:.1f} % |

LQR failure rates at B = 0.1 (% failed / catastrophic of runs with abs(x0 - r) >= 0.5):
{md(rates.reset_index(), "{:.1f}")}

LN MSE 3000 kink geometry at B = 0.1 (per seed):
{md(geo.round(3))}

## Part 2 (written {datetime.date.today().isoformat()}, after Stage A, BEFORE Stage B)
Results that exist at the time of writing: Stage A at B = 0.1 (surrogates, labels, GRU oracle repeats {reps}), the LQR study
and Lambda* above. No DPC controller trained through a B = 0.1 surrogate exists.
Stage B (Colab A100): `PLANT = L.Plant(A=1.03, B=0.1, eps_nl=0.0)`, `ROOT = "dpc_out_B01"`, `SURROGATES_ONLY = False`,
`ONLY = None`, `REPS = [0, 1, 2]`, `ARCH = "gru"`, `RUN_ROBUST = False`, `RUN_OPT = False`, `DTYPE = "float64"`.
Primary outcome: trapped (CLAUDE.md 1.4, oracle of the same plant, seed and repeat). Predictor: Lambda* of the B = 0.1 GRU
oracle through the surrogate (theory/t2_test.py, T2_PLANT = 1.03,0.1), threshold 12.1 NOT re-tuned.

| id | prediction | how it is scored |
|---|---|---|
| R1 | trapped iff Lambda* > 12.1; predicted trapped: {int(G.predicted.sum())} of {len(G)} runs (list below, `prereg/PREREG_B01_predicted.csv`) | confusion matrix; held if hit rate >= 0.75 and false-alarm rate <= 0.05 (no predicted traps: held if at most 1 run is trapped) |
| R2 | two-stage, no norm and static norm: no trapped run | count |

Per family:
{md(summ)}

Runs predicted trapped:
{lst}

## Outcomes Part 2 (fill AFTER Stage B; do not edit anything above this line)
| id | outcome (held / failed / partly) | numbers |
|---|---|---|
"""
    if dry:
        print(text)
    else:
        open(os.path.join(paths.REPO, "prereg", "PREREG_B01.md"), "a").write(text)
        print("appended Part 2 to prereg/PREREG_B01.md - commit it now, record the hash, BEFORE Stage B")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--reps", default="0,1,2")
    ap.add_argument("--out", default=os.path.join(paths.REPO, "out", "B01"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    main(a.root, [int(s) for s in a.seeds.split(",")], [int(r) for r in a.reps.split(",")], a.out, a.dry)
