"""A1 (2026-11-04): Part 2 of prereg/PREREG_B01_aug.md from the B = 0.1 values of Lambda*_aug (three partial outputs of
theory/t2_aug.py in out/B01/theory/aug_{a,b,c}). Rule fixed in Part 1 (8b26564): predicted trapped iff Lambda*_aug > 12.08.
Writes out/B01/theory/t2_aug_B01.csv, prereg/PREREG_B01_aug_predicted.csv and prints the Part 2 text (appended by the caller)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import pandas as pd
THR = 12.08
D = pd.concat([pd.read_csv(paths.out("B01", "theory", f"aug_{t}", "t2_aug.csv")) for t in "abc"])
assert len(D) == 180 and not D.duplicated(["family", "rep", "seed"]).any(), len(D)
D = D.sort_values(["family", "rep", "seed"])
D.to_csv(paths.out("B01", "theory", "t2_aug_B01.csv"), index=False)
D["predicted"] = D.L_aug_star > THR
D[["family", "seed", "rep", "L_aug_star", "L_scal_land_star", "predicted"]].to_csv(
    os.path.join(paths.REPO, "prereg", "PREREG_B01_aug_predicted.csv"), index=False)
R1 = pd.read_csv(os.path.join(paths.REPO, "prereg", "PREREG_B01_predicted.csv"))
order = ["LN MSE 3000", "LN MSE 12000", "LN slope from 0", "LN two-stage", "no norm", "static norm"]
g = D.groupby("family").agg(runs=("L_aug_star", "size"), predicted=("predicted", "sum"), max_L_aug=("L_aug_star", "max"),
                            max_L_scal_land=("L_scal_land_star", "max")).reindex(order)
lines = ["| family | runs | predicted trapped | max Lambda*_aug | max anchored scalar |", "|---|---|---|---|---|"]
lines += [f"| {f} | {int(r.runs)} | {int(r.predicted)} | {r.max_L_aug:.2f} | {r.max_L_scal_land:.2f} |" for f, r in g.iterrows()]
P = D[D.predicted]
lst = "\n".join(f"- {r.family}, seed {r.seed}, repeat {r.rep}: Lambda*_aug = {r.L_aug_star:.2f}" for _, r in P.iterrows()) or "- none"
top = D.sort_values("L_aug_star", ascending=False).head(5)
near = "; ".join(f"{r.family} s{r.seed} r{r.rep} {r.L_aug_star:.2f}" for _, r in top.iterrows())
print(f"""Values: out/B01/theory/t2_aug_B01.csv (theory/t2_aug.py, T2_PLANT = 1.03,0.1, T2_DATA = data/dpc_out_B01, the B = 0.1 GRU
oracle repeats 0-2 and the six B = 0.1 surrogates fetched on 3 Nov; nothing else from dpc_out_B01 was fetched or read).
Rule (Part 1): predicted trapped iff Lambda*_aug > 12.08. Predicted trapped: {int(D.predicted.sum())} of 180 runs.

{chr(10).join(lines)}

Runs predicted trapped (Lambda*_aug):
{lst}

Five largest values: {near}.
Run-level predictions: prereg/PREREG_B01_aug_predicted.csv. For comparison only (not a prediction of this file): the primary
direct-gain rule (PREREG_B01 Part 2, 77d8455) predicts {int(R1.predicted.sum()) if 'predicted' in R1 else 'see file'} of 180.""")
