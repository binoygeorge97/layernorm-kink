"""B4 (2026-11-04): the 20 GRU traps (B = 0.01) and every false alarm under four statistics: registered direct-gain
Lambda* (S_seg_max, t2_test), exact price Lambda*_mem (t2_exact), anchored scalar (L_scal_land_star, t2_aug) and anchored
augmented Lambda*_aug (t2_aug). Retrospective; the registered statistic stays primary. -> out/theory/b4_table.csv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import pandas as pd, numpy as np
k = ["family", "rep", "seed"]
R = pd.read_csv(paths.out("theory", "t2_runs_all_B001.csv"))
R["trapped"] = (R.gap > 0.10) & (R.own_on_model / R.ref_on_model > 1.05)
D = (R[k + ["trapped", "gap", "S_seg_max"]].rename(columns={"S_seg_max": "direct"})
     .merge(pd.read_csv(paths.out("theory", "t2_exact.csv"))[k + ["L_mem_star"]].rename(columns={"L_mem_star": "exact_price"}), on=k)
     .merge(pd.read_csv(paths.out("theory", "t2_aug_B001.csv"))[k + ["L_scal_land_star", "L_aug_star"]]
            .rename(columns={"L_scal_land_star": "anch_scalar", "L_aug_star": "anch_aug"}), on=k))
assert len(D) == 180 and D.trapped.sum() == 20
S = ["direct", "exact_price", "anch_scalar", "anch_aug"]
print("statistic | hits >12.1 | FA >12.1 | zero-FA threshold (max non-trapped) | hits above it | min trapped")
for s in S:
    nt = D.loc[~D.trapped, s].max()
    print(f"{s:12s} | {(D.trapped & (D[s] > 12.1)).sum():2d}/20 | {(~D.trapped & (D[s] > 12.1)).sum():3d}/160 | {nt:6.2f} | "
          f"{(D.trapped & (D[s] > nt)).sum():2d}/20 | {D.loc[D.trapped, s].min():6.2f}")
fa = (~D.trapped) & (D[S] > 12.1).any(axis=1)
T = D[D.trapped | fa].sort_values(["trapped", "family", "seed", "rep"], ascending=[False, True, True, True])
for s in S:
    T[s + "_flag"] = T[s] > 12.1
T.to_csv(paths.out("theory", "b4_table.csv"), index=False)
pd.set_option("display.width", 200)
print(T[k + ["trapped", "gap"] + S].round(2).to_string(index=False))
