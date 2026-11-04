import sys, time; import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "paths.py").exists()))); import paths
import numpy as np, pandas as pd, level1_core as L, dpc_core as DC, phase_core as PC
import os; _A,_B=(float(v) for v in os.environ.get("LK_PLANT","1.03,0.01").split(","))   # Task 2a: LK_PLANT="1.03,0.1"
OUT=paths.DATA; plant=L.Plant() if (_A,_B)==(1.03,0.01) else L.Plant(A=_A,B=_B); seeds=[int(v) for v in os.environ.get("LK_SEEDS","0,1,2,3,4,5,6,7,8,9").split(",")]
FAM=[("LN MSE 3000","MSE (3000)"),("LN MSE 12000","MSE (12000)"),("LN slope from step 0","J_x + J_u (3000)"),
     ("LN two-stage","J_x + J_u (two-stage)"),("no norm","no norm, MSE"),("static norm","static norm, MSE")]
if os.environ.get("LK_FAMILIES")=="rms": FAM=[("RMS MSE 3000","RMS MSE (3000)"),("RMS two-stage","RMS J_x + J_u (two-stage)")]   # Q7d
elif os.environ.get("LK_FAMILIES"): FAM=[f for f in FAM if f[0] in os.environ["LK_FAMILIES"].split(";")]                          # subset (Q6c)
models=[PC.Model.surrogate(OUT,s,seeds,n) for n,s in FAM]
EXTRA="--extra" in sys.argv     # Task 4.3: adds settle_time; writes lqr_study_extra.csv instead
t0=time.time(); df,trajs,ref=PC.lqr_study(models,plant,extra=EXTRA); print(time.time()-t0)
df.to_csv(paths.out("companion", "lqr_study_extra.csv" if EXTRA else "lqr_study.csv"),index=False)
if not EXTRA: np.savez_compressed(paths.out("companion", "lqr_ref.npz"), **{k:v for k,v in ref.items()})
for (n,_) in FAM:
    m=[mm for mm in models if mm.name==n][0]
    print(n, "rel_mse", np.array2string(m.met["rel_mse"], precision=2), "Jx_sup", np.round(m.met["Jx_sup"],3))
