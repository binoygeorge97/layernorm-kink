import sys, time; import pathlib; sys.path.insert(0, str(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "paths.py").exists()))); import paths
import numpy as np, pandas as pd, level1_core as L, dpc_core as DC, phase_core as PC
OUT=paths.DATA; plant=L.Plant(); seeds=list(range(10))
FAM=[("LN MSE 3000","MSE (3000)"),("LN MSE 12000","MSE (12000)"),("LN slope from step 0","J_x + J_u (3000)"),
     ("LN two-stage","J_x + J_u (two-stage)"),("no norm","no norm, MSE"),("static norm","static norm, MSE")]
models=[PC.Model.surrogate(OUT,s,seeds,n) for n,s in FAM]
t0=time.time(); df,trajs,ref=PC.lqr_study(models,plant); print(time.time()-t0)
df.to_csv(paths.out("companion", "lqr_study.csv"),index=False)
np.savez_compressed(paths.out("companion", "lqr_ref.npz"), **{k:v for k,v in ref.items()})
for (n,_) in FAM:
    m=[mm for mm in models if mm.name==n][0]
    print(n, "rel_mse", np.array2string(m.met["rel_mse"], precision=2), "Jx_sup", np.round(m.met["Jx_sup"],3))
