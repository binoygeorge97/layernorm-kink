"""Task 2b P1b (2026-11-04): accuracy of the slope labels in three inputs (plant P1) against the true Jacobian [A | B], compared
with the scalar study's labels (two inputs, A = 1.03, B = 0.01), both from 20 000 noisy transitions (sigma = 0.01, data key
1000 + s) with the local-quadratic Epanechnikov estimator and k nearest neighbours, evaluated at the training points (where
the slope loss uses them). Per Jacobian entry: median, 95th percentile and max of |label - true|; the u column relative to
|B| = 0.01, state entries absolute. Also on the interior (every input inside 80 % of its box half-width).
Usage: python analysis/plant2_labels.py [k=1280] [n=20000] [seeds=0,1,2] -> out/plant2/labels_accuracy_k<k>_n<n>.csv"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd
import plant2_core as P2
import level1_core as L
a = sys.argv[1:] + [None] * 3
k = int(a[0] or 1280); n = int(a[1] or 20000); seeds = [int(s) for s in (a[2] or "0,1,2").split(",")]
pl = P2.Plant2([[1.02, 0.05], [0.0, 1.01]], [0.0, 0.01])
J_true = np.column_stack([pl.A, pl.B])                         # 2 x 3
rows = []


def stats(err, scale, inner, **kw):
    for sub, m in (("all", np.ones(len(err), bool)), ("interior", inner)):
        e = np.abs(err[m]) / scale
        rows.append(dict(**kw, subset=sub, n_points=int(m.sum()), median=float(np.median(e)), p95=float(np.percentile(e, 95)),
                         max=float(e.max()), scale=scale))


for s in seeds:
    d = P2.make_data(pl, 0.01, 1000 + s, n)
    Z = np.column_stack([d["x"], d["u"]])
    t0 = time.time(); J = P2.LabelEstimator3(Z, d["y"], k)(Z); t3 = time.time() - t0
    inner = np.all(np.abs(Z) <= 0.8 * np.array(pl.box), axis=1)
    for i, out in enumerate(("x1+", "x2+")):
        for j, inp in enumerate(("x1", "x2", "u")):
            stats(J[:, i, j] - J_true[i, j], 0.01 if inp == "u" else 1.0, inner, plant="P1 (3 inputs)", seed=s, k=k, n=n,
                  entry=f"d{out}/d{inp}", time_s=t3)
    sp = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    ds = L.make_data(sp, 0.01, 1000 + s, n) if "n" in L.make_data.__code__.co_varnames else L.make_data(sp, 0.01, 1000 + s)
    xs_, us_, ys_ = (np.asarray(ds[c]) for c in ("x", "u", "y"))
    t0 = time.time(); lab = L.LabelEstimator(xs_, us_, ys_, k)(xs_, us_, want_var=False); t1 = time.time() - t0
    inner1 = (np.abs(xs_) <= 0.8 * 4.0) & (np.abs(us_) <= 0.8 * 10.0)
    stats(lab["Jx"] - 1.03, 1.0, inner1, plant="scalar (2 inputs)", seed=s, k=k, n=len(xs_), entry="dx+/dx", time_s=t1)
    stats(lab["Ju"] - 0.01, 0.01, inner1, plant="scalar (2 inputs)", seed=s, k=k, n=len(xs_), entry="dx+/du", time_s=t1)
    print(f"seed {s}: P1 labels {t3:.0f}s, scalar labels {t1:.0f}s", flush=True)
D = pd.DataFrame(rows)
D.to_csv(paths.out("plant2", f"labels_accuracy_k{k}_n{n}.csv"), index=False)
pd.set_option("display.width", 200)
print(D.groupby(["plant", "entry", "subset"])[["median", "p95", "max", "time_s"]].median().round(4).to_string())
