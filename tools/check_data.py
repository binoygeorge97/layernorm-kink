"""Inventory and integrity check of a dpc_out folder (Task 0, CLAUDE.md 3.2).

Usage: python tools/check_data.py [folder]      (default: paths.DATA = $LK_DATA or data/dpc_out)
Checks: 6 surrogate files with seeds 0-9; 7 models x 3 repeats of controller files with seeds 0-9;
dpc_results.csv has 7 models x 10 seeds x 3 reps rows (one each); dpc_diag.csv has 6 surrogate models x 10 x 3.
Exit code 1 if any check fails."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEEDS, REPS = list(range(10)), (0, 1, 2)
MODELS = ["true plant (oracle)", "LN, MSE (3000)", "LN, MSE (12000)", "LN, J_x + J_u (3000)", "LN, two-stage",
          "no norm, MSE", "static norm, MSE"]
SURR = ["MSE (3000)", "MSE (12000)", "J_x + J_u (3000)", "J_x + J_u (two-stage)", "no norm, MSE", "static norm, MSE"]


def safe(name):  # = dpc_core.safe
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")


def main(folder):
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ok   " if cond else "  FAIL ") + msg)
        ok &= bool(cond)

    print(f"dpc_out folder: {folder}")
    print("surrogates/")
    for s in SURR:
        fn = os.path.join(folder, "surrogates", safe(s) + ".npz")
        if not os.path.exists(fn):
            check(False, f"{safe(s)}.npz missing")
            continue
        z = np.load(fn, allow_pickle=True)
        plant = tuple(np.asarray(z["plant"]).tolist()) if "plant" in z.files else None
        check(list(z["seeds"]) == SEEDS, f"{safe(s)}.npz seeds {[int(v) for v in z['seeds']]}"
              + (f", plant {plant}" if plant else ", no plant key (= (1.03, 0.01, 0.01))"))
    print("controllers/")
    for m in MODELS:
        for r in REPS:
            fn = os.path.join(folder, "controllers", safe(m) + ("" if r == 0 else f"__rep{r}") + ".npz")
            if not os.path.exists(fn):
                check(False, f"{os.path.basename(fn)} missing")
                continue
            z = np.load(fn)
            check(list(z["seeds"]) == SEEDS, f"{os.path.basename(fn)} ({len(z.files) - 1} arrays)")
    for name, n_models in (("dpc_results.csv", 7), ("dpc_diag.csv", 6)):
        fn = os.path.join(folder, name)
        if not os.path.exists(fn):
            check(False, f"{name} missing")
            continue
        d = pd.read_csv(fn)
        if "rep" not in d:
            d["rep"] = 0
        d["rep"] = d.rep.fillna(0).astype(int)
        cnt = d.groupby(["model", "seed", "rep"]).size()
        check(len(d) == n_models * 30 and d.model.nunique() == n_models and (cnt == 1).all(),
              f"{name}: {len(d)} rows, {d.model.nunique()} models, reps {[int(v) for v in sorted(d.rep.unique())]}, duplicates {(cnt > 1).sum()}")
    for name in ("dpc_curves.csv", "surrogates.csv", "phase_lqr.csv", "track_results.csv"):
        fn = os.path.join(folder, name)
        print(f"  info {name}: " + (f"{os.path.getsize(fn) / 1e6:.2f} MB" if os.path.exists(fn) else "absent"))
    for sub in ("trajectories", "track_controllers", "labels"):
        d = os.path.join(folder, sub)
        print(f"  info {sub}/: " + (f"{len(os.listdir(d))} files" if os.path.isdir(d) else "absent"))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else paths.DATA) else 1)
