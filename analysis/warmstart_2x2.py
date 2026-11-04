"""B2 (ii) (2026-11-04, exploratory, not pre-registered): 2 x 2 warm start. Start AdamW (lr 1e-3) at the repeat-0 oracle GRU
on the 1000 repeat-0 training starts, N = 200, 100 epochs, as analysis/warmstart_harness.py. The FIRST step uses aggregation
F, every later step aggregation G, with F, G in {plain (mean), per-start clip 3}. The diagonal cells are the harness runs
warmstart_plain.csv / warmstart_clip3.csv (same code path); this script runs the two off-diagonal cells
(clip3 -> plain, plain -> clip3) and, with --diag, the diagonal ones too. Logged at epochs 0, 1, 5, 10, ..., 100: true-plant
cost and surrogate cost (dpc_core.true_lqr_cost on the 1000 training starts). Expulsion rule of PREREG_opt: true cost >
1.10 x epoch-0 value at any logged epoch from 10 to 100.
Usage: python analysis/warmstart_2x2.py [--cells clip3>plain,plain>clip3] [--pairs ...]  -> out/opt/warmstart_2x2.csv"""
import os, sys, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import level1_core as L, dpc_core as DC, phase_core as PC
from warmstart_harness import PAIRS, BROKEN

jax.config.update("jax_enable_x64", True)
dt, tm = jnp.float64, jax.tree_util.tree_map
N, EPOCHS = 200, 100
AGG = {"plain": ("mean", 3.0), "clip3": ("clip", 3.0)}


def run(cells, pairs, out):
    plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    seeds = list(range(10))
    Co, cs = DC.load_controllers(paths.DATA, "true plant (oracle)", 0)
    roll_t = DC._roller(DC.make_dynamics("true", plant=plant), N)
    rows = []
    for cell in cells:
        f, g = cell.split(">")
        for sur, s in pairs:
            t0 = time.time()
            norm = DC.surrogate_configs()[sur]["norm"]
            P, S0, _ = PC._stack_models(paths.DATA, sur, [s], dt)
            x0 = PC._training_starts(seeds, 0, dt)[s:s + 1]
            C = tm(lambda a: jnp.asarray(np.asarray(a)[cs.index(s):cs.index(s) + 1], dt), Co)
            dyn_s = DC.make_dynamics("surrogate", norm, plant)
            roll_s = DC._roller(dyn_s, N)
            opt = optax.adamw(DC.CTRL_LR)
            O = jax.vmap(opt.init)(C)
            first = DC.get_dpc_trainer(("2x2", f, norm, 1), dyn_s, N, 1, opt, *AGG[f])
            rest4 = DC.get_dpc_trainer(("2x2", g, norm, 4), dyn_s, N, 4, opt, *AGG[g])
            rest5 = DC.get_dpc_trainer(("2x2", g, norm, 5), dyn_s, N, 5, opt, *AGG[g])

            def record(e, C):
                Xt, Ut = (np.asarray(a, float)[0, ..., 0] for a in roll_t(C, P, S0, x0))
                Xs, Us = (np.asarray(a, float)[0, ..., 0] for a in roll_s(C, P, S0, x0))
                rows.append(dict(cell=cell, sur=sur, seed=s, broken=(sur, s) in BROKEN, epoch=e,
                                 loss_true=DC.true_lqr_cost(Xt, Ut), loss_sur=DC.true_lqr_cost(Xs, Us)))
            record(0, C)
            C, O, _ = first(C, O, P, S0, x0); record(1, C)
            C, O, _ = rest4(C, O, P, S0, x0); record(5, C)
            for e in range(5, EPOCHS, 5):
                C, O, _ = rest5(C, O, P, S0, x0); record(e + 5, C)
            D = pd.DataFrame(rows); D.to_csv(out, index=False)
            q = D[(D.cell == cell) & (D.sur == sur) & (D.seed == s)]
            c0 = q.loss_true.iloc[0]; mx = q[q.epoch >= 10].loss_true.max()
            print(f"{cell:12s} {sur:22s} s{s}: true {c0:.3f} -> e1 {q.loss_true.iloc[1]:.3f} -> e100 {q.loss_true.iloc[-1]:.3f} "
                  f"(max e10-100 {mx:.3f}, {'EXPELLED' if mx > 1.10 * c0 else 'stays'}) | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="clip3>plain,plain>clip3")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out", default=paths.out("opt", "warmstart_2x2.csv"))
    a = ap.parse_args()
    pairs = PAIRS if a.pairs is None else [(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in a.pairs.split(",")]
    run(a.cells.split(","), pairs, a.out)
