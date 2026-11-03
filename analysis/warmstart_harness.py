"""Warm-start harness (Task 3.2, local CPU): start AdamW AT the reference (oracle, repeat 0) GRU controller and train it
for EPOCHS epochs at horizon N = 200 through a surrogate, with a choice of gradient aggregation / optimiser setting.
Generalises the warm start of story/story_data.py (same controller, same 1000 training starts of repeat 0, same AdamW
lr 1e-3, same logging): every LOG epochs the true-plant cost and the surrogate loss of the current controller on the
1000 training starts (dpc_core.true_lqr_cost).

Settings (name -> dpc_core options): plain (mean), clip3 (per-start clip 3), unit, global3 (global clip 3), mom (median
of means, 10 groups), batch100 (100 starts per epoch, key 40000 + s), lr1e-4 (plain, AdamW lr 1e-4).
Pairs: LN MSE seeds 1, 2, 8, 9 and slope-from-0 seeds 2, 9 (broken); LN MSE seeds 3, 7 and two-stage seeds 2, 9 (healthy).

Usage: python analysis/warmstart_harness.py [--settings plain,clip3,...] [--epochs 100] [--pairs "MSE (3000):2,..."]
Output: out/opt/warmstart_<setting>.csv and a log line per pair. Columns: loss_true / loss_sur = true_lqr_cost of the
current controller on the 1000 training starts through the true plant / the surrogate; train_loss_mean / _max = the DPC
training loss (dpc_loss on the starts used in that epoch) over the LOG epochs before the row (= loss_sur / loss_sur_max of
the reference files data/reference_results/warmstart*.csv)."""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
import level1_core as L  # noqa: E402
import dpc_core as DC  # noqa: E402
import phase_core as PC  # noqa: E402

jax.config.update("jax_enable_x64", True)
dt, tm = jnp.float64, jax.tree_util.tree_map
N, LOG = 200, 5
SETTINGS = {"plain": dict(), "clip3": dict(aggregate="clip", clip=3.0), "unit": dict(aggregate="unit"),
            "global3": dict(aggregate="global", clip=3.0), "mom": dict(aggregate="mom", groups=10),
            "batch100": dict(batch=100), "lr1e-4": dict(lr=1e-4)}
PAIRS = [("MSE (3000)", 1), ("MSE (3000)", 2), ("MSE (3000)", 8), ("MSE (3000)", 9), ("J_x + J_u (3000)", 2),
         ("J_x + J_u (3000)", 9), ("MSE (3000)", 3), ("MSE (3000)", 7), ("J_x + J_u (two-stage)", 2),
         ("J_x + J_u (two-stage)", 9)]
BROKEN = {("MSE (3000)", 1), ("MSE (3000)", 2), ("MSE (3000)", 8), ("MSE (3000)", 9), ("J_x + J_u (3000)", 2),
          ("J_x + J_u (3000)", 9)}


def run(setting, pairs, epochs, tag=""):
    o = SETTINGS[setting]
    plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0)
    seeds = list(range(10))
    Co, cs = DC.load_controllers(paths.DATA, "true plant (oracle)", 0)
    roll_t = DC._roller(DC.make_dynamics("true", plant=plant), N)
    rows = []
    for sur, s in pairs:
        t0 = time.time()
        norm = DC.surrogate_configs()[sur]["norm"]
        P, S0, met = PC._stack_models(paths.DATA, sur, [s], dt)
        x0 = PC._training_starts(seeds, 0, dt)[s:s + 1]
        C = tm(lambda a: jnp.asarray(np.asarray(a)[cs.index(s):cs.index(s) + 1], dt), Co)
        dyn_s = DC.make_dynamics("surrogate", norm, plant)
        roll_s = DC._roller(dyn_s, N)
        opt = optax.adamw(o.get("lr", DC.CTRL_LR))
        O = jax.vmap(opt.init)(C)
        key = ("warm", setting, norm)
        if o.get("batch"):
            step = DC.get_dpc_trainer_batch(key, dyn_s, N, LOG, opt, o["batch"])
            bk = jnp.stack([jax.random.PRNGKey(DC.BATCH_KEY + s)])
        else:
            agg = o.get("aggregate", "mean")
            step = (DC.get_dpc_trainer(key, dyn_s, N, LOG, opt, agg, o.get("clip", 3.0), o.get("groups", 10)) if agg == "mom"
                    else DC.get_dpc_trainer(key, dyn_s, N, LOG, opt, agg, o.get("clip", 3.0)))

        def record(e, C, tl=None):
            Xt, Ut = (np.asarray(a, float)[0, ..., 0] for a in roll_t(C, P, S0, x0))
            Xs, Us = (np.asarray(a, float)[0, ..., 0] for a in roll_s(C, P, S0, x0))
            rows.append(dict(setting=setting, sur=sur, seed=s, epoch=e, loss_sur=DC.true_lqr_cost(Xs, Us),
                             loss_true=DC.true_lqr_cost(Xt, Ut),
                             train_loss_mean=np.nan if tl is None else float(np.mean(tl)),
                             train_loss_max=np.nan if tl is None else float(np.max(tl))))
        record(0, C)
        for e in range(0, epochs, LOG):
            if o.get("batch"):
                C, O, tl = step(C, O, P, S0, x0, bk, e)
            else:
                C, O, tl = step(C, O, P, S0, x0)
            record(e + LOG, C, np.asarray(tl)[0])
        r0, r1 = rows[-(epochs // LOG) - 1], rows[-1]
        print(f"{setting:8s} {sur:22s} seed {s}: true cost {r0['loss_true']:.3f} -> {r1['loss_true']:.3f} "
              f"(max {max(r['loss_true'] for r in rows if r['sur'] == sur and r['seed'] == s):.3f}) | "
              f"surrogate loss {r0['loss_sur']:.3f} -> {r1['loss_sur']:.3f} | {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(paths.out("opt", f"warmstart_{setting}{tag}.csv"), index=False)
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default=",".join(SETTINGS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--pairs", default=None, help='e.g. "MSE (3000):2,J_x + J_u (3000):9"')
    ap.add_argument("--tag", default="", help="suffix of the output file (exploratory re-runs, e.g. _1000ep_part1)")
    a = ap.parse_args()
    pairs = PAIRS if a.pairs is None else [(p.rsplit(":", 1)[0], int(p.rsplit(":", 1)[1])) for p in a.pairs.split(",")]
    for st in a.settings.split(","):
        run(st, pairs, a.epochs, a.tag)
