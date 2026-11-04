"""Task 2b, plant P1: Colab session driver (embedded in plant2_P1.ipynb). Stage "surrogates+oracle":
1. Level 1: the six surrogate families x seeds on P1 (plant2_core.train; data key 1000 + s, init key s, slope draws 10000 + s;
   labels k = 1280, cached once per seed in <root>/labels2 and shared by the two slope-trained families). Each family is
   saved to <root>/surrogates/<family>.npz (stacked over seeds, plant stored for the cache guard; an existing file is
   reused only if its plant and seeds match). Metrics (plant2_core.level1_metrics_split, incl. the kink ellipsoid) ->
   <root>/surrogates.csv.
2. Oracles: DPC through the TRUE plant, GRU and MLP, repeats REPS, all seeds vmapped (controller key 20000 + s + 1000 rep,
   1000 training starts key 30000 + s + 1000 rep in the start parallelogram, curriculum 5..200 as the scalar study).
   Saved to <root>/controllers/<arch>/true_plant__oracle__rep<r>.npz. Test on 1000 typical starts (key 12345) through the true
   plant; sanity check against the saturated LQR (gap printed; the scalar oracle matched it to 0.04 %) -> <root>/oracle.csv.
Nothing in the scalar modules is used or changed."""
import os
import time
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import plant2_core as P2

FAMILIES = list(P2.CONFIGS)
SAFE = lambda name: "".join(ch if ch.isalnum() else "_" for ch in name)
P1 = dict(A=[[1.02, 0.05], [0.0, 1.01]], B=[0.0, 0.01])


def plant_p1():
    return P2.Plant2(P1["A"], P1["B"])


def stage_surrogates(root, seeds, plant, scale=1.0, label_k=1280, families=None, log=print):
    rows = []
    for fam in families or FAMILIES:
        fn = os.path.join(root, "surrogates", SAFE(fam) + ".npz")
        cfg = P2.configs(scale)[fam]
        if os.path.exists(fn):
            T, ex = P2.load_npz(fn, plant)                                   # raises if made for another plant
            if list(ex["seeds"]) != list(seeds) or float(ex["scale"]) != scale:
                raise ValueError(f"{fn} has seeds {list(ex['seeds'])} / scale {float(ex['scale'])}; use a fresh ROOT")
            Ps, S0 = T, ex["s0"]
            log(f"{fam}: loaded {fn}")
        else:
            ps, s0s = [], []
            for s in seeds:
                t0 = time.time()
                p, s0, _ = P2.train(cfg, s, plant, label_dir=os.path.join(root, "labels2"), label_k=label_k)
                ps.append(p); s0s.append(float(s0))
                log(f"{fam} seed {s}: trained in {time.time() - t0:.0f}s")
            Ps = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *ps); S0 = np.array(s0s)
            P2.save_npz(fn, plant, Ps, seeds=np.array(seeds), s0=S0, scale=np.array(scale), label_k=np.array(label_k))
        for i, s in enumerate(seeds):
            p = jax.tree_util.tree_map(lambda a: a[i], Ps)
            rows.append(dict(family=fam, seed=s, **P2.level1_metrics_split(p, float(S0[i]), cfg["norm"], plant)))
        pd.DataFrame(rows).to_csv(os.path.join(root, "surrogates.csv"), index=False)
    return pd.DataFrame(rows)


def load_family(root, fam, plant):
    T, ex = P2.load_npz(os.path.join(root, "surrogates", SAFE(fam) + ".npz"), plant)
    return T, ex["s0"], [int(s) for s in ex["seeds"]], P2.CONFIGS[fam]["norm"]


def ctrl_file(root, arch, name, rep):
    return os.path.join(root, "controllers", arch, f"{SAFE(name)}__rep{rep}.npz")


def evaluate_true(C, plant, X):
    """mean true-plant cost (dpc loss, N = 200) of each stacked controller on the starts X"""
    dyn = P2.true_dyn(plant)
    return np.asarray(jax.jit(jax.vmap(lambda c: P2.dpc_loss(c, dyn, X, 200)))(C))


def stage_oracle(root, seeds, plant, reps=(0, 1, 2), archs=("gru", "mlp"), curriculum=None, n_x0=P2.N_X0, log=print):
    curriculum = curriculum or P2.CURRICULUM
    Xt = P2.test_starts()
    ref = P2.saturated_lqr_cost(plant, np.asarray(Xt))
    rows = []
    for arch in archs:
        for rep in reps:
            fn = ctrl_file(root, arch, "true plant (oracle)", rep)
            if os.path.exists(fn):
                C, ex = P2.load_npz(fn, plant)
                if list(ex["seeds"]) != list(seeds):
                    raise ValueError(f"{fn} has other seeds")
                log(f"oracle {arch} rep {rep}: loaded")
            else:
                t0 = time.time()
                keys = jnp.stack([P2.ctrl_key(s, rep) for s in seeds])
                X0s = jnp.stack([P2.training_starts(s, rep, n_x0) for s in seeds])
                C, hist = P2.train_dpc_seeds(lambda p: P2.true_dyn(plant), None, keys, X0s, curriculum, arch)
                P2.save_npz(fn, plant, C, seeds=np.array(seeds), train_s=np.array(time.time() - t0))
                log(f"oracle {arch} rep {rep}: trained {len(seeds)} seeds in {time.time() - t0:.0f}s")
            cost = evaluate_true(C, plant, Xt)
            for i, s in enumerate(seeds):
                rows.append(dict(arch=arch, rep=rep, seed=s, cost_test_true=float(cost[i]), sat_lqr=ref,
                                 gap_vs_sat_lqr=float(cost[i] / ref - 1)))
            pd.DataFrame(rows).to_csv(os.path.join(root, "oracle.csv"), index=False)
    D = pd.DataFrame(rows)
    log(f"saturated LQR on the test starts: {ref:.4f}; oracle gap vs it (median, max): "
        + "; ".join(f"{a}: {100 * g.gap_vs_sat_lqr.median():.3f} %, {100 * g.gap_vs_sat_lqr.max():.3f} %" for a, g in D.groupby("arch")))
    return D
