"""R5 (protocol prereg/PROTOCOL_trap_predictor.md, 51b4b07 + 9e62f0c): candidates C1 / C2 = warm-start expulsion from the
architecture's oracle controller. Plain AdamW (lr 1e-3, wd 1e-4) started at the oracle, 100 epochs at N = 200, full batch of the
repeat's 1000 training starts, vmapped over the 10 seeds of one (family, repeat). Every 5 epochs: true-plant cost and surrogate
cost of the current controllers on the training starts. C1: true cost > 1.10 x epoch 0 at any epoch 10-100; C2: the same on the
surrogate cost. A unit stops early once every seed has fired under both rules.
Usage: python analysis/trap_predictor.py SET [families]   SET in gru_B001, mlp, linear, gru_B01
-> out/predictor/warm_<SET>.csv (per run: c1, c2, max ratios, epoch of firing) and warm_<SET>_curves.csv"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp, optax
import level1_core as L, dpc_core as DC, phase_core as PC
jax.config.update("jax_enable_x64", True)
dt, tm = jnp.float64, jax.tree_util.tree_map
N, EPOCHS, LOG, RATIO = 200, 100, 5, 1.10
R_ = lambda *a: os.path.join(paths.REPO, *a)
SETS = {"gru_B001": dict(ctrl=R_("data", "dpc_out"), sur=R_("data", "dpc_out"), B=0.01),
        "mlp": dict(ctrl=R_("data", "dpc_out_mlp"), sur=R_("data", "dpc_out"), B=0.01),
        "linear": dict(ctrl=R_("data", "dpc_out_linear"), sur=R_("data", "dpc_out"), B=0.01),
        "gru_B01": dict(ctrl=R_("data", "dpc_out_B01"), sur=R_("data", "dpc_out_B01"), B=0.1)}
FAM = {"LN MSE 3000": "MSE (3000)", "LN MSE 12000": "MSE (12000)", "LN slope from 0": "J_x + J_u (3000)",
       "LN two-stage": "J_x + J_u (two-stage)", "no norm": "no norm, MSE", "static norm": "static norm, MSE"}


def main(name, fams):
    cfg = SETS[name]
    plant = L.Plant(A=1.03, B=cfg["B"], eps_nl=0.0)
    seeds = list(range(10))
    roll_t = DC._roller(DC.make_dynamics("true", plant=plant), N)
    out = paths.out("predictor", f"warm_{name}.csv"); outc = out.replace(".csv", "_curves.csv")
    rows, curves = [], []
    if os.path.exists(out):
        rows = pd.read_csv(out).to_dict("records"); curves = pd.read_csv(outc).to_dict("records")
    done = {(r["family"], r["rep"]) for r in rows}
    for fam in fams:
        sur = FAM[fam]; norm = DC.surrogate_configs()[sur]["norm"]
        P, S0, _ = PC._stack_models(cfg["sur"], sur, seeds, dt)
        dyn = DC.make_dynamics("surrogate", norm, plant); roll_s = DC._roller(dyn, N)
        opt = optax.adamw(DC.CTRL_LR)
        step = DC.get_dpc_trainer(("pred", name, norm), dyn, N, LOG, opt, "mean", 3.0)
        for rep in (0, 1, 2):
            if (fam, rep) in done:
                continue
            t0 = time.time()
            Co, cs = DC.load_controllers(cfg["ctrl"], "true plant (oracle)", rep)
            C = tm(lambda a: jnp.asarray(np.asarray(a)[[cs.index(s) for s in seeds]], dt), Co)
            x0 = PC._training_starts(seeds, rep, dt)
            O = jax.vmap(opt.init)(C)

            def cost(C):
                Xt, Ut = (np.asarray(a, float)[..., 0] for a in roll_t(C, P, S0, x0))
                Xs, Us = (np.asarray(a, float)[..., 0] for a in roll_s(C, P, S0, x0))
                return (np.array([DC.true_lqr_cost(Xt[i], Ut[i]) for i in range(len(seeds))]),
                        np.array([DC.true_lqr_cost(Xs[i], Us[i]) for i in range(len(seeds))]))
            t_0, s_0 = cost(C)
            mt, ms = np.ones(len(seeds)), np.ones(len(seeds)); e1, e2 = np.full(len(seeds), -1), np.full(len(seeds), -1)
            for i, s in enumerate(seeds):
                curves.append(dict(family=fam, rep=rep, seed=s, epoch=0, cost_true=t_0[i], cost_sur=s_0[i]))
            for e in range(LOG, EPOCHS + 1, LOG):
                C, O, _ = step(C, O, P, S0, x0)
                ct, csr = cost(C)
                for i, s in enumerate(seeds):
                    curves.append(dict(family=fam, rep=rep, seed=s, epoch=e, cost_true=ct[i], cost_sur=csr[i]))
                if e >= 10:
                    rt, rs = ct / t_0, csr / s_0
                    rt = np.where(np.isfinite(rt), rt, np.inf); rs = np.where(np.isfinite(rs), rs, np.inf)
                    mt, ms = np.maximum(mt, rt), np.maximum(ms, rs)
                    e1 = np.where((e1 < 0) & (rt > RATIO), e, e1); e2 = np.where((e2 < 0) & (rs > RATIO), e, e2)
                    if (e1 > 0).all() and (e2 > 0).all():
                        break
            for i, s in enumerate(seeds):
                rows.append(dict(set=name, family=fam, seed=s, rep=rep, c1=bool(e1[i] > 0), c2=bool(e2[i] > 0),
                                 max_ratio_true=mt[i], max_ratio_sur=ms[i], epoch_c1=int(e1[i]), epoch_c2=int(e2[i]),
                                 epochs_run=e, cost0_true=t_0[i], cost0_sur=s_0[i]))
            pd.DataFrame(rows).to_csv(out, index=False); pd.DataFrame(curves).to_csv(outc, index=False)
            print(f"{name} {fam:16s} rep {rep}: C1 fired on {int((e1 > 0).sum())}/10, C2 on {int((e2 > 0).sum())}/10 "
                  f"(seeds C1 {[s for s, f in zip(seeds, e1 > 0) if f]}) | {e} epochs | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else list(FAM))
