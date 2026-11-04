"""B5: why the linear DPC oracle lands at K ~ 16.7-17 instead of the LQR gain 10.29 (true plant, B = 0.01).
(1) DPC training loss (dpc_core.dpc_loss, N = 200, 1000 training starts of seed 0 rep 0) and the test cost
    (true_lqr_cost, 1000 test starts) of u = 10 tanh(-K x / 10) and of u = clip(-K x, +-10) on a grid of K.
(2) Figure: u(x) on [-3, 3] for the linear oracle, the MLP oracle and the saturated LQR (seed 0, rep 0), with slopes at x = 0
    and true-plant test costs. Output out/figs/fig_linear_gain.png, out/figs/linear_gain_scan.csv"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import level1_core as L, dpc_core as DC, phase_core as PC
plant = L.Plant(A=1.03, B=0.01, eps_nl=0.0); A, B = plant.A, plant.B
X0 = np.asarray(PC._training_starts(list(range(10)), 0))[0, :, 0]
XT = np.random.default_rng(12345).uniform(-3, 3, 1000)

def roll(pol, x0, N=200):
    xs, us = [x0], []
    for _ in range(N):
        u = pol(xs[-1]); us.append(u); xs.append(A * xs[-1] + B * u)
    return np.stack(xs), np.stack(us)

def dpc_loss(pol, x0):            # dpc_core.dpc_loss: (sum_k mean(Q x^2 + R u^2) + mean(Q_f x_N^2)) / N
    xs, us = roll(pol, x0)
    return ((5 * xs[:-1] ** 2 + 0.1 * us ** 2).mean(1).sum() + (50 * xs[-1] ** 2).mean()) / 200

def test_cost(pol):
    xs, us = roll(pol, XT); return DC.true_lqr_cost(xs, us)

tanh_pol = lambda K: (lambda x: 10 * np.tanh(-K * x / 10))
clip_pol = lambda K: (lambda x: np.clip(-K * x, -10, 10))
Ks = np.round(np.arange(6, 30.01, 0.25), 2)
rows = [dict(K=K, loss_tanh=dpc_loss(tanh_pol(K), X0), loss_clip=dpc_loss(clip_pol(K), X0),
             test_tanh=test_cost(tanh_pol(K)), test_clip=test_cost(clip_pol(K))) for K in Ks]
S = pd.DataFrame(rows); S.to_csv(paths.out("figs", "linear_gain_scan.csv"), index=False)
for c in ("loss_tanh", "loss_clip", "test_tanh", "test_clip"):
    r = S.loc[S[c].idxmin()]; print(f"argmin {c:10s}: K = {r.K:5.2f}  value {r[c]:.5f}")
print("at K = 10.29: loss tanh %.5f clip %.5f | test tanh %.5f clip %.5f" % (dpc_loss(tanh_pol(10.29), X0), dpc_loss(clip_pol(10.29), X0), test_cost(tanh_pol(10.29)), test_cost(clip_pol(10.29))))
# small-start check: only |x0| <= 0.3 (no saturation)
small = X0[np.abs(X0) <= 0.3]
Ks2 = np.arange(6, 30.01, 0.25); ls = [dpc_loss(tanh_pol(K), small) for K in Ks2]
print("argmin DPC loss of the tanh policy over small starts only (|x0| <= 0.3, n=%d): K = %.2f" % (len(small), Ks2[int(np.argmin(ls))]))
# unsaturated finite-horizon LQR with terminal weight (time-varying gain at k = 0)
P = 50.0; Kt = []
for _ in range(200):
    K = A * B * P / (0.1 + B * B * P); P = 5 + A * A * P - (A * B * P) ** 2 / (0.1 + B * B * P); Kt.append(K)
print("finite-horizon LQR (Q_f = 50): gain at the first step %.2f, at the last step %.2f" % (Kt[-1], Kt[0]))

# figure
CL, cl = DC.load_controllers(paths.REPO + "/data/dpc_out_linear", "true plant (oracle)", 0)
CM, cm = DC.load_controllers(paths.REPO + "/data/dpc_out_mlp", "true plant (oracle)", 0)
tm = jax.tree_util.tree_map
c_lin, c_mlp = tm(lambda a: a[cl.index(0)], CL), tm(lambda a: a[cm.index(0)], CM)
xx = np.linspace(-3, 3, 601)
u_of = lambda c: np.asarray(DC.ctrl_step(c, jnp.zeros((len(xx), 0)), jnp.asarray(xx)[:, None])[1][:, 0])
ul, um, uq = u_of(c_lin), u_of(c_mlp), np.clip(-DC.LQR_K * xx, -10, 10)
kl, km = DC.policy_gain(c_lin, [0.0])[0], DC.policy_gain(c_mlp, [0.0])[0]
RL = pd.read_csv(paths.REPO + "/data/dpc_out_linear/dpc_results.csv"); RM = pd.read_csv(paths.REPO + "/data/dpc_out_mlp/dpc_results.csv")
cost_l = float(RL[(RL.seed == 0) & (RL.rep == 0)].cost_test_true.iloc[0]); cost_m = float(RM[(RM.seed == 0) & (RM.rep == 0)].cost_test_true.iloc[0])
xs, us = DC.lqr_saturated(plant, XT); cost_q = DC.true_lqr_cost(xs, us)
print(f"seed 0 rep 0: linear oracle K(0) = {kl:.2f}, cost {cost_l:.4f} | MLP oracle K(0) = {km:.2f}, cost {cost_m:.4f} | saturated LQR K = 10.29, cost {cost_q:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
ax[0].plot(xx, uq, color="#888888", lw=2, ls="--", label=f"saturated LQR clip(-10.29 x): slope 10.29, cost {cost_q:.4f}")
ax[0].plot(xx, ul, color="#8B1A1A", lw=1.8, label=f"linear oracle 10 tanh(-Kx/10): slope {kl:.2f}, cost {cost_l:.4f}")
ax[0].plot(xx, um, color="#1B3A6B", lw=1.8, label=f"MLP oracle: slope {km:.2f}, cost {cost_m:.4f}")
ax[0].plot(xx, 10 * np.tanh(-10.29 * xx / 10), color="#8B1A1A", lw=1, ls=":", label="10 tanh(-10.29 x / 10) (for comparison)")
ax[0].set_xlabel("x"); ax[0].set_ylabel("u"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7.5, loc="lower left")
ax[0].set_title("Learned policies on the true plant (seed 0, repeat 0)", fontsize=10)
ax[1].plot(S.K, S.loss_tanh, color="#8B1A1A", label="DPC loss, 10 tanh(-Kx/10)")
ax[1].plot(S.K, S.loss_clip, color="#888888", ls="--", label="DPC loss, clip(-Kx)")
ax[1].axvline(10.29, color="k", lw=0.8, ls=":"); ax[1].set_xlabel("gain K"); ax[1].set_ylabel("DPC training loss (true plant)")
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8); ax[1].set_title("Training loss of the two linear policy shapes against K", fontsize=10)
fig.tight_layout(); fig.savefig(paths.out("figs", "fig_linear_gain.png"), dpi=140, facecolor="white")
