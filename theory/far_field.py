"""Q9 / CLAUDE.md Task 5.4 (2026-11-04): the far-field branch function Phi(theta) of Corollary 4 (theory.tex, cor:block) from
the trained weights, and the decay laws of the branch slope on the trained LN MSE 3000 surrogates (seeds 0-9), float64.
Coordinates: w = U^T Q_E (z - z*) (Theorem 1), rho = ||w||, w_hat = (cos th, sin th).
 1. Phi(th) = w_d^T G(gamma * sqrt(H) U w_hat + beta) (eps -> 0 far-field limit of the normalised features); check that
    the branch output at z(rho, th) converges to Phi(th) as rho grows (error vs rho).
 2. Decay of the branch gradient in w: along the ray (d/d rho) and across it ((1/rho) d/d th), at 24 angles, rho from
    10 D to 1000 D; log-log slope per angle (expected -2 along, -1 across where Phi'(th) != 0).
 3. The same in the original inputs: |J_u_hat - skip slope| along the u-axis through z* and along the x-axis (fit exponent).
Output out/theory/far_field.csv, figure out/figs/fig_far_field.png."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import level1_core as L
H, EPS = 16, 1e-6
PM = np.eye(H) - np.ones((H, H)) / H
z = np.load(os.path.join(paths.DATA, "surrogates", "MSE__3000.npz"), allow_pickle=True)
keys = [k for k in z.files if k.startswith("p:")]


def params(i):
    out = {}
    for k in keys:
        parts = k[2:].split("/")
        d = out
        for q in parts[:-1]:
            d = d.setdefault(q, {})
        d[parts[-1]] = jnp.asarray(z[k][i])
    return out


def branch_from_v(p, v):
    a = jax.nn.gelu((p["gamma"] * v + p["beta"]) @ p["br"]["W1"] + p["br"]["b1"], approximate=False)
    return (a @ p["br"]["W2"] + p["br"]["b2"]) @ p["Wd"][:, 0]


def branch_z(p, x, u):
    h = x * p["E"][0] + u * p["E"][1] + p["b"]
    r = h - jnp.mean(h); v = r / jnp.sqrt(jnp.mean(r ** 2) + EPS)
    return branch_from_v(p, v)


th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
rows, curves = [], {}
for i in range(10):
    p = params(i)
    E, b = np.asarray(p["E"]).T, np.asarray(p["b"])               # E: H x 2
    QE = PM @ E; c = PM @ b
    zs = -np.linalg.lstsq(QE, c, rcond=None)[0]
    cp = QE @ zs + c; D = np.sqrt(cp @ cp + H * EPS)
    U, S, Vt = np.linalg.svd(QE, full_matrices=False)
    M = U.T @ QE                                                     # w = M (z - z*)
    Minv = np.linalg.inv(M)
    Phi = lambda t: branch_from_v(p, jnp.sqrt(H) * jnp.asarray(U) @ jnp.array([jnp.cos(t), jnp.sin(t)]))
    zz = lambda rho, t: zs + Minv @ np.array([rho * np.cos(t), rho * np.sin(t)])
    fw = jax.jit(lambda rho, t: branch_z(p, *(jnp.asarray(zs) + jnp.asarray(Minv) @ jnp.array([rho * jnp.cos(t), rho * jnp.sin(t)]))))
    d_rho = jax.jit(jax.grad(fw, 0)); d_th = jax.jit(jax.grad(fw, 1)); dPhi = jax.jit(jax.grad(Phi))
    rhos = D * np.logspace(1, 3, 9)
    conv = [max(abs(float(fw(r, t)) - float(Phi(t))) for t in th) for r in rhos]
    ea, ec = [], []
    for t in th:
        a = np.array([abs(float(d_rho(r, t))) for r in rhos]); cr = np.array([abs(float(d_th(r, t))) / r for r in rhos])
        if np.all(a > 0):
            ea.append(np.polyfit(np.log(rhos), np.log(a), 1)[0])
        if abs(float(dPhi(t))) > 1e-3 * np.max([abs(float(dPhi(q))) for q in th]):
            ec.append(np.polyfit(np.log(rhos), np.log(cr), 1)[0])
    # in the original inputs: J_u of the branch along the u-axis and the x-axis through z*
    gu = jax.jit(jax.grad(lambda x, u: branch_z(p, x, u), 1))
    s = np.logspace(np.log10(10 * D / np.linalg.norm(QE[:, 1])), np.log10(1000 * D / np.linalg.norm(QE[:, 1])), 9)
    ju_u = np.array([abs(float(gu(zs[0], zs[1] + q))) for q in s])
    sx = np.logspace(np.log10(10 * D / np.linalg.norm(QE[:, 0])), np.log10(1000 * D / np.linalg.norm(QE[:, 0])), 9)
    ju_x = np.array([abs(float(gu(zs[0] + q, zs[1]))) for q in sx])
    rows.append(dict(seed=i, D=D, conv_err_10D=conv[0], conv_err_1000D=conv[-1],
                     conv_exponent=np.polyfit(np.log(rhos), np.log(np.maximum(conv, 1e-300)), 1)[0],
                     along_exp_median=np.median(ea), along_exp_min=np.min(ea), along_exp_max=np.max(ea),
                     across_exp_median=np.median(ec), across_exp_min=np.min(ec), across_exp_max=np.max(ec), n_across=len(ec),
                     Ju_along_u_axis_exp=np.polyfit(np.log(s), np.log(ju_u), 1)[0],
                     Ju_along_x_axis_exp=np.polyfit(np.log(sx), np.log(ju_x), 1)[0]))
    tt = np.linspace(0, 2 * np.pi, 361)
    curves[i] = (tt, np.array([float(Phi(t)) for t in tt]))
    print(f"seed {i}: D {D:.3f} | branch -> Phi: err {conv[0]:.2e} at 10 D, {conv[-1]:.2e} at 1000 D | exponents along "
          f"{np.median(ea):+.3f} [{np.min(ea):+.3f}, {np.max(ea):+.3f}], across {np.median(ec):+.3f} [{np.min(ec):+.3f}, {np.max(ec):+.3f}] | "
          f"J_u branch along u-axis {rows[-1]['Ju_along_u_axis_exp']:+.3f}, along x-axis {rows[-1]['Ju_along_x_axis_exp']:+.3f}", flush=True)
R = pd.DataFrame(rows); R.to_csv(paths.out("theory", "far_field.csv"), index=False)
fig, ax = plt.subplots(figsize=(6, 3.6))
for i, (tt, ph) in curves.items():
    ax.plot(np.degrees(tt), ph, lw=1.2, color="#8B1A1A" if i in (1, 2, 8, 9) else "#888888")
ax.set_xlabel("direction from the kink centre, theta (degrees, coordinates of U)"); ax.set_ylabel("Phi(theta)")
ax.set_title("Far-field branch function, LN MSE 3000 seeds 0-9 (red: 1, 2, 8, 9)", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(paths.out("figs", "fig_far_field.png"), dpi=150, facecolor="white")
print(R[["along_exp_median", "across_exp_median", "Ju_along_u_axis_exp", "Ju_along_x_axis_exp"]].describe().round(3))
