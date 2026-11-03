"""Task 2b proposal (no implementation): check candidate two-state plants against the CLAUDE.md criteria.
For each plant: eigenvalues, controllability, LQR (Q = diag(5, 0.5), R = 0.1) gain and closed-loop poles, equilibrium of a
position setpoint x1 = r (x2_ss(r), u_ss(r)), holdable setpoints, recoverable set of the saturated LQR (|u| <= 10) by
simulation from a grid, data box (1.2 x the recoverable set's bounding box), and the safe band of the 2x2 closed-loop
multiplier M = J_x_hat - J_u_hat K when the model's input slope is J_u_hat = g B (spectral radius < 1), against the scalar
plant's band. Output: out/plant2/plant2_summary.csv, figure out/plant2/fig_recoverable.png, printed table."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import solve_discrete_are  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UMAX, Q, R = 10.0, np.diag([5.0, 0.5]), np.array([[0.1]])
PLANTS = {
    "P1 (CLAUDE.md example)": (np.array([[1.02, 0.05], [0.0, 1.01]]), np.array([[0.0], [0.01]])),
    "P2 (scalar plant + stable lag)": (np.array([[0.98, 0.05], [0.0, 1.03]]), np.array([[0.0], [0.01]])),
    "P3 (unstable oscillation)": (np.array([[1.0, 0.1], [-0.05, 1.01]]), np.array([[0.0], [0.01]])),
}


def lqr(A, B, Qm=Q):
    P = solve_discrete_are(A, B, Qm, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def equilibrium(A, B, r):
    """x1 = r: solve (A - I) [r, x2] + B u = 0 for (x2, u) (least squares if inconsistent)."""
    M = np.column_stack([(A - np.eye(2))[:, 1], B[:, 0]])
    sol, res, rank, _ = np.linalg.lstsq(M, -(A - np.eye(2))[:, 0] * r, rcond=None)
    return sol[0], sol[1], rank


def recoverable(A, B, K, lim, n=161, steps=4000):
    g1, g2 = np.meshgrid(np.linspace(-lim[0], lim[0], n), np.linspace(-lim[1], lim[1], n))
    X = np.stack([g1.ravel(), g2.ravel()])
    ok = np.ones(X.shape[1], bool)
    for _ in range(steps):
        u = np.clip(-K @ X, -UMAX, UMAX)
        X = A @ X + B @ u
        ok &= np.isfinite(X).all(0) & (np.abs(X).max(0) < 1e3)
        X[:, ~ok] = 0.0
    ok &= np.abs(X).max(0) < 1e-3
    return g1, g2, ok.reshape(g1.shape)


def band(A, B, K, gs=np.linspace(-3, 60, 6301)):
    rho = np.array([np.max(np.abs(np.linalg.eigvals(A - g * B @ K))) for g in gs])
    inside = gs[rho < 1]
    lo = inside.min() if len(inside) else np.nan
    hi = inside.max() if len(inside) else np.nan
    rho_wrong = np.max(np.abs(np.linalg.eigvals(A + B @ K)))      # g = -1: wrong-signed input slope
    return lo, hi, rho_wrong


def main():
    rows = []
    fig, ax = plt.subplots(1, len(PLANTS), figsize=(5 * len(PLANTS), 4.2))
    for k, (name, (A, B)) in enumerate(PLANTS.items()):
        ev = np.linalg.eigvals(A)
        ctrb = np.linalg.matrix_rank(np.column_stack([B, A @ B]))
        K, P = lqr(A, B)
        cl = np.linalg.eigvals(A - B @ K)
        x2r, ur, rk = equilibrium(A, B, 1.0)
        lim = (40.0, 40.0)
        for _ in range(3):                                               # widen / shrink the grid to the recoverable set
            g1, g2, ok = recoverable(A, B, K, lim)
            ext = (np.abs(g1[ok]).max(), np.abs(g2[ok]).max()) if ok.any() else lim
            if ext[0] < 0.9 * lim[0] and ext[1] < 0.9 * lim[1]:
                lim = (1.15 * ext[0], 1.15 * ext[1])
            else:
                lim = (2 * lim[0], 2 * lim[1])
        g1, g2, ok = recoverable(A, B, K, lim)
        ext = (np.abs(g1[ok]).max(), np.abs(g2[ok]).max())
        r_hold = UMAX / abs(ur) if abs(ur) > 1e-12 else np.inf
        # setpoints whose equilibrium lies inside the recoverable set, with input margin 10 %
        rs = np.linspace(0, ext[0], 400)
        inside = [r for r in rs if abs(ur * r) <= 0.9 * UMAX and abs(x2r * r) <= ext[1]]
        lo, hi, rho_wrong = band(A, B, K)
        rows.append(dict(plant=name, eig=" ".join(f"{e:.4f}" for e in ev), ctrb_rank=ctrb,
                         K=f"[{K[0, 0]:.3f}, {K[0, 1]:.3f}]", cl_poles=" ".join(f"{e:.4f}" for e in cl),
                         spec_radius_cl=float(np.max(np.abs(cl))), x2_ss_per_r=x2r, u_ss_per_r=ur,
                         recoverable_x1=ext[0], recoverable_x2=ext[1], data_box_x1=1.2 * ext[0], data_box_x2=1.2 * ext[1],
                         r_hold_input=r_hold, r_max_setpoint=max(inside) if inside else 0.0,
                         band_g_lo=lo, band_g_hi=hi, rho_wrong_sign=rho_wrong))
        a = ax[k]
        a.contourf(g1, g2, ok.astype(float), levels=[0.5, 1.5], colors=["#1B3A6B"], alpha=0.35)
        a.contour(g1, g2, ok.astype(float), levels=[0.5], colors="#1B3A6B", linewidths=1)
        rr = np.linspace(-rows[-1]["r_max_setpoint"], rows[-1]["r_max_setpoint"], 2)
        a.plot(rr, x2r * rr, "--", color="#8B1A1A", lw=1.2, label="equilibria of setpoints x1 = r")
        a.set_title(f"{name}\nrecoverable set, saturated LQR, |u| <= 10", fontsize=9)
        a.set_xlabel("x1"), a.set_ylabel("x2"), a.legend(fontsize=7), a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths.out("plant2", "fig_recoverable.png"), dpi=150, facecolor="white")
    # the scalar reference plant for comparison
    A1, B1 = np.array([[1.03]]), np.array([[0.01]])
    P1 = solve_discrete_are(A1, B1, np.array([[5.0]]), R)
    K1 = float(np.linalg.solve(R + B1.T @ P1 @ B1, B1.T @ P1 @ A1)[0, 0])
    D = pd.DataFrame(rows)
    D.to_csv(paths.out("plant2", "plant2_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(D.T.to_string())
    print(f"\nscalar reference: K* = {K1:.4f}, pole {1.03 - 0.01 * K1:.3f}, band (g) {(1.03 - 1) / K1 / 0.01:.3f} .. "
          f"{(1.03 + 1) / K1 / 0.01:.2f}, wrong sign |m| = {1.03 + 0.01 * K1:.3f}, holdable |x| <= 3.33")
    return D


if __name__ == "__main__":
    main()
