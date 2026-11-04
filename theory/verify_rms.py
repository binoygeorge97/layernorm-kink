"""Q7a (2026-11-04): numerical check of the RMSNorm version of Theorem 1 (no centring, P = I), float64.
1. RMS(Ez + b) = sqrt(H) (E (z - z*) + c_perp) / sqrt(||E (z - z*)||^2 + D^2), z* = -(E^T E)^{-1} E^T b,
   c_perp = E z* + b, D^2 = ||c_perp||^2 + H eps; and = V phi(w) with w = U^T E (z - z*) (U = left singular vectors of E).
2. Peak gain: the largest singular value of d RMS / d z at z* equals sqrt(H) sigma_max(E) / D.
3. Comparison with LayerNorm for the SAME (E, b): ||c_perp(RMS)|| >= ||c_perp(LN)||, hence D_RMS >= D_LN (P is a
   contraction); equality iff the mean of E z_RMS* + b ... checked on random draws and on the trained LN MSE 3000 weights.
Prints the maximum errors."""
import os, sys
import numpy as np
rng = np.random.default_rng(1)
H = 16
P = np.eye(H) - np.ones((H, H)) / H


def rms(h, eps):
    return h / np.sqrt(h @ h / H + eps)


def red(E, b, eps):
    zs = -np.linalg.solve(E.T @ E, E.T @ b)
    cp = E @ zs + b
    return zs, cp, np.sqrt(cp @ cp + H * eps)


def cperp_ln(E, b):
    QE, c = P @ E, P @ b
    zs = -np.linalg.lstsq(QE, c, rcond=None)[0]
    return QE @ zs + c


err1 = err1b = err2 = 0.0
viol = 0
for trial in range(300):
    E = rng.standard_normal((H, 2)) * rng.uniform(0.1, 3)
    b = rng.standard_normal(H) * rng.uniform(0, 2)
    eps = [0.0, 1e-6, 1e-2][trial % 3]
    if eps == 0.0:
        b = b + 1e-3 * rng.standard_normal(H)
    zs, cp, D = red(E, b, eps)
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    for _ in range(5):
        z = rng.standard_normal(2) * 3
        lhs = rms(E @ z + b, eps)
        e_ = E @ (z - zs)
        rhs = np.sqrt(H) * (e_ + cp) / np.sqrt(e_ @ e_ + D ** 2)
        w = U.T @ E @ (z - zs)
        rhs2 = np.sqrt(H) * (U @ w + cp) / np.sqrt(w @ w + D ** 2)
        err1 = max(err1, np.abs(lhs - rhs).max()); err1b = max(err1b, np.abs(lhs - rhs2).max())
    # Jacobian at z* by central differences vs closed form
    J = np.column_stack([(rms(E @ (zs + d) + b, eps) - rms(E @ (zs - d) + b, eps)) / 2e-6 for d in np.eye(2) * 1e-6])
    err2 = max(err2, abs(np.linalg.svd(J, compute_uv=False)[0] / (np.sqrt(H) * s[0] / D) - 1))
    if np.linalg.norm(cp) < np.linalg.norm(cperp_ln(E, b)) - 1e-12:
        viol += 1
print(f"1. reduction max abs error {err1:.2e} (V phi form {err1b:.2e})")
print(f"2. peak gain at z*: max relative error vs sqrt(H) sigma_max / D: {err2:.2e} (finite differences, step 1e-6)")
print(f"3. ||c_perp RMS|| < ||c_perp LN|| on {viol} of 300 random draws (theory: never)")
# trained LN MSE 3000 weights, read with the RMS formula (illustration: the same (E, b) under RMSNorm)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import paths
    z = np.load(os.path.join(paths.DATA, "surrogates", "MSE__3000.npz"), allow_pickle=True)
    Es, bs = z["p:E"], z["p:b"]
    print("3b. trained LN MSE 3000 weights: seed, D_LN, D_RMS (same E, b), mean(b), ||P b||, ||b||")
    for i in range(Es.shape[0]):
        E, b = Es[i].T, bs[i]
        dl = np.sqrt(np.sum(cperp_ln(E, b) ** 2) + H * 1e-6); dr = red(E, b, 1e-6)[2]
        print(f"    seed {i}: D_LN {dl:.3f}  D_RMS {dr:.3f}  mean(b) {b.mean():+.3f}  ||Pb|| {np.linalg.norm(P @ b):.3f}  ||b|| {np.linalg.norm(b):.3f}")
except Exception as ex:  # noqa
    print("3b skipped:", ex)
