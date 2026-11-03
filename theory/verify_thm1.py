"""Numerical checks of Theorem 1 and Corollaries 1-5 (prenorm LayerNorm reduction), float64.
1. LN(Ez + b) = V phi(w(z)) for random E, b (full rank, rank-deficient Q_E, k = H-1), eps in {0, 1e-6, 1e-2}.
2. Singular values of D phi against the closed forms (tangential, radial), general eps.
3. eps = 0: phi maps R^k onto the open hemisphere; area of the image = half the area of the radius-sqrt(H) k-sphere
   (k = 1: pi sqrt(H); k = 2: 2 pi H), independent of D.
4. Block Jacobian of a full prenorm block (random branch) = skip + w_d^T J_G diag(gamma) V D phi U^T Q_E.
5. c_perp = 0 exactly iff b in range(E) + span(1); random b gives c_perp != 0.
6. On the trained LN MSE 3000 surrogates: the network output computed through the reduction equals F_point."""
import numpy as np
from scipy import integrate

rng = np.random.default_rng(0)
H = 16
P = np.eye(H) - np.ones((H, H)) / H


def ln(h, eps):
    r = P @ h
    return r / np.sqrt(r @ r / H + eps)


def reduction(E, b, eps, tol=1e-12):
    """z*, U, v, c_perp, D, k, w(.) and phi(.) of Theorem 1."""
    QE, c = P @ E, P @ b
    Uf, sv, _ = np.linalg.svd(QE, full_matrices=True)
    k = int((sv > tol * max(1.0, sv.max())).sum())
    U = Uf[:, :k]
    Pi = U @ U.T
    cp = c - Pi @ c
    zs = -np.linalg.pinv(QE) @ c
    D = np.sqrt(cp @ cp + H * eps)
    a = np.linalg.norm(cp)
    if a > 1e-12 * (1.0 + np.linalg.norm(c)):
        v = cp / a
    elif k < H - 1:  # any unit vector orthogonal to range(U) and to 1
        M = np.column_stack([U, np.ones(H) / np.sqrt(H)])
        Qf, _ = np.linalg.qr(np.column_stack([M, rng.standard_normal(H)]))
        v = Qf[:, -1]
    else:
        v = None
    if v is None or a <= 1e-12 * (1.0 + np.linalg.norm(c)):
        a = 0.0
    w = lambda z: U.T @ QE @ (z - zs)
    def phi(wv):
        num = np.append(wv, a) if v is not None else wv
        return np.sqrt(H) * num / np.sqrt(wv @ wv + D ** 2)
    V = np.column_stack([U, v]) if v is not None else U
    return dict(QE=QE, c=c, zs=zs, U=U, v=v, V=V, cp=cp, D=D, a=a, k=k, w=w, phi=phi)


def check_identity(m, rankdef=0, eps=1e-6, n=200, bias="random"):
    E = rng.standard_normal((H, m))
    if rankdef:
        E[:, -rankdef:] = E[:, :rankdef] @ rng.standard_normal((rankdef, rankdef))
    b = rng.standard_normal(H) if bias == "random" else (E @ rng.standard_normal(m) + 0.7 * np.ones(H))
    R = reduction(E, b, eps)
    err = 0.0
    for _ in range(n):
        z = 3 * rng.standard_normal(m)
        if eps == 0 and R["D"] == 0 and np.linalg.norm(R["w"](z)) < 1e-9:
            continue
        lhs = ln(E @ z + b, eps)
        rhs = R["V"] @ R["phi"](R["w"](z))
        err = max(err, np.abs(lhs - rhs).max())
    orth = np.abs(R["V"].T @ R["V"] - np.eye(R["V"].shape[1])).max()
    perp1 = np.abs(R["V"].T @ np.ones(H)).max()
    return err, R["k"], R["a"], orth, perp1


print("1. identity LN(Ez+b) = V phi(w(z)); max abs error over 200 random z")
for m, rd, label in [(2, 0, "m=2, full rank"), (5, 2, "m=5, rank 3"), (15, 0, "m=15, k=H-1"), (20, 0, "m=20 > H, k=H-1")]:
    for eps in (0.0, 1e-6, 1e-2):
        for bias in ("random", "in range(E)+span(1)"):
            e, k, a, orth, p1 = check_identity(m, rd, eps, bias=bias)
            print(f"   {label:18s} eps={eps:<6g} bias {bias:20s}: k={k:2d} ||c_perp||={a:9.2e} err={e:8.1e} "
                  f"V^T V - I {orth:7.1e}  V^T 1 {p1:7.1e}")

print("2. singular values of D phi vs closed forms (k = 4, several rho, eps)")
worst = 0.0
for eps in (0.0, 1e-6, 1e-2, 0.5):
    for a in (0.0, 0.3, 2.0):
        k = 4
        if a == 0 and eps == 0:
            continue
        D = np.sqrt(a * a + H * eps)
        for rho in (0.0, 0.01, 0.3, 1.0, 10.0, 1e3):
            wv = rng.standard_normal(k); wv *= rho / max(np.linalg.norm(wv), 1e-300) if rho > 0 else 0.0
            S = rho ** 2 + D ** 2
            Jac = np.sqrt(H) * (np.vstack([np.eye(k), np.zeros((1, k))]) / np.sqrt(S)
                                - np.outer(np.append(wv, a), wv) / S ** 1.5)
            sv = np.sort(np.linalg.svd(Jac, compute_uv=False))
            tan = np.sqrt(H) / np.sqrt(S)
            rad = np.sqrt(H) * np.sqrt(D ** 4 + a * a * rho ** 2) / S ** 1.5
            pred = np.sort(np.append(np.full(k - 1, tan), rad))
            worst = max(worst, np.max(np.abs(sv - pred) / pred))
            if eps == 0 and a > 0:
                worst = max(worst, abs(rad - np.sqrt(H) * D / S) / rad)
print(f"   max relative error over all cases: {worst:.1e}")
print("   centre: tangential = radial = sqrt(H)/D; far field: tangential ~ sqrt(H)/rho, radial ~ sqrt(H) a/rho^2")

print("3. eps = 0: image is the open hemisphere, area independent of D")
for D in (1e-3, 0.13, 0.9, 7.0):
    # k = 1: integral of the radial singular value
    I1 = integrate.quad(lambda w: np.sqrt(H) * D / (w * w + D * D), -np.inf, np.inf, limit=500)[0]
    # k = 2: integral over the plane of the area element sqrt(H)^2 D / (rho^2 + D^2)^{3/2}, polar coordinates
    I2 = integrate.quad(lambda r: 2 * np.pi * r * H * D / (r * r + D * D) ** 1.5, 0, np.inf, limit=500)[0]
    print(f"   D={D:<6g}: k=1 length {I1:.10f} (pi sqrt(H) = {np.pi*np.sqrt(H):.10f}); "
          f"k=2 area {I2:.8f} (2 pi H = {2*np.pi*H:.8f}); peak gain sqrt(H)/D = {np.sqrt(H)/D:.4g}")
E = rng.standard_normal((H, 2)); b = rng.standard_normal(H)
R = reduction(E, b, 0.0)
pts = np.array([R["phi"](10 ** rng.uniform(-3, 4) * rng.standard_normal(2)) for _ in range(2000)])
print(f"   random images: | ||phi|| - sqrt(H) | max {np.abs(np.linalg.norm(pts, axis=1) - np.sqrt(H)).max():.1e}, "
      f"min last coordinate {pts[:, -1].min():.2e} (> 0: open upper hemisphere)")
y = np.array([0.6, -0.8, 1.0]); y = np.sqrt(H) * y / np.linalg.norm(y)
winv = R["a"] * y[:2] / y[2]
print(f"   inverse gnomonic: phi(D y_1:k / y_k+1) = y, error {np.abs(R['phi'](winv) - y).max():.1e}")

print("4. block Jacobian of a full prenorm block with a random GELU branch")
from scipy.special import erf
gelu = lambda t: 0.5 * t * (1 + erf(t / np.sqrt(2)))
dgelu = lambda t: 0.5 * (1 + erf(t / np.sqrt(2))) + t * np.exp(-t * t / 2) / np.sqrt(2 * np.pi)
W1, b1, W2, b2 = rng.standard_normal((H, 32)) / 4, rng.standard_normal(32), rng.standard_normal((32, H)) / 6, rng.standard_normal(H)
gam, bet, wd, bd = rng.uniform(0.5, 1.5, H), 0.1 * rng.standard_normal(H), rng.standard_normal(H), 0.3
eps = 1e-6
E = rng.standard_normal((H, 2)); b = 0.3 * rng.standard_normal(H)
R = reduction(E, b, eps)
def Fz(z):
    h = E @ z + b
    y = gam * ln(h, eps) + bet
    return wd @ (h + W2.T @ gelu(W1.T @ y + b1) + b2) + bd
def grad_formula(z):
    wv = R["w"](z); k = R["k"]
    S = wv @ wv + R["D"] ** 2
    Dphi = np.sqrt(H) * (np.vstack([np.eye(k), np.zeros((1, k))]) / np.sqrt(S) - np.outer(np.append(wv, R["a"]), wv) / S ** 1.5)
    y = gam * (R["V"] @ R["phi"](wv)) + bet
    JG = W2.T @ np.diag(dgelu(W1.T @ y + b1)) @ W1.T
    return wd @ E + wd @ JG @ np.diag(gam) @ R["V"] @ Dphi @ R["U"].T @ R["QE"]
err = 0.0
for _ in range(100):
    z = R["zs"] + 0.5 * rng.standard_normal(2) * rng.choice([0.01, 0.3, 3.0])
    num = np.array([(Fz(z + 1e-6 * e) - Fz(z - 1e-6 * e)) / 2e-6 for e in np.eye(2)])
    err = max(err, np.abs(num - grad_formula(z)).max() / max(1.0, np.abs(num).max()))
print(f"   formula vs central differences, max relative error {err:.1e}")

print("5. c_perp = 0 iff b in range(E) + span(1)")
E = rng.standard_normal((H, 2))
for name, b in [("b = 0 (initialisation)", np.zeros(H)), ("b = E t + 2.5 * 1", E @ np.array([0.3, -1.2]) + 2.5),
                ("random b", rng.standard_normal(H)), ("random b, scale 1e-6", 1e-6 * rng.standard_normal(H))]:
    R = reduction(E, b, 1e-6)
    print(f"   {name:24s}: ||c_perp|| = {R['a']:.3e}, D = {R['D']:.4f} (sqrt(H eps) = {np.sqrt(H*1e-6):.4f}), z* = {np.round(R['zs'], 4)}")

print("6. trained LN MSE 3000 surrogates: F through the reduction vs F_point")
import t2core as T, jax.numpy as jnp
Pz, S0, norm, seeds, met = T.load_surrogate("LN MSE 3000")
worst = 0.0
for i in range(10):
    p = {k: np.asarray(v) for k, v in T.seed_slice(Pz, i).items() if k != "br"}
    br = {k: np.asarray(v) for k, v in T.seed_slice(Pz, i)["br"].items()}
    E, b = p["E"].T, p["b"]
    R = reduction(E, b, T.EPS)
    for _ in range(50):
        x, u = rng.uniform(-4, 4), rng.uniform(-10, 10)
        if rng.random() < 0.5:
            x, u = R["zs"] + 0.05 * rng.standard_normal(2)
        z = np.array([x, u])
        y = p["gamma"] * (R["V"] @ R["phi"](R["w"](z))) + p["beta"]
        g = gelu(y @ br["W1"] + br["b1"]) @ br["W2"] + br["b2"]
        Fred = (E @ z + b) @ p["Wd"][:, 0] + g @ p["Wd"][:, 0] + p["bd"][0]
        Fref = float(T.F_point(T.seed_slice(Pz, i), x, u, S0[i], "layer"))
        worst = max(worst, abs(Fred - Fref))
print(f"   max |F_reduced - F_point| over 10 seeds x 50 points: {worst:.1e}")
