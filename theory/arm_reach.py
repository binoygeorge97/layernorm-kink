import numpy as np, jax, jax.numpy as jnp
from scipy import ndimage
import t2core as T
P, S0, norm, seeds, met = T.load_surrogate("LN MSE 3000")
xg, ug = np.linspace(-4, 4, 641), np.linspace(-10, 10, 801)
X, U = np.meshgrid(xg, ug, indexing="ij")
for i in range(10):
    p, s0 = T.seed_slice(P, i), S0[i]
    ju = jax.jit(jax.vmap(jax.vmap(jax.grad(lambda x, u: T.F_point(p, x, u, s0, "layer"), 1))))(jnp.asarray(X), jnp.asarray(U))
    ju = np.asarray(ju)
    zx, zu, rx, ru = (met[k][i] for k in ("zmin_x", "zmin_u", "rstar_x", "rstar_u"))
    d = np.sqrt(((X - zx) / rx) ** 2 + ((U - zu) / ru) ** 2)
    wrong = ju <= 0
    lab, nlab = ndimage.label(wrong)
    near = np.unique(lab[(d <= 3) & wrong]); near = near[near > 0]       # components touching d <= 3
    conn = np.isin(lab, near)
    edge = wrong & ~conn
    reach_conn = d[conn].max() if conn.any() else np.nan
    reach_all = d[wrong].max() if wrong.any() else np.nan
    cell = (xg[1] - xg[0]) * (ug[1] - ug[0])
    out_share = (conn & (d > 1)).sum() / max(conn.sum(), 1)
    print(f"seed {i}: r*x {rx:.3f} r*u {ru:.3f} | wrong-sign connected to the centre: farthest d = {reach_conn:5.1f}, "
          f"area {conn.sum()*cell:.3f}, outside ellipse {100*out_share:4.0f} % | any wrong-sign in the box: farthest d = {reach_all:6.1f} "
          f"(other components area {edge.sum()*cell:.3f})")
