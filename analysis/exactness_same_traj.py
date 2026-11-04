"""B2: the memoryless exactness check sum log|m_k| = log|dx_N/dx_0| recomputed along ONE stored trajectory.
For a run (family, seed, rep) and the start with the largest disagreement in t2_arch:
  (1) one-step jvp of x -> F(x, pi(x)) at each stored x_k vs m_k = J_x_hat - J_u_hat K_k (same point)
  (2) product of the one-step jvps along the stored trajectory vs exp(sum log|m_k|)
  (3) the primal trajectory that jax.jvp of the full rollout computes internally vs the stored one (the second rollout)
Usage: python analysis/exactness_same_traj.py <controller root> <family> <seed> <rep> [...]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory"))
import numpy as np, jax, jax.numpy as jnp
import t2core as T, t2_arch as A

def check(root, fam, s, rep):
    P, S0, norm, seeds, met = T.load_surrogate(fam); ps, _ = A.make_fns(norm)
    C, cs = A.load_ref(root, rep); c = T.seed_slice(C, cs.index(s)); p = T.seed_slice(P, s); s0 = S0[s]; X0 = T.training_starts(s, rep)
    xs, us, jx, ju, K, m, sens = (np.asarray(a) for a in ps(c, p, s0, X0))
    S = np.log(np.abs(m) + 1e-300).sum(1); E = np.log(np.abs(sens) + 1e-300)
    b = int(np.argmax(np.abs(S - E) * (S > -600)))
    step = lambda x: T.F_point(p, x, A.pi(c, x), s0, norm)
    d1 = np.asarray(jax.vmap(lambda x: jax.jvp(step, (x,), (jnp.ones_like(x),))[1])(jnp.asarray(xs[b, :-1])))
    rel = np.abs(d1 - m[b]) / np.maximum(np.abs(m[b]), 1e-300)
    logprod = np.log(np.abs(d1) + 1e-300).sum()
    def roll(a):
        def st(x, _):
            xn = step(x); return xn, xn
        return jax.lax.scan(st, a, None, length=T.N_HOR)[1]
    prim, tang = jax.jvp(roll, (jnp.asarray(X0[b]),), (jnp.ones_like(X0[b]),))
    dev = np.abs(np.asarray(prim) - xs[b, 1:])
    k_sep = int(np.argmax(dev > 1e-6)) + 1 if (dev > 1e-6).any() else None
    print(f"{os.path.basename(root)} {fam} s{s} r{rep} start {float(X0[b]):+.4f}: sum log|m| {S[b]:.3f} | log|dx_N/dx_0| (second rollout) {E[b]:.3f} "
          f"| (1) one-step jvp vs m_k: max rel diff {rel.max():.1e} | (2) sum log|one-step jvp| on the stored path {logprod:.3f} "
          f"(diff to sum log|m| {abs(logprod - S[b]):.1e}) | (3) second rollout leaves the stored path by > 1e-6 at step {k_sep}, "
          f"first deviation > 1e-12 at step {int(np.argmax(dev > 1e-12)) + 1 if (dev > 1e-12).any() else None}, max |dx| {dev.max():.2e}")

if __name__ == "__main__":
    a = sys.argv[1:]
    for i in range(0, len(a), 4):
        check(a[i], a[i + 1], int(a[i + 2]), int(a[i + 3]))
