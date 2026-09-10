"""What the paper's transverse cancellation actually buys, vs the naive scheme.

Previously reported in RESULTS.md as "Test 4" from an ad-hoc run against the
superseded symmetric stencil, with no script in the repo.  This is that
measurement, reproducible, against the paper's construction.

At fixed ``lmax`` the coarse/fine level jump is varied.  ``interface_correction
= False`` forces the paper's weights to 1, which is the Losasso-style scheme
that is locally inconsistent at hanging nodes; ``True`` applies Eq. (8).

The solution is the harmonic sin(pi x) sin(pi y) exp(sqrt(2) pi z), so the
Laplacian is exactly zero and the only error is the discretization's.
"""
import os

import numpy as np
import torch

from fdm.field import Gradient
from fdm.operator import Operator
from fdm.solve import solve
from fdm.topology import build_gradient, build_rows
from fdm.tree import Octree

DEV = os.environ.get("FDM_DEVICE",
                     "cuda" if torch.cuda.is_available() else "cpu")
K = np.pi
LMAX = 5


def phi(P):
    return np.sin(K * P[:, 0]) * np.sin(K * P[:, 1]) * np.exp(np.sqrt(2) * K * P[:, 2])


def grad(P):
    s = np.sqrt(2) * K
    e = np.exp(s * P[:, 2])
    sx, sy = np.sin(K * P[:, 0]), np.sin(K * P[:, 1])
    cx, cy = np.cos(K * P[:, 0]), np.cos(K * P[:, 1])
    return np.stack([K * cx * sy * e, K * sx * cy * e, s * sx * sy * e], 1)


def run(jump, correction):
    hmin = 1.0 / (2 * (1 << LMAX))
    t = Octree((2, 2, 2), lmax=LMAX, h_min=hmin)
    t.build(lambda c, s: s > 1.001 * np.where(c[:, 2] < 0.3, hmin,
                                              hmin * (1 << jump)))
    R = build_rows(t, interface_correction=correction)
    A = Operator(R, device=DEV)
    P = t.node_positions(R.coords)
    ex = torch.as_tensor(phi(P), device=DEV, dtype=torch.float64)
    u, info = solve(A, A.rhs(ex), tol=1e-12, maxiter=40000)

    G = Gradient(build_gradient(t), device=DEV)
    ge = torch.as_tensor(grad(P), device=DEV)[G.row]
    iface = torch.as_tensor(R.irr_row, device=DEV, dtype=torch.long)
    err = (u - ex).abs()
    return dict(nodes=R.nnode, irr=R.irr_row.size, iters=info.iters,
                u=float(err.max()),
                ui=float(err[iface].max()) if iface.numel() else 0.0,
                g=float((G(u) - ge).abs().max()))


print(f"=== the paper's Eq. (8) cancellation vs forcing the weights to 1 "
      f"(lmax={LMAX}, device {DEV}) ===")
print(f"{'jump':>5} {'ratio':>6} {'nodes':>8} {'irr':>6} | "
      f"{'Linf err phi':>13} {'naive':>12} {'change':>8} | "
      f"{'err @iface':>11} {'naive':>11} {'change':>8} | "
      f"{'Linf err grad':>13} {'naive':>12} {'change':>8}")
for jump in (1, 2, 3, 4):
    c = run(jump, True)
    nv = run(jump, False)
    def rel(a, b):
        return f"{100 * (b - a) / a:+7.2f}%" if a > 0 else "      -"
    print(f"{jump:>5} {1 << jump:>5}x {c['nodes']:8d} {c['irr']:6d} | "
          f"{c['u']:13.4e} {nv['u']:12.4e} {rel(c['u'], nv['u'])} | "
          f"{c['ui']:11.4e} {nv['ui']:11.4e} {rel(c['ui'], nv['ui'])} | "
          f"{c['g']:13.4e} {nv['g']:12.4e} {rel(c['g'], nv['g'])}")

print()
print("=== local consistency: the residual on quadratics, where Eq. (1) and the")
print("=== multilinear interpolation are both exact, so a consistent scheme must")
print("=== return the Laplacian to round-off")
print(f"{'jump':>5} {'corrected':>12} {'naive (w=1)':>13}")
MONO = [("x^2", lambda P: P[:, 0] ** 2, 2.0),
        ("y^2", lambda P: P[:, 1] ** 2, 2.0),
        ("z^2", lambda P: P[:, 2] ** 2, 2.0),
        ("xy", lambda P: P[:, 0] * P[:, 1], 0.0),
        ("yz", lambda P: P[:, 1] * P[:, 2], 0.0),
        ("zx", lambda P: P[:, 2] * P[:, 0], 0.0)]


def apply(R, u):
    Au = np.zeros(R.nnode)
    Au[R.reg_row] = (R.reg_val * u[R.reg_idx]).sum(1)
    if R.irr_row.size:
        Au[R.irr_row] = (R.irr_val * u[R.irr_idx]).sum(1)
    return Au


for jump in (1, 2, 3, 4):
    out = []
    for corr in (True, False):
        hmin = 1.0 / (2 * (1 << LMAX))
        t = Octree((2, 2, 2), lmax=LMAX, h_min=hmin)
        t.build(lambda c, s: s > 1.001 * np.where(c[:, 2] < 0.3, hmin,
                                                  hmin * (1 << jump)))
        R = build_rows(t, interface_correction=corr)
        P = t.node_positions(R.coords)
        free = ~R.boundary
        out.append(max(np.abs(apply(R, f(P))[free] - lap).max()
                       for _, f, lap in MONO))
    print(f"{jump:>5} {out[0]:12.3e} {out[1]:13.3e}")

print("""
How to read the two tables together.

The naive scheme is inconsistent by O(1): forcing the weights to 1 leaves a
residual of 0.3-0.4 on quadratics at every jump ratio, and refining does not
reduce it.  The paper's Eq. (8) cancellation is exact to round-off.  That is the
statement about the scheme.

The first table is a statement about one solution.  Its L_inf error over the
whole grid is dominated by the coarse region's own cell size, which grows as
4^jump, not by the interface treatment -- interface nodes are codimension-one,
which is the supra-convergence argument holding in practice.  For this
particular harmonic the naive scheme's inconsistency happens to cancel
favourably at interface nodes, so its error there is *smaller*.  That is a
property of this test function, not evidence that inconsistency is harmless: it
is exactly the kind of coincidence the exact-identity table above exists to rule
out, and it should not be used to argue the correction is optional.""")
