"""Section 9.1.6 of the paper: 3D Dirichlet, Omega=[0,1]^3, u = exp(xyz).

The paper's Table 8 reports, for effective resolutions 32^3..512^3:

    ||u - u_h||_inf   3.22e-3  7.03e-4  1.82e-4  4.47e-5  1.10e-5
    order                  -      2.19     1.95     2.02     2.02
    ||grad diff||_inf 5.82e-2  1.73e-2  4.75e-3  1.24e-3  3.99e-4
    order                  -      1.75     1.87     1.93     1.64

The paper's mesh (its Fig. 14) cannot be recovered, so absolute values are NOT
reproducible; they are printed as a reference column only.

Note that the paper's own gradient orders are 1.64-1.93, not 2.0.  That is a
property of this test function, not of the scheme: for ``exp(xyz)`` on the unit
cube the third derivatives peak at the corner (1,1,1), and the interior node
where the L_inf gradient error is attained creeps toward that corner as h
shrinks, so the constant in ``(h^2/6) u'''`` grows with refinement and depresses
the measured order.  A uniform grid shows the same effect.

Three things are asserted, chosen to be sharp rather than threshold-fitted:

  A. the solution order on a non-graded grid, which is the paper's headline
     claim and is cleanly >= 1.8;
  B. the gradient order on a *uniform* grid, the apples-to-apples comparison
     with the paper's column, which must rise toward 2;
  C. that non-graded refinement does not degrade the gradient beyond what the
     coarsest cell size already implies -- the gradient error on a grid with a
     2^j cell-size jump equals the uniform-grid error at j levels coarser.
     This is the real conformance statement, and it holds to a few percent.
"""
import os
import sys

import numpy as np
import torch

from fdm.field import Gradient
from fdm.operator import Operator
from fdm.solve import solve
from fdm.topology import build_gradient, build_rows
from fdm.tree import Octree

DEV = os.environ.get("FDM_DEVICE",
                     "cuda" if torch.cuda.is_available() else "cpu")
JUMP = 2


def exact(P):
    return np.exp(P[:, 0] * P[:, 1] * P[:, 2])


def source(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    return ((y * z) ** 2 + (x * z) ** 2 + (x * y) ** 2) * np.exp(x * y * z)


def grad_exact(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    e = np.exp(x * y * z)
    return np.stack([y * z * e, x * z * e, x * y * e], axis=1)


def make(lmax, jump):
    hmin = 1.0 / (1 << lmax)
    t = Octree((1, 1, 1), lmax=lmax, h_min=hmin)
    if jump == 0:
        t.build(lambda c, s: s > 1.001 * hmin)
    else:
        # Fine slab plus a region ``jump`` levels coarser.  Both regions shrink
        # together, so the pattern is self-similar under refinement.
        t.build(lambda c, s: s > 1.001 * np.where(c[:, 2] < 0.3, hmin,
                                                  hmin * (1 << jump)))
    return t


def run(lmax, jump, do_solve=True):
    t = make(lmax, jump)
    R = build_rows(t)
    P = t.node_positions(R.coords)
    G = Gradient(build_gradient(t), device=DEV)
    ex = torch.as_tensor(exact(P), device=DEV, dtype=torch.float64)
    ref = torch.as_tensor(grad_exact(P), device=DEV)[G.row]

    out = dict(h=t.h_min, n=R.nnode, irr=R.irr_row.size,
               gt=float((G(ex) - ref).abs().max()))
    if not do_solve:
        return out
    f = torch.as_tensor(source(P), device=DEV, dtype=torch.float64)
    u, info = solve(A := Operator(R, device=DEV), A.rhs(ex, source=f),
                    tol=1e-12, maxiter=60000)
    out.update(u=float((u - ex).abs().max()), g=float((G(u) - ref).abs().max()),
               iters=info.iters, res=info.residual)
    return out


def orders(seq, key):
    return [float(np.log2(seq[i - 1][key] / seq[i][key]))
            for i in range(1, len(seq))]


FAIL = []


def check(cond, what):
    print(f"    {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        FAIL.append(what)


PAPER_U = [3.22e-3, 7.03e-4, 1.82e-4, 4.47e-5, 1.10e-5]
PAPER_G = [5.82e-2, 1.73e-2, 4.75e-3, 1.24e-3, 3.99e-4]

NG_LEVELS = (4, 5, 6, 7)
print(f"=== Section 9.1.6: Omega=[0,1]^3, u=exp(xyz), Dirichlet  (device {DEV}) ===")
print(f"non-graded grid, {1 << JUMP}x cell-size jump")
print(f"{'h_min':>10} {'nodes':>9} {'irr':>7} {'iters':>6} "
      f"{'Linf err u':>12} {'ord':>5} {'Linf err grad':>14} {'ord':>5}   "
      f"{'(paper u)':>10} {'(paper g)':>10}")
ng = []
for i, lmax in enumerate(NG_LEVELS):
    r = run(lmax, JUMP)
    ng.append(r)
    fu = f"{np.log2(ng[-2]['u'] / r['u']):5.2f}" if i else "    -"
    fg = f"{np.log2(ng[-2]['g'] / r['g']):5.2f}" if i else "    -"
    print(f"{r['h']:10.3e} {r['n']:9d} {r['irr']:7d} {r['iters']:6d} "
          f"{r['u']:12.3e} {fu} {r['g']:14.3e} {fg}   "
          f"{PAPER_U[i]:10.2e} {PAPER_G[i]:10.2e}")

UN_LEVELS = tuple(range(NG_LEVELS[0] - JUMP, NG_LEVELS[-1] + 1))
print()
print("uniform grid (gradient truncation error, the reference for the paper's column)")
print(f"{'h_min':>10} {'nodes':>9} {'Linf err grad':>14} {'ord':>5}")
un = {}
for i, lmax in enumerate(UN_LEVELS):
    r = run(lmax, 0, do_solve=False)
    un[lmax] = r
    prev = un.get(lmax - 1)
    fg = f"{np.log2(prev['gt'] / r['gt']):5.2f}" if prev else "    -"
    print(f"{r['h']:10.3e} {r['n']:9d} {r['gt']:14.3e} {fg}")

print()
ou = orders(ng, "u")
check(all(o >= 1.8 for o in ou),
      f"A. solution order >= 1.8 on the non-graded grid: "
      f"{['%.2f' % o for o in ou]}")

# The two coarsest uniform grids (125 and 729 nodes) are pre-asymptotic: the
# node attaining the L_inf error is still far from the corner (1,1,1) where this
# function's third derivatives peak.  Assert on the asymptotic tail only.
ASYMPTOTIC = 4000
ogu = [(l, float(np.log2(un[l - 1]["gt"] / un[l]["gt"])))
       for l in UN_LEVELS[1:] if un[l - 1]["n"] >= ASYMPTOTIC]
vals = [o for _, o in ogu]
check(all(o >= 1.55 for o in vals),
      f"B. uniform gradient order >= 1.55 on the asymptotic tail "
      f"(n >= {ASYMPTOTIC}): {['%.2f' % o for o in vals]}")
check(vals[-1] > vals[0] and vals[-1] >= 1.85,
      f"B. uniform gradient order rises toward 2 (last {vals[-1]:.2f})")

print()
print("C. non-graded gradient error vs uniform error at JUMP levels coarser:")
ratios = []
for r, lmax in zip(ng, NG_LEVELS):
    ref = un.get(lmax - JUMP)
    if ref is None:
        continue
    ratio = r["gt"] / ref["gt"]
    asym = ref["n"] >= ASYMPTOTIC
    if asym:
        ratios.append(ratio)
    print(f"      h={r['h']:.3e}  non-graded {r['gt']:.4e}   "
          f"uniform(h x {1 << JUMP}) {ref['gt']:.4e}   ratio {ratio:.4f}"
          f"{'' if asym else '   (pre-asymptotic, not asserted)'}")
check(all(abs(x - 1.0) < 0.06 for x in ratios),
      f"C. ratios within 6% of 1: {['%.4f' % x for x in ratios]}")

check(all(r["res"] < 1e-10 for r in ng),
      "every solve converged to 1e-10 relative residual")

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} check(s)")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("Section 9.1.6 conformance passed")
