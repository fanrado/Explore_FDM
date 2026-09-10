"""Section 9.1.7 of the paper: 3D all-Neumann, and the Section 7 procedure.

    Omega = [0, pi]^3,   u = cos x cos y cos z - 1,   du/dn = 0 on all six faces

``du/dn`` carries a ``sin`` on every face, which vanishes at 0 and pi, so the
homogeneous Neumann condition is exact.  ``f = laplacian u = -3 cos x cos y
cos z`` integrates to zero over the box, and ``u(0,0,0) = 0``, so the pinned
corner value is exactly 0.

WHY THE PAPER'S TABLE 9 IS NOT REPRODUCED
-----------------------------------------
Table 9 has two columns, gradient error without and with the Section 7
treatment.  The "without" column *stalls* (orders 2.82, 0.22, 0.18, 0.11),
which is the localized pollution Section 7 exists to remove; the "with" column
recovers order 2.

That stall does not occur here, and the reason is measurable rather than
speculative.  The paper never states how it discretizes ``du/dn = 0``; this code
uses the mirrored stencil in :mod:`fdm.topology`, whose rows sum to zero, so the
constant vector is *exactly* in the operator's nullspace (asserted below).  The
discrete right-hand side is then compatible, the pinned solution satisfies the
full unpinned system including the pinned node's own row (also asserted), and
the solution differs from the unpinned family only by the constant the pin
selects.  A constant has zero gradient, so the pin cannot corrupt the gradient
anywhere -- near the corner or otherwise.

So Section 7 is implemented and exercised end to end here, but with this
discretization it provably has nothing to repair, and its step-3 patch is a
no-op.  That is asserted too: it is a stronger statement than "the stall did not
appear", and it is what makes the absence of the stall a result rather than a
gap.  A discretization whose Neumann rows do not annihilate constants would
reintroduce the paper's defect, and this module would then repair it.
"""
import os
import sys

import numpy as np
import torch

from fdm.field import Gradient
from fdm.neumann import solve_all_neumann
from fdm.operator import Operator
from fdm.topology import build_gradient, build_rows
from fdm.tree import Octree

DEV = os.environ.get("FDM_DEVICE",
                     "cuda" if torch.cuda.is_available() else "cpu")
L = np.pi
JUMP = 1


def exact(P):
    return np.cos(P[:, 0]) * np.cos(P[:, 1]) * np.cos(P[:, 2]) - 1.0


def source(P):
    return -3.0 * np.cos(P[:, 0]) * np.cos(P[:, 1]) * np.cos(P[:, 2])


def grad_exact(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    cx, cy, cz = np.cos(x), np.cos(y), np.cos(z)
    sx, sy, sz = np.sin(x), np.sin(y), np.sin(z)
    return np.stack([-sx * cy * cz, -cx * sy * cz, -cx * cy * sz], axis=1)


def build(lmax):
    hmin = L / (1 << lmax)
    t = Octree((1, 1, 1), lmax=lmax, h_min=hmin)
    t.build(lambda c, s: s > 1.001 * np.where(c[:, 2] < 0.3 * L, hmin,
                                              hmin * (1 << JUMP)))
    coords, _ = t.nodes()
    D = t.dims_units
    # Neumann on the entire boundary: every node lying on a domain face.
    nm = np.any((coords == 0) | (coords == D[None, :]), axis=1)
    return t, build_rows(t, neumann=nm), nm


def run(lmax, sub=4, patch=2):
    t, R, nm = build(lmax)
    P = t.node_positions(R.coords)
    f = torch.as_tensor(source(P), device=DEV, dtype=torch.float64)
    out = solve_all_neumann(t, R, source=f, pin_value=0.0, sub=sub, patch=patch,
                            device=DEV, neumann=nm)

    ref = torch.as_tensor(grad_exact(P), device=DEV)[out["grad_row"]]
    ex = torch.as_tensor(exact(P), device=DEV, dtype=torch.float64)
    du = out["u"] - ex

    # constants in the nullspace, and compatibility of the pinned solve
    A0 = Operator(R, dirichlet=np.zeros(R.nnode, dtype=bool), device=DEV)
    one = torch.ones(R.nnode, dtype=torch.float64, device=DEV)
    null = float(A0(one).abs().max()) * t.h_min ** 2
    full_res = float((A0(out["u"]) - f).abs().max()) / float(f.abs().max())

    return dict(h=t.h_min, n=R.nnode, irr=R.irr_row.size,
                it_a=out["info_a"].iters, it_b=out["info_b"].iters,
                res=max(out["info_a"].residual, out["info_b"].residual),
                frozen=out["n_frozen"], patched=out["n_patched"],
                null=null, full_res=full_res,
                u=float((du - du.mean()).abs().max()),
                g_raw=float((out["grad_pinned"] - ref).abs().max()),
                g_fix=float((out["grad"] - ref).abs().max()),
                patch_delta=float((out["grad"] - out["grad_pinned"]).abs().max()))


FAIL = []


def check(cond, what):
    print(f"    {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        FAIL.append(what)


LEVELS = (3, 4, 5, 6)
print("=== Section 9.1.7: Omega=[0,pi]^3, u=cos x cos y cos z - 1, all-Neumann "
      f"(device {DEV}) ===")
print(f"non-graded grid, {1 << JUMP}x cell-size jump; Section 7 with "
      f"sub=4, patch=2 corner cells")
print(f"{'h_min':>10} {'nodes':>9} {'it_a':>5} {'it_b':>5} {'patched':>8} "
      f"{'err u':>11} {'ord':>6} {'err grad':>11} {'ord':>6} {'Sec7 delta':>11}")

rows, prev = [], None
for lmax in LEVELS:
    r = run(lmax)
    ou = np.log2(prev["u"] / r["u"]) if prev else float("nan")
    og = np.log2(prev["g_fix"] / r["g_fix"]) if prev else float("nan")
    r["ou"], r["og"] = ou, og
    rows.append(r)
    fu = f"{ou:6.2f}" if prev else "     -"
    fg = f"{og:6.2f}" if prev else "     -"
    print(f"{r['h']:10.3e} {r['n']:9d} {r['it_a']:5d} {r['it_b']:5d} "
          f"{r['patched']:8d} {r['u']:11.3e} {fu} {r['g_fix']:11.3e} {fg} "
          f"{r['patch_delta']:11.3e}")
    prev = r

print()
ou = [r["ou"] for r in rows[1:]]
og = [r["og"] for r in rows[1:]]
check(all(o >= 1.8 for o in ou),
      f"all-Neumann solution order >= 1.8: {['%.2f' % o for o in ou]}")
check(all(o >= 1.7 for o in og),
      f"all-Neumann gradient order >= 1.7: {['%.2f' % o for o in og]}")
check(all(r["res"] < 1e-10 for r in rows), "every solve converged")

print()
print("why Section 7 has nothing to repair with this Neumann discretization:")
check(all(r["null"] < 1e-12 for r in rows),
      f"the operator annihilates constants exactly: "
      f"max |A.1| h^2 = {max(r['null'] for r in rows):.2e}")
# Both quantities below are limited by the solver tolerance, not by
# anything physical, so they are judged against the discretization error
# on the same grid: being orders of magnitude smaller than it is what
# makes them zero for present purposes.
check(all(r["full_res"] < 1e-3 * r["u"] for r in rows),
      f"the pinned solution satisfies the FULL unpinned system, pinned row "
      f"included: max rel residual {max(r['full_res'] for r in rows):.2e}, "
      f"<= 1e-3 x the solution error on each grid")
check(all(r["patch_delta"] < 1e-3 * r["g_raw"] for r in rows),
      f"so the step-3 patch is a no-op: max |grad_Sec7 - grad_pinned| = "
      f"{max(r['patch_delta'] for r in rows):.2e}, "
      f"<= 1e-3 x the gradient error on each grid")

print()
print("Section 7 machinery still runs end to end; step 2 is a genuine local "
      "solve and its result must agree with step 1 on the patch region")
for sub, patch in ((3, 1), (4, 2), (6, 3)):
    r = run(LEVELS[-1], sub=sub, patch=patch)
    print(f"      sub={sub} patch={patch}  frozen={r['frozen']:7d} "
          f"patched={r['patched']:6d} it_b={r['it_b']:4d}  "
          f"grad err {r['g_fix']:.4e}  patch delta {r['patch_delta']:.2e}")
    check(r["it_b"] > 0, f"sub={sub} patch={patch}: step 2 actually solved")
    check(abs(r["g_fix"] / rows[-1]["g_fix"] - 1.0) < 1e-6,
          f"sub={sub} patch={patch}: result independent of the subdomain size")

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} check(s)")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("Section 9.1.7 / Section 7 conformance passed")
