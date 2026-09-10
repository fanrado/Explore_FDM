"""Conformance tests for the Min/Gibou/Ceniceros construction (Sections 5, 8, 9).

Unlike the other files in tests/, this one asserts and exits non-zero on failure,
and runs on the CPU with no GPU required.

  T1  the exact identity  sum_a w_a D_a u == laplacian(u)  for every quadratic
      monomial, on every mesh.  This is the sharpest available check: Eq. (1) is
      exact for quadratics and a multilinear interpolation reproduces a
      quadratic with error exactly (p q / 2) u_tt, so the paper's cancellation
      must be exact to round-off.  It would catch the transposed products in the
      paper's printed alpha/beta.
  T2  reduction to the standard 7-point Laplacian on a uniform grid.
  T3  the structural guarantee Eq. (8) relies on: the contamination pattern is
      triangular, one outward direction is edge-aligned, rows are at most 11
      wide, all three inward distances equal C's edge length.
"""
import sys

import numpy as np

from fdm.geometry import PixelAnode
from fdm.topology import build_gradient, build_rows, check_paper_structure
from fdm.tree import Octree

FAIL = []


def check(cond, what):
    print(f"    {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        FAIL.append(what)


def apply(R, u):
    Au = np.zeros(R.nnode)
    Au[R.reg_row] = (R.reg_val * u[R.reg_idx]).sum(1)
    if R.irr_row.size:
        Au[R.irr_row] = (R.irr_val * u[R.irr_idx]).sum(1)
    return Au


# Ten quadratic monomials and their exact Laplacians.  On a periodic axis a
# monomial in that coordinate is not single-valued, so only the subset that is
# constant along the periodic axes is admissible there.
MONOMIALS = [
    ("1",   lambda P: np.ones(len(P)),    0.0, ()),
    ("x",   lambda P: P[:, 0],            0.0, (0,)),
    ("y",   lambda P: P[:, 1],            0.0, (1,)),
    ("z",   lambda P: P[:, 2],            0.0, (2,)),
    ("x^2", lambda P: P[:, 0] ** 2,       2.0, (0,)),
    ("y^2", lambda P: P[:, 1] ** 2,       2.0, (1,)),
    ("z^2", lambda P: P[:, 2] ** 2,       2.0, (2,)),
    ("xy",  lambda P: P[:, 0] * P[:, 1],  0.0, (0, 1)),
    ("yz",  lambda P: P[:, 1] * P[:, 2],  0.0, (1, 2)),
    ("zx",  lambda P: P[:, 2] * P[:, 0],  0.0, (2, 0)),
]


def meshes():
    """A battery spanning uniform, graded, non-graded, random and periodic."""
    yield "uniform lmax=0", Octree((4, 4, 4), lmax=0, h_min=0.25)

    h = 1 / 16
    t = Octree((2, 2, 2), lmax=4, h_min=h)
    t.build(lambda c, s: s > 1.001 * np.where(c[:, 2] < 0.3, h, h * 2))
    yield "graded, jump=1", t

    t = Octree((2, 2, 2), lmax=4, h_min=h)
    t.build(lambda c, s: s > 1.001 * np.where(c[:, 2] < 0.3, h, h * 8))
    yield "non-graded, jump=3", t

    t = Octree((4, 4, 4), lmax=3, h_min=1 / 32)
    t.build(lambda c, s: (c[:, 0] < 0.3) & (c[:, 1] < 0.3))
    yield "non-graded corner", t

    t = Octree((2, 2, 2), lmax=4, h_min=h)
    t.build(lambda c, s: s > 1.001 * np.maximum(
        np.linalg.norm(c - 0.05, axis=1) / 4, h))
    yield "deep corner", t

    for seed in (7, 11, 23):
        rng = np.random.default_rng(seed)
        t = Octree((2, 2, 2), lmax=4, h_min=h)
        t.build(lambda c, s: rng.random(len(c)) < 0.55)
        yield f"random, seed={seed}", t

    t = Octree((1, 1, 4), lmax=3, h_min=1 / 8, periodic=(True, True, False))
    t.build(lambda c, s: s > 1.001 * np.where(c[:, 2] < 1.0, 1 / 8, 1 / 4))
    yield "periodic x,y", t

    lmax, pitch = 4, 4.434
    hm = pitch / (1 << lmax)
    t = Octree((3, 3, 4), lmax=lmax, h_min=hm, periodic=(True, True, False))
    pad = PixelAnode(t, pitch=pitch, pad=(11 << (lmax - 4)) * hm)
    t.build(pad.refine_predicate(grade=8.0))
    yield "production, periodic", t


print("=== T1: exact identity  sum_a w_a D_a u == laplacian(u)  (quadratics) ===")
worst = 0.0
for name, t in meshes():
    R = build_rows(t)
    P = t.node_positions(R.coords)
    free = ~R.boundary
    per = set(np.flatnonzero(t.periodic).tolist())
    scale = max(1.0, np.abs(P).max()) ** 2
    err, tested = 0.0, 0
    for nm, f, lap, axes in MONOMIALS:
        if per & set(axes):
            continue          # not single-valued across a periodic seam
        tested += 1
        if free.any():
            err = max(err, np.abs(apply(R, f(P))[free] - lap).max() / scale)
    worst = max(worst, err)
    check(err < 1e-9,
          f"{name:22s} n={R.nnode:7d} irr={R.irr_row.size:6d} "
          f"monomials={tested:2d}  max rel err {err:.2e}")
print(f"    worst over all meshes: {worst:.3e}")

print()
print("=== T2: uniform grid reduces to the standard 7-point Laplacian ===")
h = 0.25
t = Octree((4, 4, 4), lmax=0, h_min=h)
R = build_rows(t)
check(R.irr_row.size == 0, "no irregular rows on a uniform grid")
check(np.allclose(R.reg_val[:, 1:], 1 / h ** 2), f"off-diagonals == 1/h^2")
check(np.allclose(R.reg_val[:, 0], -6 / h ** 2), "diagonal == -6/h^2")
check(R.reg_idx.shape[1] == 7, f"regular width == 7 (got {R.reg_idx.shape[1]})")

print()
print("=== T3: the structural guarantee behind Eq. (8) ===")
for name, t in meshes():
    for neu in (None,):
        info = check_paper_structure(t, neumann=neu, strict=False)
        ok = (info["no_clean"] == 0 and info["not_orderable"] == 0
              and info["inward_consistent"] and info["outward_finer_than_C"] == 0
              and info["max_width"] <= 11)
        check(ok, f"{name:22s} interior={info['interior']:7d} "
                  f"max_width={info['max_width']:3d} "
                  f"no_clean={info['no_clean']} "
                  f"unorderable={info['not_orderable']} "
                  f"inward_consistent={info['inward_consistent']} "
                  f"outward_finer={info['outward_finer_than_C']}")
        print(f"          nsup multiset histogram: {info['nsup_histogram']}")

print()
print("=== T4: Section 8 gradient is exact for quadratics at EVERY node ===")
# The gradient is the derivative of the parabola through the three points, so it
# is exact for quadratics at a non-interpolated node.  Section 8's correction
# removes exactly the interpolation's (p q / 2) u_tt contribution, so with it
# applied the gradient must be exact at interpolated nodes too.
GMONO = [
    ("1",   lambda P: np.ones(len(P)),
     lambda P: np.zeros((len(P), 3)), ()),
    ("x",   lambda P: P[:, 0],
     lambda P: np.stack([np.ones(len(P)), 0 * P[:, 0], 0 * P[:, 0]], 1), (0,)),
    ("x^2", lambda P: P[:, 0] ** 2,
     lambda P: np.stack([2 * P[:, 0], 0 * P[:, 0], 0 * P[:, 0]], 1), (0,)),
    ("y^2", lambda P: P[:, 1] ** 2,
     lambda P: np.stack([0 * P[:, 0], 2 * P[:, 1], 0 * P[:, 0]], 1), (1,)),
    ("z^2", lambda P: P[:, 2] ** 2,
     lambda P: np.stack([0 * P[:, 0], 0 * P[:, 0], 2 * P[:, 2]], 1), (2,)),
    ("xy",  lambda P: P[:, 0] * P[:, 1],
     lambda P: np.stack([P[:, 1], P[:, 0], 0 * P[:, 0]], 1), (0, 1)),
    ("yz",  lambda P: P[:, 1] * P[:, 2],
     lambda P: np.stack([0 * P[:, 0], P[:, 2], P[:, 1]], 1), (1, 2)),
    ("zx",  lambda P: P[:, 2] * P[:, 0],
     lambda P: np.stack([P[:, 2], 0 * P[:, 0], P[:, 0]], 1), (2, 0)),
]


def grad_apply(G, u, a):
    g = (G.val[a] * u[G.idx[a]]).sum(1)
    if G.corr_pos.size:
        np.add.at(g, G.corr_pos, (G.corr_val[a] * u[G.corr_idx[a]]).sum(1))
    return g


for name, t in meshes():
    G = build_gradient(t)
    if G.row.size == 0:
        continue
    P = t.node_positions(t.nodes()[0])
    per = set(np.flatnonzero(t.periodic).tolist())
    scale = max(1.0, np.abs(P).max())
    err, tested = 0.0, 0
    for lab, f, gf, axes in GMONO:
        if per & set(axes):
            continue
        tested += 1
        u, ref = f(P), gf(P)
        for a in range(3):
            err = max(err, np.abs(grad_apply(G, u, a)
                                  - ref[G.row, a]).max() / scale)
    check(err < 1e-9,
          f"{name:22s} rows={G.row.size:7d} corrected={G.corr_pos.size:6d} "
          f"({100 * G.corr_pos.size / G.row.size:4.1f}%) "
          f"monomials={tested:2d}  max rel err {err:.2e}")

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} check(s)")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("all paper-conformance checks passed")
