"""Tests 1 and 2: the operator reduces to the 7-point Laplacian on a uniform
grid, and annihilates a linear field at every node including T-junctions."""
import numpy as np
from fdm.tree import Octree
from fdm.topology import build_rows

def apply(R, u):
    Au = np.zeros(R.nnode)
    Au[R.reg_row] = (R.reg_val * u[R.reg_idx]).sum(1)
    if R.irr_row.size:
        Au[R.irr_row] = (R.irr_val * u[R.irr_idx]).sum(1)
    return Au

print("=== TEST 1: uniform tree reduces to the 7-point Laplacian ===")
t = Octree((4,4,4), lmax=0, h_min=0.25)
R = build_rows(t)
print(f"nodes={R.nnode} regular={R.reg_row.size} irregular={R.irr_row.size} boundary={R.boundary.sum()}")
h = 0.25
off = np.sort(R.reg_val[:,1:], axis=1)
print("off-diag == 1/h^2 :", np.allclose(off, 1/h**2), " diag == -6/h^2 :", np.allclose(R.reg_val[:,0], -6/h**2))
assert R.irr_row.size == 0

print()
print("=== TEST 2: linear field must give exactly zero at EVERY node ===")
for name, (nd, lmax, pred) in {
  "uniform      ": ((4,4,4), 0, None),
  "graded x     ": ((4,4,4), 3, lambda c,s: c[:,0] < 0.5),
  "non-graded   ": ((4,4,4), 3, lambda c,s: (c[:,0] < 0.3) & (c[:,1] < 0.3)),
  "deep corner  ": ((2,2,2), 4, lambda c,s: np.linalg.norm(c-0.0,axis=1) < 0.35),
}.items():
    t = Octree(nd, lmax=lmax, h_min=1.0/(nd[0]*(1<<lmax)))
    if pred: t.build(pred)
    R = build_rows(t)
    P = t.node_positions(R.coords)
    for coef in ([1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.7,-1.3,2.1]):
        u = P @ np.array(coef) + 0.37
        r = apply(R, u)[~R.boundary]
        e = np.abs(r).max() if r.size else 0.0
        scale = max(1.0, np.abs(np.array(coef)).max()) / t.h_min**2
        assert e/scale < 1e-9, (name, coef, e/scale)
    lv = np.bincount(t.cells[:,0])
    print(f"{name} cells={len(t):6d} nodes={R.nnode:7d} reg={R.reg_row.size:7d} irr={R.irr_row.size:6d} levels={lv}  max|Au|/scale < 1e-9  OK")
