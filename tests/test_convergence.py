"""Test 3/4: convergence of phi and grad(phi) on a non-graded octree."""
import numpy as np, torch, sys
from fdm.tree import Octree
from fdm.topology import build_rows, build_gradient
from fdm.operator import Operator
from fdm.solve import solve
from fdm.field import Gradient

K = np.pi
def phi(P):   # harmonic: laplacian == 0 exactly
    return np.sin(K*P[:,0])*np.sin(K*P[:,1])*np.exp(np.sqrt(2)*K*P[:,2])
def grad(P):
    s=np.sqrt(2)*K; e=np.exp(s*P[:,2]); sx,sy=np.sin(K*P[:,0]),np.sin(K*P[:,1])
    cx,cy=np.cos(K*P[:,0]),np.cos(K*P[:,1])
    return np.stack([K*cx*sy*e, K*sx*cy*e, s*sx*sy*e],1)

def run(lmax, correction=True, dev="cuda"):
    hmin = 1.0/(2*(1<<lmax))
    t = Octree((2,2,2), lmax=lmax, h_min=hmin)
    # Fixed refinement *pattern* that scales with hmin: a fine slab near z=0 and
    # a region JUMP levels coarser elsewhere.  Both regions must shrink together
    # or the coarse region's truncation error stays constant and dominates.
    JUMP = 2
    t.build(lambda c,s: s > 1.001*np.where(c[:,2] < 0.3, hmin, hmin*(1<<JUMP)))
    R = build_rows(t, interface_correction=correction)
    A = Operator(R, device=dev)
    P = t.node_positions(R.coords)
    ex = torch.as_tensor(phi(P), device=dev, dtype=torch.float64)
    b = A.rhs(ex)
    u, info = solve(A, b, tol=1e-12, maxiter=20000)
    err = (u - ex).abs()
    G = Gradient(build_gradient(t), device=dev)
    ge = torch.as_tensor(grad(P), device=dev)[G.row]
    gerr = float((G(u) - ge).abs().max())
    # error restricted to nodes with a non-trivial (interpolated) stencil
    iface = torch.as_tensor(R.irr_row, device=dev, dtype=torch.long)
    return dict(h=t.h_min, n=R.nnode, cells=len(t), iters=info.iters, res=info.residual,
                u=float(err.max()), ui=float(err[iface].max()) if iface.numel() else 0.0,
                g=gerr, irr=R.irr_row.size)

def order(a,b): return np.log2(a/b) if (a>0 and b>0) else float("nan")

for corr in (True, False):
    print(f"\n=== interface_correction = {corr} ===")
    print(f"{'h_min':>10} {'nodes':>9} {'iters':>6} {'L8 err phi':>12} {'ord':>5} "
          f"{'err @iface':>12} {'ord':>5} {'L8 err gradphi':>15} {'ord':>5}")
    prev=None
    for lmax in (3,4,5,6):
        r = run(lmax, corr)
        o = lambda k: f"{order(prev[k],r[k]):5.2f}" if prev else "    -"
        print(f"{r['h']:10.3e} {r['n']:9d} {r['iters']:6d} {r['u']:12.3e} {o('u')} "
              f"{r['ui']:12.3e} {o('ui')} {r['g']:15.3e} {o('g')}")
        prev=r
