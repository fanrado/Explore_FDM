"""Test 7: weighting field of a single pad in an NxN island.

Centre pad at 1 V, all other pads and the cathode grounded.  The weighting
potential must decay away from the centre pad; convergence is checked by
growing the island and watching the near-pad field stop changing.
"""
import numpy as np, torch, time
from fdm.tree import Octree
from fdm.geometry import PixelAnode
from fdm.topology import build_rows
from fdm.operator import Operator
from fdm.solve import solve
from fdm.field import efield

PITCH=4.434; LMAX=5; NZ=24          # 24*pitch = 106 mm of drift is ample here
hmin=PITCH/(1<<LMAX)

print(f"h_min = {hmin*1000:.1f} um    pitch = {PITCH} mm")
print(f"{'island':>7} {'nodes':>9} {'irr%':>6} {'iters':>7} {'t(s)':>7} "
      f"{'phi_w(z=p/4)':>13} {'|E| on axis':>12} {'d vs prev':>10}")
prev=None
for N in (3,5,7):
    t=Octree((N,N,NZ), lmax=LMAX, h_min=hmin)
    pad=PixelAnode(t, pitch=PITCH, pad=22*hmin)
    t.build(pad.refine_predicate(grade=8.0))
    R=build_rows(t); co=R.coords
    # centre the pad indexing on the middle pitch cell
    shift=(N//2)*int(PITCH/hmin)
    cc=co.copy(); cc[:,0]-=shift; cc[:,1]-=shift
    centre = pad.on_pad(cc, centre_only=True)
    anode  = pad.on_anode(co)
    V=np.zeros(R.nnode); V[centre]=1.0            # weighting field: centre pad at 1 V
    A=Operator(R, dirichlet=(anode | R.boundary), device="cuda")
    t0=time.time(); u,info=solve(A, A.rhs(torch.as_tensor(V,device="cuda")),
                                 tol=1e-11, maxiter=60000); ts=time.time()-t0
    rid,E=efield(t,u)
    P=t.node_positions(co)
    # probe on the axis above the centre pad, a quarter pitch up
    zt=PITCH/4
    axis=(np.abs(P[:,0]-shift*hmin)<1e-9)&(np.abs(P[:,1]-shift*hmin)<1e-9)
    k=np.flatnonzero(axis & (np.abs(P[:,2]-zt)<hmin))
    phiw=float(u[k].mean()) if k.size else float("nan")
    m=np.isin(rid.cpu().numpy(), k)
    Emag=float(E[torch.as_tensor(m,device=E.device)].norm(dim=1).mean()) if m.any() else float("nan")
    d = "" if prev is None else f"{abs(phiw-prev)/max(abs(prev),1e-30):10.2%}"
    print(f"{N}x{N:<5} {R.nnode:9d} {100*R.irr_row.size/R.nnode:6.1f} {info.iters:7d} {ts:7.2f} "
          f"{phiw:13.5e} {Emag:12.5e} {d:>10}")
    prev=phiw
