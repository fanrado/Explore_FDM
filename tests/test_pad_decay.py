"""Test 6: pad-induced field ripple must decay as exp(-2 pi z / pitch).

Single pixel cell, transverse periodic.  The anode plane carries a pad at 0 V
and the inter-pad gap at 1 V; the cathode is at 0 V.  The transverse variation
of phi at height z is then a pure surface effect whose leading Fourier mode
decays as exp(-2 pi z / pitch).  Recovering that exponent validates the solver
AND measures how thick the refined slab has to be.
"""
import numpy as np, torch, time
from fdm.tree import Octree
from fdm.geometry import PixelAnode
from fdm.topology import build_rows
from fdm.operator import Operator
from fdm.solve import solve

PITCH = 4.434
LMAX  = 6                      # h_min = pitch/64 = 69.3 um
NZ    = 113                    # drift = 113 * pitch = 501.0 mm
hmin  = PITCH / (1 << LMAX)

t = Octree((1,1,NZ), lmax=LMAX, h_min=hmin, periodic=(True,True,False))
pad = PixelAnode(t, pitch=PITCH, pad=44*hmin)
t0=time.time(); t.build(pad.refine_predicate(grade=8.0)); tb=time.time()-t0
print(f"pitch={PITCH} mm  pad={pad.pad:.4f} mm  gap={pad.gap:.4f} mm  h_min={hmin*1000:.1f} um")
print(f"drift={NZ*PITCH:.1f} mm   cells={len(t)}  (build {tb:.1f}s)  levels={np.bincount(t.cells[:,0])}")

t0=time.time(); R = build_rows(t); ta=time.time()-t0
co = R.coords
zmax = int(t.dims_units[2])
dirich = pad.on_anode(co) | (co[:,2] == zmax)
V = np.zeros(R.nnode)
V[pad.on_anode(co) & ~pad.on_pad(co)] = 1.0        # gap at 1 V, pad at 0 V
A = Operator(R, dirichlet=dirich | R.boundary, device="cuda")
print(f"nodes={R.nnode}  regular={R.reg_row.size}  irregular={R.irr_row.size}  "
      f"nnz={R.nnz/1e6:.1f}M  assembly {ta:.1f}s")

b = A.rhs(torch.as_tensor(V, device="cuda"))
torch.cuda.synchronize(); t0=time.time()
u, info = solve(A, b, tol=1e-12, maxiter=20000)
torch.cuda.synchronize(); ts=time.time()-t0
print(f"solve: {info.iters} iters, rel.res {info.residual:.2e}, converged={info.converged}, "
      f"{ts:.2f}s  ({1e3*ts/max(info.iters,1):.2f} ms/iter)  "
      f"peak GPU {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

phi = u.cpu().numpy()
print(f"\n{'z (mm)':>10} {'z/pitch':>8} {'ripple':>11} {'exp(-2pi z/p)':>14} {'ratio':>8}")
zu_all = np.unique(co[:,2])
rows=[]
for frac in (0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,3.0):
    zt = frac*PITCH/hmin
    zu = zu_all[np.argmin(np.abs(zu_all-zt))]
    sel = co[:,2]==zu
    if sel.sum() < 8: continue
    z = zu*hmin
    rip = phi[sel].max()-phi[sel].min()
    pred = np.exp(-2*np.pi*z/PITCH)
    rows.append((z,rip,pred))
    print(f"{z:10.4f} {z/PITCH:8.3f} {rip:11.3e} {pred:14.3e} {rip/pred:8.3f}")

z=np.array([r[0] for r in rows]); r_=np.array([r[1] for r in rows])
# fit only where a single Fourier mode dominates and the amplitude is
# well above the solver tolerance floor
m=(z>0.4*PITCH)&(z<2.2*PITCH)&(r_>1e-7)
slope=np.polyfit(z[m], np.log(r_[m]),1)[0]
print(f"\nfitted decay constant = {-slope:.4f} /mm   expected 2*pi/pitch = {2*np.pi/PITCH:.4f} /mm"
      f"   -> {100*abs(-slope/(2*np.pi/PITCH)-1):.2f}% error")
