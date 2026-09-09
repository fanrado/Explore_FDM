"""End-to-end verification: drift field -> drift paths -> induced current.

Geometry: an NxN pixel supercell with periodic transverse boundaries, so the
pad array is effectively infinite.  The anode plane at z=0 carries conducting
pads (Dirichlet, 0 V) separated by bare dielectric (homogeneous Neumann), which
is what focuses drift lines onto the pads.  The cathode sets a 500 V/cm drift
field.

Two solves on the SAME grid:
  drift field    -- all pads 0 V, cathode at -E0*L
  weighting field-- centre pad 1 V, all other pads and the cathode 0 V

Then 10x10 electrons are tracked from random points in the volume and the
Shockley-Ramo current on the centre pad is computed.  The decisive check is
that the induced charge from the Ramo integral agrees with the independent
weighting-potential difference, and that a collected electron induces exactly
one elementary charge.
"""
import numpy as np, torch, time
from fdm.tree import Octree
from fdm.geometry import PixelAnode
from fdm.topology import build_rows, build_gradient
from fdm.operator import Operator
from fdm.solve import solve
from fdm.field import Gradient
from fdm.drift import drift_paths, induced_current, fill_missing, drift_velocity, E_CHARGE

PITCH, LMAX, NPAD, NZ = 4.434, 5, 5, 12
E0 = 500.0/10.0          # 500 V/cm -> 50 V/mm
hmin = PITCH/(1<<LMAX)
LDRIFT = NZ*PITCH
rng = np.random.default_rng(20260907)

t = Octree((NPAD,NPAD,NZ), lmax=LMAX, h_min=hmin, periodic=(True,True,False))
pad = PixelAnode(t, pitch=PITCH, pad=22*hmin)
t.build(pad.refine_predicate(grade=8.0))
co_cells = len(t)

R0 = build_rows(t); co = R0.coords
zmax = int(t.dims_units[2])
shift = (NPAD//2)*int(PITCH/hmin)
cc = co.copy(); cc[:,0] -= shift; cc[:,1] -= shift

on_anode  = pad.on_anode(co)
on_pads   = pad.on_pad(co)                       # every pad
centre    = pad.on_pad(cc, centre_only=True)     # just the middle one
gap       = on_anode & ~on_pads                  # bare dielectric -> Neumann
cathode   = co[:,2] == zmax

R = build_rows(t, neumann=gap)
dirich = on_pads | cathode
print(f"pitch {PITCH} mm  pad {pad.pad:.3f} mm  gap {pad.gap:.3f} mm  h_min {hmin*1000:.1f} um")
print(f"supercell {NPAD}x{NPAD} pads, drift {LDRIFT:.1f} mm, E0 {E0*10:.0f} V/cm")
print(f"cells {co_cells}  nodes {R.nnode}  irregular {R.irr_row.size}  "
      f"pads {int(on_pads.sum())}  gap(Neumann) {int(gap.sum())}")

A = Operator(R, dirichlet=dirich, device="cuda")
def run(V, tag):
    t0=time.time(); u,info = solve(A, A.rhs(torch.as_tensor(V,device="cuda")),
                                   tol=1e-11, maxiter=60000)
    print(f"  {tag}: {info.iters} iters, res {info.residual:.1e}, {time.time()-t0:.2f}s")
    return u

Vd = np.zeros(R.nnode); Vd[cathode] = -E0*LDRIFT      # pads at 0 V
Vw = np.zeros(R.nnode); Vw[centre]  = 1.0             # weighting: centre pad 1 V
phi_d = run(Vd, "drift    ")
phi_w = run(Vw, "weighting")

G = Gradient(build_gradient(t, neumann=gap), device="cuda")
have = np.zeros(R.nnode, bool); have[G.row.cpu().numpy()] = True
def Efield(u):
    E = np.zeros((R.nnode,3)); E[G.row.cpu().numpy()] = (-G(u)).cpu().numpy()
    return fill_missing(t, E, have)
Ed, Ew = Efield(phi_d), Efield(phi_w)
wpot = fill_missing(t, phi_w.cpu().numpy(), np.ones(R.nnode,bool))

# ---- sanity: far from the anode the drift field must be uniform E0 ----------
from fdm.drift import Interpolator
I = Interpolator(t)
pz = np.stack([np.full(6, shift*hmin), np.full(6, shift*hmin),
               np.array([0.25,0.5,1,2,4,8])*PITCH],1)
Emag = np.linalg.norm(I(Ed, pz),axis=1)
print(f"\n  |E| on axis vs z/pitch: " +
      "  ".join(f"{z/PITCH:.2f}:{e*10:.0f}V/cm" for z,e in zip(pz[:,2],Emag)))
print(f"  bulk |E| = {Emag[-1]*10:.1f} V/cm (target {E0*10:.0f})   "
      f"v_drift = {drift_velocity(Emag[-1]/100):.4f} mm/us")

# ---- 10x10 drift paths -----------------------------------------------------
NG=10
gx,gy = np.meshgrid(np.linspace(0.1,0.9,NG), np.linspace(0.1,0.9,NG), indexing="ij")
starts = np.stack([ (gx.ravel()+rng.uniform(-.04,.04,NG*NG))*NPAD*PITCH,
                    (gy.ravel()+rng.uniform(-.04,.04,NG*NG))*NPAD*PITCH,
                    rng.uniform(0.15,0.95,NG*NG)*LDRIFT ],1)
t0=time.time(); paths = drift_paths(t, Ed, starts, z_stop=0.0); tp=time.time()-t0
lens = np.array([len(p) for p in paths]); zend=np.array([p[-1,2] for p in paths])
tend=np.array([p[-1,3] for p in paths])
print(f"\n{NG}x{NG} = {len(paths)} paths tracked in {tp:.1f}s "
      f"(steps {lens.min()}-{lens.max()}, median {int(np.median(lens))})")
print(f"  all reached the anode: {bool((zend<1e-9).all())}   "
      f"drift time {tend.min():.1f}-{tend.max():.1f} us")
vfit = (starts[:,2]-zend)/tend
print(f"  mean drift speed {vfit.mean():.4f} mm/us (expect {drift_velocity(E0/100):.4f}), "
      f"spread {100*vfit.std()/vfit.mean():.2f}%")
# transverse displacement: focusing onto pads
d0 = np.linalg.norm(starts[:,:2]-np.array([p[-1,0] for p in paths] and
     [[p[-1,0],p[-1,1]] for p in paths]),axis=1)
print(f"  transverse focusing displacement: mean {d0.mean()*1000:.0f} um, max {d0.max()*1000:.0f} um")

# where did they land?
land = np.array([[p[-1,0],p[-1,1]] for p in paths])
onc = pad.on_pad(np.c_[np.round(land/hmin).astype(np.int64)-shift,
                        np.zeros(len(land),np.int64)], centre_only=True)
print(f"  landed on the centre pad: {int(onc.sum())} of {len(paths)}")

# ---- induced current -------------------------------------------------------
print(f"\n  Ramo vs weighting-potential charge (fC), electron charge = {E_CHARGE:.4e} fC")
qr=[];qp=[];pk=[]
for p in paths:
    _,i_r,q_r,q_p = induced_current(t, wpot, Ew, p)
    qr.append(q_r[-1]); qp.append(q_p[-1]); pk.append(np.abs(i_r).max())
qr=np.array(qr); qp=np.array(qp); pk=np.array(pk)
rel = np.abs(qr-qp)/np.maximum(np.abs(qp),1e-12)
# Relative error is meaningless where Q_phi ~ 0 (bipolar neighbour signals that
# net to zero), so report it against the electron charge as well.
print(f"  |Q_ramo - Q_phi| / |Q_phi| :  median {np.median(rel):.2e}")
print(f"  |Q_ramo - Q_phi| / e       :  median {np.median(np.abs(qr-qp))/E_CHARGE:.2e}"
      f"   max {np.max(np.abs(qr-qp))/E_CHARGE:.2e}")
coll = qp > 0.5*E_CHARGE
print(f"  collected on centre pad ({int(coll.sum())}): "
      f"Q/e = {np.mean(qp[coll]/E_CHARGE):.6f} +- {np.std(qp[coll]/E_CHARGE):.2e}")
if (~coll).any():
    print(f"  neighbours    ({int((~coll).sum())}): "
          f"|Q|/e max = {np.max(np.abs(qp[~coll]))/E_CHARGE:.2e} (bipolar, nets to ~0)")
print(f"  peak |i| on centre pad: {pk[coll].min():.5f} - {pk[coll].max():.5f} nA")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
for k,p in enumerate(paths):
    ax[0].plot(p[:,0],p[:,2],lw=.5,color="C0",alpha=.5)
for e in range(NPAD+1):
    ax[0].axvline(e*PITCH,color="k",lw=.4,ls=":")
h2=(PITCH-pad.pad)/2
for e in range(NPAD):
    ax[0].plot([e*PITCH+h2,e*PITCH+h2+pad.pad],[0,0],color="C3",lw=3)
ax[0].set(xlabel="x (mm)",ylabel="z (mm)",title="drift paths (pads red)",ylim=(-1,LDRIFT))
for k in np.flatnonzero(coll):
    tt,ii,_,_=induced_current(t,wpot,Ew,paths[k]); ax[1].plot(tt-tt[-1],ii*1e3,lw=1)
ax[1].set(xlabel="t - t_arrival (us)",ylabel="i (pA)",title="induced current, collected",xlim=(-4,0.05))
for k in np.flatnonzero(~coll)[:20]:
    tt,ii,_,_=induced_current(t,wpot,Ew,paths[k]); ax[2].plot(tt-tt[-1],ii*1e3,lw=1)
ax[2].set(xlabel="t - t_arrival (us)",ylabel="i (pA)",title="induced current, neighbours (bipolar)",xlim=(-4,0.05))
for a in ax: a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("drift_verification.png",dpi=110)
print("\n  wrote drift_verification.png")
