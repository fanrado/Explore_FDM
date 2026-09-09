"""Completeness sum rule for weighting potentials -- the decisive check.

Setting every electrode to 1 V makes phi == 1 everywhere (it solves Laplace and
matches all BCs, including homogeneous Neumann on the gaps).  By superposition,

    phi_w,cathode(x) + SUM over all pads phi_w,pad(x) = 1   for every x.

By transverse periodicity the sum over pads is the centre pad's weighting
potential evaluated at the N^2 translated points, so this needs only two solves.
Any error in the operator, the Neumann treatment, the periodic wrap or the
interpolation breaks it.
"""
import numpy as np, torch
from fdm.tree import Octree
from fdm.geometry import PixelAnode
from fdm.topology import build_rows
from fdm.operator import Operator
from fdm.solve import solve
from fdm.drift import Interpolator

PITCH, NZ, LMAX = 4.434, 8, 5
hmin = PITCH/(1<<LMAX); LD = NZ*PITCH
rng = np.random.default_rng(3)
print(f"{'N':>3} {'nodes':>9} {'max |sum - 1|':>15} {'mean |sum - 1|':>15}")
for NPAD in (3,5,7):
    t=Octree((NPAD,NPAD,NZ),lmax=LMAX,h_min=hmin,periodic=(True,True,False))
    pad=PixelAnode(t,pitch=PITCH,pad=22*hmin); t.build(pad.refine_predicate(grade=8.0))
    R0=build_rows(t); co=R0.coords; zmax=int(t.dims_units[2])
    shift=(NPAD//2)*int(PITCH/hmin); cc=co.copy(); cc[:,0]-=shift; cc[:,1]-=shift
    on_pads=pad.on_pad(co); centre=pad.on_pad(cc,centre_only=True)
    gap=pad.on_anode(co)&~on_pads; cathode=co[:,2]==zmax
    R=build_rows(t,neumann=gap); A=Operator(R,dirichlet=on_pads|cathode,device="cuda")

    Vp=np.zeros(R.nnode); Vp[centre]=1.0          # centre pad at 1 V
    Vc=np.zeros(R.nnode); Vc[cathode]=1.0         # cathode at 1 V, all pads 0
    up,_=solve(A,A.rhs(torch.as_tensor(Vp,device="cuda")),tol=1e-12,maxiter=80000)
    uc,_=solve(A,A.rhs(torch.as_tensor(Vc,device="cuda")),tol=1e-12,maxiter=80000)
    wp, wc = up.cpu().numpy(), uc.cpu().numpy()

    I=Interpolator(t)
    NS=400
    P=np.stack([rng.uniform(0,NPAD*PITCH,NS), rng.uniform(0,NPAD*PITCH,NS),
                rng.uniform(0.02,0.98,NS)*LD],1)
    tot=I(wc,P).copy()
    for i in range(NPAD):
        for j in range(NPAD):
            Q=P.copy(); Q[:,0]+=i*PITCH; Q[:,1]+=j*PITCH
            tot += I(wp,Q)
    e=np.abs(tot-1.0)
    print(f"{NPAD:3d} {R.nnode:9d} {e.max():15.3e} {e.mean():15.3e}")
