"""The 'missing' induced charge is a finite-supercell effect, with a formula.

In an NxN periodic supercell one pad at 1 V carries 1/N^2 of the uniform (k=0)
Fourier mode of the anode plane.  That mode is not screened by the pad
structure: it decays LINEARLY to the grounded cathode rather than as
exp(-2 pi z / pitch).  So an electron starting at height z has

    phi_w(start) = (1/N^2) (1 - z/L)   =>   Q/e = 1 - (1/N^2)(1 - z/L)

In a real detector one pad among millions has k=0 weight ~0 and Q/e -> 1.  This
test checks the code reproduces the formula, which is the strongest available
statement that the weighting solve is right.
"""
import numpy as np, torch
from fdm.tree import Octree
from fdm.geometry import PixelAnode
from fdm.topology import build_rows
from fdm.operator import Operator
from fdm.solve import solve
from fdm.drift import Interpolator

PITCH, NZ, LMAX = 4.434, 8, 4
hmin = PITCH/(1<<LMAX); LD = NZ*PITCH
print(f"{'N':>3} {'nodes':>9} {'z/L':>6} {'phi_w measured':>15} {'1/N^2*(1-z/L)':>15} {'ratio':>7}")
for NPAD in (3,5,7,9):
    t=Octree((NPAD,NPAD,NZ),lmax=LMAX,h_min=hmin,periodic=(True,True,False))
    pad=PixelAnode(t,pitch=PITCH,pad=11*hmin); t.build(pad.refine_predicate(grade=8.0))
    R0=build_rows(t); co=R0.coords; zmax=int(t.dims_units[2])
    shift=(NPAD//2)*int(PITCH/hmin); cc=co.copy(); cc[:,0]-=shift; cc[:,1]-=shift
    on_pads=pad.on_pad(co); centre=pad.on_pad(cc,centre_only=True)
    gap=pad.on_anode(co)&~on_pads; cathode=co[:,2]==zmax
    R=build_rows(t,neumann=gap); A=Operator(R,dirichlet=on_pads|cathode,device="cuda")
    V=np.zeros(R.nnode); V[centre]=1.0
    u,_=solve(A,A.rhs(torch.as_tensor(V,device="cuda")),tol=1e-11,maxiter=80000)
    I=Interpolator(t)
    for zf in (0.5,):
        p=np.array([[shift*hmin+PITCH/2, shift*hmin+PITCH/2, zf*LD]])
        m=float(I(u.cpu().numpy(),p)[0]); pr=(1/NPAD**2)*(1-zf)
        print(f"{NPAD:3d} {R.nnode:9d} {zf:6.2f} {m:15.5f} {pr:15.5f} {m/pr:7.3f}")
