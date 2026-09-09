"""Does a collected electron induce exactly one elementary charge?

Q/e must -> 1 for an electron that lands on the pad.  Any deficit comes from
trilinear interpolation of phi_w at the termination point: focusing drives
electrons preferentially towards pad EDGES, where the interpolation cell
straddles the pad boundary.  If that is the cause, the deficit must fall with
h_min.  If it does not, something is wrong with the weighting solve.
"""
import numpy as np, torch, time
from fdm.tree import Octree
from fdm.geometry import PixelAnode
from fdm.topology import build_rows, build_gradient
from fdm.operator import Operator
from fdm.solve import solve
from fdm.field import Gradient
from fdm.drift import drift_paths, induced_current, fill_missing, E_CHARGE

PITCH, NPAD, NZ, E0 = 4.434, 3, 8, 50.0
print(f"{'h_min(um)':>10} {'nodes':>10} {'collected':>10} {'<Q/e>':>10} {'1-<Q/e>':>10} {'ratio':>7}")
prev=None
for LMAX in (4,5,6):
    hmin=PITCH/(1<<LMAX); LD=NZ*PITCH
    t=Octree((NPAD,NPAD,NZ),lmax=LMAX,h_min=hmin,periodic=(True,True,False))
    pad=PixelAnode(t,pitch=PITCH,pad=(11<<(LMAX-4))*hmin)
    t.build(pad.refine_predicate(grade=8.0))
    R0=build_rows(t); co=R0.coords; zmax=int(t.dims_units[2])
    shift=(NPAD//2)*int(PITCH/hmin); cc=co.copy(); cc[:,0]-=shift; cc[:,1]-=shift
    on_pads=pad.on_pad(co); centre=pad.on_pad(cc,centre_only=True)
    gap=pad.on_anode(co)&~on_pads; cathode=co[:,2]==zmax
    R=build_rows(t,neumann=gap); A=Operator(R,dirichlet=on_pads|cathode,device="cuda")
    Vd=np.zeros(R.nnode); Vd[cathode]=-E0*LD
    Vw=np.zeros(R.nnode); Vw[centre]=1.0
    pd,_=solve(A,A.rhs(torch.as_tensor(Vd,device="cuda")),tol=1e-11,maxiter=80000)
    pw,_=solve(A,A.rhs(torch.as_tensor(Vw,device="cuda")),tol=1e-11,maxiter=80000)
    G=Gradient(build_gradient(t,neumann=gap),device="cuda")
    have=np.zeros(R.nnode,bool); have[G.row.cpu().numpy()]=True
    def EF(u):
        E=np.zeros((R.nnode,3)); E[G.row.cpu().numpy()]=(-G(u)).cpu().numpy()
        return fill_missing(t,E,have)
    Ed,Ew=EF(pd),EF(pw); wp=pw.cpu().numpy()
    # start electrons directly above the centre pad so they are all collected
    rng=np.random.default_rng(7)
    g=np.linspace(0.2,0.8,10)
    gx,gy=np.meshgrid(g,g,indexing="ij")
    h2=(PITCH-pad.pad)/2
    st=np.stack([shift*hmin+h2+gx.ravel()*pad.pad,
                 shift*hmin+h2+gy.ravel()*pad.pad,
                 np.full(100,0.5*LD)],1)
    paths=drift_paths(t,Ed,st,z_stop=0.0)
    q=np.array([induced_current(t,wp,Ew,p)[3][-1] for p in paths])/E_CHARGE
    keep=q>0.5
    d=1-q[keep].mean()
    r=f"{prev/d:7.2f}" if prev else "      -"
    print(f"{hmin*1000:10.1f} {R.nnode:10d} {int(keep.sum()):10d} {q[keep].mean():10.5f} {d:10.2e} {r}")
    prev=d
