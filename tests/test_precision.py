"""Test 5: does fp32 storage + fp64 reductions + fp64 refinement track fp64?"""
import numpy as np, torch, time
from fdm.tree import Octree
from fdm.topology import build_rows
from fdm.operator import Operator
from fdm.solve import solve
K=np.pi
def phi(P): return np.sin(K*P[:,0])*np.sin(K*P[:,1])*np.exp(np.sqrt(2)*K*P[:,2])

print(f"{'h_min':>10} {'nodes':>9} | {'fp64 err':>11} {'t(s)':>7} | {'fp32+ref err':>13} {'t(s)':>7} | {'ratio':>7}")
for lmax in (4,5,6):
    hmin=1.0/(2*(1<<lmax))
    t=Octree((2,2,2),lmax=lmax,h_min=hmin)
    t.build(lambda c,s: s>1.001*np.where(c[:,2]<0.3, hmin, hmin*4))
    R=build_rows(t); P=t.node_positions(R.coords)
    out={}
    for tag,dt,ref in (("fp64",torch.float64,0), ("fp32",torch.float32,2)):
        A=Operator(R, device="cuda", dtype=dt)
        ex=torch.as_tensor(phi(P), device="cuda", dtype=dt)
        torch.cuda.synchronize(); t0=time.time()
        u,info=solve(A, A.rhs(ex), tol=1e-12 if dt==torch.float64 else 1e-6,
                     maxiter=30000, refine_steps=ref)
        torch.cuda.synchronize()
        out[tag]=(float((u.double()-torch.as_tensor(phi(P),device="cuda")).abs().max()),
                  time.time()-t0)
    print(f"{hmin:10.3e} {R.nnode:9d} | {out['fp64'][0]:11.3e} {out['fp64'][1]:7.2f} | "
          f"{out['fp32'][0]:13.3e} {out['fp32'][1]:7.2f} | {out['fp32'][0]/out['fp64'][0]:7.3f}")
