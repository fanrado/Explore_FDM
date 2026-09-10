"""Assembly and solve scaling on the production geometry.

Previously reported in RESULTS.md as "Test 8" from an ad-hoc run against the
superseded symmetric stencil, with no script in the repo.  This is that
measurement, reproducible, against the paper's construction.  The numbers are
not comparable to the old table: rows narrowed from 25 to 16 entries and the
neighbour probe went from 24 locate calls per node to 8.

Single pixel cell, transverse periodic, 501 mm drift -- the same geometry as
test_pad_decay, swept over h_min.  Set FDM_BENCH_LMAX to change the range.

The bandwidth column uses an explicit model, stated here so it is reproducible:
one matvec reads, per stored entry, a 4-byte column index, an 8-byte
coefficient and an 8-byte gathered value, and writes 8 bytes per row.  Padding
entries are counted because they are read.  It is an upper bound on useful
traffic, not a hardware counter.
"""
import os
import time

import numpy as np
import torch

from fdm.geometry import PixelAnode
from fdm.operator import Operator
from fdm.solve import solve
from fdm.topology import build_rows
from fdm.tree import Octree

DEV = os.environ.get("FDM_DEVICE",
                     "cuda" if torch.cuda.is_available() else "cpu")
PITCH = 4.434
NZ = 113                        # drift = 113 * pitch = 501.0 mm
LEVELS = tuple(int(x) for x in
               os.environ.get("FDM_BENCH_LMAX", "4,5,6,7").split(","))
BYTES_PER_ENTRY = 4 + 8 + 8


def run(lmax):
    hmin = PITCH / (1 << lmax)
    t = Octree((1, 1, NZ), lmax=lmax, h_min=hmin, periodic=(True, True, False))
    pad = PixelAnode(t, pitch=PITCH, pad=(11 << (lmax - 4)) * hmin)
    t0 = time.time()
    t.build(pad.refine_predicate(grade=8.0))
    t_build = time.time() - t0

    t0 = time.time()
    R = build_rows(t)
    t_asm = time.time() - t0

    co = R.coords
    dirich = pad.on_anode(co) | (co[:, 2] == int(t.dims_units[2]))
    V = np.zeros(R.nnode)
    V[pad.on_anode(co) & ~pad.on_pad(co)] = 1.0
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    A = Operator(R, dirichlet=dirich | R.boundary, device=DEV)
    b = A.rhs(torch.as_tensor(V, device=DEV))

    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    u, info = solve(A, b, tol=1e-12, maxiter=20000)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t_solve = time.time() - t0

    # two matvecs per BiCGSTAB iteration
    nrow = R.reg_row.size + R.irr_row.size
    per_mv = R.nnz * BYTES_PER_ENTRY + 8 * nrow
    gbs = (2 * per_mv * info.iters / t_solve / 1e9) if t_solve > 0 else 0.0
    peak = (torch.cuda.max_memory_allocated() / 2 ** 30
            if DEV == "cuda" else float("nan"))
    return dict(h=hmin, cells=len(t), nodes=R.nnode, nnz=R.nnz,
                irr=100 * R.irr_row.size / R.nnode, build=t_build, asm=t_asm,
                iters=info.iters, solve=t_solve, res=info.residual,
                msit=1e3 * t_solve / max(info.iters, 1), gbs=gbs, peak=peak)


print(f"=== assembly and solve scaling, single pixel cell, {NZ * PITCH:.0f} mm "
      f"drift  (device {DEV}) ===")
print(f"{'h_min':>9} {'cells':>10} {'nodes':>10} {'irr%':>6} {'nnz':>8} "
      f"{'build':>7} {'assembly':>9} {'iters':>6} {'solve':>8} {'ms/iter':>8} "
      f"{'GB/s':>6} {'GPU':>8}")
for lmax in LEVELS:
    r = run(lmax)
    print(f"{r['h'] * 1000:8.1f}u {r['cells']:10d} {r['nodes']:10d} "
          f"{r['irr']:5.1f}% {r['nnz'] / 1e6:7.1f}M {r['build']:6.1f}s "
          f"{r['asm']:8.1f}s {r['iters']:6d} {r['solve']:7.2f}s "
          f"{r['msit']:8.2f} {r['gbs']:6.0f} {r['peak']:7.2f}G")
    assert r["res"] < 1e-10, (lmax, r["res"])
