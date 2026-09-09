#!/usr/bin/env python3
"""Solve the pixel-anode drift and weighting fields and write .npz outputs.

    python3 make_outputs.py --outdir out --lmax 5 --npad 5 --ngrid 10

Writes, into --outdir:

    drift_field.npz        phi, E on the octree nodes  (native, exact)
    weighting_field.npz    phi_w, E_w on the octree nodes
    drift_field_grid.npz   phi, E on a regular grid over the whole volume
    weighting_field_grid.npz
    near_anode_grid.npz    both fields at h_min over the near-anode slab
    induced_current.npz    i(t), Q(t) per drift path + the trajectories
    README_outputs.txt     what every array is, with units and shapes
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from fdm.drift import (E_CHARGE, Interpolator, drift_paths, drift_velocity,
                       fill_missing, induced_current)
from fdm.field import Gradient
from fdm.geometry import PixelAnode
from fdm.io import sample_grid, save_currents, save_field, save_grid
from fdm.operator import Operator
from fdm.solve import solve
from fdm.topology import build_gradient, build_rows
from fdm.tree import Octree


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--pitch", type=float, default=4.434, help="pad pitch, mm")
    ap.add_argument("--lmax", type=int, default=5, help="h_min = pitch/2**lmax")
    ap.add_argument("--npad", type=int, default=5, help="NxN periodic supercell")
    ap.add_argument("--nz", type=int, default=12, help="drift length in pitches")
    ap.add_argument("--efield", type=float, default=500.0, help="drift field, V/cm")
    ap.add_argument("--ngrid", type=int, default=10, help="NxN drift start points")
    ap.add_argument("--slab", type=float, default=3.0,
                    help="near-anode fine-grid thickness, in pitches")
    ap.add_argument("--pad-before", type=float, default=None,
                    help="us of baseline before the earliest emission "
                         "(default: 20%% of the longest drift)")
    ap.add_argument("--pad-after", type=float, default=2.0,
                    help="us of tail after the latest arrival")
    ap.add_argument("--dt", type=float, default=0.05, help="sample period, us")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    hmin = a.pitch / (1 << a.lmax)
    LD = a.nz * a.pitch
    E0 = a.efield / 10.0                      # V/cm -> V/mm
    rng = np.random.default_rng(20260907)

    # ---------------------------------------------------------------- grid
    t0 = time.time()
    tree = Octree((a.npad, a.npad, a.nz), lmax=a.lmax, h_min=hmin,
                  periodic=(True, True, False))
    pad = PixelAnode(tree, pitch=a.pitch, pad=(11 << (a.lmax - 4)) * hmin)
    tree.build(pad.refine_predicate(grade=8.0))

    R0 = build_rows(tree)
    co = R0.coords
    zmax = int(tree.dims_units[2])
    shift = (a.npad // 2) * int(a.pitch / hmin)
    cc = co.copy(); cc[:, 0] -= shift; cc[:, 1] -= shift

    on_pads = pad.on_pad(co)
    centre = pad.on_pad(cc, centre_only=True)
    gap = pad.on_anode(co) & ~on_pads          # bare dielectric -> Neumann
    cathode = co[:, 2] == zmax

    rows = build_rows(tree, neumann=gap)
    A = Operator(rows, dirichlet=on_pads | cathode, device=a.device)
    print(f"grid: {len(tree)} cells, {rows.nnode} nodes "
          f"({rows.irr_row.size} irregular), h_min {hmin*1000:.1f} um, "
          f"{time.time()-t0:.1f}s")

    # --------------------------------------------------------------- solve
    def run(V, tag):
        s = time.time()
        u, info = solve(A, A.rhs(torch.as_tensor(V, device=a.device)),
                        tol=1e-11, maxiter=80000)
        print(f"  {tag}: {info.iters} iters, res {info.residual:.1e}, "
              f"{time.time()-s:.2f}s")
        return u

    Vd = np.zeros(rows.nnode); Vd[cathode] = -E0 * LD
    Vw = np.zeros(rows.nnode); Vw[centre] = 1.0
    phi_d = run(Vd, "drift    ")
    phi_w = run(Vw, "weighting")

    G = Gradient(build_gradient(tree, neumann=gap), device=a.device)
    grow = G.row.cpu().numpy()
    have = np.zeros(rows.nnode, bool); have[grow] = True

    def Efield(u):
        E = np.zeros((rows.nnode, 3)); E[grow] = (-G(u)).cpu().numpy()
        return fill_missing(tree, E, have)

    Ed, Ew = Efield(phi_d), Efield(phi_w)
    pd_, pw_ = phi_d.cpu().numpy(), phi_w.cpu().numpy()

    geom = dict(pitch=a.pitch, pad_width=pad.pad, gap=pad.gap, h_min=hmin,
                npad=a.npad, drift_length=LD, efield_Vcm=a.efield,
                anode_z=0.0, cathode_z=LD)

    # ------------------------------------------------------- native fields
    p = save_field(f"{a.outdir}/drift_field.npz", tree, co, pd_, Ed,
                   description="drift field: all pads 0 V, cathode at -E0*L, "
                               "inter-pad gap Neumann",
                   **geom)
    print(f"wrote {p}  ({rows.nnode} nodes)")
    p = save_field(f"{a.outdir}/weighting_field.npz", tree, co, pw_, Ew,
                   description="weighting field of the centre pad (1 V), all "
                               "other pads and cathode 0 V",
                   centre_pad_xy=np.array([shift * hmin, shift * hmin]), **geom)
    print(f"wrote {p}")

    # --------------------------------------------------------- regular grids
    L = a.npad * a.pitch
    p, shp = save_grid(f"{a.outdir}/drift_field_grid.npz", tree, pd_, Ed,
                       (0, L), (0, L), (0, LD), a.pitch / 8, **geom)
    print(f"wrote {p}  shape {shp}")
    p, shp = save_grid(f"{a.outdir}/weighting_field_grid.npz", tree, pw_, Ew,
                       (0, L), (0, L), (0, LD), a.pitch / 8, **geom)
    print(f"wrote {p}  shape {shp}")

    zs = a.slab * a.pitch
    x, y, z, pgd, Egd = sample_grid(tree, pd_, Ed, (0, L), (0, L), (0, zs), hmin)
    _, _, _, pgw, Egw = sample_grid(tree, pw_, Ew, (0, L), (0, L), (0, zs), hmin)
    np.savez_compressed(f"{a.outdir}/near_anode_grid.npz", x=x, y=y, z=z,
                        phi_drift=pgd, E_drift=Egd,
                        phi_weight=pgw, E_weight=Egw,
                        spacing=hmin, units_length="mm", units_field="V/mm",
                        **{k: np.asarray(v) for k, v in geom.items()})
    print(f"wrote {a.outdir}/near_anode_grid.npz  shape {pgd.shape} at h_min")

    # ---------------------------------------------------------- drift + Ramo
    NG = a.ngrid
    g = np.linspace(0.1, 0.9, NG)
    gx, gy = np.meshgrid(g, g, indexing="ij")
    starts = np.stack([(gx.ravel() + rng.uniform(-.04, .04, NG * NG)) * L,
                       (gy.ravel() + rng.uniform(-.04, .04, NG * NG)) * L,
                       rng.uniform(0.15, 0.95, NG * NG) * LD], 1)
    s = time.time()
    paths = drift_paths(tree, Ed, starts, z_stop=0.0)
    currents = [induced_current(tree, pw_, Ew, pp) for pp in paths]
    print(f"tracked {len(paths)} paths + Ramo in {time.time()-s:.1f}s")

    qp = np.array([c[3][-1] for c in currents])
    p, shp = save_currents(f"{a.outdir}/induced_current.npz", paths, currents,
                           starts, dt=a.dt, pad_before=a.pad_before,
                           pad_after=a.pad_after,
                           collected=qp > 0.5 * E_CHARGE,
                           electron_charge_fC=E_CHARGE,
                           v_drift_bulk=drift_velocity(E0 / 100), **geom)
    print(f"wrote {p}  i shape {shp}")

    with open(f"{a.outdir}/README_outputs.txt", "w") as f:
        f.write(README.format(
            pitch=a.pitch, pad=pad.pad, gap=pad.gap, hmin=hmin * 1000,
            npad=a.npad, LD=LD, E0=a.efield, nnode=rows.nnode,
            ncell=len(tree), npath=len(paths), ngrid=shp[1],
            slab=a.slab, nearshape=pgd.shape))
    print(f"wrote {a.outdir}/README_outputs.txt")


README = """Pixel-anode field and signal outputs
===================================

Geometry: pitch {pitch} mm, pad {pad:.4f} mm, gap {gap:.4f} mm,
h_min {hmin:.1f} um, {npad}x{npad} periodic supercell, drift {LD:.1f} mm
at {E0:.0f} V/cm.  Octree: {ncell} cells, {nnode} nodes.
Anode plane at z=0 (pads Dirichlet, gap Neumann); cathode at z={LD:.1f} mm.

Units: mm, V, V/mm, us, fC, pA.

drift_field.npz / weighting_field.npz   -- native, exact, on the octree nodes
    phi        (nnode,)     potential, V
    E          (nnode, 3)   E = -grad(phi), V/mm
    pos        (nnode, 3)   node positions, mm
    coords     (nnode, 3)   integer lattice coordinates (units of h_min)
    tree_*                  the octree, so the field can be re-interpolated:

        from fdm.io import field_interpolator
        tree, d, phi_at, E_at = field_interpolator("drift_field.npz")
        E = E_at(np.array([[x, y, z]]))     # anywhere, mm

drift_field_grid.npz / weighting_field_grid.npz   -- regular grid, whole volume
    x, y, z    axes, mm (spacing = pitch/8)
    phi        (nx, ny, nz)      V
    E          (nx, ny, nz, 3)   V/mm

near_anode_grid.npz    -- both fields at full h_min over the first {slab} pitches
    x, y, z                       axes, mm (spacing = h_min)
    phi_drift, phi_weight         {nearshape}
    E_drift,  E_weight            {nearshape} + (3,)
    This is the region where the pixel structure matters; the rest of the
    drift volume is uniform to <1e-3 and the coarse grid is sufficient there.

induced_current.npz    -- {npath} drift paths, Shockley-Ramo on the centre pad
    t          ({ngrid},)         us, uniform, t=0 is the ARRIVAL time
    i          (npath, {ngrid})   induced current, pA
    q          (npath, {ngrid})   cumulative induced charge, fC
    t_abs      (nt_abs,)          us, t=0 is emission (absolute readout time)
    i_abs, q_abs (npath, nt_abs)  same signals on the absolute time base
    t_drift    (npath,)           drift duration, us
    pad_before, pad_after         us of margin either side of the drift

    The time base is wider than the longest drift on purpose, so the pulse is
    not clipped at the array edges.  Outside its own drift a waveform is zero
    (the carrier is not moving); the charge holds its final value instead.
    q_total    (npath,)           final induced charge, fC
    t_arrival  (npath,)           total drift time, us
    start, end (npath, 3)         start and landing positions, mm
    traj       (npath, maxstep, 4)  [x, y, z, t], NaN-padded
    traj_len   (npath,)           valid length of each trajectory
    collected  (npath,) bool      landed on the centre pad
    electron_charge_fC            1 e in fC, for normalising q

Sign convention: for a carrier of charge q the induced charge on electrode k is
Q_k = -q [phi_w(x) - phi_w(x0)] and the current is i_k = dQ/dt = q v . E_w.
For an electron (q = -e) that means Q = +e[phi_w - phi_w0] and i = -e v . E_w.
"""


if __name__ == "__main__":
    main()
