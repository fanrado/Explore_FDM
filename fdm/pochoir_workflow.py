"""Drift field, weighting field, paths and currents from a pochoir config.

The same solver as :mod:`fdm.workflow`, driven by a pochoir JSON config
(e.g. ``example_gen_pcb_drift_pixel_task13_fine.json``) and arranged so every
output lines up with the corresponding pochoir store entry:

=====================  ======================================================
ours (``outdir/``)      pochoir store
=====================  ======================================================
drift_lattice.npz       potential/drift3d     (N, N, Nz) drift tile
weighting_lattice.npz   potential/weight3d    (Nw, Nw, Nz) weighting probe
paths.npz               paths/drift3d         (ngrid**2, nsteps, 3)
current.npz             current/induced_current  (n_imp**2, nsteps-1)
=====================  ======================================================

The geometry translation is in :mod:`fdm.pochoir`; read its docstring for what
is modelled and why.  Two separate trees are solved, because pochoir solves two
different problems: a periodic single-pitch tile for the drift field and an
``Npixels x Npixels`` probe with held-at-zero walls for the weighting field.

Boundaries that do not fall on a root-cell face (the weighting walls, which are
22.1 mm apart, and the cathode, 150.6 mm from the anode) are *embedded*: the
octree box is rounded up to whole root cells, every cell that straddles the
boundary plane is refined until the plane is a cell face, and every node on or
beyond the plane is held at the boundary value.  No stencil of a free node can
then reach past the plane, so the result is the same as if the box ended there.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import pochoir as P
from .drift import E_CHARGE, Interpolator, drift_paths, fill_missing
from .field import Gradient
from .io import load_field, save_field
from .operator import Operator
from .solve import solve
from .topology import build_gradient, build_rows, node_stencil
from .tree import Octree
from .workflow import Context, StageLog


@dataclass
class PochoirConfig:
    outdir: str = "out_pochoir"
    # pochoir inputs
    drift_config: str = ""            # pochoir drift JSON (required)
    weighting_config: str | None = None  # pochoir weighting JSON (optional)
    spacing: float = 0.1              # pochoir grid spacing the configs are for, mm
    cathode_idx: int | None = None    # override pochoir's last z plane index
    # our mesh
    refine: int = 0                   # h_min = spacing / 2**refine
    grade: float = 8.0                # cell size ~ distance_to_anode / grade
    edge_levels: float = 2.0          # h_min within this many cells of a pad edge
    max_root: float = 4.0             # mm, largest root cell (weighting tree)
    # weighting side walls: "fixed" holds them at 0 (pochoir true_fixed);
    # "mirror" makes them zero-slope (pochoir field-solve --mirror-edges), on
    # the planes x, y = -s and (Nw - 1) s, 22.0 mm apart and symmetric about the
    # pads (pochoir's own halo copy puts them s/2 further in).  They must be
    # root-cell faces, so the root cell must divide Nw cells (max_root 0.4).
    weighting_walls: str = "fixed"
    lmax_cap: int = 8
    # solver
    tol: float = 1e-11
    maxiter: int = 80000
    device: str = "cuda"
    dtype: str = "float64"
    # export on pochoir's lattice
    export_drift_zmax: float | None = None   # mm above the anode; None = to cathode
    export_weighting_zmax: float | None = 30.0
    # drift and currents (pochoir run-pixel-field.sh defaults)
    velocity: str = "bnl"             # "bnl" (pochoir) or "icarus"
    temperature: float = 87.0         # K
    tick: float = 0.05                # us
    tmax: float = 200.0               # us

    @property
    def torch_dtype(self):
        return dict(float64=torch.float64, float32=torch.float32)[self.dtype]

    def geometry(self) -> P.PochoirGeometry:
        if not self.drift_config:
            raise ValueError("drift_config (a pochoir drift JSON) is required")
        return P.from_configs(self.drift_config, self.weighting_config,
                              spacing=self.spacing, cathode_idx=self.cathode_idx)


# ------------------------------------------------------------------ meshing

def _predicate(origin, h, pads, pad_offset, z_phys_top, walls_hi, planes,
               grade, edge_levels, pitch):
    """Refinement rule shared by both trees.

    ``planes`` are (axis, units) boundary planes that must become cell faces.
    ``walls_hi`` are (axis, units) upper limits of the physical box; cells
    wholly beyond them, or beyond the cathode, are left coarse.
    """
    def pred(centers, sizes):
        cu = (centers - origin) / h
        su = sizes / h
        lo = cu - 0.5 * su[:, None]
        hi = cu + 0.5 * su[:, None]
        dz = centers[:, 2] - origin[2]
        target = np.maximum(dz / grade, h)
        near = pads.near_edge(cu[:, 0] - pad_offset, cu[:, 1] - pad_offset,
                              edge_levels * su)
        target = np.where(near & (dz < edge_levels * pitch), h, target)
        split = sizes > 1.001 * target
        outside = lo[:, 2] >= z_phys_top
        for a, q in walls_hi:
            outside |= lo[:, a] >= q
        split &= ~outside
        for a, q in planes:
            split |= (lo[:, a] < q) & (hi[:, a] > q)
        return split
    return pred


def build_drift_mesh(cfg: PochoirConfig, geom: P.PochoirGeometry):
    m = 1 << cfg.refine
    h = geom.spacing / m
    Pu = geom.N * m                           # period, units
    lmax = P.root_level(Pu, cfg.lmax_cap)
    U = 1 << lmax
    Lc = (geom.cathode_idx - geom.anode_idx) * m
    nzr = P.ceil_div(Lc, U)
    origin = np.array([0.0, 0.0, geom.z_anode])
    tree = Octree((Pu // U, Pu // U, nzr), lmax=lmax, h_min=h, origin=origin,
                  periodic=(True, True, False))
    pads = P.drift_pads(geom, m)
    tree.build(_predicate(origin, h, pads, 0, Lc, [], [(2, Lc)],
                          cfg.grade, cfg.edge_levels, geom.pitch))

    coords, _ = tree.nodes()
    anode = coords[:, 2] == 0
    on_pad = anode & pads.contains(coords[:, 0], coords[:, 1])
    gap = anode & ~on_pad
    cathode = coords[:, 2] >= Lc
    V = np.zeros(len(coords))
    V[cathode] = geom.cathode_potential
    return dict(tree=tree, coords=coords, dirichlet=on_pad | cathode, gap=gap,
                V=V, on_pad=on_pad, root_mm=U * h, Lc=Lc, m=m)


def build_weighting_mesh(cfg: PochoirConfig, geom: P.PochoirGeometry):
    m = 1 << cfg.refine
    h = geom.spacing / m
    lmax = min(cfg.lmax_cap, int(np.floor(np.log2(cfg.max_root / h + 1e-9))))
    U = 1 << lmax
    if cfg.weighting_walls not in ("fixed", "mirror"):
        raise ValueError(f"weighting_walls must be 'fixed' or 'mirror', not {cfg.weighting_walls!r}")
    mirror = cfg.weighting_walls == "mirror"
    # far wall, units from the near wall: fixed walls sit on pochoir's halo
    # nodes (-s and Nw s); mirror walls on -s and (Nw - 1) s
    W = (geom.Nw if mirror else geom.Nw + 1) * m
    Lc = (geom.w_cathode_idx - geom.anode_idx) * m
    if mirror and W % U:
        raise ValueError(f"mirror walls must be root-cell faces: the root cell ({U} cells) "
                         f"does not divide the {W}-cell box; lower max_root")
    nxr, nzr = P.ceil_div(W, U), P.ceil_div(Lc, U)
    origin = np.array([-geom.spacing, -geom.spacing, geom.z_anode])
    tree = Octree((nxr, nxr, nzr), lmax=lmax, h_min=h, origin=origin,
                  periodic=(False, False, False))
    pads = P.weighting_pads(geom, m)
    tree.build(_predicate(origin, h, pads, m, Lc, [(0, W), (1, W)],
                          [(0, W), (1, W), (2, Lc)],
                          cfg.grade, cfg.edge_levels, geom.pitch))

    coords, _ = tree.nodes()
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    sides = (x <= 0) | (y <= 0) | (x >= W) | (y >= W)
    cathode = z >= Lc
    walls = sides | cathode
    anode = (z == 0) & ~(cathode if mirror else walls)
    on_pad = anode & pads.contains(x - m, y - m)       # pochoir frame
    centre = anode & pads.contains(x - m, y - m, which=0)
    gap = anode & ~on_pad
    if mirror:
        # zero-slope side walls: mirrored rows, like the anode gaps (the walls
        # are the tree's outer faces)
        gap = gap | (sides & ~cathode)
        walls = cathode
    V = np.zeros(len(coords))
    V[centre] = 1.0
    return dict(tree=tree, coords=coords, dirichlet=on_pad | walls, gap=gap,
                V=V, on_pad=on_pad, centre=centre, root_mm=U * h, Lc=Lc, m=m)


# ------------------------------------------------------------------- stages

def stage_geometry(cfg: PochoirConfig, ctx: Context):
    """Translate the pochoir config(s) into integer geometry."""
    geom = cfg.geometry()
    ctx.geom = geom
    print("    " + P.describe(geom).replace("\n", "\n    "))
    with open(f"{cfg.outdir}/geometry.json", "w") as f:
        json.dump(dict(pochoir=geom.summary(), config=asdict(cfg)), f, indent=2)
    return dict(N=geom.N, Nw=geom.Nw, anode_idx=geom.anode_idx,
                cathode_idx=geom.cathode_idx)


def _mesh(cfg, ctx, which):
    builder = build_drift_mesh if which == "drift" else build_weighting_mesh
    mesh = builder(cfg, ctx.geom)
    # one stencil for the operator here and the gradient in stage_efields
    mesh["stencil"] = node_stencil(mesh["tree"], neumann=mesh["gap"])
    rows = build_rows(mesh["tree"], neumann=mesh["gap"], st=mesh["stencil"])
    mesh["rows"] = rows
    mesh["A"] = Operator(rows, dirichlet=mesh["dirichlet"], device=cfg.device,
                         dtype=cfg.torch_dtype)
    ctx[which] = mesh
    t = mesh["tree"]
    print(f"    {which}: {len(t)} cells, {rows.nnode} nodes "
          f"({100 * rows.irr_row.size / rows.nnode:.1f}% irregular), "
          f"h_min {t.h_min * 1000:.1f} um, root {mesh['root_mm']:.3f} mm, "
          f"box {t.root_dims.tolist()} roots")
    return dict(cells=len(t), nodes=rows.nnode, irregular=int(rows.irr_row.size),
                root_mm=mesh["root_mm"], h_min=t.h_min)


def _solve(cfg, ctx, which):
    mesh = ctx[which]
    A = mesh["A"]
    b = A.rhs(torch.as_tensor(mesh["V"], device=cfg.device, dtype=cfg.torch_dtype))
    u, info = solve(A, b, tol=cfg.tol, maxiter=cfg.maxiter)
    print(f"    {which}: {info.iters} iters, rel.res {info.residual:.2e}, "
          f"converged={info.converged}")
    if not info.converged:
        print(f"    WARNING: {which} did not reach tol={cfg.tol}")
    mesh["u"] = u
    return dict(iters=info.iters, residual=info.residual,
                converged=bool(info.converged))


def stage_grid_drift(cfg, ctx):
    """Periodic single-pitch drift tile: octree, operator, electrode masks."""
    return _mesh(cfg, ctx, "drift")


def stage_solve_drift(cfg, ctx):
    """Drift field: pads 0 V, gap Neumann, cathode at CathodePotential."""
    return _solve(cfg, ctx, "drift")


def stage_grid_weighting(cfg, ctx):
    """Npixels x Npixels probe with held walls: octree, operator, masks."""
    return _mesh(cfg, ctx, "weighting")


def stage_solve_weighting(cfg, ctx):
    """Weighting field: centre pad 1, other pads, walls and cathode 0."""
    return _solve(cfg, ctx, "weighting")


def stage_efields(cfg, ctx):
    """E = -grad(phi) on both trees, from the discretization's own gradient."""
    out = {}
    for which in ("drift", "weighting"):
        mesh = ctx[which]
        G = Gradient(build_gradient(mesh["tree"], neumann=mesh["gap"],
                                    st=mesh.pop("stencil", None)),
                     device=cfg.device, dtype=cfg.torch_dtype)
        grow = G.row.cpu().numpy()
        have = np.zeros(mesh["rows"].nnode, bool)
        have[grow] = True
        E = np.zeros((mesh["rows"].nnode, 3))
        E[grow] = (-G(mesh["u"])).cpu().numpy()
        mesh["E"] = fill_missing(mesh["tree"], E, have)
        mesh["phi"] = mesh["u"].cpu().numpy()
        mesh["interp"] = Interpolator(mesh["tree"])
        out[f"{which}_gradient_nodes"] = int(len(grow))
    return out


def _meta(cfg, geom, which):
    d = dict(pochoir_spacing=geom.spacing, pitch=geom.pitch,
             z_anode=geom.z_anode, z_cathode=geom.z_cathode,
             cathode_potential=geom.cathode_potential,
             pochoir_geometry=json.dumps(geom.summary()),
             frame="pochoir drift tile" if which == "drift"
             else "pochoir weighting probe")
    return d


class _Saves:
    """File writes on background threads, so the GPU and the next stages carry
    on while the CPU compresses and writes (zlib and file I/O release the GIL).
    The arrays handed over must not be modified afterwards.  :meth:`wait` joins
    every write and re-raises the first error; :func:`run` calls it before
    returning."""

    def __init__(self):
        self.threads, self.errors = [], []

    def submit(self, fn, *args, **kwargs):
        def work():
            try:
                fn(*args, **kwargs)
            except BaseException as e:   # noqa: BLE001 -- re-raised in wait()
                self.errors.append(e)
        th = threading.Thread(target=work)
        th.start()
        self.threads.append(th)

    def wait(self):
        for th in self.threads:
            th.join()
        self.threads = []
        if self.errors:
            err, self.errors = self.errors[0], []
            raise err


def _saves(ctx) -> _Saves:
    if "saves" not in ctx:
        ctx.saves = _Saves()
    return ctx.saves


def stage_export_fields(cfg, ctx):
    """Native octree-node fields (exact; reload with fdm.io.load_field).
    Written in the background; see :class:`_Saves`."""
    geom = ctx.geom
    for which, desc in (("drift", "drift field from pochoir config"),
                        ("weighting", "weighting field of the centre pad")):
        mesh = ctx[which]
        _saves(ctx).submit(save_field, f"{cfg.outdir}/{which}_field.npz", mesh["tree"],
                           mesh["rows"].coords, mesh["phi"], mesh["E"], description=desc,
                           **_meta(cfg, geom, which))
    return dict(files=2)


def _lattice(interp, phi, E, nx, ny, k0, nz, s, chunk_planes=None):
    """Sample a field at pochoir lattice nodes (i s, j s, k s)."""
    i = np.arange(nx) * s
    j = np.arange(ny) * s
    X, Y = np.meshgrid(i, j, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    pg = np.empty((nx, ny, nz))
    Eg = np.empty((nx, ny, nz, 3), np.float32)
    per = max(1, 4_000_000 // len(xy)) if chunk_planes is None else chunk_planes
    for k in range(0, nz, per):
        ks = np.arange(k, min(k + per, nz))
        z = (k0 + ks) * s
        pts = np.concatenate([np.column_stack([xy, np.full(len(xy), zz)])
                              for zz in z])
        ids, w = interp.weights(pts)
        ph = (phi[ids] * w).sum(1)
        ee = (E[ids] * w[:, :, None]).sum(1)
        pg[:, :, ks] = ph.reshape(len(ks), nx, ny).transpose(1, 2, 0)
        Eg[:, :, ks] = ee.reshape(len(ks), nx, ny, 3).transpose(1, 2, 0, 3)
    return pg, Eg


def _lattice_torch(tree, phi, E, nx, ny, k0, nz, s, device, chunk_points=4_000_000):
    """:func:`_lattice` on ``device`` with torch: the same leaf lookup as
    ``Octree.locate_float`` (per-level sorted keys, finest first) and the same
    trilinear weights as ``Interpolator.weights``, so the result agrees with the
    NumPy path to round-off.

    The lattice-unit coordinates ``q`` are computed in NumPy: torch divides by a
    scalar as a multiply by its reciprocal, which moves points lying exactly on
    a cell face by an ulp into the neighbouring leaf, and the trilinear field is
    not exactly continuous across a coarse/fine face."""
    dt = torch.float64
    t = lambda a, d=None: torch.as_tensor(np.ascontiguousarray(a), device=device, dtype=d)  # noqa: E731
    _, cid = tree.nodes()
    cid = t(cid, torch.int64)
    org = t(tree.origins_units(), dt)
    size = t(tree.sizes_units(), dt)
    lut = {lv: (t(k, torch.int64), t(c, torch.int64)) for lv, (k, c) in tree._lookup_tables().items()}
    dims = [float(d) for d in tree.dims_units]
    phi_t, E_t = t(phi, dt), t(E, dt)

    i = np.arange(nx) * s
    j = np.arange(ny) * s
    X, Y = np.meshgrid(i, j, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], 1)
    pg = np.empty((nx, ny, nz))
    Eg = np.empty((nx, ny, nz, 3), np.float32)
    per = max(1, chunk_points // len(xy))
    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    for k in range(0, nz, per):
        ks = np.arange(k, min(k + per, nz))
        z = (k0 + ks) * s
        pts = np.concatenate([np.column_stack([xy, np.full(len(xy), zz)]) for zz in z])
        q = t((pts - tree.origin) / tree.h_min, dt)
        for a in range(3):
            if tree.periodic[a]:
                q[:, a] = torch.remainder(q[:, a], dims[a])
            else:
                q[:, a] = q[:, a].clamp(0.0, float(np.nextafter(dims[a], 0.0)))
        cell = torch.full((q.shape[0],), -1, device=device, dtype=torch.int64)
        todo = torch.arange(q.shape[0], device=device)
        for level in range(tree.lmax, -1, -1):
            if todo.numel() == 0:
                break
            if level not in lut:
                continue
            keys, cells = lut[level]
            c = float(tree.cell_size_units(level))
            ijk = torch.floor(q[todo] / c).to(torch.int64)
            ldy, ldz = (int(v) for v in tree.level_dims(level)[1:])
            kk = (ijk[:, 0] * ldy + ijk[:, 1]) * ldz + ijk[:, 2]
            pos = torch.searchsorted(keys, kk).clamp_(0, keys.numel() - 1)
            hit = keys[pos] == kk
            cell[todo[hit]] = cells[pos[hit]]
            todo = todo[~hit]
        f = ((q - org[cell]) / size[cell][:, None]).clamp_(0.0, 1.0)
        w = torch.stack([(f[:, 0] if a else 1 - f[:, 0]) * (f[:, 1] if b else 1 - f[:, 1])
                         * (f[:, 2] if c else 1 - f[:, 2]) for a, b, c in corners], 1)
        ids = cid[cell]
        ph = (phi_t[ids] * w).sum(1)
        ee = (E_t[ids] * w[:, :, None]).sum(1)
        pg[:, :, ks] = ph.view(len(ks), nx, ny).permute(1, 2, 0).cpu().numpy()
        Eg[:, :, ks] = ee.view(len(ks), nx, ny, 3).permute(1, 2, 0, 3).cpu().numpy()
    return pg, Eg


def stage_export_lattice(cfg, ctx):
    """Both fields resampled on pochoir's own lattice, index-aligned (on the
    GPU; written in the background)."""
    geom = ctx.geom
    s = geom.spacing
    out = {}
    for which, n, kc, zmax in (
            ("drift", geom.N, geom.cathode_idx, cfg.export_drift_zmax),
            ("weighting", geom.Nw, geom.w_cathode_idx, cfg.export_weighting_zmax)):
        mesh = ctx[which]
        k0 = geom.anode_idx
        k1 = kc if zmax is None else min(kc, k0 + int(round(zmax / s)))
        nz = k1 - k0 + 1
        pg, Eg = _lattice_torch(mesh["tree"], mesh["phi"], mesh["E"], n, n, k0, nz, s, cfg.device)
        _saves(ctx).submit(np.savez, f"{cfg.outdir}/{which}_lattice.npz", phi=pg, E=Eg,
                 i0=0, j0=0, k0=k0, spacing=s,
                 note=f"phi[i, j, k] is pochoir potential/"
                      f"{'drift3d' if which == 'drift' else 'weight3d'}"
                      f"[i, j, k0 + k]; x = i*s, y = j*s, z = (k0+k)*s mm",
                 **_meta(cfg, geom, which))
        out[f"{which}_shape"] = list(pg.shape)
        print(f"    {which}_lattice.npz {pg.shape}, planes {k0}..{k1}")
    return out


def _need_fields(cfg, ctx):
    """Reload both fields from disk when not in memory (for --only runs)."""
    if "geom" not in ctx:
        ctx.geom = cfg.geometry()
    for which in ("drift", "weighting"):
        if which in ctx and "E" in ctx[which]:
            continue
        print(f"    (loading {which} field from disk)")
        tree, d = load_field(f"{cfg.outdir}/{which}_field.npz")
        ctx[which] = dict(tree=tree, phi=d["phi"], E=d["E"],
                          interp=Interpolator(tree))


def stage_drift(cfg, ctx):
    """Track pochoir's start grid on the drift tile, sample on pochoir's ticks."""
    _need_fields(cfg, ctx)
    geom = ctx.geom
    mesh = ctx.drift
    starts = P.pochoir_start_points(geom)
    if cfg.velocity == "bnl":
        speed = lambda E: P.bnl_drift_velocity(E, cfg.temperature)   # noqa: E731
    elif cfg.velocity == "icarus":
        speed = None
    else:
        raise ValueError(f"velocity must be 'bnl' or 'icarus', not {cfg.velocity!r}")
    paths, arrived = drift_paths(mesh["tree"], mesh["E"], starts,
                                 z_stop=geom.z_anode, max_time=cfg.tmax,
                                 temperature=cfg.temperature, speed=speed,
                                 return_status=True)
    nsteps = int(round(cfg.tmax / cfg.tick))
    t = cfg.tick * np.arange(nsteps)
    xyz = np.empty((len(paths), nsteps, 3))
    for n, p in enumerate(paths):
        for a in range(3):
            xyz[n, :, a] = np.interp(t, p[:, 3], p[:, a])   # held after arrival
    t_arr = np.array([p[-1, 3] for p in paths])
    if not arrived.all():
        print(f"    WARNING: {int((~arrived).sum())} of {len(paths)} paths "
              f"did not reach the anode within {cfg.tmax} us")
    np.savez_compressed(f"{cfg.outdir}/paths.npz", paths=xyz, t=t, starts=starts,
                        arrived=arrived, t_arrival=t_arr,
                        end=np.array([p[-1, :3] for p in paths]),
                        velocity=cfg.velocity, temperature=cfg.temperature,
                        note="paths[n, k] = position at t[k] (us), drift-tile "
                             "frame, same layout as pochoir paths/drift3d")
    ctx.paths, ctx.t, ctx.starts, ctx.arrived = xyz, t, starts, arrived
    print(f"    {int(arrived.sum())}/{len(paths)} arrived, drift time "
          f"{t_arr[arrived].min():.2f}..{t_arr[arrived].max():.2f} us")
    return dict(npaths=len(paths), arrived=int(arrived.sum()),
                t_min=float(t_arr.min()), t_max=float(t_arr.max()))


def impact_layout(geom: P.PochoirGeometry):
    """pochoir induce-pixel's impact grid: path index and pitch shift per axis.

    Mirrors ``_shift_paths_pixel_grid`` (pochoir/__main__.py), which tiles the
    single-pitch paths over ``Npixels // 2`` full pitches plus a final half
    pitch, starting at the collecting pad's centre.  For Npixels = 5 and 10
    start points per side that is 2*10 + 5 = 25 impact positions per axis.
    (pochoir's ``--npixels`` flag only switches this tiling on; the extent
    comes from ``Npixels`` in the config.)
    """
    n = geom.ngrid
    a = np.arange((geom.npixels // 2) * n + n // 2)
    return a % n, a // n


def stage_currents(cfg, ctx):
    """Induced charge and current on the centre pad, pochoir's impact grid."""
    _need_fields(cfg, ctx)
    geom = ctx.geom
    if "paths" not in ctx:
        z = np.load(f"{cfg.outdir}/paths.npz")
        ctx.paths, ctx.t = z["paths"], z["t"]
    xyz, t = ctx.paths, ctx.t
    lvl, kpix = impact_layout(geom)
    na = len(lvl)
    c = geom.weighting_pad_centre() - geom.drift_pad_centre()
    W = ctx.weighting
    Q = np.empty((na, na, len(t)))
    for ia in range(na):
        for ib in range(na):
            path = xyz[lvl[ib] + lvl[ia] * geom.ngrid]
            shift = np.array([c + kpix[ia] * geom.pitch,
                              c + kpix[ib] * geom.pitch, 0.0])
            Q[ia, ib] = W["interp"](W["phi"], path + shift)
    dW = np.diff(Q, axis=2)
    dt = np.diff(t)
    current_nA = E_CHARGE * dW / dt                     # fC/us = nA
    imp = geom.start_spacing * (0.5 + np.arange(na))
    np.savez_compressed(
        f"{cfg.outdir}/current.npz", Q=Q, t=t, current_nA=current_nA,
        current_pochoir=(dW / (dt * 1000.0)).reshape(na * na, -1),
        impact=imp, path_index=lvl, pitch_shift=kpix,
        pad_centre=np.array([geom.weighting_pad_centre()] * 2),
        note="Q[a, b, k] = weighting potential of the centre pad at the "
             "electron position, impact (impact[a], impact[b]) from the pad "
             "centre; current_pochoir is dQ/dt per ns flattened a-major, the "
             "layout and units of pochoir current/induced_current")
    q_end = Q[:, :, -1] - Q[:, :, 0]
    print(f"    {na}x{na} impacts; collected on the centre pad: "
          f"{q_end[0, 0]:.4f} at its centre, {q_end[-1, -1]:.4f} at "
          f"({imp[-1]:.2f}, {imp[-1]:.2f}) mm")
    ctx.Q = Q
    return dict(impacts=na, q_centre=float(q_end[0, 0]))


STAGES = {
    "geometry": stage_geometry,
    "grid_drift": stage_grid_drift,
    "solve_drift": stage_solve_drift,
    "grid_weighting": stage_grid_weighting,
    "solve_weighting": stage_solve_weighting,
    "efields": stage_efields,
    "export_fields": stage_export_fields,
    "export_lattice": stage_export_lattice,
    "drift": stage_drift,
    "currents": stage_currents,
}


def run(cfg: PochoirConfig, stages=None, log_path=None):
    os.makedirs(cfg.outdir, exist_ok=True)
    names = list(STAGES) if stages is None else list(stages)
    unknown = [s for s in names if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}\nknown: {list(STAGES)}")
    log = StageLog(log_path or f"{cfg.outdir}/stages.jsonl",
                   meta=dict(config=asdict(cfg), stages=names, mode="pochoir",
                             device=torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else "cpu"))
    ctx = Context()
    for name in names:
        with log.stage(name) as extra:
            info = STAGES[name](cfg, ctx)
            if info:
                extra.update(info)
    if "saves" in ctx and ctx.saves.threads:
        with log.stage("wait_saves"):
            ctx.saves.wait()
    print(f"\ntotal {log.total():.1f}s over {len(log.records)} stages "
          f"-> {log.path}")
    return ctx, log
