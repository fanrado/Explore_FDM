"""The end-to-end pipeline as a sequence of named, individually runnable stages.

One place defines what the workflow *is*; ``run_workflow.py`` is only a CLI over
it, and the profiling scripts consume the stage log it writes.

Each stage records its wall time and GPU/CPU memory into a JSONL log, so a run
is self-describing after the fact without needing to be re-run under a profiler.

Stages communicate through a :class:`Context`.  A stage whose inputs are not in
the context loads them from the output directory instead, so any suffix of the
pipeline can be run on its own against an earlier run's files.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .drift import (E_CHARGE, Interpolator, drift_paths, drift_velocity,
                    fill_missing, induced_current)
from .field import Gradient
from .geometry import PixelAnode
from .io import load_field, sample_grid, save_currents, save_field, save_grid
from .operator import Operator
from .solve import solve
from .topology import build_gradient, build_rows
from .tree import Octree


# --------------------------------------------------------------------- config

# Pad width as a fraction of the pitch, when not set explicitly.  11/16 keeps
# both the pad and the gap on the h_min lattice for any lmax >= 4, and
# reproduces the 3.048 mm pad / 1.386 mm gap of the reference 4.434 mm pitch.
PAD_FRACTION = 11.0 / 16.0


@dataclass
class Config:
    outdir: str = "out"
    # geometry
    pitch: float = 4.434          # mm, pad pitch
    pad_width: float | None = None  # mm, conducting pad; None -> PAD_FRACTION
    lmax: int = 5                 # h_min = pitch / 2**lmax
    npad: int = 5                 # NxN periodic supercell
    drift_length: float = 53.208  # mm, anode plane to cathode
    efield: float = 500.0         # V/cm
    grade: float = 8.0            # cell size ~ dist_to_anode / grade
    # solver
    tol: float = 1e-11
    maxiter: int = 80000
    device: str = "cuda"
    dtype: str = "float64"
    # field export
    grid_div: int = 8             # coarse export spacing = pitch / grid_div
    slab: float = 3.0             # near-anode fine grid thickness, in pitches
    # drift / signals
    ngrid: int = 10               # NxN random-ish start points for the survey
    tick: float = 0.05            # us
    pad_before: float | None = None
    pad_after: float = 2.0
    # response table
    paths_per_pixel: int = 10
    npix: float = 2.4
    start_z: float | None = None
    grid_offset: float = 0.5
    scale: str = "electron"
    seed: int = 20260907

    @property
    def h_min(self) -> float:
        return self.pitch / (1 << self.lmax)

    @property
    def pad_mm(self) -> float:
        """Conducting pad width (mm).

        Defaults to ``PAD_FRACTION`` of the pitch, which for the reference
        4.434 mm pitch gives the 3.048 mm pad / 1.386 mm gap this study uses.
        Must be an integer multiple of ``h_min`` so every pad edge lands exactly
        on a grid line; ``PixelAnode`` raises if it does not.
        """
        if self.pad_width is not None:
            return float(self.pad_width)
        return PAD_FRACTION * self.pitch

    @property
    def gap_mm(self) -> float:
        """Inter-pad gap width (mm) -- the length scale that sets h_min."""
        return self.pitch - self.pad_mm

    @property
    def nz(self) -> int:
        """Drift length in pitches: the tree's root-cell count along z.

        Derived from ``drift_length`` rather than configured, so the config
        states the physical length and the mesh follows.  The drift must be an
        integer number of pitches; violations raise rather than snapping, for
        the same reason pad edges must land on the lattice.
        """
        n = self.drift_length / self.pitch
        if abs(n - round(n)) > 1e-9:
            raise ValueError(
                f"drift_length = {self.drift_length} mm is not an integer "
                f"multiple of pitch = {self.pitch} mm "
                f"(would be {n:.6f} pitches)")
        return int(round(n))

    @property
    def e0(self) -> float:
        """Drift field in V/mm."""
        return self.efield / 10.0

    @property
    def torch_dtype(self):
        return dict(float64=torch.float64, float32=torch.float32)[self.dtype]


class Context(dict):
    """Loose bag of stage outputs; attribute access for convenience."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


# -------------------------------------------------------------------- logging

def _rss_mib() -> float:
    """Resident set size of this process, MiB (no psutil dependency)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


class StageLog:
    """Append-only JSONL record of stage timings and memory."""

    def __init__(self, path: str, meta: dict | None = None):
        self.path = path
        self.records: list[dict] = []
        self.t0 = time.time()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"kind": "meta", "t0": self.t0,
                                **(meta or {})}) + "\n")

    @contextmanager
    def stage(self, name: str, **extra):
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        print(f"[{t_start - self.t0:7.1f}s] {name} ...", flush=True)
        # Yielded dict: a stage adds its own summary fields, which land in the
        # same JSONL record rather than being appended after it is written.
        extra_out: dict = {}
        try:
            yield extra_out
        finally:
            if cuda:
                torch.cuda.synchronize()
            t_end = time.time()
            rec = dict(
                kind="stage", stage=name,
                t_start=t_start - self.t0, t_end=t_end - self.t0,
                wall=t_end - t_start,
                gpu_peak_MiB=(torch.cuda.max_memory_allocated() / 2**20
                              if cuda else 0.0),
                gpu_now_MiB=(torch.cuda.memory_allocated() / 2**20
                             if cuda else 0.0),
                gpu_reserved_MiB=(torch.cuda.max_memory_reserved() / 2**20
                                  if cuda else 0.0),
                rss_MiB=_rss_mib(), **extra, **extra_out)
            self.records.append(rec)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{t_end - self.t0:7.1f}s] {name} done in {rec['wall']:.2f}s "
                  f"(GPU peak {rec['gpu_peak_MiB']:.0f} MiB, "
                  f"RSS {rec['rss_MiB']:.0f} MiB)", flush=True)

    def total(self) -> float:
        return sum(r["wall"] for r in self.records)


# --------------------------------------------------------------------- stages

def stage_grid(cfg: Config, ctx: Context):
    """Build the octree, the composite operator and the electrode masks."""
    tree = Octree((cfg.npad, cfg.npad, cfg.nz), lmax=cfg.lmax, h_min=cfg.h_min,
                  periodic=(True, True, False))
    pad = PixelAnode(tree, pitch=cfg.pitch, pad=cfg.pad_mm)
    tree.build(pad.refine_predicate(grade=cfg.grade))

    coords, _ = tree.nodes()
    zmax = int(tree.dims_units[2])
    shift = (cfg.npad // 2) * int(cfg.pitch / cfg.h_min)
    cc = coords.copy(); cc[:, 0] -= shift; cc[:, 1] -= shift

    on_pads = pad.on_pad(coords)
    centre = pad.on_pad(cc, centre_only=True)
    gap = pad.on_anode(coords) & ~on_pads          # bare dielectric -> Neumann
    cathode = coords[:, 2] == zmax

    rows = build_rows(tree, neumann=gap)
    A = Operator(rows, dirichlet=on_pads | cathode, device=cfg.device,
                 dtype=cfg.torch_dtype)

    ctx.update(tree=tree, pad=pad, rows=rows, A=A, coords=rows.coords,
               on_pads=on_pads, centre=centre, gap=gap, cathode=cathode,
               shift=shift)
    print(f"    {len(tree)} cells, {rows.nnode} nodes "
          f"({100*rows.irr_row.size/rows.nnode:.1f}% irregular), "
          f"h_min {cfg.h_min*1000:.1f} um")
    return dict(cells=len(tree), nodes=rows.nnode, irregular=rows.irr_row.size)


def _solve(cfg, ctx, V, tag):
    b = ctx.A.rhs(torch.as_tensor(V, device=cfg.device, dtype=cfg.torch_dtype))
    u, info = solve(ctx.A, b, tol=cfg.tol, maxiter=cfg.maxiter)
    print(f"    {tag}: {info.iters} iters, rel.res {info.residual:.2e}, "
          f"converged={info.converged}")
    if not info.converged:
        print(f"    WARNING: {tag} did not reach tol={cfg.tol}")
    return u, info


def stage_solve_drift(cfg: Config, ctx: Context):
    """Drift field: all pads at 0 V, cathode at -E0*L, gap Neumann."""
    V = np.zeros(ctx.rows.nnode)
    V[ctx.cathode] = -cfg.e0 * cfg.drift_length
    u, info = _solve(cfg, ctx, V, "drift")
    ctx.phi_d = u
    return dict(iters=info.iters, residual=info.residual,
                converged=bool(info.converged))


def stage_solve_weighting(cfg: Config, ctx: Context):
    """Weighting field of the centre pad: it at 1 V, everything else at 0."""
    V = np.zeros(ctx.rows.nnode)
    V[ctx.centre] = 1.0
    u, info = _solve(cfg, ctx, V, "weighting")
    ctx.phi_w = u
    return dict(iters=info.iters, residual=info.residual,
                converged=bool(info.converged))


def stage_efields(cfg: Config, ctx: Context):
    """E = -grad(phi) for both solves, using the discretization's own gradient."""
    G = Gradient(build_gradient(ctx.tree, neumann=ctx.gap), device=cfg.device,
                 dtype=cfg.torch_dtype)
    grow = G.row.cpu().numpy()
    have = np.zeros(ctx.rows.nnode, bool); have[grow] = True

    def E_of(u):
        E = np.zeros((ctx.rows.nnode, 3)); E[grow] = (-G(u)).cpu().numpy()
        return fill_missing(ctx.tree, E, have)

    ctx.Ed, ctx.Ew = E_of(ctx.phi_d), E_of(ctx.phi_w)
    ctx.pd = ctx.phi_d.cpu().numpy()
    ctx.pw = ctx.phi_w.cpu().numpy()
    ctx.interp = Interpolator(ctx.tree)
    return dict(gradient_nodes=int(len(grow)))


def _geom(cfg, ctx):
    return dict(pitch=cfg.pitch, pad_width=ctx.pad.pad, gap=ctx.pad.gap,
                h_min=cfg.h_min, npad=cfg.npad, drift_length=cfg.drift_length,
                efield_Vcm=cfg.efield, anode_z=0.0, cathode_z=cfg.drift_length)


def stage_export_fields(cfg: Config, ctx: Context):
    """Native (octree-node) field files -- exact and re-interpolatable."""
    g = _geom(cfg, ctx)
    save_field(f"{cfg.outdir}/drift_field.npz", ctx.tree, ctx.coords, ctx.pd,
               ctx.Ed, description="drift field: pads 0 V, cathode -E0*L, "
                                   "inter-pad gap Neumann", **g)
    save_field(f"{cfg.outdir}/weighting_field.npz", ctx.tree, ctx.coords, ctx.pw,
               ctx.Ew, description="weighting field of the centre pad",
               centre_pad_xy=np.array([ctx.shift * cfg.h_min,
                                       ctx.shift * cfg.h_min]), **g)
    return dict(files=2)


def stage_export_grids(cfg: Config, ctx: Context):
    """Regular-grid resamplings: coarse over the volume, fine near the anode."""
    g = _geom(cfg, ctx)
    L, LD = cfg.npad * cfg.pitch, cfg.drift_length
    sp = cfg.pitch / cfg.grid_div
    _, shp = save_grid(f"{cfg.outdir}/drift_field_grid.npz", ctx.tree, ctx.pd,
                       ctx.Ed, (0, L), (0, L), (0, LD), sp, **g)
    save_grid(f"{cfg.outdir}/weighting_field_grid.npz", ctx.tree, ctx.pw,
              ctx.Ew, (0, L), (0, L), (0, LD), sp, **g)

    zs = cfg.slab * cfg.pitch
    x, y, z, pgd, Egd = sample_grid(ctx.tree, ctx.pd, ctx.Ed,
                                    (0, L), (0, L), (0, zs), cfg.h_min)
    _, _, _, pgw, Egw = sample_grid(ctx.tree, ctx.pw, ctx.Ew,
                                    (0, L), (0, L), (0, zs), cfg.h_min)
    np.savez_compressed(f"{cfg.outdir}/near_anode_grid.npz", x=x, y=y, z=z,
                        phi_drift=pgd, E_drift=Egd, phi_weight=pgw,
                        E_weight=Egw, spacing=cfg.h_min, units_length="mm",
                        units_field="V/mm",
                        **{k: np.asarray(v) for k, v in g.items()})
    return dict(coarse_shape=list(shp), fine_shape=list(pgd.shape))


def _need_fields(cfg: Config, ctx: Context):
    """Load fields from disk if an earlier stage did not put them in context."""
    if "Ed" in ctx:
        return
    print("    (loading fields from disk)")
    tree, dd = load_field(f"{cfg.outdir}/drift_field.npz")
    _, dw = load_field(f"{cfg.outdir}/weighting_field.npz")
    ctx.update(tree=tree, Ed=dd["E"], Ew=dw["E"], pd=dd["phi"], pw=dw["phi"],
               coords=dd["coords"], loaded_meta=(dd, dw),
               interp=Interpolator(tree))
    if "pad" not in ctx:
        ctx.pad = PixelAnode(tree, pitch=float(dd["pitch"]),
                             pad=float(dd["pad_width"]))
        ctx.shift = int(round(float(dw["centre_pad_xy"][0]) / float(dd["h_min"])))


def stage_drift(cfg: Config, ctx: Context):
    """Track a survey of electrons from random points through the volume."""
    _need_fields(cfg, ctx)
    rng = np.random.default_rng(cfg.seed)
    L, LD = cfg.npad * cfg.pitch, cfg.drift_length
    n = cfg.ngrid
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, n), np.linspace(0.1, 0.9, n),
                         indexing="ij")
    starts = np.stack([(gx.ravel() + rng.uniform(-.04, .04, n * n)) * L,
                       (gy.ravel() + rng.uniform(-.04, .04, n * n)) * L,
                       rng.uniform(0.15, 0.95, n * n) * LD], 1)
    tmax = 5.0 * LD / drift_velocity(cfg.e0 / 100)
    paths, arrived = drift_paths(ctx.tree, ctx.Ed, starts, z_stop=0.0,
                                 max_time=tmax, return_status=True)
    if not arrived.all():
        print(f"    WARNING: {int((~arrived).sum())} of {len(paths)} paths "
              f"stalled (gap-symmetry surface)")
    ctx.paths, ctx.starts, ctx.arrived = paths, starts, arrived
    return dict(npaths=len(paths), arrived=int(arrived.sum()))


def stage_currents(cfg: Config, ctx: Context):
    """Shockley-Ramo current for the survey paths."""
    _need_fields(cfg, ctx)
    cur = [induced_current(ctx.tree, ctx.pw, ctx.Ew, p, interp=ctx.interp)
           for p in ctx.paths]
    qp = np.array([c[3][-1] for c in cur])
    _, shp = save_currents(f"{cfg.outdir}/induced_current.npz", ctx.paths, cur,
                           ctx.starts, dt=cfg.tick, pad_before=cfg.pad_before,
                           pad_after=cfg.pad_after,
                           collected=qp > 0.5 * E_CHARGE,
                           electron_charge_fC=E_CHARGE,
                           v_drift_bulk=drift_velocity(cfg.e0 / 100),
                           **_geom(cfg, ctx))
    ctx.currents = cur
    return dict(shape=list(shp), collected=int((qp > 0.5 * E_CHARGE).sum()))


def stage_response(cfg: Config, ctx: Context):
    """Field-response table on a regular grid of impact positions."""
    _need_fields(cfg, ctx)
    LD = cfg.drift_length
    z0 = cfg.start_z if cfg.start_z is not None else 0.75 * LD
    cx = cy = ctx.shift * cfg.h_min

    n = int(round(cfg.npix * cfg.paths_per_pixel)) + 1
    span = cfg.npix * cfg.pitch
    step = span / (n - 1)
    ax_ = np.linspace(-span / 2, span / 2, n) + cfg.grid_offset * step
    gx, gy = np.meshgrid(cx + ax_, cy + ax_, indexing="ij")
    starts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0)], 1)

    tmax = 5.0 * z0 / drift_velocity(cfg.e0 / 100)
    paths, arrived = drift_paths(ctx.tree, ctx.Ed, starts, z_stop=0.0,
                                 max_time=tmax, return_status=True)
    if not arrived.all():
        print(f"    WARNING: {int((~arrived).sum())} paths stalled")
    cur = [induced_current(ctx.tree, ctx.pw, ctx.Ew, p, interp=ctx.interp)
           for p in paths]

    tdrift = np.array([c[0][-1] - c[0][0] for c in cur])
    nt = int(np.ceil((tdrift[arrived].max() + cfg.pad_after) / cfg.tick)) + 1
    edges = cfg.tick * np.arange(nt)
    Q = np.zeros((len(paths), nt))
    for k, (t, i_r, q_r, q_p) in enumerate(cur):
        if arrived[k]:
            Q[k] = np.interp(edges, t - t[0], q_p, left=0.0, right=q_p[-1])

    resp = np.diff(Q, axis=1) / E_CHARGE
    if cfg.scale == "fC":
        resp *= E_CHARGE
    elif cfg.scale == "pA":
        resp *= E_CHARGE * 1e3 / cfg.tick

    qtot = Q[:, -1] / E_CHARGE
    meta = dict(response=resp, t=0.5 * (edges[1:] + edges[:-1]),
                tick=np.float64(cfg.tick), start=starts,
                end=np.array([p[-1, :3] for p in paths]),
                q_total=qtot, collected=qtot > 0.5, arrived=arrived,
                t_drift=tdrift, impact_shape=np.array([n, n]),
                paths_per_pixel=np.int32(cfg.paths_per_pixel),
                centre_pad_xy=np.array([cx, cy]), start_z=np.float64(z0),
                scale=cfg.scale, electron_charge_fC=np.float64(E_CHARGE),
                v_drift_bulk=np.float64(drift_velocity(cfg.e0 / 100)),
                **_geom(cfg, ctx))
    np.savez_compressed(f"{cfg.outdir}/response.npz", **meta)
    np.save(f"{cfg.outdir}/response.npy", resp)
    print(f"    response {resp.shape} ({cfg.scale}/tick), "
          f"{int((qtot > 0.5).sum())} of {len(qtot)} collected")
    ctx.response = resp
    return dict(shape=list(resp.shape), impact_grid=[n, n],
                collected=int((qtot > 0.5).sum()))


def stage_plots(cfg: Config, ctx: Context):
    """Induced-current summary figure."""
    import subprocess
    import sys
    r = subprocess.run(
        [sys.executable, "plot_induced_current.py",
         "--npz", f"{cfg.outdir}/induced_current.npz",
         "--out", f"{cfg.outdir}/induced_current.png"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()})
    if r.returncode:
        print("    plotting failed:\n" + r.stdout + r.stderr)
    return dict(ok=r.returncode == 0)


STAGES = {
    "grid": stage_grid,
    "solve_drift": stage_solve_drift,
    "solve_weighting": stage_solve_weighting,
    "efields": stage_efields,
    "export_fields": stage_export_fields,
    "export_grids": stage_export_grids,
    "drift": stage_drift,
    "currents": stage_currents,
    "response": stage_response,
    "plots": stage_plots,
}


def run(cfg: Config, stages=None, log_path=None):
    """Run the selected stages in order, writing a stage log."""
    os.makedirs(cfg.outdir, exist_ok=True)
    names = list(STAGES) if stages is None else list(stages)
    unknown = [s for s in names if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}\nknown: {list(STAGES)}")

    log = StageLog(log_path or f"{cfg.outdir}/stages.jsonl",
                   meta=dict(config=asdict(cfg), stages=names,
                             device=torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else "cpu"))
    ctx = Context()
    for name in names:
        with log.stage(name) as extra:
            info = STAGES[name](cfg, ctx)
            if info:
                extra.update(info)
    print(f"\ntotal {log.total():.1f}s over {len(log.records)} stages "
          f"-> {log.path}")
    return ctx, log
