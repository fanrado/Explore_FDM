"""Save and reload fields and induced currents as .npz.

Two representations are written for every field:

* **native** -- values on the octree nodes, plus enough of the tree to rebuild
  the interpolator.  Exact, compact, and the only form that preserves the
  adaptive resolution.
* **grid** -- resampled onto a regular Cartesian lattice.  Lossy wherever the
  grid is coarser than the octree, but directly indexable as a numpy array,
  which is usually what downstream code wants.

Units throughout: mm, V, V/mm for fields; us, fC, nA for signals.
"""

from __future__ import annotations

import numpy as np

from .drift import Interpolator
from .tree import Octree

UNITS = dict(length="mm", potential="V", field="V/mm",
             time="us", charge="fC", current="nA")


def tree_arrays(tree: Octree) -> dict:
    """Everything needed to reconstruct the octree and its interpolator."""
    _, corner_ids = tree.nodes()
    return dict(
        tree_cells=tree.cells.astype(np.int32),
        tree_corner_ids=corner_ids.astype(np.int32),
        tree_root_dims=tree.root_dims.astype(np.int32),
        tree_lmax=np.int32(tree.lmax),
        tree_h_min=np.float64(tree.h_min),
        tree_origin=tree.origin.astype(np.float64),
        tree_periodic=tree.periodic,
    )


def load_tree(z) -> Octree:
    """Rebuild an Octree from arrays written by :func:`tree_arrays`."""
    t = Octree(z["tree_root_dims"], int(z["tree_lmax"]), float(z["tree_h_min"]),
               origin=z["tree_origin"], periodic=z["tree_periodic"])
    t.cells = z["tree_cells"].astype(np.int64)
    t._invalidate()
    return t


def save_field(path, tree: Octree, coords, phi, E, **meta):
    """Write a field on the octree nodes, with the tree needed to reuse it.

    ``phi`` is (nnode,) in V and ``E`` is (nnode, 3) in V/mm.
    """
    d = tree_arrays(tree)
    d.update(
        coords=np.asarray(coords, np.int32),          # integer lattice units
        pos=tree.node_positions(coords).astype(np.float64),   # mm
        phi=np.asarray(phi, np.float64),
        E=np.asarray(E, np.float64),
    )
    for k, v in meta.items():
        d[k] = np.asarray(v)
    for k, v in UNITS.items():
        d[f"units_{k}"] = v
    np.savez_compressed(path, **d)
    return path


def load_field(path):
    """Return ``(tree, dict_of_arrays)`` for a file written by save_field."""
    z = np.load(path, allow_pickle=False)
    return load_tree(z), {k: z[k] for k in z.files}


def field_interpolator(path):
    """Convenience: reload a saved field and return a ready callable.

    ``f(points_mm)`` gives phi, ``fE(points_mm)`` gives E.
    """
    tree, d = load_field(path)
    interp = Interpolator(tree)
    return tree, d, (lambda p: interp(d["phi"], p)), (lambda p: interp(d["E"], p))


def sample_grid(tree: Octree, phi, E, xlim, ylim, zlim, spacing):
    """Resample a node field onto a regular Cartesian grid.

    ``*lim`` are ``(lo, hi)`` in mm and ``spacing`` is the step in mm.  Returns
    ``(x, y, z, phi_grid, E_grid)`` with shapes (nx,), (ny,), (nz,),
    (nx,ny,nz) and (nx,ny,nz,3).
    """
    def axis(lim):
        n = int(round((lim[1] - lim[0]) / spacing)) + 1
        return lim[0] + spacing * np.arange(n)

    x, y, z = axis(xlim), axis(ylim), axis(zlim)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)

    interp = Interpolator(tree)
    # Chunked so a large grid does not blow up the point-location temporaries.
    chunk = 4_000_000
    pg = np.empty(len(pts))
    Eg = np.empty((len(pts), 3))
    for i in range(0, len(pts), chunk):
        s = slice(i, min(i + chunk, len(pts)))
        pg[s] = interp(phi, pts[s])
        Eg[s] = interp(E, pts[s])
    return x, y, z, pg.reshape(X.shape), Eg.reshape(X.shape + (3,))


def save_grid(path, tree, phi, E, xlim, ylim, zlim, spacing, **meta):
    """Resample onto a regular grid and write it."""
    x, y, z, pg, Eg = sample_grid(tree, phi, E, xlim, ylim, zlim, spacing)
    d = dict(x=x, y=y, z=z, phi=pg, E=Eg, spacing=np.float64(spacing))
    for k, v in meta.items():
        d[k] = np.asarray(v)
    for k, v in UNITS.items():
        d[f"units_{k}"] = v
    np.savez_compressed(path, **d)
    return path, pg.shape


def save_currents(path, paths, currents, starts, dt=0.05,
                  pad_before=None, pad_after=2.0, **meta):
    """Write induced-current waveforms on common uniform time bases.

    ``currents`` is a list of ``(t, i, q_ramo, q_phi)`` as returned by
    :func:`fdm.drift.induced_current`.

    The time base is deliberately WIDER than the longest drift, with
    ``pad_before`` us of baseline ahead of the earliest emission and
    ``pad_after`` us of tail past the latest arrival.  Without that margin the
    pulse is clipped at the array edges and its shape cannot be read (or
    filtered, or fitted) near the ends.  ``pad_before`` defaults to 20% of the
    longest drift.

    Outside a path's own drift the waveform is not extrapolated: before
    emission and after collection the electron is not moving, so the current is
    exactly zero.  The induced *charge* instead holds its final value after
    collection, which is what makes the step visible.

    Two bases are stored:

    * ``t``, ``i``, ``q``           -- aligned so t = 0 is each electron's
      arrival.  Use this to compare pulse *shapes*.
    * ``t_abs``, ``i_abs``, ``q_abs`` -- absolute time from a common t = 0 at
      emission, so each electron sits at its own drift time.  This is what a
      readout actually sees.

    Trajectories are stored NaN-padded to a rectangular array so the file needs
    no pickling.
    """
    n = len(paths)
    tdrift = np.array([c[0][-1] - c[0][0] for c in currents])
    tmax = float(tdrift.max())
    if pad_before is None:
        pad_before = 0.2 * tmax

    # Arrival-aligned base; 0 is exactly on a sample.
    nb = int(np.ceil((tmax + pad_before) / dt))
    na = int(np.ceil(pad_after / dt))
    tgrid = dt * np.arange(-nb, na + 1)
    # Absolute base, emission at 0.
    tabs = dt * np.arange(0, int(np.ceil((tmax + pad_after) / dt)) + 1)

    I = np.zeros((n, tgrid.size));  Q = np.zeros((n, tgrid.size))
    Ia = np.zeros((n, tabs.size));  Qa = np.zeros((n, tabs.size))
    for k, (t, i_r, q_r, q_p) in enumerate(currents):
        # right=0: after collection the carrier has stopped, so no current.
        tt = t - t[-1]
        I[k] = np.interp(tgrid, tt, i_r, left=0.0, right=0.0)
        Q[k] = np.interp(tgrid, tt, q_p, left=0.0, right=q_p[-1])
        ta = t - t[0]
        Ia[k] = np.interp(tabs, ta, i_r, left=0.0, right=0.0)
        Qa[k] = np.interp(tabs, ta, q_p, left=0.0, right=q_p[-1])

    lens = np.array([len(p) for p in paths])
    traj = np.full((n, lens.max(), 4), np.nan)
    for k, p in enumerate(paths):
        traj[k, :len(p)] = p

    d = dict(
        t=tgrid,                                  # us, 0 == arrival
        i=I * 1e3,                                # fC/us -> pA
        q=Q,                                      # fC, cumulative induced
        t_abs=tabs,                               # us, 0 == emission
        i_abs=Ia * 1e3,
        q_abs=Qa,
        q_total=Q[:, -1],
        t_drift=tdrift,
        pad_before=np.float64(pad_before),
        pad_after=np.float64(pad_after),
        t_arrival=np.array([c[0][-1] for c in currents]),
        start=np.asarray(starts, np.float64),
        end=np.array([p[-1, :3] for p in paths]),
        traj=traj, traj_len=lens,
        dt=np.float64(dt),
    )
    for k, v in meta.items():
        d[k] = np.asarray(v)
    for k, v in UNITS.items():
        d[f"units_{k}"] = v
    d["units_current"] = "pA"
    np.savez_compressed(path, **d)
    return path, I.shape
