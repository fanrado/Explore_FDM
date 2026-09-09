"""Electron drift and Shockley-Ramo induced current.

Three pieces, none of which the field solver can be trusted without:

* the liquid-argon drift-velocity parameterization,
* trilinear interpolation of a node-sampled field on the octree,
* an RK4 tracker plus the Ramo current on each pad.

Sign convention: the anode is at z = 0 held at 0 V, the cathode at z = L held
at a negative potential, so E points towards +z and electrons (q = -e) drift
along -E, towards the anode.
"""

from __future__ import annotations

import numpy as np

from .tree import Octree

# Elementary charge (fC), so currents come out in fC/us == nA.
E_CHARGE = 1.602176634e-4


def drift_velocity(E_kVcm, temperature=87.3):
    """Electron drift speed in LAr (mm/us) for |E| in kV/cm.

    ICARUS parameterization as used by LArSoft; see W. Walkowiak,
    NIM A 449 (2000) 288 and the ICARUS fit therein.  Gives 1.596 mm/us at
    500 V/cm and 87.3 K, matching the standard value.
    """
    P1, P2, P3 = -0.04640, 0.01712, 1.88125
    P4, P5, P6 = 0.99408, 0.01172, 4.20214
    T0 = 105.749
    E = np.maximum(np.asarray(E_kVcm, dtype=np.float64), 1e-9)
    dT = temperature - T0
    v = (P1 * dT + 1.0) * (P3 * E * np.log1p(P4 / E) + P5 * E**P6) + P2 * dT
    return np.maximum(v, 0.0)


class Interpolator:
    """Trilinear interpolation of node-sampled fields inside each leaf cell.

    Note the field is only C0 within a level: across a coarse/fine face the
    coarse cell's trilinear form and the fine cells' do not agree exactly,
    because the hanging-node values are solved degrees of freedom rather than
    interpolants of the coarse face.  The mismatch is O(h^2) and shows up as a
    small kink in a drift path crossing a refinement boundary.
    """

    def __init__(self, tree: Octree):
        self.tree = tree
        _, self.corner_ids = tree.nodes()
        self.org = tree.origins_units().astype(np.float64)
        self.size = tree.sizes_units().astype(np.float64)

    def weights(self, points_mm):
        """Corner ids and trilinear weights for each query point."""
        q = (np.asarray(points_mm, dtype=np.float64) - self.tree.origin) / self.tree.h_min
        cell, qw = self.tree.locate_float(q)
        f = (qw - self.org[cell]) / self.size[cell][:, None]
        np.clip(f, 0.0, 1.0, out=f)
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        w = np.empty((len(cell), 8))
        for m, (di, dj, dk) in enumerate(
                [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]):
            w[:, m] = ((fx if di else 1 - fx)
                       * (fy if dj else 1 - fy)
                       * (fz if dk else 1 - fz))
        return self.corner_ids[cell], w

    def __call__(self, values, points_mm):
        """Interpolate a scalar (n,) or vector (n, 3) node field."""
        ids, w = self.weights(points_mm)
        v = np.asarray(values)
        if v.ndim == 1:
            return (v[ids] * w).sum(1)
        return (v[ids] * w[:, :, None]).sum(1)


def fill_missing(tree: Octree, values, have, passes=6):
    """Fill nodes with no value by averaging covered corners of adjacent cells.

    Gradient rows exist only at nodes with a full stencil, so Dirichlet nodes
    (pad surfaces, cathode) carry no field value.  Those nodes are exactly where
    a drift path terminates, so they cannot simply be left at zero.  Repeated
    cell-averaging fills them from their neighbours; a couple of passes is
    enough because uncovered nodes form a surface.
    """
    _, cid = tree.nodes()
    v = np.array(values, dtype=np.float64, copy=True)
    have = np.array(have, dtype=bool, copy=True)
    v[~have] = 0.0
    for _ in range(passes):
        if have.all():
            break
        acc = np.zeros_like(v)
        cnt = np.zeros(v.shape[0])
        src = have[cid]                       # (ncell, 8)
        for m in range(8):
            tgt = cid[:, m]
            for s in range(8):
                if s == m:
                    continue
                ok = src[:, s]
                np.add.at(cnt, tgt[ok], 1.0)
                np.add.at(acc, tgt[ok], v[cid[ok, s]])
        fill = (~have) & (cnt > 0)
        v[fill] = acc[fill] / cnt[fill][:, None] if v.ndim == 2 else acc[fill] / cnt[fill]
        have |= fill
    return v


def drift_paths(tree, Efield, starts_mm, z_stop=0.0, dt_frac=0.25,
                max_steps=200000, temperature=87.3, max_time=None,
                return_status=False):
    """RK4-track electrons from ``starts_mm`` down to ``z_stop``.

    The step is limited to ``dt_frac`` of the local cell crossing time, so the
    tracker automatically takes small steps where the grid is fine -- i.e. near
    the pads, where the trajectory actually bends.

    ``max_time`` (us) abandons a path that has not arrived.  This is not
    belt-and-braces: with a perfectly insulating (Neumann) inter-pad gap, field
    lines lying exactly on the gap symmetry surface never terminate on a pad, so
    an electron launched exactly above a gap centre-line stalls near the anode
    with v -> 0.  Such starting points are a measure-zero pathology of the ideal
    BC (real gaps are resolved by diffusion), but they must be detected rather
    than run to ``max_steps``.

    Returns a list of (npt, 4) arrays holding ``[x, y, z, t]`` (mm, us), and
    with ``return_status`` also a bool array flagging which paths arrived.
    """
    interp = Interpolator(tree)
    hmin = tree.h_min

    def vel(p):
        """Electron velocity (mm/us): speed from |E|, direction along -E."""
        E = interp(Efield, p)                       # V/mm
        mag = np.linalg.norm(E, axis=1)
        speed = drift_velocity(mag * 10.0 / 1000.0, temperature)   # V/mm -> kV/cm
        d = -E / np.maximum(mag, 1e-30)[:, None]
        return d * speed[:, None], mag

    p = np.array(starts_mm, dtype=np.float64)
    n = len(p)
    t = np.zeros(n)
    alive = np.ones(n, bool)
    arrived = np.zeros(n, bool)
    paths = [[np.r_[p[i], 0.0]] for i in range(n)]

    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        k1, mag = vel(p[idx])
        speed = np.linalg.norm(k1, axis=1)
        cell, _ = tree.locate_float((p[idx] - tree.origin) / hmin)
        h = dt_frac * tree.sizes_units()[cell] * hmin / np.maximum(speed, 1e-12)
        h = np.minimum(h, 50.0)

        k2, _ = vel(p[idx] + 0.5 * h[:, None] * k1)
        k3, _ = vel(p[idx] + 0.5 * h[:, None] * k2)
        k4, _ = vel(p[idx] + h[:, None] * k3)
        step = (h / 6.0)[:, None] * (k1 + 2 * k2 + 2 * k3 + k4)

        # Do not overshoot the anode: rescale the final step to land on it.
        newz = p[idx, 2] + step[:, 2]
        over = newz < z_stop
        if over.any():
            frac = (p[idx[over], 2] - z_stop) / np.maximum(
                p[idx[over], 2] - newz[over], 1e-30)
            step[over] *= frac[:, None]
            h[over] *= frac

        p[idx] += step
        t[idx] += h
        for m, i in enumerate(idx):
            paths[i].append(np.r_[p[i], t[i]])
        done = p[idx, 2] <= z_stop + 1e-12
        arrived[idx[done]] = True
        alive[idx[done]] = False
        if max_time is not None:
            alive[idx[t[idx] > max_time]] = False

    out = [np.array(pp) for pp in paths]
    return (out, arrived) if return_status else out


def induced_current(tree, wpot, wfield, path, interp=None):
    """Shockley-Ramo current and induced charge on one pad for one path.

    Convention: the charge induced on electrode k by a carrier of charge ``q``
    is ``Q_k = -q [phi_w(x) - phi_w(x0)]``, and the current is its time
    derivative, ``i_k = dQ/dt = q v . E_w`` with ``E_w = -grad(phi_w)``.  For an
    electron ``q = -e``, so ``Q = +e [phi_w - phi_w0]`` but ``i = -e v . E_w``.
    Getting these two signs consistent is the point of returning both: the Ramo
    integral and the potential difference are computed by completely different
    routes (one differentiates the weighting field, one does not), so their
    agreement checks the gradient operator and the tracker together.
    """
    # Building an Interpolator walks every cell corner, so reuse one across
    # paths rather than rebuilding it per call.
    if interp is None:
        interp = Interpolator(tree)
    xyz, t = path[:, :3], path[:, 3]
    Ew = interp(wfield, xyz)
    v = np.zeros_like(xyz)
    if len(t) > 2:
        v[1:-1] = (xyz[2:] - xyz[:-2]) / np.maximum(t[2:] - t[:-2], 1e-30)[:, None]
        v[0] = (xyz[1] - xyz[0]) / max(t[1] - t[0], 1e-30)
        v[-1] = (xyz[-1] - xyz[-2]) / max(t[-1] - t[-2], 1e-30)
    i_ramo = -E_CHARGE * (v * Ew).sum(1)
    phi = interp(wpot, xyz)
    q_pot = E_CHARGE * (phi - phi[0])
    q_ramo = np.concatenate([[0.0], np.cumsum(0.5 * (i_ramo[1:] + i_ramo[:-1])
                                              * np.diff(t))])
    return t, i_ramo, q_ramo, q_pot
