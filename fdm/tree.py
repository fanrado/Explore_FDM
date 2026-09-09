"""Non-graded octree on an exact integer lattice.

All geometry is carried in integer *units*, where one unit is the finest
possible cell size ``h_min``.  A cell at level ``L`` has size ``2**(lmax-L)``
units.  Working in integers means node identity is exact equality -- there is
never a floating-point comparison anywhere in the topology construction, which
is the single largest source of bugs in adaptive-grid codes.

The tree is *non-graded*: no constraint is placed on the level difference
between adjacent cells.  This is required by the Min-Gibou discretization and is
what lets the grid coarsen aggressively away from the anode.
"""

from __future__ import annotations

import numpy as np


class Octree:
    """Leaf cells of a non-graded octree, stored as a flat integer array.

    Parameters
    ----------
    root_dims : (3,) int
        Number of level-0 cells along x, y, z.
    lmax : int
        Deepest permitted refinement level.  Cell size at ``lmax`` is one unit.
    h_min : float
        Physical size (mm) of one unit, i.e. of a level-``lmax`` cell.
    origin : (3,) float
        Physical position (mm) of the lattice origin.
    """

    def __init__(self, root_dims, lmax: int, h_min: float, origin=(0.0, 0.0, 0.0),
                 periodic=(False, False, False)):
        self.root_dims = np.asarray(root_dims, dtype=np.int64)
        if self.root_dims.shape != (3,):
            raise ValueError("root_dims must have shape (3,)")
        self.lmax = int(lmax)
        self.h_min = float(h_min)
        self.origin = np.asarray(origin, dtype=np.float64)
        self.periodic = np.asarray(periodic, dtype=bool)
        if self.periodic.shape != (3,):
            raise ValueError("periodic must have shape (3,)")

        grids = np.meshgrid(*[np.arange(d, dtype=np.int64) for d in self.root_dims],
                            indexing="ij")
        n = grids[0].size
        self.cells = np.stack(
            [np.zeros(n, np.int64)] + [g.ravel() for g in grids], axis=1
        )
        self._invalidate()

    # ------------------------------------------------------------------ basics

    def _invalidate(self):
        """Drop cached per-level lookup tables (call after any topology change)."""
        self._lut = None

    def __len__(self) -> int:
        return self.cells.shape[0]

    @property
    def unit(self) -> int:
        """Units per level-0 cell."""
        return 1 << self.lmax

    @property
    def dims_units(self) -> np.ndarray:
        """Domain extent in units, along each axis."""
        return self.root_dims * self.unit

    @property
    def key_dims(self) -> np.ndarray:
        """Node-lattice extent per axis: a periodic axis identifies its two
        faces, so it has one fewer distinct node plane than a bounded one."""
        return self.dims_units + np.where(self.periodic, 0, 1)

    def fold(self, coords: np.ndarray) -> np.ndarray:
        """Wrap lattice coordinates on periodic axes into the canonical range."""
        if not self.periodic.any():
            return coords
        out = np.asarray(coords).copy()
        for a in range(3):
            if self.periodic[a]:
                out[..., a] = np.mod(out[..., a], self.dims_units[a])
        return out

    def cell_size_units(self, level):
        """Cell edge length in units, for the given level(s)."""
        return np.left_shift(np.int64(1), self.lmax - np.asarray(level, np.int64))

    def level_dims(self, level: int) -> np.ndarray:
        """Number of cells along each axis at ``level``, if fully refined."""
        return self.root_dims << level

    def sizes_units(self) -> np.ndarray:
        """Edge length in units of every leaf cell."""
        return self.cell_size_units(self.cells[:, 0])

    def origins_units(self) -> np.ndarray:
        """Lower corner in units of every leaf cell, shape (ncell, 3)."""
        return self.cells[:, 1:] * self.sizes_units()[:, None]

    def centers(self) -> np.ndarray:
        """Physical cell centres (mm), shape (ncell, 3)."""
        s = self.sizes_units()[:, None]
        return self.origin + (self.origins_units() + 0.5 * s) * self.h_min

    def cell_sizes(self) -> np.ndarray:
        """Physical cell edge lengths (mm), shape (ncell,)."""
        return self.sizes_units() * self.h_min

    # ------------------------------------------------------------- refinement

    def refine(self, mask) -> int:
        """Split every leaf selected by boolean ``mask`` into eight children.

        Cells already at ``lmax`` are silently skipped.  Returns the number of
        cells actually split.
        """
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self),):
            raise ValueError(f"mask must have shape ({len(self)},)")
        mask = mask & (self.cells[:, 0] < self.lmax)
        nsplit = int(mask.sum())
        if nsplit == 0:
            return 0

        parents = self.cells[mask]
        offs = np.array(
            [[di, dj, dk] for di in (0, 1) for dj in (0, 1) for dk in (0, 1)],
            dtype=np.int64,
        )
        kids = np.empty((nsplit * 8, 4), dtype=np.int64)
        kids[:, 0] = np.repeat(parents[:, 0] + 1, 8)
        kids[:, 1:] = (np.repeat(parents[:, 1:] * 2, 8, axis=0)
                       + np.tile(offs, (nsplit, 1)))

        self.cells = np.concatenate([self.cells[~mask], kids], axis=0)
        self._sort_cells()
        self._invalidate()
        return nsplit

    def build(self, predicate, max_passes: int | None = None) -> "Octree":
        """Refine repeatedly until ``predicate`` is satisfied everywhere.

        ``predicate(centers, sizes) -> bool array`` is given physical cell
        centres (mm, shape ``(n, 3)``) and physical edge lengths (mm, shape
        ``(n,)``) and returns True for cells that should be split further.
        """
        passes = self.lmax if max_passes is None else max_passes
        for _ in range(passes):
            if self.refine(predicate(self.centers(), self.cell_sizes())) == 0:
                break
        return self

    def _sort_cells(self):
        order = np.lexsort((self.cells[:, 3], self.cells[:, 2],
                            self.cells[:, 1], self.cells[:, 0]))
        self.cells = self.cells[order]

    # ------------------------------------------------------------- cell lookup

    def _keys(self, level: int, ijk: np.ndarray) -> np.ndarray:
        """Pack cell indices at ``level`` into a single sortable integer."""
        ny, nz = self.level_dims(level)[1:]
        return (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]

    def _lookup_tables(self):
        """Per-level sorted key arrays, built lazily and cached."""
        if self._lut is None:
            lut = {}
            lv = self.cells[:, 0]
            for level in range(self.lmax + 1):
                sel = np.flatnonzero(lv == level)
                if sel.size == 0:
                    continue
                keys = self._keys(level, self.cells[sel, 1:])
                order = np.argsort(keys, kind="stable")
                lut[level] = (keys[order], sel[order])
            self._lut = lut
        return self._lut

    def locate(self, points_units: np.ndarray, toward: np.ndarray):
        """Find the leaf cell adjacent to each lattice point in a given octant.

        ``points_units`` are integer lattice coordinates (n, 3).  ``toward`` is
        (n, 3) or (3,) with entries in {-1, +1} selecting which of the eight
        octants around the point to probe.  Because the point sits exactly on
        the lattice, the probe is done in pure integer arithmetic: probing
        towards +x means the cell containing ``p + eps``, i.e. index ``p // c``;
        towards -x it is ``(p - 1) // c``.

        Returns ``(cell_index, valid)``.  ``valid`` is False where the probe
        falls outside the domain.  Leaves partition the domain, so exactly one
        level matches for every valid probe.
        """
        p = np.asarray(points_units, dtype=np.int64)
        t = np.broadcast_to(np.asarray(toward, dtype=np.int64), p.shape)

        # Shift by one unit on the negative side so integer floor-division lands
        # in the correct cell; +1 side needs no shift.
        q = np.where(t > 0, p, p - 1)
        # A periodic axis wraps instead of leaving the domain.
        if self.periodic.any():
            q = np.where(self.periodic, np.mod(q, self.dims_units), q)

        inside = np.all((q >= 0) & (q < self.dims_units) | self.periodic, axis=1)
        out = np.full(p.shape[0], -1, dtype=np.int64)
        todo = np.flatnonzero(inside)

        lut = self._lookup_tables()
        # Search finest level first: the first hit is the unique containing leaf.
        for level in range(self.lmax, -1, -1):
            if todo.size == 0:
                break
            if level not in lut:
                continue
            keys, cells = lut[level]
            c = int(self.cell_size_units(level))
            ijk = q[todo] // c
            k = self._keys(level, ijk)
            pos = np.searchsorted(keys, k)
            np.clip(pos, 0, keys.size - 1, out=pos)
            hit = keys[pos] == k
            if hit.any():
                found = todo[hit]
                out[found] = cells[pos[hit]]
                todo = todo[~hit]
        return out, out >= 0

    def locate_float(self, q):
        """Locate the leaf cell containing arbitrary (non-lattice) points.

        ``q`` is in float lattice units.  Periodic axes are wrapped; points
        outside a bounded axis are clamped to the domain, so a tracker that
        steps marginally past a wall still gets a usable cell.
        """
        q = np.array(q, dtype=np.float64, copy=True)
        for a in range(3):
            D = float(self.dims_units[a])
            if self.periodic[a]:
                q[:, a] = np.mod(q[:, a], D)
            else:
                q[:, a] = np.clip(q[:, a], 0.0, np.nextafter(D, 0.0))

        out = np.full(q.shape[0], -1, dtype=np.int64)
        todo = np.arange(q.shape[0])
        lut = self._lookup_tables()
        for level in range(self.lmax, -1, -1):
            if todo.size == 0:
                break
            if level not in lut:
                continue
            keys, cells = lut[level]
            c = float(self.cell_size_units(level))
            ijk = np.floor(q[todo] / c).astype(np.int64)
            k = self._keys(level, ijk)
            pos = np.searchsorted(keys, k)
            np.clip(pos, 0, keys.size - 1, out=pos)
            hit = keys[pos] == k
            if hit.any():
                out[todo[hit]] = cells[pos[hit]]
                todo = todo[~hit]
        return out, q

    # ------------------------------------------------------------------- nodes

    def nodes(self):
        """Unique mesh nodes: the corners of every leaf cell.

        Returns ``(coords, corner_ids)`` where ``coords`` is (nnode, 3) integer
        lattice coordinates sorted by packed key, and ``corner_ids`` is
        (ncell, 8) giving the node id of each cell corner in the octant order
        ``(0,0,0), (0,0,1), (0,1,0), ... (1,1,1)``.
        """
        s = self.sizes_units()[:, None]
        org = self.origins_units()
        offs = np.array(
            [[di, dj, dk] for di in (0, 1) for dj in (0, 1) for dk in (0, 1)],
            dtype=np.int64,
        )
        # (ncell, 8, 3)
        corners = org[:, None, :] + offs[None, :, :] * s[:, None, :]
        flat = self.fold(corners.reshape(-1, 3))

        ny, nz = self.key_dims[1], self.key_dims[2]
        keys = (flat[:, 0] * ny + flat[:, 1]) * nz + flat[:, 2]

        uniq, inv = np.unique(keys, return_inverse=True)
        coords = np.empty((uniq.size, 3), dtype=np.int64)
        coords[:, 2] = uniq % nz
        rest = uniq // nz
        coords[:, 1] = rest % ny
        coords[:, 0] = rest // ny
        return coords, inv.reshape(-1, 8).astype(np.int64)

    def node_positions(self, coords: np.ndarray) -> np.ndarray:
        """Physical positions (mm) of integer lattice coordinates."""
        return self.origin + np.asarray(coords, np.float64) * self.h_min
