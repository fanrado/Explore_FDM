"""Pixel-anode geometry as integer lattice masks -- no meshing anywhere.

The pad array is axis-aligned and rectangular, so if the pitch and pad width are
integer multiples of ``h_min`` every pad boundary lands exactly on a grid line.
The geometry is then a boolean mask built by integer comparison, with no cut
cells, no stair-stepping and no geometric tolerance.  This is the property that
makes finite differences a better fit than finite elements for this problem.
"""

from __future__ import annotations

import numpy as np

from .tree import Octree


class PixelAnode:
    """A periodic array of square pads on the z = 0 plane.

    All lengths are in mm and are required to be integer multiples of the
    tree's ``h_min``; violations raise rather than silently snapping, because a
    pad edge that misses the lattice reintroduces exactly the geometric error
    this scheme is designed to avoid.
    """

    def __init__(self, tree: Octree, pitch: float, pad: float, z_anode: float = 0.0):
        self.tree = tree
        self.pitch_u = self._units(tree, pitch, "pitch")
        self.pad_u = self._units(tree, pad, "pad width")
        self.z_u = self._units(tree, z_anode - tree.origin[2], "anode z", signed=True)
        if self.pad_u >= self.pitch_u:
            raise ValueError("pad width must be smaller than the pitch")
        self.pitch = pitch
        self.pad = pad

    @staticmethod
    def _units(tree: Octree, length: float, what: str, signed: bool = False) -> int:
        u = length / tree.h_min
        if abs(u - round(u)) > 1e-9:
            raise ValueError(
                f"{what} = {length} mm is not an integer multiple of h_min = "
                f"{tree.h_min} mm (would be {u:.6f} units)"
            )
        return int(round(u))

    @property
    def gap(self) -> float:
        """Inter-pad gap width (mm) -- the length scale that sets h_min."""
        return (self.pitch_u - self.pad_u) * self.tree.h_min

    def on_anode(self, coords: np.ndarray) -> np.ndarray:
        """True for nodes lying on the anode plane."""
        return coords[:, 2] == self.z_u

    def on_pad(self, coords: np.ndarray, centre_only: bool = False) -> np.ndarray:
        """True for anode-plane nodes covered by a pad.

        Pads are centred on each pitch cell.  With ``centre_only`` just the pad
        containing the origin is selected, which is what the weighting-field
        calculation needs.
        """
        half = (self.pitch_u - self.pad_u) // 2
        r = np.mod(coords[:, :2], self.pitch_u)
        inside = np.all((r >= half) & (r <= half + self.pad_u), axis=1)
        if centre_only:
            cell = coords[:, :2] // self.pitch_u
            inside &= np.all(cell == 0, axis=1)
        return self.on_anode(coords) & inside

    def refine_predicate(self, grade: float = 8.0, edge_levels: int = 2):
        """Refinement rule: fine at the pads, coarsening into the bulk.

        Cell size is capped at ``dist_to_anode / grade`` so resolution follows
        the exp(-2 pi z / pitch) decay of the pad-induced distortion, and forced
        to ``h_min`` within ``edge_levels`` cells of a pad edge, where the field
        is singular.
        """
        t = self.tree
        z0 = self.z_u * t.h_min + t.origin[2]
        half = (self.pitch_u - self.pad_u) // 2
        edge_lo = half * t.h_min
        edge_hi = (half + self.pad_u) * t.h_min

        def predicate(centers, sizes):
            dz = np.abs(centers[:, 2] - z0)
            target = np.maximum(dz / grade, t.h_min)
            # Near a pad edge in x or y, and close to the plane, go to h_min.
            r = np.mod(centers[:, :2] - t.origin[:2], self.pitch)
            near_edge = np.any(
                (np.abs(r - edge_lo) < edge_levels * sizes[:, None])
                | (np.abs(r - edge_hi) < edge_levels * sizes[:, None]), axis=1)
            target = np.where(near_edge & (dz < edge_levels * self.pitch),
                              t.h_min, target)
            return sizes > 1.001 * target

        return predicate
