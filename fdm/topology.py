"""Build the composite Laplacian rows for every node of a non-graded octree.

This implements the Min-Gibou node-based scheme (JCP 218 (2006) 300-321) in a
generalized form.  For each node and each of the six axis directions we take the
*finest leaf cell adjacent to the node on that side*, call it C, and place the
neighbour on C's far face at the point aligned with the node:

  * if the node is a corner of C, that point is a real mesh node (weight 1);
  * otherwise the node is hanging on a face or edge of C, and the point is
    obtained by bilinear interpolation from C's four far-face corners.

Bilinear interpolation of a smooth field at the aligned point carries a known
error.  Interpolating between neighbours at distances ``a`` and ``b`` either
side gives ``u_interp = u_exact + (a*b/2) * u''``, so each interpolated
neighbour injects spurious transverse second-derivative terms:

    D_a u = u_aa + C_ab u_bb + C_ac u_cc + O(h)

Min & Gibou cancel these with two scalar weights alpha, beta, derived for the
specific configurations that arise in their figures.  We instead solve the
general 3x3 system

    M^T w = (1, 1, 1),    M[a][a] = 1,  M[a][b] = C_ab

for the per-node weights ``w``, so that ``sum_a w_a D_a = laplacian + O(h)``.
This reduces to their alpha/beta in their configurations, but is uniform in
code, needs no case analysis, and covers configurations where more than one
direction is interpolated.  ``M`` is diagonally dominant on isotropic cells, so
the solve is well conditioned.

Rows are emitted in two groups, because on a graded grid the overwhelming
majority of nodes have all six neighbours landing on real nodes:

  * ``regular``   -- 7 entries per row (diagonal + 6 neighbours)
  * ``irregular`` -- up to 25 entries per row (diagonal + 6 x 4 interpolants)

which keeps the memory traffic of a matvec close to the ideal 7-point cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import Octree

# Axis, sign for the six directions, in the order -x +x -y +y -z +z.
DIRS = [(0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)]


@dataclass
class Rows:
    """Composite Laplacian in two padded gather-reduce blocks.

    ``A @ u`` is  ``zeros(n).index_add(reg_row, (reg_val * u[reg_idx]).sum(1))``
    plus the same for the irregular block.  Both blocks are dense in their
    second dimension, so the matvec is a pure gather-multiply-reduce with no
    sparse-format overhead.
    """

    nnode: int
    reg_row: np.ndarray   # (nr,)      node id owning each regular row
    reg_idx: np.ndarray   # (nr, 7)    column node ids
    reg_val: np.ndarray   # (nr, 7)    coefficients
    irr_row: np.ndarray   # (ni,)
    irr_idx: np.ndarray   # (ni, 25)
    irr_val: np.ndarray   # (ni, 25)
    boundary: np.ndarray  # (nnode,) bool -- on the domain boundary
    coords: np.ndarray    # (nnode, 3) integer lattice coordinates

    @property
    def nnz(self) -> int:
        return self.reg_idx.size + self.irr_idx.size


def _node_ids(keys_sorted, coords, ny, nz, tree=None) -> np.ndarray:
    """Map integer lattice coordinates to node ids via the sorted key table."""
    if tree is not None:
        coords = tree.fold(coords)
    k = (coords[..., 0] * ny + coords[..., 1]) * nz + coords[..., 2]
    pos = np.searchsorted(keys_sorted, k.ravel())
    np.clip(pos, 0, keys_sorted.size - 1, out=pos)
    if not np.all(keys_sorted[pos] == k.ravel()):
        raise AssertionError("interpolation stencil referenced a non-existent node")
    return pos.reshape(k.shape)


def neighbours(tree: Octree, coords: np.ndarray, keys_sorted: np.ndarray):
    """Neighbour descriptors for every node, for all six directions.

    Returns ``(nid, wgt, s, valid)`` with shapes ``(6, n, 4)``, ``(6, n, 4)``,
    ``(6, n)`` and ``(6, n)``.  ``s`` is the physical distance (mm) to the
    neighbour; ``valid`` is False where the direction leaves the domain.
    """
    n = coords.shape[0]
    ny, nz = int(tree.key_dims[1]), int(tree.key_dims[2])

    nid = np.zeros((6, n, 4), dtype=np.int64)
    wgt = np.zeros((6, n, 4), dtype=np.float64)
    sdist = np.zeros((6, n), dtype=np.float64)
    valid = np.zeros((6, n), dtype=bool)
    # Transverse half-widths of the interpolation, per direction and per
    # transverse axis: the product a*b entering the interpolation error.
    prod = np.zeros((6, n, 3), dtype=np.float64)

    for d, (axis, sign) in enumerate(DIRS):
        b, c = [a for a in (0, 1, 2) if a != axis]

        # Probe the four octants on this side of the node and keep the finest
        # leaf found: that is the cell the stencil leaves through.
        best_level = np.full(n, -1, dtype=np.int64)
        best_cell = np.full(n, -1, dtype=np.int64)
        for sb in (-1, 1):
            for sc in (-1, 1):
                toward = np.empty((n, 3), dtype=np.int64)
                toward[:, axis] = sign
                toward[:, b] = sb
                toward[:, c] = sc
                cell, ok = tree.locate(coords, toward)
                lv = np.where(ok, tree.cells[np.maximum(cell, 0), 0], -1)
                take = ok & (lv > best_level)
                best_level[take] = lv[take]
                best_cell[take] = cell[take]

        have = best_cell >= 0
        valid[d] = have
        if not have.any():
            continue

        sel = np.flatnonzero(have)
        cell = best_cell[sel]
        csz = tree.cell_size_units(tree.cells[cell, 0])
        org = tree.cells[cell, 1:] * csz[:, None]
        p = coords[sel]

        # The neighbour cell may have been located through a periodic wrap, in
        # which case its origin is on the far side of the domain.  Shift it into
        # the unwrapped frame nearest the node so that distances and
        # interpolation offsets are measured locally; the resulting corner
        # coordinates are folded back when they are looked up.
        for a in range(3):
            if tree.periodic[a]:
                D = int(tree.dims_units[a])
                shift = np.round((p[:, a] - (org[:, a] + 0.5 * csz)) / D)
                org[:, a] = org[:, a] + (D * shift).astype(np.int64)

        # Distance to the far face along the stencil axis.  A whole-period cell
        # can put the node exactly on the face it is leaving through; on a
        # periodic axis that means the neighbour is one full period away.
        if sign > 0:
            step = org[:, axis] + csz - p[:, axis]
        else:
            step = p[:, axis] - org[:, axis]
        if tree.periodic[axis]:
            step = np.where(step <= 0, step + int(tree.dims_units[axis]), step)
        face = p[:, axis] + sign * step
        sdist[d, sel] = step * tree.h_min

        # Bilinear weights on the far face from the node's transverse offsets.
        fb = (p[:, b] - org[:, b]) / csz.astype(np.float64)
        fc = (p[:, c] - org[:, c]) / csz.astype(np.float64)
        w = np.stack([(1 - fb) * (1 - fc), (1 - fb) * fc,
                      fb * (1 - fc), fb * fc], axis=1)

        corners = np.empty((sel.size, 4, 3), dtype=np.int64)
        for m, (ob, oc) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            corners[:, m, axis] = face
            corners[:, m, b] = org[:, b] + ob * csz
            corners[:, m, c] = org[:, c] + oc * csz
        nid[d, sel] = _node_ids(keys_sorted, corners, ny, nz, tree)
        wgt[d, sel] = w

        # Interpolation error products a*b in each transverse direction.
        hb = csz.astype(np.float64) * tree.h_min
        prod[d, sel, b] = (fb * (1 - fb)) * hb * hb
        prod[d, sel, c] = (fc * (1 - fc)) * hb * hb

    return nid, wgt, sdist, valid, prod


def build_rows(tree: Octree, interface_correction: bool = True,
               neumann=None) -> Rows:
    """Assemble the composite Laplacian rows for every node of ``tree``.

    With ``interface_correction=False`` the transverse-error weights are forced
    to 1, giving the naive scheme that simply sums the three directional
    differences.  That variant is locally inconsistent at hanging nodes (the
    Losasso-style first-order treatment); it exists so the cost of the
    correction can be measured rather than assumed.
    """
    coords, _ = tree.nodes()
    n = coords.shape[0]
    ny, nz = int(tree.key_dims[1]), int(tree.key_dims[2])
    keys_sorted = (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]

    nid, wgt, sdist, valid, prod = neighbours(tree, coords, keys_sorted)

    # Homogeneous Neumann (dphi/dn = 0) by mirroring: where a direction leaves
    # the domain, reuse the opposite direction's neighbour and distance.  The
    # axis difference then collapses to 2*(u_nb - u0)/s^2, which is exactly the
    # reflected stencil.  This is what models a bare dielectric substrate in the
    # inter-pad gap, and without it the anode plane is an equipotential and the
    # pixel structure does not act on the drift field at all.
    if neumann is not None:
        nm = np.asarray(neumann, dtype=bool)
        for a in range(3):
            dm, dp = 2 * a, 2 * a + 1
            for dst, src in ((dm, dp), (dp, dm)):
                take = nm & ~valid[dst] & valid[src]
                if not take.any():
                    continue
                nid[dst, take] = nid[src, take]
                wgt[dst, take] = wgt[src, take]
                sdist[dst, take] = sdist[src, take]
                prod[dst, take] = prod[src, take]
                valid[dst, take] = True

    # A node is interior only if all six directions found a cell.
    interior = valid.all(axis=0)
    boundary = ~interior

    # Per-axis difference coefficients:  D_a u = cm*(u_minus - u0) + cp*(u_plus - u0)
    cm = np.zeros((3, n))
    cp = np.zeros((3, n))
    C = np.zeros((n, 3, 3))
    for a in range(3):
        dm, dp = 2 * a, 2 * a + 1
        sm, sp = sdist[dm], sdist[dp]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(interior, 2.0 / (sm + sp), 0.0)
            cm[a] = np.where(interior, scale / sm, 0.0)
            cp[a] = np.where(interior, scale / sp, 0.0)
        C[:, a, a] = 1.0
        for t in range(3):
            if t == a:
                continue
            # Each interpolated side contributes (a*b/2) * 2/((sm+sp)*s_side).
            C[:, a, t] = np.where(
                interior,
                0.5 * (prod[dm, :, t] * cm[a] + prod[dp, :, t] * cp[a]),
                0.0,
            )

    # Solve M^T w = 1 per node so that sum_a w_a D_a = laplacian.
    w = np.zeros((n, 3))
    if interior.any():
        if interface_correction:
            Mi = np.transpose(C[interior], (0, 2, 1))
            w[interior] = np.linalg.solve(Mi, np.ones((int(interior.sum()), 3)))
        else:
            w[interior] = 1.0

    # A row is "regular" when every direction landed exactly on a real node.
    exact = (np.abs(wgt) > 1e-14).sum(axis=2) == 1
    regular = interior & exact.all(axis=0)
    irregular = interior & ~regular

    def assemble(sel, width):
        m = sel.size
        idx = np.zeros((m, width), dtype=np.int64)
        val = np.zeros((m, width), dtype=np.float64)
        diag = np.zeros(m)
        col = 1
        for d, (axis, sign) in enumerate(DIRS):
            coef = (cp[axis] if sign > 0 else cm[axis])[sel] * w[sel, axis]
            diag -= coef
            k = 1 if width == 7 else 4
            if k == 1:
                # weight is exactly 1 on a single interpolant
                pick = np.argmax(np.abs(wgt[d, sel]), axis=1)
                idx[:, col] = nid[d, sel, pick]
                val[:, col] = coef
            else:
                idx[:, col:col + 4] = nid[d, sel]
                val[:, col:col + 4] = coef[:, None] * wgt[d, sel]
            col += k
        idx[:, 0] = sel
        val[:, 0] = diag
        return idx, val

    reg_sel = np.flatnonzero(regular)
    irr_sel = np.flatnonzero(irregular)
    reg_idx, reg_val = assemble(reg_sel, 7)
    irr_idx, irr_val = assemble(irr_sel, 25)

    return Rows(
        nnode=n,
        reg_row=reg_sel, reg_idx=reg_idx, reg_val=reg_val,
        irr_row=irr_sel, irr_idx=irr_idx, irr_val=irr_val,
        boundary=boundary, coords=coords,
    )


@dataclass
class GradRows:
    """Per-axis first-derivative rows, same padded gather-reduce layout."""

    nnode: int
    row: np.ndarray          # (m,)        nodes with a full 3-D stencil
    idx: np.ndarray          # (3, m, 9)
    val: np.ndarray          # (3, m, 9)

    def axis(self, a: int):
        return self.row, self.idx[a], self.val[a]


def build_gradient(tree: Octree, neumann=None) -> GradRows:
    """Rows for d/dx, d/dy, d/dz at every interior node.

    Uses the same neighbours and the same ghost interpolants as the Laplacian,
    so the gradient is a property of the discretization rather than a
    post-hoc difference of the solution.  The coefficients are those of the
    quadratic through the two neighbours and the node, which on unequal
    spacings ``s_m``, ``s_p`` gives

        u' = [ s_m^2 (u_p - u0) + s_p^2 (u0 - u_m) ] / (s_m s_p (s_m + s_p))

    and reduces to the usual centred difference when ``s_m == s_p``.

    Note: this recomputes the neighbour probes rather than caching them on
    ``Rows``, which would cost ~770 bytes/node.  Gradients are needed once, at
    the end, so recomputing is the cheaper trade.
    """
    coords, _ = tree.nodes()
    n = coords.shape[0]
    ny, nz = int(tree.key_dims[1]), int(tree.key_dims[2])
    keys_sorted = (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]

    nid, wgt, sdist, valid, _ = neighbours(tree, coords, keys_sorted)
    # Mirror the same way the Laplacian does, so the gradient exists (and is
    # correctly zero in the normal direction) on Neumann surfaces.
    if neumann is not None:
        nm = np.asarray(neumann, dtype=bool)
        for a in range(3):
            dm, dp = 2 * a, 2 * a + 1
            for dst, src in ((dm, dp), (dp, dm)):
                take = nm & ~valid[dst] & valid[src]
                if take.any():
                    nid[dst, take] = nid[src, take]
                    wgt[dst, take] = wgt[src, take]
                    sdist[dst, take] = sdist[src, take]
                    valid[dst, take] = True
    interior = valid.all(axis=0)
    sel = np.flatnonzero(interior)
    m = sel.size

    idx = np.zeros((3, m, 9), dtype=np.int64)
    val = np.zeros((3, m, 9), dtype=np.float64)
    for a in range(3):
        dm, dp = 2 * a, 2 * a + 1
        sm, sp = sdist[dm, sel], sdist[dp, sel]
        den = sm * sp * (sm + sp)
        cp = sm * sm / den            # coefficient on the + neighbour
        cmm = -sp * sp / den          # coefficient on the - neighbour
        idx[a, :, 0] = sel
        val[a, :, 0] = -(cp + cmm)
        idx[a, :, 1:5] = nid[dm, sel]
        val[a, :, 1:5] = cmm[:, None] * wgt[dm, sel]
        idx[a, :, 5:9] = nid[dp, sel]
        val[a, :, 5:9] = cp[:, None] * wgt[dp, sel]
    return GradRows(nnode=n, row=sel, idx=idx, val=val)
