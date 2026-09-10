"""Node-based Laplacian rows, as constructed in Min, Gibou & Ceniceros.

This implements Sections 5.1-5.3 of

    C. Min, F. Gibou, H.D. Ceniceros, "A supra-convergent finite difference
    scheme for the variable coefficient Poisson equation on non-graded grids",
    J. Comput. Phys. 218 (2006) 300-321,

as written.  Only the constant-coefficient (Laplace) case of Section 5 is
implemented; the paper's Section 6 variable coefficient div(rho grad u) and its
Section 8 gradient correction term are deliberately absent.

The construction, per node ``u0``
---------------------------------
``u0`` is a *corner* of a chosen leaf cell ``C``.  ``C`` fixes an octant, and
for each axis the direction pointing into ``C`` is *inward*, the other
*outward*:

  * the **inward** neighbour is ``C``'s adjacent corner along that axis.  It is
    always a real mesh node, at distance ``s_in = h_C``.  These are the paper's
    ``u1, u2, u3`` at ``s1, s2, s3`` (Fig. 6 in 2D, Fig. 7 in 3D).
  * the **outward** neighbour is the point on the far face of ``C_a`` -- the
    leaf on the outward side of ``C`` along that axis that also contains ``u0``
    -- aligned with ``u0``, at distance ``s_out = h_{C_a}``.  These are the
    paper's ``u4, u5, u6`` at ``s4, s5, s6``.  It is multilinearly interpolated
    from ``C_a``'s four far-face corners, which degenerates to two nodes when
    ``u0`` lies on an edge of ``C_a`` ("face-aligned") and to a single node when
    ``u0`` is a corner of it ("edge-aligned", no interpolation).

The stencil at one node therefore touches ``C`` and at most three of its
neighbouring cells, which is the locality property the paper is built around.

Each axis then carries the standard nonuniform difference, the paper's Eq. (1),

    D_a u = (2/(s_in + s_out)) * ( (u_in - u0)/s_in + (u_out - u0)/s_out )

An outward multilinear interpolation with offsets ``p, q`` either side along a
transverse axis ``t`` reproduces a smooth field with error ``(p*q/2) u_tt``, so

    D_a u = u_aa + sum_t M[a][t] u_tt + O(h),
    M[a][t] = (p_t q_t / 2) * 2/((s_in + s_out) s_out),   M[a][a] = 1.

The paper cancels the spurious transverse terms by the linear weighting of its
Eq. (8), ``alpha * D_x + D_y + beta * D_z = laplacian + O(h)``.  Solving
``M^T w = (1,1,1)`` returns exactly that: the paper's configuration makes ``M``
unit-triangular under the permutation (edge-aligned, face-aligned,
face-interior), so the solve reproduces ``alpha``, ``beta`` and ``w == 1`` on
the fully interpolated axis, including the paper's normalization.  It is the
paper's formula, not a generalization of it.

One erratum, recorded because the code deliberately departs from the printed
text: the paper's Eq. (8) defines ``alpha = 1 - s10 s11/(s5(s2+s5))`` and
``beta = 1 - s9 s12/(s5(s2+s5)) - alpha s7 s8/(s4(s1+s4))``.  Read against
Eq. (6), Fig. 7's geometry and the ``u5`` interpolation formula -- where
``{s9,s12}`` are the x-offsets and ``{s10,s11}`` the z-offsets of ``u5`` on
``C_y``'s far face -- those two products are interchanged.  The self-consistent
values are ``alpha = 1 - s9 s12/(s5(s2+s5))`` and
``beta = 1 - s10 s11/(s5(s2+s5)) - alpha s7 s8/(s4(s1+s4))``, which is what the
``M^T w = 1`` solve produces.

Rows are emitted in two padded gather-reduce blocks, because on any practical
grid the overwhelming majority of nodes have every neighbour landing on a real
node:

  * ``regular``   -- 7 entries (diagonal + 3 inward + 3 outward)
  * ``irregular`` -- 16 entries (diagonal + 3 inward + 3 x 4 interpolants)

which keeps the memory traffic of a matvec close to the ideal 7-point cost.
The paper's structural guarantee bounds a row at 11 entries; the block is padded
to 16 with zero-valued columns, which are inert in the gather-reduce and cost
nothing numerically.

Transverse periodicity is an extension the paper does not consider.  It is
preserved here, and shows up only in the origin un-wrap of a cell located
through a periodic seam (:func:`_unwrap_origin`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tree import Octree

# The eight octants around a node.  Octant ``m`` has sign ``+1`` on axis ``a``
# when bit ``2 - a`` of ``m`` is set, so flipping the direction along axis ``a``
# is the single bit operation ``m ^ (4 >> a)``.  The enumeration order matches
# the corner order of ``Octree.nodes()``.
OCT_SIGN = np.array([[((m >> (2 - a)) & 1) * 2 - 1 for a in range(3)]
                     for m in range(8)], dtype=np.int64)


def _flip(m: int, a: int) -> int:
    """Octant ``m`` with its direction along axis ``a`` reversed."""
    return m ^ (4 >> a)


@dataclass
class Stencil:
    """The paper's node stencil, anchored on one incident leaf ``C`` per node.

    ``in_*`` describes the three inward neighbours (single real nodes), ``out_*``
    the three outward neighbours (up to four interpolants each).  All arrays are
    axis-major and defined only where ``have`` (resp. ``out_ok``) is True.
    """

    cell: np.ndarray      # (n,)        chosen leaf C, -1 where there is none
    toward: np.ndarray    # (n, 3)      +-1, direction from u0 into C
    have: np.ndarray      # (n,)  bool  a usable C was found
    in_nid: np.ndarray    # (3, n)      inward neighbour node id
    in_s: np.ndarray      # (3, n)      inward distance (mm), == C's edge length
    out_nid: np.ndarray   # (3, n, 4)   far-face corners of C_a
    out_wgt: np.ndarray   # (3, n, 4)   multilinear weights, summing to 1
    out_s: np.ndarray     # (3, n)      outward distance (mm)
    out_ok: np.ndarray    # (3, n) bool the outward neighbour exists
    out_prod: np.ndarray  # (3, n, 3)   p_t * q_t; entry [a, :, a] is always 0
    out_nsup: np.ndarray  # (3, n) int8 1 edge-aligned, 2 face-aligned, 4 interior


@dataclass
class Rows:
    """Composite Laplacian in two padded gather-reduce blocks.

    ``A @ u`` is  ``zeros(n).index_add(reg_row, (reg_val * u[reg_idx]).sum(1))``
    plus the same for the irregular block.  Both blocks are dense in their
    second dimension, so the matvec is a pure gather-multiply-reduce with no
    sparse-format overhead.  Column 0 of each block is the diagonal.
    """

    nnode: int
    reg_row: np.ndarray   # (nr,)      node id owning each regular row
    reg_idx: np.ndarray   # (nr, 7)    column node ids
    reg_val: np.ndarray   # (nr, 7)    coefficients
    irr_row: np.ndarray   # (ni,)
    irr_idx: np.ndarray   # (ni, 16)
    irr_val: np.ndarray   # (ni, 16)
    boundary: np.ndarray  # (nnode,) bool -- no stencil row (domain boundary)
    coords: np.ndarray    # (nnode, 3) integer lattice coordinates

    @property
    def nnz(self) -> int:
        return self.reg_idx.size + self.irr_idx.size


REG_WIDTH = 7
IRR_WIDTH = 16


def _node_ids(keys_sorted, coords, ny, nz, tree=None) -> np.ndarray:
    """Map integer lattice coordinates to node ids via the sorted key table."""
    if tree is not None:
        coords = tree.fold(coords)
    k = (coords[..., 0] * ny + coords[..., 1]) * nz + coords[..., 2]
    pos = np.searchsorted(keys_sorted, k.ravel())
    np.clip(pos, 0, keys_sorted.size - 1, out=pos)
    if not np.all(keys_sorted[pos] == k.ravel()):
        raise AssertionError("stencil referenced a non-existent node")
    return pos.reshape(k.shape)


def _unwrap_origin(tree: Octree, org: np.ndarray, csz: np.ndarray,
                   p: np.ndarray) -> np.ndarray:
    """Shift a located cell's origin into the unwrapped frame nearest ``p``.

    A cell reached through a periodic seam has its origin on the far side of the
    domain.  Distances and interpolation offsets must be measured locally, so the
    origin is translated by whole periods until the cell straddles ``p``; the
    resulting corner coordinates are folded back when they are looked up.
    """
    for a in range(3):
        if tree.periodic[a]:
            D = int(tree.dims_units[a])
            shift = np.round((p[:, a] - (org[:, a] + 0.5 * csz)) / D)
            org[:, a] = org[:, a] + (D * shift).astype(np.int64)
    return org


def stencil(tree: Octree, coords: np.ndarray, keys_sorted: np.ndarray,
            neumann=None) -> Stencil:
    """Build the paper's cell-anchored stencil for every node.

    ``C`` is chosen as the finest leaf incident to the node that is *usable*:
    the node must be a corner of it, and all three of its axis-neighbours
    containing the node must exist.  Ties between equally fine candidates go to
    the lowest octant code, which is the same convention ``Octree.nodes()`` uses
    for corner ordering.  The paper does not specify the choice of ``C``; the
    corner requirement, however, is the paper's (the node is a corner of ``C``
    in both Fig. 6 and Fig. 7), and it is what makes an edge-aligned outward
    direction available at every node.

    Where ``neumann`` is set, a node whose outward probe leaves the domain is
    still given a row, by mirroring the inward neighbour onto the outward side.
    The axis difference then collapses to ``2 (u_in - u0)/s^2``, the reflected
    stencil for ``du/dn = 0``.  A mirrored side is a real node, so it injects no
    interpolation error.  This is a boundary condition the paper does not treat;
    it is unrelated to the paper's Section 7, which removes the singularity of
    an all-Neumann system.
    """
    n = coords.shape[0]
    ny, nz = int(tree.key_dims[1]), int(tree.key_dims[2])

    # ---- probe all eight octants once.  The same eight probes give both C and
    # every C_a, so the whole stencil costs 8 locate calls per node.  These are
    # the largest transient arrays in the build, so they use narrow dtypes:
    # int32 indexes any plausible cell count and levels are bounded by lmax.
    cid = np.full((8, n), -1, dtype=np.int32)
    lev = np.full((8, n), -1, dtype=np.int16)
    for m in range(8):
        c, ok = tree.locate(coords, OCT_SIGN[m])
        cid[m] = np.where(ok, c, -1)
        lev[m] = np.where(ok, tree.cells[np.maximum(c, 0), 0], -1)

    # ---- is the node aligned with the octant-m leaf's lattice, per axis?
    # A periodic wrap shifts an origin by a whole number of periods, and every
    # period is a multiple of every cell size, so this modular test needs no
    # un-wrap.
    aligned = np.zeros((8, n, 3), dtype=bool)
    for m in range(8):
        csz = tree.cell_size_units(np.maximum(lev[m], 0))
        org = tree.cells[np.maximum(cid[m], 0), 1:] * csz[:, None]
        ok = lev[m] >= 0
        for a in range(3):
            aligned[m, :, a] = ok & (((coords[:, a] - org[:, a]) % csz) == 0)

    # ---- usable C: the node is a corner of it, and all three C_a exist.
    corner = aligned.all(axis=2) & (lev >= 0)
    out_exists = np.stack(
        [np.stack([lev[_flip(m, a)] >= 0 for a in range(3)]).all(axis=0)
         for m in range(8)])
    full = corner & out_exists

    # Prefer a fully usable C at any level over a relaxed one; among equals take
    # the finest, then the lowest octant code (np.argmax returns the first
    # maximum).  The relaxed tier admits a C whose outward probe leaves the
    # domain, and exists only for Neumann nodes, where that side gets mirrored.
    TIER = np.int32(1 << 8)          # > any level, so the tier dominates
    lv = lev.astype(np.int32)
    score = np.where(full, TIER + lv, -1)
    if neumann is not None:
        nm = np.asarray(neumann, dtype=bool)
        score = np.where(full, TIER + lv, np.where(corner & nm, lv, -1))
    pick = np.argmax(score, axis=0)
    rows = np.arange(n)
    have = score[pick, rows] >= 0

    toward = np.where(have[:, None], OCT_SIGN[pick], 0)
    cellC = np.where(have, cid[pick, rows], -1)

    in_nid = np.zeros((3, n), dtype=np.int64)
    in_s = np.zeros((3, n), dtype=np.float64)
    out_nid = np.zeros((3, n, 4), dtype=np.int64)
    out_wgt = np.zeros((3, n, 4), dtype=np.float64)
    out_s = np.zeros((3, n), dtype=np.float64)
    out_ok = np.zeros((3, n), dtype=bool)
    out_prod = np.zeros((3, n, 3), dtype=np.float64)
    out_nsup = np.zeros((3, n), dtype=np.int8)

    sel = np.flatnonzero(have)
    if sel.size:
        p = coords[sel]
        mC = pick[sel]
        cszC = tree.cell_size_units(lev[mC, sel])
        sgn = OCT_SIGN[mC]

        for a in range(3):
            b, c = [x for x in (0, 1, 2) if x != a]

            # ---- inward: C's adjacent corner along this axis, always a node
            q = p.copy()
            q[:, a] = p[:, a] + sgn[:, a] * cszC
            in_nid[a, sel] = _node_ids(keys_sorted, q, ny, nz, tree)
            in_s[a, sel] = cszC.astype(np.float64) * tree.h_min

            # ---- outward: the aligned point on C_a's far face
            mA = np.array([_flip(int(m), a) for m in range(8)])[mC]
            okA = lev[mA, sel] >= 0
            if not okA.any():
                continue
            s2 = sel[okA]
            p2 = p[okA]
            cszA = tree.cell_size_units(lev[mA[okA], s2])
            orgA = (tree.cells[cid[mA[okA], s2], 1:] * cszA[:, None]).copy()
            _unwrap_origin(tree, orgA, cszA, p2)

            osign = -sgn[okA, a]
            step = np.where(osign > 0,
                            orgA[:, a] + cszA - p2[:, a],
                            p2[:, a] - orgA[:, a])
            if tree.periodic[a]:
                # A cell spanning a whole period can put the node on the very
                # face the stencil leaves through; then the far face is one full
                # period away.
                step = np.where(step <= 0, step + int(tree.dims_units[a]), step)
            face = p2[:, a] + osign * step
            out_s[a, s2] = step.astype(np.float64) * tree.h_min

            fb = (p2[:, b] - orgA[:, b]) / cszA.astype(np.float64)
            fc = (p2[:, c] - orgA[:, c]) / cszA.astype(np.float64)
            w = np.stack([(1 - fb) * (1 - fc), (1 - fb) * fc,
                          fb * (1 - fc), fb * fc], axis=1)

            corners = np.empty((s2.size, 4, 3), dtype=np.int64)
            for k, (ob, oc) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                corners[:, k, a] = face
                corners[:, k, b] = orgA[:, b] + ob * cszA
                corners[:, k, c] = orgA[:, c] + oc * cszA
            out_nid[a, s2] = _node_ids(keys_sorted, corners, ny, nz, tree)
            out_wgt[a, s2] = w

            # Interpolation error products p*q, in each transverse direction.
            # These use C_a's size, not C's -- the two decouple here.
            hA = cszA.astype(np.float64) * tree.h_min
            out_prod[a, s2, b] = (fb * (1 - fb)) * hA * hA
            out_prod[a, s2, c] = (fc * (1 - fc)) * hA * hA
            out_ok[a, s2] = True
            out_nsup[a, s2] = (np.abs(w) > 1e-14).sum(axis=1)

    # ---- homogeneous Neumann by mirroring the inward neighbour outward.
    if neumann is not None:
        nm = np.asarray(neumann, dtype=bool)
        for a in range(3):
            take = nm & have & ~out_ok[a]
            if not take.any():
                continue
            out_nid[a, take, 0] = in_nid[a, take]
            out_wgt[a, take, 0] = 1.0
            out_s[a, take] = in_s[a, take]
            out_nsup[a, take] = 1
            out_ok[a, take] = True

    return Stencil(cell=cellC, toward=toward, have=have,
                   in_nid=in_nid, in_s=in_s,
                   out_nid=out_nid, out_wgt=out_wgt, out_s=out_s,
                   out_ok=out_ok, out_prod=out_prod, out_nsup=out_nsup)


def _axis_coefficients(st: Stencil, interior: np.ndarray):
    """Eq. (1) coefficients and the transverse-contamination matrix ``M``."""
    n = interior.size
    c_in = np.zeros((3, n))
    c_out = np.zeros((3, n))
    M = np.zeros((n, 3, 3))
    for a in range(3):
        si, so = st.in_s[a], st.out_s[a]
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(interior, 2.0 / (si + so), 0.0)
            c_in[a] = np.where(interior, scale / si, 0.0)
            c_out[a] = np.where(interior, scale / so, 0.0)
        M[:, a, a] = 1.0
        for t in range(3):
            if t == a:
                continue
            # Only the outward side is interpolated; the inward neighbour is a
            # real node and contributes no error.
            M[:, a, t] = np.where(interior,
                                  0.5 * st.out_prod[a, :, t] * c_out[a], 0.0)
    return c_in, c_out, M


def paper_weights(M: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """The weights of the paper's Eq. (8): solve ``M^T w = (1,1,1)`` per node.

    Under the paper's structure ``M`` is unit-triangular in the order
    (edge-aligned, face-aligned, face-interior), so this returns exactly
    ``alpha``, ``beta`` and ``w == 1`` on the fully interpolated axis.  A
    residual check guards the case where the structure does not hold and the
    triangular reading would be invalid.
    """
    w = np.zeros((interior.size, 3))
    if not interior.any():
        return w
    Mi = np.transpose(M[interior], (0, 2, 1))
    w[interior] = np.linalg.solve(Mi, np.ones((int(interior.sum()), 3)))
    res = np.abs(np.einsum("nat,na->nt", M[interior], w[interior]) - 1.0).max()
    if res > 1e-9:
        raise AssertionError(
            f"transverse cancellation failed: max residual {res:.3e}")
    return w


def build_rows(tree: Octree, interface_correction: bool = True,
               neumann=None) -> Rows:
    """Assemble the composite Laplacian rows for every node of ``tree``.

    With ``interface_correction=False`` the paper's weights are forced to 1,
    giving the naive scheme that simply sums the three directional differences.
    That variant is locally inconsistent at hanging nodes (the Losasso-style
    first-order treatment); it exists so the cost of the paper's cancellation can
    be measured rather than assumed.
    """
    coords, _ = tree.nodes()
    n = coords.shape[0]
    ny, nz = int(tree.key_dims[1]), int(tree.key_dims[2])
    keys_sorted = (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]

    st = stencil(tree, coords, keys_sorted, neumann=neumann)

    interior = st.have & st.out_ok.all(axis=0)
    boundary = ~interior

    c_in, c_out, M = _axis_coefficients(st, interior)
    if interface_correction:
        w = paper_weights(M, interior)
    else:
        w = np.zeros((n, 3))
        w[interior] = 1.0

    # A row is regular when every neighbour, inward and outward, is a real node.
    regular = interior & (st.out_nsup == 1).all(axis=0)
    irregular = interior & ~regular

    def assemble(sel, width):
        m = sel.size
        idx = np.zeros((m, width), dtype=np.int64)
        val = np.zeros((m, width), dtype=np.float64)
        diag = np.zeros(m)
        col = 1
        for a in range(3):
            coef = c_in[a][sel] * w[sel, a]
            idx[:, col] = st.in_nid[a, sel]
            val[:, col] = coef
            diag -= coef
            col += 1
        k = 1 if width == REG_WIDTH else 4
        for a in range(3):
            coef = c_out[a][sel] * w[sel, a]
            diag -= coef
            if k == 1:
                # the weight is exactly 1 on a single interpolant
                pick = np.argmax(np.abs(st.out_wgt[a, sel]), axis=1)
                idx[:, col] = st.out_nid[a, sel, pick]
                val[:, col] = coef
            else:
                idx[:, col:col + 4] = st.out_nid[a, sel]
                val[:, col:col + 4] = coef[:, None] * st.out_wgt[a, sel]
            col += k
        idx[:, 0] = sel
        val[:, 0] = diag
        return idx, val

    reg_sel = np.flatnonzero(regular)
    irr_sel = np.flatnonzero(irregular)
    reg_idx, reg_val = assemble(reg_sel, REG_WIDTH)
    irr_idx, irr_val = assemble(irr_sel, IRR_WIDTH)

    return Rows(
        nnode=n,
        reg_row=reg_sel, reg_idx=reg_idx, reg_val=reg_val,
        irr_row=irr_sel, irr_idx=irr_idx, irr_val=irr_val,
        boundary=boundary, coords=coords,
    )


@dataclass
class GradRows:
    """Per-axis first-derivative rows, same padded gather-reduce layout.

    Two blocks, for the same reason the Laplacian has two: the Section 8
    interpolation correction is needed only where an outward neighbour was
    actually interpolated, which is a few percent of nodes.  ``idx``/``val``
    carry the difference itself at every node; ``corr_idx``/``corr_val`` carry
    the correction, and apply to the rows named by ``corr_pos`` (positions
    *within* ``row``, not node ids).  The correction's own diagonal term is
    folded into column 0 of the base block, so the correction block has no
    diagonal.
    """

    nnode: int
    row: np.ndarray          # (m,)         nodes with a full 3-D stencil
    idx: np.ndarray          # (3, m, 6)    diagonal, inward, 4 outward
    val: np.ndarray          # (3, m, 6)
    corr_pos: np.ndarray     # (k,)         positions within ``row``
    corr_idx: np.ndarray     # (3, k, 10)   2 x (inward + 4 outward), transverse
    corr_val: np.ndarray     # (3, k, 10)


def build_gradient(tree: Octree, neumann=None) -> GradRows:
    """Rows for d/dx, d/dy, d/dz at every interior node.

    Uses the same stencil as the Laplacian -- the same cell ``C``, the same
    neighbours and the same interpolants -- so the gradient is a property of the
    discretization rather than a post-hoc difference of the solution.  The
    coefficients are those of the quadratic through the two neighbours and the
    node, which on unequal spacings ``s_m``, ``s_p`` gives

        u' = [ s_m^2 (u_p - u0) + s_p^2 (u0 - u_m) ] / (s_m s_p (s_m + s_p))

    equivalently the paper's Section 8 weighted average of the forward and
    backward differences, and reduces to the usual centred difference when
    ``s_m == s_p``.  Which side is which is now per node: the inward neighbour
    lies on the ``+`` side exactly where ``toward`` is positive.

    Section 8's interpolation correction is applied.  Where the outward
    neighbour was interpolated it carries ``sum_t (p_t q_t / 2) u_tt``, which
    enters the difference multiplied by that side's coefficient, so

        u'_a  -=  c_out[a] * sum_{t != a} (p_t q_t / 2) * u_tt

    This is the paper's ``- s5 s6 s1/(2 s4 (s1+s4)) u_yy`` in 2D and its 3D
    counterpart, and it uses the same cells as the Laplacian, so the locality of
    the scheme is preserved.

    ``u_tt`` here must be the *consistent* second difference -- what the paper
    means by "the finite differences for uxx, uyy and uzz are given in
    Section 5.3".  The raw axis differences satisfy ``D = M u_vec`` with the
    same contamination matrix the Laplacian weights invert, so

        u_vec = M^{-1} D.

    Using the raw ``D_t`` instead is only correct when direction ``t`` is itself
    uninterpolated; where two directions are both interpolated it leaves an O(h)
    residual.  With ``M^{-1}`` the gradient is exact for every quadratic at every
    node; without Section 8 at all it is only first order at interpolated nodes.

    This recomputes the octant probes rather than caching them on ``Rows``,
    which would cost ~600 bytes/node.  Gradients are needed once, at the end, so
    recomputing is the cheaper trade.
    """
    coords, _ = tree.nodes()
    n = coords.shape[0]
    ny, nz = int(tree.key_dims[1]), int(tree.key_dims[2])
    keys_sorted = (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]

    st = stencil(tree, coords, keys_sorted, neumann=neumann)
    interior = st.have & st.out_ok.all(axis=0)
    sel = np.flatnonzero(interior)
    m = sel.size

    # Second-difference coefficients and the contamination matrix, for the
    # Section 8 correction.
    d_in, d_out, M = _axis_coefficients(st, interior)

    # Rows needing the correction: some outward neighbour was interpolated.
    needs = (st.out_nsup[:, sel] > 1).any(axis=0)
    corr_pos = np.flatnonzero(needs)
    k = corr_pos.size
    csel = sel[corr_pos]

    # kappa[a][s]: the coefficient of the raw axis-s difference D_s in the
    # Section 8 correction to u'_a, after decontaminating via u_vec = M^-1 D.
    kappa = np.zeros((3, k, 3))
    if k:
        N = np.linalg.inv(M[csel])
        for a in range(3):
            for s in range(3):
                kappa[a, :, s] = -sum(
                    st.out_prod[a, csel, t] / 2.0 * N[:, t, s]
                    for t in range(3) if t != a)

    idx = np.zeros((3, m, 6), dtype=np.int64)
    val = np.zeros((3, m, 6), dtype=np.float64)
    corr_idx = np.zeros((3, k, 10), dtype=np.int64)
    corr_val = np.zeros((3, k, 10), dtype=np.float64)

    for a in range(3):
        tw = st.toward[sel, a]
        s_in = st.in_s[a, sel]
        s_out = st.out_s[a, sel]
        sp = np.where(tw > 0, s_in, s_out)     # spacing on the + side
        sm = np.where(tw > 0, s_out, s_in)     # spacing on the - side
        den = sm * sp * (sm + sp)
        cp = sm * sm / den                     # coefficient on the + neighbour
        cm = -sp * sp / den                    # coefficient on the - neighbour
        c_in = np.where(tw > 0, cp, cm)
        c_out = np.where(tw > 0, cm, cp)
        idx[a, :, 0] = sel
        val[a, :, 0] = -(cp + cm)
        idx[a, :, 1] = st.in_nid[a, sel]
        val[a, :, 1] = c_in
        idx[a, :, 2:6] = st.out_nid[a, sel]
        val[a, :, 2:6] = c_out[:, None] * st.out_wgt[a, sel]

        if k == 0:
            continue
        # Section 8: add sum_s kappa[a][s] * D_s.  The s == a term references
        # the same nodes as the base block, so it folds into it; the two
        # transverse terms go in the correction block.  Every D_s diagonal
        # folds into base column 0, which is the node itself.
        co = c_out[corr_pos]
        col = 0
        for s in range(3):
            ks = kappa[a, :, s] * co
            val[a, corr_pos, 0] += ks * -(d_in[s][csel] + d_out[s][csel])
            if s == a:
                val[a, corr_pos, 1] += ks * d_in[s][csel]
                val[a, corr_pos, 2:6] += (
                    (ks * d_out[s][csel])[:, None] * st.out_wgt[s, csel])
                continue
            corr_idx[a, :, col] = st.in_nid[s, csel]
            corr_val[a, :, col] = ks * d_in[s][csel]
            corr_idx[a, :, col + 1:col + 5] = st.out_nid[s, csel]
            corr_val[a, :, col + 1:col + 5] = (
                (ks * d_out[s][csel])[:, None] * st.out_wgt[s, csel])
            col += 5

    return GradRows(nnode=n, row=sel, idx=idx, val=val,
                    corr_pos=corr_pos, corr_idx=corr_idx, corr_val=corr_val)


def check_paper_structure(tree: Octree, neumann=None, strict: bool = True):
    """Verify the structural guarantee the paper's Eq. (8) relies on.

    Returns a dict of diagnostics.  The load-bearing property is that the
    transverse-contamination pattern admits a triangular ordering; the paper's
    specific realization of it (Fig. 7) is one axis edge-aligned, one
    face-aligned and one face-interior.
    """
    coords, _ = tree.nodes()
    ny, nz = int(tree.key_dims[1]), int(tree.key_dims[2])
    keys_sorted = (coords[:, 0] * ny + coords[:, 1]) * nz + coords[:, 2]
    st = stencil(tree, coords, keys_sorted, neumann=neumann)
    interior = st.have & st.out_ok.all(axis=0)
    ni = int(interior.sum())

    nsup = st.out_nsup[:, interior]
    multiset = np.sort(nsup, axis=0).T
    no_clean = int((multiset.min(axis=1) != 1).sum())

    # nilpotency of the strict off-diagonal support: P^3 == 0 <=> orderable
    _, _, M = _axis_coefficients(st, interior)
    P = (np.abs(M[interior]) > 0).astype(np.int64)
    for a in range(3):
        P[:, a, a] = 0
    P3 = P @ P @ P
    not_orderable = int((P3 != 0).any(axis=(1, 2)).sum())

    # all three inward distances are C's edge length
    si = st.in_s[:, interior]
    same_in = bool(np.allclose(si[0], si[1]) and np.allclose(si[1], si[2]))
    coarser_out = int((st.out_s[:, interior] < si - 1e-12).sum())

    width = 1 + 3 + nsup.sum(axis=0)
    info = dict(interior=ni, no_clean=no_clean, not_orderable=not_orderable,
                inward_consistent=same_in, outward_finer_than_C=coarser_out,
                max_width=int(width.max()) if ni else 0,
                nsup_histogram={tuple(int(x) for x in k): int(v) for k, v in
                                zip(*np.unique(multiset, axis=0,
                                               return_counts=True))})
    if strict and (no_clean or not_orderable or not same_in or coarser_out):
        raise AssertionError(f"paper structure violated: {info}")
    return info
