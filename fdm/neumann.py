"""Section 7 of Min, Gibou & Ceniceros: the all-Neumann three-step procedure.

When Neumann conditions are imposed on the *entire* boundary the linear system
is singular -- adding a constant gives another solution.  Only the gradient
matters in that case, so it is enough to pin down one solution.  Pinning a
single corner node makes the system nonsingular, but the artificial Dirichlet
value corrupts the solution's gradient near that corner, leaving it only first
order there.  The paper's remedy (its Fig. 8) is three steps:

  1. impose Dirichlet at one corner of the domain and solve;
  2. re-solve the same equation on a small subdomain containing that corner,
     taking Dirichlet data from step 1 on the subdomain's *interior-facing*
     boundary only.  The pinned corner is free in this solve, which is what
     removes the first-order error;
  3. overwrite the gradients at the nodes near the corner using step 2.

Step 2 needs no new machinery and no submesh: freezing every node outside the
subdomain into the Dirichlet mask of the same full-size operator is
algebraically the subdomain problem.  The cost is one extra solve with O(sub^3)
free unknowns, which converges in a handful of iterations.

This is unrelated to the ``neumann=`` mirroring in :mod:`fdm.topology`, which is
the *discretization* of ``du/dn = 0`` on a face and is what writes a Neumann row
in the first place.  Section 7 addresses the singularity of the resulting
system.  With any Dirichlet electrode present the system is nonsingular and this
module must not be used -- hence the explicit check in :func:`solve_all_neumann`.
"""

from __future__ import annotations

import numpy as np
import torch

from .field import Gradient
from .operator import Operator
from .solve import solve
from .topology import Rows, build_gradient
from .tree import Octree


def pin_node(rows: Rows, tree: Octree) -> int:
    """The domain corner node to pin.

    ``Octree.nodes()`` sorts by packed key, so on a bounded domain the
    ``(0,0,0)`` corner is node 0.  Asserted rather than assumed.
    """
    if tree.periodic.any():
        raise ValueError("Section 7 assumes a bounded domain on every axis")
    if not np.all(rows.coords[0] == 0):
        raise AssertionError(
            f"expected node 0 at the (0,0,0) corner, got {rows.coords[0]}")
    return 0


def corner_cell_size(tree: Octree, rows: Rows, pin: int) -> int:
    """Edge length in units of the leaf cell at the pinned corner."""
    cell, ok = tree.locate(rows.coords[pin:pin + 1], np.array([1, 1, 1]))
    if not bool(ok[0]):
        raise AssertionError("no leaf cell at the pinned corner")
    return int(tree.cell_size_units(tree.cells[int(cell[0]), 0]))


def subdomain(tree: Octree, rows: Rows, pin: int, sub: int, patch: int):
    """Masks for step 2 and step 3.

    Returns ``(frozen, patch_nodes)``.  ``frozen`` is the Dirichlet mask of the
    step-2 problem: every node outside the subdomain box, plus the box's
    interior-facing skin.  A node on a *physical* domain face keeps its
    mirrored-Neumann row, because the cells beyond it do not exist and so it
    never lands in the skin -- which is exactly the paper's "Dirichlet on the
    subdomain's interior-facing edges" and preserves the physical condition.
    ``patch_nodes`` marks the (strictly smaller) region whose gradients step 3
    overwrites, kept clear of the artificial skin.
    """
    if not 0 < patch < sub:
        raise ValueError(f"need 0 < patch < sub, got patch={patch}, sub={sub}")
    h0 = corner_cell_size(tree, rows, pin)
    lo = rows.coords[pin]
    csz = tree.sizes_units()
    org = tree.origins_units()
    _, corner_ids = tree.nodes()

    def cells_within(extent):
        hi = lo + extent
        return np.all((org >= lo) & (org + csz[:, None] <= hi), axis=1)

    inside = cells_within(sub * h0)
    if not inside.any():
        raise ValueError(f"subdomain of {sub} corner cells contains no cell")

    innode = np.zeros(rows.nnode, dtype=bool)
    innode[corner_ids[inside]] = True
    skin = np.zeros(rows.nnode, dtype=bool)
    if (~inside).any():
        skin[corner_ids[~inside]] = True
    frozen = ~innode | (innode & skin)

    patch_cells = cells_within(patch * h0)
    patch_nodes = np.zeros(rows.nnode, dtype=bool)
    if patch_cells.any():
        patch_nodes[corner_ids[patch_cells]] = True
    patch_nodes &= ~frozen
    return frozen, patch_nodes


def solve_all_neumann(tree: Octree, rows: Rows, source=None, pin_value=0.0,
                      sub: int = 4, patch: int = 2, device="cuda",
                      dtype=torch.float64, tol=1e-12, maxiter=100000,
                      neumann=None, verbose=False):
    """Run the paper's Section 7 procedure.

    ``rows`` must have been built with ``neumann=`` covering the whole boundary,
    so that every node carries a row.  ``source`` is the right-hand side ``f``
    of ``laplacian u = f`` as a tensor, or None for the Laplace equation.

    Returns a dict with the step-1 solution ``u``, the step-2 local solution
    ``u_local``, the gradient rows ``grad_row``, the uncorrected gradient
    ``grad_pinned`` and the step-3 patched gradient ``grad``.  Both gradients
    are returned so that the two columns of the paper's Table 9 -- before and
    after treating the localized error -- can be produced from one run.
    """
    if rows.boundary.any():
        raise ValueError(
            f"{int(rows.boundary.sum())} nodes have no row; build_rows must be "
            "called with neumann= covering the whole boundary for Section 7")

    pin = pin_node(rows, tree)
    dev = torch.device(device)

    # ---- step 1: pin one corner, solve the full problem
    pin_mask = np.zeros(rows.nnode, dtype=bool)
    pin_mask[pin] = True
    A = Operator(rows, dirichlet=pin_mask, device=dev, dtype=dtype)
    values = torch.zeros(rows.nnode, dtype=dtype, device=dev)
    values[pin] = pin_value
    u_a, info_a = solve(A, A.rhs(values, source=source), tol=tol,
                        maxiter=maxiter, verbose=verbose)

    # ---- step 2: re-solve on a small box at the corner, pin now free
    frozen, patch_nodes = subdomain(tree, rows, pin, sub, patch)
    B = Operator(rows, dirichlet=frozen, device=dev, dtype=dtype)
    u_b, info_b = solve(B, B.rhs(u_a, source=source), tol=tol,
                        maxiter=maxiter, verbose=verbose)

    # ---- step 3: patch the gradients near the corner
    G = Gradient(build_gradient(tree, neumann=neumann), device=dev, dtype=dtype)
    g_a = G(u_a)
    g_b = G(u_b)
    sel = torch.as_tensor(patch_nodes, device=dev)[G.row]
    g = torch.where(sel[:, None], g_b, g_a)

    return dict(u=u_a, u_local=u_b, grad_row=G.row, grad_pinned=g_a, grad=g,
                info_a=info_a, info_b=info_b, pin=pin,
                n_frozen=int(frozen.sum()), n_patched=int(patch_nodes.sum()))
