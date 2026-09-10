"""Drift field E = -grad(phi) evaluated with the discretization's own gradient.

The gradient reuses the cell C, the neighbours and the interpolants of the
Laplacian rather than differencing the solution afterwards, and it applies the
paper's Section 8 correction for the interpolation error, so it is exact for
quadratics at every node rather than degrading to first order at hanging nodes.
E is the deliverable for drift simulation, so this matters more than the
accuracy of phi itself.
"""

from __future__ import annotations

import numpy as np
import torch

from .topology import GradRows, build_gradient
from .tree import Octree


class Gradient:
    """GPU gradient operator: a padded gather-reduce block per axis, plus the
    Section 8 interpolation correction on the rows that need it."""

    def __init__(self, grad: GradRows, device="cuda", dtype=torch.float64):
        self.n = grad.nnode
        self.device = torch.device(device)
        self.dtype = dtype

        def T(a, d):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=d,
                                   device=self.device)

        self.row = T(grad.row, torch.int64)
        self.idx = T(grad.idx, torch.int64)
        self.val = T(grad.val, dtype)
        self.corr_pos = T(grad.corr_pos, torch.int64)
        self.corr_idx = T(grad.corr_idx, torch.int64)
        self.corr_val = T(grad.corr_val, dtype)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """Return grad(u) at every interior node, shape ``(len(row), 3)``."""
        cols = []
        for a in range(3):
            g = (self.val[a] * u[self.idx[a]]).sum(1)
            if self.corr_pos.numel():
                g = g.index_add(
                    0, self.corr_pos,
                    (self.corr_val[a] * u[self.corr_idx[a]]).sum(1))
            cols.append(g)
        return torch.stack(cols, dim=1)

    def efield(self, u: torch.Tensor) -> torch.Tensor:
        """E = -grad(phi) at every interior node, shape ``(len(row), 3)``."""
        return -self(u)


def efield(tree: Octree, u: torch.Tensor, device=None, neumann=None):
    """Convenience wrapper: build the gradient and evaluate E at interior nodes.

    Returns ``(node_ids, E)``.  Node ids index into the solution vector, so
    positions come from ``tree.node_positions(rows.coords[node_ids])``.
    """
    g = Gradient(build_gradient(tree, neumann=neumann),
                 device=device or u.device, dtype=u.dtype)
    return g.row, g.efield(u)
