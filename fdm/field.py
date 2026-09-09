"""Drift field E = -grad(phi) evaluated with the discretization's own gradient.

The gradient reuses the neighbours and ghost interpolants of the Laplacian
rather than differencing the solution afterwards, so it inherits the scheme's
accuracy at hanging nodes instead of degrading there.  E is the deliverable for
drift simulation, so this matters more than the accuracy of phi itself.
"""

from __future__ import annotations

import numpy as np
import torch

from .topology import GradRows, build_gradient
from .tree import Octree


class Gradient:
    """GPU gradient operator: three padded gather-reduce blocks."""

    def __init__(self, grad: GradRows, device="cuda", dtype=torch.float64):
        self.n = grad.nnode
        self.device = torch.device(device)
        self.dtype = dtype
        self.row = torch.as_tensor(grad.row, dtype=torch.int64, device=self.device)
        self.idx = torch.as_tensor(grad.idx, dtype=torch.int64, device=self.device)
        self.val = torch.as_tensor(np.ascontiguousarray(grad.val),
                                   dtype=dtype, device=self.device)

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        """Return grad(u) at every interior node, shape ``(len(row), 3)``."""
        return torch.stack(
            [(self.val[a] * u[self.idx[a]]).sum(1) for a in range(3)], dim=1)

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
