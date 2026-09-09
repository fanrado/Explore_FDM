"""The composite Laplacian as a matrix-free PyTorch operator.

One call to :meth:`Operator.__call__` applies the operator to *every* node at
*every* refinement level simultaneously -- two gather-multiply-reduce kernels
and one masked write.  There is no level ordering, no coarse/fine sequencing and
no interface pass; the refinement hierarchy is entirely absorbed into the
precomputed index and coefficient tensors.
"""

from __future__ import annotations

import numpy as np
import torch

from .topology import Rows


class Operator:
    """Matrix-free composite Laplacian with Dirichlet rows folded in.

    Dirichlet nodes carry an identity row, so ``A @ u = b`` with ``b`` holding
    the prescribed values there imposes the boundary condition exactly.
    """

    def __init__(self, rows: Rows, dirichlet=None, device="cuda",
                 dtype=torch.float64, idx_dtype=torch.int32):
        self.n = rows.nnode
        self.device = torch.device(device)
        self.dtype = dtype
        # int32 indices halve index traffic; torch needs int64 to index, so we
        # keep int64 when the node count would overflow int32 anyway.
        use32 = idx_dtype == torch.int32 and self.n < 2**31
        ix = torch.int32 if use32 else torch.int64

        def T(a, d):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=d, device=self.device)

        self.reg_row = T(rows.reg_row, torch.int64)
        self.reg_idx = T(rows.reg_idx, ix)
        self.reg_val = T(rows.reg_val, dtype)
        self.irr_row = T(rows.irr_row, torch.int64)
        self.irr_idx = T(rows.irr_idx, ix)
        self.irr_val = T(rows.irr_val, dtype)

        if dirichlet is None:
            dirichlet = rows.boundary
        self.dirichlet = T(np.asarray(dirichlet, bool), torch.bool)

        # Free nodes are those with a real row and no prescribed value.  Any
        # node that is neither (an orphan) would make the system singular.
        has_row = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        has_row[self.reg_row] = True
        has_row[self.irr_row] = True
        orphan = ~(has_row | self.dirichlet)
        if bool(orphan.any()):
            raise ValueError(f"{int(orphan.sum())} nodes have neither a stencil "
                             "row nor a Dirichlet value")
        self.free = ~self.dirichlet
        self._diag = self._build_diag()

    def _build_diag(self) -> torch.Tensor:
        d = torch.ones(self.n, dtype=self.dtype, device=self.device)
        d[self.reg_row] = self.reg_val[:, 0]
        if self.irr_row.numel():
            d[self.irr_row] = self.irr_val[:, 0]
        d[self.dirichlet] = 1.0
        return d

    @property
    def diag(self) -> torch.Tensor:
        """Main diagonal; varies by orders of magnitude across levels, so it is
        the natural (and near-essential) Jacobi preconditioner here."""
        return self._diag

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(u)
        out.scatter_(0, self.reg_row,
                     (self.reg_val * u[self.reg_idx.long()]).sum(1))
        if self.irr_row.numel():
            out.scatter_(0, self.irr_row,
                         (self.irr_val * u[self.irr_idx.long()]).sum(1))
        return torch.where(self.dirichlet, u, out)

    def rhs(self, values: torch.Tensor, source=None) -> torch.Tensor:
        """Right-hand side for ``A u = b``: Dirichlet data where prescribed,
        the source term (zero for Laplace) elsewhere."""
        b = torch.zeros(self.n, dtype=self.dtype, device=self.device) \
            if source is None else source.clone()
        return torch.where(self.dirichlet, values, b)
