"""Matrix-free Krylov solvers in pure PyTorch.

Everything runs on the GPU with no host round-trips.  The composite operator is
non-symmetric (node-based sampling is not reflexive: the row for a node may
reference a neighbour whose own row does not reference it back), so CG is not
applicable and BiCGSTAB is used instead.

All inner products are accumulated in float64 regardless of the working dtype:
for 10^7-element reductions this is where accuracy is actually lost, and it
costs nothing on a bandwidth-bound problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SolveInfo:
    iters: int
    residual: float
    converged: bool
    history: list = field(default_factory=list)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Inner product accumulated in float64, returned in the working dtype."""
    return torch.sum(a.double() * b.double()).to(a.dtype)


def bicgstab(A, b, x0=None, M=None, tol=1e-10, maxiter=10000, verbose=False):
    """Left-Jacobi-preconditioned BiCGSTAB.

    ``A`` is any callable taking and returning a 1-D tensor.  ``M`` is the
    preconditioner diagonal (not its inverse); pass ``A.diag``.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    minv = None if M is None else 1.0 / M

    def prec(v):
        return v if minv is None else v * minv

    r = b - A(x)
    r0 = r.clone()
    bnorm = torch.sqrt(_dot(b, b))
    if float(bnorm) == 0.0:
        bnorm = torch.ones_like(bnorm)

    rho = alpha = omega = torch.ones((), dtype=b.dtype, device=b.device)
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    hist = []
    tiny = torch.finfo(b.dtype).tiny

    for it in range(1, maxiter + 1):
        rho_new = _dot(r0, r)
        if abs(float(rho_new)) < float(tiny):
            # Breakdown: restart against the current residual.
            r0 = r.clone()
            rho_new = _dot(r0, r)
            p = r.clone()
            if abs(float(rho_new)) < float(tiny):
                break
        else:
            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
        rho = rho_new

        ph = prec(p)
        v = A(ph)
        denom = _dot(r0, v)
        if abs(float(denom)) < float(tiny):
            break
        alpha = rho / denom
        s = r - alpha * v

        sh = prec(s)
        t = A(sh)
        tt = _dot(t, t)
        omega = _dot(t, s) / tt if float(tt) > float(tiny) else torch.zeros_like(tt)

        x = x + alpha * ph + omega * sh
        r = s - omega * t

        res = float(torch.sqrt(_dot(r, r)) / bnorm)
        hist.append(res)
        if verbose and (it % 50 == 0 or res < tol):
            print(f"  bicgstab {it:6d}  rel.res {res:.3e}")
        if res < tol:
            return x, SolveInfo(it, res, True, hist)
        if float(omega) == 0.0:
            break

    res = float(torch.sqrt(_dot(r, r)) / bnorm)
    return x, SolveInfo(len(hist), res, res < tol, hist)


def solve(A, b, tol=1e-10, maxiter=10000, refine_steps=0, verbose=False):
    """Solve ``A x = b``.

    With ``refine_steps > 0`` and a float32 operator, performs outer iterative
    refinement: the residual is formed in float64 and the correction solved in
    the working precision.  This recovers near-float64 accuracy at float32
    bandwidth, and is the intended path for large runs.  At the sizes this code
    targets, plain float64 is cheap enough that ``refine_steps=0`` is the
    default.
    """
    x, info = bicgstab(A, b, M=A.diag, tol=tol, maxiter=maxiter, verbose=verbose)
    for _ in range(refine_steps):
        r = (b.double() - A(x).double()).to(b.dtype)
        dx, info = bicgstab(A, r, M=A.diag, tol=tol, maxiter=maxiter, verbose=verbose)
        x = x + dx
    return x, info


def sweep(A, b, x, omega=1.0, n=1):
    """Damped Jacobi sweeps -- one pass over every node at every level at once.

    Far slower than Krylov for 3-D Laplace, but it is the plainest possible
    demonstration that the composite operator needs no level ordering, and it is
    useful for debugging the stencil in isolation.
    """
    dinv = omega / A.diag
    for _ in range(n):
        x = x + dinv * (b - A(x))
    return x
