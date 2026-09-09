# GPU single-pass FDM Laplace solver on a non-uniform octree (PyTorch)

## Context

We need the 3D drift field (∇²φ = 0, **E = −∇φ**) for a LArTPC with a **pixel**
anode. Pixel geometry demands fine resolution near the anode, and because the
pad-induced distortion extends a finite distance off the plane, the fine region
must be a genuine 3D box — not just a graded drift axis. The bulk drift region is
nearly uniform and needs only coarse resolution.

Priorities: **accuracy first** (E = −∇φ is the deliverable, so *gradient* accuracy
is what counts), then speed and GPU memory. PyTorch on 2× RTX 4090 (24 GB each,
sm_89, torch 2.10+cu128).

Two structural requirements drive the method choice:

1. **No meshing.** The anode is an axis-aligned rectangular pad array, so pad
   boundaries land exactly on grid lines when spacings divide the pitch. Geometry
   is a boolean mask, not a mesh. This is why FDM beats FEM here.
2. **One pass over the whole grid.** Explicitly *not* AMR-style level-by-level
   cycling (relax fine, relax coarse, interpolate, reflux, repeat). One composite
   operator over every node at every refinement level; one sweep touches
   everything at once. This rules out block-structured AMR (Chombo / AMReX /
   Martin & Cartwright), which is inherently level-sequential.

## Problem sizing — this drives everything

Geometry: LArPix / ND-LAr, **pitch 4.434 mm, drift 500 mm**. Resolution set by the
*inter-pad gap*, not the pitch, since the field singularity lives at the pad edge:
**h_min = gap/12**.

Graded octree (cell size grows with distance from the pad plane, capped at
pitch/8 in the bulk), node counts and solver-vector memory:

| gap | h_min | 1 pad (periodic) | 7×7 island | mem (7×7) |
|---|---|---|---|---|
| 0.5 mm | 42 µm | 0.20 M | 9.9 M | 0.26 GiB |
| 1.0 mm | 83 µm | 0.09 M | 4.6 M | 0.12 GiB |
| 2.2 mm | 183 µm | 0.06 M | 3.2 M | 0.08 GiB |

Two findings that shaped the design:

- **The coarse bulk is essentially free** — under 1% of nodes in every case. All
  the cost is in the near-anode slab.
- **Grading *within* the anode slab is where the win is.** For 7×7 pads at
  34.6 µm: uniform = 11.6 G nodes (302 GiB, impossible); naive two-block
  fine-slab + coarse-bulk = 311 M (8.1 GiB); **graded octree = 13 M (0.34 GiB)**.
  A two-array fine/coarse split captures only ~35× of a ~2000× available gain.
  This is exactly what the multi-level octree buys and what two arrays cannot
  express.

**Consequence: the problem is small.** ~10 M nodes ≈ 1 GB of memory traffic per
iteration ≈ 1 ms on a 4090; a few thousand Krylov iterations is a **seconds-long
solve**. GPU memory is a non-issue by two orders of magnitude, and we can afford
to refine well past the requirement.

### Recommended change: default to fp64, not mixed precision

Given the above, I recommend **fp64 throughout** as the default rather than the
fp32/fp64-refinement scheme. Reasoning: this is bandwidth-bound at ~1 FLOP/byte,
so Ada's 1/64 fp64 *arithmetic* rate is largely irrelevant — fp64 costs ~2×, not
64×. Two seconds versus one second is not worth the complexity or the risk to
accuracy, which is priority #1. The mixed-precision path stays in the plan
(`fdm/solve.py`, tests 5) and becomes worthwhile only if we later scale to a
full-plane run. **Flagging this as a deviation from the earlier choice — say the
word and I'll make mixed precision the default instead.**

## Discretization: Min–Gibou node-based scheme on a non-graded octree

**Reference:** C. Min, F. Gibou, H. Ceniceros, *"A supra-convergent finite
difference scheme for the variable coefficient Poisson equation on non-graded
grids,"* J. Comput. Phys. 218 (2006) 300–321.
Preprint: <https://web.math.ucsb.edu/~hdc/public/poisson_on_amr.pdf>
Companion: Chen, Min & Gibou, J. Sci. Comput. 31 (2007),
<https://link.springer.com/article/10.1007/s10915-006-9122-8>

Why this one:

- **Solution sampled at cell *nodes* (vertices), not centers.** This is the crux.
  Min et al. prove (Theorem 1) that *no* locally consistent linear cell-centered
  stencil exists on a non-uniform Cartesian grid using only adjacent cells —
  which is precisely why cell-centered AMR codes are forced into refluxing and
  level cycling. Node sampling escapes the theorem, and with it the requirement
  we're trying to avoid.
- **Non-graded**: no constraint on level jumps between adjacent cells, no buffer
  layers. Required for the aggressive grading in the table above.
- **Compact support:** a node's stencil touches its own cell plus at most 3
  adjacent cells in 3D. One row per node, no interface pass.
- **Second order in both φ and ∇φ.**

### The stencil (their Eq. 8, 3D)

At node `u0` with six direction-neighbours `u1..u6` at distances `s1..s6`, the
neighbours falling on a T-junction are replaced by interpolated ghosts:

    u4 = (u7*s8 + u8*s7) / (s8 + s7)                        # linear, on an edge
    u5 = (u11*s11*s12 + u12*s11*s9
          + u9*s10*s12 + u10*s10*s9) / ((s11+s10)*(s9+s12))  # bilinear, on a face

Differencing naively with those ghosts injects spurious `uxx`/`uzz` terms, which
are cancelled by weighting the three directional differences:

    [ 2/(s1+s4) * ( (u1-u0)/s1 + (u4-u0)/s4 ) ] * α
  + [ 2/(s2+s5) * ( (u2-u0)/s2 + (u5-u0)/s5 ) ]
  + [ 2/(s3+s6) * ( (u3-u0)/s3 + (u6-u0)/s6 ) ] * β
  = Δu + O(h)

    α = 1 − s10*s11 / (s5*(s2+s5))
    β = 1 − s9*s12 / (s5*(s2+s5)) − s7*s8 / (s4*(s1+s4))

On a locally uniform patch α = β = 1 and this collapses exactly to the standard
7-point Laplacian — the interior costs nothing extra. **This is what makes the
GPU hybrid below possible.**

**Properties.** M-matrix (diagonally dominant) provided cell anisotropy ≤ √2 → use
cubic cells. The matrix is **non-symmetric** (node sampling is not reflexive: the
row for `u0` may reference `u6` while `u6`'s row does not reference `u0`), so CG
is invalid — use BiCGSTAB/GMRES. Diagonal dominance guarantees Jacobi/SOR converge.

## GPU data structure: block-structured octree with halos

Not one cell per leaf. Each leaf is a **uniform tile of S³ cells** (S = 16 or 32,
tuned) at some refinement level, stored with a one-node halo:

    u : Tensor[B, S+2, S+2, S+2]     # B blocks, all levels, ONE tensor
    h : Tensor[B, 1, 1, 1]           # per-block spacing

This buys everything at once:

- **All levels live in one batched tensor.** A dense stencil applied to `u` with
  per-block `h` processes every block at every refinement level in a single
  batched op. That *is* the "all at once" pass — one kernel, no level ordering.
- **Coalesced access** and near-zero index storage for interior nodes, which are
  the overwhelming majority.
- Halo fill is one gather with a precomputed index/weight map, handling
  same-level neighbours *and* T-junction interpolation uniformly.

### One iteration = two kernels

1. `halo_fill(u)` — gather + weighted sum into halo nodes. Precomputed
   `idx[M,K] int32`, `w[M,K]`, with M = halo node count only (scales as N^(2/3)).
2. `apply_stencil(u, h)` — dense 7-point on all block interiors, batched over B.
   Interface-adjacent nodes (where α, β ≠ 1) are a small separate node list
   carrying their own Eq. 8 rows.

Both `torch.compile`-fusable; wrap the iteration in a **CUDA graph** to eliminate
launch overhead, which at these kernel sizes would otherwise dominate.

## Implementation

**Step 0 — before writing any code:** commit this plan verbatim to
`/nfs/data/1/rrazakami/work/Explore_FDM/PLAN.md` so it lives with the project and
is version-controllable, then implement against it.

New package `fdm/` in `/nfs/data/1/rrazakami/work/Explore_FDM`.

Write every stage as a **pure function of explicit tensors** (no hidden state, no
in-place ops on anything an adjoint would need). That is the cost of keeping the
autograd door open; a custom `torch.autograd.Function` using the adjoint solve can
then be added later without restructuring. Run the solve under `torch.no_grad()`
for now.

- **`fdm/tree.py`** — block octree. Leaves as a flat int32 tensor of
  `(level, i, j, k)`; all coordinates in integer units of `h_min`, so node
  identity is exact integer equality, never float comparison. `refine(predicate)`
  splits blocks. Production predicate: cell size ≈ max(h_min, dist_to_anode/8),
  plus max level on blocks touching a pad edge. No grading enforced.
- **`fdm/topology.py`** — build the halo gather map: for each halo node, source
  node indices and interpolation weights (same-level copy, or Min–Gibou edge/face
  interpolation at a T-junction). Built once on CPU with numpy, moved to GPU as
  int32/float tensors. Also emits the interface node list with per-node α, β.
- **`fdm/operator.py`** — `apply(u, h, halo_map, iface) -> Au`, the two kernels
  above. Flag `interface_correction: bool`; `False` forces α = β = 1, reproducing
  the naive/Losasso-style 1st-order-at-interface variant. **This flag is how we
  measure the accuracy tradeoff** — same code, same grid, two settings.
- **`fdm/geometry.py`** — pad array as a boolean Dirichlet mask built by integer
  slicing on pitch/pad-width; cathode and field-cage BCs likewise. Assert pitch,
  pad width and gap are integer multiples of h_min.
- **`fdm/solve.py`** — matrix-free BiCGSTAB in pure torch (no CPU round-trips, no
  scipy). fp64 default; `dtype`/`refine` options implement the fp32-storage +
  fp64-reduction + fp64-iterative-refinement path for the large-scale case. Also
  expose `sweep()` (damped Jacobi, `u ← ω(b − R u)/d + (1−ω)u`) for debugging —
  one pass over every node at every level, no level ordering.
- **`fdm/field.py`** — `E = −∇φ` at nodes using the same non-uniform differences
  and the same ghost interpolations, so the gradient inherits 2nd-order accuracy
  rather than being post-hoc differenced.

## Verification

`tests/`, in order:

1. **Uniform reduction.** On a fully uniform tree, assert the operator equals the
   standard 7-point Laplacian to machine precision and α = β = 1 everywhere.
   Catches sign/indexing errors before refinement is in play.
2. **Halo map correctness.** Apply the operator to an exactly-representable linear
   field `φ = ax + by + cz + d`; the result must be 0 to round-off at *every* node
   including T-junctions. Isolates topology bugs from discretization bugs — the
   single highest-value test.
3. **Analytic convergence, non-graded.** Impose a harmonic function (e.g.
   `x² + y² − 2z²`) as Dirichlet data; measure L∞ error in **both φ and ∇φ** vs
   h_min over a refinement sequence. Expect slope 2 for both. fp64.
4. **Quantify the interface tradeoff.** Repeat (3) with
   `interface_correction=False`. Report error **restricted to interface nodes**,
   not just the global norm — the global norm hides it, which is exactly what
   "supra-convergence" means. This is the measured number requested earlier.
5. **Precision study.** Repeat (3) in fp32-with-refinement; confirm it tracks the
   fp64 curve until discretization error dominates. Decides whether mixed
   precision is safe to enable for large runs.
6. **Physics — pad field decay.** Single pad, 4.434 mm periodic cell, transverse
   periodic BCs, 500 mm drift. Verify the transverse field ripple decays as
   exp(−2πz/pitch). Validates the solver *and* sizes the refined slab.
7. **Physics — weighting field.** 7×7 pad island (31 × 31 mm transverse),
   centre pad at 1 V, others grounded. Check convergence vs. island size by
   comparing to a 15×15 run (~60 M nodes, still only ~1.6 GiB).
8. **Benchmark.** Nodes/s, achieved GB/s vs the 4090's ~1 TB/s roofline, memory
   high-water mark, iterations to convergence vs problem size.

## Risks / open items

- **Need the actual LArPix pad dimension / inter-pad gap** from the PCB spec to
  fix h_min. Table above brackets it; I'll default to gap = 0.5 mm (the finest,
  most conservative case, 10 M nodes) until confirmed.
- **Solver scaling is the real risk, not memory.** Unpreconditioned Krylov on 3D
  Laplace needs O(N^(1/3)) iterations. At 10 M nodes the budget says seconds, so
  this should be fine. If it isn't, the level-free accelerator to try first is
  **Chebyshev-accelerated Jacobi** (matrix-free, maps perfectly to the GPU). A
  multigrid preconditioner is the fallback; note it would reintroduce level
  cycling *inside the preconditioner only* — the discretization stays a single
  level-free composite operator, so the design goal is preserved.
- **Non-symmetry** costs us CG. The symmetric alternative
  (Losasso–Gibou–Fedkiw, cell-centred) is only 1st order with degraded gradients —
  a bad trade when ∇φ *is* the answer. Test 4 decides on measured numbers.
- **Keep cells cubic.** Anisotropy > √2 loses the M-matrix property and with it
  the sweep convergence guarantee.
- **Second GPU unused.** At ~1 GiB per problem it is not needed; the block
  structure makes decomposition natural later if a full-plane run is wanted.
  (4090s have no NVLink — P2P over PCIe — so multi-GPU is only worth it at much
  larger scale.)
- Block size S (16 vs 32) trades halo overhead against refinement granularity;
  tune in test 8.
