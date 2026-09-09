# Results

Measured on 1× RTX 4090, torch 2.10+cu128, float64 unless stated.
Reproduce with `PYTHONPATH=. python3 tests/<name>.py`.

## Status against PLAN.md

| Test | What | Status |
|---|---|---|
| 1 | Uniform grid reduces to the 7-point Laplacian | pass (exact) |
| 2 | Linear field gives 0 at every node incl. T-junctions | pass (< 1e-9 relative) |
| 3 | Analytic convergence of φ and ∇φ on a non-graded grid | pass (1.98 / 1.90) |
| 4 | Cost of the interface correction | measured — see below |
| 5 | fp32 + refinement vs fp64 | measured — fp64 preferred at these sizes |
| 6 | Pad field decay exp(−2πz/pitch) | pass (0.14% on the exponent) |
| 7 | Weighting field, island-size convergence | pass (0.46% at 7×7) |
| 8 | Benchmark / scaling | done — see below |

Not yet implemented: the **block-structured halo fast path**. See "Deviations".

## Test 3 — convergence, non-graded octree (2-level jump)

| h_min | nodes | iters | L∞ err φ | order | L∞ err ∇φ | order |
|---|---|---|---|---|---|---|
| 6.25e-2 | 1 520 | 21 | 2.167 | – | 24.34 | – |
| 3.13e-2 | 10 287 | 40 | 0.582 | 1.90 | 8.99 | 1.44 |
| 1.56e-2 | 91 904 | 60 | 0.149 | 1.97 | 2.66 | 1.76 |
| 7.81e-3 | 677 182 | 122 | 0.0376 | **1.98** | 0.714 | **1.90** |

Second order in the solution *and* its gradient, which is the property the whole
method was chosen for. Verified separately with periodic transverse BCs (1.97).

## Test 4 — the interface correction is not the accuracy driver

L∞ error at fixed `lmax=5`, varying the coarse/fine level jump:

| jump | nodes | irregular | corrected | naive (α=β=1) |
|---|---|---|---|---|
| 1 | 112 683 | 3 008 | 3.753e-2 | 3.594e-2 |
| 2 | 91 904 | 3 744 | 1.485e-1 | 1.436e-1 |
| 3 | 72 311 | 3 920 | 5.819e-1 | 5.760e-1 |
| 4 | 71 900 | 3 960 | 2.166 | 2.153 |

**The correction changes the error by ~1%, at every jump ratio.** It is genuinely
active (912 interpolated nodes at `lmax=4`, transverse error products up to
4h²), and both variants are second order. The error is dominated by the *coarse
region's own cell size*, which grows as 4^jump, not by the interface treatment.
This is the supra-convergence argument holding in practice: interface nodes are
codimension-one and their contribution is absorbed.

**Design consequence:** grading should be set by local truncation error, not
only by "the pad field has decayed here". Doubling the cell size costs 4× in
local error everywhere, whereas the coarse/fine interface itself is nearly free.
The correction is kept on (it costs nothing — irregular rows are the same width
either way) but it is not something to design around.

## Test 6 — pad-induced field decay

Single pixel cell, transverse periodic, pitch 4.434 mm, pad 3.048 mm,
gap 1.386 mm, h_min 69.3 µm, 501 mm drift, 307 513 nodes, solve 1.1 s.

| z/pitch | ripple | exp(−2πz/p) | ratio |
|---|---|---|---|
| 0.20 | 3.70e-1 | 2.79e-1 | 1.33 |
| 0.50 | 6.18e-2 | 4.32e-2 | 1.43 |
| 1.00 | 2.72e-3 | 1.87e-3 | 1.46 |
| 1.50 | 1.20e-4 | 8.07e-5 | 1.49 |
| 2.00 | 4.88e-6 | 3.49e-6 | 1.40 |

Fitted decay constant **1.4190 /mm** vs 2π/pitch = 1.4170 /mm — **0.14% error**.
The flat ratio ≈1.45 across four decades is the Fourier amplitude of the square
pad, not an error.

**Sizing consequence:** the pad distortion is at 1e-3 by z = 1 pitch and 1e-5 by
2 pitches. A refined slab of **2–3 pitches (~9–13 mm)** is ample out of 500 mm
of drift.

## Test 7 — weighting field, island convergence

Centre pad at 1 V, h_min 138.6 µm, probe on-axis at z = pitch/4:

| island | nodes | iters | t (s) | φ_w | change |
|---|---|---|---|---|---|
| 3×3 | 513 385 | 302 | 0.23 | 6.782e-2 | – |
| 5×5 | 1 437 033 | 441 | 0.87 | 6.981e-2 | 2.94% |
| 7×7 | 2 817 329 | 571 | 2.33 | 7.013e-2 | **0.46%** |

Converging geometrically (~6× per step), so 7×7 is good to ~0.5% and 9×9 would
reach ~0.1%.

## Test 5 — precision

fp32 storage + fp64 reductions + 2 fp64 refinement steps reproduces the fp64
error to 4 significant figures (ratio 1.000) at every size — discretization
error dominates, as expected. But it is **slower** (0.65 s vs 0.11 s at 677 k
nodes) because of the extra refinement solves. **fp64 stays the default**, as
recommended in PLAN.md; the mixed path is available for runs large enough to be
memory-limited, which these are not.

## Test 8 — benchmark (single pixel cell, 501 mm drift)

| h_min | cells | nodes | irr% | assembly | iters | solve | ms/iter | GB/s | GPU |
|---|---|---|---|---|---|---|---|---|---|
| 277 µm | 8 233 | 8 577 | 4.7% | 0.0 s | 527 | 0.19 s | 0.37 | 8 | ~0 |
| 139 µm | 51 969 | 55 577 | 10.6% | 0.2 s | 1 309 | 0.44 s | 0.34 | 60 | 0.02 GiB |
| 69 µm | 284 929 | 307 513 | 13.1% | 1.2 s | 1 968 | 0.90 s | 0.46 | 261 | 0.09 GiB |
| 35 µm | 1 320 705 | 1 435 737 | 14.7% | 6.7 s | 771 | 1.73 s | 2.25 | 255 | 0.44 GiB |
| 17 µm | 5 682 433 | 6 202 745 | 15.6% | 61.8 s | 6 982 | 78.6 s | 11.25 | 224 | **1.89 GiB** |

Memory is a non-issue exactly as predicted: 6.2 M nodes at 17 µm resolution fits
in 1.9 GiB of 24 GiB.

## What the benchmark says to do next, in priority order

1. **Solver iteration count is the bottleneck, not bandwidth or memory.** Counts
   are erratic and grow badly (6 982 iterations at 6.2 M nodes). The driver is
   the ratio h_max/h_min (256:1 here), not N. This is the risk flagged in
   PLAN.md and it is now the measured limiter. Try Chebyshev-accelerated Jacobi
   first (matrix-free, level-free); a multigrid preconditioner is the fallback.
2. **Assembly is now comparable to the solve** (62 s vs 79 s at 6.2 M nodes) and
   is single-threaded CPU numpy. The neighbour probe is the hot loop and is
   `numba`-shaped.
3. **Bandwidth is 224–261 GB/s of the 4090's ~1 TB/s.** The block-structured
   halo fast path should recover most of the remaining ~4×.

## Deviations from PLAN.md

**The operator is a flat node list, not the block-structured halo layout.**

Rows are emitted in two padded gather-reduce blocks — `regular` (7 entries) and
`irregular` (25 entries) — rather than as batched `[B,S+2,S+2,S+2]` tiles.

Reasoning: the plan's block/halo design was justified when the budget assumed
~10^8 nodes. Measured node counts are ~10^7 and peak memory is 1.9 GiB of 24
GiB, so the block layout's benefit is a ~3–4× bandwidth constant, not a
feasibility requirement. The node list is far simpler to prove correct and is
needed anyway as the reference implementation to validate a blocked version
against. It satisfies the actual requirement — one pass over every node at
every level, no level ordering — identically.

The block fast path remains worth doing for point 3 above, and `Operator` is
already split into regular/irregular sets, which is the same partition a blocked
implementation needs.

**Also generalized:** rather than Min & Gibou's α/β (derived for the specific
configurations in their figures), the transverse-error cancellation is obtained
by solving a 3×3 system `Mᵀw = 1` per node. This reduces to their weights in
their configurations, needs no case analysis, and additionally covers nodes
where more than one direction is interpolated. See `fdm/topology.py`.

---

# End-to-end physics verification

`tests/test_drift.py`, `test_sumrule.py`, `test_supercell.py`,
`test_charge_conservation.py`.

Geometry: 5×5 pixel supercell, transverse periodic (so the pad array is
effectively infinite), pitch 4.434 mm, pad 3.048 mm, gap 1.386 mm,
h_min 138.6 µm, 53.2 mm drift at 500 V/cm. 1 420 836 nodes.

The anode plane carries conducting pads (Dirichlet) separated by bare dielectric
(**homogeneous Neumann**). The Neumann gap is essential: without it the anode is
an equipotential and the pixel structure does not act on the drift field at all,
so there is nothing to verify. Two solves on the same grid, ~1.05 s each:
the drift field (all pads 0 V, cathode at −E₀L) and the weighting field (centre
pad 1 V, all others and the cathode 0 V).

## Drift velocity

ICARUS parameterization (Walkowiak, NIM A 449 (2000) 288) at 87.3 K:

| E (kV/cm) | 0.10 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|
| v (mm/µs) | 0.520 | 1.085 | **1.596** | 2.116 |

1.596 mm/µs at 500 V/cm is the standard literature value.

## Field and 10×10 drift paths

| quantity | result |
|---|---|
| bulk \|E\| on axis | 496.8 V/cm (target 500) |
| \|E\| at z = pitch/4 | 400 V/cm — pad focusing suppresses the near field |
| paths tracked | 100 in 2.0 s, 130–338 RK4 steps each |
| all reached the anode | yes |
| drift time | 5.1 – 31.5 µs |
| mean drift speed | 1.5895 mm/µs vs 1.5964 expected (**0.43%**) |
| speed spread | 1.29% |
| transverse focusing | mean 310 µm, max 771 µm |
| landed on centre pad | 5 of 100 (expected 1/25 = 4) |

The focusing displacement is the physically meaningful number: electrons
starting above a gap are pulled up to ~0.77 mm sideways onto the nearest pad,
which is about half the 1.386 mm gap, exactly as it should be. This is the
effect that only exists because the near-anode region is resolved in 3-D.

## Shockley–Ramo induced current

Induced charge computed by two independent routes — the Ramo integral
∫ q v·E_w dt (which differentiates the weighting field) and the weighting
*potential* difference −q[φ_w(x)−φ_w(x₀)] (which does not):

| check | result |
|---|---|
| \|Q_ramo − Q_φ\| / \|Q_φ\| | median **4.9e-3** |
| \|Q_ramo − Q_φ\| / e | median 2.4e-6 |
| collected on centre pad | Q/e = 0.9656 ± 0.036 |
| neighbour pads | bipolar, \|Q\|/e ≤ 4.7e-2, nets to ~0 |
| peak current on centre pad | 0.13 – 0.46 pA |

Sign convention matters and was initially wrong: Q_k = −q[φ_w(x) − φ_w(x₀)] and
i_k = dQ/dt = q v·E_w with E_w = −∇φ_w, so for an electron (q = −e)
Q = +e[φ_w − φ_w₀] but **i = −e v·E_w**. A median relative disagreement of
exactly 2.00 is the signature of that flip.

## The Q/e = 0.966 deficit is a finite-supercell artifact, not an error

The deficit is **resolution-independent** (5.50e-2 at h = 277, 139 and 69 µm —
ratio 1.00), so it is not interpolation. It is φ_w at the electron's *starting*
point being nonzero: in an N×N periodic supercell one "one-hot" pad necessarily
carries 1/N² of the uniform (k = 0) transverse mode, and that mode is not
screened by the pad structure — it decays *linearly* to the grounded cathode
rather than as exp(−2πz/pitch).

| N | φ_w(z = L/2) measured | (1/N²)(1 − z/L) |
|---|---|---|
| 3 | 0.05520 | 0.05556 (0.6%) |
| 5 | 0.02089 | 0.02000 |
| 7 | 0.01274 | 0.01020 |
| 9 | 0.01019 | 0.00617 |

The k = 0 formula is accurate at N = 3–5 where that mode dominates; at larger N
the supercell's own lowest transverse harmonic (decay length N·pitch/2π, which
approaches the drift length) also contributes. Both are supercell artifacts. In
a real detector one pad among millions has k = 0 weight ≈ 0 and Q/e → 1.

## Sum rule — the decisive check

Setting every electrode to 1 V makes φ ≡ 1 (it solves Laplace and satisfies all
BCs including homogeneous Neumann on the gaps), so by superposition

    φ_w,cathode(x) + Σ_all pads φ_w,pad(x) = 1   for every x

Evaluated at 400 random points per configuration, using periodicity to get the
pad sum from N² translations of one solve:

| N | nodes | max \|Σ − 1\| | mean \|Σ − 1\| |
|---|---|---|---|
| 3 | 503 808 | 1.25e-4 | 5.3e-6 |
| 5 | 1 420 736 | 5.62e-5 | 2.1e-6 |
| 7 | 2 794 232 | 6.51e-5 | 3.0e-6 |

This is the strongest available statement: it simultaneously validates the
composite operator, the Neumann mirroring, the periodic wrapping, the
coarse/fine interface treatment and the trilinear interpolation. Nothing in the
chain can be wrong at more than the 1e-4 level.

`drift_verification.png` shows the drift paths with the pads marked, and the
induced-current waveforms for collected and neighbouring pads.

## Known limitation

The interpolated field is only C⁰ *within* a refinement level: across a
coarse/fine face the coarse cell's trilinear form and the fine cells' do not
agree exactly, because hanging-node values are solved degrees of freedom rather
than interpolants of the coarse face. The mismatch is O(h²) and appears as a
small kink in a drift path crossing a refinement boundary. It is bounded by the
sum-rule result above and did not measurably affect drift speed (0.43%) or
induced charge (4.9e-3).
