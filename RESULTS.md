# Results

Measured on 1× RTX 4090, torch 2.10+cu128, float64 unless stated.
Reproduce with `PYTHONPATH=. python3 tests/<name>.py`.

## Status against PLAN.md

| Test | What | Status |
|---|---|---|
| 1 | Uniform grid reduces to the 7-point Laplacian | pass (exact) |
| 2 | Linear field gives 0 at every node incl. T-junctions | pass (< 1e-9 relative) |
| 3 | Analytic convergence of φ and ∇φ on a non-graded grid | pass (1.98 / 1.90) |
| 4 | What the Eq. (8) cancellation buys | measured — see below |
| 5 | fp32 + refinement vs fp64 | measured — fp64 preferred at these sizes |
| 6 | Pad field decay exp(−2πz/pitch) | pass (0.19% on the exponent) |
| 7 | Weighting field, island-size convergence | pass (0.46% at 7×7) |
| 8 | Benchmark / scaling | done — see below |

**Conformance to the source paper** (Min, Gibou & Ceniceros 2006) is verified
separately — see the last section. Those three tests assert and gate; everything
in this table is print-and-inspect.

Not yet implemented: the **block-structured halo fast path**. See "Deviations".

## Test 3 — convergence, non-graded octree (2-level jump)

| h_min | nodes | iters | L∞ err φ | order | L∞ err ∇φ | order |
|---|---|---|---|---|---|---|
| 6.25e-2 | 1 520 | 21 | 2.167 | – | 24.34 | – |
| 3.13e-2 | 10 287 | 42 | 0.582 | 1.90 | 8.99 | 1.44 |
| 1.56e-2 | 91 904 | 60 | 0.149 | 1.97 | 2.66 | 1.76 |
| 7.81e-3 | 677 182 | 210 | 0.0376 | **1.98** | 0.714 | **1.90** |

Second order in the solution *and* its gradient, which is the property the whole
method was chosen for. Verified separately with periodic transverse BCs (1.97).

The gradient order here is limited by the *coarse* region's cell size, not by the
interface treatment — see "Section 9.1.6" in the conformance section, where the
same effect is isolated and measured against a uniform grid.

## Test 4 — what the paper's Eq. (8) cancellation buys

`tests/test_interface_correction.py`. Fixed `lmax=5`, varying the coarse/fine
level jump. `interface_correction=False` forces the paper's weights to 1, the
Losasso-style scheme.

**Local consistency** — residual on quadratics, where Eq. (1) and the multilinear
interpolation are both exact, so a consistent scheme must return the Laplacian to
round-off:

| jump | corrected (Eq. 8) | naive (w = 1) |
|---|---|---|
| 1 (2×) | 5.5e-12 | 3.33e-1 |
| 2 (4×) | 4.5e-12 | 4.00e-1 |
| 3 (8×) | 3.2e-12 | 4.44e-1 |
| 4 (16×) | 4.5e-12 | 4.71e-1 |

The naive scheme is inconsistent by **O(1)**, and refining does not reduce it.
Eq. (8) is exact.

**Error on one solution** — the harmonic `sin(πx)sin(πy)exp(√2πz)`:

| jump | nodes | irr | L∞ err φ | naive | err @iface | naive | L∞ err ∇φ | naive |
|---|---|---|---|---|---|---|---|---|
| 1 | 112 683 | 3 008 | 3.753e-2 | 3.594e-2 | 1.401e-2 | 1.476e-3 | 7.135e-1 | 7.198e-1 |
| 2 | 91 904 | 3 744 | 1.485e-1 | 1.437e-1 | 5.545e-2 | 1.679e-2 | 2.656e+0 | 2.673e+0 |
| 3 | 72 311 | 3 920 | 5.819e-1 | 5.760e-1 | 1.661e-1 | 1.070e-1 | 8.988e+0 | 9.012e+0 |
| 4 | 71 900 | 3 960 | 2.166e+0 | 2.153e+0 | 5.775e-1 | 4.552e-1 | 2.434e+1 | 2.443e+1 |

The L∞ error over the whole grid is dominated by the coarse region's own cell
size, which grows as 4^jump, not by the interface treatment — interface nodes are
codimension-one, which is the supra-convergence argument holding in practice.

**A correction to the previous write-up.** This section used to conclude, from
the ~1% whole-grid column alone, that the correction "is not something to design
around". That was wrong. For this particular harmonic the naive scheme's
inconsistency happens to cancel favourably *at interface nodes*, so its error
there is smaller — which the old text read as the correction being nearly
irrelevant. The consistency table above is what rules that coincidence out: an
O(1) locally inconsistent operator is not a scheme you can reason about from one
test function. Eq. (8) is kept because it is the paper's scheme and because it is
consistent, not because it wins on this solution.

**Design consequence** (unchanged): grading should be set by local truncation
error, not only by "the pad field has decayed here". Doubling the cell size costs
4× in local error everywhere, whereas the coarse/fine interface itself is cheap.

## Test 6 — pad-induced field decay

Single pixel cell, transverse periodic, pitch 4.434 mm, pad 3.048 mm,
gap 1.386 mm, h_min 69.3 µm, 501 mm drift, 307 513 nodes, solve 0.52 s.

| z/pitch | ripple | exp(−2πz/p) | ratio |
|---|---|---|---|
| 0.20 | 3.70e-1 | 2.79e-1 | 1.33 |
| 0.50 | 6.18e-2 | 4.32e-2 | 1.43 |
| 1.00 | 2.72e-3 | 1.87e-3 | 1.46 |
| 1.50 | 1.20e-4 | 8.07e-5 | 1.49 |
| 2.00 | 4.88e-6 | 3.49e-6 | 1.40 |

Fitted decay constant **1.4198 /mm** vs 2π/pitch = 1.4170 /mm — **0.19% error**.
The flat ratio ≈1.45 across four decades is the Fourier amplitude of the square
pad, not an error.

**Sizing consequence:** the pad distortion is at 1e-3 by z = 1 pitch and 1e-5 by
2 pitches. A refined slab of **2–3 pitches (~9–13 mm)** is ample out of 500 mm
of drift.

## Test 7 — weighting field, island convergence

Centre pad at 1 V, h_min 138.6 µm, probe on-axis at z = pitch/4:

| island | nodes | iters | t (s) | φ_w | change |
|---|---|---|---|---|---|
| 3×3 | 513 385 | 302 | 0.23 | 6.78184e-2 | – |
| 5×5 | 1 437 033 | 476 | 0.89 | 6.98093e-2 | 2.94% |
| 7×7 | 2 817 329 | 511 | 1.93 | 7.01299e-2 | **0.46%** |

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

`tests/test_benchmark.py`, which defaults to the first four rows; the 17 µm row
needs `FDM_BENCH_LMAX=8`. The bandwidth column uses an explicit model stated in
that file (4-byte index + 8-byte coefficient + 8-byte gathered value per stored
entry, two matvecs per iteration); it is an upper bound on useful traffic, not a
hardware counter.

| h_min | cells | nodes | irr% | nnz | assembly | iters | solve | ms/iter | GB/s | GPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 277 µm | 8 233 | 8 577 | 5.1% | 0.1 M | 0.0 s | 405 | 0.16 s | 0.39 | 7 | ~0 |
| 139 µm | 51 969 | 55 577 | 11.3% | 0.4 M | 0.1 s | 1 448 | 0.48 s | 0.33 | 55 | 0.02 GiB |
| 69 µm | 284 929 | 307 513 | 13.4% | 2.5 M | 0.8 s | 1 163 | 0.49 s | 0.42 | 251 | 0.09 GiB |
| 35 µm | 1 320 705 | 1 435 737 | 14.9% | 11.9 M | 4.4 s | 1 189 | 2.38 s | 2.01 | 248 | 0.41 GiB |
| 17 µm | 5 682 433 | 6 202 745 | 15.7% | 51.7 M | 45.0 s | 2 737 | 27.7 s | 10.13 | 214 | **1.78 GiB** |

Memory is a non-issue exactly as predicted: 6.2 M nodes at 17 µm fits in 1.78 GiB
of 24 GiB.

The paper's construction is materially cheaper than the superseded stencil at the
top end — at 6.2 M nodes, assembly 45.0 s vs 61.8 s (8 `locate` probes per node
instead of 24), solve 27.7 s vs 78.6 s on 2 737 iterations instead of 6 982, and
peak memory 1.78 GiB vs 1.89 GiB. The iteration-count drop is the largest single
effect and was not anticipated.

## What the benchmark says to do next, in priority order

1. **Solver iteration count is still the bottleneck, and still the thing to
   attack.** It is much better than before (2 737 rather than 6 982 at 6.2 M
   nodes) but still grows non-monotonically with h_max/h_min (256:1 here) rather
   than with N — 405, 1 448, 1 163, 1 189, 2 737. This remains the risk flagged
   in PLAN.md. Try Chebyshev-accelerated Jacobi first (matrix-free, level-free);
   a multigrid preconditioner is the fallback.
2. **Assembly now dominates at the top end** (45 s vs 28 s of solve at 6.2 M
   nodes) and is single-threaded CPU numpy. The eight octant probes and the
   per-octant alignment test in `stencil()` are the hot loops and are
   `numba`-shaped, or could be chunked over nodes to cut the transient arrays.
3. **Bandwidth is 214–251 GB/s of the 4090's ~1 TB/s.** The block-structured
   halo fast path should recover most of the remaining ~4×.

## Deviations from PLAN.md

**The operator is a flat node list, not the block-structured halo layout.**

Rows are emitted in two padded gather-reduce blocks — `regular` (7 entries) and
`irregular` (16 entries) — rather than as batched `[B,S+2,S+2,S+2]` tiles.

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

**Superseded:** an earlier version used a symmetric six-direction stencil, each
direction independently taking the finest leaf on its own side, and described
the `Mᵀw = 1` weight solve as a generalization of Min & Gibou's α/β. The stencil
has since been rewritten to the paper's own cell-anchored construction, and the
weight solve is now known to *be* their α/β rather than a generalization of it.
See "Conformance to Min, Gibou & Ceniceros (2006)" below.

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

---

# Conformance to Min, Gibou & Ceniceros (2006)

`poisson_on_amr.pdf` — C. Min, F. Gibou, H.D. Ceniceros, *J. Comput. Phys.* 218
(2006) 300–321. The scheme is now implemented as the paper constructs it, rather
than as an equivalent-looking variant. Reproduce with
`PYTHONPATH=. python3 tests/test_paper.py`, `tests/test_paper_convergence.py`,
`tests/test_paper_neumann.py` — unlike the rest of `tests/`, these three assert
and exit non-zero on failure, and run on the CPU if no GPU is present.

## What changed

**The stencil is now asymmetric, as in the paper's Figs. 6 and 7.** A node `u0`
is a *corner* of a chosen leaf `C`. The three directions pointing into `C` take
`C`'s adjacent corners — always real nodes, at `s1 = s2 = s3 = h_C`. The three
pointing away take the aligned point on the far face of `C`'s axis-neighbours
`Cx/Cy/Cz`, at `s4/s5/s6`, multilinearly interpolated. The previous code treated
all six directions symmetrically and independently.

`C` is the finest leaf incident to the node that is *usable*: the node must be a
corner of it and all three axis-neighbours must exist. The paper does not specify
which incident cell to use; the corner requirement is the paper's, and it is what
makes an edge-aligned outward direction available at every node. Dropping it
produces ~0.01% of production nodes (on the doubly-periodic seam at `x=y=0`) with
no clean direction, a configuration the paper's α/β cannot express.

**Eight octant probes replace twenty-four.** The same eight `tree.locate` calls
give both `C` and all three `C_a`, so the rewrite is also a simplification.

**Row width drops from 25 to 16** (padded; the paper's structural bound on the
support is 11, and 11 is what is measured). Regular rows stay 7.

**The α/β erratum.** The paper's Eq. (8) prints
`α = 1 − s10·s11/(s5(s2+s5))` and `β = 1 − s9·s12/(s5(s2+s5)) − α·s7·s8/(s4(s1+s4))`.
Read against Eq. (6), Fig. 7's geometry and the `u5` interpolation formula —
where `{s9,s12}` are the x-offsets and `{s10,s11}` the z-offsets of `u5` on
`Cy`'s far face — those two products are interchanged. The consistent values are
`α = 1 − s9·s12/(s5(s2+s5))`, `β = 1 − s10·s11/(s5(s2+s5)) − α·s7·s8/(s4(s1+s4))`,
which is what the `Mᵀw = 1` solve returns.

**Section 8 is now applied.** Its correction term was previously omitted. It must
use the *decontaminated* second differences `u_vec = M⁻¹ D` — what the paper means
by "the finite differences for uxx, uyy and uzz are given in Section 5.3". Using
the raw `D_t` is correct only where direction `t` is itself uninterpolated, and
leaves an O(h) residual where two directions are both interpolated.

**Section 7** (all-Neumann three-step procedure) is implemented in
`fdm/neumann.py`. It is gated on the absence of any Dirichlet node and so is
dormant in production, where the pads and cathode are Dirichlet.

**Still not implemented: Section 6**, the variable coefficient `∇·(ρ∇u) = f`. The
solver is constant-coefficient throughout.

## The exact-identity test

Eq. (1) is exact for quadratics, and a multilinear interpolation reproduces a
quadratic with error exactly `(p·q/2)·u_tt`, so the paper's cancellation must be
exact to round-off. This is sharper than any convergence study and would have
caught the transposed α/β products.

`Σ_a w_a D_a u = ∇²u` and, with Section 8, `∇u` exactly, for all ten quadratic
monomials, over uniform / graded / non-graded / deep-corner / three randomly
refined / periodic / production meshes:

| | worst relative error |
|---|---|
| Laplacian, all meshes | 1.3e-12 |
| gradient, all meshes | 1.1e-14 |

Only the randomly refined meshes produce two simultaneously interpolated
directions, and they are what exposed the `M⁻¹` requirement in Section 8: with
the raw `D_t` the gradient error there was 2.1e-3, not round-off.

## Structure

Every node has at least one edge-aligned outward direction; the contamination
pattern is always triangular; all three inward distances equal `h_C`; no `C_a` is
finer than `C`; maximum support is 11 — the paper's exact bound — on every mesh
above. Outward support multisets, production grid (74 344 interior nodes):

| `(nsup)` | nodes | share |
|---|---|---|
| (1,1,1) | 71 128 | 95.7% |
| (1,1,2) | 1 520 | 2.0% |
| (1,1,4) | 1 648 | 2.2% |
| (1,2,2) | 16 | 0.02% |
| (1,2,4) | 32 | 0.04% |

## Section 9.1.6 — 3D Dirichlet, Ω=[0,1]³, u = exp(xyz)

Non-graded grid, 4× cell-size jump. The paper's mesh (its Fig. 14) cannot be
recovered, so absolute values are not reproducible; the paper's column is shown
for scale only.

| h_min | nodes | iters | L∞ err φ | order | L∞ err ∇φ | order | (paper φ) | (paper ∇φ) |
|---|---|---|---|---|---|---|---|---|
| 6.25e-2 | 1 520 | 44 | 6.97e-4 | – | 8.74e-3 | – | 3.22e-3 | 5.82e-2 |
| 3.13e-2 | 10 287 | 90 | 1.61e-4 | 2.11 | 2.43e-3 | 1.85 | 7.03e-4 | 1.73e-2 |
| 1.56e-2 | 91 904 | 185 | 4.47e-5 | 1.85 | 9.88e-4 | 1.30 | 1.82e-4 | 4.75e-3 |
| 7.81e-3 | 677 182 | 395 | 1.04e-5 | **2.10** | 3.29e-4 | 1.59 | 4.47e-5 | 1.24e-3 |

Solution errors are below the paper's at every resolution.

**Why the gradient order is not 2 — for the paper either.** The paper's own
Table 8 gradient orders are 1.75/1.87/1.93/**1.64**. For `exp(xyz)` on the unit
cube the third derivatives peak at the corner (1,1,1), and the interior node
attaining the L∞ gradient error creeps toward that corner as h shrinks, so the
constant in `(h²/6)u'''` grows with refinement. A *uniform* grid shows the same
thing: order 1.59 → 1.80 → 1.90, rising toward 2.

The gradient error is set by the **coarsest** cells, not by `h_min`, and the
non-graded and uniform sequences coincide exactly:

| h_min | non-graded (4× jump) | uniform at 4×h | ratio |
|---|---|---|---|
| 1.56e-2 | 1.0077e-3 | 1.0077e-3 | **1.0000** |
| 7.81e-3 | 3.3395e-4 | 3.3395e-4 | **1.0000** |

So non-graded refinement costs nothing in gradient accuracy beyond what the
coarse cell size already implies. That equality is what the test asserts, rather
than a fitted order threshold.

## Section 9.1.7 — 3D all-Neumann, Ω=[0,π]³, u = cos x cos y cos z − 1

| h_min | nodes | it₁ | it₂ | L∞ err φ | order | L∞ err ∇φ | order | §7 change |
|---|---|---|---|---|---|---|---|---|
| 3.93e-1 | 318 | 45 | 18 | 5.18e-2 | – | 7.97e-2 | – | 2.8e-11 |
| 1.96e-1 | 1 931 | 93 | 21 | 1.41e-2 | 1.88 | 2.19e-2 | 1.87 | 9.9e-12 |
| 9.82e-2 | 15 158 | 197 | 16 | 3.47e-3 | 2.02 | 6.14e-3 | 1.83 | 7.6e-11 |
| 4.91e-2 | 112 683 | 369 | 16 | 8.64e-4 | **2.01** | 1.53e-3 | **2.01** | 2.3e-9 |

**The paper's Table 9 is not reproduced, and the reason is measured rather than
guessed.** Table 9's point is that the pinned solution's gradient *stalls*
(orders 2.82/0.22/0.18/0.11) until Section 7 repairs it. That stall does not
occur here. The paper never states how it discretizes `∂u/∂n = 0`; this code uses
the mirrored stencil, whose rows sum to zero, so:

- the operator annihilates constants **exactly** (`max|A·1| = 0.00e+00`);
- the discrete RHS is therefore compatible, and the pinned solution satisfies the
  **full unpinned system including the pinned node's own row** (relative residual
  ≤ 1e-3 × the solution error on every grid);
- so the solution differs from the unpinned family only by the constant the pin
  selects, and a constant has zero gradient.

The pin therefore cannot corrupt the gradient anywhere, and Section 7's step-3
patch is provably a no-op — asserted directly (last column above), which is a
stronger statement than "the stall did not appear". Steps 1–3 all still execute,
and the result is independent of the subdomain size (`sub`=3/4/6, `patch`=1/2/3
agree to 1e-6). A Neumann discretization whose rows did not annihilate constants
would reintroduce the paper's defect, and `fdm/neumann.py` would then repair it.

## Differences from the previous stencil

The previous implementation was **not** the paper's method, so agreement with it
is not evidence of correctness and disagreement is not a regression. This table
records what moved, for orientation only. Correctness is established by the exact
identities and structural checks above, not by this column.

| check | superseded scheme | paper's construction |
|---|---|---|
| uniform grid → 7-point Laplacian | exact | exact |
| convergence of φ (`test_convergence`) | 1.90/1.97/1.98 | 1.90/1.97/1.98 |
| sum rule, max \|Σ−1\| (N=3/5/7) | 1.25e-4/5.62e-5/6.51e-5 | 1.41e-4/6.31e-5/6.38e-5 |
| supercell φ_w (N=3/5/7/9) | 0.05520/0.02089/0.01274/0.01019 | identical |
| pad-decay exponent | 0.14% | 0.19% |
| drift speed vs parameterization | 0.43% | 0.43% |
| Q_ramo vs Q_φ, median | 4.9e-3 | **4.70e-3** |
| Q/e on the centre pad | 0.966 | 0.9656 |
| charge deficit, h=277/139/69 µm | 5.50e-2, ratio 1.00 | 5.52e-2/5.51e-2/5.50e-2, ratio 1.00 |
| local consistency on quadratics | (not measured) | exact to 5e-12 |
| assembly, 6.2 M nodes | 61.8 s | 45.0 s |
| solve, 6.2 M nodes | 78.6 s / 6 982 iters | 27.7 s / 2 737 iters |
| peak memory, 6.2 M nodes | 1.89 GiB | 1.78 GiB |

The end-to-end physics lands in the same place, and the Ramo-vs-potential
agreement improves, which is the expected signature of the Section 8 gradient
correction: `Q_ramo` differentiates the weighting field and `Q_φ` does not. The
solve and the assembly are both faster — rows narrowed from 25 to 16 entries and
the neighbour probe from 24 `locate` calls per node to 8.
