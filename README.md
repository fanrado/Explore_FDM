# Explore_FDM

GPU finite-difference Laplace solver on a non-graded octree, for LArTPC
**pixel**-anode drift and weighting fields, and the induced-current signals
that follow from them.

`∇²φ = 0`, `E = −∇φ`, in PyTorch, matrix-free, on the GPU. The anode is an
axis-aligned pad array, so the geometry is an exact integer-lattice boolean
mask — no meshing, no cut cells. One composite operator covers every node at
every refinement level, so a single sweep touches the whole grid; there is no
level-by-level AMR cycling.

See [PLAN.md](PLAN.md) for the method and the sizing argument behind it. The
scheme is that of Min, Gibou & Ceniceros, *J. Comput. Phys.* 218 (2006) 300–321,
implemented as they construct it; `tests/test_paper*.py` verify that.

## Install

Needs Python 3.10+, PyTorch (CUDA build), NumPy, and Matplotlib for the plots.

```bash
pip install torch numpy matplotlib
```

## Running the calculations

The whole pipeline — solve the fields, drift the electrons, build the induced
current and the response table, make the plots — is one command:

```bash
python3 run_workflow.py --config configs/reference.json
```

That is the reference geometry: a 5×5 periodic supercell of 4.434 mm pads,
3.048 mm pad / 1.386 mm gap, 297.078 mm of drift at 500 V/cm, `h_min` = 138.6 µm.
It writes the fields, the resampled grids, the induced current, the response
table and the plots into `out/`, plus `out/README_outputs.txt` describing each
file.

Start with `configs/quick.json` if you only want to check that the pipeline
runs — same anode geometry, but a quarter of the resolution on a 3×3 cell with a
17.736 mm drift. It finishes in a few seconds and is too coarse for physics:

```bash
python3 run_workflow.py --config configs/quick.json
```

### Configuration files

Geometry and run settings live in JSON, in `configs/`:

| file | what it is |
|---|---|
| `reference.json` | the reference geometry, 297.078 mm drift (67 pitches) |
| `quick.json` | smallest config that exercises every stage, seconds to run |
| `fine.json` | `h_min` = 34.6 µm on a 7×7 cell, 53.208 mm drift |
| `coarse_pad.json` | a different pad/gap ratio at the same pitch |

Every field of `fdm.workflow.Config` may appear in a config file, and every one
is also a command-line flag. **The CLI wins over the file**, so a config is a
baseline you override:

```bash
python3 run_workflow.py --config configs/reference.json --lmax 6 --outdir out_lmax6
python3 run_workflow.py --config configs/reference.json --drift-length 22.17
```

Keys beginning with `_` are treated as comments and ignored, so a config file can
document itself. To capture the fully resolved settings of a run — file plus CLI
plus defaults — use `--dump-config`:

```bash
python3 run_workflow.py --config configs/reference.json --npad 7 \
        --dump-config my_run.json
```

### The geometry parameters

Every config file lists **all 24 `Config` fields explicitly**, so a file is the
complete input to a run and nothing falls back to a code default.
`tests/test_configs.py` asserts that, so the files cannot rot as `Config` grows.

The geometry keys — all lengths in mm:

| key | meaning |
|---|---|
| `pitch` | pad pitch |
| `pad_width` | conducting pad width; the inter-pad gap is `pitch - pad_width` |
| `lmax` | refinement depth; `h_min = pitch / 2**lmax` |
| `npad` | N×N pads in the transversely periodic supercell |
| `drift_length` | anode plane to cathode |
| `efield` | drift field, V/cm |
| `grade` | coarsening rate: cell size ≈ distance-to-anode / `grade` |

Two exactness constraints, both of which **raise rather than silently snapping**:

- `pitch` and `pad_width` must be integer multiples of `h_min`, so every pad edge
  lands exactly on a grid line. That exactness is the reason this problem suits
  finite differences — it removes cut cells and geometric tolerances entirely.
- `drift_length` must be a whole number of pitches, since it sets the octree's
  root-cell count along z.

```
ValueError: pad width = 3.048375 mm is not an integer multiple of
h_min = 0.55425 mm (would be 5.500000 units)

ValueError: drift_length = 50.0 mm is not an integer multiple of
pitch = 4.434 mm (would be 11.276500 pitches)
```

If you hit the first, raise `lmax` or pick a `pad_width` on the lattice; the gap
is the length scale that sets the resolution you need. If you hit the second,
round `drift_length` to a multiple of the pitch.

Exactly one value is deliberately left as `null`: `pad_before`, the baseline
ahead of each waveform, which defaults to 20% of the longest drift time and so
is not knowable until the paths have been tracked. Set a number in µs to pin it.

### Running part of the pipeline

Stages run in order and hand results to each other in memory. A stage whose
inputs are not in memory loads them from `--outdir` instead, so any suffix of the
pipeline can run on its own against an earlier run's files:

```bash
python3 run_workflow.py --list                    # show the stages
python3 run_workflow.py --only response           # reuse a previous run's fields
python3 run_workflow.py --skip export_grids plots  # everything else
```

Stages: `grid` → `solve_drift` → `solve_weighting` → `efields` →
`export_fields` → `export_grids` → `drift` → `currents` → `response` → `plots`.

Each run writes `<outdir>/stages.jsonl` with per-stage wall time and memory;
`profile_runtime.py` turns that into a report, and `monitor_gpu.py` samples GPU
usage while a run is in flight.

## Package layout

| module | contents |
|---|---|
| `fdm/tree.py` | non-graded octree on an exact integer lattice |
| `fdm/geometry.py` | pixel-anode pads and gaps as lattice masks |
| `fdm/topology.py` | Min/Gibou/Ceniceros composite Laplacian and gradient rows |
| `fdm/operator.py` | the Laplacian as a matrix-free PyTorch operator |
| `fdm/solve.py` | BiCGSTAB with fp64 reductions |
| `fdm/field.py` | `E = −∇φ` from the discretization's own gradient |
| `fdm/neumann.py` | the paper's Section 7 all-Neumann procedure |
| `fdm/drift.py` | drift velocity, RK4 tracking, Shockley-Ramo current |
| `fdm/io.py` | `.npz` I/O, native-octree and resampled-grid forms |
| `fdm/workflow.py` | the stages, the config and the stage log |

## Standalone scripts

- `make_outputs.py` — solve the two fields and write the `.npz` outputs, without
  the rest of the pipeline (`--pitch`, `--pad-width`, `--lmax`, `--npad`, …)
- `make_response.py` — induced-current response table on an impact grid
  (pochoir-compatible layout)
- `plot_induced_current.py` — example figure from `out/induced_current.npz`
- `monitor_gpu.py` — sample GPU memory/utilisation during a run
- `profile_runtime.py` — turn `stages.jsonl` into a runtime and memory report

## Tests

Three files verify conformance to the source paper. They assert, exit non-zero
on failure, and run on the CPU if no GPU is present:

```bash
PYTHONPATH=. python3 tests/test_paper.py              # exact identities, structure
PYTHONPATH=. python3 tests/test_paper_convergence.py  # paper Section 9.1.6
PYTHONPATH=. python3 tests/test_paper_neumann.py      # paper Sections 7 and 9.1.7
```

The sharpest of these is an exact identity rather than a convergence study: the
scheme's 1-D difference is exact for quadratics and a multilinear interpolation
reproduces a quadratic with a known error, so the transverse cancellation must be
exact to round-off. `test_paper.py` checks that for all ten quadratic monomials,
for both the Laplacian and the gradient, on uniform, graded, non-graded,
randomly refined, periodic and production meshes.

The rest of the suite is physics verification, and is print-and-inspect rather
than asserting:

```bash
for t in configs operator convergence interface_correction precision pad_decay \
         weighting sumrule supercell charge_conservation drift benchmark; do
    PYTHONPATH=. python3 tests/test_$t.py
done
```

reduction to the 7-point Laplacian, exactness on linear fields, `φ` and `∇φ`
convergence on a non-graded tree, fp32-vs-fp64 agreement, single-pad
weighting-field convergence, the weighting-potential sum rule, exponential decay
of pad ripple, one elementary charge induced per collected electron, the
finite-supercell induced-charge deficit, what the Eq. (8) cancellation buys, and
assembly/solve scaling to 6.2 M nodes.

## Note
If proven successful, this method will be implemented in the package `pochoir` for pixelated readout LArTPC.

## License

MIT — see [LICENSE](LICENSE).
