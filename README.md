# Explore_FDM

GPU finite-difference Laplace solver on a non-graded octree, for LArTPC
**pixel**-anode drift and weighting fields, and the induced-current signals
that follow from them.

`∇²φ = 0`, `E = −∇φ`, in PyTorch, matrix-free, on the GPU. The anode is an
axis-aligned pad array, so the geometry is an exact integer-lattice boolean
mask — no meshing, no cut cells. One composite operator covers every node at
every refinement level, so a single sweep touches the whole grid; there is no
level-by-level AMR cycling.

See [PLAN.md](PLAN.md) for the method and the sizing argument behind it, and
[RESULTS.md](RESULTS.md) for the verification suite.

## Install

Needs Python 3.10+, PyTorch (CUDA build), NumPy, and Matplotlib for the plots.

```bash
pip install torch numpy matplotlib
```

## Run

```bash
python3 run_workflow.py                       # everything, into out/
python3 run_workflow.py --lmax 6 --npad 7     # finer grid, bigger supercell
python3 run_workflow.py --only response       # reuse a previous run's fields
python3 run_workflow.py --list                # show the stages
```

Stages run in order and hand results to each other in memory; a stage whose
inputs are missing loads them from `--outdir`, so any suffix of the pipeline
can run on its own against earlier output. Every field of `fdm.workflow.Config`
is exposed as a CLI flag, and `--config cfg.json` / `--dump-config` round-trip
the settings.

Stages: `grid` → `solve_drift` → `solve_weighting` → `efields` →
`export_fields` → `export_grids` → `drift` → `currents` → `response` → `plots`.

Each run writes `<outdir>/stages.jsonl` with per-stage wall time and memory.

## Package layout

| module | contents |
|---|---|
| `fdm/tree.py` | non-graded octree on an exact integer lattice |
| `fdm/geometry.py` | pixel-anode pads and gaps as lattice masks |
| `fdm/topology.py` | Min-Gibou composite Laplacian and gradient rows |
| `fdm/operator.py` | the Laplacian as a matrix-free PyTorch operator |
| `fdm/solve.py` | BiCGSTAB with fp64 reductions |
| `fdm/field.py` | `E = −∇φ` from the discretization's own gradient |
| `fdm/drift.py` | drift velocity, RK4 tracking, Shockley-Ramo current |
| `fdm/io.py` | `.npz` I/O, native-octree and resampled-grid forms |
| `fdm/workflow.py` | the stages, the config and the stage log |

## Standalone scripts

- `make_outputs.py` — solve the two fields and write the `.npz` outputs
- `make_response.py` — induced-current response table on an impact grid
  (pochoir-compatible layout)
- `plot_induced_current.py` — example figure from `out/induced_current.npz`
- `monitor_gpu.py` — sample GPU memory/utilisation during a run
- `profile_runtime.py` — turn `stages.jsonl` into a runtime and memory report

## Tests

```bash
python3 -m pytest tests
```

The suite is physics verification, not just unit tests: reduction to the
7-point Laplacian, exactness on linear fields, `φ` and `∇φ` convergence on a
non-graded tree, fp32-vs-fp64 agreement, single-pad weighting-field
convergence, the weighting-potential sum rule, exponential decay of pad
ripple, one elementary charge induced per collected electron, and the
finite-supercell induced-charge deficit.

## Note
If proven successful, this method will be implemented in the package `pochoir` for pixelated readout LArTPC.

## License

MIT — see [LICENSE](LICENSE).
