"""pochoir config translation: geometry must match pochoir node for node.

    PYTHONPATH=. python3 tests/test_pochoir.py

1. The anode-plane node sets we derive (drift tile pads, weighting probe pads,
   the W = 1 collecting pad) equal the ones pochoir's own generators draw, for
   the task13 fine (0.1 mm) and coarse (0.55 mm) configs.  Needs the pochoir
   source tree next to this repo (../pochoir); skipped otherwise.
2. At refine 1 the pad set restricted to the coarse lattice is unchanged, so
   refinement adds nodes without moving the electrodes.
3. An embedded cathode (not on a root face) gives a uniform bulk field whose
   effective anode plane sits within a fraction of a pitch of z_anode, and the
   solve on the coarse 0.55 mm config converges.

Asserts, and exits non-zero on failure.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

import numpy as np

from fdm import pochoir as P

HERE = os.path.dirname(os.path.abspath(__file__))
POCHOIR = os.path.abspath(os.path.join(HERE, "..", "..", "pochoir"))
CFG = os.path.join(HERE, "..", "configs", "pochoir")


def _pochoir_generators():
    """pochoir's two generator modules, loaded by file path.

    Importing them as ``pochoir.*`` would run pochoir's package __init__, which
    needs pochoir's full dependency set; the generators themselves need only
    numpy and matplotlib.
    """
    import importlib
    src = os.path.join(POCHOIR, "pochoir")
    if not os.path.isdir(src):
        return None
    # A stub package whose __path__ is pochoir's source directory: the
    # generators' relative imports resolve, pochoir's __init__ never runs.
    pkg = types.ModuleType("_pochoir_src")
    pkg.__path__ = [src]
    sys.modules["_pochoir_src"] = pkg
    try:
        return [importlib.import_module(f"_pochoir_src.{name}") for name in
                ("gen_pcb_drift_pixel_with_grid", "gen_pcb_pixel_with_grid")]
    except Exception as e:                       # pragma: no cover
        print(f"  (cannot load pochoir generators: {e})")
        return None


def _coarse_cfgs():
    a = os.path.join(POCHOIR, "scripts", "example_gen_pcb_drift_pixel_task13_coarse.json")
    b = os.path.join(POCHOIR, "scripts", "example_gen_pixel_with_grid_task13_coarse.json")
    return (a, b) if os.path.exists(a) and os.path.exists(b) else None


def check_against_generators():
    gens = _pochoir_generators()
    if gens is None:
        print("SKIP generator comparison (no ../pochoir)")
        return
    gd, gw = gens
    cases = [(os.path.join(CFG, "example_gen_pcb_drift_pixel_task13_fine.json"),
              os.path.join(CFG, "example_gen_pixel_with_grid_task13_fine.json"), 0.1)]
    if _coarse_cfgs():
        cases.append((*_coarse_cfgs(), 0.55))
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["POCHOIR_STORE"] = tmp
        os.chdir(tmp)                       # pochoir writes plots into ./store
        os.makedirs("store", exist_ok=True)
        try:
            for dcfg, wcfg, s in cases:
                geom = P.from_configs(dcfg, wcfg, spacing=s)
                nz = geom.anode_idx + 4     # enough planes to hold the pad
                # drift generator on a short (N, N, nz) domain
                cfg = P.load_json(dcfg)
                dom = types.SimpleNamespace(shape=(geom.N, geom.N, nz),
                                            spacing=(s, s, s))
                _, barr, *_ = gd.generator(dom, cfg)
                I, J = np.meshgrid(np.arange(geom.N), np.arange(geom.N), indexing="ij")
                ours = P.drift_pads(geom, 1).contains(I, J)
                assert (ours == (barr[:, :, geom.anode_idx] > 0)).all(), \
                    f"drift pad plane differs from pochoir at s={s}"
                assert not (barr[:, :, geom.anode_idx + 1:-1] > 0).any(), \
                    "pochoir has electrodes above our anode plane"
                # weighting generator on (Nw, Nw, nz)
                cfgw = dict(P.load_json(wcfg))
                cfgw.pop("driftZDepth", None)      # keep the short domain
                dom = types.SimpleNamespace(shape=(geom.Nw, geom.Nw, nz),
                                            spacing=(s, s, s))
                arr, barr, *_ = gw.generator(dom, cfgw)
                I, J = np.meshgrid(np.arange(geom.Nw), np.arange(geom.Nw),
                                   indexing="ij")
                ps = P.weighting_pads(geom, 1)
                plane = geom.anode_idx
                assert (ps.contains(I, J) == (barr[:, :, plane] > 0)).all(), \
                    f"weighting pad plane differs from pochoir at s={s}"
                assert (ps.contains(I, J, which=0)
                        == np.isclose(arr[:, :, plane], 1.0)).all(), \
                    f"collecting pad differs from pochoir at s={s}"
                print(f"OK  s={s}: drift {int(ours.sum())} pad nodes, weighting "
                      f"{int(ps.contains(I, J).sum())}, collecting "
                      f"{int(ps.contains(I, J, which=0).sum())} -- identical to pochoir")
        finally:
            os.chdir(cwd)


def check_refinement_keeps_pads():
    geom = P.from_configs(os.path.join(CFG, "example_gen_pcb_drift_pixel_task13_fine.json"))
    I, J = np.meshgrid(np.arange(geom.N), np.arange(geom.N), indexing="ij")
    a = P.drift_pads(geom, 1).contains(I, J)
    b = P.drift_pads(geom, 2).contains(2 * I, 2 * J)
    assert (a == b).all(), "refine 1 moves pad nodes that exist on both lattices"
    print("OK  refine 1 keeps every coarse-lattice pad node")


def check_embedded_cathode():
    pair = _coarse_cfgs()
    if pair is None:
        print("SKIP embedded-cathode solve (no coarse configs)")
        return
    import torch
    from fdm.pochoir_workflow import PochoirConfig, run
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = PochoirConfig(outdir=tmp, drift_config=pair[0], spacing=0.55,
                            device=dev, tol=1e-12)
        ctx, _ = run(cfg, stages=["geometry", "grid_drift", "solve_drift"])
    g, m = ctx.geom, ctx.drift
    assert m["Lc"] % (m["root_mm"] / (g.spacing)) != 0, "cathode is not embedded"
    u = m["u"].cpu().numpy()
    z = m["tree"].node_positions(m["rows"].coords)[:, 2]
    bulk = (z > g.z_anode + 30) & (z < g.z_cathode - 1)
    A = np.vstack([z[bulk], np.ones(bulk.sum())]).T
    (k, c), *_ = np.linalg.lstsq(A, u[bulk], rcond=None)
    resid = np.abs(A @ [k, c] - u[bulk]).max()
    z_eff = -c / k
    assert resid < 1e-6 * abs(g.cathode_potential), f"bulk not linear ({resid})"
    assert abs(k * (g.z_cathode - z_eff) - g.cathode_potential) < 1e-6 * abs(
        g.cathode_potential), "cathode potential not reached at the cathode plane"
    assert abs(z_eff - g.z_anode) < 0.5 * g.pitch, f"effective anode {z_eff}"
    print(f"OK  embedded cathode at {g.z_cathode} mm (root {m['root_mm']} mm): "
          f"bulk E {-k:.5f} V/mm, linear to {resid:.1e} V, effective anode "
          f"{z_eff:.4f} mm")


if __name__ == "__main__":
    check_against_generators()
    check_refinement_keeps_pads()
    check_embedded_cathode()
    print("all pochoir checks passed")
