"""Slides on the pochoir-config mode, appended by make_methods_slides.py.

The comparison numbers are from the first task13-fine run (out_pochoir/,
compare_pochoir.txt); the figure slide is included only when that run's
compare_pochoir.png exists.
"""
from __future__ import annotations

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "out_pochoir")


def _report():
    p = os.path.join(OUT, "compare_pochoir.txt")
    return open(p).read().splitlines() if os.path.exists(p) else []


def add_slides(prs, helpers):
    tb, img, two = helpers["title_bullets"], helpers["image_slide"], helpers["two_col"]

    tb(prs, "New: solving a pochoir config directly", [
        "**run_pochoir.py** reads pochoir's own pixel JSON (e.g. "
        "example_gen_pcb_drift_pixel_task13_fine.json + weighting sibling)",
        "fdm/pochoir.py repeats pochoir's integer arithmetic: every int()/round() "
        "on pad, gap, chamfer, pad-plane z, FR4/pad thickness",
        ("Pads = pochoir's node sets, rounded corners included (trimCorner rule)", 1),
        ("Anode = drift-facing pad plane (10.0 mm): pads Dirichlet, gaps "
         "node-centred no-flux mirror, as pochoir's enableInsulatorFR4", 1),
        "Two trees, because pochoir solves two problems:",
        ("Drift: periodic 4.4 mm tile, cathode −7535 V at 160.6 mm", 1),
        ("Weighting: 5×5 probe, walls held at 0 on pochoir's halo nodes "
         "(x = −0.1 and 22.0 mm)", 1),
        "Electrons: pochoir start grid at launchZ, BNL mobility at 87 K, "
        "0.05 µs ticks, induce-pixel's 25×25 impact grid",
        "Outputs sit on pochoir's lattice and layout, so compare_pochoir.py "
        "differences index by index",
    ])

    two(prs, "Exact geometry, embedded boundaries", [
        "tests/test_pochoir.py runs pochoir's own generators:",
        ("0.1 mm: 1201 drift pad nodes, 30025 weighting, 1201 collecting: identical", 1),
        ("0.55 mm: 36 / 900 / 36: identical", 1),
        "Also identical to the boundary arrays of the stored pochoir run",
        "Unsupported features raise rather than approximate: shield grid, "
        "fixed_cell chamfer, enableFR4, no insulator",
    ], [
        "The walls are 22.1 mm apart (221 cells) and the cathode is 150.6 mm "
        "from the anode: neither is a whole number of root cells",
        "So they are embedded:",
        ("the box is rounded up to whole root cells", 1),
        ("cells straddling the plane are refined until it is a face", 1),
        ("nodes on or beyond the plane are held at the boundary value", 1),
        "No free node's stencil reaches past the plane, so this is exact",
        "Check: embedded cathode gives a linear bulk to 8e-9 V",
    ], left_head="Node-for-node match", right_head="Boundaries off the root grid")

    fig = os.path.join(OUT, "compare_pochoir.png")
    if os.path.exists(fig) and _report():
        img(prs, "First comparison with pochoir (task13 fine, 0.1 mm)", fig,
            caption="vs pochoir store_singlespacing_16cmdrift_16092026, "
                    "index-for-index on pochoir's lattice",
            bullets=[
                "Drift φ: max |Δ| 0.029 V = 3.9e-6 of 7535 V",
                "Bulk E_z: 49.9693 vs 49.9695 V/mm",
                "Weighting W: max |Δ| 9.7e-4 (rms 8.6e-5) up to 30 mm",
                "Arrival: 94.048 vs 94.038 µs (Δ 0.010 ± 0.020)",
                "Landing xy: median 3.5 µm, max 49 µm",
                "Charge per impact: max |Δ| 0.023 e",
            ])

    tb(prs, "Status and open points", [
        "Meshes at h_min 0.1 mm: drift tile 196,885 nodes, weighting probe "
        "6,164,259 nodes",
        ("Both BiCGSTAB solves converge to 4e-12 relative residual in under a "
         "minute on the GPU", 1),
        "Cost is now CPU-side assembly (build_rows / build_gradient: tens of "
        "minutes at 6 M nodes on a shared machine)",
        ("fill_missing rewritten with bincount: 278 s → 12 s, same result", 1),
        "Agreement: drift φ to 4e-6 of the cathode voltage, bulk E_z equal to "
        "5 digits, drift times within 0.01 µs, charges within 0.023 e",
        "Open: W far above the pad: ours higher by 1% at 10 mm, 10% at 20 mm, "
        "37% at 30 mm (values 2e-2 → 3e-4)",
        ("Box with walls 22.1 mm apart: slowest mode decays ×7.47 per 10 mm. "
         "Ours ×7.53 (20→30 mm); pochoir ×9.43", 1),
        ("Points to pochoir's Jacobi stopping early (precision 2e-8). "
         "Confirm with a tighter pochoir run and a grade/refine study here", 1),
        "Next: the same comparison against firep, using the lattice files",
    ])
