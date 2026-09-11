#!/usr/bin/env python3
"""Draw the workflow diagram: the ten stages, what they pass, what they write.

    python3 plot_workflow_diagram.py                  # out/workflow_diagram.png
    python3 plot_workflow_diagram.py --out x.png --dpi 200

Drawn rather than generated from the code, so it says what the pipeline *means*
(two solves feeding three consumer branches) rather than re-listing ``STAGES``.
Two things it encodes that a stage list cannot:

* solid arrows are values handed between stages in memory through the Context;
  dashed arrows cross the filesystem,
* the single arrow back out of the file band is ``_need_fields()`` -- the reason
  ``--only response`` runs without re-solving.

Keep it in step with fdm/workflow.py when stages are added or reordered.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, INK3 = "#131a21", "#4c5966", "#77838f"
EDGE, PANEL = "#ccd2d9", "#f7f8fa"
FIELD = "#0d6e8c"        # the potential / field-solve path
CHARGE = "#a4531d"       # drifting charge, signals, files on disk

W, H = 1180.0, 512.0     # drawing units; y is flipped so it reads top-down
MONO = ["DejaVu Sans Mono", "monospace"]


def box(ax, x, y, w, h, title, lines=(), color=EDGE, lw=1.0, tcolor=INK):
    ax.add_patch(FancyBboxPatch(
        (x, H - y - h), w, h, boxstyle="round,pad=0,rounding_size=4",
        fc=PANEL, ec=color, lw=lw, zorder=2))
    ax.text(x + 14, H - y - 21, title, color=tcolor, fontsize=9.5,
            family=MONO, weight="bold", zorder=3)
    for k, s in enumerate(lines):
        ax.text(x + 14, H - y - 37 - 12 * k, s, color=INK2, fontsize=7.6,
                family=MONO, zorder=3)


def chip(ax, x, y, w, names, color=CHARGE, tcolors=None):
    ax.add_patch(FancyBboxPatch(
        (x, H - y - 44), w, 44, boxstyle="round,pad=0,rounding_size=4",
        fc=PANEL, ec=color, lw=1.0, ls=(0, (3, 3)), zorder=2))
    tc = tcolors or [color] * len(names)
    for k, s in enumerate(names):
        ax.text(x + 12, H - y - 20 - 17 * k, s, color=tc[k], fontsize=8,
                family=MONO, zorder=3)


def arrow(ax, pts, color=INK, ls="-", lw=1.3):
    """Elbow arrow through a list of (x, y) points in drawing coordinates."""
    fy = [(px, H - py) for px, py in pts]
    for a, b in zip(fy, fy[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=1,
                solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(
        fy[-2], fy[-1], arrowstyle="-|>", mutation_scale=11, color=color,
        lw=lw, ls=ls, shrinkA=0, shrinkB=0, zorder=1))


def label(ax, x, y, s, color=INK3, size=7.6, weight="normal", ha="left"):
    ax.text(x, H - y, s, color=color, fontsize=size, family=MONO, ha=ha,
            weight=weight, zorder=3)


def draw(out, dpi):
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    # --- configuration -----------------------------------------------------
    box(ax, 16, 34, 152, 72, "Config",
        ["configs/*.json", "+ CLI flags (CLI wins)"])
    arrow(ax, [(92, 106), (92, 148)])

    # --- one grid, two solves ---------------------------------------------
    box(ax, 16, 150, 152, 58, "grid",
        ["octree, operator,", "pad/gap/cathode masks"], FIELD, 1.4, FIELD)
    box(ax, 212, 96, 152, 58, "solve_drift",
        ["pads 0 V, cathode", "-E0*L, gap Neumann"], FIELD, 1.4, FIELD)
    box(ax, 212, 206, 152, 58, "solve_weighting",
        ["centre pad 1 V,", "all others 0 V"], FIELD, 1.4, FIELD)
    arrow(ax, [(168, 170), (196, 170), (196, 125), (210, 125)])
    arrow(ax, [(168, 190), (196, 190), (196, 235), (210, 235)])
    label(ax, 172, 186, "rows, operator, masks")

    box(ax, 408, 150, 152, 58, "efields",
        ["E = -grad phi, with the", "scheme's own gradient"], FIELD, 1.4, FIELD)
    arrow(ax, [(364, 125), (392, 125), (392, 170), (406, 170)], FIELD)
    arrow(ax, [(364, 235), (392, 235), (392, 188), (406, 188)], FIELD)
    label(ax, 370, 118, "phi_drift", FIELD)
    label(ax, 370, 256, "phi_weight", FIELD)

    # --- three consumer branches ------------------------------------------
    box(ax, 604, 30, 152, 50, "export_fields", ["values on octree nodes"])
    box(ax, 604, 94, 152, 50, "export_grids", ["resampled to regular grids"])
    box(ax, 604, 164, 152, 58, "drift",
        ["electrons tracked", "to the anode plane"], CHARGE, 1.4, CHARGE)
    box(ax, 800, 164, 152, 58, "currents",
        ["Shockley-Ramo", "i = q v . E_w"], CHARGE, 1.4, CHARGE)
    box(ax, 996, 164, 152, 58, "plots",
        ["subprocess:", "plot_induced_current.py"])
    box(ax, 604, 244, 152, 58, "response",
        ["re-drifts a grid of", "impact positions"], CHARGE, 1.4, CHARGE)

    arrow(ax, [(560, 170), (582, 170), (582, 55), (602, 55)])
    arrow(ax, [(560, 176), (582, 176), (582, 119), (602, 119)])
    arrow(ax, [(560, 184), (582, 184), (582, 193), (602, 193)], CHARGE)
    arrow(ax, [(560, 190), (582, 190), (582, 273), (602, 273)], CHARGE)
    label(ax, 412, 140, "phi, E on every node")
    arrow(ax, [(756, 193), (794, 193)], CHARGE)
    label(ax, 762, 186, "paths", CHARGE)
    arrow(ax, [(952, 193), (994, 193)])

    # --- the file band -----------------------------------------------------
    ax.plot([16, 1164], [H - 352, H - 352], color=EDGE, lw=1.0, zorder=1)
    label(ax, 16, 342, "W R I T T E N   T O   o u t /")

    chip(ax, 16, 372, 152, ["stages.jsonl", "every stage"], EDGE, [INK2, INK3])
    chip(ax, 188, 372, 184, ["response.npz", "response.npy"])
    chip(ax, 396, 372, 184, ["induced_current.npz", "induced_current.png"])
    chip(ax, 604, 372, 184, ["drift_field.npz", "weighting_field.npz"])
    chip(ax, 812, 372, 196, ["*_field_grid.npz", "near_anode_grid.npz"])

    d = (0, (4, 4))
    arrow(ax, [(680, 80), (680, 370)], CHARGE, d, 1.1)
    arrow(ax, [(900, 144), (900, 370)], CHARGE, d, 1.1)
    arrow(ax, [(876, 222), (876, 300), (488, 300), (488, 370)], CHARGE, d, 1.1)
    arrow(ax, [(640, 302), (640, 330), (280, 330), (280, 370)], CHARGE, d, 1.1)

    # The one arrow that leaves the file band: why a partial run works at all.
    arrow(ax, [(770, 370), (770, 208), (758, 208)], FIELD, (0, (5, 4)), 1.3)
    label(ax, 1164, 342,
          "_need_fields(): a stage run on its own reloads the fields it did "
          "not compute", FIELD, 8.2, ha="right")

    label(ax, 16, 462,
          "Stage order is fixed; --only and --skip choose a subset, and every "
          "stage appends wall time and GPU/host memory to stages.jsonl.",
          INK2, 8.4)
    label(ax, 16, 482,
          "Solid arrows carry values in memory through one Context object.  "
          "Dashed arrows cross the filesystem.", INK3, 8.4)

    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/workflow_diagram.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    draw(a.out, a.dpi)


if __name__ == "__main__":
    main()
