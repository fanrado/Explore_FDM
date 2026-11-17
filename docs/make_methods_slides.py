#!/usr/bin/env python3
"""Build docs/Explore_FDM_methods.pptx: the numerical methods of this repository.

    python3 docs/make_methods_slides.py            # writes docs/Explore_FDM_methods.pptx
    python3 docs/make_methods_slides.py --out x.pptx

Every slide is produced by one function in ``SLIDES``; ``build(prs)`` runs them
in order.  Numbers quoted on the slides are read from ``out/stages.jsonl`` (the
reference run) where possible, so they follow the run rather than this file.

Extension hook
--------------
If ``docs/slides_extra.py`` exists and defines ``add_slides(prs, helpers)``, it
is called after the built-in slides and before saving.  ``helpers`` is a dict of
the styling functions used here, so added slides look the same:

    helpers["title_bullets"](prs, title, bullets, notes=None)
    helpers["image_slide"](prs, title, image_path, caption=None, bullets=None)
    helpers["two_col"](prs, title, left, right, left_head=None, right_head=None)
    helpers["eq_box"](slide, lines, left, top, width, height)   # monospace box
    helpers["table"](slide, rows, left, top, width, col_widths=None)
    helpers["blank"](prs, title)                                # titled empty slide

A bullet is a string, or ``(text, level)`` with level 0 or 1.  Lengths are
python-pptx ``Emu`` (use ``pptx.util.Inches``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out")

# ------------------------------------------------------------------ styling

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)
TITLE_TOP, TITLE_H = Inches(0.35), Inches(0.85)
BODY_TOP = Inches(1.45)
BODY_H = H - BODY_TOP - Inches(0.5)

FONT = "Calibri"
MONO = "Consolas"
INK = RGBColor(0x1F, 0x29, 0x33)
MUTED = RGBColor(0x5B, 0x67, 0x72)
ACCENT = RGBColor(0x0B, 0x6E, 0x8A)   # teal, as in workflow_diagram.png
ACCENT2 = RGBColor(0xA3, 0x52, 0x1A)  # rust, as in workflow_diagram.png
PANEL = RGBColor(0xF3, 0xF5, 0xF7)
RULE = RGBColor(0xCF, 0xD6, 0xDC)

BODY_PT, SUB_PT, CAP_PT, EQ_PT = 20, 17, 14, 16


def _run(p, text, size, color=INK, bold=False, font=FONT, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = color
    return r


def _textbox(slide, left, top, width, height, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(0.05))
    return tf


def _rich(p, text, size, color=INK, font=FONT):
    """Add ``text`` to paragraph ``p``; segments between ** are bold."""
    for k, seg in enumerate(text.split("**")):
        if seg:
            _run(p, seg, size, color=color, bold=bool(k % 2), font=font)


def blank(prs, title):
    """New slide with the deck's title bar and footer rule."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(1, 0, 0, Inches(0.18), H)  # left accent stripe
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()
    tf = _textbox(s, MARGIN, TITLE_TOP, W - 2 * MARGIN, TITLE_H, MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], title, 30 if len(title) <= 46 else 24,
         color=INK, bold=True)
    ln = s.shapes.add_connector(1, MARGIN, TITLE_TOP + TITLE_H + Inches(0.05),
                                W - MARGIN, TITLE_TOP + TITLE_H + Inches(0.05))
    ln.line.color.rgb = RULE
    ln.line.width = Pt(1.25)
    n = len(prs.slides)
    ft = _textbox(s, W - MARGIN - Inches(3), H - Inches(0.42), Inches(3), Inches(0.3))
    ft.paragraphs[0].alignment = PP_ALIGN.RIGHT
    _run(ft.paragraphs[0], f"Explore_FDM methods  ·  {n}", 11, color=MUTED)
    return s


def _hang(p, mar, ind):
    """Hanging indent: wrapped lines align with the text, not the marker."""
    ppr = p._p.get_or_add_pPr()
    ppr.set("marL", str(int(mar)))
    ppr.set("indent", str(-int(ind)))


def _bullets(slide, bullets, left, top, width, height, size=BODY_PT):
    tf = _textbox(slide, left, top, width, height)
    first = True
    for b in bullets:
        text, lvl = (b, 0) if isinstance(b, str) else b
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        sz = size if lvl == 0 else size - 3
        p.space_before = Pt(9 if lvl == 0 else 3)
        _hang(p, Inches(0.3 if lvl == 0 else 0.6), Inches(0.28))
        mark = "▪\t" if lvl == 0 else "–\t"
        _run(p, mark, sz, color=ACCENT if lvl == 0 else MUTED, bold=True)
        _rich(p, text, sz, color=INK if lvl == 0 else MUTED)
    return tf


def title_bullets(prs, title, bullets, notes=None, size=BODY_PT):
    s = blank(prs, title)
    _bullets(s, bullets, MARGIN, BODY_TOP, W - 2 * MARGIN, BODY_H, size=size)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def eq_box(slide, lines, left, top, width, height, size=EQ_PT):
    """Shaded monospace panel for equations written as plain Unicode."""
    box = slide.shapes.add_shape(1, left, top, width, height)
    box.fill.solid(); box.fill.fore_color.rgb = PANEL
    box.line.color.rgb = RULE
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for side in ("left", "right"):
        setattr(tf, f"margin_{side}", Inches(0.2))
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        _run(p, line, size, color=INK, font=MONO)
    return box


def _fit(path, left, top, width, height):
    iw, ih = Image.open(path).size
    sc = min(width / iw, height / ih)
    w, h = int(iw * sc), int(ih * sc)
    return left + (width - w) // 2, top + (height - h) // 2, w, h


def picture(slide, path, left, top, width, height):
    x, y, w, h = _fit(path, left, top, width, height)
    return slide.shapes.add_picture(path, Emu(x), Emu(y), Emu(w), Emu(h))


def image_slide(prs, title, image_path, caption=None, bullets=None,
                img_frac=0.62):
    """Image with an optional caption below; with ``bullets`` the image takes
    the left ``img_frac`` of the width and the bullets the rest."""
    s = blank(prs, title)
    cap_h = Inches(0.75) if caption else 0
    top, avail_h = BODY_TOP, BODY_H - cap_h
    if bullets:
        iw = int((W - 2 * MARGIN) * img_frac)
        picture(s, image_path, MARGIN, top, iw, avail_h)
        _bullets(s, bullets, MARGIN + iw + Inches(0.3), top,
                 W - 2 * MARGIN - iw - Inches(0.3), avail_h, size=SUB_PT)
    else:
        picture(s, image_path, MARGIN, top, W - 2 * MARGIN, avail_h)
    if caption:
        tf = _textbox(s, MARGIN, top + avail_h + Inches(0.05), W - 2 * MARGIN, cap_h)
        _rich(tf.paragraphs[0], caption, CAP_PT, color=MUTED)
    return s


def two_col(prs, title, left, right, left_head=None, right_head=None,
            size=SUB_PT):
    """Two bullet columns, each with an optional coloured heading."""
    s = blank(prs, title)
    gap = Inches(0.4)
    cw = (W - 2 * MARGIN - gap) // 2
    for k, (items, head) in enumerate(((left, left_head), (right, right_head))):
        x = MARGIN + k * (cw + gap)
        y = BODY_TOP
        if head:
            tf = _textbox(s, x, y, cw, Inches(0.5))
            _run(tf.paragraphs[0], head, 21, color=ACCENT if k == 0 else ACCENT2,
                 bold=True)
            y += Inches(0.5)
        _bullets(s, items, x, y, cw, BODY_TOP + BODY_H - y, size=size)
    return s


def table(slide, rows, left, top, width, col_widths=None, size=15,
          row_h=Inches(0.42)):
    """Simple table; first row is the header."""
    nr, nc = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nr, nc, left, top, width, row_h * nr)
    t = shp.table
    if col_widths:
        tot = sum(col_widths)
        for j, cw in enumerate(col_widths):
            t.columns[j].width = int(width * cw / tot)
    for i, row in enumerate(rows):
        t.rows[i].height = row_h
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = ACCENT if i == 0 else (
                PANEL if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF))
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            if i == 0:
                _run(p, str(val), size, color=RGBColor(0xFF, 0xFF, 0xFF), bold=True)
            else:
                _rich(p, str(val), size)
    return shp


HELPERS = dict(title_bullets=title_bullets, image_slide=image_slide,
               two_col=two_col, eq_box=eq_box, table=table, blank=blank,
               picture=picture)

# ------------------------------------------------------------ run numbers


def _stages():
    """Per-stage records of the reference run, keyed by stage name."""
    meta, st = {}, {}
    path = os.path.join(OUT, "stages.jsonl")
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "meta":
                meta = r
            elif r.get("kind") == "stage":
                st[r["stage"]] = r
    return meta, st


META, ST = _stages()
CFG = META.get("config", {})


def _f(x, nd=1):
    return f"{x:.{nd}f}"


# ------------------------------------------------------------------ slides


def s_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor(0x12, 0x2B, 0x3A)
    bg.line.fill.background()
    stripe = s.shapes.add_shape(1, 0, Inches(4.45), W, Inches(0.08))
    stripe.fill.solid(); stripe.fill.fore_color.rgb = ACCENT2
    stripe.line.fill.background()
    tf = _textbox(s, Inches(0.9), Inches(1.4), W - Inches(1.8), Inches(1.4),
                  MSO_ANCHOR.BOTTOM)
    _run(tf.paragraphs[0], "Explore_FDM: numerical methods", 44,
         color=RGBColor(0xFF, 0xFF, 0xFF), bold=True)
    tf = _textbox(s, Inches(0.9), Inches(2.9), W - Inches(1.8), Inches(1.4))
    _run(tf.paragraphs[0],
         "GPU finite-difference Laplace solver on a non-graded octree for "
         "LArTPC pixel-anode drift and weighting fields, and the induced "
         "currents that follow from them", 22, color=RGBColor(0xDD, 0xE6, 0xEC))
    tf = _textbox(s, Inches(0.9), Inches(4.8), W - Inches(1.8), Inches(1.6))
    p = tf.paragraphs[0]
    _run(p, "∇²φ = 0,   E = −∇φ   ·   PyTorch, matrix-free, fp64 on one GPU", 20,
         color=RGBColor(0xFF, 0xFF, 0xFF), font=MONO)
    p = tf.add_paragraph(); p.space_before = Pt(14)
    _run(p, "Scheme: C. Min, F. Gibou, H. Ceniceros, J. Comput. Phys. 218 (2006) "
            "300–321", 17, color=RGBColor(0xB8, 0xC7, 0xD1))


def s_problem(prs):
    title_bullets(prs, "Problem and requirements", [
        "Drift field and per-pad weighting field above a **pixel** anode: an "
        "axis-aligned array of square pads with dielectric gaps between them",
        "**Accuracy first** — E = −∇φ is the deliverable, so gradient accuracy "
        "is what counts; then speed and GPU memory",
        "Pad-induced distortion extends a finite distance off the plane, so the "
        "fine region is a genuine 3-D slab, not just a graded drift axis; the "
        "bulk is nearly uniform",
        "Structural requirements that drive the method:",
        ("**No meshing** — geometry is a boolean mask on the grid", 1),
        ("**One composite operator** over every node at every refinement level; "
         "one sweep touches everything — no level-by-level AMR cycling "
         "(rules out block-structured AMR: Chombo, AMReX)", 1),
        "Outputs: φ and E for both problems, drift paths, Shockley–Ramo "
        "currents and a field-response table",
    ])


def s_lattice(prs):
    s = blank(prs, "Why finite differences on an exact integer lattice")
    _bullets(s, [
        "Pads are axis-aligned rectangles: if pitch and pad width are integer "
        "multiples of h_min, **every pad edge lands exactly on a grid line**",
        "Geometry becomes a boolean mask built by integer comparison — no cut "
        "cells, no stair-stepping, no geometric tolerance. This is why FDM "
        "beats FEM here",
        "All topology is carried in integer **units** of h_min; node identity is "
        "exact integer equality, never a float comparison",
        "Exactness is enforced: off-lattice pitch / pad width, or a drift length "
        "that is not a whole number of pitches, **raises** instead of snapping",
    ], MARGIN, BODY_TOP, Inches(7.4), BODY_H, size=18)
    rows = [["reference geometry", "value"],
            ["pitch", "4.434 mm = 32 h_min"],
            ["pad / gap", "3.048 mm (22) / 1.386 mm (10)"],
            ["lmax  →  h_min", "5  →  138.6 µm"],
            ["supercell", "5 × 5 pads, periodic in x, y"],
            ["drift length", "297.078 mm = 67 pitches"],
            ["drift field", "500 V/cm"]]
    table(s, rows, Inches(8.3), BODY_TOP + Inches(0.15), Inches(4.4),
          col_widths=[1.3, 2.0], size=14, row_h=Inches(0.5))
    tf = _textbox(s, Inches(8.3), BODY_TOP + Inches(3.85), Inches(4.4), Inches(1.6))
    _rich(tf.paragraphs[0], "From configs/reference.json. The gap, not the "
          "pitch, sets the resolution: the field singularity is at the pad edge.",
          CAP_PT, color=MUTED)


def s_sizing(prs):
    s = blank(prs, "Problem sizing: why an octree")
    rows = [["gap", "h_min", "1 pad (periodic)", "7×7 island", "mem (7×7)"],
            ["0.5 mm", "42 µm", "0.20 M", "9.9 M", "0.26 GiB"],
            ["1.0 mm", "83 µm", "0.09 M", "4.6 M", "0.12 GiB"],
            ["2.2 mm", "183 µm", "0.06 M", "3.2 M", "0.08 GiB"]]
    table(s, rows, MARGIN, BODY_TOP, Inches(6.6), size=15, row_h=Inches(0.48))
    tf = _textbox(s, MARGIN, BODY_TOP + Inches(2.0), Inches(6.6), Inches(0.8))
    _rich(tf.paragraphs[0], "PLAN.md sizing (500 mm drift, h_min = gap/12, cell "
          "size grows with distance from the pads, capped at pitch/8 in the bulk).",
          CAP_PT, color=MUTED)
    rows2 = [["7×7 pads at 34.6 µm", "nodes", "memory"],
             ["uniform grid", "11.6 G", "302 GiB"],
             ["fine slab + coarse bulk", "311 M", "8.1 GiB"],
             ["**graded octree**", "**13 M**", "**0.34 GiB**"]]
    table(s, rows2, MARGIN, BODY_TOP + Inches(2.95), Inches(6.6),
          col_widths=[2.4, 1, 1], size=15, row_h=Inches(0.48))
    _bullets(s, [
        "The coarse bulk is essentially free: < 1 % of nodes in every case",
        "The win is grading **within** the anode slab: a two-array split "
        "captures ~35× of a ~2000× available gain — only a multi-level tree "
        "expresses it",
        "Consequence: the problem is small. ~10 M nodes ≈ 1 GB of traffic per "
        "iteration ≈ 1 ms on an RTX 4090 → seconds-long Krylov solves",
        "Bandwidth-bound at ~1 FLOP/byte, so fp64 costs ~2×, not 64× → "
        "**fp64 by default**",
    ], Inches(7.6), BODY_TOP, W - Inches(7.6) - MARGIN, BODY_H, size=17)


def s_octree(prs):
    g = ST.get("grid", {})
    two_col(prs, "Non-graded octree on an integer lattice  (tree.py, geometry.py)", [
        "Leaves stored as a flat integer array of (level, i, j, k); a level-L "
        "cell is 2^(lmax−L) units wide",
        "Root cells: npad × npad × (drift / pitch) — one level-0 cell per pitch",
        "**Non-graded**: no constraint on the level jump between neighbours, no "
        "buffer layers — required by the aggressive grading",
        "Periodic x, y: coordinates folded onto the canonical range; a periodic "
        "axis has one fewer node plane",
        "Nodes = cell corners, deduplicated by integer key",
    ], [
        "Cell size ≤ max(h_min, |z − z_anode| / grade), grade = 8 — follows the "
        "exp(−2πz/pitch) decay of pad ripple",
        "Forced to h_min within 2 cells of a pad edge (in x or y) and within 2 "
        "pitches of the plane — where the field is singular",
        "Refine passes split every leaf the predicate flags until none remain",
        f"Reference run: **{g.get('cells', 0)/1e6:.2f} M cells, "
        f"{g.get('nodes', 0)/1e6:.2f} M nodes**",
    ], left_head="Data structure", right_head="Refinement rule")


def s_nodes(prs):
    s = blank(prs, "Min–Gibou–Ceniceros: a node-based scheme")
    _bullets(s, [
        "Unknowns live at cell **nodes** (vertices), not centres. Their "
        "Theorem 1: no locally consistent linear cell-centred stencil exists on "
        "a non-uniform Cartesian grid using only adjacent cells — the reason "
        "cell-centred AMR needs refluxing and level cycling",
        "Each node u0 is a corner of a chosen leaf C (the finest usable incident "
        "leaf). Per axis:",
        ("**inward** neighbour: C's adjacent corner — always a real node, "
         "distance s_in = h_C  (paper's u1, u2, u3)", 1),
        ("**outward** neighbour: point on the far face of the adjacent leaf C_a, "
         "distance s_out = h_Ca  (u4, u5, u6); multilinear from up to 4 corners, "
         "2 if face-aligned, 1 if edge-aligned", 1),
        "Compact: C plus at most 3 neighbouring cells; one row per node, no "
        "interface pass. Second order in both φ and ∇φ",
    ], MARGIN, BODY_TOP, Inches(7.5), BODY_H, size=17)
    eq_box(s, [
        "Per-axis difference (Eq. 1):",
        "",
        "D_a u = 2/(s_in+s_out) ·",
        "   [ (u_in−u0)/s_in",
        "   + (u_out−u0)/s_out ]",
        "",
        "Interpolating u_out with",
        "offsets p, q along a",
        "transverse axis t adds",
        "(p·q/2)·u_tt, so",
        "",
        "D_a u = u_aa",
        "   + Σ_t M[a][t]·u_tt + O(h)",
    ], Inches(8.5), BODY_TOP + Inches(0.1), W - Inches(8.5) - MARGIN,
        Inches(5.1), size=16)


def s_eq8(prs):
    s = blank(prs, "Eq. (8): cancelling the interpolation error")
    eq_box(s, [
        "α·D_x u  +  D_y u  +  β·D_z u  =  Δu + O(h)",
        "",
        "α = 1 − s9·s12 / (s5·(s2+s5))",
        "β = 1 − s10·s11 / (s5·(s2+s5)) − α·s7·s8 / (s4·(s1+s4))",
        "",
        "implemented as: solve  Mᵀ w = (1, 1, 1)  per node",
    ], MARGIN, BODY_TOP, W - 2 * MARGIN, Inches(2.25), size=17)
    _bullets(s, [
        "M is unit-triangular in the order (edge-aligned, face-aligned, "
        "face-interior), so the solve returns exactly α, β and w = 1 on the "
        "interpolated axis; a residual check guards the structure",
        "**Locally uniform patch: α = β = 1** and the row collapses to the "
        "standard 7-point Laplacian — the interior costs nothing extra",
        "Erratum kept in code: the printed α, β have s9·s12 and s10·s11 "
        "interchanged; the forms above are the self-consistent ones "
        "(tests/test_paper.py T1 would catch the printed version)",
        "M-matrix (diagonally dominant) for cubic cells; **non-symmetric** "
        "(node sampling is not reflexive) → CG invalid, use BiCGSTAB",
        "interface_correction=False forces w = 1: the naive, locally "
        "inconsistent scheme, kept to measure what Eq. (8) buys",
    ], MARGIN, BODY_TOP + Inches(2.45), W - 2 * MARGIN, BODY_H - Inches(2.45),
        size=16)


def s_bcs(prs):
    two_col(prs, "Boundary conditions", [
        "**Dirichlet** — pads and cathode carry identity rows; b holds the "
        "prescribed value",
        ("drift solve: all pads 0 V, cathode −E0·L", 1),
        ("weighting solve: centre pad 1 V, other pads and cathode 0 V", 1),
        "**Neumann** du/dn = 0 on the bare inter-pad gap: the missing outward "
        "neighbour is the inward one mirrored",
        ("axis difference collapses to 2(u_in − u0)/s²; mirrored side is a real "
         "node, so no interpolation error", 1),
        ("an extension: the paper does not say how it discretizes du/dn = 0", 1),
    ], [
        "**Periodic** in x and y over the N×N supercell: an infinite pad array; "
        "a cell reached through the seam has its origin un-wrapped by a whole "
        "period (also beyond the paper)",
        "**All-Neumann systems** (fdm/neumann.py, paper Section 7): singular; "
        "pin one corner, re-solve a small corner subdomain with the pin free, "
        "patch the gradients there",
        ("not used in production: any Dirichlet electrode makes the system "
         "nonsingular, and the code refuses it", 1),
        ("with the mirrored rows (row sums 0) the pin cannot corrupt ∇u, so the "
         "paper's Table 9 stall does not occur — test_paper_neumann.py", 1),
    ], left_head="Electrodes and gaps", right_head="Supercell and Section 7")


def s_operator(prs):
    g = ST.get("grid", {})
    frac = 100 * g.get("irregular", 0) / max(g.get("nodes", 1), 1)
    s = blank(prs, "Matrix-free composite operator on the GPU  (fdm/operator.py)")
    _bullets(s, [
        "Rows assembled once (NumPy, CPU) into **two padded dense blocks**, "
        "moved to the GPU:",
        ("regular: 7 entries (diagonal + 3 inward + 3 outward, all real nodes)", 1),
        ("irregular: 16 entries (diagonal + 3 inward + 3 × 4 interpolants); "
         "paper bounds a row at 11, padding is zero-valued and inert", 1),
        "A·u = gather → multiply → row-sum → scatter, for both blocks, then one "
        "masked write of the Dirichlet identity rows",
        "**Every node at every level in one call** — no level ordering, no "
        "coarse/fine sequencing; the hierarchy is absorbed into the index "
        "and coefficient tensors",
        "int32 column indices halve index traffic",
        f"Reference run: **{frac:.1f} %** of {g.get('nodes', 0)/1e6:.2f} M nodes "
        "are irregular, so a matvec costs close to a 7-point sweep",
        "Note: PLAN.md proposed S³ tiles with halos; the code uses flat per-node "
        "row blocks instead",
    ], MARGIN, BODY_TOP, Inches(7.9), BODY_H, size=17)
    eq_box(s, [
        "out = zeros(n)",
        "out[reg_row] =",
        "  (reg_val *",
        "   u[reg_idx]).sum(1)",
        "out[irr_row] =",
        "  (irr_val *",
        "   u[irr_idx]).sum(1)",
        "A·u = where(dirichlet,",
        "            u, out)",
        "",
        "reg_idx: (n_r, 7)",
        "irr_idx: (n_i, 16)",
    ], Inches(8.9), BODY_TOP + Inches(0.1), W - Inches(8.9) - MARGIN,
        Inches(4.9), size=16)


def s_solver(prs):
    d, w = ST.get("solve_drift", {}), ST.get("solve_weighting", {})
    two_col(prs, "Linear solver  (fdm/solve.py)", [
        "**BiCGSTAB**, pure PyTorch, no host round-trips — the operator is "
        "non-symmetric so CG does not apply",
        "**Jacobi preconditioner** = operator diagonal; it varies by orders of "
        "magnitude across levels, so it is near-essential",
        "All inner products accumulated in **fp64** whatever the working dtype "
        "— where accuracy is lost in 10⁷-element reductions",
        "Breakdown handling: restart against current residual when ρ → 0",
        "Stop on relative residual ‖r‖/‖b‖ < tol "
        f"(reference: {CFG.get('tol', 1e-11):.0e})",
    ], [
        "**fp64 by default**: bandwidth-bound, so it costs ~2× fp32",
        "Optional fp32 storage + fp64 residual + iterative refinement "
        "(refine_steps) for very large runs; at these sizes it is slower "
        "(test_precision.py)",
        "sweep(): damped Jacobi over all levels at once — debugging aid and "
        "proof that no level ordering is needed",
        f"Reference run: drift solve **{d.get('iters')} iterations, "
        f"{_f(d.get('wall', 0))} s**; weighting {w.get('iters')} iterations, "
        f"{_f(w.get('wall', 0))} s; final residuals "
        f"{d.get('residual', 0):.1e} / {w.get('residual', 0):.1e}",
    ], left_head="Method", right_head="Precision and cost")


def s_gradient(prs):
    s = blank(prs, "E = −∇φ from the scheme's own gradient  (fdm/field.py)")
    eq_box(s, [
        "u'_a = [ s_m²(u_p − u0) + s_p²(u0 − u_m) ] / ( s_m·s_p·(s_m + s_p) )",
        "",
        "Section 8 correction where u_out was interpolated:",
        "u'_a −= c_out[a] · Σ_{t≠a} (p_t·q_t / 2) · u_tt,     u_vec = M⁻¹ D",
    ], MARGIN, BODY_TOP, W - 2 * MARGIN, Inches(1.9), size=17)
    _bullets(s, [
        "Same cell C, same neighbours, same interpolants as the Laplacian — the "
        "gradient is part of the discretization, not a post-hoc difference",
        "Quadratic through the node and its two neighbours on unequal spacings; "
        "centred difference when s_m = s_p",
        "u_tt must be the **decontaminated** second difference M⁻¹D: raw D_t "
        "leaves an O(h) residual where two directions are interpolated. With "
        "it, ∇u is exact for every quadratic at every node",
        "GPU form: per-axis 6-wide gather-reduce, plus a 10-wide correction "
        "block only on rows with an interpolated neighbour",
        "Dirichlet nodes (pad surfaces, cathode) have no row; their E is filled "
        "by averaging covered corners of adjacent cells (fill_missing)",
    ], MARGIN, BODY_TOP + Inches(2.05), W - 2 * MARGIN, BODY_H - Inches(2.05),
        size=17)


def s_fields(prs):
    image_slide(prs, "Reference solution near the anode",
                os.path.join(OUT, "near_anode.png"),
                caption="Fine resampled grid (161×161×97 at h_min = 0.139 mm). "
                "Top: drift potential minus its plane average and |E|; bottom: "
                "centre-pad weighting potential and |E_w|; right: values on the "
                "anode plane. Pads 3.05 mm, gaps 1.39 mm, 500 V/cm.")


def s_drift(prs):
    s = blank(prs, "Electron drift  (fdm/drift.py)")
    _bullets(s, [
        "**Velocity**: default ICARUS LAr fit v(|E|, T) (Walkowiak, NIM A 449 "
        "(2000) 288), as in LArSoft — 1.596 mm/µs at 500 V/cm, 87.3 K; "
        "direction −E/|E|",
        "**Interpolation**: trilinear within each leaf from its 8 corner nodes; "
        "C⁰ within a level, O(h²) kink across a coarse/fine face",
        "**RK4** tracker, step = 0.25 × local cell-crossing time → small steps "
        "near the pads, where paths bend; final step rescaled to land on z = 0",
        "Stall detection (max_time): with an ideal Neumann gap, field lines on "
        "the gap symmetry surface never reach a pad",
    ], MARGIN, BODY_TOP, W - 2 * MARGIN, Inches(2.6), size=16)
    picture(s, os.path.join(OUT, "trajectories.png"), MARGIN,
            BODY_TOP + Inches(2.6), W - 2 * MARGIN, Inches(2.6))
    tf = _textbox(s, MARGIN, BODY_TOP + Inches(5.2), W - 2 * MARGIN, Inches(0.4))
    _rich(tf.paragraphs[0], "Reference run, 100 electrons: transverse pull vs "
          "height, landing points, drift time vs start height (slope = bulk "
          "velocity).", CAP_PT, color=MUTED)


def s_ramo(prs):
    s = blank(prs, "Shockley–Ramo current and response table")
    eq_box(s, [
        "i(t) = −e · v · E_w(x(t))",
        "E_w  = −∇φ_w",
        "Q(t) = +e · [ φ_w(x(t)) − φ_w(x0) ]",
    ], MARGIN, BODY_TOP, Inches(5.7), Inches(1.25), size=16)
    _bullets(s, [
        "∫i dt (differentiates φ_w) and the potential difference (does not) "
        "are independent routes: their agreement checks gradient and tracker "
        "together",
        "A collected electron must induce exactly Q/e = 1; a finite N×N "
        "supercell predicts the deficit (1/N²)(1 − z/L)",
        "**Response table** (make_response / response stage): electrons from a "
        "regular impact grid at fixed start z, binned into Q(t) on a fixed "
        "tick, differenced → current per electron per tick (pochoir-compatible "
        "layout)",
    ], MARGIN, BODY_TOP + Inches(1.35), Inches(5.7), BODY_H - Inches(1.35), size=15)
    r = ST.get("response", {})
    picture(s, os.path.join(OUT, "response.png"), Inches(6.5), BODY_TOP,
            W - Inches(6.5) - MARGIN, Inches(4.2))
    tf = _textbox(s, Inches(6.5), BODY_TOP + Inches(4.25), W - Inches(6.5) - MARGIN,
                  Inches(1.2))
    ig = r.get("impact_grid", ["?", "?"])
    sh = r.get("shape", ["?", "?"])
    _rich(tf.paragraphs[0], f"out/ run: {ig[0]}×{ig[1]} impact grid, "
          f"{CFG.get('paths_per_pixel')} paths/pixel, start z = "
          f"{CFG.get('start_z', 0):.1f} mm → table {sh[0]} × {sh[1]} ticks of "
          f"{CFG.get('tick')} µs; {r.get('collected')} of {sh[0]} impacts "
          "collected on the centre pad.", CAP_PT, color=MUTED)


def s_pipeline(prs):
    image_slide(prs, "Pipeline: ten named stages  (fdm/workflow.py)",
                os.path.join(OUT, "workflow_diagram.png"),
                caption="One JSON config (+ CLI overrides) drives all stages. "
                "Solid arrows pass data in memory; dashed arrows go through out/, "
                "so any suffix of the pipeline (--only / --skip) can rerun "
                "against an earlier run's files. Every stage logs wall time and "
                "memory to stages.jsonl.")


def s_verify(prs):
    s = blank(prs, "Verification  (tests/)")
    rows = [["asserting conformance tests", "what is checked"],
            ["test_paper.py",
             "Σ w_a D_a u = Δu exactly for all 10 quadratic monomials (Laplacian "
             "and gradient) on uniform, graded, non-graded, random, periodic and "
             "production meshes; 7-point reduction; Eq. 8 structure (triangular, "
             "≤ 11 wide)"],
            ["test_paper_convergence.py",
             "Paper §9.1.6, u = exp(xyz): φ order ≥ 1.8 on a non-graded grid; "
             "gradient order → 2 on uniform; no extra degradation from level jumps"],
            ["test_paper_neumann.py",
             "Paper §7 / §9.1.7 all-Neumann: constant in the nullspace, pin "
             "harmless, Section 7 procedure exercised end to end"],
            ["test_configs.py",
             "every config lists all Config fields and is exact on the lattice"]]
    table(s, rows, MARGIN, BODY_TOP, W - 2 * MARGIN, col_widths=[1.15, 4],
          size=14, row_h=Inches(0.62))
    tf = _textbox(s, MARGIN, BODY_TOP + Inches(3.3), W - 2 * MARGIN, Inches(0.4))
    _run(tf.paragraphs[0], "Physics checks (print-and-inspect):", 17,
         color=ACCENT, bold=True)
    _bullets(s, [
        "operator: 7-point reduction, linear fields annihilated incl. T-junctions"
        "  ·  convergence / interface_correction: φ, ∇φ orders and what Eq. 8 buys",
        "pad_decay: ripple ∝ exp(−2πz/pitch)  ·  sumrule: φ_cathode + Σ φ_pad = 1"
        "  ·  weighting: island-size convergence",
        "charge_conservation: Q/e → 1  ·  supercell: deficit (1/N²)(1 − z/L)"
        "  ·  drift: Ramo ∫i dt = e Δφ_w  ·  precision, benchmark (to 6.2 M nodes)",
    ], MARGIN, BODY_TOP + Inches(3.7), W - 2 * MARGIN, Inches(1.9), size=15)


def s_perf(prs):
    g, d, w, e = (ST.get(k, {}) for k in ("grid", "solve_drift",
                                          "solve_weighting", "efields"))
    tot = sum(r["wall"] for r in ST.values())
    image_slide(prs, "Performance: reference run",
                os.path.join(OUT, "stages.png"), bullets=[
                    f"{META.get('device', '')}, {CFG.get('dtype')}",
                    f"{g.get('nodes', 0)/1e6:.2f} M nodes, "
                    f"{g.get('cells', 0)/1e6:.2f} M cells",
                    f"Drift solve {_f(d.get('wall', 0))} s, {d.get('iters')} "
                    f"its; weighting {_f(w.get('wall', 0))} s, {w.get('iters')} its",
                    f"Peak GPU memory in a solve ≈ "
                    f"{max(d.get('gpu_peak_MiB', 0), w.get('gpu_peak_MiB', 0)):.0f} "
                    f"MiB; overall {e.get('gpu_peak_MiB', 0):.0f} MiB (efields)",
                    f"Total {tot:.1f} s: the solves are ~5 %; gradient assembly, "
                    "grid build and export (CPU NumPy) dominate",
                ], img_frac=0.68)


def s_summary(prs):
    title_bullets(prs, "Summary and outlook", [
        "Exact integer lattice + non-graded octree: geometry is a mask, "
        "resolution goes only where the pad structure lives",
        "Min–Gibou–Ceniceros node scheme with Eq. (8) weights: one composite, "
        "level-free operator, second order in φ and ∇φ, 7-point in the interior",
        "Matrix-free gather-reduce operator + Jacobi-BiCGSTAB in fp64: "
        "~1.4 M-node pixel problem solved in seconds on one GPU",
        "E from the scheme's own corrected gradient → RK4 drift → Shockley–Ramo "
        "currents with an independent charge cross-check → response table",
        "Conformance to the paper asserted by tests/test_paper*.py",
        "Outlook: CPU-side assembly / export now dominate the wall time; "
        "if proven, the method is to be implemented in **pochoir** for "
        "pixelated-readout LArTPCs",
    ])


SLIDES = [s_title, s_problem, s_lattice, s_sizing, s_octree, s_nodes, s_eq8,
          s_bcs, s_operator, s_solver, s_gradient, s_fields, s_drift, s_ramo,
          s_pipeline, s_verify, s_perf, s_summary]


def build(prs):
    """Append every built-in slide to ``prs``."""
    for f in SLIDES:
        f(prs)
    return prs


def _extra(prs):
    path = os.path.join(HERE, "slides_extra.py")
    if not os.path.exists(path):
        return
    spec = importlib.util.spec_from_file_location("slides_extra", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "add_slides"):
        mod.add_slides(prs, HELPERS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "Explore_FDM_methods.pptx"))
    a = ap.parse_args()
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    build(prs)
    _extra(prs)
    prs.save(a.out)
    print(f"wrote {a.out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
