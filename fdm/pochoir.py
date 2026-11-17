"""Read a pochoir pixel-anode JSON config and reproduce its geometry exactly.

pochoir (``gen_pcb_drift_pixel_with_grid.py`` for the drift field,
``gen_pcb_pixel_with_grid.py`` for the weighting field) rasterizes the anode on
a uniform lattice of spacing ``s`` (0.1 mm for ``*_fine.json``, 0.55 mm for
``*_coarse.json``).  Every length in the config is turned into a whole number of
cells with a specific ``int()`` or ``round()``, and the electrodes are *node
sets* on that lattice.  To compare against pochoir, this module performs the
same integer arithmetic and then describes the result as exact integer
geometry on our own lattice ``h_min = s / 2**refine``.  At ``refine = 0`` the
Dirichlet node set on the anode plane is identical to pochoir's, node for node
(``tests/test_pochoir.py`` checks this against pochoir's own generator and
against the boundary arrays of a stored pochoir run).

What pochoir's task13-style configs actually solve, and therefore what we solve:

* **Anode plane** at ``z_a = (pp_loweredge + pp_width) * s`` -- the drift-facing
  top of the pad conductor (10.0 mm for task13 fine).  With
  ``enableInsulatorFR4`` pochoir imposes a *node-centred* zero-flux mirror at the
  gap nodes of this plane (``fdm_generic.padplane_noflux_geom``): the ghost node
  below each gap node is overwritten with the node above.  That is exactly the
  reflected stencil our ``build_rows(neumann=...)`` uses, so the anode is
  "pads Dirichlet, gaps Neumann" on the plane ``z = z_a``.  Everything below
  that plane (the FR4 cell, the extra ``padThicknessCells``, the bulk down to
  pochoir's z = 0) cannot influence the drift side and is not modelled.
* **Pads**: node intervals ``[lo, hi]`` per axis with ``hi - lo = p - 1``
  (``p = round(pixelSize/s)``), i.e. a pad spans ``(p-1) s`` between its outer
  nodes, with the four corners rounded by pochoir's ``trimCorner``: inside an
  ``r x r`` corner box, nodes farther than ``r`` cells from a centre placed
  ``r - 1`` cells in from both edges are removed (``r = round(chamfer_r/s)``).
  On a finer lattice the same rule is applied in continuous coordinates.
* **Drift field**: one pitch tile, ``N = p + g`` nodes, periodic in x and y, one
  pad centred on node 0 (pochoir draws it as four corner quadrants).  Cathode
  plane at the last lattice plane, ``round(driftZDepth/s)``, held at
  ``CathodePotential``.
* **Weighting field**: ``Npixels x Npixels`` pads, ``Nw = Npixels*(p+g)`` nodes,
  pad ``k`` occupying nodes ``g//2 + k(p+g) ... + p - 1``, the centre pad
  (``k = Npixels//2`` in both axes) at 1 and the others at 0.  The transverse
  walls are *held* at 0 (``true_fixed``): pochoir pads the array with a halo,
  so the walls sit on the halo nodes at ``x = -s`` and ``x = Nw s``, not on the
  first and last array nodes.  Cathode plane at 0.

Coordinates are pochoir's own: ``x = i s``, ``y = j s``, ``z = k s`` in mm, in
the drift-tile frame for the drift field and in the weighting frame for the
weighting field, so arrays can be compared index by index.

Unsupported pochoir features raise ``NotImplementedError`` rather than being
approximated: a shield grid (``GridHoleShape`` other than ``"None"``), the
``fixed_cell`` chamfer, ``enableFR4``, and running without
``enableInsulatorFR4`` (pochoir then solves the region below the pad plane as
well, which is a different problem).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _strip(cfg: dict) -> dict:
    """Drop pochoir's prose keys (``_description`` and friends)."""
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


# ----------------------------------------------------------- pochoir integers

@dataclass
class PochoirGeometry:
    """The integer geometry pochoir derives from a config at spacing ``s``.

    All ``*_idx`` and cell counts are pochoir lattice indices (units of ``s``).
    """
    spacing: float          # s, mm
    p: int                  # pad size, nodes
    g: int                  # gap, cells (p + g = nodes per pitch)
    r: int                  # chamfer radius, cells (0 = square corners)
    npixels: int            # weighting probe is npixels x npixels pads
    anode_idx: int          # z index of the drift-facing pad plane
    cathode_idx: int        # z index of the drift cathode plane
    w_cathode_idx: int      # z index of the weighting cathode plane
    cathode_potential: float  # V
    launch_z: float         # mm, electron start height (pochoir frame)
    ngrid: int              # start points per side
    start_spacing: float    # mm between start points
    # provenance, for the record
    pp_loweredge: int = 0
    pp_width: int = 0
    n_fr4: int = 0
    n_pad: int = 0
    pad_cells: int = 1

    # -- derived ------------------------------------------------------------
    @property
    def N(self) -> int:
        """Nodes per pitch (drift tile width, periodic)."""
        return self.p + self.g

    @property
    def Nw(self) -> int:
        """Weighting array width in nodes."""
        return self.npixels * self.N

    @property
    def pitch(self) -> float:
        return self.N * self.spacing

    @property
    def z_anode(self) -> float:
        return self.anode_idx * self.spacing

    @property
    def z_cathode(self) -> float:
        return self.cathode_idx * self.spacing

    @property
    def drift_length(self) -> float:
        return (self.cathode_idx - self.anode_idx) * self.spacing

    @property
    def bulk_field(self) -> float:
        """Nominal |E| (V/mm) between anode plane and cathode."""
        return abs(self.cathode_potential) / self.drift_length

    @property
    def centre_pad(self) -> int:
        return self.npixels // 2

    def drift_pad_nodes(self):
        """(lo, hi) node interval of the drift-tile pad, centred on node 0."""
        half = (self.p + 1) // 2
        return -(self.p - half), half - 1

    def weighting_pad_nodes(self, k: int):
        lo = self.g // 2 + k * self.N
        return lo, lo + self.p - 1

    def weighting_pad_centre(self) -> float:
        """Physical centre (mm) of the collecting pad, per axis.

        Same definition pochoir's ``induce-pixel`` uses: the midpoint of the
        outermost W = 1 nodes.
        """
        lo, hi = self.weighting_pad_nodes(self.centre_pad)
        return 0.5 * (lo + hi) * self.spacing

    def drift_pad_centre(self) -> float:
        lo, hi = self.drift_pad_nodes()
        return 0.5 * (lo + hi) * self.spacing

    def summary(self) -> dict:
        d = asdict(self)
        d.update(N=self.N, Nw=self.Nw, pitch=self.pitch, z_anode=self.z_anode,
                 z_cathode=self.z_cathode, drift_length=self.drift_length,
                 bulk_field_V_per_mm=self.bulk_field,
                 drift_pad_nodes=self.drift_pad_nodes(),
                 weighting_centre_pad_nodes=self.weighting_pad_nodes(
                     self.centre_pad),
                 weighting_pad_centre=self.weighting_pad_centre(),
                 pad_span_mm=(self.p - 1) * self.spacing,
                 gap_span_mm=(self.g + 1) * self.spacing)
        return d


def _check_supported(cfg: dict, which: str):
    shape = cfg.get("GridHoleShape", "None")
    if shape not in ("None", None):
        raise NotImplementedError(
            f"{which}: GridHoleShape={shape!r} (shield grid) is not supported; "
            "only the no-grid geometry is")
    if cfg.get("chamferMode", "dynamic") != "dynamic":
        raise NotImplementedError(
            f"{which}: chamferMode={cfg['chamferMode']!r}; only 'dynamic' is "
            "supported")
    if cfg.get("enableFR4", False):
        raise NotImplementedError(f"{which}: enableFR4 is not supported")
    if not cfg.get("enableInsulatorFR4", False):
        raise NotImplementedError(
            f"{which}: enableInsulatorFR4 is false.  pochoir then relaxes the "
            "gap nodes against the volume below the pad plane, which this "
            "package does not model; only the insulating (no-flux) pad plane "
            "is supported")


def _anode_index(cfg: dict, s: float):
    """pochoir's pad-plane z index, with the generator's exact rounding."""
    pp_loweredge = int(cfg["pixelPlaneLowEdgePosition"] / s)      # int(), not round
    pp_width = int(cfg["pixelPlaneWidth"] / s)
    n_fr4 = n_pad = 0
    if cfg.get("FR4Thickness") is not None and cfg.get("padThickness") is not None:
        n_fr4 = max(1, int(round(cfg["FR4Thickness"] / s)))
        n_pad = max(1, int(round(cfg["padThickness"] / s)))
        pp_width = n_pad + n_fr4 - 1
    else:
        n_fr4 = max(1, pp_width // 2)
    return pp_loweredge, pp_width, n_fr4, n_pad


def from_configs(drift_cfg: dict | str, weighting_cfg: dict | str | None = None,
                 spacing: float = 0.1, cathode_idx: int | None = None,
                 start_spacing: float | None = None) -> PochoirGeometry:
    """Translate pochoir drift (and optionally weighting) configs.

    ``spacing`` is the pochoir grid spacing the configs are written for (it is
    a command-line argument to pochoir, not a config key).  If a weighting
    config is given, the anode geometry it implies must agree with the drift
    config's; a mismatch raises, since the two fields must share one pad
    surface for the drift paths and the weighting potential to be consistent.
    """
    if isinstance(drift_cfg, str):
        drift_cfg = load_json(drift_cfg)
    d = _strip(drift_cfg)
    _check_supported(d, "drift config")
    s = float(spacing)

    p = int(round(d["pixelSize"] / s))
    g = int(round(d["pixelGap"] / s))
    r = int(round(d.get("chamfer_r", 0.0) / s))
    pp_lo, pp_w, n_fr4, n_pad = _anode_index(d, s)
    anode = pp_lo + pp_w
    cat = int(round(d["driftZDepth"] / s)) if cathode_idx is None else int(cathode_idx)
    w_cat = cat

    if weighting_cfg is not None:
        if isinstance(weighting_cfg, str):
            weighting_cfg = load_json(weighting_cfg)
        w = _strip(weighting_cfg)
        _check_supported(w, "weighting config")
        wp = int(round(w["pixelSize"] / s))
        wg = int(round(w["pixelGap"] / s))
        wr = int(round(w.get("chamfer_r", 0.0) / s))
        wlo, ww, _, _ = _anode_index(w, s)
        bad = [(k, a, b) for k, a, b in (("pixelSize cells", p, wp),
                                         ("pixelGap cells", g, wg),
                                         ("chamfer cells", r, wr),
                                         ("pad-plane z index", anode, wlo + ww))
               if a != b]
        if bad:
            raise ValueError("drift and weighting configs disagree on the anode: "
                             + ", ".join(f"{k} {a} vs {b}" for k, a, b in bad))
        npix = int(w["Npixels"])
        if "driftZDepth" in w:
            w_cat = int(round(w["driftZDepth"] / s))
    else:
        npix = int(d["Npixels"])

    if not (0 <= anode < cat):
        raise ValueError(f"pad plane index {anode} is not below cathode {cat}")
    if 2 * r > p:
        raise ValueError(f"chamfer radius {r} cells exceeds half the pad ({p})")

    launch = float(d["launchZ"]) if "launchZ" in d else float(d["driftZDepth"])
    ngrid = int(d.get("nGridPoints", 10))
    if start_spacing is None:
        start_spacing = float(d["gridSpacing"]) if "gridSpacing" in d \
            else (p + g) * s / ngrid

    return PochoirGeometry(
        spacing=s, p=p, g=g, r=r, npixels=npix,
        anode_idx=anode, cathode_idx=cat, w_cathode_idx=w_cat,
        cathode_potential=float(d["CathodePotential"]),
        launch_z=launch, ngrid=ngrid, start_spacing=float(start_spacing),
        pp_loweredge=pp_lo, pp_width=pp_w, n_fr4=n_fr4, n_pad=n_pad,
        pad_cells=int(d.get("padThicknessCells", 1)))


# --------------------------------------------- pad footprints on our lattice

class PadSet:
    """Rounded-corner square pads in integer units of our ``h_min``.

    ``m = s / h_min`` lattice units per pochoir cell.  A pad is the node
    interval ``[lo, hi]`` (pochoir nodes, scaled by ``m``) in x and y, minus the
    four corner regions pochoir's ``trimCorner`` carves.  Membership is decided
    by integer arithmetic only.
    """

    def __init__(self, geom: PochoirGeometry, m: int, pads, period=None):
        self.m = int(m)
        self.r = geom.r * self.m                     # chamfer radius, units
        self.rc = max(geom.r - 1, 0) * self.m        # centre inset, units
        # (npad, 4): xlo, xhi, ylo, yhi in units
        self.pads = np.array([(xl * self.m, xh * self.m, yl * self.m, yh * self.m)
                              for (xl, xh), (yl, yh) in pads], dtype=np.int64)
        self.period = period                         # units, or None

    def _inside_1pad(self, x, y, xl, xh, yl, yh):
        inside = (x >= xl) & (x <= xh) & (y >= yl) & (y <= yh)
        if self.r <= self.m:          # pochoir radius <= 1 cell: carves nothing
            return inside
        r2 = self.r * self.r
        for cx, sx in ((xl + self.rc, -1), (xh - self.rc, +1)):
            for cy, sy in ((yl + self.rc, -1), (yh - self.rc, +1)):
                box = (sx * (x - cx) >= 0) & (sy * (y - cy) >= 0)
                far = (x - cx) ** 2 + (y - cy) ** 2 > r2
                inside &= ~(box & far)
        return inside

    def contains(self, x, y, which=None):
        """True where the lattice point (x, y) [units] is pad metal.

        ``which`` restricts to one pad index.  For a periodic pad set the
        coordinates are wrapped first.
        """
        x = np.asarray(x, np.int64)
        y = np.asarray(y, np.int64)
        out = np.zeros(np.broadcast(x, y).shape, bool)
        idx = range(len(self.pads)) if which is None else [which]
        for k in idx:
            xl, xh, yl, yh = self.pads[k]
            if self.period is None:
                out |= self._inside_1pad(x, y, xl, xh, yl, yh)
            else:
                P = self.period
                # wrap into [lo, lo + P); a pad never spans more than a period
                xx = xl + np.mod(x - xl, P)
                yy = yl + np.mod(y - yl, P)
                out |= self._inside_1pad(xx, yy, xl, xh, yl, yh)
        return out

    def near_edge(self, cx, cy, band):
        """True where a point (float units) is within ``band`` of a pad edge.

        Used only to drive refinement, so it errs on the generous side: both
        the straight edges and the corner arcs count as edges.
        """
        cx = np.asarray(cx, float)
        cy = np.asarray(cy, float)
        band = np.asarray(band, float)
        out = np.zeros(np.broadcast(cx, cy).shape, bool)
        for xl, xh, yl, yh in self.pads:
            if self.period is None:
                shifts = [(0.0, 0.0)]
                x, y = cx, cy
            else:
                P = float(self.period)
                x = xl + np.mod(cx - xl, P)
                y = yl + np.mod(cy - yl, P)
                shifts = [(a * P, b * P) for a in (-1, 0, 1) for b in (-1, 0, 1)]
            for sx, sy in shifts:
                xx, yy = x - sx, y - sy
                # distance to the rectangle's boundary (unsigned)
                dx = np.maximum(np.maximum(xl - xx, xx - xh), 0.0)
                dy = np.maximum(np.maximum(yl - yy, yy - yh), 0.0)
                outside = np.hypot(dx, dy)
                inside = np.minimum(np.minimum(xx - xl, xh - xx),
                                    np.minimum(yy - yl, yh - yy))
                d = np.where((dx > 0) | (dy > 0), outside, np.abs(inside))
                hit = d < band
                if self.r > self.m:
                    for ccx in (xl + self.rc, xh - self.rc):
                        for ccy in (yl + self.rc, yh - self.rc):
                            hit |= np.abs(np.hypot(xx - ccx, yy - ccy)
                                          - self.r) < band
                out |= hit
        return out


def drift_pads(geom: PochoirGeometry, m: int) -> PadSet:
    iv = geom.drift_pad_nodes()
    return PadSet(geom, m, [(iv, iv)], period=geom.N * m)


def weighting_pads(geom: PochoirGeometry, m: int) -> PadSet:
    """All pads of the weighting probe; index 0 is the collecting centre pad."""
    c = geom.centre_pad
    order = [(c, c)] + [(i, j) for i in range(geom.npixels)
                        for j in range(geom.npixels) if (i, j) != (c, c)]
    return PadSet(geom, m, [(geom.weighting_pad_nodes(i),
                             geom.weighting_pad_nodes(j)) for i, j in order])


def pochoir_start_points(geom: PochoirGeometry) -> np.ndarray:
    """pochoir's ``starts -m yes`` grid, in its order (x outer, y inner).

    Cell-centred: ``x = sp/2 + j sp``, ``y = sp/2 + i sp``, at ``launchZ``.
    Note pochoir's ``make_pixel_start_points`` currently hardcodes
    ``sp = 0.44 mm``; we use ``gridSpacing`` or ``pitch / nGridPoints``, which
    is the same number for a 4.4 mm pitch and 10 points.
    """
    sp = geom.start_spacing
    half = 0.5 * sp
    pts = [[half + j * sp, half + i * sp, geom.launch_z]
           for j in range(geom.ngrid) for i in range(geom.ngrid)]
    return np.array(pts, dtype=np.float64)


def bnl_drift_velocity(E_V_per_mm, temperature=87.0):
    """Electron drift speed (mm/us) with pochoir's mobility (``pochoir/lar.py``).

    BNL parameterization, https://lar.bnl.gov/properties/trans.html, with the
    same constants and temperature scaling as pochoir, so drift times are
    comparable.  ``E`` in V/mm.
    """
    E = np.maximum(np.asarray(E_V_per_mm, np.float64), 1e-12) / 100.0   # kV/cm
    Trel = temperature / 89.0
    a0, a1, a2, a3, a4, a5 = 551.6, 7158.3, 4440.43, 4.29, 43.63, 0.2053
    mu = (a0 + a1 * E + a2 * E**1.5 + a3 * E**2.5)
    mu /= (1 + (a1 / a0) * E + a4 * E**2 + a5 * E**3) * Trel**1.5   # cm2/s/V
    v_cm_per_s = mu * E * 1000.0                                       # E in V/cm
    return v_cm_per_s * 1e-5                                           # mm/us


def root_level(period_units: int, lmax_cap: int) -> int:
    """Largest l <= lmax_cap with 2**l dividing ``period_units``."""
    l = 0
    while l < lmax_cap and period_units % (1 << (l + 1)) == 0:
        l += 1
    return l


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def describe(geom: PochoirGeometry) -> str:
    s = geom.spacing
    lo, hi = geom.drift_pad_nodes()
    return "\n".join([
        f"pochoir spacing {s} mm: pad {geom.p} nodes ({(geom.p-1)*s:.3f} mm "
        f"node-to-node), gap {geom.g} cells, pitch {geom.N} cells = "
        f"{geom.pitch:.4f} mm, chamfer r = {geom.r} cells",
        f"anode (pad top, Neumann gaps) z = {geom.z_anode:.4f} mm "
        f"[index {geom.anode_idx} = {geom.pp_loweredge} + {geom.pp_width}]",
        f"cathode z = {geom.z_cathode:.4f} mm [index {geom.cathode_idx}] at "
        f"{geom.cathode_potential} V -> {geom.bulk_field:.4f} V/mm over "
        f"{geom.drift_length:.4f} mm",
        f"drift tile: {geom.N}x{geom.N} periodic, pad nodes {lo}..{hi}",
        f"weighting probe: {geom.npixels}x{geom.npixels} pads, "
        f"{geom.Nw}x{geom.Nw} nodes, walls held at 0 at x = {-s:.3f} and "
        f"{geom.Nw*s:.3f} mm, collecting pad centre "
        f"{geom.weighting_pad_centre():.4f} mm",
        f"starts: {geom.ngrid}x{geom.ngrid} at {geom.start_spacing:.4f} mm, "
        f"z = {geom.launch_z} mm",
    ])


__all__ = ["PochoirGeometry", "from_configs", "PadSet", "drift_pads",
           "weighting_pads", "pochoir_start_points", "bnl_drift_velocity",
           "root_level", "ceil_div", "describe"]
