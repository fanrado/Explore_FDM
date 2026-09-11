#!/usr/bin/env python3
"""Plot everything a workflow run leaves in out/.

    python3 plot_outputs.py                 # reads out/, writes out/*.png
    python3 plot_outputs.py --outdir out --dpi 130

One figure per product, each named after the .npz it came from:

    drift_field.png       potential and E of the drift solution
    weighting_field.png   the Ramo weighting potential of the centre pad
    near_anode.png        the fine grid where the pad structure lives
    trajectories.png      the drifted electron paths
    induced_current.png   current / charge waveforms (see plot_induced_current.py)
    response.png          the 2D field response, impact position x time
    stages.png            wall time and GPU memory per stage

Conventions, kept the same as plot_induced_current.py so the set reads as one
document:

* Signed quantities (a potential that goes negative, a transverse field, a
  bipolar induced current) get a DIVERGING map with a neutral midpoint pinned
  at zero, so the sign is readable without consulting the colorbar.
* Unsigned magnitudes get a single-hue SEQUENTIAL ramp.
* Never two y-scales on one axes; a second measure gets its own panel.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LogNorm

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SEQ, DIV = "Blues", "RdBu_r"
ACCENT = "#1f5fa9"


def style(ax, xlabel, ylabel, title=None):
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, pad=8, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(label, color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=INK2, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)
    return cb


def suptitle(fig, d, text):
    fig.suptitle(
        f"{text}     pitch {float(d['pitch'])} mm, pad {float(d['pad_width']):.2f} mm, "
        f"gap {float(d['gap']):.2f} mm, {float(d['efield_Vcm']):.0f} V/cm",
        color=INK, fontsize=11.5, y=1.04, x=0.01, ha="left")


def save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def diverging(a):
    """Norm centred on zero, symmetric, robust to a few extreme cells."""
    m = float(np.percentile(np.abs(a), 99.5)) or float(np.abs(a).max()) or 1.0
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


# ---------------------------------------------------------------- coarse grids

def plot_coarse(path, out, what, dpi):
    """drift_field_grid.npz / weighting_field_grid.npz: phi and E on (x,y,z)."""
    d = np.load(path)
    x, y, z, phi, E = d["x"], d["y"], d["z"], d["phi"], d["E"]
    jy = len(y) // 2                      # slice through the pad centre row
    Emag = np.linalg.norm(E, axis=-1)
    pitch = float(d["pitch"])
    drift = what == "Drift"

    # The pad structure lives within a couple of pitches of the anode; beyond
    # that both solutions are one-dimensional.  Map that region, profile the rest.
    zmax = min(z[-1], 3 * pitch)
    kz = z <= zmax

    fig, ax = plt.subplots(1, 4, figsize=(19, 4.0), constrained_layout=True)

    # The drift potential is dominated by the -E0*z ramp, which would flatten
    # everything else to one colour, so map the DEVIATION from the plane-parallel
    # solution - that is the part the pads actually cause.  The weighting
    # potential has no such background and is mapped as it stands.
    if drift:
        field = phi[:, jy, kz] - phi[:, :, kz].mean(axis=(0, 1))
        lab, ttl = "phi - <phi>(z)  (V)", "Drift potential, pad-induced part"
        norm, cmap = diverging(field), DIV
    else:
        field = phi[:, jy, kz]
        lab, ttl = "weighting phi (1)", "Weighting potential"
        norm, cmap = Normalize(0.0, max(field.max(), 1e-12)), SEQ
    im = ax[0].pcolormesh(z[kz], x, field, cmap=cmap, norm=norm, shading="nearest")
    style(ax[0], "z, distance from anode (mm)", "x (mm)", f"{ttl}, y = {y[jy]:.2f} mm")
    cbar(fig, im, ax[0], lab)

    # |E| spans orders of magnitude only right at the pad edges; a linear ramp
    # keeps the bulk readable and a 99.5th-percentile top keeps one hot cell
    # from eating the scale.
    em = Emag[:, jy, kz]
    im = ax[1].pcolormesh(z[kz], x, em, cmap=SEQ, shading="nearest",
                          norm=Normalize(0.0, float(np.percentile(em, 99.5))))
    style(ax[1], "z, distance from anode (mm)", "x (mm)", "|E|, same slice")
    cbar(fig, im, ax[1], "|E| (V/mm)")

    # Axial profiles over the FULL drift, at the two transverse extremes.
    ix_pad, ix_gap = len(x) // 2, 0
    for k, lab2, c in ((ix_pad, "above pad centre", ACCENT),
                       (ix_gap, "above pad edge/gap", INK2)):
        ax[2].plot(z, phi[k, jy, :], lw=1.8, color=c, label=lab2)
        ax[3].plot(z, Emag[k, jy, :], lw=1.8, color=c, label=lab2)
    style(ax[2], "z (mm)", "phi (V)" if drift else "weighting phi (1)",
          "Axial potential, full drift")
    # The field only departs from uniform within a pitch or two of the pads, so
    # the magnitude panel zooms there while the potential panel keeps the full
    # drift; their x-axes deliberately differ and both say so.
    style(ax[3], "z (mm), near-anode zoom", "|E| (V/mm)", "Axial field magnitude")
    if drift:
        ax[3].set_xlim(0, zmax)
    else:
        # The weighting quantities fall off over many decades; log both axes and
        # floor the potential where it stops meaning anything.
        for a in (ax[2], ax[3]):
            a.set_xscale("log")
            a.set_yscale("log")
        ax[2].set_xlim(z[1], z[-1])
        ax[2].set_ylim(bottom=1e-6)
        ax[3].set_xlim(z[1], zmax)
    for a in (ax[2], ax[3]):
        a.legend(frameon=False, fontsize=8, labelcolor=INK2)

    suptitle(fig, d, f"{what} field solution")
    save(fig, out, dpi)


# ------------------------------------------------------------ near-anode grid

def emag_norm(a):
    """Sequential norm for a magnitude: log only when it really spans decades."""
    hi, mid = float(np.percentile(a, 99.5)), float(np.median(a))
    # Log only when the TYPICAL cell is decades below the top; a magnitude that
    # is uniform apart from a thin hot layer reads better linear.
    if mid > 0 and hi / mid > 1e2:
        lo = max(float(np.percentile(a[a > 0], 1)), hi * 1e-6)
        return LogNorm(vmin=lo, vmax=hi)
    return Normalize(0.0, hi)


def plot_near_anode(path, out, dpi):
    d = np.load(path)
    x, y, z = d["x"], d["y"], d["z"]
    jy = len(y) // 2
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.2), constrained_layout=True)

    for r, (pk, ek, name, unit) in enumerate((
            ("phi_drift", "E_drift", "Drift", "V"),
            ("phi_weight", "E_weight", "Weighting", "1"))):
        phi, E = d[pk], d[ek]
        Emag = np.linalg.norm(E, axis=-1)
        drift = name == "Drift"

        # Same rule as the coarse figure: strip the -E0*z ramp off the drift
        # potential so what remains is the pad structure, and map the weighting
        # potential (0 to 1, one sign) on a sequential ramp.
        if drift:
            sl = phi[:, jy, :] - phi.mean(axis=(0, 1))
            plan = phi[:, :, 0] - phi[:, :, 0].mean()
            nsl, npl, cm = diverging(sl), diverging(plan), DIV
            plab = "phi - <phi>(z)  (V)"
            ttl = "Drift potential, pad-induced part"
        else:
            sl, plan = phi[:, jy, :], phi[:, :, 0]
            nsl = npl = Normalize(0.0, 1.0)
            cm, plab, ttl = SEQ, "weighting phi (1)", "Weighting potential"

        im = ax[r, 0].pcolormesh(z, x, sl, cmap=cm, norm=nsl, shading="nearest")
        if not drift:        # contours of a near-flat field are just noise
            ax[r, 0].contour(z, x, sl, levels=10, colors=INK2,
                             linewidths=0.4, alpha=0.55)
        style(ax[r, 0], "z (mm)", "x (mm)", f"{ttl}, y = {y[jy]:.2f} mm")
        cbar(fig, im, ax[r, 0], plab)

        im = ax[r, 1].pcolormesh(z, x, Emag[:, jy, :], cmap=SEQ,
                                 norm=emag_norm(Emag[:, jy, :]), shading="nearest")
        style(ax[r, 1], "z (mm)", "x (mm)", f"{name} |E|, same slice")
        cbar(fig, im, ax[r, 1], "|E| (V/mm)")

        # Looking down on the anode plane: the pad footprint in plan view.
        im = ax[r, 2].pcolormesh(x, y, plan.T, cmap=cm, norm=npl, shading="nearest")
        ax[r, 2].set_aspect("equal")
        style(ax[r, 2], "x (mm)", "y (mm)", f"{ttl} at the anode, z = 0")
        cbar(fig, im, ax[r, 2], plab)

    suptitle(fig, d, "Near-anode fine grid "
                     f"({len(x)}x{len(y)}x{len(z)} at {float(d['spacing']):.3f} mm)")
    save(fig, out, dpi)


# --------------------------------------------------------------- trajectories

def plot_trajectories(path, out, dpi):
    d = np.load(path)
    traj, n = d["traj"], d["traj_len"]
    coll, z0 = d["collected"], d["start"][:, 2]
    start, end = d["start"], d["end"]
    half = float(d["pad_width"]) / 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    norm = Normalize(z0.min(), z0.max())
    cmap = plt.get_cmap(SEQ)

    # A path is 250 mm long and wanders a few tens of microns, so plotting raw x
    # against z draws 100 vertical lines and says nothing.  Plot the transverse
    # DISPLACEMENT from each path's own start, in microns: that is the whole
    # physics of the panel, and it only appears in the last millimetre of drift.
    for k in range(len(traj)):
        p = traj[k, :int(n[k])]
        c = cmap(0.30 + 0.70 * norm(z0[k]))
        ax[0].plot(1e3 * (p[:, 0] - p[0, 0]), p[:, 2], color=c, lw=0.8, alpha=0.85)
    ax[0].set_yscale("log")
    ax[0].set_ylim(float(d["h_min"]), z0.max())
    ax[0].axvline(0, color=INK2, lw=1.0)
    style(ax[0], "x - x$_0$ (um)", "z, distance from anode (mm), log",
          f"Transverse pull vs height  (n={len(traj)})")

    # Where each electron lands, relative to the pad it was aimed at.
    cx, cy = end[coll][:, 0].mean(), end[coll][:, 1].mean()
    pad = plt.Rectangle((-half, -half), 2 * half, 2 * half, fill=False,
                        ec=INK2, lw=1.2, ls="--")
    ax[1].add_patch(pad)
    ax[1].scatter(end[coll][:, 0] - cx, end[coll][:, 1] - cy, s=26, c=ACCENT,
                  label="collected on the pad")
    ax[1].scatter(end[~coll][:, 0] - cx, end[~coll][:, 1] - cy, s=26,
                  facecolors="none", edgecolors=INK2, lw=0.8,
                  label="landed on a neighbour")
    ax[1].set_aspect("equal")
    style(ax[1], "landing x, relative to the centre pad (mm)", "landing y (mm)",
          "Landing points (dashed: one pad)")
    ax[1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    ax[2].scatter(z0[coll], d["t_arrival"][coll], s=16, c=ACCENT,
                  label="collected on the pad")
    ax[2].scatter(z0[~coll], d["t_arrival"][~coll], s=16, facecolors="none",
                  edgecolors=INK2, lw=0.8, label="neighbouring pad")
    v = float(d["v_drift_bulk"])
    zz = np.linspace(z0.min(), z0.max(), 2)
    ax[2].plot(zz, zz / v, color=INK2, lw=1.0, ls="--")
    ax[2].annotate(f"z / v_bulk, v = {v:.3f} mm/us", (zz[1], zz[1] / v),
                   xytext=(-4, 10), textcoords="offset points",
                   ha="right", color=INK2, fontsize=8)
    style(ax[2], "start height z$_0$ (mm)", "arrival time (us)",
          "Drift time vs start height")
    ax[2].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")

    suptitle(fig, d, "Drifted electron trajectories")
    save(fig, out, dpi)


# ------------------------------------------------------------------- response

def plot_response(path, out, dpi):
    d = np.load(path)
    R, t = d["response"], d["t"]
    ny, nx = (int(v) for v in d["impact_shape"])
    start, cxy = d["start"], d["centre_pad_xy"]
    coll = d["collected"]

    # Impact position of each path, relative to the pad centre, in pitch units.
    pitch = float(d["pitch"])
    dx = (start[:, 0] - cxy[0]) / pitch
    dy = (start[:, 1] - cxy[1]) / pitch
    r = np.hypot(dx, dy)

    # Every waveform is flat until the electron is within a pitch of the pad, so
    # the panels that show time zoom to the window that carries the signal.
    act = np.abs(R).sum(0)
    live = np.flatnonzero(act > 1e-3 * act.max())
    t0, t1 = t[live[0]], t[live[-1]]
    pad_t = 0.1 * (t1 - t0)
    win = (t0 - pad_t, t1 + pad_t)

    fig = plt.figure(figsize=(16, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # The response proper: every impact's waveform as one row, ordered by how
    # far off-centre it started.  Bipolar, so a diverging map centred on zero.
    ax = fig.add_subplot(gs[0, :2])
    o = np.argsort(r)
    im = ax.pcolormesh(t, np.arange(len(o)), R[o], cmap=DIV,
                       norm=diverging(R), shading="nearest")
    ax.set_xlim(*win)
    style(ax, "time (us), zoomed to the active window",
          "impact, sorted by distance from pad centre",
          f"Field response, {R.shape[0]} impacts x {R.shape[1]} ticks "
          f"({float(d['tick'])} us); flat outside {win[0]:.1f}-{win[1]:.1f} us")
    cbar(fig, im, ax, f"current per {str(d['scale'])} (arb.)")

    # Waveforms, grouped in radius bands: the shape change is what matters.
    ax = fig.add_subplot(gs[0, 2])
    edges = np.linspace(0, r.max(), 6)
    cmap = plt.get_cmap(SEQ)
    for b in range(len(edges) - 1):
        m = (r >= edges[b]) & (r < edges[b + 1] + (b == len(edges) - 2) * 1e9)
        if not m.any():
            continue
        ax.plot(t, R[m].mean(0), lw=1.8,
                color=cmap(0.30 + 0.70 * b / (len(edges) - 2)),
                label=f"{edges[b]:.2f}-{edges[b+1]:.2f} pitch  (n={int(m.sum())})")
    ax.axhline(0, color=INK2, lw=1.0)
    ax.set_xlim(*win)
    style(ax, "time (us)", "mean response (arb.)", "Mean waveform by impact radius")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)

    # Where the paths started, and what each one delivered.
    # Only the impacts that land on the centre pad deposit charge on it; the
    # rest are drawn as empty markers rather than forced onto the same ramp.
    ax = fig.add_subplot(gs[1, 0])
    q = np.abs(d["q_total"])
    ax.scatter(dx[~coll], dy[~coll], s=10, facecolors="none", edgecolors=GRID,
               lw=0.7, label="not collected here")
    sc = ax.scatter(dx[coll], dy[coll], c=q[coll], s=18, cmap=SEQ,
                    norm=Normalize(0.0, float(q[coll].max())), label="collected")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    style(ax, "impact x (pitch)", "impact y (pitch)", "Induced charge by impact")
    cbar(fig, sc, ax, "|Q| (fC)")

    ax = fig.add_subplot(gs[1, 1])
    sc = ax.scatter(dx, dy, c=d["t_drift"], s=14, cmap=SEQ)
    ax.set_aspect("equal")
    style(ax, "impact x (pitch)", "impact y (pitch)", "Drift time by impact")
    cbar(fig, sc, ax, "t_drift (us)")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, act, lw=1.8, color=ACCENT)
    ax.set_xlim(*win)
    style(ax, "time (us)", "sum |response| (arb.)",
          "Where in time the response lives")
    ax.axhline(0, color=INK2, lw=1.0)

    suptitle(fig, d, f"Field response, {int(d['paths_per_pixel'])} paths/pixel "
                     f"on a {nx}x{ny} impact grid, start z = {float(d['start_z']):.1f} mm")
    fig.text(0.01, -0.02, f"collected {int(coll.sum())}/{len(coll)}   "
                          f"arrived {int(d['arrived'].sum())}/{len(coll)}",
             color=INK2, fontsize=8.5)
    save(fig, out, dpi)


# --------------------------------------------------------------------- stages

def plot_stages(path, out, dpi):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    meta = next((r for r in recs if r["kind"] == "meta"), {})
    st = [r for r in recs if r["kind"] == "stage"]
    if not st:
        print(f"no stage records in {path}, skipping")
        return
    names = [r["stage"] for r in st]
    y = np.arange(len(st))[::-1]

    fig, ax = plt.subplots(1, 3, figsize=(15, 0.45 * len(st) + 2.6),
                           constrained_layout=True)
    ax[0].barh(y, [r["wall"] for r in st], color=ACCENT, height=0.62)
    for k, r in enumerate(st):
        ax[0].text(r["wall"], y[k], f" {r['wall']:.1f}", va="center",
                   color=INK2, fontsize=8)
    style(ax[0], "wall time (s)", "", "Time per stage")

    ax[1].barh(y, [r.get("gpu_peak_MiB", 0) for r in st], color=ACCENT, height=0.62)
    style(ax[1], "peak GPU memory (MiB)", "", "GPU high-water mark per stage")

    ax[2].plot([r["t_end"] for r in st], [r["rss_MiB"] for r in st],
               marker="o", ms=5, lw=1.8, color=ACCENT)
    style(ax[2], "elapsed (s)", "host RSS (MiB)", "Host memory over the run")

    for a in (ax[0], ax[1]):
        a.set_yticks(y)
        a.set_yticklabels(names, color=INK2, fontsize=8.5)
        a.grid(axis="y", visible=False)

    total = st[-1]["t_end"] - st[0]["t_start"]
    fig.suptitle(f"Run profile     {total:.1f} s total on "
                 f"{meta.get('device', 'unknown device')}, "
                 f"{meta.get('config', {}).get('dtype', '')}",
                 color=INK, fontsize=11.5, y=1.03, x=0.01, ha="left")
    save(fig, out, dpi)


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="out", help="read inputs and write PNGs here")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of: drift weighting near_anode trajectories "
                         "current response stages")
    a = ap.parse_args()
    o = a.outdir

    def p(name):
        return os.path.join(o, name)

    jobs = {
        "drift": lambda: plot_coarse(p("drift_field_grid.npz"),
                                     p("drift_field.png"), "Drift", a.dpi),
        "weighting": lambda: plot_coarse(p("weighting_field_grid.npz"),
                                         p("weighting_field.png"), "Weighting", a.dpi),
        "near_anode": lambda: plot_near_anode(p("near_anode_grid.npz"),
                                              p("near_anode.png"), a.dpi),
        "trajectories": lambda: plot_trajectories(p("induced_current.npz"),
                                                  p("trajectories.png"), a.dpi),
        "current": lambda: plot_current(p("induced_current.npz"),
                                        p("induced_current.png"), a.dpi),
        "response": lambda: plot_response(p("response.npz"),
                                          p("response.png"), a.dpi),
        "stages": lambda: plot_stages(p("stages.jsonl"), p("stages.png"), a.dpi),
    }
    for name, fn in jobs.items():
        if a.only and name not in a.only:
            continue
        fn()


def plot_current(path, out, dpi):
    """Delegate to the dedicated script so there is one definition of this figure."""
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([sys.executable, os.path.join(here, "plot_induced_current.py"),
                    "--npz", path, "--out", out], check=True)


if __name__ == "__main__":
    main()
