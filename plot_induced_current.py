#!/usr/bin/env python3
"""Example: plot the induced current from out/induced_current.npz

    python3 plot_induced_current.py                       # writes induced_current.png
    python3 plot_induced_current.py --npz out/induced_current.npz --show

The minimal version is three lines:

    d = np.load("out/induced_current.npz")
    plt.plot(d["t"], d["i"][d["collected"]].T)
    plt.xlabel("t - t_arrival (us)"); plt.ylabel("induced current (pA)")

Everything below is presentation.  Two choices worth copying:

* Current and cumulative charge go in SEPARATE panels sharing the x-axis, not on
  twin y-axes.  A dual-axis chart lets the reader infer a crossing point that
  depends only on the arbitrary relative scaling of the two axes.
* The waveforms differ by a continuous parameter (how far the electron started
  from the anode), not by identity, so they are colored with a single-hue
  sequential ramp plus a colorbar rather than a categorical cycle.
"""
from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, color=INK, fontsize=11, pad=8, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="out/induced_current.npz")
    ap.add_argument("--out", default="induced_current.png")
    ap.add_argument("--window", type=float, default=None,
                    help="us before arrival to display (default: the full "
                         "padded record, so the pulse is never clipped)")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    d = np.load(a.npz)
    t = d["t"]                      # us, 0 == arrival
    i = d["i"]                      # pA,  (npath, nt)
    q = d["q"]                      # fC,  (npath, nt) cumulative
    coll = d["collected"]           # landed on the centre pad
    z0 = d["start"][:, 2]           # mm, starting height
    e = float(d["electron_charge_fC"])

    # Color by starting height: a magnitude, so one hue light -> dark.
    norm = Normalize(vmin=z0.min(), vmax=z0.max())
    cmap = plt.get_cmap("Blues")
    def col(k):                     # avoid the near-white low end
        return cmap(0.30 + 0.70 * norm(z0[k]))

    # Show the whole padded record by default.  The saved window is already
    # wider than the longest drift on both sides, so nothing is cut off.
    keep = np.ones_like(t, bool) if a.window is None else (t >= -a.window)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.3), constrained_layout=True)

    for k in np.flatnonzero(coll):
        ax[0].plot(t[keep], i[k][keep], color=col(k), lw=1.6)
    style(ax[0], "t - t_arrival (us)", "induced current (pA)",
          f"Collected on the pad  (n={int(coll.sum())})")

    for k in np.flatnonzero(coll):
        ax[1].plot(t[keep], q[k][keep] / e, color=col(k), lw=1.6)
    ax[1].axhline(1.0, color=INK2, lw=1.0, ls="--")
    ax[1].annotate("one electron", (t[keep][0], 1.0), xytext=(4, -12),
                   textcoords="offset points", color=INK2, fontsize=8)
    style(ax[1], "t - t_arrival (us)", "cumulative induced charge  Q/e",
          "Charge collected")

    nb = np.flatnonzero(~coll)
    for k in nb[np.argsort(z0[nb])][::4]:
        ax[2].plot(t[keep], i[k][keep], color=col(k), lw=1.0, alpha=0.85)
    ax[2].axhline(0.0, color=INK2, lw=1.0)
    style(ax[2], "t - t_arrival (us)", "induced current (pA)",
          f"Neighbouring pads  (n={len(nb)}, every 4th shown)")

    # Absolute time: what a readout sees.  Each pulse sits at its own drift
    # time, and the padding either side keeps every shape complete.
    ta, ia = d["t_abs"], d["i_abs"]
    for k in np.argsort(z0)[::4]:
        ax[3].plot(ta, ia[k], color=col(k), lw=1.0, alpha=0.9)
    ax[3].axhline(0.0, color=INK2, lw=1.0)
    style(ax[3], "absolute time since emission (us)", "induced current (pA)",
          "All pads, absolute time (every 4th)")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, shrink=0.85, pad=0.012)
    cb.set_label("electron start height  z$_0$ (mm)", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_edgecolor(GRID)

    fig.suptitle(
        f"Shockley-Ramo induced current on one pixel     "
        f"pitch {float(d['pitch'])} mm, gap {float(d['gap']):.2f} mm, "
        f"{float(d['efield_Vcm']):.0f} V/cm, drift {float(d['drift_length']):.0f} mm",
        color=INK, fontsize=11.5, y=1.06, x=0.01, ha="left")
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {a.out}")

    print(f"\n  collected {int(coll.sum())}/{len(coll)}   "
          f"peak |i| {np.abs(i[coll]).max():.3f} pA   "
          f"Q/e = {np.mean(q[coll][:, -1])/e:.4f}")
    print(f"  neighbour |Q|/e max {np.max(np.abs(q[~coll][:, -1]))/e:.2e}   "
          f"drift times {d['t_arrival'].min():.1f}-{d['t_arrival'].max():.1f} us")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
