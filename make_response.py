#!/usr/bin/env python3
"""Induced-current field response on a regular grid of impact positions.

Matches the layout of pochoir's
``fr_<pitch>_<npix>_nogrid_<n>pathsperpixel.npy``: a single 2-D float64 array,
one row per drift path, one column per uniform time tick, holding the charge
induced in that tick.  Paths are launched from a regular transverse grid at a
fixed height, ordered with y varying fastest.

    python3 make_response.py --fields out --tick 0.05 --paths-per-pixel 10

Writes ``response.npz`` (key ``response`` plus everything needed to interpret
it) and ``response.npy`` (the bare array, drop-in for the pochoir file).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from fdm.drift import (E_CHARGE, Interpolator, drift_paths, drift_velocity,
                       induced_current)
from fdm.io import load_field


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fields", default="out",
                    help="directory holding drift_field.npz / weighting_field.npz")
    ap.add_argument("--out", default="out/response")
    ap.add_argument("--tick", type=float, default=0.05, help="tick period, us")
    ap.add_argument("--paths-per-pixel", type=int, default=10,
                    help="impact points per pitch, per axis")
    ap.add_argument("--npix", type=float, default=2.4,
                    help="transverse extent of the impact grid, in pitches")
    ap.add_argument("--start-z", type=float, default=None,
                    help="launch height, mm (default: 0.75 of the drift length)")
    ap.add_argument("--pad-after", type=float, default=2.0,
                    help="us of tail kept after the last arrival")
    ap.add_argument("--grid-offset", type=float, default=0.5,
                    help="shift the impact grid by this fraction of a step, to "
                         "keep points off the gap symmetry surfaces where an "
                         "electron would stall (0 = centred on the pad)")
    ap.add_argument("--scale", choices=["electron", "fC", "pA"],
                    default="electron",
                    help="response units: charge/tick in electrons (default, so "
                         "response.sum(1) is the induced charge in units of e), "
                         "charge/tick in fC, or mean current in the tick in pA")
    a = ap.parse_args()

    tree, dd = load_field(f"{a.fields}/drift_field.npz")
    _, dw = load_field(f"{a.fields}/weighting_field.npz")
    Ed, Ew, wpot = dd["E"], dw["E"], dw["phi"]
    pitch = float(dd["pitch"])
    LD = float(dd["drift_length"])
    cx, cy = dw["centre_pad_xy"]
    z0 = a.start_z if a.start_z is not None else 0.75 * LD

    # Regular impact grid centred on the pad, y fastest -- the same ordering as
    # the reference file, so index = ix * n + iy.
    n = int(round(a.npix * a.paths_per_pixel)) + 1
    span = a.npix * pitch
    step = span / (n - 1)
    ax_ = np.linspace(-span / 2, span / 2, n) + a.grid_offset * step
    gx, gy = np.meshgrid(cx + ax_, cy + ax_, indexing="ij")
    starts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z0)], 1)
    print(f"impact grid {n}x{n} = {len(starts)} paths, spacing "
          f"{pitch/a.paths_per_pixel*1000:.1f} um ({a.paths_per_pixel}/pitch), "
          f"z0 = {z0:.3f} mm of {LD:.1f} mm")

    vnom = drift_velocity(float(dd["efield_Vcm"]) / 1000)
    tmax_allowed = 5.0 * z0 / vnom
    t0 = time.time()
    paths, arrived = drift_paths(tree, Ed, starts, z_stop=0.0,
                                 max_time=tmax_allowed, return_status=True)
    if not arrived.all():
        bad = np.flatnonzero(~arrived)
        print(f"  WARNING: {bad.size} of {len(paths)} paths did not reach the "
              f"anode within {tmax_allowed:.0f} us (gap-symmetry stalls); "
              f"their rows are zeroed and flagged in 'arrived'")
    interp = Interpolator(tree)          # built once, reused for every path
    cur = [induced_current(tree, wpot, Ew, p, interp=interp) for p in paths]
    print(f"tracked + induced in {time.time()-t0:.1f}s")

    # Uniform tick grid on absolute time (emission at 0), wider than the longest
    # drift so no pulse is clipped.  response[k, j] is the charge induced in
    # tick j, i.e. the difference of the cumulative induced charge across it.
    tdrift = np.array([c[0][-1] - c[0][0] for c in cur])
    # Size the record from the paths that actually arrived, so one stalled path
    # cannot inflate the tick grid.
    nt = int(np.ceil((tdrift[arrived].max() + a.pad_after) / a.tick)) + 1
    edges = a.tick * np.arange(nt)                    # tick boundaries, us
    Q = np.zeros((len(paths), nt))
    for k, (t, i_r, q_r, q_p) in enumerate(cur):
        if arrived[k]:
            Q[k] = np.interp(edges, t - t[0], q_p, left=0.0, right=q_p[-1])

    resp = np.diff(Q, axis=1) / E_CHARGE              # electrons per tick
    if a.scale == "fC":
        resp *= E_CHARGE
    elif a.scale == "pA":
        resp *= E_CHARGE * 1e3 / a.tick               # fC/us -> pA

    end = np.array([p[-1, :3] for p in paths])
    qtot = Q[:, -1] / E_CHARGE
    coll = qtot > 0.5
    meta = dict(
        response=resp,                                # (npath, ntick) float64
        t=0.5 * (edges[1:] + edges[:-1]),             # tick centres, us
        tick=np.float64(a.tick),
        start=starts, end=end,
        q_total=qtot, collected=coll, arrived=arrived,
        t_drift=tdrift,
        impact_shape=np.array([n, n]),                # response.reshape(n,n,-1)
        paths_per_pixel=np.int32(a.paths_per_pixel),
        centre_pad_xy=np.array([cx, cy]),
        start_z=np.float64(z0),
        scale=a.scale,
        electron_charge_fC=np.float64(E_CHARGE),
        v_drift_bulk=np.float64(drift_velocity(float(dd["efield_Vcm"]) / 1000)),
    )
    for k in ("pitch", "pad_width", "gap", "h_min", "npad", "drift_length",
              "efield_Vcm"):
        meta[k] = dd[k]
    np.savez_compressed(f"{a.out}.npz", **meta)
    np.save(f"{a.out}.npy", resp)

    unit = {"electron": "e/tick", "fC": "fC/tick", "pA": "pA"}[a.scale]
    print(f"\nwrote {a.out}.npz  (key 'response')  and  {a.out}.npy")
    print(f"  response {resp.shape} {resp.dtype}, {unit}, tick {a.tick} us")
    print(f"  drift times {tdrift.min():.2f}-{tdrift.max():.2f} us, "
          f"record {edges[-1]:.2f} us  ({edges[-1]-tdrift.max():.2f} us of tail)")
    print(f"  arrived {int(arrived.sum())}/{len(arrived)};  "
          f"collected {int(coll.sum())}/{len(coll)};  "
          f"row sums: min {resp.sum(1).min():.4f} max {resp.sum(1).max():.4f}")
    print(f"  reshape to the impact grid: response.reshape({n}, {n}, -1)  "
          f"[x, y, tick]")


if __name__ == "__main__":
    main()
