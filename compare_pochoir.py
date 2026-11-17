#!/usr/bin/env python3
"""Compare a run_pochoir.py output directory with a pochoir store.

    python3 compare_pochoir.py --ours out_pochoir \\
        --store ../pochoir/scripts/store_singlespacing_16cmdrift_16092026

Reads, where present, pochoir's potential/drift3d, potential/weight3d,
paths/drift3d and current/induced_current, and the matching files written by
run_pochoir.py, which are on pochoir's lattice and in its layout so no
regridding is involved.  Prints a text report and writes a figure.

Every comparison is index-for-index: ours[i, j, k] vs pochoir[i, j, k0 + k].
Differences therefore mix discretization error of *both* codes; at refine 0 the
boundary node sets are identical, so what remains is the solvers' own error
(pochoir's 0.1 mm uniform 7-point Jacobi vs our octree) and convergence.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

POCHOIR_C = "#2a78d6"   # categorical slot 1
OURS_C = "#eb6834"      # categorical slot 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _arr(path):
    z = np.load(path)
    return z[z.files[0]]


def compare_field(name, ours_path, poch_path, scale, rep):
    o = np.load(ours_path)
    phi, k0 = o["phi"], int(o["k0"])
    pc = _arr(poch_path)
    nz = min(phi.shape[2], pc.shape[2] - k0)
    p = pc[:, :, k0:k0 + nz]
    d = phi[:, :, :nz] - p
    z = (k0 + np.arange(nz)) * float(o["spacing"])
    mx = np.abs(d).max(axis=(0, 1))
    rms = np.sqrt((d ** 2).mean(axis=(0, 1)))
    rep.append(f"{name}: {phi.shape[0]}x{phi.shape[1]}x{nz} nodes compared "
               f"(z {z[0]:.2f}..{z[-1]:.2f} mm)")
    rep.append(f"  max |diff| {np.abs(d).max():.3e} ({np.abs(d).max()/scale:.2e} "
               f"of {scale:g}), rms {np.sqrt((d**2).mean()):.3e}")
    i = int(np.argmax(np.abs(d)))
    a, b, c = np.unravel_index(i, d.shape)
    rep.append(f"  worst at node ({a}, {b}, {k0 + c}), z = {z[c]:.2f} mm: "
               f"ours {phi[a, b, c]:.6g}, pochoir {p[a, b, c]:.6g}")
    for dz in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        k = int(round(dz / float(o["spacing"])))
        if k < nz:
            rep.append(f"  z = anode + {dz:4.1f} mm: max {mx[k]:.3e}  rms {rms[k]:.3e}")
    return dict(z=z, max=mx, rms=rms, ours=phi[:, :, :nz], poch=p)


def first_arrival(paths, t, z_anode, tol=0.01):
    hit = paths[:, :, 2] <= z_anode + tol
    k = np.where(hit.any(1), hit.argmax(1), -1)
    return np.where(k >= 0, t[np.maximum(k, 0)], np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", default="out_pochoir")
    ap.add_argument("--store", required=True, help="pochoir store directory")
    ap.add_argument("--out", default=None, help="figure path "
                    "(default <ours>/compare_pochoir.png)")
    args = ap.parse_args()
    S, O = args.store, args.ours
    rep, res = [], {}

    geo = np.load(f"{O}/drift_lattice.npz")
    z_anode = float(geo["z_anode"])
    vcat = abs(float(geo["cathode_potential"]))

    if os.path.exists(f"{S}/potential/drift3d.npz"):
        res["drift"] = compare_field("drift potential (V)",
                                     f"{O}/drift_lattice.npz",
                                     f"{S}/potential/drift3d.npz", vcat, rep)
        d = res["drift"]
        s = float(geo["spacing"])
        for lab, phi in (("ours", d["ours"]), ("pochoir", d["poch"])):
            kz = np.searchsorted(d["z"], [60.0, 140.0])
            ez = -(phi[:, :, kz[1]].mean() - phi[:, :, kz[0]].mean()) / (
                d["z"][kz[1]] - d["z"][kz[0]])
            rep.append(f"  bulk E_z (60..140 mm) {lab}: {ez:.6f} V/mm")
    if os.path.exists(f"{S}/potential/weight3d.npz"):
        res["weight"] = compare_field("weighting potential",
                                      f"{O}/weighting_lattice.npz",
                                      f"{S}/potential/weight3d.npz", 1.0, rep)

    t = None
    if os.path.exists(f"{S}/paths/drift3d.npz") and os.path.exists(f"{O}/paths.npz"):
        pp = _arr(f"{S}/paths/drift3d.npz")
        oz = np.load(f"{O}/paths.npz")
        po, t = oz["paths"], oz["t"]
        ta_o = first_arrival(po, t, z_anode)
        ta_p = first_arrival(pp, t, z_anode)
        end_d = np.hypot(*(po[:, -1, :2] - pp[:, -1, :2]).T)
        rep.append(f"paths: {len(po)} starts")
        rep.append(f"  arrival time ours {np.nanmean(ta_o):.3f} us, pochoir "
                   f"{np.nanmean(ta_p):.3f} us (mean); per-path diff "
                   f"{np.nanmean(ta_o - ta_p):+.3f} +- {np.nanstd(ta_o - ta_p):.3f} us")
        rep.append(f"  landing point xy difference: median {np.median(end_d)*1e3:.1f} um, "
                   f"max {end_d.max()*1e3:.1f} um")
        res["paths"] = (ta_o, ta_p, end_d)

    if os.path.exists(f"{S}/current/induced_current.npz") and \
            os.path.exists(f"{O}/current.npz"):
        ip = _arr(f"{S}/current/induced_current.npz")
        oc = np.load(f"{O}/current.npz")
        io = oc["current_pochoir"]
        t = oc["t"]
        na = int(round(np.sqrt(io.shape[0])))
        if ip.shape != io.shape:
            rep.append(f"currents: shape mismatch ours {io.shape} vs pochoir "
                       f"{ip.shape}; skipped")
        else:
            dt_ns = np.diff(t) * 1000.0
            qo = (io * dt_ns).sum(1).reshape(na, na)
            qp = (ip * dt_ns).sum(1).reshape(na, na)
            rep.append(f"currents: {io.shape[0]} waveforms ({na}x{na} impacts)")
            rep.append(f"  collected charge, centre impact: ours {qo[0, 0]:.4f}, "
                       f"pochoir {qp[0, 0]:.4f}")
            rep.append(f"  integrated charge, max |diff| over impacts "
                       f"{np.abs(qo - qp).max():.3e}")
            pk = np.abs(ip).max(1)
            rel = np.abs(io - ip).max(1) / np.maximum(pk.max(), 1e-30)
            rep.append(f"  max |i_ours - i_pochoir| / global peak: median "
                       f"{np.median(rel):.3e}, worst {rel.max():.3e}")
            res["cur"] = (io.reshape(na, na, -1), ip.reshape(na, na, -1),
                          qo, qp, oc["impact"])

    text = "\n".join(rep)
    print(text)
    with open(f"{O}/compare_pochoir.txt", "w") as f:
        f.write(f"ours  {os.path.abspath(O)}\npochoir {os.path.abspath(S)}\n\n"
                + text + "\n")

    _figure(res, t, z_anode, args.out or f"{O}/compare_pochoir.png")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _figure(res, t, z_anode, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axs.flat:
        ax.set_facecolor("#fcfcfb")

    ax = axs[0, 0]
    if "drift" in res:
        d = res["drift"]
        ax.semilogy(d["z"] - z_anode, d["max"], color=OURS_C, lw=2, label="max")
        ax.semilogy(d["z"] - z_anode, d["rms"], color=POCHOIR_C, lw=2, label="rms")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.legend(frameon=False, fontsize=8)
    _style(ax, "Drift potential: |ours - pochoir| per z plane",
           "height above anode (mm)", "V")

    ax = axs[0, 1]
    if "weight" in res:
        w = res["weight"]
        i = w["ours"].shape[0] // 2 - 1
        zz = w["z"] - z_anode
        ax.semilogy(zz, np.abs(w["poch"][i, i]), color=POCHOIR_C, lw=2, label="pochoir")
        ax.semilogy(zz, np.abs(w["ours"][i, i]), color=OURS_C, lw=2, ls="--",
                    label="ours")
        ax.legend(frameon=False, fontsize=8)
        _style(ax, f"Weighting potential above the collecting pad (node {i},{i})",
               "height above anode (mm)", "W")

    ax = axs[0, 2]
    if "weight" in res:
        w = res["weight"]
        ax.semilogy(w["z"] - z_anode, w["max"], color=OURS_C, lw=2, label="max")
        ax.semilogy(w["z"] - z_anode, w["rms"], color=POCHOIR_C, lw=2, label="rms")
        ax.legend(frameon=False, fontsize=8)
        _style(ax, "Weighting potential: |ours - pochoir| per z plane",
               "height above anode (mm)", "|dW|")

    if "cur" in res and t is not None:
        io, ip, qo, qp, imp = res["cur"]
        tc = 0.5 * (t[1:] + t[:-1])
        for ax, (a, b) in zip((axs[1, 0], axs[1, 1]), ((0, 0), (4, 0))):
            k = np.flatnonzero(np.abs(ip[a, b]) > 1e-3 * np.abs(ip[a, b]).max())
            sl = slice(max(k.min() - 20, 0), min(k.max() + 20, len(tc))) if len(k) \
                else slice(None)
            ax.plot(tc[sl], ip[a, b, sl] * 1e3, color=POCHOIR_C, lw=2, label="pochoir")
            ax.plot(tc[sl], io[a, b, sl] * 1e3, color=OURS_C, lw=2, ls="--",
                    label="ours")
            ax.legend(frameon=False, fontsize=8)
            _style(ax, f"Centre-pad current, impact ({imp[a]:.2f}, {imp[b]:.2f}) mm",
                   "time (us)", "dW/dt (per us)")
        ax = axs[1, 2]
        ax.plot(imp, qp[:, 0], color=POCHOIR_C, lw=2, marker="o", ms=4, label="pochoir")
        ax.plot(imp, qo[:, 0], color=OURS_C, lw=2, ls="--", marker="s", ms=4,
                label="ours")
        ax.legend(frameon=False, fontsize=8)
        _style(ax, "Charge on centre pad vs impact x (y = first row)",
               "impact x from pad centre (mm)", "collected fraction of e")
    else:
        for ax in axs[1]:
            ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
