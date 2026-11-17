#!/usr/bin/env python3
"""What halving the spacing buys: Explore_FDM at 0.1 mm and 0.05 mm against the
pochoir hybrid and firep, all with zero-slope weighting walls.

    python3 compare_refine.py [OUTDIR]          # default out_compare_refine/

Reads (environment variables, defaults in brackets; nothing is written there):

* ``EXPLORE_COARSE`` / ``EXPLORE_FINE``  run_pochoir.py outputs for
  configs/pochoir_task13_fine_mirror.json at refine 0 and refine 1
  [out_pochoir_mirror, out_pochoir_mirror_r1_nothp];
* ``POCHOIR_STORE``  the pochoir hybrid store run with --mirror-edges
  [../pochoir/scripts/store_hybrid_11092026];
* ``FIREP_RUN``      firep's solves, paths and currents for that store
  [../firep/runs/pixel/runfield_mirror], read with firep's own package.

Figures: weight, potentials, paths, currents, current-average (as in
pochoir_Analysis/cross_validation/three_way_mirror.py, with both Explore_FDM
spacings), and ``improvement.png``: the change from 0.1 to 0.05 mm, the
codes' distance from the 0.05 mm solution, and the remaining discretization
error estimated by Richardson extrapolation assuming second order,
err(0.05) ~ (W_0.1 - W_0.05)/3 and err(0.1) ~ 4/3 (W_0.1 - W_0.05).  Near pad
edges and corners the field is singular and convergence is slower than
second order, so there the estimate is a lower bound.  Numbers go to
summary.json.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from fdm.drift import Interpolator  # noqa: E402
from fdm.io import load_field  # noqa: E402

FIREP = os.environ.get("FIREP", os.path.join(WORK, "firep"))
sys.path.insert(0, os.path.join(FIREP, "src"))
from firep import pixel as X  # noqa: E402

RUNS = dict(c=os.environ.get("EXPLORE_COARSE", os.path.join(HERE, "out_pochoir_mirror")),
            r=os.environ.get("EXPLORE_FINE", os.path.join(HERE, "out_pochoir_mirror_r1_nothp")))
STORE = os.environ.get("POCHOIR_STORE", os.path.join(WORK, "pochoir/scripts/store_hybrid_11092026"))
FIREP_RUN = os.environ.get("FIREP_RUN", os.path.join(FIREP, "runs/pixel/runfield_mirror"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
PITCH, Y_PCB, Y_PAD, TOP, INTERFACE = 4.4, 9.9, 10.0, 160.6, 29.7
TICK, NT, S = 0.05, 4000, 0.1
LAB = dict(p="pochoir hybrid", f="firep", c="Explore_FDM 0.1 mm", r="Explore_FDM 0.05 mm")
STY = dict(p=dict(color="C0", ls="-"), f=dict(color="C1", ls="--"),
           c=dict(color="C2", ls=":"), r=dict(color="C3", ls="-."))
MK = dict(p="s", f="o", c="^", r="v")
t64 = lambda a: torch.as_tensor(np.asarray(a, float), device=DEV, dtype=DT)  # noqa: E731


def wrap(u):
    return (u + PITCH / 2) % PITCH - PITCH / 2


def store(key):
    g, n = key.split("/")
    return np.load(os.path.join(STORE, g, n + ".npz"))[n]


def firep_field(name):
    st = torch.load(os.path.join(FIREP_RUN, name + ".pt"), weights_only=False)
    return X.SheetPotential.from_state(st["field"], device=DEV)


def evaluate(field, xyz, grad=False, chunk=4000):
    """firep's SheetPotential.evaluate in chunks (the GPU may be shared)."""
    xyz = np.asarray(xyz, float)
    vs, gs = [], []
    for i in range(0, len(xyz), chunk):
        r = field.evaluate(t64(xyz[i:i + chunk]), grad=grad)
        if grad:
            vs.append(r[0].cpu().numpy())
            gs.append(r[1].cpu().numpy())
        else:
            vs.append(r.cpu().numpy())
    return (np.concatenate(vs), np.concatenate(gs)) if grad else np.concatenate(vs)


def main(out=None):
    out = out or os.path.join(HERE, "out_compare_refine")
    os.makedirs(out, exist_ok=True)
    R = dict(inputs=dict(pochoir=STORE, firep=FIREP_RUN, explore_0p1=RUNS["c"], explore_0p05=RUNS["r"]))
    for k in "cr":
        cfg = json.load(open(os.path.join(RUNS[k], "geometry.json")))["config"]
        assert cfg.get("weighting_walls") == "mirror", f"{RUNS[k]} is not a mirror-wall run"
        R[f"refine_{k}"] = cfg["refine"]
    assert (R["refine_c"], R["refine_r"]) == (0, 1), "expected refine 0 and refine 1 runs"

    drift, W = firep_field("drift"), firep_field("weighting5")
    WL = {k: np.load(os.path.join(RUNS[k], "weighting_lattice.npz")) for k in "cr"}
    k0 = int(WL["c"]["k0"])
    zmax_lat = (k0 + WL["c"]["phi"].shape[2] - 1) * S
    oct_ = {}
    for k in "cr":
        tr, d = load_field(os.path.join(RUNS[k], "weighting_field.npz"))
        oct_[k] = (Interpolator(tr), d["phi"], d["E"])
        print(f"loaded {LAB[k]} weighting octree: {len(d['phi'])} nodes", flush=True)

    # ---- drift potential --------------------------------------------------------
    P = store("potential/drift3d")
    DL = {k: np.load(os.path.join(RUNS[k], "drift_lattice.npz"))["phi"] for k in "cr"}
    xs = wrap(np.arange(P.shape[0]) * S)
    Xg, Zg = np.meshgrid(xs, xs, indexing="ij")
    kzs = [101, 102, 103, 105, 107, 110, 115, 120, 150, 200, 297, 400, 800, 1200, 1600]
    pairs_d = (("r", "c"), ("p", "r"), ("p", "c"), ("f", "r"))
    drows = []
    for kz in kzs:
        v = dict(p=P[:, :, kz], c=DL["c"][:, :, kz - k0], r=DL["r"][:, :, kz - k0])
        v["f"] = evaluate(drift, np.column_stack((Xg.ravel(), np.full(Xg.size, kz * S), Zg.ravel()))
                          ).reshape(Xg.shape)
        row = dict(y=kz * S)
        for a, b in pairs_d:
            dd = v[a] - v[b]
            row[f"{a}-{b}"] = dict(mean=float(dd.mean()), rms=float(np.sqrt((dd ** 2).mean())))
        drows.append(row)
    R["drift_planes"] = drows
    R["bulk_field_V_per_mm"] = {k: float(-np.gradient(DL[k].mean(axis=(0, 1)), S)[800 - k0]) for k in "cr"}
    R["bulk_field_V_per_mm"]["p"] = float(-np.gradient(P.mean(axis=(0, 1)), S)[800])
    _, g = drift.evaluate(t64([[0.0, 80.0, 0.0]]), grad=True)
    R["bulk_field_V_per_mm"]["f"] = float(-g[0, 1])
    print("bulk field V/mm:", R["bulk_field_V_per_mm"], flush=True)
    del P, DL

    # ---- weighting potential -----------------------------------------------------
    Wp = store("potential/weight3d")
    SP = np.load(os.path.join(STORE, "startpoints.npy"))
    cen = float(SP[:, 0].min() - 0.22)
    c = int(round(cen / S))
    n = Wp.shape[0]
    gx = np.arange(n) * S
    GX, GY = np.meshgrid(gx, gx, indexing="ij")
    kk0 = torch.zeros(1, device=DEV, dtype=DT)

    def f_mean(y):
        y = np.atleast_1d(np.asarray(y, float))
        return sum(float(W.A[ip, 0, 0]) * W.arr.green(kk0, t64(y), yp, W.kind)[:, 0].cpu().numpy()
                   for ip, yp in enumerate(W.planes)) + W.offset

    def e_plane(k, y):
        if y <= zmax_lat + 1e-9:
            return WL[k]["phi"][:, :, int(round(y / S)) - k0]
        I, ph, _ = oct_[k]
        return I(ph, np.column_stack((GX.ravel(), GY.ravel(), np.full(GX.size, y)))).reshape(GX.shape)

    def e_mean(k, y):
        I, ph, _ = oct_[k]
        return float(I(ph, np.column_stack((GX.ravel(), GY.ravel(), np.full(GX.size, y)))).mean())

    ys = np.array([10.1, 10.2, 10.3, 10.5, 10.7, 11.0, 11.5, 12.0, 13.0, 15.0, 20.0, 25.0, 29.7, 35.0, 40.0,
                   60.0, 100.0, 150.0])
    pairs_w = (("c", "r"), ("p", "r"), ("f", "r"), ("p", "c"), ("f", "c"))
    wrows = []
    for y in ys:
        v = dict(p=Wp[:, :, int(round(y * 10))], c=e_plane("c", y), r=e_plane("r", y))
        v["f"] = evaluate(W, np.column_stack((GX.ravel() - cen, np.full(GX.size, y), GY.ravel() - cen))
                          ).reshape(GX.shape)
        row = dict(y=float(y), axis={k: float(v[k][c, c]) for k in "pfcr"},
                   mean={k: float(v[k].mean()) for k in "pcr"})
        row["mean"]["f"] = float(f_mean(y)[0])
        for a, b in pairs_w:
            dd = v[a] - v[b]
            row[f"{a}-{b}"] = dict(rms=float(np.sqrt((dd ** 2).mean())), max=float(np.abs(dd).max()))
        # Richardson (second order) estimate of the remaining grid error
        dd = v["c"] - v["r"]
        row["richardson"] = dict(err_0p05_axis=float(dd[c, c] / 3), err_0p1_axis=float(4 * dd[c, c] / 3),
                                 err_0p05_rms=float(np.sqrt((dd ** 2).mean()) / 3))
        wrows.append(row)
        a = row["axis"]
        print(f"W y={y:6.1f}: axis p {a['p']:.5f} f {a['f']:.5f} E0.1 {a['c']:.5f} E0.05 {a['r']:.5f}; "
              f"(E0.1-E0.05)/E0.05 {100 * (a['c'] - a['r']) / a['r']:+.3f}%", flush=True)
    R["weight_planes"] = wrows

    mz = Wp.mean(axis=(0, 1))
    hs = (15.0, 20.0, 25.0, 40.0, 80.0, 120.0, 150.0)
    R["weight_mean_slope"] = dict(
        p={f"{y:g}": float(-(mz[int(round(y * 10)) + 1] - mz[int(round(y * 10)) - 1]) / 0.2) for y in hs},
        f={f"{y:g}": float((f_mean(y - 0.1)[0] - f_mean(y + 0.1)[0]) / 0.2) for y in hs},
        **{k: {f"{y:g}": (e_mean(k, y - 0.1) - e_mean(k, y + 0.1)) / 0.2 for y in hs} for k in "cr"})
    print("plane-mean slope 1/mm:", json.dumps(R["weight_mean_slope"]), flush=True)

    hd = np.geomspace(0.1, TOP - Y_PAD - 0.1, 160)
    kv = np.unique(np.round(np.geomspace(1, int(round((TOP - Y_PAD) * 10)) - 2, 40)).astype(int)) + int(round(Y_PAD * 10))
    hp = kv * S - Y_PAD
    e = 5e-3
    curves = dict(
        f=(hd, np.abs(f_mean(Y_PAD + hd - e) - f_mean(Y_PAD + hd + e)) / (2 * e),
           np.linalg.norm(evaluate(W, np.column_stack((np.zeros(len(hd)), Y_PAD + hd, np.zeros(len(hd)))),
                                   grad=True)[1], axis=1)),
        p=(hp, np.abs(mz[kv + 1] - mz[kv - 1]) / 0.2, np.abs(Wp[c, c, kv + 1] - Wp[c, c, kv - 1]) / 0.2))
    for k in "cr":
        I, _, E = oct_[k]
        curves[k] = (hp, np.array([abs(e_mean(k, z - 0.1) - e_mean(k, z + 0.1)) / 0.2 for z in kv * S]),
                     np.linalg.norm(I(E, np.column_stack((np.full(len(hp), cen), np.full(len(hp), cen), kv * S))), axis=1))
    del Wp

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    yy = ys - Y_PAD
    for k in "pfcr":
        ax[0, 0].plot(yy, [r["mean"][k] for r in wrows], marker=MK[k], mfc="none", ms=5, label=LAB[k], **STY[k])
        ax[0, 1].semilogy(yy, [r["axis"][k] for r in wrows], marker=MK[k], mfc="none", ms=5, label=LAB[k], **STY[k])
        h, em, ea = curves[k]
        line = k == "f"
        ax[1, 0].plot(h, em, label=LAB[k], **(STY[k] if line else dict(color=STY[k]["color"], ls="none", marker=MK[k], mfc="none", ms=4)))
        ax[1, 1].plot(h, ea, label=LAB[k], **(STY[k] if line else dict(color=STY[k]["color"], ls="none", marker=MK[k], mfc="none", ms=4)))
    for a in ax.ravel():
        a.set_xscale("log")
        a.axvline(INTERFACE - Y_PAD, color="0.6", ls=":", lw=1)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    for a in ax[1]:
        a.set_xlabel("height above the pad top [mm]  (dotted: pochoir's 0.1/0.55 mm interface)")
        a.set_yscale("log")
    ax[0, 0].set_ylabel("plane-averaged weighting potential")
    ax[0, 1].set_ylabel("weighting potential above the sensing pad")
    ax[1, 0].set_ylabel("|plane-averaged weighting field| [1/mm]")
    ax[1, 1].set_ylabel("|weighting field| above the sensing pad [1/mm]")
    fig.suptitle("weighting, 5x5 pads, zero-slope walls")
    fig.savefig(os.path.join(out, "weight.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for (a, b), col in zip(pairs_d, ("C3", "C0", "C2", "C1")):
        ax[0].semilogy([r["y"] - Y_PAD for r in drows], [r[f"{a}-{b}"]["rms"] for r in drows], "o-", color=col,
                       label=f"{LAB[a]} - {LAB[b]}")
    for (a, b), col in zip(pairs_w, ("C3", "C0", "C1", "C9", "C5")):
        ax[1].semilogy(yy, [r[f"{a}-{b}"]["rms"] for r in wrows], "o-", color=col, label=f"{LAB[a]} - {LAB[b]}")
    ax[0].set_ylabel("drift potential, rms difference over a plane [V]")
    ax[1].set_ylabel("weighting potential, rms difference over a plane")
    for a in ax:
        a.set_xscale("log")
        a.axvline(INTERFACE - Y_PAD, color="0.6", ls=":", lw=1)
        a.set_xlabel("height above the pad top [mm]")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.savefig(os.path.join(out, "potentials.png"), dpi=120)
    plt.close(fig)

    # ---- paths -------------------------------------------------------------------
    S0 = store("starts/drift3d")
    loc = np.column_stack((wrap(S0[:, 0]), wrap(S0[:, 1])))
    PP = store("paths/drift3d")
    EP = {k: np.load(os.path.join(RUNS[k], "paths.npz")) for k in "cr"}
    for k in "cr":
        assert np.allclose(EP[k]["starts"], S0)
    pa = torch.load(os.path.join(FIREP_RUN, "paths.pt"), weights_only=False)
    fx = pa["xyz"].numpy()
    tt = np.arange(NT) * TICK
    hgt = dict(p=(PP[:, :, 2], tt), f=(fx[:, :, 1], pa["t"].numpy()),
               c=(EP["c"]["paths"][:, :, 2], EP["c"]["t"]), r=(EP["r"]["paths"][:, :, 2], EP["r"]["t"]))
    R["time_to_height_us"] = {f"{h:g}mm": {k: float(np.mean([np.interp(-h, -z, hgt[k][1]) for z in hgt[k][0]])) for k in "pfcr"}
                              for h in (100.0, 50.0, 20.0, 10.5)}
    ends = dict(p=PP[:, -1, :2], f=fx[:, -1][:, [0, 2]], c=EP["c"]["paths"][:, -1, :2], r=EP["r"]["paths"][:, -1, :2])
    dist = lambda a, b: np.hypot(wrap(a[:, 0]) - wrap(b[:, 0]), wrap(a[:, 1]) - wrap(b[:, 1]))  # noqa: E731
    R["end_points_mm"] = {f"{a}-{b}": dict(mean=float(dist(ends[a], ends[b]).mean()), max=float(dist(ends[a], ends[b]).max()))
                          for a, b in (("c", "r"), ("p", "r"), ("f", "r"), ("p", "c"))}
    print("time to height:", json.dumps(R["time_to_height_us"]), "\nend points:", R["end_points_mm"], flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    xpath = dict(p=(PP[:, :, 0], PP[:, :, 2]), f=(fx[:, :, 0], fx[:, :, 1]),
                 c=(EP["c"]["paths"][:, :, 0], EP["c"]["paths"][:, :, 2]), r=(EP["r"]["paths"][:, :, 0], EP["r"]["paths"][:, :, 2]))
    for k in "pfcr":
        for j in range(len(PP)):
            ax[0].plot(wrap(xpath[k][0][j]), xpath[k][1][j], color=STY[k]["color"], ls=STY[k]["ls"], lw=0.6)
        ax[0].plot([], [], label=LAB[k], **STY[k])
        ax[1].plot(wrap(ends[k][:, 0]), wrap(ends[k][:, 1]), MK[k], mfc="none", ms=5, color=STY[k]["color"], label=LAB[k])
    ax[1].plot(loc[:, 0], loc[:, 1], ".", color="0.7", ms=3, label="launch")
    ax[0].set_ylim(Y_PCB - 0.2, 16)
    ax[0].set_xlabel("x, folded into the cell [mm]")
    ax[0].set_ylabel("height [mm]")
    ax[0].set_title("paths near the pads", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].set_aspect("equal")
    ax[1].set_title("transverse end points", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.savefig(os.path.join(out, "paths.png"), dpi=120)
    plt.close(fig)

    # ---- currents ------------------------------------------------------------------
    tc = np.arange(NT - 1) * TICK + 0.5 * TICK
    rel = SP[:, :2] - cen
    I = dict(p=store("current/induced_current") * 1e3, f=np.load(os.path.join(FIREP_RUN, "currents.npz"))["I"])
    for k in "cr":
        EC = np.load(os.path.join(RUNS[k], "current.npz"))
        imp = EC["impact"]
        assert np.allclose(np.stack(np.meshgrid(imp, imp, indexing="ij"), -1).reshape(-1, 2), rel)
        I[k] = EC["current_pochoir"] * 1e3

    def pick(x, z):
        return int(np.argmin(np.hypot(rel[:, 0] - x, rel[:, 1] - z)))

    picks = [(0.22, 0.22, "over the pad, near its centre"), (1.54, 0.22, "over the pad, near an edge"),
             (1.98, 1.98, "near the gap crossing"), (2.42, 0.22, "over the gap, next pad's side"),
             (4.62, 4.62, "over the diagonal pad"), (9.02, 9.02, "two pads away")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    R["picks"] = []
    for a, (x, z, lab) in zip(axes.ravel(), picks):
        j = pick(x, z)
        for k in "pfcr":
            a.plot(tc, I[k][j], lw=1.2, label=LAB[k], **STY[k])
        tpk = tc[np.argmax(np.abs(I["p"][j]))]
        a.set_xlim(tpk - 6, tpk + 1.5)
        a.set_title(f"{lab}: launch ({rel[j, 0]:.2f}, {rel[j, 1]:.2f}) mm", fontsize=9)
        a.set_xlabel(r"time [$\mu$s]")
        a.set_ylabel(r"current [e/$\mu$s]")
        a.grid(alpha=0.3)
        pk = {k: float(I[k][j][np.argmax(np.abs(I[k][j]))]) for k in "pfcr"}
        R["picks"].append(dict(label=lab, launch=[float(rel[j, 0]), float(rel[j, 1])], peak=pk))
        print(f"{lab:32s} peak " + "  ".join(f"{LAB[k]} {pk[k]:+.4f}" for k in "pfcr"), flush=True)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(os.path.join(out, "currents.png"), dpi=120)
    plt.close(fig)

    own = (rel[:, 0] < PITCH / 2) & (rel[:, 1] < PITCH / 2)
    near = own & (np.abs(rel[:, 0]) < 1.3) & (np.abs(rel[:, 1]) < 1.3)
    allm = np.ones(len(rel), bool)
    pk_ref = np.abs(I["r"][near]).max()
    rms = lambda a, b, m: float(np.sqrt(((I[a] - I[b])[m] ** 2).mean()) / pk_ref * 100)  # noqa: E731
    R["currents"] = dict(
        collected={k: float(I[k][own].sum(1).mean() * TICK) for k in "pfcr"},
        avg_peak={k: float(I[k][own].mean(0).max()) for k in "pfcr"},
        interior_rms_pct_of_peak={f"{a}-{b}": rms(a, b, near) for a, b in (("c", "r"), ("p", "r"), ("f", "r"), ("p", "c"), ("f", "c"))},
        all_rms_pct_of_peak={f"{a}-{b}": rms(a, b, allm) for a, b in (("c", "r"), ("p", "r"), ("f", "r"), ("p", "c"), ("f", "c"))})
    print("currents:", json.dumps(R["currents"], indent=1), flush=True)

    fig, a = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for k in "pfcr":
        a.plot(tc, I[k][own].mean(0), label=LAB[k], **STY[k])
    tpk = tc[np.argmax(I["p"][own].mean(0))]
    a.set_xlim(tpk - 8, tpk + 1.5)
    a.set_xlabel(r"time [$\mu$s]")
    a.set_ylabel(r"current [e/$\mu$s]")
    a.set_title("average over launches above the collecting pad's pitch square", fontsize=9)
    a.grid(alpha=0.3)
    a.legend()
    fig.savefig(os.path.join(out, "current-average.png"), dpi=120)
    plt.close(fig)

    # ---- improvement: what 0.05 mm changes --------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ref = np.array([r["axis"]["r"] for r in wrows])
    for k, lab in (("c", "Explore_FDM 0.1 mm"), ("p", "pochoir hybrid (0.1 mm near grid)"), ("f", "firep")):
        ax[0, 0].plot(yy, 100 * (np.array([r["axis"][k] for r in wrows]) - ref) / ref, marker=MK[k], mfc="none",
                      color=STY[k]["color"], ls=STY[k]["ls"], label=f"{lab} - Explore_FDM 0.05 mm")
    rich = 100 * np.array([r["richardson"]["err_0p05_axis"] for r in wrows]) / ref
    ax[0, 0].fill_between(yy, -np.abs(rich), np.abs(rich), color="C3", alpha=0.25,
                          label="estimated remaining error of 0.05 mm (Richardson)")
    ax[0, 0].axhline(0, color="0.5", lw=0.8)
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_ylabel("W above the sensing pad, relative to Explore_FDM 0.05 mm [%]")
    ax[0, 0].set_xlabel("height above the pad top [mm]")
    for (a_, b_), col in zip(pairs_w, ("C3", "C0", "C1", "C9", "C5")):
        ax[0, 1].semilogy(yy, [r[f"{a_}-{b_}"]["rms"] for r in wrows], "o-", color=col, label=f"{LAB[a_]} - {LAB[b_]}")
    ax[0, 1].semilogy(yy, [r["richardson"]["err_0p05_rms"] for r in wrows], "k:", label="est. remaining error, 0.05 mm")
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_ylabel("W, rms difference over a plane")
    ax[0, 1].set_xlabel("height above the pad top [mm]")
    r0 = np.hypot(rel[:, 0], rel[:, 1])
    m = own
    for k, col in (("c", "C2"), ("p", "C0"), ("f", "C1")):
        pk_k = np.abs(I[k][m]).max(1)
        pk_r = np.abs(I["r"][m]).max(1)
        ax[1, 0].plot(r0[m], 100 * (pk_k - pk_r) / pk_r, MK[k], mfc="none", color=col,
                      label=f"{LAB[k]} vs Explore_FDM 0.05 mm")
    ax[1, 0].axhline(0, color="0.5", lw=0.8)
    ax[1, 0].set_xlabel("launch distance from the pad centre [mm]")
    ax[1, 0].set_ylabel("peak current, relative difference [%]")
    ax[1, 0].set_title("launches over the collecting pad's pitch square", fontsize=9)
    for k, col in (("c", "C2"), ("p", "C0"), ("f", "C1")):
        ax[1, 1].plot(tc, I[k][own].mean(0) - I["r"][own].mean(0), color=col, ls=STY[k]["ls"],
                      label=f"{LAB[k]} - Explore_FDM 0.05 mm")
    ax[1, 1].set_xlim(tpk - 8, tpk + 1.5)
    ax[1, 1].axhline(0, color="0.5", lw=0.8)
    ax[1, 1].set_xlabel(r"time [$\mu$s]")
    ax[1, 1].set_ylabel(r"average pulse difference [e/$\mu$s]")
    for a in ax.ravel():
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("what halving the spacing changes (reference: Explore_FDM 0.05 mm)")
    fig.savefig(os.path.join(out, "improvement.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(R, f, indent=1)
    print(f"wrote {out}/summary.json and figures", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
