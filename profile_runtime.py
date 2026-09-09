#!/usr/bin/env python3
"""Turn a workflow stage log into a runtime and memory report.

    # report on the last run
    python3 profile_runtime.py --stages out/stages.jsonl

    # run the workflow and report in one go
    python3 profile_runtime.py --run -- python3 run_workflow.py --outdir out

    # compare two runs (e.g. two resolutions)
    python3 profile_runtime.py --stages out/stages.jsonl out_fine/stages.jsonl

Writes ``<out>.csv``, ``<out>.json`` and ``<out>.png``.  Reports per-stage wall
time, share of the total, the torch peak GPU allocation and process RSS.  If a
GPU trace from monitor_gpu.py is present, the peak *device* memory during each
stage is attributed to that stage too -- torch's allocator only sees its own
tensors, so the two numbers answer different questions.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
BAR = "#2a78d6"
BAR2 = "#eb6834"


def read_log(path):
    meta, stages = None, []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            (stages if r.get("kind") == "stage" else [])
            if r.get("kind") == "meta":
                meta = r
            elif r.get("kind") == "stage":
                stages.append(r)
    if meta is None:
        raise SystemExit(f"{path}: no meta record; is this a stage log?")
    return meta, stages


def attach_device_peak(stages, meta, trace_path):
    """Attribute peak device memory to each stage from a monitor_gpu trace."""
    if not trace_path or not os.path.exists(trace_path):
        return False
    z = np.load(trace_path, allow_pickle=False)
    t = z["t"] + float(z["t0_epoch"]) - meta["t0"]      # -> stage-log clock
    # Busiest single GPU, not the sum: summing mixes in cards this run
    # never touched.  Note this is still whole-device (all users);
    # proc_peak_MiB below is this workflow alone.
    mem = z["mem_used_MiB"].max(axis=1)
    proc = z["proc_mem_MiB"]
    for r in stages:
        m = (t >= r["t_start"]) & (t <= r["t_end"])
        r["device_peak_MiB"] = float(mem[m].max()) if m.any() else float("nan")
        r["proc_peak_MiB"] = (float(np.nanmax(proc[m]))
                              if m.any() and np.isfinite(proc[m]).any()
                              else float("nan"))
    return True


def report(meta, stages, have_dev):
    total = sum(r["wall"] for r in stages)
    cols = [("stage", 18, "s"), ("wall_s", 9, "f2"), ("share", 8, "pct"),
            ("torch_peak_MiB", 15, "f0"), ("rss_MiB", 10, "f0")]
    if have_dev:
        cols += [("device_peak_MiB", 16, "f0"), ("proc_peak_MiB", 14, "f0")]

    head = "".join(f"{c[0]:>{c[1]}}" for c in cols)
    print(head)
    print("-" * len(head))
    for r in sorted(stages, key=lambda r: -r["wall"]):
        cells = []
        for name, w, kind in cols:
            v = r.get(name if name != "wall_s" else "wall",
                      r.get("torch_peak", r.get("gpu_peak_MiB")))
            if name == "stage":
                cells.append(f"{r['stage']:>{w}}")
            elif name == "wall_s":
                cells.append(f"{r['wall']:>{w}.2f}")
            elif name == "share":
                cells.append(f"{100*r['wall']/total:>{w-1}.1f}%")
            elif name == "torch_peak_MiB":
                cells.append(f"{r['gpu_peak_MiB']:>{w}.0f}")
            else:
                x = r.get(name, float("nan"))
                cells.append(f"{x:>{w}.0f}" if np.isfinite(x) else f"{'-':>{w}}")
        print("".join(cells))
    print("-" * len(head))
    print(f"{'TOTAL':>18}{total:>9.2f}{'100.0%':>8}"
          f"{max(r['gpu_peak_MiB'] for r in stages):>15.0f}"
          f"{max(r['rss_MiB'] for r in stages):>10.0f}")
    cfg = meta.get("config", {})
    if cfg:
        print(f"\n  {cfg.get('npad')}x{cfg.get('npad')} pads, "
              f"h_min {1000*cfg.get('pitch', 0)/(1 << cfg.get('lmax', 0)):.1f} um, "
              f"{cfg.get('dtype')}, {meta.get('device', '?')}")
    for r in stages:
        if "nodes" in r:
            print(f"  {r['nodes']} nodes, {r.get('cells')} cells, "
                  f"{100*r.get('irregular', 0)/max(r['nodes'], 1):.1f}% irregular")
    return total


def plot(base, runs, have_dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["stage"] for r in runs[0][1]]
    fig, ax = plt.subplots(1, 2 if have_dev else 1,
                           figsize=(13 if have_dev else 8, 0.45 * len(names) + 2.2),
                           constrained_layout=True, squeeze=False)
    ax = ax[0]
    y = np.arange(len(names))[::-1]

    # Wall time: one series per run, so a single hue unless we are comparing.
    nrun = len(runs)
    h = 0.8 / nrun
    for j, (label, stages) in enumerate(runs):
        w = [next((r["wall"] for r in stages if r["stage"] == n), 0.0)
             for n in names]
        color = BAR if j == 0 else BAR2
        off = (j - (nrun - 1) / 2) * h
        ax[0].barh(y + off, w, height=h * 0.9, color=color,
                   label=label if nrun > 1 else None)
        if nrun == 1:
            for yy, ww in zip(y, w):
                ax[0].annotate(f"{ww:.1f}s", (ww, yy), xytext=(4, 0),
                               textcoords="offset points", va="center",
                               fontsize=8, color=INK2)
    ax[0].set_yticks(y, names, fontsize=9)
    ax[0].set_xlabel("wall time (s)", color=INK2, fontsize=9)
    ax[0].set_title("Runtime by stage", color=INK, fontsize=11, loc="left")
    if nrun > 1:
        ax[0].legend(fontsize=8, frameon=False)

    if have_dev:
        label, stages = runs[0]
        tp = [next((r["gpu_peak_MiB"] for r in stages if r["stage"] == n), 0.0)
              for n in names]
        dp = [next((r.get("device_peak_MiB", np.nan)
                    for r in stages if r["stage"] == n), np.nan) for n in names]
        ax[1].barh(y + 0.2, tp, height=0.36, color=BAR, label="torch allocator")
        ax[1].barh(y - 0.2, dp, height=0.36, color=BAR2, label="whole device")
        ax[1].set_yticks(y, names, fontsize=9)
        ax[1].set_xlabel("peak GPU memory (MiB)", color=INK2, fontsize=9)
        ax[1].set_title("Peak GPU memory by stage", color=INK, fontsize=11,
                        loc="left")
        ax[1].legend(fontsize=8, frameon=False)

    for A in ax:
        A.grid(True, axis="x", color=GRID, lw=0.6)
        A.set_axisbelow(True)
        for s_ in ("top", "right", "left"):
            A.spines[s_].set_visible(False)
        A.spines["bottom"].set_color(GRID)
        A.tick_params(colors=INK2, labelsize=8, length=0)
    fig.savefig(f"{base}.png", dpi=130, facecolor="white", bbox_inches="tight")
    print(f"\nwrote {base}.png")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["out/stages.jsonl"],
                    help="one or more stage logs (two or more = comparison)")
    ap.add_argument("--gpu-trace", default=None,
                    help="monitor_gpu .npz (default: next to the stage log)")
    ap.add_argument("--out", default=None, help="report basename")
    ap.add_argument("--run", action="store_true",
                    help="run the command after -- first, then report")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    if a.run:
        cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
        if not cmd:
            raise SystemExit("--run needs a command after --")
        env = {**os.environ,
               "PYTHONPATH": os.environ.get("PYTHONPATH", os.getcwd())}
        rc = subprocess.run(cmd, env=env).returncode
        if rc:
            raise SystemExit(rc)
        print()

    runs, have_dev = [], False
    for path in a.stages:
        meta, stages = read_log(path)
        trace = a.gpu_trace or os.path.join(os.path.dirname(path) or ".",
                                            "gpu_trace.npz")
        have_dev |= attach_device_peak(stages, meta, trace)
        label = os.path.basename(os.path.dirname(path) or path)
        print(f"\n=== {path} ===")
        report(meta, stages, have_dev)
        runs.append((label, stages))

    base = a.out or os.path.join(os.path.dirname(a.stages[0]) or ".", "runtime")
    rows = []
    for label, stages in runs:
        for r in stages:
            rows.append(dict(run=label, **r))
    with open(f"{base}.json", "w") as f:
        json.dump(rows, f, indent=2)
    keys = ["run", "stage", "wall", "gpu_peak_MiB", "gpu_reserved_MiB",
            "rss_MiB", "device_peak_MiB", "proc_peak_MiB"]
    with open(f"{base}.csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"\nwrote {base}.csv and {base}.json")

    if not a.no_plot:
        plot(base, runs, have_dev)


if __name__ == "__main__":
    main()
