#!/usr/bin/env python3
"""Sample GPU memory and utilisation while the workflow runs, and save the trace.

    # run the workflow under the monitor (the usual way)
    python3 monitor_gpu.py -- python3 run_workflow.py --outdir out

    # or watch an already-running process, or just watch for a while
    python3 monitor_gpu.py --pid 12345
    python3 monitor_gpu.py --duration 60

Writes ``<out>.csv``, ``<out>.npz`` and ``<out>.png``.  The trace records total
device memory and utilisation for every GPU, plus the memory attributable to the
monitored process, which is the number that actually matters when the machine is
shared.

If a stage log (``<outdir>/stages.jsonl``) is present, the plot shades and
labels each workflow stage, so a memory spike can be attributed to the stage
that caused it.

Uses pynvml when installed and falls back to polling nvidia-smi, so it has no
hard dependency beyond the driver.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

FIELDS = ["index", "memory.used", "memory.total", "utilization.gpu",
          "utilization.memory", "temperature.gpu", "power.draw"]


class Sampler:
    """GPU sampling through pynvml if available, else nvidia-smi."""

    def __init__(self):
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                            for i in range(pynvml.nvmlDeviceGetCount())]
            self.n = len(self.handles)
        except Exception:
            out = self._smi()
            self.n = len(out)
        self.backend = "pynvml" if self.nvml else "nvidia-smi"

    @staticmethod
    def _smi():
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        rows = []
        for line in r.stdout.strip().splitlines():
            v = [x.strip() for x in line.split(",")]
            rows.append([float(x) if x not in ("[N/A]", "N/A") else float("nan")
                         for x in v])
        return rows

    @staticmethod
    def compute_apps():
        """(pid, name, MiB) for every process holding GPU memory."""
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True)
        except Exception:
            return []
        out = []
        for line in r.stdout.strip().splitlines():
            if not line.strip():
                continue
            p_, n_, m_ = [x.strip() for x in line.split(",")]
            out.append((int(p_), n_, float(m_)))
        return out

    @staticmethod
    def _proc_mem(pids):
        """MiB of GPU memory used by the given pids (summed over devices)."""
        if not pids:
            return 0.0
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True)
        except Exception:
            return float("nan")
        tot = 0.0
        for line in r.stdout.strip().splitlines():
            if not line.strip():
                continue
            p, m = [x.strip() for x in line.split(",")]
            if int(p) in pids:
                tot += float(m)
        return tot

    def sample(self, pids=()):
        """One sample: (mem_used, util, temp, power) per GPU + process memory."""
        if self.nvml:
            mem, util, temp, pw, tot = [], [], [], [], []
            for h in self.handles:
                mi = self.nvml.nvmlDeviceGetMemoryInfo(h)
                ut = self.nvml.nvmlDeviceGetUtilizationRates(h)
                mem.append(mi.used / 2**20); tot.append(mi.total / 2**20)
                util.append(ut.gpu)
                try:
                    temp.append(self.nvml.nvmlDeviceGetTemperature(h, 0))
                    pw.append(self.nvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                except Exception:
                    temp.append(float("nan")); pw.append(float("nan"))
        else:
            rows = self._smi()
            mem = [r[1] for r in rows]; tot = [r[2] for r in rows]
            util = [r[3] for r in rows]; temp = [r[5] for r in rows]
            pw = [r[6] for r in rows]
        return mem, tot, util, temp, pw, self._proc_mem(set(pids))


def child_pids(pid):
    """The process plus its descendants, so a subprocess's GPU use is counted."""
    out = {pid}
    try:
        r = subprocess.run(["ps", "-o", "pid=", "--ppid", str(pid)],
                           capture_output=True, text=True)
        for line in r.stdout.split():
            out |= child_pids(int(line))
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/gpu_trace", help="output basename")
    ap.add_argument("--interval", type=float, default=0.25, help="seconds")
    ap.add_argument("--pid", type=int, help="watch an existing process")
    ap.add_argument("--duration", type=float, help="watch for this many seconds")
    ap.add_argument("--stages", default=None,
                    help="stage log to overlay (default <outdir>/stages.jsonl)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- command to run under the monitor")
    a = ap.parse_args()

    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd and not a.pid and not a.duration:
        raise SystemExit("give a command after --, or --pid, or --duration")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    s = Sampler()
    proc = None
    if cmd:
        env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", os.getcwd())}
        proc = subprocess.Popen(cmd, env=env)
        watch = proc.pid
        print(f"monitor_gpu: {s.backend}, {s.n} GPU(s), interval {a.interval}s, "
              f"running: {' '.join(cmd)}")
    else:
        watch = a.pid
        print(f"monitor_gpu: {s.backend}, {s.n} GPU(s), interval {a.interval}s, "
              f"{'pid ' + str(watch) if watch else f'{a.duration}s'}")

    t0 = time.time()
    T, MEM, UTIL, TEMP, PW, PROC = [], [], [], [], [], []
    pids = child_pids(watch) if watch else set()

    # Anything else already on the GPU will dominate the device-total and
    # utilisation traces, so say so up front rather than letting it be read as
    # this workflow's usage.
    foreign = [(p_, n_, m_) for p_, n_, m_ in s.compute_apps() if p_ not in pids]
    if foreign:
        print("  WARNING: other processes are already using the GPU; "
              "'device total' and utilisation include them.")
        for p_, n_, m_ in foreign:
            print(f"    pid {p_}  {m_:.0f} MiB  {os.path.basename(n_)}")
        print("    Trust the 'this workflow's process' line, or isolate with "
              "CUDA_VISIBLE_DEVICES.")
    last_refresh = 0.0
    try:
        while True:
            now = time.time()
            if watch and now - last_refresh > 2.0:
                pids = child_pids(watch)       # the tree changes as it runs
                last_refresh = now
            mem, tot, util, temp, pw, pm = s.sample(pids)
            T.append(now - t0); MEM.append(mem); UTIL.append(util)
            TEMP.append(temp); PW.append(pw); PROC.append(pm)

            if proc is not None and proc.poll() is not None:
                break
            if a.duration and now - t0 >= a.duration:
                break
            if a.pid and not os.path.exists(f"/proc/{a.pid}"):
                break
            time.sleep(max(0.0, a.interval - (time.time() - now)))
    except KeyboardInterrupt:
        print("\nmonitor_gpu: interrupted, saving what we have")

    rc = proc.returncode if proc is not None else 0
    T = np.array(T); MEM = np.array(MEM); UTIL = np.array(UTIL)
    TEMP = np.array(TEMP); PW = np.array(PW); PROC = np.array(PROC)
    total = np.array(tot)

    np.savez_compressed(f"{a.out}.npz", t=T, mem_used_MiB=MEM,
                        mem_total_MiB=total, util_pct=UTIL, temp_C=TEMP,
                        power_W=PW, proc_mem_MiB=PROC,
                        interval=np.float64(a.interval),
                        t0_epoch=np.float64(t0), backend=s.backend,
                        command=" ".join(cmd) if cmd else "")
    with open(f"{a.out}.csv", "w") as f:
        cols = ["t_s"] + [f"gpu{i}_mem_MiB" for i in range(s.n)] \
               + [f"gpu{i}_util_pct" for i in range(s.n)] + ["proc_mem_MiB"]
        f.write(",".join(cols) + "\n")
        for k in range(len(T)):
            f.write(",".join([f"{T[k]:.3f}"]
                             + [f"{v:.1f}" for v in MEM[k]]
                             + [f"{v:.0f}" for v in UTIL[k]]
                             + [f"{PROC[k]:.1f}"]) + "\n")

    print(f"\nmonitor_gpu: {len(T)} samples over {T[-1]:.1f}s -> "
          f"{a.out}.csv / .npz")
    for i in range(s.n):
        print(f"  GPU{i}: peak {MEM[:, i].max():.0f} MiB of {total[i]:.0f} "
              f"({100*MEM[:, i].max()/total[i]:.1f}%), "
              f"mean util {UTIL[:, i].mean():.0f}%, peak util {UTIL[:, i].max():.0f}%")
    if np.isfinite(PROC).any():
        print(f"  monitored process: peak {np.nanmax(PROC):.0f} MiB   <-- this "
              f"workflow")
    if foreign:
        print(f"  NOTE: {len(foreign)} foreign process(es) shared the GPU; "
              f"device totals and utilisation above are not this workflow alone.")

    if not a.no_plot:
        stages = a.stages
        if stages is None:
            guess = os.path.join(os.path.dirname(a.out) or ".", "stages.jsonl")
            stages = guess if os.path.exists(guess) else None
        plot(a.out, T, MEM, total, UTIL, PROC, t0, stages, bool(foreign))
    sys.exit(rc)


# Categorical slots, fixed order, validated for CVD separation.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def plot(base, T, MEM, total, UTIL, PROC, t0, stage_log, foreign=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = []
    if stage_log and os.path.exists(stage_log):
        meta = None
        with open(stage_log) as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "meta":
                    meta = r
                elif r.get("kind") == "stage" and meta:
                    off = meta["t0"] - t0
                    bands.append((r["stage"], r["t_start"] + off,
                                  r["t_end"] + off))

    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                           constrained_layout=True)
    for A in ax:
        for k, (name, s, e) in enumerate(bands):
            A.axvspan(s, e, color=GRID, alpha=0.35 if k % 2 else 0.15, lw=0)
    if bands:
        ymax = ax[0].get_ylim()[1]
        for name, s, e in bands:
            ax[0].annotate(name, ((s + e) / 2, 1.0), xycoords=("data", "axes fraction"),
                           rotation=90, ha="center", va="top", fontsize=7,
                           color=INK2, xytext=(0, -3), textcoords="offset points")

    for i in range(MEM.shape[1]):
        ax[0].plot(T, MEM[:, i], color=SERIES[i], lw=2,
                   label=f"GPU{i} device total (of {total[i]:.0f} MiB)")
    if np.isfinite(PROC).any() and np.nanmax(PROC) > 0:
        ax[0].plot(T, PROC, color=SERIES[MEM.shape[1]], lw=2, ls="--",
                   label="this workflow's process")
    ax[0].set_ylabel("GPU memory (MiB)", color=INK2, fontsize=9)
    ax[0].legend(loc="upper left", fontsize=8, frameon=False)
    ax[0].set_title("GPU memory", color=INK, fontsize=11, loc="left")

    for i in range(UTIL.shape[1]):
        ax[1].plot(T, UTIL[:, i], color=SERIES[i], lw=2, label=f"GPU{i}")
    ax[1].set_ylabel("utilisation (%)", color=INK2, fontsize=9)
    ax[1].set_xlabel("wall time (s)", color=INK2, fontsize=9)
    ax[1].set_ylim(-2, 102)
    ax[1].legend(loc="upper left", fontsize=8, frameon=False)
    ax[1].set_title("GPU utilisation", color=INK, fontsize=11, loc="left")

    for A in ax:
        A.grid(True, color=GRID, lw=0.6)
        A.set_axisbelow(True)
        for s_ in ("top", "right"):
            A.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            A.spines[s_].set_color(GRID)
        A.tick_params(colors=INK2, labelsize=8)

    if foreign:
        ax[0].annotate("device total includes other processes on this GPU",
                       (0.995, 0.04), xycoords="axes fraction", ha="right",
                       fontsize=8, color=INK2)
    fig.savefig(f"{base}.png", dpi=130, facecolor="white", bbox_inches="tight")
    print(f"  wrote {base}.png")


if __name__ == "__main__":
    main()
