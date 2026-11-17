#!/usr/bin/env python3
"""Solve a pochoir pixel-anode config with the octree FDM solver.

    python3 run_pochoir.py --config configs/pochoir_task13_fine.json
    python3 run_pochoir.py --drift-config X.json --weighting-config Y.json
    python3 run_pochoir.py --config configs/pochoir_task13_fine.json --refine 1
    python3 run_pochoir.py --config ... --only drift currents   # reuse fields
    python3 run_pochoir.py --config ... --no-thp     # on a fragmented host

``--no-thp``: on a long-running host with transparent huge pages set to
"always" and fragmented memory, every fresh large NumPy temporary can stall in
kernel memory compaction (seen here: a 0.05 mm run spent 1191 s in the kernel
against 59 s of its own work).  The flag disables huge pages for this process
(prctl PR_SET_THP_DISABLE, kept across exec) and re-executes once with glibc
told to keep freed memory for reuse (MALLOC_MMAP_MAX_=0, a large
MALLOC_TRIM_THRESHOLD_), which glibc only reads at start-up.

Outputs line up index-for-index with a pochoir store; compare with
compare_pochoir.py.  See fdm/pochoir.py for what is modelled.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import os
import sys

from fdm.pochoir_workflow import STAGES, PochoirConfig as Config, run


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list stages and exit")
    p.add_argument("--no-thp", action="store_true",
                   help="disable transparent huge pages and keep freed memory "
                        "for reuse (see the module docstring)")
    p.add_argument("--only", nargs="+", metavar="STAGE",
                   help="run only these stages, in the given order")
    p.add_argument("--skip", nargs="+", metavar="STAGE", default=[],
                   help="run everything except these")
    p.add_argument("--config", help="JSON file of Config fields (CLI wins)")
    p.add_argument("--dump-config", metavar="PATH",
                   help="write the resolved config to PATH and exit")

    # Every Config field becomes a flag, so the CLI never drifts from the config.
    g = p.add_argument_group("workflow settings")
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            g.add_argument(flag, action="store_true", default=None)
        else:
            typ = {"int": int, "float": float, "str": str,
                   "float | None": float, "int | None": int,
                   "str | None": str}.get(f.type, str)
            g.add_argument(flag, type=typ, default=None,
                           help=f"(default {f.default})")
    return p


_NOTHP_ENV = dict(MALLOC_MMAP_MAX_="0", MALLOC_TRIM_THRESHOLD_=str(1 << 36),
                  MALLOC_TOP_PAD_=str(1 << 30))


def _no_thp():
    """Disable THP for this process and re-exec once with the malloc settings."""
    if os.environ.get("_EXPLORE_FDM_NO_THP") == "1":
        return
    PR_SET_THP_DISABLE = 41
    if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0:
        print(f"warning: PR_SET_THP_DISABLE failed (errno {ctypes.get_errno()})")
    env = dict(os.environ, _EXPLORE_FDM_NO_THP="1", **_NOTHP_ENV)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def main():
    args = build_parser().parse_args()
    if args.no_thp:
        _no_thp()

    if args.list:
        print("stages, in order:\n")
        for name, fn in STAGES.items():
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:16s} {doc}")
        return

    values = {}
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
        # JSON has no comments, so keys beginning with "_" are treated as prose
        # and ignored.  That lets a config file document itself.
        values.update({k: v for k, v in loaded.items() if not k.startswith("_")})
    for f in dataclasses.fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            values[f.name] = v
    unknown = set(values) - {f.name for f in dataclasses.fields(Config)}
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(**values)

    if args.dump_config:
        with open(args.dump_config, "w") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        print(f"wrote {args.dump_config}")
        return

    stages = args.only if args.only else [s for s in STAGES if s not in args.skip]
    bad = [s for s in (args.only or []) + list(args.skip) if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s): {bad}\nknown: {list(STAGES)}")

    print(f"outdir  {cfg.outdir}")
    print(f"pochoir {cfg.drift_config}  +  {cfg.weighting_config}")
    print(f"mesh    h_min {cfg.spacing / 2**cfg.refine * 1000:.1f} um, "
          f"device {cfg.device} {cfg.dtype}")
    print(f"stages  {' -> '.join(stages)}\n")
    run(cfg, stages=stages)


if __name__ == "__main__":
    main()
