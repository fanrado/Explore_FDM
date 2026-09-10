#!/usr/bin/env python3
"""Single entry point for the whole pixel-anode field/signal workflow.

    python3 run_workflow.py                          # everything, into out/
    python3 run_workflow.py --lmax 6 --npad 7        # finer grid, bigger cell
    python3 run_workflow.py --only response          # just the response table
    python3 run_workflow.py --skip export_grids plots
    python3 run_workflow.py --list                   # show the stages
    python3 run_workflow.py --config cfg.json        # load settings from JSON

Stages run in order and hand results to each other in memory.  A stage whose
inputs are missing loads them from --outdir instead, so `--only response` works
against a previous run's files without re-solving.

Every run writes <outdir>/stages.jsonl with per-stage wall time and memory.
Use monitor_gpu.py to sample GPU usage while it runs, and profile_runtime.py to
turn the stage log into a report.
"""
from __future__ import annotations

import argparse
import dataclasses
import json

from fdm.workflow import STAGES, Config, run


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list stages and exit")
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
                   "float | None": float}.get(f.type, str)
            g.add_argument(flag, type=typ, default=None,
                           help=f"(default {f.default})")
    return p


def main():
    args = build_parser().parse_args()

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
    print(f"grid    {cfg.npad}x{cfg.npad} pads, pitch {cfg.pitch} mm, "
          f"h_min {cfg.h_min*1000:.1f} um, drift {cfg.drift_length:.1f} mm")
    print(f"anode   pad {cfg.pad_mm:.4f} mm ({cfg.pad_mm/cfg.h_min:.0f} h_min), "
          f"gap {cfg.gap_mm:.4f} mm ({cfg.gap_mm/cfg.h_min:.0f} h_min)")
    print(f"field   {cfg.efield} V/cm    device {cfg.device} {cfg.dtype}")
    print(f"stages  {' -> '.join(stages)}\n")
    run(cfg, stages=stages)


if __name__ == "__main__":
    main()
