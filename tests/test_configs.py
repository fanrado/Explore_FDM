"""Every file in configs/ must be a complete, self-consistent run input.

Asserting this is what stops the config files from rotting as ``Config`` gains
fields: a new field that nobody adds to the JSON would otherwise fall back to a
code default silently, which is exactly the implicitness the files exist to
remove.

Checks, per file:

  1. it parses, and every non-comment key is a real ``Config`` field;
  2. **every** ``Config`` field is present -- no value is left to a default;
  3. no value is null except ``pad_before``, which cannot be known before the
     paths are tracked (null = 20% of the longest drift time);
  4. the geometry is exact on the integer lattice: pad and gap are whole
     multiples of ``h_min`` and the drift is a whole number of pitches;
  5. the ``_derived`` comment matches what the values actually resolve to, so
     the prose cannot go stale.

Files with a ``drift_config`` key are run_pochoir.py inputs: they are checked
against ``PochoirConfig`` instead (checks 1 and 2, the only nulls being the
documented optional ones, and the pochoir JSON they name must translate).
"""
import dataclasses
import glob
import json
import os
import sys

from fdm.pochoir_workflow import PochoirConfig
from fdm.workflow import Config

FIELDS = {f.name for f in dataclasses.fields(Config)}
RUNTIME_NULL = {"pad_before"}      # genuinely not knowable from the config
P_FIELDS = {f.name for f in dataclasses.fields(PochoirConfig)}
# null means "derive from the pochoir config" or "no limit"
P_NULL = {"weighting_config", "cathode_idx", "export_drift_zmax",
          "export_weighting_zmax"}

FAIL = []


def check(cond, what):
    print(f"    {'ok  ' if cond else 'FAIL'}  {what}")
    if not cond:
        FAIL.append(what)


paths = sorted(glob.glob(os.path.join("configs", "*.json")))
print(f"=== {len(paths)} config file(s), {len(FIELDS)} Config fields each ===")
check(bool(paths), "configs/ contains at least one file")

for path in paths:
    print(f"\n{path}")
    with open(path) as f:
        raw = json.load(f)
    values = {k: v for k, v in raw.items() if not k.startswith("_")}

    if "drift_config" in values:
        unknown = sorted(set(values) - P_FIELDS)
        check(not unknown, f"[pochoir] no unknown keys"
              f"{'' if not unknown else f': {unknown}'}")
        missing = sorted(P_FIELDS - set(values))
        check(not missing, f"[pochoir] all {len(P_FIELDS)} PochoirConfig fields "
              f"present{'' if not missing else f'; missing {missing}'}")
        if unknown or missing:
            continue
        nulls = sorted(k for k, v in values.items() if v is None)
        check(set(nulls) <= P_NULL, f"[pochoir] nulls only where documented "
              f"({nulls or 'none'})")
        geom = PochoirConfig(**values).geometry()
        check(geom.anode_idx < geom.cathode_idx,
              f"[pochoir] translates: anode {geom.z_anode:.3f} mm, cathode "
              f"{geom.z_cathode:.3f} mm, {geom.npixels}x{geom.npixels} probe")
        continue

    unknown = sorted(set(values) - FIELDS)
    check(not unknown, f"no unknown keys{'' if not unknown else f': {unknown}'}")

    missing = sorted(FIELDS - set(values))
    check(not missing,
          f"all {len(FIELDS)} Config fields present"
          f"{'' if not missing else f'; missing {missing}'}")
    if unknown or missing:
        continue

    nulls = sorted(k for k, v in values.items() if v is None)
    check(set(nulls) <= RUNTIME_NULL,
          f"nothing left to a code default (null: {nulls or 'none'})")

    cfg = Config(**values)

    # Geometry must be exact on the lattice.  Config.nz raises if the drift is
    # not a whole number of pitches, so reaching this line is part of the test.
    pad_u = cfg.pad_mm / cfg.h_min
    gap_u = cfg.gap_mm / cfg.h_min
    check(abs(pad_u - round(pad_u)) < 1e-9,
          f"pad {cfg.pad_mm:.6f} mm is {pad_u:.4f} h_min (integer)")
    check(abs(gap_u - round(gap_u)) < 1e-9,
          f"gap {cfg.gap_mm:.6f} mm is {gap_u:.4f} h_min (integer)")
    check(cfg.drift_length > 0 and cfg.nz >= 1,
          f"drift {cfg.drift_length:.3f} mm = {cfg.nz} pitches")
    check(0.0 < cfg.pad_mm < cfg.pitch,
          f"pad fits inside the pitch ({cfg.pad_mm:.4f} < {cfg.pitch})")

    expect = (f"h_min {cfg.h_min*1000:.1f} um; pad {cfg.pad_mm:.6f} mm "
              f"= {cfg.pad_mm/cfg.h_min:.0f} h_min; gap {cfg.gap_mm:.6f} mm "
              f"= {cfg.gap_mm/cfg.h_min:.0f} h_min; "
              f"drift {cfg.drift_length:.3f} mm = {cfg.nz} pitches; "
              f"E {cfg.e0:.1f} V/mm")
    check(raw.get("_derived") == expect,
          "_derived comment matches the resolved values"
          + ("" if raw.get("_derived") == expect
             else f"\n           file: {raw.get('_derived')}\n           real: {expect}"))

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} check(s)")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("all config files are complete and self-consistent")
