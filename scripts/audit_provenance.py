#!/usr/bin/env python3
"""Which run artefacts carry no record of what they were computed from?

Written 2026-08-30 after finding `runs/f4b_recovery.json` recorded no input paths, which forced a
load-bearing same-data claim to rest on two scripts happening to share an argparse default.

The failure is subtle: `inputs_from_args` only sees paths that reach the argparse Namespace, so a
script hardcoding its checkpoints and datasets as module constants records `inputs: {}` and looks
stamped while carrying no input record at all.

Note on the unstamped artefacts: they predate M29 and are deliberately left alone. Adding a stamp now
would assert a provenance nobody can verify, which is worse than an honest gap.

Usage:  uv run python scripts/audit_provenance.py
"""
from __future__ import annotations

import json, pathlib, sys

RUNS = pathlib.Path(__file__).resolve().parent.parent / "runs"


def main() -> int:
    ok, empty, unstamped, unstampable = [], [], [], []
    for p in sorted(RUNS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict):
            # top-level lists (training histories, older outputs) cannot carry a provenance dict at
            # all. Excluding them is legitimate; excluding them SILENTLY is not -- an earlier version
            # dropped 49 files this way while printing the full glob count as the total.
            unstampable.append(p.name)
            continue
        prov = d.get("provenance")
        if not prov or not prov.get("runs"):
            unstamped.append(p.name)
        elif any(r.get("inputs") for r in prov["runs"]):
            ok.append(p.name)
        else:
            empty.append(p.name)

    total = len(ok) + len(empty) + len(unstamped)
    print(f"  {total} stampable artefacts: {len(ok)} record inputs, {len(empty)} stamped with EMPTY "
          f"inputs, {len(unstamped)} unstamped (pre-M29, left alone deliberately)")
    print(f"  {len(unstampable)} further files are top-level JSON lists (training histories and "
          f"older outputs)\n  and cannot carry a provenance dict in this format -- excluded, and "
          f"counted here so the\n  exclusion is visible rather than silent.\n")
    if empty:
        print("  stamped but recording no inputs -- check whether the script hardcodes its paths:")
        for n in empty:
            print(f"    {n}")
        print("\n  A script with no file inputs (a pure simulation) is legitimately empty here.")
        print("  One that reads checkpoints or datasets must pass inputs=[...] to attach().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
