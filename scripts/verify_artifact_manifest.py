#!/usr/bin/env python3
"""Do the sha256 hashes in docs/ARTIFACT_MANIFEST.md match the local files?

The manifest's stated purpose is that "a download can be checked against the record the experiments
were run on". That is only true if the hashes are right, which was never checked until 2026-08-30.

Usage:  uv run python scripts/verify_artifact_manifest.py
Exit 0 if every listed file is present and matches; 1 otherwise.
"""
from __future__ import annotations

import hashlib, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROW = re.compile(r"\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*`([0-9a-f]{16})`\s*\|")


def main() -> int:
    rows = [m.groups() for m in
            (ROW.match(l) for l in (ROOT / "docs" / "ARTIFACT_MANIFEST.md").read_text().splitlines())
            if m]
    if not rows:
        print("  no manifest rows parsed -- has the table format changed?")
        return 1
    ok, bad, missing = 0, [], []
    for name, _mb, h16 in rows:
        p = ROOT / "runs" / name
        if not p.exists():
            missing.append(name); continue
        d = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                d.update(chunk)
        (ok := ok + 1) if d.hexdigest()[:16] == h16 else bad.append((name, h16, d.hexdigest()[:16]))
    print(f"  {len(rows)} entries: {ok} match, {len(bad)} mismatch, {len(missing)} missing")
    for n, e, g in bad:
        print(f"    MISMATCH {n}: manifest {e}, actual {g}")
    for n in missing:
        print(f"    MISSING  {n}")
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
