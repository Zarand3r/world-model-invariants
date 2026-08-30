#!/usr/bin/env python3
"""Does each number guard actually catch its claim being falsified?

`verify_paper_numbers.py` asserts values appear in the paper. Some checks anchor a value to the claim
it belongs to (`near=`); the rest only assert the digits occur somewhere. This script tells them
apart by breaking the paper and seeing which checks notice.

Written because I got this wrong by hand twice on 2026-08-29, in the way
`cppgpt/tools/mutate.sh` warns about: **a mutation that does not apply everywhere the claim appears
"survives" and looks like a guard failure.** Both times I mutated one of several occurrences and
concluded the guards were useless. This harness mutates EVERY occurrence and asserts it applied
before drawing any conclusion.

Usage:  uv run python scripts/mutate_guards.py
Exit 0 if every anchored guard is caught; 1 if any anchored guard sleeps through its mutation.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEC = ROOT / "paper1.2" / "sections"
# CLAIMS.md is part of the corpus verify_paper_numbers.py reads, so a mutation that skips it is
# incomplete and the guard "survives" for the wrong reason. Found on 2026-08-29 when the E18
# label-free guard slept through a mutation applied only to sections/.
CLAIMS = ROOT / "paper1.2" / "CLAIMS.md"

# (value to break, replacement, substring identifying the guard that must notice)
CASES = [
    ("6.7",   "9.9",   "E18 rho_obs ratio"),
    ("12",    None,    "E19 wrong-sign control"),      # None -> handled specially, see below
    ("2.484", "9.999", "F6 origin-forced slope"),
    ("-42.2", "-99.9", "E18 label-free effect"),
    ("591",   "999",   "E19 physics improvement"),
    ("0.31",  "9.99",  "F1 variance of dC"),
    ("0.0014", "0.9999", "F5 largest arm effect"),
    ("0.575", "9.999", "F5 across-episode return SD"),
    ("0.125", "0.999", "E19 shadow coefficient"),
    ("1.10",  "9.99",  "E19 residual to label-free"),
    ("0.058", "0.999", "F6 origin-forced slope CI"),
    ("1.65",  "9.99",  "F5 Gate 0 paired margin"),
    ("2.24",  "9.99",  "F6 separation dt=0.035"),
    ("5.72",  "9.99",  "F6 separation dt=0.05"),
    ("0.977", "0.999", "E19 rho_E at c*"),
    ("-49.5", "-99.9", "E19 effect at c*, seed 3"),
    ("767",   "999",   "F4b degradation vs RSSM"),
    ("0.032", "9.999", "F3 rho(G,Gtrue) high"),
    ("0.740", "9.999", "F1 action-use range low"),
    ("+0.57", "+9.99", "E10b Spearman _s1_step30000"),
]


def guard_status(fragment: str) -> str | None:
    out = subprocess.run([sys.executable, "scripts/verify_paper_numbers.py"],
                         capture_output=True, text=True, cwd=ROOT).stdout
    for line in out.splitlines():
        if fragment in line:
            return line.strip().split()[0]
    return None


def main() -> int:
    backup = pathlib.Path(tempfile.mkdtemp()) / "sections"
    shutil.copytree(SEC, backup)
    claims_backup = backup.parent / "CLAIMS.md"
    shutil.copy(CLAIMS, claims_backup)
    failures = []
    try:
        for old, new, fragment in CASES:
            if new is None:
                # "12" is a bare digit pair that occurs everywhere; break only the anchored phrase
                old, new = r"$12\times$ worse", r"$99\times$ worse"
            n = 0
            for p in list(SEC.glob("*.tex")) + [CLAIMS]:
                s = p.read_text()
                if old in s:
                    n += s.count(old)
                    p.write_text(s.replace(old, new))
            if n == 0:
                print(f"  SKIP   {fragment:34s} (value {old!r} not present)")
                continue
            status = guard_status(fragment)
            caught = status == "FAIL"
            print(f"  {'CAUGHT' if caught else 'MISSED':6s} {fragment:34s} "
                  f"broke {n} occurrence(s) of {old!r}")
            if not caught:
                failures.append(fragment)
            for p in backup.glob("*.tex"):
                shutil.copy(p, SEC / p.name)
            shutil.copy(claims_backup, CLAIMS)
    finally:
        for p in backup.glob("*.tex"):
            shutil.copy(p, SEC / p.name)
        shutil.copy(claims_backup, CLAIMS)
        shutil.rmtree(backup.parent, ignore_errors=True)

    print(f"\n  {len(CASES) - len(failures)}/{len(CASES)} guards caught their mutation")
    if failures:
        print("  guards that slept through a falsified claim:")
        for f in failures:
            print(f"    - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
