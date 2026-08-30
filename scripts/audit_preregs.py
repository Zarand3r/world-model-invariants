#!/usr/bin/env python3
"""Which preregistered predictions have no recorded verdict in any run artefact?

Written 2026-08-30 after three separate cases in one week where a script evaluated LESS than its
preregistration registered, each found by accident:

  * F7's P3 checked two of three registered acceptance criteria.
  * F7b's P2 checked a two-arm gap where the prereg registered a three-arm ordering.
  * F7 Gate 0's G2 recorded `pass` off an `inf` produced by a comparison that never ran.

Each was caught only because I happened to look. This sweeps all of them at once.

This is a REPORT FOR HUMAN REVIEW, not a pass/fail gate. It matches prediction labels textually, so
it cannot tell that a recorded verdict is *narrower* than the registered one -- only that some
verdict bearing the label exists. A clean run here does not mean the predictions were evaluated
faithfully; a flagged line means one plausibly was not evaluated at all.

Usage:  uv run python scripts/audit_preregs.py [--verbose]
"""
from __future__ import annotations

import argparse, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Deviations that are known, recorded in docs/EXECUTION_LOG.md, and deliberately not repaired.
# They stay VISIBLE in the output -- the point is only that a NEW flag is not lost among them.
# A permanently-red check is a check nobody reads, which is how the next real one gets missed.
# Adding an entry here requires a log entry saying why it was accepted rather than fixed.
KNOWN = {
    ("F8_PREREG.md", "P6"): (
        "registered mandatory before reading P5; T=10 passed the power gate so it was owed, and was "
        "not run. P5 came out NEGATIVE and P6 guards against reading a confounded POSITIVE, and the "
        "whole axis was withdrawn hours later, so running it now would interrogate a contrast that "
        "does not exist. Logged 2026-08-30."),
}
DOCS, RUNS = ROOT / "docs", ROOT / "runs"

# A registered prediction is introduced in bold: **P1**, **P1 (primary).**, **G0 (control).**, **A1**
LABEL = re.compile(r"\*\*([PGACDS]\d+[a-z]?)\b")
# C1/C2/C3 are ALSO the paper's claim identifiers ("Claim addressed: **C2 -- physical validity**"),
# and the first version of this script reported all ten such references as missing verdicts. They
# are not predictions. But C-labels are genuinely used as controls elsewhere (C1-C4 in
# run_dreamer_mechanism_controls.py), so this excludes by context rather than dropping the letter.
CLAIM_CONTEXT = re.compile(r"claims?\s+addressed", re.I)


def registered(path: pathlib.Path) -> list[str]:
    out, seen = [], set()
    lines = path.read_text().splitlines()
    for i, line in enumerate(lines):
        # the context can sit on the previous line when the sentence wraps:
        #   "... Claim addressed:\n   **C3 -- failure mechanism.**"
        if CLAIM_CONTEXT.search(line) or (i and CLAIM_CONTEXT.search(lines[i - 1])):
            continue
        for m in LABEL.finditer(line):
            lab = m.group(1)
            if lab not in seen:
                seen.add(lab); out.append(lab)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    blobs = {}
    for p in sorted(RUNS.glob("*.json")):
        try:
            blobs[p.name] = p.read_text()
        except Exception:
            pass

    rows, missing_total, checked_total, unlabelled = [], 0, 0, []
    for pre in sorted(DOCS.glob("*_PREREG.md")):
        labs = registered(pre)
        if not labs:
            unlabelled.append(pre.name)
            continue
        missing = []
        for lab in labs:
            # a recorded verdict mentions the label as a JSON key: "P1_pass", "G2_evaluable", ...
            # the label can sit anywhere in the key: "P5_pass", but also "separates_P5".
            # Matching only the prefix flagged F8's P5 as unevaluated when it is recorded as
            # `separates_P5` -- a false alarm that buried a real one (P6) in the same line.
            key = re.compile(rf'"[A-Za-z0-9_]*{lab}[_"]')
            if not any(key.search(b) for b in blobs.values()):
                missing.append(lab)
        known = [l for l in missing if (pre.name, l) in KNOWN]
        new = [l for l in missing if l not in known]
        checked_total += len(labs); missing_total += len(new)
        rows.append((pre.name, labs, new, known))

    total = len(rows) + len(unlabelled)
    print(f"  {len(rows)} preregistrations, {checked_total} registered predictions, "
          f"{missing_total} with no recorded verdict\n")
    print(f"  COVERAGE {len(rows)}/{total} preregistrations. The other {len(unlabelled)} do not use\n"
          f"  the **P1** bold-label convention and are INVISIBLE to this audit, not clean:\n"
          f"    {', '.join(unlabelled)}\n")
    flagged = [r for r in rows if r[2]]
    for name, labs, new, known in rows:
        if new or known or a.verbose:
            mark = "FLAG" if new else ("known" if known else "ok  ")
            extra = ""
            if new:
                extra = f" NO VERDICT: {','.join(new)}"
            elif known:
                extra = f" accepted deviation: {','.join(known)}"
            print(f"  {mark:5s} {name:24s} registered {','.join(labs):<28s}{extra}")
    n_known = sum(len(r[3]) for r in rows)
    if n_known:
        print(f"\n  {n_known} accepted deviation(s), each with a reason recorded in the log:")
        for name, _labs, _new, known in rows:
            for lab in known:
                print(f"    {name} {lab}: {KNOWN[(name, lab)]}")
    if not flagged:
        print("  No labelled prediction lacks a verdict -- but note what this CANNOT see: whether a\n"
              "  recorded verdict is narrower than what was registered. All three defects that\n"
              "  motivated this script were of that kind. A clean sweep here is weak reassurance.")
    if flagged:
        print(f"\n  {len(flagged)} preregistration(s) have predictions with no recorded verdict.")
        print("  Textual match only -- a label may be evaluated under a different name, and a")
        print("  recorded verdict may still be narrower than what was registered. Review by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
