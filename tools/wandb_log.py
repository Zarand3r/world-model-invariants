#!/usr/bin/env python3
"""Ship this project's completed runs to Weights & Biases.

Dev tooling, deliberately a SIDECAR rather than a feature of the experiment scripts, following the
same rule as `cppgpt/tools/wandb_log.py`: experiments write append-only records under `runs/` and
this process reads them. No experiment imports wandb, so none can be slowed by it, fail because of
it, or have its raw rows perturbed by it. `docs/ROADMAP.md` requires raw rows in `runs/` to stay
immutable; a logger that wrote into them would break that.

Everything here is BACKFILL. Unlike cppgpt, there is no live mode, because every experiment in this
project has already finished -- a live wrapper would be dead code pretending to be a feature.

Two modes:

    --training     one W&B run per `runs/*_hist.json`  (real training curves)
    --results      one W&B run carrying the paper's verified headline numbers

`--results` deliberately reuses `scripts/verify_paper_numbers.py` as its source rather than
recomputing anything. That script is the project's single source of truth for what the paper claims,
and it is checked in CI-style on every iteration; a second extraction path could silently disagree
with the manuscript, which is the exact failure this project has spent the most effort preventing.

Auth failures abort BEFORE anything is shipped, so a broken credential is discovered immediately
rather than halfway through a batch.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return ""


def _provenance(rec: dict) -> dict:
    """Flatten an M29 provenance block into W&B config, if the record carries one."""
    runs = (rec.get("provenance") or {}).get("runs") or []
    if not runs:
        return {}
    p = runs[0]
    return {"prov_git": p.get("git", "")[:12], "prov_utc": p.get("utc"),
            "prov_dirty": p.get("git_dirty"), "prov_argv": " ".join(p.get("argv", [])[:6])}


def ship_training(wandb, entity: str | None, project: str, dry: bool) -> int:
    hists = sorted(RUNS.glob("*_hist.json"))
    if not hists:
        print("no runs/*_hist.json found", file=sys.stderr)
        return 1
    for h in hists:
        rows = json.loads(h.read_text())
        if not isinstance(rows, list) or not rows:
            print(f"  skip {h.name}: not a non-empty list")
            continue
        name = h.stem.removesuffix("_hist")
        cfg = {"model": name, "kind": "training", "backfilled": True,
               "git": _git("rev-parse", "HEAD")[:12], "steps": rows[-1].get("step")}
        if dry:
            print(f"  [dry] {name}: {len(rows)} points, final step {rows[-1].get('step')}")
            continue
        run = wandb.init(entity=entity, project=project, name=name, config=cfg,
                         job_type="training", reinit=True)
        for r in rows:
            step = r.get("step")
            if step is None:
                continue
            m = {f"train/{k}": v for k, v in r.items()
                 if k not in ("step",) and isinstance(v, (int, float))}
            run.log(m, step=int(step))
        last = rows[-1]
        run.summary["train/final_recon"] = last.get("recon")
        run.summary["train/final_val_recon"] = last.get("val_recon")
        run.summary["train/hours"] = last.get("hours")
        print(f"  {name}: {len(rows)} points -> {run.url}", flush=True)
        run.finish()
    return 0


def ship_results(wandb, entity: str | None, project: str, dry: bool) -> int:
    """One run carrying every headline number, taken from the verifier's own output."""
    proc = subprocess.run([sys.executable, "scripts/verify_paper_numbers.py"],
                          capture_output=True, text=True, cwd=ROOT)
    # strip first: the verifier indents its rows, so counting on the raw line silently reports 0
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip().startswith(("PASS", "FAIL"))]
    metrics, failed = {}, []
    for l in lines:
        parts = l.split()
        status, value = parts[0], parts[-1]
        label = " ".join(parts[1:-1]) if len(parts) > 2 else " ".join(parts[1:])
        if status == "FAIL":
            failed.append(label)
        try:
            metrics["paper/" + label.replace(" ", "_").replace("/", "-")] = float(value)
        except ValueError:
            pass                      # boolean guards carry "-" rather than a number
    n_pass = sum(1 for l in lines if l.startswith("PASS"))
    summary = {"paper/checks_total": len(lines), "paper/checks_passed": n_pass,
               "paper/checks_failed": len(failed)}
    if dry:
        print(f"  [dry] results: {n_pass}/{len(lines)} checks, {len(metrics)} numeric metrics")
        for k, v in sorted(metrics.items())[:8]:
            print(f"        {k} = {v}")
        return 0
    cfg = {"kind": "paper-results", "backfilled": True, "git": _git("rev-parse", "HEAD")[:12],
           "verifier_exit": proc.returncode}
    run = wandb.init(entity=entity, project=project, name="paper1.2-headline-numbers",
                     config=cfg, job_type="results", reinit=True)
    run.summary.update({**metrics, **summary})
    for f in failed:
        print(f"  WARNING: verifier reports FAIL for {f!r}", file=sys.stderr)
    print(f"  results: {n_pass}/{len(lines)} checks, {len(metrics)} metrics -> {run.url}", flush=True)
    run.finish()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="world-model-invariants")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    ap.add_argument("--training", action="store_true", help="ship runs/*_hist.json")
    ap.add_argument("--results", action="store_true", help="ship the paper's verified numbers")
    ap.add_argument("--dry-run", action="store_true", help="print what would ship, contact nothing")
    a = ap.parse_args()
    if not (a.training or a.results):
        print("nothing to do: pass --training and/or --results", file=sys.stderr)
        return 2

    wandb = None
    if not a.dry_run:
        import wandb as _w
        wandb = _w
        try:                        # fail fast, before shipping anything
            _w.Api().default_entity
        except Exception as e:
            print(f"wandb auth failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
    rc = 0
    if a.training:
        rc |= ship_training(wandb, a.entity, a.project, a.dry_run)
    if a.results:
        rc |= ship_results(wandb, a.entity, a.project, a.dry_run)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
