"""M29 provenance: record how a run record was produced, inside the record itself.

Why this module exists. An audit on 2026-08-28 found that **98 of 101 run records carried no
recorded invocation**. The scripts take `--ckpts/--data/--out` on the command line, so a record's
filename never appears in any source file, and the execution log stores prose rather than commands.
The mapping from a record back to (script, arguments) therefore existed nowhere in the repo.

That is not hypothetical. E10b's numbers were produced by an uncommitted ad-hoc script; when the
registered design was reimplemented, seed 1 reproduced and seed 0 did not, and no update rule tested
could reconstruct the original values. A supporting claim in the paper turned into a negative result.

`stamp()` returns a dict recording argv, resolved input hashes, git HEAD and the working-tree state.
`attach()` puts it inside a dict-shaped record; for list-shaped records it writes a `.prov.json`
sidecar so the raw rows stay byte-identical and immutable.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import subprocess
import sys


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=pathlib.Path(__file__).resolve().parent.parent).stdout.strip()
    except Exception:
        return ""


def _sha256(p: pathlib.Path, cap: int = 512 * 1024 * 1024) -> str | None:
    """Hash a file. Large arrays (frames npz) are hashed in full; cap guards pathological inputs."""
    try:
        if not p.is_file() or p.stat().st_size > cap:
            return None
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def stamp(*, inputs: list[str] | None = None, **extra) -> dict:
    """Everything needed to re-run this invocation, recorded at the moment it runs.

    `inputs` are paths whose content defines the result -- data files and checkpoints. They are
    hashed, because a path is not identity: this project has already had checkpoints silently
    overwritten by an accidental retraining.
    """
    ins = {}
    for s in inputs or []:
        p = pathlib.Path(s)
        ins[s] = {"sha256": _sha256(p),
                  "mtime": (datetime.datetime.fromtimestamp(p.stat().st_mtime,
                                                            datetime.timezone.utc).isoformat(timespec="seconds")
                            if p.exists() else None),
                  "bytes": p.stat().st_size if p.exists() else None}
    return {"argv": sys.argv,
            "cwd": str(pathlib.Path.cwd()),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": sys.version.split()[0],
            "inputs": ins,
            **extra}


def attach(record, out_path, *, inputs=None, **extra):
    """Attach provenance to `record` if it is a dict; otherwise write a `.prov.json` sidecar.

    Appends rather than overwrites, because several scripts here are resumable and a record can be
    built across sessions -- one entry per invocation.
    """
    s = stamp(inputs=inputs, **extra)
    if isinstance(record, dict):
        record.setdefault("provenance", {}).setdefault("runs", []).append(s)
        return record
    side = pathlib.Path(str(out_path).removesuffix(".json") + ".prov.json")
    prev = json.loads(side.read_text()) if side.exists() else {"runs": []}
    prev["runs"].append(s)
    side.write_text(json.dumps(prev, indent=1) + "\n")
    return record


def inputs_from_args(ns) -> list[str]:
    """Pull path-like inputs out of an argparse Namespace.

    Scripts here name their inputs inconsistently -- `--ckpt` or `--ckpts`, `--data`, `--eval-data`,
    `--ood-data` -- so this selects by extension rather than by argument name, and therefore keeps
    working when a script grows a new input flag.

    LIMITATION, found 2026-08-30 by auditing every artefact in runs/: this can only see paths that
    reach the argparse Namespace. A script that hardcodes its checkpoints or datasets as module
    constants -- which every experiment script written that week did -- records `inputs: {}` and
    looks stamped while carrying no input record at all. M29 did not break; the inputs stopped being
    where it looks. Such scripts must pass `inputs=[...]` to `attach()` explicitly.
    """
    out: list[str] = []
    for v in vars(ns).values():
        vals = v if isinstance(v, (list, tuple)) else [v]
        for x in vals:
            if isinstance(x, str) and x.endswith((".pt", ".npz")) and x not in out:
                out.append(x)
    return out
