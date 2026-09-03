"""Download the checkpoints and datasets that reproduce paper 1.2 but that git does not carry.

`tools/hf_upload.py` publishes them and writes the manifest on the HuggingFace side; this is the
other half, and until now it did not exist — the artifacts were uploaded but nothing in the
repository told a reader how to get them back, or checked what came down.

**Every file is verified against a sha256 committed in `docs/ASSETS.json`.** A reproduction that
starts from a download is then as checkable as one that starts from training: a truncated transfer
or a re-uploaded artifact fails loudly instead of quietly producing numbers about a different model.
The manifest also records, per file, which claim it backs, so `--what-backs` answers "which of these
do I actually need for the result I care about" without downloading two gigabytes.

**Local sourcing.** On the machine this study was run on, the artifacts already exist in a sibling
checkout. `--from-local` copies from there instead of the network; `--from-local PATH` points it
somewhere else. The hashes are checked either way, so a local copy is not trusted more than a
download.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "ASSETS.json"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: pathlib.Path, expect: dict) -> tuple[bool, str]:
    """(ok, reason). Size first because it is free and catches a truncated transfer."""
    if not path.exists():
        return False, "missing"
    if path.stat().st_size != expect["bytes"]:
        return False, f"size {path.stat().st_size} != {expect['bytes']}"
    got = sha256(path)
    if got != expect["sha256"]:
        return False, f"sha256 {got[:12]}... != {expect['sha256'][:12]}..."
    return True, "ok"


def main() -> int:
    m = json.loads(MANIFEST.read_text())
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="*", metavar="SUBSTR",
                   help="fetch only files whose name or backing claim contains one of these")
    p.add_argument("--from-local", nargs="?", const=m.get("local_fallback"), default=None,
                   metavar="DIR", help="copy from a local checkout instead of downloading")
    p.add_argument("--verify-only", action="store_true",
                   help="check what is already in runs/ and download nothing")
    p.add_argument("--what-backs", action="store_true",
                   help="list each file and the claim it supports, then exit")
    p.add_argument("--out", default=str(ROOT / "runs"))
    a = p.parse_args()

    if a.what_backs:
        for remote, e in m["files"].items():
            print(f"{e['bytes'] / 1e6:8.1f} MB  {remote:<44}{e['backs']}")
        return 0

    wanted = m["files"]
    if a.only:
        wanted = {k: v for k, v in wanted.items()
                  if any(s.lower() in (k + v["backs"]).lower() for s in a.only)}
        if not wanted:
            print(f"nothing matches {a.only}; try --what-backs")
            return 1

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(a.from_local) if a.from_local else None
    if src is not None and not src.is_dir():
        print(f"--from-local {src} is not a directory")
        return 1
    if not a.verify_only and src is None:
        from huggingface_hub import hf_hub_download          # late: not needed to verify

    failures, got = [], 0
    for remote, expect in wanted.items():
        local = out / pathlib.Path(remote).name
        ok, why = verify(local, expect)
        if ok:
            print(f"  ok       {local.name}")
            continue
        if a.verify_only:
            print(f"  FAIL     {local.name}  ({why})")
            failures.append(local.name)
            continue
        print(f"  fetching {local.name}  ({why}, {expect['bytes'] / 1e6:.0f} MB)", flush=True)
        if src is not None:
            candidate = src / pathlib.Path(remote).name
            if not candidate.exists():
                print(f"  MISSING  {candidate}")
                failures.append(local.name)
                continue
            shutil.copyfile(candidate, local)
        else:
            # Copy out of the HF cache rather than symlinking into it: the cache is prunable, and a
            # dangling runs/*.pt is a worse failure than a second copy on disk.
            shutil.copyfile(hf_hub_download(repo_id=m["repo_id"], repo_type=m["repo_type"],
                                            filename=remote), local)
        ok, why = verify(local, expect)
        if not ok:
            print(f"  FAIL     {local.name} after transfer ({why})")
            failures.append(local.name)
        got += 1

    if failures:
        print(f"\n{len(failures)} file(s) failed verification: {', '.join(failures)}")
        print("Do not report numbers computed from these.")
        return 1
    print(f"\nall {len(wanted)} asset(s) verified against docs/ASSETS.json"
          + (f" ({got} transferred)" if got else " (nothing to transfer)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
