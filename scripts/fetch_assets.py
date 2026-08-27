"""Download the checkpoints and datasets that reproduction needs but git does not carry.

**Why these live outside git.** The six world-model checkpoints are 54 MB each, past the point
where GitHub warns and close to where it refuses; the three rendered datasets are another 35 MB and
are deterministically regenerable from `make_pendulum_pixels.py`. Keeping them out leaves the
tracked repository at 1.9 MB. They are hosted instead at the HuggingFace repo named in
`docs/ASSETS.json`.

**Every file is checked against a committed sha256**, so a reproduction that starts here is
verifiable rather than merely convenient: if a download is truncated or the remote is ever
re-uploaded with different weights, this fails loudly instead of producing numbers that are quietly
about a different model. `docs/ASSETS.json` also records each checkpoint's optimizer steps, wall
clock and seed, which is the metadata `docs/REPRODUCE.md` §3 pins.

The datasets are the one thing you can decline to download: `--skip-data` regenerates them locally
instead, which takes about a minute and is byte-identical (verified across two independently created
checkouts).
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
    """(ok, reason). Size is checked before the hash because it is free and catches truncation."""
    if not path.exists():
        return False, "missing"
    if path.stat().st_size != expect["bytes"]:
        return False, f"size {path.stat().st_size} != {expect['bytes']}"
    got = sha256(path)
    if got != expect["sha256"]:
        return False, f"sha256 {got[:12]}... != {expect['sha256'][:12]}..."
    return True, "ok"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--skip-data", action="store_true",
                   help="fetch checkpoints only; regenerate the .npz with make_pendulum_pixels.py")
    p.add_argument("--verify-only", action="store_true",
                   help="check what is already in runs/ against the manifest and download nothing")
    p.add_argument("--out", default=str(ROOT / "runs"))
    args = p.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = {k: v for k, v in manifest["files"].items()
              if not (args.skip_data and k.startswith("data/"))}

    if not args.verify_only:
        from huggingface_hub import hf_hub_download          # imported late: not needed to verify

    failures, fetched = [], 0
    for remote, expect in wanted.items():
        local = out / pathlib.Path(remote).name
        ok, why = verify(local, expect)
        if ok:
            print(f"  ok       {local.name}")
            continue
        if args.verify_only:
            print(f"  FAIL     {local.name}  ({why})")
            failures.append(local.name)
            continue
        print(f"  fetching {local.name}  ({why}, {expect['bytes'] / 1e6:.0f} MB)", flush=True)
        cached = hf_hub_download(repo_id=manifest["repo_id"], repo_type=manifest["repo_type"],
                                 filename=remote)
        # Copy out of the HF cache rather than symlinking into it: the cache is prunable, and a
        # dangling runs/*.pt is a worse failure than a second copy on disk.
        shutil.copyfile(cached, local)
        ok, why = verify(local, expect)
        if not ok:
            print(f"  FAIL     {local.name} after download ({why})")
            failures.append(local.name)
        fetched += 1

    if failures:
        print(f"\n{len(failures)} file(s) failed verification: {', '.join(failures)}")
        print("Do not report numbers computed from these.")
        return 1
    print(f"\nall {len(wanted)} asset(s) verified against docs/ASSETS.json"
          + (f" ({fetched} downloaded)" if fetched else " (nothing to download)"))
    if args.skip_data:
        print("datasets skipped — regenerate them with docs/REPRODUCE.md step 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
