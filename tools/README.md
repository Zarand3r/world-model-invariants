# tools

Dev sidecars. **Nothing in `latent_noether/` or `scripts/` imports anything here**, and nothing here
writes into `runs/`. That is deliberate, and it is the same rule `cppgpt/tools/` follows: experiments
write append-only records, tooling reads them. An experiment that opened a network connection could
be slowed by it, fail because of it, or perturb the raw rows `docs/ROADMAP.md` requires to stay
immutable.

| tool | question it answers | why it exists |
|---|---|---|
| `wandb_log.py` | what did training and the headline numbers actually do? | keeps the network out of the experiment path |
| `hf_upload.py` | can someone else reproduce this? | checkpoints and datasets are too large for git |

## Notes that matter

**`wandb_log.py` is backfill-only.** cppgpt's version wraps a live training run; ours does not,
because every experiment here has already finished. A live mode would be dead code pretending to be
a feature.

**`--results` reuses `scripts/verify_paper_numbers.py` rather than recomputing.** That script is the
project's single source of truth for what the paper claims. A second extraction path could silently
disagree with the manuscript, which is the failure this project has spent the most effort preventing.
It also means a `FAIL` in the verifier surfaces as a warning during upload rather than being shipped
as if fine.

**`hf_upload.py` checks the repo's REAL visibility.** `create_repo(exist_ok=True)` does *not* change
the visibility of a repo that already exists, so it returns a public repo while the caller believes it
asked for a private one. That happened on 2026-08-29: the tool printed "private" and uploaded into a
repo that had been public since 2026-08-27. It now reads back the actual setting and refuses unless
`--public` is passed knowingly. Verified in both directions.

**`hf_upload.py` uploads a curated selection, not `runs/`.** `runs/` is 6.2 GB, most of it
intermediate checkpoint ladders supporting the E8 saturation sweep. Every uploaded file is listed in
`docs/ARTIFACT_MANIFEST.md` with its sha256 and the claim it backs; anything absent is absent on
purpose.
