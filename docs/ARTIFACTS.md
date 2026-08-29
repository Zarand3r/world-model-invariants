# Published artifacts

Where the runs, checkpoints and datasets live, and what they are for.

## Weights & Biases

Project **`richardbao419-substrate/world-model-invariants`**, populated by `tools/wandb_log.py`.

- **14 training runs**, one per `runs/*_hist.json` — the real optimisation curves for every model the
  paper uses: the three reference DreamerV3 seeds, the three dissipative controls, the 2-DoF central
  and non-central arms, and the three action-conditioned models.
- **One results run**, `paper1.2-headline-numbers`, carrying the 39 verified headline numbers with
  their pass/fail state.

Everything is marked `backfilled: true` in its config, so a shipped run is never mistaken for one
that was tracked live. Regenerate with:

```bash
uv run python tools/wandb_log.py --training --results     # --dry-run to see it first
```

## Hugging Face

Repo **`Zarand3r/world-model-invariants`** — 34 files, 1.1 GB, uploaded by `tools/hf_upload.py`.

`docs/ARTIFACT_MANIFEST.md` lists every file with its **sha256** and **the claim it backs**, so a
download can be checked against the record the experiments actually ran on.

| directory | contents |
|---|---|
| `checkpoints/` | the models behind each claim: reference seeds at step 6,500 and 60,000, dissipative controls, 2-DoF arms, both conv-GRU families, and the action-conditioned models |
| `data/` | all 10 datasets — every result is reproducible from these plus the checkpoints |
| `meta/` | the frozen pixel-readout calibration, which the decoded-energy metric depends on |

### Visibility

**The repo is private.** It was created public on 2026-08-27 and set private on 2026-08-29, because
the manuscript is unpublished and NeurIPS, ICML and ICLR all review double-blind — a public repo under
an author's account, named for the paper, is an anonymity risk. Making it public again is a one-line
change; the reverse is not, since a fetched checkpoint cannot be recalled.

## What is *not* published

`runs/` holds 6.2 GB. The intermediate checkpoint ladders (`*_step1000.pt` … `*_step30000.pt`) exist
to support the E8 training-saturation sweep and are not uploaded; the two checkpoints per seed that
back a claim are. The `runs/*.json` records stay in git, where they are small, diffable, and
regenerate `docs/RESULTS.md` byte-identically.
