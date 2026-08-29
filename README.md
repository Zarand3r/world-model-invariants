# A conserved quantity inside a pixel-trained world model

Code, run logs and paper source for the DreamerV3 study. A frozen world model trained only to
predict pendulum video contains a scalar that its own transition map holds constant; the search
returns nothing when the same model is trained with damping; and enforcing the recovered quantity
during imagination lowers the model's rollout error.

Everything in `runs/` is the output of the script named beside it in the table below. The figures in
the paper are generated from those files and nothing else, so every number is checkable without a
GPU.

The current manuscript is **`paper1.2/`**. `paper/` is the superseded arXiv version and carries a
documented defect in its random-constraint null — see `paper/README.md`.

## Artifacts

Training curves and the paper's verified numbers are on **Weights & Biases**; checkpoints and
datasets are on **Hugging Face**. See [`docs/ARTIFACTS.md`](docs/ARTIFACTS.md) for what is published,
what is deliberately not, and how to regenerate both. The tooling is sidecar-only — no experiment
imports it, and nothing writes into `runs/` — as described in [`tools/README.md`](tools/README.md).

## Regenerating the figures (no GPU, seconds)

```bash
pip install -e .
python paper1.2/make_fig1.py          # Figure 1
python paper1.2/make_fig2.py          # Figure 2
python paper1.2/make_figures.py       # Figures 3-5 (appendix)
cd paper1.2 && tectonic -X compile main.tex
```

`make_figures.py` prints every number it plots. Compare its output against the paper.

## Running the tests

```bash
pytest tests/ --ignore=tests/test_timing_convention.py     # 37 tests, ~4s, no GPU
```

`test_timing_convention.py` checks our adapter against the reference DreamerV3 implementation and
needs the vendored checkout described in `docs/REPRODUCE.md`.

## Reproducing from scratch (GPU, a few hours)

`docs/REPRODUCE.md` has the full procedure, including the pinned upstream commit of the DreamerV3
implementation. In outline:

```bash
python scripts/make_pendulum_pixels.py                      # render the dataset
python scripts/make_pendulum_pixels.py --zeta 0.03 \
       --out runs/pendulum_pixels_damped.npz                # the dissipative arm
python scripts/train_dreamer_pendulum.py --seed 3 --out runs/dreamer_ref_s3.pt
python scripts/run_dreamer_extraction.py --ckpts runs/dreamer_ref_s3.pt --ld 12
```

Model checkpoints (~54 MB each) and rendered datasets are not committed. Training is capped by wall
clock rather than step count, so a re-run lands near but not exactly on the committed numbers.

## What produced what

| paper claim | script | run log | pre-registration |
|---|---|---|---|
| recovery at LD=12 | `run_dreamer_extraction.py --ld 12` | `dreamer_extraction_prereg_ld12.json` | D36 |
| untrained null | `run_dreamer_extraction.py --untrained` | `dreamer_untrained_null.json` | `gauge.decodability` docstring |
| refusal on a damped model | `run_dreamer_refusal.py` | `dreamer_refusal.json` | `docs/DISSIPATIVE_PREREG.md` |
| the edit, 20-draw null | `run_dreamer_edit.py` | `dreamer_edit.json` | `docs/S4_PREREG.md` |
| extraction-dimension sweep | `run_dreamer_ld_sweep.py` | `dreamer_ld_sweep.json` | D36 |
| where the energy lives | `run_dreamer_residual_decomp.py` | `dreamer_residual_decomp.json` | D37 |
| what the correction acts on | `run_dreamer_leverage.py` | `dreamer_leverage.json` | D46 |
| the same, on damped models | `run_dreamer_leverage.py --ckpts runs/dreamer_damped_s*.pt` | `dreamer_leverage_damped.json` | M26 |
| flow-generation ablation | `run_pairing_ablation.py` | `pairing_ablation.json` | D47 |
| leverage stability by horizon | `run_leverage_stability.py` | `leverage_stability.json` | D48 |
| edit compactness | `run_edit_compactness.py` | `edit_compactness.json` | D49 |
| frequency weighting (unresolved) | `run_dreamer_nested_kappa.py` | `dreamer_nested_kappa.json` | D38 |

## Notes on reading the logs

- **Per-seed values, not just medians.** Two results in this paper turn on seed-level disagreement
  that a median hides: the intervention's specificity holds on two models of three, and the
  low-variance energy claim inverts on one seed. Both are visible in the raw JSON.
- **The random-law arm is a distribution.** `dreamer_edit.json` holds 20 draws per checkpoint under
  `B_conservative_random`. An earlier version drew one polynomial and reused it across models; the
  figures and the paper now report the null's median and each model's percentile inside it.
- **Two error metrics are not interchangeable.** The intervention here is scored on pixel error
  against held-out video. The precursor study described in the paper's appendix scored trajectory
  error in a two-dimensional state space, so its percentages are not comparable with these.

## Layout

```
latent_noether/   extraction machinery: PCA/effective-rank basis, polynomial invariants,
                  the joint f = B grad C fit, and the DreamerV3 adapter
scripts/          one script per experiment; each prints its own verdict
runs/             the JSON output of those scripts, as committed
paper/            LaTeX source, figure generator, generated figures
docs/             reproduction instructions and the pre-registrations
tests/            37 tests over the extraction machinery
```
