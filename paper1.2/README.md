# paper1.2 — build and figure provenance

This is the **current** manuscript. `paper/` is the superseded arXiv version and carries a documented
defect in its random-constraint null; see `paper/README.md`.

```bash
uv run python paper1.2/make_fig1.py       # Figure 1
uv run python paper1.2/make_fig2.py       # Figure 2
uv run python paper1.2/make_figures.py    # Figures 3-5 (appendix)
cd paper1.2 && tectonic -X compile main.tex
../scripts/make_arxiv_archive.sh paper1.2 # submission tarball
```

Every figure regenerates from committed run records, so the numbers in the figures cannot drift from
the experiments. Run all three generators before a build and confirm the diff is empty.

| figure | rendered as | generator | run records |
|---|---|---|---|
| `fig1_probe_vs_operator.pdf` | Figure 1 | `make_fig1.py` | `e18_supervised_baseline.json` |
| `fig2_shadow_sweep.pdf` | Figure 2 | `make_fig2.py` | `e19_shadow_sweep.json`, `e18_supervised_baseline.json` |
| `fig3_leverage.pdf` | Figure 3 | `make_figures.py` | `dreamer_leverage.json`, `dreamer_leverage_damped.json` |
| `fig4_ld_sweep.pdf` | Figure 4 | `make_figures.py` | `dreamer_ld_sweep.json` |
| `fig5_low_variance.pdf` | Figure 5 | `make_figures.py` | `dreamer_residual_decomp.json` |

**Filename matches rendered number.** It did not until 2026-08-28: the directory was copied from
`paper/`, so `fig4_leverage` rendered as Figure 3 and two files began `fig2_`. That is the collision
`make_arxiv_archive.sh`'s own header warns about, having once shipped stale figures.
`scripts/verify_paper_numbers.py` now asserts one file per number and that every shipped figure is
referenced by a section.

**`make_figures.py` writes to `paper1.2/figures/`.** It was copied from `paper/` with the output path
hardcoded, so until 2026-08-28 it wrote this manuscript's figures into the superseded one's directory
— silently, because both exist. It no longer emits `fig1_three_claims.pdf` or `fig5_random_null.pdf`,
which belong to paper 1.0 only.

## Figure grammar

Inherited from `paper/README.md` and still enforced:

- **No dual axes.** Two y-scales make a crossing an artefact of scale alignment.
- **The palette is validated, not chosen** — all-pairs CVD, normal vision, and 3:1 contrast on white.
- **Colour encodes the arm, shape encodes the seed.** Replicates are never averaged away.
- **Render every figure and look at it before shipping.** Three defects this session were invisible
  in code and obvious on sight: an axis label asserting the opposite of the result, a legend sitting
  on data, and an annotation clipped off the axis.

## Claims

`CLAIMS.md` holds the claim architecture — written before the prose, with every number traceable to a
run record. `scripts/verify_paper_numbers.py` recomputes each headline number from `runs/*.json` and
greps the sources for it; it reads `CLAIMS.md` as well as `sections/*.tex`.
