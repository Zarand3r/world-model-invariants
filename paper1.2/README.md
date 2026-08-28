# Paper build

```bash
uv run python paper/make_figures.py     # regenerate all figures from runs/*.json
cd paper && tectonic -X compile main.tex
```

`make_figures.py` reads only committed run logs, so every number and mark in every figure
regenerates from the experiment output. Run it before each build and confirm the diff is empty.

| figure | source |
|---|---|
| `fig1_three_claims.pdf` | `runs/dreamer_refusal.json`, `runs/dreamer_untrained_null.json`, `runs/dreamer_edit.json` |
| `fig2_ld_sweep.pdf` | `runs/dreamer_ld_sweep.json` |
| `fig3_low_variance.pdf` | `runs/dreamer_residual_decomp.json` |
| `fig4_leverage.pdf` | `runs/dreamer_leverage.json` |
| `fig5_random_null.pdf` | `runs/dreamer_edit.json` |

Figure grammar enforced here, beyond "no bars on a log axis" and "per-seed points always":

- **No dual axes.** `fig2` was a `twinx()` chart until 2026-08-13. Two y-scales make the crossing an
  artefact of scale alignment, and that crossing is the figure's whole claim. Both quantities are
  dimensionless, so they now share one axis.
- **The palette is validated, not chosen.** The previous "random law" orange and "dissipative" red
  failed colourblind separation (deutan ΔE 5.6) *and* the normal-vision floor (ΔE 7.1 against 15).
  The current set passes all-pairs CVD, normal vision, and 3:1 contrast on white.
- **Colour encodes the arm, shape encodes the seed.** `fig4`'s three seeds are one arm making one
  point, so they share a hue and differ by marker.

Label map from paper claims to the code and pre-registration that produced them:

| paper claim | script | pre-registration | decision entry |
|---|---|---|---|
| recovery at LD=12 | `run_dreamer_extraction.py --ld 12` | D36 | D37, D39 |
| untrained null | `run_dreamer_extraction.py --untrained` | `gauge.decodability` docstring | D43 |
| refusal | `run_dreamer_refusal.py` | `docs/DISSIPATIVE_PREREG.md` | D44 |
| the edit, 20-draw null | `run_dreamer_edit.py` | `docs/S4_PREREG.md` | D45, D67 |
| LD sweep / residual anti-correlation | `run_dreamer_ld_sweep.py` | D36 | D37, D67 |
| what the correction acts on | `run_dreamer_leverage.py` | D46 registered block | D46, D48, D52 |
| dissipative control on the above | `run_dreamer_leverage.py --ckpts runs/dreamer_damped_s*.pt --data runs/pendulum_pixels_damped.npz` | M26 | D67 |
| mechanism (unresolved) | `run_dreamer_nested_kappa.py` | D38 | D40, D41, D42 |

Two runs in this table are re-generations, not the originals, and the reason is recorded in D67:
`dreamer_edit.json` because its random-law arm was a single reused draw, and `dreamer_ld_sweep.json`
because the committed file no longer reproduced under the deterministic `encode` of D39.
