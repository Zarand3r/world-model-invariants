# Invariant Probe Bench

An interactive rig over the frozen DreamerV3 world models in this repository: recover the conserved
scalar, steer it by hand, project the latent back onto its level set during imagination, and watch
what that does to fifty steps of predicted pendulum video.

```bash
uv run python scripts/fetch_assets.py     # once, if runs/ is empty
./viz/run.sh                              # http://127.0.0.1:8130
./viz/run.sh dev                          # Vite with hot reload, API proxied
```

## What it shows

| panel | what it is |
|---|---|
| **theatre** | held-out truth, free imagination and corrected imagination over the same steps, with per-frame pixel error and a shared cursor |
| **the invariant** | C against true energy (\|ρ\|_E), and C(t) drifting away from its own starting level set during the rollout |
| **the law bench** | eight sliders over the fitted basis, `C = Σ aᵢφᵢ`, with \|ρ\|_E, drift and the pairing residual live, and the dose response over the published α grid |
| **directions that matter** | variance V(u) against causal leverage D(u) per extracted direction, sized by how hard the projection pushes each one |

Anywhere the paper published a number, it is drawn behind the live one.

## Why it is fast

The fit is a 1819×1819 eigenproblem — 13–22 s depending on the extraction dimension. Everything
else is milliseconds. So a *bundle* (projection, conserved basis, mixing weights, and the basis
invariants evaluated on the analysis trajectories) is computed once per (checkpoint, LD, degree) and
cached to `runs/viz_cache/`; the sliders are then arithmetic on that. Measured on an RTX PRO 6000:

| interaction | work | time |
|---|---|---|
| move a law slider | `C = G @ a`, then rescore | **30–40 ms**, no GPU |
| move α | one 50-step rollout, two tracks | **140 ms** |
| dose response | 5 rollouts over 52 trajectories | 620 ms |
| change checkpoint or LD | encode, PCA, flow, eigenproblem, fit | 13–22 s, then cached |

## Reading it honestly

- The dose response is scored as the **slope over the whole fixed α grid**, never the best α.
- Rollout MSE differences below about **0.2%** are the size of the arithmetic — see the measurement
  in `viz/server/rollout.py`. The intervention's effect is 2.9–3.5%.
- C is fitted and scored on the same analysis trajectories, so the absolute effect is in-sample with
  respect to the fit. The comparison against other constraints stays matched.
- The law bench's "random draw" mixes the *already-conserved* basis, which is a harder null than the
  paper's arm B (a random polynomial over the whole degree-4 basis). The published null is shown
  underneath from `runs/dreamer_edit.json`.

## Layout

```
viz/server/    FastAPI: assets, model registry, bundle cache + job queue, law, rollout, leverage
viz/web/       React + Vite + TypeScript; hand-drawn SVG plots, no chart library
viz/run.sh     both processes
```

`latent_noether/extraction.py` holds the extraction the bench and the experiment scripts share.
`tests/test_viz_bench.py` pins the bench's numbers to the committed run logs.
