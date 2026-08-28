"""Figure 1 for paper 1.2: the right answer, fitted perfectly, does not work.

Grammar, stated here so it survives edits:
  colour  = arm (label-free C / supervised energy probe / magnitude-matched random)
  marker  = model seed (3, 4, 5) -- replicates are never averaged away
  panels  = the three quantities, never a dual axis

Every value is read from `runs/e18_supervised_baseline.json`. Nothing is typed in.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Palette checked for deuteranopia/protanopia separation; do not swap without re-checking.
COL = {"label-free": "#1b6ca8", "supervised": "#d1495b", "random": "#8d99ae"}
MRK = {"3": "o", "4": "s", "5": "^"}

d = json.loads((ROOT / "runs" / "e18_supervised_baseline.json").read_text())["models"]
rows = []
for M in d:
    s = M["ckpt"].split("ref_s")[1].split("_")[0]
    rows.append(("label-free", s, M["unsupervised"]["rho_E"], M["unsupervised"]["rho_obs"],
                 M["unsupervised"]["effect_pct"]))
    rows.append(("supervised", s, M["supervised"]["rho_E"], M["supervised"]["rho_obs"],
                 M["supervised"]["effect_pct"]))
    rn = M["random"]
    rows.append(("random", s, float(np.median([x["rho_E"] for x in rn])),
                 float(np.median([x["rho_obs"] for x in rn])),
                 float(np.median([x["effect_pct"] for x in rn]))))

fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.5))
order = ["label-free", "supervised", "random"]
xpos = {a: i for i, a in enumerate(order)}

panels = [
    (0, 2, "probe quality\n$|\\rho|$ with true energy", False, None),
    (1, 3, "conserved by the transition\n$\\rho_{\\mathrm{obs}}$  (lower is better)", True, None),
    (2, 4, "effect on imagined physics\n% change in energy drift", False, 0.0),
]
for pi, col, title, logy, hline in panels:
    a = ax[pi]
    for arm, s, rE, ro, eff in rows:
        v = (rE, ro, eff)[col - 2]
        a.scatter(xpos[arm] + (int(s) - 4) * 0.13, v, c=COL[arm], marker=MRK[s],
                  s=64, edgecolor="white", linewidth=0.7, zorder=3)
    if hline is not None:
        a.axhline(hline, color="0.35", lw=0.9, ls="--", zorder=1)
    if logy:
        a.set_yscale("log")
    a.set_xticks(range(3))
    a.set_xticklabels(["label-free\n$C$", "supervised\nenergy probe", "random\n(matched)"], fontsize=8.5)
    a.set_title(title, fontsize=9.5, pad=8)
    a.grid(axis="y", alpha=0.25, lw=0.6)
    a.set_xlim(-0.55, 2.55)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

# Negative = drift reduced = the correction HELPS. The first draft labelled this axis
# "harms <- -> helps", which put "harms" at the bottom where the label-free arm sits at -42% --
# the label asserted the opposite of the result. Caught by rendering the figure and looking at it.
ax[2].margins(y=0.16)   # clearance so the direction annotations never touch a point
ax[2].annotate("drift reduced", xy=(0.02, 0.06), xycoords="axes fraction", fontsize=7.5, color="0.35")
ax[2].annotate("drift increased", xy=(0.02, 0.93), xycoords="axes fraction", fontsize=7.5, color="0.35")
handles = [plt.Line2D([], [], color="0.25", marker=MRK[s], ls="", ms=7, label=f"seed {s}")
           for s in ("3", "4", "5")]
ax[0].legend(handles=handles, fontsize=8, frameon=False, loc="lower left")

fig.tight_layout(rect=(0, 0, 1, 0.99))
out = ROOT / "paper1.2" / "figures" / "fig1_probe_vs_operator.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
for arm in order:
    sel = [r for r in rows if r[0] == arm]
    print(f"  {arm:12s} rho_E {np.median([r[2] for r in sel]):.4f}  "
          f"rho_obs {np.median([r[3] for r in sel]):.3e}  effect {np.median([r[4] for r in sel]):+.1f}%")
