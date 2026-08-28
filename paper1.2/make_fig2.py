"""Figure 2: the model's transition conserves the integrator's shadow Hamiltonian.

Grammar, kept identical to make_fig1.py so the two figures read together:
  colour  = arm  -- blue is the label-free scalar, red is the supervised probe on textbook energy
  marker  = model seed (3, 4, 5); replicates are never averaged away
  panels  = ground-truth physics (no model) | the learned transition

The horizontal reference lines are fig1's two arms, so the reader can see the sweep pass from the
red level (textbook energy) down to the blue level (label-free) as c approaches the predicted
shadow coefficient. Nothing here is fitted: c* = (dt/2) mg(l/2) = 0.125 comes from the integrator.
"""
import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = pathlib.Path(__file__).resolve().parent.parent / "runs"
d = json.loads((R / "e19_shadow_sweep.json").read_text())
e18 = json.loads((R / "e18_supervised_baseline.json").read_text())
CS = 0.125
COL = {"label-free": "#1b6ca8", "supervised": "#d1495b", "sweep": "#2a2a2a"}
MRK = {"3": "o", "4": "s", "5": "^"}

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.5))

# ---- left: ground truth, no model ----
g = d["physics_P1"]["grid"]
cs = np.array([r["c"] for r in g]); rt = np.array([r["ratio"] for r in g])
ax[0].plot(cs, rt, "-", color=COL["sweep"], lw=1.2, zorder=2)
ax[0].plot(cs, rt, "o", color=COL["sweep"], ms=4.5, zorder=3)
ax[0].set_yscale("log")
ax[0].set_title("ground-truth trajectories\n(no model)", fontsize=9)
ax[0].set_ylabel("invariance ratio of $T_c$", fontsize=8.5)

# ---- right: the learned transition ----
for m in d["models"]:
    s = m["ckpt"].split("_s")[1][0]
    c = np.array([r["c"] for r in m["sweep"]]); y = np.array([r["rho_obs"] for r in m["sweep"]])
    ax[1].plot(c, y, "-", color=COL["sweep"], lw=0.8, alpha=0.5, zorder=2)
    ax[1].plot(c, y, MRK[s], color=COL["sweep"], ms=5, label=f"seed {s}", zorder=3)
lf = float(np.median([mm["unsupervised"]["rho_obs"] for mm in e18["models"]]))
sup = float(np.median([mm["supervised"]["rho_obs"] for mm in e18["models"]]))
ax[1].axhline(sup, color=COL["supervised"], ls="--", lw=1.1, zorder=1)
ax[1].axhline(lf, color=COL["label-free"], ls="--", lw=1.1, zorder=1)
ax[1].set_yscale("log")
# Headroom first, so the two reference annotations have somewhere to sit that is not on their
# own line and not off the axis. Both were invisible or struck through before this was set.
ax[1].set_ylim(lf * 0.30, 0.55)
ax[1].annotate("supervised probe", xy=(-0.48, sup * 1.6), fontsize=7.2,
               color=COL["supervised"], va="bottom")
# Right side, below the line: the left is taken by the legend and the curve is high on both
# flanks, so this is the only clear spot. Checked on the rendered page, not just in code.
ax[1].annotate("label-free $C$", xy=(0.50, lf * 0.80), fontsize=7.2,
               color=COL["label-free"], va="top", ha="right")
ax[1].set_title("the model's own transition", fontsize=9)
ax[1].set_ylabel(r"$\rho_{\mathrm{obs}}$   (lower is better)", fontsize=8.5)
# Lower right is the only quadrant with no data and no reference line.
# Lower LEFT: the only region clear of both the data (which is high on the left) and the
# blue reference line, which spans the full width and struck through the legend at lower right.
ax[1].legend(fontsize=7.5, frameon=False, loc="lower left", handletextpad=0.4,
             borderaxespad=0.6, labelspacing=0.3)

for a in ax:
    a.axvline(CS, color="0.55", ls=":", lw=1.1, zorder=1)
    a.annotate(r"$c^\star$ predicted", xy=(CS, 0.965), xycoords=("data", "axes fraction"),
               fontsize=7.2, color="0.35", ha="left", va="top", rotation=0,
               xytext=(4, 0), textcoords="offset points")
    a.set_xlabel(r"target coefficient $c$   in   $T_c = E + c\,\dot\theta\sin\theta$", fontsize=8.5)
    a.grid(axis="y", color="0.9", lw=0.6)
    a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.tick_params(labelsize=7.5)

fig.tight_layout()
out = pathlib.Path(__file__).resolve().parent / "figures" / "fig2_shadow_sweep.pdf"
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
print(f"  physics argmin c = {cs[rt.argmin()]:+.4f}  ratio {rt.min():.3e}  ({rt[cs==0][0]/rt.min():.0f}x better than c=0)")
for m in d["models"]:
    b = min(m["sweep"], key=lambda r: r["rho_obs"])
    print(f"  {m['ckpt'][-24:]}  argmin c={b['c']:+.4f}  rho_obs={b['rho_obs']:.5f}")
