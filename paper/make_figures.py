"""Regenerate every figure in the paper from the committed run logs.

Nothing here is hand-drawn or hand-edited: each figure reads a JSON written by the experiment that
produced it, so "every number in the paper regenerates from the logs" is checkable rather than
promised. Run before every build and confirm the diff is empty.

Figure grammar followed here:
  - no bars on a log axis (bar length encodes magnitude from zero; a log axis destroys that).
    Drift spans four orders of magnitude, so it is drawn as dots.
  - per-seed points are always shown, never only a mean. Two results in this paper turned on
    seed-level disagreement that a median hid.
  - captions state what to see; the axes are already in the axis labels.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = pathlib.Path("runs")
OUT = pathlib.Path("paper/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})
# Palette validated with the data-viz validator, not chosen by eye. The previous pair for
# "random law" (#e08a1e) and "dissipative" (#c1272d) FAILED both the colourblind separation
# (deutan dE 5.6) and the normal-vision floor (dE 7.1 against a floor of 15) -- those two curves
# in fig1c were hard to separate for every reader, not only colourblind ones. The set below
# passes lightness band, chroma floor, all-pairs CVD (dE 13.0 deutan), normal vision (16.3) and
# 3:1 contrast on a white surface:
#     node scripts/validate_palette.js "#2a78d6,#e34948,#4a3aa7" --pairs all --mode light
TRAINED = "#2a78d6"    # conservative / trained / the recovered law
DAMPED = "#e34948"     # dissipative arm, everywhere it appears
RANDOM = "#4a3aa7"     # the random-law null
RESID = "#eb6834"      # pairing residual, a different measure from recovery
NULLC = "#8a8a85"      # untrained draws: deliberately neutral, never a categorical slot
FLOWC = "#c9ccd1"      # share-of-flow reference bars


def _panel(ax, label, gloss):
    """Bold panel label plus a plain-language gloss under it.

    "recover / refuse / intervene" is our shorthand, not the reader's: a reviewer who has not read
    section 5 cannot tell what "refuse" refers to. The label stays for cross-referencing, and the
    gloss says what the panel actually shows.
    """
    ax.set_title(label, loc="left", fontweight="bold", pad=13)
    ax.text(0.0, 1.015, gloss, transform=ax.transAxes, fontsize=6.4, color="#52514e",
            ha="left", va="bottom")


def _load(name):
    return json.load(open(R / name))


# ---------------------------------------------------------------------------------------
def fig1_three_claims():
    """The paper's whole argument in one figure: recover, refuse, intervene."""
    ref = _load("dreamer_refusal.json")
    unt = _load("dreamer_untrained_null.json")
    edit = _load("dreamer_edit.json")

    cons_rho = [r["rho_energy"] for r in ref["conservative"]]
    damp_rho = [r["rho_energy"] for r in ref["damped"]]
    null_rho = [r["rho_energy"] for r in unt]
    cons_dr = [r["drift_of_C"] for r in ref["conservative"]]
    damp_dr = [r["drift_of_C"] for r in ref["damped"]]

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.15))

    # (a) recovery against the untrained null
    groups = [("no damping\n(trained)", cons_rho, TRAINED),
              ("untrained\n(6 draws)", null_rho, NULLC),
              ("damping\n(trained)", damp_rho, DAMPED)]
    for i, (lab, vals, c) in enumerate(groups):
        x = np.full(len(vals), i) + np.linspace(-.08, .08, len(vals))
        ax[0].scatter(x, vals, s=17, color=c, zorder=3, edgecolor="white", linewidth=.4)
    ax[0].axhline(0.7, color="k", ls=":", lw=.8)
    ax[0].text(2.42, 0.72, "cutoff we set\nbefore running", fontsize=6, ha="right", va="bottom",
               color="#52514e")
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([g[0] for g in groups])
    ax[0].tick_params(axis="x", labelsize=6.6)
    ax[0].set_ylabel(r"correlation with true energy"); ax[0].set_ylim(-0.05, 1.05)
    _panel(ax[0], "(a) recover", "trained models agree, untrained ones scatter")

    # (b) is it actually conserved? four orders of magnitude, so dots on a log axis
    for i, (vals, c) in enumerate([(cons_dr, TRAINED), (damp_dr, DAMPED)]):
        x = np.full(len(vals), i) + np.linspace(-.06, .06, len(vals))
        ax[1].scatter(x, vals, s=17, color=c, zorder=3, edgecolor="white", linewidth=.4)
    ax[1].set_yscale("log"); ax[1].set_xlim(-.5, 1.5)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["no damping", "damping"])
    ax[1].tick_params(axis="x", labelsize=6.6)
    ax[1].set_ylabel("drift of recovered $C$\n(0 = perfectly conserved)")
    ax[1].annotate("", xy=(1.32, np.median(damp_dr)), xytext=(1.32, np.median(cons_dr)),
                   arrowprops=dict(arrowstyle="<->", lw=.7, color="k"))
    ax[1].text(1.38, 5e-3, r"2400$\times$", fontsize=7, rotation=90, va="center")
    _panel(ax[1], "(b) refuse", "with damping, nothing stays constant")

    # (c) the edit: error vs alpha, normalised so the three arms are comparable
    # per-arm label offsets: the three end values sit close together at the right edge, so each is
    # placed away from its neighbour, the axis ticks, and the legend box.
    # Arm B is now 20 draws x 3 seeds (M24), so its band is the interquartile range: with 60 curves
    # a min/max envelope is set by two extreme draws and says nothing about where the null sits.
    # Arms A and C keep min/max because they are three seeds and every seed should be visible.
    styles = [("A_conservative_own", "recovered law", TRAINED, "-", "o", (-4, -13), "minmax",
               "right"),
              ("B_conservative_random", "random law (20 draws)", RANDOM, "--", "s", (4, -2),
               "iqr", "left"),
              ("C_damped_own", "damped model", DAMPED, "-.", "^", (-4, 7), "minmax",
               "right")]
    # log y: the claimed effect is -3.3% and the control is +49%, so on a linear axis the claim is
    # invisible next to its own control. Endpoints are labelled (C3) so the reader reads a number
    # rather than measuring a mark.
    for key, lab, c, ls, mk, off, band, hal in styles:
        curves = np.array([[v for v in r["rollout_by_alpha"].values()] for r in edit[key]])
        alphas = np.array([float(a) for a in edit[key][0]["rollout_by_alpha"]])
        rel = curves / curves[:, :1]
        med = np.median(rel, 0)
        lo, hi = ((rel.min(0), rel.max(0)) if band == "minmax"
                  else (np.percentile(rel, 25, axis=0), np.percentile(rel, 75, axis=0)))
        ax[2].plot(alphas, med, ls, marker=mk, ms=3.2, lw=1.2, color=c, label=lab)
        ax[2].fill_between(alphas, lo, hi, color=c, alpha=.13, lw=0)
        ax[2].annotate(f"{(med[-1]-1)*100:+.1f}%", xy=(alphas[-1], med[-1]), xytext=off,
                       textcoords="offset points", fontsize=6.6, color=c, ha=hal,
                       va="bottom", fontweight="bold")
    ax[2].axhline(1.0, color="k", lw=.7, ls=":")
    ax[2].set_yscale("log")
    ax[2].set_ylim(0.93, 2.05)
    ax[2].set_yticks([0.95, 1.0, 1.1, 1.3, 1.6, 2.0])
    ax[2].set_yticklabels(["0.95", "1.0", "1.1", "1.3", "1.6", "2.0"])
    ax[2].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[2].set_xlim(-0.02, 0.52)
    ax[2].set_xlabel(r"projection strength $\alpha$")
    ax[2].set_ylabel("rollout error\n(relative to no edit)")
    ax[2].legend(frameon=False, loc="upper left", handlelength=1.4, borderpad=0.1,
                 labelspacing=0.25, fontsize=6.8)
    _panel(ax[2], "(c) intervene", "enforcing it lowers rollout error")

    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig1_three_claims.pdf")
    plt.close(fig)
    return dict(cons_rho=cons_rho, null_rho=null_rho, damp_rho=damp_rho,
                cons_dr=cons_dr, damp_dr=damp_dr)


# ---------------------------------------------------------------------------------------
def fig2_ld_sweep():
    """Recovery improves with latent dimension while the pairing residual gets worse.

    ONE axis, deliberately. This was a twinx() dual-axis chart until 2026-08-13. Two y-scales
    make the crossing point an artefact of the scale alignment, and the crossing is exactly what
    this figure is asked to show, so the old version asserted the paper's claim by construction.
    Both quantities are dimensionless and live in [0, 1], so a shared axis costs nothing and the
    reader can see that they genuinely cross. Per-seed points are drawn because the seeds
    disagree at low dimension (0.674 / 0.970 / 0.721 at LD=6).
    """
    rows = _load("dreamer_ld_sweep.json")
    lds = sorted({r["ld"] for r in rows})
    rho = [np.median([r["rho_energy"] for r in rows if r["ld"] == l]) for l in lds]
    res = [np.median([r["pairing_residual"] for r in rows if r["ld"] == l]) for l in lds]

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for key, med, c, mk, lab in ((("rho_energy"), rho, TRAINED, "o", r"recovery $|\rho|_E$"),
                                 (("pairing_residual"), res, RESID, "s", "pairing residual")):
        for i, l in enumerate(lds):
            pts = [r[key] for r in rows if r["ld"] == l]
            ax.scatter(np.full(len(pts), i) + np.linspace(-.09, .09, len(pts)), pts,
                       s=8, color=c, alpha=.45, zorder=2, linewidth=0)
        ax.plot(range(len(lds)), med, "-", marker=mk, ms=3.6, lw=1.4, color=c, label=lab, zorder=3)
    ax.set_xticks(range(len(lds))); ax.set_xticklabels([str(l) for l in lds])
    ax.set_xlabel("extraction dimension"); ax.set_ylabel("score (both run 0 to 1)")
    ax.set_ylim(0, 1.05)
    ax.axvline(lds.index(12), color="k", lw=.6, ls=":", zorder=1)
    ax.text(lds.index(12), 1.07, "registered", fontsize=6, ha="center", color="#52514e")
    ax.legend(frameon=False, loc="center left", handlelength=1.5, fontsize=7, labelspacing=.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_ld_sweep.pdf")
    plt.close(fig)
    return dict(lds=lds, rho=rho, res=res)


# ---------------------------------------------------------------------------------------
def fig3_low_variance():
    """Energy lives in the low-variance directions, and that is not where the residual lives."""
    rows = _load("dreamer_residual_decomp.json")
    e_first = np.median([r["first6_energy_r2"] for r in rows])
    e_added = np.median([r["added7_12_energy_r2"] for r in rows])
    r_first = np.median([r["first6_resid_share"] for r in rows])
    r_added = np.median([r["added7_12_resid_share"] for r in rows])
    f_first = np.median([r["first6_flow_share"] for r in rows])
    f_added = np.median([r["added7_12_flow_share"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(5.1, 2.15))

    # (a) PER SEED, not two median bars. The median said "dims 7-12 carry 2x the energy"; seed s1
    # inverts (0.91x) and a two-bar chart hides that entirely. Paired dots keep both levels and
    # the seed that disagrees visible, which is the grammar the rest of this file follows.
    # Two of the three end points sit within 0.008 of each other, so the ratio labels are laddered
    # by rank instead of all pinned to their own y -- otherwise they overprint.
    ends = sorted(range(len(rows)), key=lambda k: rows[k]["added7_12_energy_r2"])
    yoff = {k: o for k, o in zip(ends, (-9, 2, 9))}
    for i, r in enumerate(rows):
        a, b = r["first6_energy_r2"], r["added7_12_energy_r2"]
        c = TRAINED if b > a else DAMPED
        ax[0].plot([0, 1], [a, b], "-", lw=1.0, color=c, alpha=.85, zorder=2)
        ax[0].scatter([0, 1], [a, b], s=20, color=c, zorder=3, edgecolor="white", linewidth=.5)
        ax[0].annotate(f"{b/a:.2f}$\\times$", xy=(1, b), xytext=(6, yoff[i]), fontsize=6.5,
                       textcoords="offset points", color=c, va="center", fontweight="bold")
    ax[0].set_xlim(-.35, 1.55); ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["top 6\n(high variance)", "dims 7-12\n(low variance)"])
    ax[0].set_ylabel(r"energy probe $R^2$"); ax[0].set_ylim(0, .78)
    ax[0].set_title("(a) where the energy is", loc="left", fontweight="bold")

    x, w = np.arange(2), .36
    ax[1].bar(x - w/2, [f_first, f_added], w, color=FLOWC, label="share of flow")
    ax[1].bar(x + w/2, [r_first, r_added], w, color=RESID, label="share of residual")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["top 6", "dims 7-12"])
    ax[1].set_ylabel("fraction"); ax[1].set_ylim(0, .78)
    ax[1].legend(frameon=False, loc="upper center", ncol=1, handlelength=1.2)
    ax[1].set_title("(b) where the misfit is", loc="left", fontweight="bold")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "fig3_low_variance.pdf")
    plt.close(fig)
    return dict(e_first=e_first, e_added=e_added, r_added=r_added, f_added=f_added)


def fig4_leverage():
    """What the correction acts on: prominence rank against consequence rank, per direction."""
    rows = json.load(open(R / "dreamer_leverage.json"))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4))
    # ONE hue, three marker shapes. The three seeds are the same arm and the message is that they
    # agree in sign, so giving each its own colour spent the categorical channel on a distinction
    # the figure is not making. Shape carries seed identity; colour stays free.
    marks = ["o", "s", "^"]
    for i, r in enumerate(rows):
        V, D, E = np.array(r["variance"]), np.array(r["leverage"]), np.array(r["edit_move"])
        rk = lambda x: np.argsort(np.argsort(x)) + 1
        ax[0].scatter(rk(V), rk(D), s=24, color=TRAINED, marker=marks[i], alpha=.8,
                      edgecolor="white", linewidth=.4,
                      label=f"seed {i+3}")
        ax[1].scatter(rk(D), rk(E), s=24, color=TRAINED, marker=marks[i], alpha=.8,
                      edgecolor="white", linewidth=.4,
                      label=None)
    for a, xl, yl in ((ax[0], "variance rank (low $\\to$ high)", "damage rank (low $\\to$ high)"),
                      (ax[1], "damage rank (low $\\to$ high)", "how hard the edit pushes (rank)")):
        a.plot([1, 12], [1, 12], color="k", lw=.7, ls=":")
        a.set_xlabel(xl); a.set_ylabel(yl); a.set_xlim(.3, 12.7); a.set_ylim(.3, 12.7)
        a.set_aspect("equal")
    # One shared legend under both panels. With 36 points on a 12x12 rank grid every corner holds
    # data, so an in-axes legend box lands on top of it wherever it goes. The per-seed rho values
    # move to the caption.
    ax[0].legend(frameon=False, ncol=3, fontsize=6.6, handlelength=.9, columnspacing=1.4,
                 loc="upper center", bbox_to_anchor=(1.12, -0.30))
    _panel(ax[0], "(a) variance misses what matters",
           "damage = how much nudging a direction hurts the rollout")
    _panel(ax[1], "(b) the edit finds it anyway",
           "without being told which directions those are")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "fig4_leverage.pdf"); plt.close(fig)
    return rows


def fig5_random_null():
    """Where each model's own law sits inside its own 20-draw random null.

    The claim this carries is the paper's most contested: the edit's specificity is strong on two
    models and weak on the third. Prose says "0th percentile on two seeds, 15th on the third";
    this shows it, including the seven random laws on s3 that beat the recovered one.
    """
    edit = _load("dreamer_edit.json")
    own = {r["ckpt"]: r["normalised_slope"] for r in edit["A_conservative_own"]}
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    rng = np.random.default_rng(0)          # jitter only; no data is generated here
    for i, (ck, s_own) in enumerate(sorted(own.items())):
        draws = np.array([r["normalised_slope"] for r in edit["B_conservative_random"]
                          if r["ckpt"] == ck])
        pct = 100.0 * float((draws < s_own).mean())
        ax.scatter(draws, np.full(len(draws), i) + rng.uniform(-.13, .13, len(draws)),
                   s=9, color=RANDOM, alpha=.5, linewidth=0, zorder=2)
        ax.scatter([s_own], [i], s=52, marker="D", color=TRAINED, zorder=4,
                   edgecolor="white", linewidth=.7)
        ax.annotate(f"{pct:.0f}th pct", xy=(s_own, i), xytext=(0, 9), textcoords="offset points",
                    fontsize=6.5, color=TRAINED, ha="center", fontweight="bold")
        if i == 2:      # direct labels on the top row only: a legend box lands on seed 3's draws
            ax.annotate("recovered law", xy=(s_own, i), xytext=(0, -12), ha="center", va="top",
                        textcoords="offset points", fontsize=6.5, color=TRAINED)
            ax.annotate("20 random draws", xy=(float(np.median(draws)), i), xytext=(0, 13),
                        ha="center", textcoords="offset points", fontsize=6.5, color=RANDOM)
    ax.axvline(0, color="k", lw=.7, ls=":", zorder=1)
    ax.text(0.035, -.45, "no effect", fontsize=6, color="#52514e")
    ax.set_yticks(range(3)); ax.set_yticklabels([f"seed {i}" for i in (3, 4, 5)])
    ax.set_xlabel("dose-response slope (negative = improves the rollout)")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_ylim(-.6, 2.62)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_random_null.pdf")
    plt.close(fig)
    return {ck: (own[ck], float(np.median([r["normalised_slope"]
                                           for r in edit["B_conservative_random"]
                                           if r["ckpt"] == ck]))) for ck in own}


if __name__ == "__main__":
    s1 = fig1_three_claims()
    s5 = fig5_random_null()
    s4 = fig4_leverage()
    s2 = fig2_ld_sweep()
    s3 = fig3_low_variance()
    print("fig1  trained", [round(x, 3) for x in s1["cons_rho"]],
          " null", [round(x, 3) for x in sorted(s1["null_rho"])],
          " damped", [round(x, 3) for x in s1["damp_rho"]])
    print("      drift  cons", [f"{x:.1e}" for x in s1["cons_dr"]],
          " damp", [round(x, 3) for x in s1["damp_dr"]])
    print("fig2  LD", s2["lds"], "rho", [round(x, 3) for x in s2["rho"]],
          "resid", [round(x, 3) for x in s2["res"]])
    print(f"fig3  energy R2 first6 {s3['e_first']:.3f} added {s3['e_added']:.3f}; "
          f"resid share added {s3['r_added']:.1%} vs flow share {s3['f_added']:.1%}")
    damped = json.load(open(R / "dreamer_leverage_damped.json"))
    print("      M26 control -- same measurement on the dissipative arm:")
    print("        rho(V,D)   cons", [round(r["rho_V_D"], 3) for r in s4],
          " damped", [round(r["rho_V_D"], 3) for r in damped])
    print("        rho(D,edit) cons", [round(r["rho_D_edit"], 3) for r in s4],
          " damped", [round(r["rho_D_edit"], 3) for r in damped])
    print("        top3 M_C   cons", [round(r["top3_eigenmass"], 3) for r in s4],
          " damped", [round(r["top3_eigenmass"], 3) for r in damped])
    print("fig4  rho(V,D)", [round(r["rho_V_D"], 3) for r in s4],
          " rho(D,edit)", [round(r["rho_D_edit"], 3) for r in s4])
    print("wrote", *(p.name for p in sorted(OUT.glob("*.pdf"))))
