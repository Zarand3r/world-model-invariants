"""The paper's published numbers, read from the committed run logs.

The bench recomputes everything live, which is the point of it — but a live number is only
interesting next to the one that was published. These come from `runs/*.json`, need no GPU and no
checkpoint, and are the same files `paper/make_figures.py` reads, so a reference line drawn here
cannot drift from a figure in the paper.
"""
import functools
import json
import pathlib

import numpy as np

RUNS = pathlib.Path(__file__).resolve().parents[2] / "runs"


def _load(name):
    return json.loads((RUNS / name).read_text())


@functools.lru_cache(maxsize=1)
def refs() -> dict:
    ext = {r["ckpt"].split("/")[-1].replace(".pt", ""): r
           for r in _load("dreamer_extraction_prereg_ld12.json")}
    ref = _load("dreamer_refusal.json")
    edit = _load("dreamer_edit.json")
    lev = {r["ckpt"].split("/")[-1].replace(".pt", ""): r for r in _load("dreamer_leverage.json")}
    untrained = [r["rho_energy"] for r in _load("dreamer_untrained_null.json")]

    per_model = {}
    for arm in ("conservative", "damped"):
        for r in ref[arm]:
            k = r["ckpt"].split("/")[-1].replace(".pt", "")
            per_model.setdefault(k, {}).update(
                {"arm": arm, "rho_energy": r["rho_energy"], "drift_of_C": r["drift_of_C"],
                 "pairing_residual": r["pairing_residual"],
                 "heldout_invariance_ratio": r.get("heldout_invariance_ratio")})
    for k, r in ext.items():
        per_model.setdefault(k, {}).update({"rho_energy": r["rho_energy"],
                                            "pairing_residual": r["pairing_residual"],
                                            "participation_ratio": r["participation_ratio"]})
    for k, r in lev.items():
        per_model.setdefault(k, {}).update({"rho_V_D": r["rho_V_D"], "rho_D_edit": r["rho_D_edit"]})
    for r in edit["A_conservative_own"]:
        k = r["ckpt"].split("/")[-1].replace(".pt", "")
        per_model.setdefault(k, {}).update(
            {"dose": r["rollout_by_alpha"], "normalised_slope": r["normalised_slope"],
             "relative_change_at_max_alpha": r["relative_change_at_max_alpha"]})

    null = [r["relative_change_at_max_alpha"] for r in edit["B_conservative_random"]]
    return {
        "per_model": per_model,
        "untrained_rho": sorted(untrained),
        "random_null": {
            # np.median, not the upper middle value: with 60 draws the two differ (55.2% vs the
            # 53.9% the paper reports) and the UI would be quoting a number the paper does not.
            "n": len(null), "median": float(np.median(null)),
            "improving": sum(1 for x in null if x < 0),
            "values": sorted(null),
            "slopes": sorted(r["normalised_slope"] for r in edit["B_conservative_random"]),
        },
        "damped_dose": [r["relative_change_at_max_alpha"] for r in edit["C_damped_own"]],
        "alphas": [0.0, 0.05, 0.1, 0.2, 0.4],
    }
