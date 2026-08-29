"""Re-derive every headline number in paper1.2 from the run records and check the prose agrees.

Why this exists: two number defects reached the paper in a single day. One was an overclaim
("increases drift on 3 of 3 models" when one seed reduces it) caught only because a figure was
rendered and looked at. The other was a flattering-seed quote (14x, which was seed 3 of 14/10/12)
caught only because the numbers happened to be re-traced. Neither was catchable by reading prose.

Each check recomputes a value from `runs/*.json`, formats it exactly as the paper states it, and
greps the .tex sources for that literal string. A FAIL means the prose and the run records have
drifted apart -- it does not say which one is wrong.
"""
import json, pathlib, re, statistics as st, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS, TEX = ROOT / "runs", ROOT / "paper1.2" / "sections"
# CLAIMS.md is checked too: it carried stale mean-based numbers for a full day because this
# verifier only ever globbed sections/*.tex, and nothing else compared the two.
_RAW = "\n".join(p.read_text() for p in sorted(TEX.glob("*.tex")))
_RAW += "\n" + (TEX.parent / "CLAIMS.md").read_text()
# The reviewer handoff restates the paper's figures, so it carries the same drift risk and is
# guarded by the same checks. CLAIMS.md silently held mean-based numbers for a day before it
# was added here; the handoff is the same shape of risk.
_RAW += "\n" + (ROOT / "docs" / "REVIEWER_HANDOFF.md").read_text()
# Lines marked `<!-- superseded: ... -->` deliberately NAME an old value in order to record that it
# was wrong. A repo that documents its own corrections will otherwise trip every stale-value guard
# it owns, so those annotations are excluded from the corpus the guards read.
_RAW = "\n".join(l for l in _RAW.split("\n") if "<!-- superseded:" not in l)
# LaTeX writes a minus as `$-$`, so a literal "-42.2" never appears in the source.
CORPUS = _RAW.replace("$-$", "-").replace("--", "-")
CS = 0.125

def load(n): return json.loads((RUNS / n).read_text())

e18, e19 = load("e18_supervised_baseline.json"), load("e19_shadow_sweep.json")
f4b = load("f4b_recovery.json")
e10b = load("e10b_matched_band_pool400.json")
f2 = load("f2_trust_signal.json")
f5 = load("f5_planning.json")
f5g = load("f5_gate0.json")
f1b = load("f1_balance_measured.json")
f1a = load("f1_action_use.json")
f3 = load("f3_constraint.json")
f6 = load("f6_models.json")
lf  = [m["unsupervised"] for m in e18["models"]]
sup = [m["supervised"] for m in e18["models"]]
at  = lambda m, c: next(r for r in m["sweep"] if r["c"] == c)

CHECKS = []
def check(label, value, *, fmt="{:.1f}", must_appear=None):
    s = must_appear if must_appear is not None else fmt.format(value)
    CHECKS.append((label, s, s in CORPUS))

# --- E18 ---
check("E18 label-free rho_obs median (6.9e-3)", st.median(r["rho_obs"] for r in lf) * 1e3, fmt="{:.1f}")
check("E18 supervised rho_obs median (4.58e-2)", st.median(r["rho_obs"] for r in sup) * 1e2, fmt="{:.2f}")
check("E18 label-free effect median", st.median(r["effect_pct"] for r in lf), fmt="{:.1f}")
check("E18 supervised effect median", st.median(r["effect_pct"] for r in sup), fmt="{:.1f}")
check("E18 rho_obs ratio sup/lf",
      st.median(r["rho_obs"] for r in sup) / st.median(r["rho_obs"] for r in lf), fmt="{:.1f}")

# --- E19 ---
phys = e19["physics_P1"]
check("E19 physics improvement at c*", phys["improvement_x"], fmt="{:.0f}")
check("E19 shadow coefficient", CS, fmt="{:.3f}")
wrong = [at(m, -CS)["rho_obs"] / at(m, CS)["rho_obs"] for m in e19["models"]]
check("E19 wrong-sign control (median, NOT best seed)", st.median(wrong), fmt="{:.0f}")
red = sorted(at(m, 0.0)["rho_obs"] / at(m, CS)["rho_obs"] for m in e19["models"])
check("E19 P3 reduction range low", red[0], fmt="{:.1f}")
check("E19 P3 reduction range high", red[-1], fmt="{:.1f}")
check("E19 residual to label-free",
      st.median(at(m, CS)["rho_obs"] for m in e19["models"]) / st.median(r["rho_obs"] for r in lf),
      fmt="{:.2f}")
check("E19 rho_E at c*", st.median(at(m, CS)["rho_E"] for m in e19["models"]), fmt="{:.3f}")
for m in e19["models"]:
    s = m["ckpt"].split("_s")[1][0]
    check(f"E19 effect at c*, seed {s}", at(m, CS)["effect_pct"], fmt="{:.1f}")

# --- F4b ---
term = [m["recovered"]["rho_obs"] for m in f4b["models"] if m["ckpt"].endswith("step60000.pt")]
check("F4b degradation vs RSSM",
      st.median(term) / st.median(r["rho_obs"] for r in lf), fmt="{:.0f}")

# --- F2 (registered NEGATIVE: drift is not a useful trust signal) ---
def _rank_np(x):
    x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
def _spear(a, b):
    ra, rb = _rank_np(a) - _rank_np(a).mean(), _rank_np(b) - _rank_np(b).mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-30))
for _sig in ("acc_drift", "latent_disp"):
    for _m in f2["models"]:
        _s = _spear(_m["signals"][_sig], _m["target_energy_error"])
        check(f"F2 {_sig} {_m['ckpt'][-16:-3]}", _s, fmt="{:+.2f}")
# Guard the RISK (claiming it is untested), not a phrasing. An earlier version required the
# literal "was tested" and failed on a legitimate rewrite during compression, which is a guard
# training you to ignore it. The numbers themselves are checked above.
CHECKS.append(("F2 not described as untested", "-",
               not re.search(r"drift[^.]{0,60}(is|remains) untested", CORPUS, re.I)))

# --- F6 timestep scaling (the paper's positive general result) ---
import collections as _c
_by = _c.defaultdict(list)
for _m in f6["models"]:
    _by[_m["dt"]].append(_m)
for _dt in sorted(_by):
    _r0 = float(np.median([z["rho_obs_at_r0"] for z in _by[_dt]]))
    _r1 = float(np.median([z["rho_obs_at_r1"] for z in _by[_dt]]))
    # the paper quotes these to the precision it uses: 2 dp for the small ones, 1 dp for 13.5
    check(f"F6 separation dt={_dt}", _r0 / _r1, fmt=("{:.1f}" if _r0 / _r1 >= 10 else "{:.2f}"))
_x = np.array([m["dt"] for m in f6["models"]]); _y = np.array([m["c_recovered"] for m in f6["models"]])
_slope0 = float((_x @ _y) / (_x @ _x))
_res = _y - _slope0 * _x
_se0 = float(np.sqrt(((_res ** 2).sum() / (len(_x) - 1)) / (_x @ _x)))
check("F6 origin-forced slope", _slope0, fmt="{:.3f}")
check("F6 origin-forced slope CI", 1.96 * _se0, fmt="{:.3f}")
_exact = sum(1 for m in f6["models"] if abs(m["argmin_r"] - 1.0) < 1e-9)
CHECKS.append((f"F6 argmin exactly at r=1 on {_exact} of {len(f6['models'])} models", "-",
               f"{_exact} of {len(f6['models'])}" in CORPUS or f"{_exact} of\n12" in CORPUS))
_p4 = sum(1 for m in f6["models"] if m["rho_obs_at_rm1"] > m["rho_obs_at_r1"])
CHECKS.append((f"F6 wrong-sign control beaten {_p4}/12", "-", _p4 == len(f6["models"])))
CHECKS.append(("F6 registered P3 failure is stated", "-",
               "registered failure" in CORPUS.lower() or "excludes zero" in CORPUS.lower()))

# --- F3 constraint-residual bound (registered NEGATIVE; A1 gate fired) ---
_r = sorted(m["rho_G_Gtrue"] for m in f3["models"])
check("F3 rho(G,Gtrue) low", _r[0], fmt="{:.3f}")
check("F3 rho(G,Gtrue) high", _r[-1], fmt="{:.3f}")
CHECKS.append((f"F3 A1 failed on all {len(f3['models'])} models (gate fired)", "-",
               not any(m["A1_pass"] for m in f3["models"])))
CHECKS.append(("F3: no released-checkpoint result claimed", "-",
               not re.search(r"walker|DMC|released checkpoint experiment", CORPUS, re.I)))

# --- F1 actuation axis (the paper's fifth dissociation axis) ---
for _m in f1b["models"]:
    _s = _m["ckpt"][-9:-3]
    check(f"F1 rho(C,E) {_s}", _m["rho_C_energy"], fmt="{:.2f}")
    check(f"F1 Spearman(power,dC) {_s}", _m["spearman_pred_obs"], fmt="{:.3f}")
_var = 100 * float(np.mean([m["pearson_pred_obs"] ** 2 for m in f1b["models"]]))
check("F1 variance of dC explained by power (%)", _var, fmt="{:.2f}")
_scale = [m["scale_obs_over_pred"] for m in f1b["models"]]
check("F1 obs/pred scale low", min(_scale), fmt="{:.1f}")
check("F1 obs/pred scale high", max(_scale), fmt="{:.1f}")
# The paper quotes the RANGE, not per-seed values, so check the endpoints. Checking each seed
# fails on a value that is inside the stated range but never written down -- which it did.
_fin = [m["ratio_true_over_shuffled"] for m in f1a["models"] if not re.search(r"step\d+", m["ckpt"])]
check("F1 action-use range low", min(_fin), fmt="{:.3f}")
check("F1 action-use range high", max(_fin), fmt="{:.3f}")

# --- F5 (registered NEGATIVE: no control benefit) ---
_arms = ("none", "conserve", "balance", "probe", "random")
_ret = {a: np.concatenate([[r["return"] for r in m["arms"][a]["rows"]] for m in f5["models"]])
        for a in _arms}
_base = _ret["none"]
_largest = max(abs(_ret[a].mean() - _base.mean()) for a in _arms[1:])
check("F5 largest arm effect", _largest, fmt="{:.4f}")
check("F5 across-episode return SD", _base.std(), fmt="{:.3f}")
_g = f5g["models"][0]["arms"]
_d = (np.array([r["return"] for r in _g["none"]["rows"]])
      - np.array([r["return"] for r in _g["__random_policy__"]["rows"]]))
check("F5 Gate 0 paired margin", _d.mean(), fmt="{:.2f}")
CHECKS.append(("F5 not described as untested", "-",
               not re.search(r"helps a planner is untested|planner[^.]{0,40}untested", CORPUS, re.I)))
CHECKS.append(("F5: no arm claimed to beat no-correction", "-",
               not re.search(r"correction (?:significantly )?improv\w+ (?:planning|control|return)",
                             CORPUS, re.I)))

# --- E10b (400-pool, the matched-design primary set) ---
import math
def _ci(r, n):
    z = math.atanh(r); se = 1 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
for m in e10b["models"]:
    check(f"E10b Spearman {m['ckpt'][-16:-3]}", m["spearman_ratio_repair"], fmt="{:+.2f}")
n_excl = sum(1 for m in e10b["models"]
             if (lambda t: t[0] * t[1] > 0)(_ci(m["spearman_ratio_repair"], len(m["rows"]))))
CHECKS.append((f"E10b CI excludes zero on {n_excl} of 3 (paper must say 'one of')",
               "-", ("one of" in CORPUS.lower()) == (n_excl == 1)))
CHECKS.append(("E10b: no unreproducible +0.71 anywhere in the paper",
               "-", "+0.71" not in CORPUS))
# Guard the stale CLAIM, not the digits. The first version matched the ASCII "6.3x" and missed a
# stale "6.3\\times" in the introduction for many iterations; the second matched bare "6.3" and
# flagged two legitimate uses (a displacement ratio, and the low end of the rho_obs range).
CHECKS.append(("no stale '6.3x less/better preserved' claim (the ratio is 6.7)", "-",
               not re.search(r"6\.3\\times[^.]{0,40}(less|better) preserved", CORPUS)))
for stale, why in (
                   ("7.26e-03", "label-free rho_obs median is 6.85e-03, not the mean"),
                   ("+20.0", "supervised effect median is +26.8%, not the mean +20.0%")):
    CHECKS.append((f"no stale mean-based value {stale!r} ({why})", "-", stale not in CORPUS))

# --- ICLR submission format ---
_main = (TEX.parent / "main.tex").read_text()
CHECKS.append(("uses the ICLR 2025 style, not a geometry approximation", "-",
               "iclr2025_conference" in _main and "geometry}" not in _main))
_pdf = TEX.parent / "main.pdf"
if _pdf.exists():
    import subprocess as _sp
    _txt = _sp.run(["pdftotext", str(_pdf), "-"], capture_output=True, text=True).stdout
    # double-blind: the author name must not appear anywhere in the rendered PDF
    CHECKS.append(("double-blind: author name absent from the PDF", "-",
                   not re.search(r"Richard\s+Bao|richardbao419", _txt, re.I)))
    CHECKS.append(("anonymous submission header present", "-",
                   "Anonymous authors" in _txt))

# --- figure hygiene: filename prefix must be unique, and match the figure it renders as ---
# paper1.2 inherited paper 1.0's filenames, so `fig4_leverage` rendered as Figure 3 and TWO files
# began `fig2_`. That is the collision the archive script's own header warns about, having once
# shipped stale figures. Assert one file per figure number.
_figs = sorted(p.name for p in (TEX.parent / "figures").glob("*.pdf"))
_prefixes = [f.split("_")[0] for f in _figs]
CHECKS.append((f"one figure file per number ({', '.join(_figs)})", "-",
               len(_prefixes) == len(set(_prefixes))))
CHECKS.append(("every shipped figure is referenced by a section", "-",
               all(f in CORPUS for f in _figs)))

# --- an overclaim guard: any "N of N" claim about the supervised probe increasing drift ---
inc = sum(1 for r in sup if r["effect_pct"] > 0)
# Catch "the supervised probe increases drift on 3 of 3" while allowing the CORRECT sentence,
# which says it increases on 2 of 3 and that the label-free scalar repairs on 3 of 3. The negated
# span rejects any match with an intervening "2 of 3".
bad = re.search(r"increases(?:(?!2 of 3)[^.]){0,80}?3 of 3", CORPUS, re.S | re.I)
CHECKS.append((f"no 'increases on 3 of 3' claim (actual: {inc} of {len(sup)} increase)",
               "-", bad is None))

w = max(len(c[0]) for c in CHECKS)
fails = 0
for label, s, ok in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<{w}}  {s}")
    fails += not ok
print(f"\n{len(CHECKS) - fails}/{len(CHECKS)} checks pass")
sys.exit(1 if fails else 0)
