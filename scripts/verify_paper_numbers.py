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

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS, TEX = ROOT / "runs", ROOT / "paper1.2" / "sections"
_RAW = "\n".join(p.read_text() for p in sorted(TEX.glob("*.tex")))
# LaTeX writes a minus as `$-$`, so a literal "-42.2" never appears in the source.
CORPUS = _RAW.replace("$-$", "-").replace("--", "-")
CS = 0.125

def load(n): return json.loads((RUNS / n).read_text())

e18, e19 = load("e18_supervised_baseline.json"), load("e19_shadow_sweep.json")
f4b = load("f4b_recovery.json")
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

# --- an overclaim guard: any "N of N" claim about the supervised probe increasing drift ---
inc = sum(1 for r in sup if r["effect_pct"] > 0)
bad = re.search(r"increases[^.]{0,80}?3 of 3", CORPUS, re.S | re.I)
CHECKS.append((f"no 'increases on 3 of 3' claim (actual: {inc} of {len(sup)} increase)",
               "-", bad is None))

w = max(len(c[0]) for c in CHECKS)
fails = 0
for label, s, ok in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {label:<{w}}  {s}")
    fails += not ok
print(f"\n{len(CHECKS) - fails}/{len(CHECKS)} checks pass")
sys.exit(1 if fails else 0)
