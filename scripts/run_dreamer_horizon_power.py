"""Is S3 Test 1 underpowered, or is the mechanism absent on DreamerV3?

At HORIZON=40 (2.0 s) every model scored held-out R^2 < 0.1 and the integral coefficient came out
b = -1.28 -- the WRONG SIGN against a mechanism that predicts b ~ +1. An ordering between two
near-zero R^2 values is not evidence, so the honest reading is "underpowered", and that is a
testable claim rather than an excuse.

The physics says so directly. The pendulum's period is 2*pi/omega0 = 1.62 s, so a 2.0 s window is
1.2 cycles. Secular phase error grows like the TIME INTEGRAL of the invariant bias; over roughly
one cycle it is swamped by the model's instantaneous state error, which is not secular at all.
The mechanism's signature only separates from that background as t grows.

So sweep the horizon and watch three things move together (or fail to):
  - held-out R^2 of the integral predictor
  - its coefficient b, which should approach +1 rather than wander
  - the across-trajectory rank correlation between the accumulated bias X(T) and the observed
    timing error Delta_tau(T). That is the causal claim in its most robust form: trajectories that
    accumulate more bias must drift more, and Spearman does not care about scale or offset.

If R^2 and Spearman both climb with horizon, the earlier null was power. If they stay flat while
the instrument stays valid, the mechanism is genuinely absent here and that is the reportable
result. The G3 noise floor is re-checked at every horizon, so a longer window cannot buy power by
quietly degrading the instrument.
"""
import json

import numpy as np
import scripts.run_dreamer_mechanism as MECH

CKPTS = [f"runs/dreamer_ref_s{i}.pt" for i in range(3)]
DATA, PROBE = "runs/pendulum_pixels_eval.npz", "runs/pendulum_pixels.npz"
HORIZONS = [20, 40, 80, 120, 160, 189]   # eval data is T=200; WARMUP=10 caps the window at 189

print(__doc__.split("\n\n")[0])
print(f"\npendulum period = {2*np.pi/MECH.OMEGA0:.2f} s;  dt = {MECH.DT}\n")
print(f"  {'H':>4}{'sec':>7}{'cyc':>6}{'flr/dE':>8}| pooled: {'R2_int':>8}{'R2_t^p':>8}{'b':>7}"
      f" | across-traj: {'R2':>7}{'b':>7}{'rho':>7}{'n':>4}")

rows = []
for H in HORIZONS:
    per = []
    for ck in CKPTS:
        r = MECH.run(ck, DATA, ck, HORIZON=H, probe_data=PROBE)
        if not r["VOID"]:
            per.append(r)
    if not per:
        print(f"  {H:>4}{H*MECH.DT:>9.2f}{H*MECH.DT*MECH.OMEGA0/(2*np.pi):>8.2f}"
              f"{'':>12}{'ALL VOID (instrument)':>38}")
        continue
    med = lambda k: float(np.median([x[k] for x in per]))
    ratio = med("floor_E") / max(med("dE_median_abs"), 1e-9)
    row = {"horizon": H, "seconds": H * MECH.DT, "r2_int": med("r2_integral"),
           "r2_pow": med("r2_power"), "b": med("b"), "spearman": med("spearman_XT"),
           "r2_cross": med("r2_cross"), "b_cross": med("b_cross"),
           "floor_ratio": ratio, "n_seeds": len(per)}
    rows.append(row)
    print(f"  {H:>4}{H*MECH.DT:>7.2f}{H*MECH.DT*MECH.OMEGA0/(2*np.pi):>6.2f}{ratio:>8.2f}|"
          f"{row['r2_int']:>17.3f}{row['r2_pow']:>8.3f}{row['b']:>7.2f} |"
          f"{row['r2_cross']:>20.3f}{row['b_cross']:>7.2f}{row['spearman']:>7.3f}"
          f"{len(per):>4}", flush=True)
    json.dump(rows, open("runs/dreamer_horizon_power.json", "w"), indent=2)

print("\n--- VERDICT")
if len(rows) < 2:
    print("  Not enough valid horizons to judge power.")
else:
    best = max(rows, key=lambda r: r["r2_cross"])
    trend = rows[-1]["r2_cross"] - rows[0]["r2_cross"]
    sp_trend = abs(rows[-1]["spearman"]) - abs(rows[0]["spearman"])
    print(f"  best horizon {best['horizon']} ({best['seconds']:.1f} s): across-trajectory R2 "
          f"{best['r2_cross']:+.3f}  b {best['b_cross']:+.2f}  spearman {best['spearman']:+.3f}"
          f"   |  pooled R2_int {best['r2_int']:+.3f} vs R2_t^p {best['r2_pow']:+.3f}")
    if best["r2_cross"] > 0.2 and trend > 0.1:
        print("  POWER was the problem, and the discriminating statistic was the across-trajectory")
        print("  one. The accumulated bias X predicts WHICH trajectories drift most -- something no")
        print("  function of t alone can do -- and it strengthens with horizon as a secular")
        print("  mechanism requires. The frequency-action mechanism reproduces on a real DreamerV3.")
    elif abs(best["spearman"]) > 0.3 and sp_trend > 0:
        print("  PARTIAL: the pooled R^2 stays low, but the across-trajectory dose-response")
        print("  (Spearman between accumulated bias and observed drift) strengthens with horizon.")
        print("  The causal ordering survives; the quantitative form does not yet.")
    else:
        print("  NOT power. R^2 and Spearman stay flat while the instrument stays valid across")
        print("  a 10x range of horizons. The frequency-action mechanism is ABSENT on DreamerV3 —")
        print("  a real negative result, and the paper's mechanism claim stays GRU-only.")
