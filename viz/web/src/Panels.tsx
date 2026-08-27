/** The three analysis panels: the invariant, the law bench, and the directions. */
import type { BundleInfo, Dose, LawScores, Leverage, PaperRefs, Rollout } from "./api";
import { Dots, Frame, Line, extent, fmt, scale, ticksFor } from "./plot";

/** C against true energy, and C(t) drifting away from its own starting level set. */
export function InvariantPanel({ law, roll }: { law: LawScores | null; roll: Rollout | null }) {
  const W = 330, H = 168;
  return (
    <section className="panel">
      <h2>The invariant</h2>
      <div className="pair">
        <div>
          <div className="plottitle">C vs true energy</div>
          {law && (() => {
            const [ex, exh] = extent(law.scatter.E), [cy, cyh] = extent(law.scatter.C);
            const sx = scale(ex, exh, 44, W - 10), sy = scale(cy, cyh, H - 26, 10);
            return (
              <Frame w={W} h={H} xLabel="true energy" yLabel="C"
                     xTicks={ticksFor(ex, exh, sx, 3).ticks}
                     yTicks={ticksFor(cy, cyh, sy, 3).ticks}>
                <Dots xs={law.scatter.E} ys={law.scatter.C} sx={sx} sy={sy} cls="pt-energy" r={1.3} />
              </Frame>
            );
          })()}
          <div className="stat">|ρ|<sub>E</sub> <b>{law ? law.rho_energy.toFixed(4) : "—"}</b></div>
        </div>
        <div>
          <div className="plottitle">C during the rollout</div>
          {roll && (() => {
            const n = roll.C.free.length;
            const all = [...roll.C.free, ...roll.C.corrected, roll.C0];
            const [lo, hi] = extent(all);
            const sx = scale(0, n - 1, 44, W - 10), sy = scale(lo, hi, H - 26, 10);
            const xs = roll.C.free.map((_, i) => i);
            return (
              <Frame w={W} h={H} xLabel="imagined step" yLabel="C"
                     xTicks={ticksFor(0, n - 1, sx, 3).ticks.map(t => ({ ...t, label: t.label.split(".")[0] }))}
                     yTicks={ticksFor(lo, hi, sy, 3).ticks}>
                <line x1={44} y1={sy(roll.C0)} x2={W - 10} y2={sy(roll.C0)} className="level" />
                <text x={W - 12} y={sy(roll.C0) - 4} className="tick" textAnchor="end">C₀</text>
                <Line xs={xs} ys={roll.C.free} sx={sx} sy={sy} cls="ln-free" width={2.4} />
                <Line xs={xs} ys={roll.C.corrected} sx={sx} sy={sy} cls="ln-corrected" width={1.4} />
              </Frame>
            );
          })()}
          <div className="stat">
            drift <b>{law ? law.drift_of_C.toExponential(2) : "—"}</b>
            <span className="hint">within-trajectory variance share, held-out half</span>
          </div>
        </div>
      </div>
    </section>
  );
}

/** Eight mixing sliders over the fitted basis, plus the dose response they produce. */
export function LawBench({ info, law, weights, onWeights, onReset, onRandom, dose, paper, busy }: {
  info: BundleInfo | null; law: LawScores | null; weights: number[];
  onWeights: (w: number[]) => void; onReset: () => void; onRandom: () => void;
  dose: Dose | null; paper: PaperRefs | null; busy: boolean;
}) {
  const W = 330, H = 150;
  const published = info && paper
    ? (paper.per_model[info.ckpt.split("/").pop()!.replace(".pt", "")] as
       Record<string, number | Record<string, number>> | undefined)
    : undefined;
  const pubDose = published?.dose as Record<string, number> | undefined;

  return (
    <section className="panel">
      <h2>The law bench <span className="sub">C = Σ aᵢφᵢ</span></h2>
      <div className="sliders">
        {weights.map((w, i) => (
          <label key={i} className="slider">
            <span className="ix">a{i + 1}</span>
            <input type="range" min={-1} max={1} step={0.005} value={w}
                   onChange={(e) => {
                     const next = weights.slice();
                     next[i] = Number(e.target.value);
                     onWeights(next);
                   }} />
            <span className="val">{w.toFixed(3)}</span>
          </label>
        ))}
      </div>
      <div className="btns">
        <button onClick={onReset}>fitted</button>
        <button onClick={onRandom}>random draw</button>
      </div>
      <div className="scores">
        <div><span>|ρ|<sub>E</sub></span><b>{law ? law.rho_energy.toFixed(4) : "—"}</b></div>
        <div><span>drift</span><b>{law ? law.drift_of_C.toExponential(1) : "—"}</b></div>
        <div><span>pairing</span><b>{law ? law.pairing_residual.toFixed(4) : "—"}</b></div>
      </div>

      <div className="plottitle">
        dose response over the fixed alpha grid {busy && <em>· running</em>}
      </div>
      {dose && (() => {
        const norm = dose.mse.map((v) => v / dose.mse[0]);
        // The committed log keys alphas as the strings Python wrote ("0.0", "0.05", ...), and
        // String(0.0) in JS is "0" — so a naive lookup misses exactly the baseline and poisons the
        // whole normalised series with NaN. Match numerically instead.
        const pubAt = (a: number) => {
          const hit = Object.entries(pubDose ?? {}).find(([k]) => Number(k) === a);
          return hit ? hit[1] : undefined;
        };
        const base = pubAt(0);
        const pub = pubDose && base
          ? dose.alphas.map((a) => { const v = pubAt(a); return v === undefined ? NaN : v / base; })
          : null;
        const all = [...norm, ...(pub ? pub.filter(Number.isFinite) : [])];
        const [lo, hi] = extent(all, 0.12);
        const sx = scale(0, dose.alphas.length - 1, 44, W - 10);
        const sy = scale(lo, hi, H - 26, 10);
        const xs = dose.alphas.map((_, i) => i);
        return (
          <>
            <Frame w={W} h={H} xLabel="α" yLabel="rel. error"
                   xTicks={dose.alphas.map((a, i) => ({ at: sx(i), label: String(a) }))}
                   yTicks={ticksFor(lo, hi, sy, 3).ticks}>
              <line x1={44} y1={sy(1)} x2={W - 10} y2={sy(1)} className="level" />
              {pub && <Line xs={xs} ys={pub} sx={sx} sy={sy} cls="ln-published" width={1.2} />}
              <Line xs={xs} ys={norm} sx={sx} sy={sy} cls="ln-corrected" />
              <Dots xs={xs} ys={norm} sx={sx} sy={sy} cls="pt-corrected" r={2.6} />
            </Frame>
            <div className="stat">
              slope <b>{dose.normalised_slope.toFixed(4)}</b> · at α max{" "}
              <b className={dose.relative_change_at_max_alpha < 0 ? "good" : "bad"}>
                {(dose.relative_change_at_max_alpha * 100).toFixed(2)}%
              </b>
              {published?.relative_change_at_max_alpha !== undefined && (
                <span className="hint">
                  published {((published.relative_change_at_max_alpha as number) * 100).toFixed(2)}%
                </span>
              )}
            </div>
            {paper && (
              <div className="null">
                published random-constraint null over {paper.random_null.n} draws: median{" "}
                <b>{(paper.random_null.median * 100).toFixed(1)}%</b>,{" "}
                {paper.random_null.improving} of {paper.random_null.n} lower the error
              </div>
            )}
          </>
        );
      })()}
      <p className="caveat">
        Scored as the slope over the whole grid, never the best α. Differences below ~0.2% are the
        size of the arithmetic — see the note in <code>rollout.py</code>.
      </p>
    </section>
  );
}

/** Statistical prominence against causal leverage, one dot per extracted direction. */
export function Directions({ lev, published }:
  { lev: Leverage | null; published?: { rho_V_D?: number; rho_D_edit?: number } }) {
  const W = 700, H = 260;
  return (
    <section className="panel">
      <h2>Directions that matter</h2>
      {!lev && <div className="placeholder">measuring…</div>}
      {lev && (() => {
        const [vx, vxh] = extent(lev.variance), [dy, dyh] = extent(lev.damage);
        const sx = scale(vx, vxh, 56, W - 14), sy = scale(dy, dyh, H - 32, 12);
        const mv = Math.max(...lev.edit_move);
        return (
          <>
            <Frame w={W} h={H} pad={[12, 14, 32, 56]} xLabel="variance V(u)" yLabel="damage D(u)"
                   xTicks={ticksFor(vx, vxh, sx, 3).ticks}
                   yTicks={ticksFor(dy, dyh, sy, 3).ticks}>
              <line x1={56} y1={sy(0)} x2={W - 14} y2={sy(0)} className="level" />
              {lev.variance.map((v, i) => (
                <circle key={i} cx={sx(v)} cy={sy(lev.damage[i])}
                        r={3 + 7 * (lev.edit_move[i] / mv)} className="pt-dir">
                  <title>{`direction ${i}\nV ${fmt(v)}\nD ${fmt(lev.damage[i])}\nedit ${fmt(lev.edit_move[i])}`}</title>
                </circle>
              ))}
            </Frame>
            <div className="stat">
              ρ(V, D) <b className={lev.rho_V_D < 0 ? "good" : "bad"}>{lev.rho_V_D.toFixed(4)}</b>
              {published?.rho_V_D !== undefined &&
                <span className="hint">published {published.rho_V_D.toFixed(4)}</span>}
            </div>
            <div className="stat">
              ρ(D, edit) <b>{lev.rho_D_edit.toFixed(4)}</b>
              {published?.rho_D_edit !== undefined &&
                <span className="hint">published {published.rho_D_edit.toFixed(4)}</span>}
            </div>
            <p className="caveat">
              Dot size is how hard the projection pushes that direction. A negative ρ(V, D) is the
              claim: the model gives least variance to directions whose displacement costs most.
            </p>
          </>
        );
      })()}
    </section>
  );
}
