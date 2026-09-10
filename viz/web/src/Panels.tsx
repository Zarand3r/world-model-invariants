/** The three analysis panels: the invariant, the law bench, and the directions.
 *
 * Each plot is its own component and returns `null` until its data arrives, rather than being an
 * inline function called in place — the panels read top to bottom, and a plot that needs different
 * scales does not have to fight for space inside a conditional expression.
 */
import type { BundleInfo, Dose, LawScores, Leverage, PaperRefs, Published, Rollout } from "./api";
import { Dots, Frame, Line, extent, fmt, scale, ticksFor } from "./plot";
import { Term } from "./Term";

const W = 330;

function EnergyScatter({ law }: { law: LawScores | null }) {
  if (!law) return null;
  const H = 168;
  const [ex, exh] = extent(law.scatter.E);
  const [cy, cyh] = extent(law.scatter.C);
  const sx = scale(ex, exh, 44, W - 10);
  const sy = scale(cy, cyh, H - 26, 10);
  return (
    <Frame w={W} h={H} xLabel="real energy" yLabel="C"
           xTicks={ticksFor(ex, exh, sx, 3).ticks} yTicks={ticksFor(cy, cyh, sy, 3).ticks}>
      <Dots xs={law.scatter.E} ys={law.scatter.C} sx={sx} sy={sy} cls="pt-energy" r={1.3} />
    </Frame>
  );
}

function DriftTrace({ roll }: { roll: Rollout | null }) {
  if (!roll) return null;
  const H = 168;
  const n = roll.C.free.length;
  const [lo, hi] = extent([...roll.C.free, ...roll.C.corrected, roll.C0]);
  const sx = scale(0, n - 1, 44, W - 10);
  const sy = scale(lo, hi, H - 26, 10);
  const xs = roll.C.free.map((_, i) => i);
  return (
    <Frame w={W} h={H} xLabel="imagined step" yLabel="C"
           xTicks={ticksFor(0, n - 1, sx, 3).ticks.map((t) => ({ ...t, label: t.label.split(".")[0] }))}
           yTicks={ticksFor(lo, hi, sy, 3).ticks}>
      <line x1={44} y1={sy(roll.C0)} x2={W - 10} y2={sy(roll.C0)} className="level" />
      <text x={W - 12} y={sy(roll.C0) - 4} className="tick" textAnchor="end">C₀</text>
      <Line xs={xs} ys={roll.C.free} sx={sx} sy={sy} cls="ln-free" width={2.4} />
      <Line xs={xs} ys={roll.C.corrected} sx={sx} sy={sy} cls="ln-corrected" width={1.4} />
    </Frame>
  );
}

/** C against true energy, and C(t) drifting away from its own starting level set. */
export function InvariantPanel({ law, roll }: { law: LawScores | null; roll: Rollout | null }) {
  return (
    <section className="panel">
      <h2>What the model conserves</h2>
      <p className="blurb">
        The search read only the model's internal state and its own one-step dynamics — never
        position, velocity or energy. It came back with{" "}
        <Term id="C">a single number, <b>C</b></Term>, that the model holds almost constant.
        Below: how well it lines up with the pendulum's real energy, and whether it stays put once
        the model starts imagining.
      </p>
      <div className="pair">
        <div>
          <div className="plottitle">C against the pendulum's real energy</div>
          <EnergyScatter law={law} />
          <div className="stat">
            <Term id="rho_E">agreement with true energy</Term>{" "}
            <b>{law ? law.rho_energy.toFixed(4) : "—"}</b>
            <span className="hint">1.0 = perfect · the search never saw energy</span>
          </div>
        </div>
        <div>
          <div className="plottitle">C while the model imagines forward</div>
          <DriftTrace roll={roll} />
          <div className="stat">
            <Term id="drift">how much it wanders</Term>{" "}
            <b>{law ? law.drift_of_C.toExponential(2) : "—"}</b>
            <span className="hint">0 = perfectly held · dashed line is its starting value</span>
          </div>
        </div>
      </div>
    </section>
  );
}

/** The committed log keys alphas as Python wrote them ("0.0", "0.05"); `String(0.0)` in JS is "0",
 *  so a string lookup misses exactly the baseline and poisons the normalised series with NaN. */
function publishedDose(dose: Dose, pub?: Record<string, number>): number[] | null {
  if (!pub) return null;
  const at = (a: number) => Object.entries(pub).find(([k]) => Number(k) === a)?.[1];
  const base = at(0);
  if (!base) return null;
  return dose.alphas.map((a) => { const v = at(a); return v === undefined ? NaN : v / base; });
}

function DoseCurve({ dose, published }: { dose: Dose | null; published?: Published }) {
  if (!dose) return null;
  const H = 150;
  const norm = dose.mse.map((v) => v / dose.mse[0]);
  const pub = publishedDose(dose, published?.dose);
  const [lo, hi] = extent([...norm, ...(pub?.filter(Number.isFinite) ?? [])], 0.12);
  const sx = scale(0, dose.alphas.length - 1, 44, W - 10);
  const sy = scale(lo, hi, H - 26, 10);
  const xs = dose.alphas.map((_, i) => i);
  return (
    <Frame w={W} h={H} xLabel="correction strength" yLabel="error vs none"
           xTicks={dose.alphas.map((a, i) => ({ at: sx(i), label: String(a) }))}
           yTicks={ticksFor(lo, hi, sy, 3).ticks}>
      <line x1={44} y1={sy(1)} x2={W - 10} y2={sy(1)} className="level" />
      {pub && <Line xs={xs} ys={pub} sx={sx} sy={sy} cls="ln-published" width={1.2} />}
      <Line xs={xs} ys={norm} sx={sx} sy={sy} cls="ln-corrected" />
      <Dots xs={xs} ys={norm} sx={sx} sy={sy} cls="pt-corrected" r={2.6} />
    </Frame>
  );
}

/** Eight mixing sliders over the fitted basis, plus the dose response they produce. */
export function LawBench({ info, law, weights, onWeights, onReset, onRandom, dose, paper, busy }: {
  info: BundleInfo | null; law: LawScores | null; weights: number[];
  onWeights: (w: number[]) => void; onReset: () => void; onRandom: () => void;
  dose: Dose | null; paper: PaperRefs | null; busy: boolean;
}) {
  const published = info && paper ? paper.per_model[info.model_key] : undefined;
  return (
    <section className="panel">
      <h2>Propose your own <span className="sub">C = Σ aᵢφᵢ</span></h2>
      <p className="blurb">
        The search returns eight candidate quantities and a best mix of them.{" "}
        <Term id="weights">These eight dials are that mix</Term> — move one and you are proposing a
        different conserved quantity. The three scores judge it instantly.
      </p>
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
        <div>
          <span><Term id="rho_E">matches energy</Term></span>
          <b>{law ? law.rho_energy.toFixed(4) : "—"}</b><em>higher better</em>
        </div>
        <div>
          <span><Term id="drift">wanders</Term></span>
          <b>{law ? law.drift_of_C.toExponential(1) : "—"}</b><em>lower better</em>
        </div>
        <div>
          <span><Term id="pairing">drives the dynamics</Term></span>
          <b>{law ? law.pairing_residual.toFixed(4) : "—"}</b><em>lower better</em>
        </div>
      </div>

      <div className="plottitle">
        does correcting actually help? {busy && <em>· running</em>}
      </div>
      <p className="blurb small">
        Prediction error against{" "}
        <Term id="alpha">how hard the correction pushes</Term>, over all 52 held-out episodes.
        Below 1.0 means correcting beat leaving the model alone. The grey dashed line is the
        published result for this model.
      </p>
      <DoseCurve dose={dose} published={published} />
      {dose && (
        <div className="stat">
          <Term id="slope">overall effect</Term> <b>{dose.normalised_slope.toFixed(4)}</b>
          {" "}· at the strongest correction{" "}
          <b className={dose.relative_change_at_max_alpha < 0 ? "good" : "bad"}>
            {(dose.relative_change_at_max_alpha * 100).toFixed(2)}%
          </b>
          {published?.relative_change_at_max_alpha !== undefined && (
            <span className="hint">
              published {(published.relative_change_at_max_alpha * 100).toFixed(2)}%
            </span>
          )}
        </div>
      )}
      {dose && paper && (
        <div className="null">
          published random-constraint null over {paper.random_null.n} draws: median{" "}
          <b>{(paper.random_null.median * 100).toFixed(1)}%</b>,{" "}
          {paper.random_null.improving} of {paper.random_null.n} lower the error
        </div>
      )}
      <p className="caveat">
        Scored across every correction strength, never the single best one — otherwise any method
        gets five chances to look good. Changes smaller than about 0.2% are below the noise floor of
        the arithmetic itself, so treat them as no change.
      </p>
    </section>
  );
}

function LeverageScatter({ lev }: { lev: Leverage }) {
  const w = 700, h = 260;
  const [vx, vxh] = extent(lev.variance);
  const [dy, dyh] = extent(lev.damage);
  const sx = scale(vx, vxh, 56, w - 14);
  const sy = scale(dy, dyh, h - 32, 12);
  const mv = Math.max(...lev.edit_move);
  return (
    <Frame w={w} h={h} pad={[12, 14, 32, 56]} xLabel="how much the model uses it" yLabel="how much nudging it hurts"
           xTicks={ticksFor(vx, vxh, sx, 3).ticks} yTicks={ticksFor(dy, dyh, sy, 3).ticks}>
      <line x1={56} y1={sy(0)} x2={w - 14} y2={sy(0)} className="level" />
      {lev.variance.map((v, i) => (
        <circle key={i} cx={sx(v)} cy={sy(lev.damage[i])}
                r={3 + 7 * (lev.edit_move[i] / mv)} className="pt-dir">
          <title>{`direction ${i}\nV ${fmt(v)}\nD ${fmt(lev.damage[i])}\nedit ${fmt(lev.edit_move[i])}`}</title>
        </circle>
      ))}
    </Frame>
  );
}

/** Statistical prominence against causal leverage, one dot per extracted direction. */
export function Directions({ lev, published }: { lev: Leverage | null; published?: Published }) {
  if (!lev) return <section className="panel"><h2>Directions that matter</h2>
    <div className="placeholder">measuring…</div></section>;
  return (
    <section className="panel">
      <h2>Which directions matter</h2>
      <p className="blurb">
        Every direction inside the model's state, plotted by{" "}
        <Term id="variance">how much the model uses it</Term> against{" "}
        <Term id="damage">how much damage nudging it does</Term>. If the directions that matter
        most are the ones the model varies least, then a standard look at the latent would miss
        exactly the state that counts. Dot size is how hard the correction pushes that direction.
      </p>
      <LeverageScatter lev={lev} />
      <div className="stat">
        used-vs-matters correlation{" "}
        <b className={lev.rho_V_D < 0 ? "good" : "bad"}>{lev.rho_V_D.toFixed(4)}</b>
        <span className="hint">negative = the model under-uses the directions that matter</span>
        {published?.rho_V_D !== undefined &&
          <span className="hint">published {published.rho_V_D.toFixed(4)}</span>}
      </div>
      <div className="stat">
        matters-vs-corrected correlation <b>{lev.rho_D_edit.toFixed(4)}</b>
        <span className="hint">positive = the correction pushes where it counts</span>
        {published?.rho_D_edit !== undefined &&
          <span className="hint">published {published.rho_D_edit.toFixed(4)}</span>}
      </div>

    </section>
  );
}
