/** Hand-drawn SVG plot primitives.
 *
 * Every plot here is bespoke — a scatter you select in, a drift trace against a level line, a dose
 * curve with a published reference behind it — so a charting library would be configuration on top
 * of the same work. These are the four pieces they share: a scale, an axis frame, a polyline and a
 * point cloud.
 */
import type { ReactNode } from "react";

export type Scale = (v: number) => number;

export function scale(dLo: number, dHi: number, rLo: number, rHi: number): Scale {
  const span = dHi - dLo || 1;
  return (v) => rLo + ((v - dLo) / span) * (rHi - rLo);
}

export function extent(vals: number[], pad = 0.05): [number, number] {
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) { if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  if (!Number.isFinite(lo)) return [0, 1];
  const p = (hi - lo) * pad || Math.abs(hi) * 0.05 || 1;
  return [lo - p, hi + p];
}

export function fmt(v: number, digits = 3): string {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(1);
  return v.toFixed(digits).replace(/\.?0+$/, "");
}

/** Axis ticks that stay distinct.
 *
 * Rounding each tick independently produced "0.01" twice on a range of 0.0069 to 0.0104 — two
 * different gridlines carrying the same label, which is worse than no label. Ticks are formatted
 * against the SPAN rather than the values, so the number of decimals is whatever it takes to
 * separate adjacent ticks, and a common exponent is factored into the axis label when the values
 * are small.
 */
export function ticksFor(lo: number, hi: number, s: Scale, n = 4): {
  ticks: { at: number; label: string }[]; suffix: string;
} {
  const span = Math.abs(hi - lo) || Math.abs(hi) || 1;
  const mag = Math.floor(Math.log10(Math.max(Math.abs(lo), Math.abs(hi)) || 1));
  const useExp = mag <= -3 || mag >= 5;
  const scaleBy = useExp ? Math.pow(10, mag) : 1;
  // enough decimals that one step is visible in the last digit
  const step = span / n / scaleBy;
  const dec = Math.max(0, Math.min(4, Math.ceil(-Math.log10(step)) + 1));
  const out: { at: number; label: string }[] = [];
  for (let i = 0; i <= n; i++) {
    const v = lo + ((hi - lo) * i) / n;
    out.push({ at: s(v), label: (v / scaleBy).toFixed(dec) });
  }
  return { ticks: out, suffix: useExp ? ` ×10${sup(mag)}` : "" };
}

const SUP = ["\u2070", "\u00b9", "\u00b2", "\u00b3", "\u2074", "\u2075", "\u2076", "\u2077",
             "\u2078", "\u2079"];
function sup(n: number): string {
  const s = Math.abs(n).toString().split("").map((d) => SUP[Number(d)]).join("");
  return (n < 0 ? "\u207b" : "") + s;
}

type FrameProps = {
  w: number; h: number; pad?: [number, number, number, number];
  xLabel?: string; yLabel?: string;
  xTicks?: { at: number; label: string }[];
  yTicks?: { at: number; label: string }[];
  children: ReactNode;
};

/** Axis frame with ticks. `children` draw in pixel space; use the scales you built from `inner`. */
export function Frame({ w, h, pad = [10, 10, 26, 42], xLabel, yLabel, xTicks = [], yTicks = [],
                        children }: FrameProps) {
  const [pt, pr, pb, pl] = pad;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="plot" role="img"
         aria-label={`${yLabel ?? "value"} against ${xLabel ?? "index"}`}>
      <line x1={pl} y1={pt} x2={pl} y2={h - pb} className="axis" />
      <line x1={pl} y1={h - pb} x2={w - pr} y2={h - pb} className="axis" />
      {yTicks.map((t, i) => (
        <g key={i}>
          <line x1={pl - 3} y1={t.at} x2={pl} y2={t.at} className="axis" />
          <text x={pl - 6} y={t.at + 3} className="tick" textAnchor="end">{t.label}</text>
        </g>
      ))}
      {xTicks.map((t, i) => (
        <g key={i}>
          <line x1={t.at} y1={h - pb} x2={t.at} y2={h - pb + 3} className="axis" />
          <text x={t.at} y={h - pb + 14} className="tick" textAnchor="middle">{t.label}</text>
        </g>
      ))}
      {xLabel && <text x={(pl + w - pr) / 2} y={h - 2} className="axlabel" textAnchor="middle">{xLabel}</text>}
      {yLabel && <text transform={`translate(9 ${(pt + h - pb) / 2}) rotate(-90)`} className="axlabel"
                       textAnchor="middle">{yLabel}</text>}
      {children}
    </svg>
  );
}

export function Line({ xs, ys, sx, sy, cls, width = 1.6 }:
  { xs: number[]; ys: number[]; sx: Scale; sy: Scale; cls: string; width?: number }) {
  // Break the path at any non-finite point rather than emitting "NaN" into `d`, which makes the
  // browser discard the whole path and log an SVG error.
  let pen = "M";
  const d = xs.map((x, i) => {
    const y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) { pen = "M"; return ""; }
    const seg = `${pen}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
    pen = "L";
    return seg;
  }).join("");
  return <path d={d} className={cls} fill="none" strokeWidth={width} />;
}

export function Dots({ xs, ys, sx, sy, cls, r = 1.6 }:
  { xs: number[]; ys: number[]; sx: Scale; sy: Scale; cls: string; r?: number }) {
  return (
    <g className={cls}>
      {xs.map((x, i) => <circle key={i} cx={sx(x)} cy={sy(ys[i])} r={r} />)}
    </g>
  );
}
