/** Three synchronised film strips, and the per-frame error underneath.
 *
 * Each track arrives as one vertical sprite sheet; a canvas draws the slice for the current frame,
 * which is why scrubbing is free and playback does not touch the network. The three tracks share a
 * single cursor with the error curve, so the frame where the corrected rollout pulls away from the
 * free one can be read off both at once.
 */
import { useEffect, useRef } from "react";
import type { Rollout } from "./api";
import { Frame, Line, extent, fmt, scale, ticksFor } from "./plot";

const RES = 64;

function Strip({ src, frame, label, cls, scaleUp = 3 }:
  { src: string; frame: number; label: string; cls: string; scaleUp?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const img = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const im = new Image();
    im.onload = () => { img.current = im; draw(); };
    im.src = src;
    return () => { img.current = null; };
  }, [src]);

  const draw = () => {
    const cv = ref.current, im = img.current;
    if (!cv || !im) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(im, 0, frame * RES, RES, RES, 0, 0, cv.width, cv.height);
  };
  useEffect(draw, [frame, src]);

  return (
    <figure className={`strip ${cls}`}>
      <canvas ref={ref} width={RES * scaleUp} height={RES * scaleUp} />
      <figcaption>{label}</figcaption>
    </figure>
  );
}

export function Theatre({ roll, frame, onFrame, alpha }:
  { roll: Rollout | null; frame: number; onFrame: (f: number) => void; alpha: number }) {
  if (!roll) return <div className="panel theatre placeholder">imagining…</div>;

  const n = roll.mse.free.length;
  const W = 560, H = 150;
  const sx = scale(0, n - 1, 52, W - 10);
  const all = [...roll.mse.free, ...roll.mse.corrected];
  // pixel MSE is non-negative; padding below zero produced a "-3.7e-4" tick on a quantity that
  // cannot be negative
  const [lo0, hi] = extent(all);
  const lo = Math.max(0, lo0);
  const sy = scale(lo, hi, H - 26, 10);
  const xs = roll.mse.free.map((_, i) => i);
  const xA = ticksFor(0, n - 1, sx, 5);
  const yA = ticksFor(lo, hi, sy, 3);

  return (
    <div className="panel theatre">
      <div className="strips">
        <Strip src={roll.truth} frame={frame} label="held-out truth" cls="truth" />
        <Strip src={roll.free} frame={frame} label="free imagination" cls="free" />
        <Strip src={roll.corrected} frame={frame} label={`corrected · α ${alpha}`} cls="corrected" />
      </div>

      <div className="plotwrap">
        <div className="plothead">
          <span className="plottitle">pixel MSE per imagined step</span>
          <span className="legend">
            <i className="sw free" /> free
            <i className="sw corrected" /> corrected
          </span>
        </div>
        <Frame w={W} h={H} pad={[10, 10, 26, 52]} xLabel="imagined step"
               yLabel={`pixel MSE${yA.suffix}`}
               xTicks={xA.ticks.map((t) => ({ ...t, label: t.label.split(".")[0] }))}
               yTicks={yA.ticks}>
          <Line xs={xs} ys={roll.mse.free} sx={sx} sy={sy} cls="ln-free" width={2.6} />
          <Line xs={xs} ys={roll.mse.corrected} sx={sx} sy={sy} cls="ln-corrected" width={1.5} />
          <line x1={sx(frame)} y1={10} x2={sx(frame)} y2={H - 26} className="cursor" />
        </Frame>
      </div>

      <input className="scrub" type="range" min={0} max={n - 1} value={frame}
             onChange={(e) => onFrame(Number(e.target.value))}
             aria-label="imagined frame" />
      <div className="readout">
        <span>frame <b>{frame}</b> of {n - 1}</span>
        <span>free <b>{fmt(roll.mse.free[frame], 5)}</b></span>
        <span>corrected <b>{fmt(roll.mse.corrected[frame], 5)}</b></span>
        <span className={roll.mse.corrected[frame] < roll.mse.free[frame] ? "good" : "bad"}>
          {((roll.mse.corrected[frame] / roll.mse.free[frame] - 1) * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
