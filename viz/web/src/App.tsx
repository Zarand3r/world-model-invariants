/** The bench. One state object; every panel reads from it and the URL mirrors it. */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { aborted, api, watchJob } from "./api";
import { useDebounced } from "./useDebounced";
import type { BundleInfo, Dose, LawScores, Leverage, Model, PaperRefs, Rollout } from "./api";
import { Theatre } from "./Theatre";
import { Directions, InvariantPanel, LawBench } from "./Panels";
import { Term } from "./Term";

const PAPER = { ld: 12, degree: 4, horizon: 50, alpha: 0.4 };

/** `max` on a number input is advisory — the browser still reports a typed 999. */
const clamp = (v: number, lo: number, hi: number) =>
  Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo;

function readUrl() {
  const q = new URLSearchParams(location.search);
  const num = (k: string, d: number) => (q.has(k) ? Number(q.get(k)) : d);
  return {
    model: q.get("model") ?? "dreamer_ref_s3",
    ld: num("ld", PAPER.ld), degree: num("degree", PAPER.degree),
    traj: num("traj", 0), horizon: num("horizon", PAPER.horizon), alpha: num("alpha", PAPER.alpha),
  };
}

export default function App() {
  const [models, setModels] = useState<Model[]>([]);
  const [paper, setPaper] = useState<PaperRefs | null>(null);
  const [cfg, setCfg] = useState(readUrl);
  const [status, setStatus] = useState<string>("");
  const [bundleKey, setBundleKey] = useState<string | null>(null);
  const [info, setInfo] = useState<BundleInfo | null>(null);
  const [weights, setWeights] = useState<number[]>([]);
  const [law, setLaw] = useState<LawScores | null>(null);
  const [roll, setRoll] = useState<Rollout | null>(null);
  const [dose, setDose] = useState<Dose | null>(null);
  const [lev, setLev] = useState<Leverage | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [doseBusy, setDoseBusy] = useState(false);
  const drawRef = useRef(0);

  // The GPU work is driven by the SETTLED values, so a drag runs one rollout instead of twenty.
  const alpha = useDebounced(cfg.alpha, 120);
  const traj = useDebounced(cfg.traj, 120);
  const horizon = useDebounced(cfg.horizon, 300);
  const settledWeights = useDebounced(weights, 120);

  useEffect(() => {
    api.models().then((d) => setModels(d.models)).catch((e) => setStatus(String(e)));
    api.paper().then(setPaper).catch(() => undefined);
  }, []);

  // keep the URL in step, so any configuration is a link
  useEffect(() => {
    const q = new URLSearchParams(Object.entries(cfg).map(([k, v]) => [k, String(v)]));
    history.replaceState(null, "", `?${q}`);
  }, [cfg]);

  // (model, ld, degree) -> a bundle, built if this is the first time anyone asked
  useEffect(() => {
    let cancelled = false;
    setInfo(null); setLaw(null); setRoll(null); setDose(null); setLev(null); setBundleKey(null);
    setStatus("checking for a cached extraction…");
    api.requestBundle(cfg.model, cfg.ld, cfg.degree)
      .then(async (r) => {
        if (!r.cached && r.job) {
          setStatus("fitting the invariant — this is the slow one, about 15 s");
          await watchJob(r.job, (m) => setStatus(m));
        }
        if (cancelled) return;
        setStatus("loading the extraction…");
        const b = await api.bundle(r.key);
        if (cancelled) return;
        setBundleKey(r.key); setInfo(b); setWeights(b.weights); setStatus("");
        api.leverage(r.key).then((l) => !cancelled && setLev(l)).catch(() => undefined);
      })
      .catch((e) => !cancelled && setStatus(String(e)));
    return () => { cancelled = true; };
  }, [cfg.model, cfg.ld, cfg.degree]);

  // weights -> scores. No GPU, ~35 ms, so this runs on the raw slider value and stays live.
  useEffect(() => {
    if (!bundleKey || !weights.length) return;
    const ctl = new AbortController();
    const id = ++drawRef.current;
    api.law(bundleKey, { weights }, ctl.signal)
      .then((s) => { if (id === drawRef.current) setLaw(s); })
      .catch((e) => { if (!aborted(e)) setStatus(String(e)); });
    return () => ctl.abort();
  }, [bundleKey, weights]);

  // the theatre: one trajectory, free vs corrected. Debounced, and the in-flight request is
  // abandoned when the inputs move again, so the server is never working on a stale frame.
  useEffect(() => {
    if (!bundleKey || !settledWeights.length) return;
    const ctl = new AbortController();
    api.rollout(bundleKey, { traj, horizon, alpha, weights: settledWeights }, ctl.signal)
      .then((r) => { setRoll(r); setFrame((f) => Math.min(f, r.mse.free.length - 1)); setStatus(""); })
      .catch((e) => { if (!aborted(e)) setStatus(String(e)); });
    return () => ctl.abort();
  }, [bundleKey, traj, horizon, alpha, settledWeights]);

  const runDose = useCallback(() => {
    if (!bundleKey) return;
    setDoseBusy(true);
    api.dose(bundleKey, { horizon, weights })
      .then(setDose)
      .catch((e) => { if (!aborted(e)) setStatus(String(e)); })
      .finally(() => setDoseBusy(false));
  }, [bundleKey, horizon, weights]);

  useEffect(() => { if (bundleKey && weights.length) runDose(); }, [bundleKey]);   // once per bundle

  useEffect(() => {
    if (!playing || !roll) return;
    const n = roll.mse.free.length;
    const t = setInterval(() => setFrame((f) => (f + 1) % n), 90);
    return () => clearInterval(t);
  }, [playing, roll]);

  const set = (patch: Partial<typeof cfg>) => setCfg((c) => ({ ...c, ...patch }));
  const grouped = useMemo(() => {
    const g: Record<string, Model[]> = {};
    for (const m of models) (g[m.arm] ??= []).push(m);
    return g;
  }, [models]);
  const published = paper && info ? paper.per_model[info.model_key] : undefined;

  return (
    <div className="app">
      <header className="rig">
        <div className="brand">
          <span className="mark">C</span>
          <div>
            <h1>Invariant Probe Bench</h1>
            <p>a video-prediction model that was never taught physics, and the conservation law
              hiding inside it</p>
          </div>
        </div>

        <div className="controls">
          <label>model
            <select value={cfg.model} onChange={(e) => set({ model: e.target.value })}>
              {Object.entries(grouped).map(([arm, ms]) => (
                <optgroup key={arm} label={arm}>
                  {ms.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
                </optgroup>
              ))}
            </select>
          </label>
          <label><Term id="ld">search dimensions</Term>
            <select value={cfg.ld} onChange={(e) => set({ ld: Number(e.target.value) })}>
              {[6, 8, 12, 16].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
          <label><Term id="trajectory">episode</Term>
            <input type="number" min={0} max={(info?.n_traj ?? 52) - 1} value={cfg.traj}
                   onChange={(e) => set({
                     traj: clamp(Number(e.target.value), 0, (info?.n_traj ?? 52) - 1) })} />
          </label>
          <label><Term id="horizon">steps imagined</Term>
            <input type="number" min={5} max={info?.max_horizon ?? 110} step={5} value={cfg.horizon}
                   onChange={(e) => set({
                     horizon: clamp(Number(e.target.value), 1, info?.max_horizon ?? 110) })} />
          </label>
          <label className="alpha">
            <Term id="alpha">correction strength</Term> {cfg.alpha.toFixed(2)}
            <input type="range" min={0} max={1} step={0.01} value={cfg.alpha}
                   onChange={(e) => set({ alpha: Number(e.target.value) })} />
          </label>
          <button onClick={() => setPlaying((p) => !p)}>{playing ? "pause" : "play"}</button>
          <button className="ghost" onClick={() => { set(PAPER); }}>paper mode</button>
        </div>
      </header>

      <p className="intro">
        This model was trained to do one thing: predict pendulum video, frame by frame. Nobody told
        it about energy. A label-free search of its internal state nonetheless finds{" "}
        <Term id="C">a quantity it holds almost constant</Term> — and when the model runs on its
        own, that quantity slips. Push it back and the predictions get better.{" "}
        <b>Try it:</b> drag <em>correction strength</em> and watch the third video and the error
        curve, then move the dials on the right to propose a different quantity and see the scores
        fall apart.
      </p>

      {status && <div className="status">{status}</div>}

      {info && (
        <div className="meta">
          <span>{info.n_traj} held-out episodes, none used for training</span>
          <span>searched in {info.retained_rank} directions of the model's state · {info.n_basis}{" "}
            candidate quantities · polynomials up to degree {info.degree}</span>
          {published?.rho_energy !== undefined &&
            <span className="pub">the paper reports {published.rho_energy.toFixed(3)} for this
              model</span>}
        </div>
      )}

      <main className="grid">
        <div className="main-col">
          <Theatre roll={roll} frame={frame} onFrame={(f) => { setPlaying(false); setFrame(f); }}
                   alpha={cfg.alpha} />
          <Directions lev={lev} published={published} />
        </div>
        <div className="side">
          <InvariantPanel law={law} roll={roll} />
          <LawBench info={info} law={law} weights={weights} onWeights={setWeights}
                    onReset={() => info && setWeights(info.weights)}
                    onRandom={() => {
                      if (!bundleKey) return;
                      api.law(bundleKey, { draw: Math.floor(Math.random() * 1000) })
                        .then((s) => setWeights(s.weights)).catch(() => undefined);
                    }}
                    dose={dose} paper={paper} busy={doseBusy} />
          <div className="dosebtn"><button onClick={runDose} disabled={doseBusy}>
            re-run dose response with these weights
          </button></div>
        </div>
      </main>

      <footer>
        Every number on this page is computed live from the frozen model, not read from a file.
        One caveat worth stating plainly: the quantity is found and judged on the same set of
        episodes, so its absolute benefit is flattered — the honest comparison is against the other
        quantities tested on exactly those episodes, which is what the scores above do. The model is
        never retrained or altered; the correction only nudges its internal state as it predicts.
      </footer>
    </div>
  );
}
