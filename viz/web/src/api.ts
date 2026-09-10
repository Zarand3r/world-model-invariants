/** Typed access to the bench API. One place that knows the wire format. */

export type Model = {
  key: string; path: string; arm: "conservative" | "damped" | "ladder";
  data: string; seed: number | null; steps: number | null; hours: number | null;
  released: boolean; label: string;
};

export type BundleInfo = {
  key: string; ckpt: string; model_key: string; ld: number; degree: number; warmup: number;
  max_horizon: number; max_alpha: number;
  eigenvalues: number[]; rho_energy: number; drift_of_C: number;
  n_traj: number; n_steps: number; retained_rank: number; n_basis: number; n_monomials: number;
  weights: number[]; pairing_residual: number; rank_and_test_residual: number;
  scatter: { C: number[]; E: number[] }; energy_range: [number, number];
};

export type LawScores = {
  rho_energy: number; drift_of_C: number; pairing_residual: number;
  weights: number[]; scatter: { C: number[]; E: number[] };
};

export type Rollout = {
  alpha: number; traj: number; horizon: number;
  truth: string; free: string; corrected: string;
  mse: { free: number[]; corrected: number[] };
  C: { free: number[]; corrected: number[] };
  C0: number;
};

export type Dose = {
  alphas: number[]; mse: number[]; normalised_slope: number;
  relative_change_at_max_alpha: number; n_traj: number;
};

export type Leverage = {
  eps: number; baseline_loss: number; horizon: number;
  variance: number[]; damage: number[]; edit_move: number[];
  rho_V_D: number; rho_D_edit: number; damage_ratio: number;
};

/** What the committed run logs say about one model. Every field is optional: a ladder checkpoint
 *  has no published counterpart, and not every arm reports every statistic. */
export type Published = {
  arm?: string;
  rho_energy?: number;
  drift_of_C?: number;
  pairing_residual?: number;
  heldout_invariance_ratio?: number;
  participation_ratio?: number;
  rho_V_D?: number;
  rho_D_edit?: number;
  dose?: Record<string, number>;
  normalised_slope?: number;
  relative_change_at_max_alpha?: number;
};

export type PaperRefs = {
  per_model: Record<string, Published | undefined>;
  untrained_rho: number[];
  random_null: { n: number; median: number; improving: number; values: number[]; slopes: number[] };
  damped_dose: number[];
  alphas: number[];
};

async function jsonOrThrow(r: Response) {
  if (!r.ok) {
    // the API answers a rejected request with {"detail": "..."}; show that, not "422 {...}"
    const body = await r.text();
    let detail = body;
    try { detail = JSON.parse(body).detail ?? body; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return r.json();
}

const post = (url: string, body: unknown, signal?: AbortSignal) =>
  fetch(url, { method: "POST", headers: { "content-type": "application/json" },
               body: JSON.stringify(body), signal }).then(jsonOrThrow);

export const aborted = (e: unknown) => e instanceof DOMException && e.name === "AbortError";

export const api = {
  models: (): Promise<{ models: Model[]; resident: string[]; cached_bundles: string[] }> =>
    fetch("/api/models").then(jsonOrThrow),
  paper: (): Promise<PaperRefs> => fetch("/api/paper").then(jsonOrThrow),
  requestBundle: (model: string, ld: number, degree: number):
    Promise<{ key: string; cached: boolean; job?: string }> =>
    post("/api/bundles", { model, ld, degree }),
  bundle: (key: string): Promise<BundleInfo> => fetch(`/api/bundles/${key}`).then(jsonOrThrow),
  law: (key: string, body: { weights?: number[]; draw?: number }, signal?: AbortSignal):
    Promise<LawScores> => post(`/api/bundles/${key}/law`, body, signal),
  rollout: (key: string, body: { traj: number; horizon: number; alpha: number; weights?: number[] },
            signal?: AbortSignal): Promise<Rollout> =>
    post(`/api/bundles/${key}/rollout`, body, signal),
  dose: (key: string, body: { horizon: number; weights?: number[] }, signal?: AbortSignal):
    Promise<Dose> => post(`/api/bundles/${key}/dose`, body, signal),
  leverage: (key: string): Promise<Leverage> =>
    fetch(`/api/bundles/${key}/leverage`).then(jsonOrThrow),
};

/** Follow a bundle-build job, calling `onMessage` for each progress line. */
export function watchJob(jid: string, onMessage: (m: string) => void): Promise<string> {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${jid}/events`);
    es.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.message) onMessage(d.message);
      if (d.state === "done") { es.close(); resolve(d.key); }
      if (d.state === "failed") { es.close(); reject(new Error(d.message ?? "fit failed")); }
    };
    es.onerror = () => { es.close(); reject(new Error("lost the job stream")); };
  });
}
