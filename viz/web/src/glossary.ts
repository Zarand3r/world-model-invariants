/** Plain-language meaning for every symbol the bench puts on screen.
 *
 * The notation is the paper's, and it stays — a reader holding the paper should recognise what they
 * are looking at. But the bench is now reachable by people who have never read it, and a page
 * covered in `C`, `|ρ|_E`, `a_i` and `D(u)` is unreadable to them. Every symbol therefore carries
 * its name and a sentence saying what it measures, and the sentence is the thing a newcomer needs:
 * what does a big number mean, what does a small one mean.
 */
export type Entry = { name: string; what: string; good?: string };

export const GLOSSARY: Record<string, Entry> = {
  C: {
    name: "the conserved quantity",
    what: "A single number computed from the model's internal state. The search found it without " +
      "ever being shown physics: it is the scalar the model's own step-to-step dynamics leave " +
      "almost unchanged. On a pendulum it turns out to track energy.",
  },
  C0: {
    name: "its starting value",
    what: "What the conserved quantity equalled on the first real frame. The correction pulls the " +
      "model back toward this value as it imagines forward.",
  },
  rho_E: {
    name: "agreement with true energy",
    what: "How closely the recovered quantity tracks the pendulum's actual energy, which the " +
      "search never saw. 1.0 would be perfect agreement, 0 none at all.",
    good: "higher is better",
  },
  drift: {
    name: "how much it wanders",
    what: "How much the quantity moves along a single real trajectory, as a fraction of how much " +
      "it varies between trajectories. Near zero means the model really is holding it constant.",
    good: "lower is better",
  },
  pairing: {
    name: "how well it generates the dynamics",
    what: "Whether the model's latent motion looks like motion that conserves this quantity, " +
      "rather than merely correlating with it. 0 would mean the dynamics are exactly generated " +
      "by it.",
    good: "lower is better",
  },
  weights: {
    name: "the recipe for the quantity",
    what: "The search returns eight candidate conserved quantities. These eight dials mix them " +
      "into the one being tested — move them and you are proposing a different conserved " +
      "quantity, which the scores immediately judge.",
  },
  alpha: {
    name: "correction strength",
    what: "How hard the model is pushed back toward its starting value of the conserved quantity " +
      "at every imagined step. At 0 the model imagines freely; higher values correct harder.",
  },
  ld: {
    name: "search dimensions",
    what: "How many directions of the model's internal state the search looks in. The paper fixed " +
      "this at 12 in advance. Changing it means a new search, which takes about fifteen seconds.",
  },
  horizon: {
    name: "steps imagined",
    what: "How far the model predicts forward on its own, with no further frames to look at.",
  },
  trajectory: {
    name: "episode",
    what: "Which held-out pendulum swing to replay. These were never used to train the model.",
  },
  variance: {
    name: "how much the model uses this direction",
    what: "How widely the model's internal state actually varies along this direction. This is " +
      "what a standard analysis of the latent would notice first.",
  },
  damage: {
    name: "how much it matters",
    what: "How much worse the model's prediction gets if this direction is nudged off course. " +
      "High means the direction carries real consequences for what the model imagines.",
  },
  slope: {
    name: "overall effect of correcting",
    what: "How prediction error changes across the whole range of correction strengths. Negative " +
      "means correcting helps; the whole range is used so no single flattering setting can be " +
      "cherry-picked.",
    good: "negative is better",
  },
  free: {
    name: "free imagination",
    what: "The model predicting forward on its own, with no correction applied.",
  },
  corrected: {
    name: "corrected imagination",
    what: "The same prediction, with the latent state nudged back toward its starting value of " +
      "the conserved quantity at every step.",
  },
  truth: {
    name: "what actually happened",
    what: "The real held-out video the model is being scored against.",
  },
};
