# Working agreements for this repository

## Keep `docs/SUMMARY.md` current

`docs/SUMMARY.md` is the one-page executive summary: what we are trying to find out, what the plan
is, and what we have established. **Update it after every significant development**, in the same
commit as the work.

Significant means: a result landed, a claim changed status (established / open / dead), a defect was
found, the plan changed, or an experiment was withdrawn. Routine work — a passing test run, a
refactor, a doc tidy — is not significant and should not touch it.

Rules for that file:

- **Plain language.** Someone outside the project should follow it without the codenames. Say "a
  probe fitted to the pendulum's true energy", not "E18's supervised arm".
- **Extremely concise.** One paragraph of objective and context at the top, then `## Roadmap` and
  `## Results` as bullets. If it grows past roughly a page, cut rather than append.
- **Status, not history.** It says what is true now. The narrative of how we got there lives in
  `docs/EXECUTION_LOG.md`, which is append-only.
- **Move claims between the Established / Open / Dead groupings** rather than deleting them. A claim
  that turned out wrong is information; readers need to know it was tried.
- **Numbers must trace to a run record**, and stay consistent with `scripts/verify_paper_numbers.py`.

## The rest of the repository

- `docs/EXECUTION_LOG.md` — append-only, dated, negatives and retractions recorded equally.
- `docs/ROADMAP.md` — plan, plus a maintained execution-status table of outcomes.
- `docs/*_PREREG.md` — predictions and falsifiers, written before results; amendments dated and
  stating which way they move the expected outcome.
- `runs/` — raw records, immutable. Derived summaries live in `docs/`.
- Before comparing two data-generating processes, check the difference exists in what the model
  actually observes: `tests/test_observable_difference.py`. Four experiments were lost to skipping it.
- `origin` is the published repository and is guarded by `.git/hooks/pre-push`. Day-to-day work
  pushes to `extension`.
