# Locked Final Two-Week Plan

## Planning Window

This plan covers **May 18, 2026 through May 31, 2026** and assumes the final presentation/submission lands at the end of that two-week window.

## Locked Priorities

1. Reproduce the controlled baseline and primary best result.
2. Finalize the evidence package around the Week 4 confound and Week 5 controlled fix.
3. Turn the locked story into a clear final presentation.

## Schedule

| Date | Task | Output |
|---|---|---|
| May 18, 2026 | Restore a runnable Python environment from `requirements.txt` and confirm `pandas`/`scikit-learn` imports work. | Working local environment |
| May 19, 2026 | Re-run the controlled baseline (`Run 03` equivalent) and the locked best result (`Run 06` equivalent) in a fresh Week 6 session. | Fresh `results.csv` and `autoresearch_summary.md` under a Week 6 session folder |
| May 20, 2026 | Verify row counts, RMSE, and selected features against `artifacts/week5-20260510-1/`. | Reproducibility check memo |
| May 21, 2026 | Build final comparison table and one final figure showing baseline vs locked best result. | Presentation-ready table and figure |
| May 22, 2026 | Write the methods/results narrative for the final report or slide notes. | Draft narrative text |
| May 23, 2026 | Write the "what we dropped and why" section using the controlled negative results. | Final scope-cut summary |
| May 24, 2026 | Add one limitations/interpretation section explaining missing county-level indicators, macro as background conditions, and lagged prices as a proxy for local fundamentals. | Final interpretation note or slide |
| May 25, 2026 | Update repo-facing docs so the project statement, strategy, and evidence all say the same thing. | Cleaned documentation set |
| May 26, 2026 | Draft presentation slides around the locked claim, the confound, the fix, the final comparison, and the interpretation limits. | First full slide draft |
| May 27, 2026 | Review slides for overclaiming and cut anything not backed by controlled evidence. | Narrowed slide deck |
| May 28, 2026 | Rehearse the presentation once and patch any weak explanations. | Revised talking points |
| May 29, 2026 | Run one final end-to-end repo QA pass: commands, filenames, metrics, and references. | Final QA checklist |
| May 30, 2026 | Freeze code and documentation unless a reproducibility bug is discovered. | Scope-locked final repo |
| May 31, 2026 | Final rehearsal, polish, and delivery. | Final presentation build and submission |

## Only Remaining Open Question

Can the locked Week 5 result be reproduced cleanly in the current local environment after restoring dependencies?

That is the only confirmation question still worth spending project time on.

## Explicit Non-Goals For These Two Weeks

- No new model-family search
- No new feature-family expansion
- No evaluation redesign
- No "one more big idea" experiments
