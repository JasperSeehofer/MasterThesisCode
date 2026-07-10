---
phase: quick-260710-pp-coverage-deepvenue-mode
plan: 01
subsystem: testing
tags: [pp_coverage, dark-siren, h0-estimator, zero-host-completion, issue-29, calibration-harness]

# Dependency graph
requires:
  - phase: physics/zero-host-completion-fallback (commits ed46390, 8db6c6e, f29a5e7)
    provides: production pure-completion B_num/D zero-host fallback estimator (issue #29) and the Z_MAX_POP cap (issue #30) this harness mode is the synthetic-universe analog of
provides:
  - "master_thesis_code/validation/pp_coverage.py z_support catalogue-support-truncated mode (config field + CLI flag + B_num/D completion branch + completion_fraction reporting)"
  - "results/pp_coverage_deepvenue_20260710/RUNBOOK.md orchestrator sweep spec (8-cell grid + anchor bit-identity re-run + SUMMARY verdict format)"
affects: [bias-investigation-20260710, campaign-phase2-execution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pin-test-first for harness behavior changes: golden pin commit (bit-identical default path) BEFORE the feature commit, mirroring ed46390 -> 8db6c6e"
    - "mypy Optional-narrowing via inline `if x is not None and ...:` + local rebinding, not a precomputed bool flag, when the whole-tree mypy pre-commit hook must pass on new Optional-typed branches"

key-files:
  created:
    - results/pp_coverage_deepvenue_20260710/RUNBOOK.md
  modified:
    - master_thesis_code/validation/pp_coverage.py
    - master_thesis_code_test/validation/test_pp_coverage.py

key-decisions:
  - "Squashed the TDD RED/GREEN split into Task 2's single feat commit (tests b/c/d + implementation together) instead of a separate failing-test commit, because the RED-phase tests reference PPCoverageConfig(z_support=...) which does not exist yet and would fail the repo's whole-tree mypy pre-commit hook on a standalone RED commit; Task 1's golden-pin commit remains the plan's mandated standalone pin-first commit."
  - "Picked z_support=0.35/0.2 (not the plan's illustrative 0.5/0.2) for the monotonicity test after measuring that the tiny 30-event config produces completion_fraction=0.0 at z_support=0.5 (no host redshifts that deep in only 30 draws) -- kept the same TINY_DEEPVENUE seed/config, just adjusted the two z_support probe points so both land strictly in (0,1)."

requirements-completed: [L-A]

# Metrics
duration: 9min
completed: 2026-07-10
---

# Quick Task 260710-sjm: pp_coverage deep-venue (`z_support`) mode Summary

**Added a catalogue-support-truncated mode to the independent pp_coverage P-P/calibration harness that routes deep-catalogue true hosts into the issue-#29 pure-completion B_num/D likelihood, plus the orchestrator's ready-to-run 8-cell sweep RUNBOOK.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-07-10T18:45:45Z
- **Completed:** 2026-07-10T18:53:39Z
- **Tasks:** 3
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments
- `PPCoverageConfig.z_support: float | None = None` + `--z-support` CLI flag: true hosts with `z_host >= z_support` become zero-host events using the pure-completion likelihood `B_num(h)/D(h)` — the exact `L_cat -> 0` limit of the Gray et al. (2020) mixture that production commit `8db6c6e` (issue #29) installed, integral capped at `Z_MAX_POP` (issue #30 parallel), sharing `D(h)`'s exact unnormalized measure.
- `completion_fraction` reported per truth (mean fraction of zero-host events per realization); verified 0 at `z_support=None`/`>=Z_MAX_POP`, strictly increasing in `(0,1)` as `z_support` decreases, and `~1` at `z_support≈0.05` with a finite/normalizable posterior (no NaN).
- Golden pin (Task 1, own commit) proves the default `z_support=None` path stayed bit-identical through the Task 2 change.
- `results/pp_coverage_deepvenue_20260710/RUNBOOK.md`: the orchestrator's unambiguous 8-cell sweep (`z_support` in `{0.2,0.3,0.5,1.0}` x `sigma_z` in `{0.015,0.035}`) + anchor bit-identity re-run instructions + the SUMMARY.md verdict format (table columns, control comparison, `+/-2*SE~=0.085` coverage-collapse / `2*SEM` bias-flag criteria, 3 carried caveats).

## Task Commits

Each task was committed atomically:

1. **Task 1: Pin the z_support=None behaviour (pin-first commit)** - `a9733bb` (test)
2. **Task 2: Add the z_support truncated mode (B_num/D completion branch) + tests** - `e0eddd3` (feat)
3. **Task 3: Write the orchestrator sweep RUNBOOK + SUMMARY verdict format** - `a8100f1` (docs)

_Note: Task 2 is a TDD-flagged task; per the "Key Decisions" above, its RED-phase tests and GREEN-phase implementation were verified separately (RED confirmed failing via `TypeError: unexpected keyword argument 'z_support'` before any source edit) but committed together in one `feat` commit — see rationale above._

## Files Created/Modified
- `master_thesis_code/validation/pp_coverage.py` - `z_support` config field, `--z-support` CLI flag, `_run_realization` membership split + `B_num(h)/D(h)` completion branch (return type now `tuple[NDArray, int]`), `run_coverage` `completion_fraction` aggregation, module/function docstring citations (Gray et al. 2020 Eqs. 29+32; Gray/Messenger/Veitch 2022 Eq. 5; G2a derivation doc; issues #29/#30)
- `master_thesis_code_test/validation/test_pp_coverage.py` - golden pin (`test_z_support_none_golden_pin`), limiting-case (`z_support=Z_MAX_POP`), small-`z_support` finite/normalizable-posterior test, and monotonic-completion-fraction test
- `results/pp_coverage_deepvenue_20260710/RUNBOOK.md` (new) - orchestrator sweep spec

## Decisions Made
- Squashed Task 2's TDD RED/GREEN split into a single `feat` commit — see `key-decisions` above (mypy whole-tree pre-commit hook would reject a standalone RED commit referencing the not-yet-existing `z_support` field). RED-phase failure was still verified (and is documented) before writing any implementation code, satisfying the spirit of the pin-first/TDD convention without violating the repo's commit-time quality gate.
- Adjusted the monotonicity test's two `z_support` probe points from the plan's illustrative `{0.5, 0.2}` to `{0.35, 0.2}` after measuring `completion_fraction=0.0` at `z_support=0.5` for the tiny 30-event test config (the plan's own numbers were illustrative, not measured-exact for this config size).
- Applied both plan-checker notes verbatim: (1) bound a local `zs: float = config.z_support` immediately inside an inline `if config.z_support is not None and ...:` (not via a precomputed `is_zero_host` bool) so mypy narrows correctly; (2) used `0.085` (not `0.086`) for the `+/-2*SE` coverage-collapse threshold in the RUNBOOK.

## Deviations from Plan

None beyond the two items already documented above under "Decisions Made" (both are Rule-3-class blocking-issue accommodations — the mypy pre-commit gate — resolved by adjusting commit granularity and a test-parameter choice, not by changing the estimator design).

## Issues Encountered
- Running `pytest -k "golden_pin"` (a filtered subset) trips the repo-wide `fail-under=25%` coverage gate (expected — coverage is computed over the whole `master_thesis_code` package, not the filtered test count). Confirmed this is a pre-existing artifact of running module-scoped subsets, not a regression, by also running the full `master_thesis_code_test/validation/` suite and the whole-repo `-m "not gpu and not slow"` suite (829 passed, 15 skipped, 25 deselected, no failures).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness
- `pp_coverage.py`'s `z_support` mode is ready for the orchestrator to run the 8-cell sweep per `results/pp_coverage_deepvenue_20260710/RUNBOOK.md` (handoff item L-A). No sweep was executed by this task per the plan's constraints.
- The RUNBOOK's SUMMARY.md verdict format gives the orchestrator a ready template to fill in post-sweep, including the carried caveats (1D-only; single-host clean-limit vs production's `B_num` admixture on host-found events; hard vs soft/M_BH-prune truncation) that must be stated verbatim in that later SUMMARY.
- No blockers. This task did not touch production code (`bayesian_statistics.py` or any `/physics-change`-gated file) — it is entirely within the deliberately independent `pp_coverage.py` harness.

---
*Phase: quick-260710-pp-coverage-deepvenue-mode*
*Completed: 2026-07-10*

## Self-Check: PASSED

- FOUND: master_thesis_code/validation/pp_coverage.py
- FOUND: master_thesis_code_test/validation/test_pp_coverage.py
- FOUND: results/pp_coverage_deepvenue_20260710/RUNBOOK.md
- FOUND: .planning/quick/260710-sjm-pp-coverage-deepvenue-mode/260710-sjm-SUMMARY.md
- FOUND commit: a9733bb (test: pin z_support=None golden behaviour)
- FOUND commit: e0eddd3 (feat: add z_support catalogue-support-truncated mode)
- FOUND commit: a8100f1 (docs: author the pp_coverage deep-venue sweep RUNBOOK)
