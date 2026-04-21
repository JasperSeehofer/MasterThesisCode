---
phase: 35-coordinate-bug-characterization
plan: "03"
subsystem: test-infrastructure
tags: [audit, baseline, coordinate-frame, cli, gitignore-fix]
dependency_graph:
  requires: [35-01]
  provides: [audit-baseline-artifacts, audit-cli-script]
  affects: [phase-40-verify-04]
tech_stack:
  added: []
  patterns: [argparse-cli, json-md-dual-write, evaluation-report-pattern]
key_files:
  created:
    - scripts/audit_coordinate_bug.py
    - .planning/audit_coordinate_bug.md
    - .planning/audit_coordinate_bug.json
    - .planning/audit_coordinate_bug_histogram.png
  modified:
    - .gitignore
    - CHANGELOG.md
decisions:
  - "Passed full CSV path via --csv since simulations/ is gitignored and not in the worktree"
  - "Added !.planning/*.png exception to .gitignore so committed audit histogram is not silently excluded by global *.png rule"
  - "Restored .planning/debug/baseline.json and comparison_current.md that were modified as out-of-scope side effects of the pytest run"
metrics:
  duration: "4m (221s)"
  completed: "2026-04-21T21:43:59Z"
  tasks_completed: 2
  files_created: 4
  files_modified: 2
---

# Phase 35 Plan 03: Coordinate Bug Baseline Audit Summary

**One-liner:** CLI audit generator + committed pre-fix baseline showing 0/42 CRB events in ±5° ecliptic-equator band (vs 8.7% isotropic expected), JSON sidecar for Phase 40 diff.

## Observed Band Counts (Pre-Fix Baseline)

Source: `simulations/cramer_rao_bounds.csv` (42 events, `qS` column — recovered ecliptic polar angle in radians)

| Band | Count | Fraction (observed) | Fraction (isotropic prior) | Deviation |
|------|-------|---------------------|----------------------------|-----------|
| ±5°  | 0     | 0.0000              | 0.0872                     | -0.0872   |
| ±10° | 2     | 0.0476              | 0.1736                     | -0.1260   |
| ±15° | 5     | 0.1190              | 0.2588                     | -0.1398   |

**Interpretation:** Events are *underrepresented* near the ecliptic equator. The ±5° deviation is −0.0872 (all events below the equator threshold). This means the coordinate bug does not pile events *onto* the ecliptic equator — instead, the broken BallTree embedding collapses equatorial-plane galaxies to `(0,0,1)` (the polar direction), causing sky-position mismatches that could manifest anywhere in the distribution rather than a specific equatorial pile-up.

Phase 40 VERIFY-04 will re-run the CLI and diff the JSON sidecar against the post-fix result to verify the fix altered the distribution.

## CLI Invocation (verbatim for Phase 40)

```bash
uv run python scripts/audit_coordinate_bug.py \
    --csv simulations/cramer_rao_bounds.csv \
    --output-dir .planning/
```

Expected output line: `Wrote audit artifacts to .planning// for 42 events.`

## Confirmed Artifacts (git ls-files)

```
.planning/audit_coordinate_bug.json
.planning/audit_coordinate_bug.md
.planning/audit_coordinate_bug_histogram.png
```

All three artifacts are git-tracked (confirmed by `git ls-files` output above).

## Commits

| Task | Commit | Files |
|------|--------|-------|
| 1 — audit CLI module | edd6f03 | scripts/audit_coordinate_bug.py |
| 2 — baseline artifacts | d0573b1 | .planning/audit_coordinate_bug.{md,json,png}, .gitignore, CHANGELOG.md |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] .gitignore *.png rule prevented committing histogram artifact**

- **Found during:** Task 2 — `git add .planning/audit_coordinate_bug_histogram.png` failed with "ignored by .gitignore"
- **Issue:** The global `*.png` rule in `.gitignore` (line 25) would silently exclude the committed baseline histogram. D-12 requires the PNG to be git-tracked.
- **Fix:** Added `!.planning/*.png` exception line to `.gitignore` immediately after the `*.png` rule. Committed alongside the artifacts.
- **Files modified:** `.gitignore`
- **Commit:** d0573b1

**2. [Rule 1 - Bug] pytest run regenerated .planning/debug/ files as side-effect**

- **Found during:** Task 2 — after running `pytest -m "not gpu and not slow"`, `git status` showed `.planning/debug/baseline.json` and `.planning/debug/comparison_current.md` modified with new timestamps/commit hashes.
- **Issue:** The test suite imports evaluation_report.py and regenerates these debug files as a side-effect of some test fixture. These are out-of-scope changes from this plan.
- **Fix:** `git checkout -- .planning/debug/baseline.json .planning/debug/comparison_current.md` to restore originals before committing.
- **Files modified:** None (restored to HEAD state)
- **Commit:** N/A (restoration, not a fix)

## Known Stubs

None — the audit is complete with real data from the production CSV.

## Threat Flags

None — script reads a trusted local CSV and writes to `.planning/`. No new network endpoints, auth paths, or schema changes at trust boundaries.

## Self-Check: PASSED

- `scripts/audit_coordinate_bug.py` exists: CONFIRMED
- `.planning/audit_coordinate_bug.md` exists: CONFIRMED
- `.planning/audit_coordinate_bug.json` exists: CONFIRMED
- `.planning/audit_coordinate_bug_histogram.png` exists: CONFIRMED (valid PNG, 37kB)
- Commits edd6f03 and d0573b1 exist in git log: CONFIRMED
- JSON schema valid (event_count=42, all required keys present): CONFIRMED
- expected_fraction_5deg ≈ 0.0872: CONFIRMED (0.08715574274765817)
