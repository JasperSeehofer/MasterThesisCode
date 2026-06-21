---
phase: 36-coordinate-frame-fix
plan: 04
subsystem: galaxy-catalogue
tags: [balltree, spherical-embedding, coordinate-frame, numpy, sklearn]

# Dependency graph
requires:
  - phase: 36-coordinate-frame-fix
    plan: 02
    provides: "_polar_to_cartesian helper + polar-correct 3D BallTree (COORD-02)"

provides:
  - "setup_4d_galaxy_catalog_balltree rewritten with 5-D spherical sky embedding (3 sky Cartesian + z_norm + log_M_norm)"
  - "find_closest_galaxy_to_coordinates uses matching 5-D query point via _polar_to_cartesian"
  - "COORD-02b bug eliminated from simulation pipeline host-assignment path"

affects:
  - "36-05-verification (D-28 stop/rethink gate checks this)"
  - "Phase 41/42 conditional injection campaigns (protected from latent COORD-02b bug)"
  - "main.py:388 get_hosts_from_parameter_samples (consumer of 4D tree)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "5-D BallTree composed of 3-D spherical sky sub-space + 1-D z_norm + 1-D log_M_norm"
    - "Structural symmetry: both setup and query call the same _polar_to_cartesian helper (D-17 lock)"
    - "Backward-compat attribute name kept (catalog_4d_ball_tree) with misnomer documented in docstring"

key-files:
  created: []
  modified:
    - "master_thesis_code/galaxy_catalogue/handler.py — setup_4d_galaxy_catalog_balltree (now 5-D), find_closest_galaxy_to_coordinates (5-D query)"

key-decisions:
  - "Metric weights: sky chord-length [0,2] + z_norm [0,1] + log_M_norm [0,1], euclidean on R^5 (planner default per D-18 / Claude's Discretion; checkpoint Task 3 auto-resolved as planner-default)"
  - "Attribute name catalog_4d_ball_tree kept for backward compatibility; documented as historical misnomer in docstring"
  - "Structural symmetry enforced: both setup_4d_galaxy_catalog_balltree and find_closest_galaxy_to_coordinates call _polar_to_cartesian (D-17)"

patterns-established:
  - "Query-point dimensionality must match tree dimensionality: both built via _polar_to_cartesian ensures structural lock"

requirements-completed: [COORD-02b]

# Metrics
duration: 2min
completed: 2026-04-22
---

# Phase 36 Plan 04: COORD-02b Summary

**COORD-02b fixed: 4D BallTree sky sub-space now uses spherical Cartesian embedding (_polar_to_cartesian) → 5-D tree replacing the flat (phi/2pi, theta/pi) bug that collapsed equatorial points to a corner**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-04-22T11:38:24Z
- **Completed:** 2026-04-22T11:40:43Z
- **Tasks:** 3 executed (Task 3 checkpoint auto-resolved as planner-default)
- **Files modified:** 1

## Accomplishments
- Rewrote `setup_4d_galaxy_catalog_balltree` to embed sky coordinates as 3-D Cartesian unit vectors via `_polar_to_cartesian(theta, phi)`, concatenated with z_norm and log_M_norm for a 5-D BallTree — eliminating the COORD-02b flat-metric bug
- Rewrote `find_closest_galaxy_to_coordinates` to produce a matching (1, 5) query point using the same `_polar_to_cartesian` helper, preserving structural symmetry (D-17)
- Smoke test confirmed: tree is 5-D, sky columns are unit vectors (||v||=1 to 1e-12), z_norm and log_M_norm are in [0,1], and self-lookup returns the correct galaxy
- All 517 non-GPU/non-slow tests pass; 9 coordinate roundtrip tests pass (0 xfail, 0 XPASS)

## Task Commits

1. **Task 1+2: Rewrite setup_4d + find_closest (combined into single atomic commit)** - `5b9cfbf` ([PHYSICS])
3. **Task 3: Checkpoint auto-resolved** — planner-default metric weights (no code change required)

**Plan metadata:** committed with SUMMARY.md in final docs commit

## Files Created/Modified
- `/home/jasper/Repositories/MasterThesisCode/master_thesis_code/galaxy_catalogue/handler.py` — `setup_4d_galaxy_catalog_balltree` rebuilt as 5-D tree; `find_closest_galaxy_to_coordinates` uses 5-D query; both call `_polar_to_cartesian`; docstrings reference COORD-02b, D-17, D-18

## Decisions Made
- **Metric weights (D-18, Claude's Discretion):** sky chord-length natural range [0,2] + z_norm [0,1] + log_M_norm [0,1], euclidean on R^5. Sky axes slightly over-weighted vs legacy (which had all axes in [0,1]) — matches physical intuition that sky localization is the primary discriminator for EMRI host assignment. Task 3 checkpoint was auto-resolved with this planner-default choice.
- **Attribute name preserved:** `catalog_4d_ball_tree` kept for backward compatibility (all callers: main.py:388, scripts/quick_snr_calibration.py:53,69 unaffected). The "4d" is a historical misnomer; documented in docstring.

## Deviations from Plan

None — plan executed exactly as written. Task 3 checkpoint was pre-resolved by the orchestrator (auto_advance=true, planner-default selected). Tasks 1 and 2 were committed as a single `[PHYSICS]` commit per the locked commit message.

## Issues Encountered
- BallTree `.data` attribute returns a `_cyutility._memoryviewslice`, not a numpy array — required `np.array(obj.catalog_4d_ball_tree.data)` conversion in the smoke test verification. Not a production issue (the BallTree query path does not expose `.data` directly).

## Known Stubs
None — both methods are fully wired. The 5-D tree is built from the real catalog and queried with real coordinates.

## Threat Flags
None — no new network endpoints, auth paths, or schema changes. The 4D BallTree is an in-memory data structure used only in the simulation pipeline (not the evaluation/inference pipeline).

## Next Phase Readiness
- COORD-02b is committed and verified. All four Phase 36 REQ-IDs now have atomic `[PHYSICS]` commits: COORD-03 (36-01), COORD-02 (36-02), COORD-04 (36-03), COORD-02b (36-04).
- Proceed to Plan 36-05: final verification + D-28 stop/rethink gate + superset regression pickle.
- D-28 gate: `uv run pytest master_thesis_code_test/test_coordinate_roundtrip.py -v` must show ≥9 passed, 0 xfail, 0 XPASS — confirmed green in this plan.

---
*Phase: 36-coordinate-frame-fix*
*Completed: 2026-04-22*
