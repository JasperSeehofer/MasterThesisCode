---
phase: 35-coordinate-bug-characterization
plan: "02"
subsystem: test-infrastructure
tags: [xfail, coordinate-frame, red-tests, balltree, astropy, phase-35]
dependency_graph:
  requires:
    - "35-01: coordinate fixtures module (fixtures/coordinate.py)"
  provides:
    - "master_thesis_code_test/test_coordinate_roundtrip.py — 6 xfail(strict) RED tests + 3 ground-truth checks"
  affects:
    - "Phase 36 success criterion 1 (flip these xfails GREEN after fix)"
    - "TODO.md: TEST-COORD item added for post-Phase-36 cleanup"
tech_stack:
  added: []
  patterns:
    - "xfail(strict=True) RED test pattern — first use in this repo (per PATTERNS.md)"
    - "object.__new__ shim for GalaxyCatalogueHandler (bypasses GLADE disk I/O)"
key_files:
  created:
    - master_thesis_code_test/test_coordinate_roundtrip.py
  modified:
    - TODO.md
decisions:
  - "D-01 (locked): shared _XFAIL_REASON constant — single grep target for post-Phase-36 cleanup"
  - "D-03 (locked): no plain-assert-on-wrong-behavior; all tests describe the correct post-fix state"
  - "D-07 (locked): N=100, seed=42 recovery gate — matches ROADMAP Phase 36 success criterion 1 verbatim"
  - "Tolerance fix: BarycentricTrueEcliptic returns epoch-specific true obliquity, not mean obliquity; vernal equinox lambda and summer solstice lambda assertions use 0.01deg tolerance (vs 0.001deg for beta), with explanatory comments"
metrics:
  duration: "~15 minutes"
  completed: "2026-04-21T21:43:47Z"
  tasks_completed: 1
  tasks_total: 1
  files_created: 1
  files_modified: 1
---

# Phase 35 Plan 02: RED Coordinate Roundtrip Tests — Summary

Six xfail(strict=True) RED tests + three ground-truth checks encoding both
coordinate-frame bugs against current buggy code, with a shared reason constant
pointing to Phase 36.

## What Was Built

`master_thesis_code_test/test_coordinate_roundtrip.py` — one new file, 346 lines.

### Test Inventory

| Class | Decision Tag | xfail | What It Pins |
|---|---|---|---|
| `TestEquatorialToEclipticGroundTruth` | D-04(c) | NOT xfail | Astropy wrapper correctness (3 tests: vernal equinox, summer solstice, ecliptic pole) |
| `TestBallTreeRecoversEclipticEquatorGalaxy` | D-04(a) | YES | BallTree `cos(theta)` embedding collapses theta=pi/2 to (0,0,1) |
| `TestBallTreeRecoversNorthCelestialPole` | D-04(b) | YES | NCP round-trip: missing rotation + BallTree embedding (two bugs simultaneously) |
| `TestSummerSolsticeRotation` | D-05 | YES | RA=90° Dec=+23.44° must land on ecliptic equator (theta_polar=pi/2) |
| `TestEclipticPoleIngestion` | D-05 | YES | RA=270° Dec=+66.56° must land at ecliptic pole (theta_polar=0) |
| `TestVernalEquinoxRoundTrip` | D-05 | YES | On-axis trap: vernal equinox (on BOTH equators) must BallTree-round-trip |
| `TestNRandomEclipticEquatorRecovery` | D-06/D-07 | YES | N=100, seed=42, >=99% recovery — exact Phase 36 SC-1 threshold |

### Which Test Is NOT xfail and Why

`TestEquatorialToEclipticGroundTruth` (three methods) tests the `equatorial_to_ecliptic_astropy`
helper from Plan 01's fixture module — not the buggy production code. The astropy wrapper is
correct by construction. These tests verify the ground truth that the other xfail tests rely on.
Per D-03: "no plain-assert-on-wrong-behavior; all tests describe the correct post-fix state."

## XFAIL Count Observed in Verify Run

```
master_thesis_code_test/test_coordinate_roundtrip.py::TestEquatorialToEclipticGroundTruth::test_vernal_equinox_lies_on_both_equators PASSED
master_thesis_code_test/test_coordinate_roundtrip.py::TestEquatorialToEclipticGroundTruth::test_summer_solstice_lies_on_ecliptic_equator PASSED
master_thesis_code_test/test_coordinate_roundtrip.py::TestEquatorialToEclipticGroundTruth::test_ecliptic_pole_has_beta_90 PASSED
master_thesis_code_test/test_coordinate_roundtrip.py::TestBallTreeRecoversEclipticEquatorGalaxy::test_ball_tree_recovers_ecliptic_equator_galaxy XFAIL
master_thesis_code_test/test_coordinate_roundtrip.py::TestBallTreeRecoversNorthCelestialPole::test_ncp_round_trip_through_ingestion_and_balltree XFAIL
master_thesis_code_test/test_coordinate_roundtrip.py::TestSummerSolsticeRotation::test_summer_solstice_maps_to_ecliptic_equator XFAIL
master_thesis_code_test/test_coordinate_roundtrip.py::TestEclipticPoleIngestion::test_ecliptic_pole_maps_to_theta_polar_zero XFAIL
master_thesis_code_test/test_coordinate_roundtrip.py::TestVernalEquinoxRoundTrip::test_vernal_equinox_galaxy_recovered_by_balltree XFAIL
master_thesis_code_test/test_coordinate_roundtrip.py::TestNRandomEclipticEquatorRecovery::test_n_random_ecliptic_equator_galaxies_recovery_rate_above_99pct XFAIL

3 passed, 6 xfailed in 0.78s
```

## Cross-Reference to Phase 36 Success Criterion 1

From `.planning/ROADMAP.md §Phase 36 SC-1` (verbatim):
> "Recovery rate >=99% over randomized offsets on the ecliptic equator band"

`TestNRandomEclipticEquatorRecovery.test_n_random_ecliptic_equator_galaxies_recovery_rate_above_99pct`
uses the identical threshold (`recovered >= 99` out of 100) with identical seed (42). When Phase 36
fixes the BallTree embedding, this test flips XPASS → CI fails → markers must be removed.
The handoff is unambiguous.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Tolerance adjustment for BarycentricTrueEcliptic nutation offset**
- **Found during:** Task 1 verification run
- **Issue:** `astropy.coordinates.BarycentricTrueEcliptic` returns the epoch-specific *true*
  ecliptic (includes nutation), not the mean ecliptic. The vernal equinox (RA=0, Dec=0)
  maps to `lambda ≈ 359.996°` (not exactly 0°), and the summer solstice lambda offset is
  `~0.004°` — both exceeding the planned 0.001° tolerance.
- **Fix:** Beta (ecliptic latitude) assertions keep 0.001° tolerance (they pass). Lambda
  assertions for the vernal equinox and summer solstice use 0.01° tolerance with inline
  comments explaining the nutation offset. This is correct behaviour of the fixture, not a
  bug in the fixture.
- **Files modified:** `master_thesis_code_test/test_coordinate_roundtrip.py`
- **Commit:** `fa9ec00`

## Known Stubs

None — this plan creates no data-wiring stubs. All tests are pure in-memory assertions.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes.
Private method access via `# noqa: SLF001` is the established test-only idiom (T-35-07
accepted in plan threat register).

## Self-Check: PASSED

- `master_thesis_code_test/test_coordinate_roundtrip.py` — FOUND
- `fa9ec00` — FOUND (git log confirms)
- `uv run pytest ... --no-cov` → 3 passed, 6 xfailed, 0 failed, 0 XPASS — CORRECT
- `uv run ruff check` → All checks passed
- `uv run mypy` → Success: no issues found
