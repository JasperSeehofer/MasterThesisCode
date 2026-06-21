---
phase: 35-coordinate-bug-characterization
plan: "01"
subsystem: test-infrastructure
tags: [test-infrastructure, fixtures, coordinate-frame, astropy, galaxy-catalog]
dependency_graph:
  requires: []
  provides:
    - master_thesis_code_test/fixtures/coordinate.py
  affects:
    - master_thesis_code_test/test_coordinate_roundtrip.py (Plan 35-02, imports from here)
    - Phase 36 regression tests (imports from here per D-09)
tech_stack:
  added: []
  patterns:
    - object.__new__ shim to bypass GalaxyCatalogueHandler.__init__ (per test_catalog_only_diagnostic.py idiom)
    - NamedTuple return type for clean multi-value unpacking (EclipticCoords)
    - np.random.default_rng(seed) seed-pinned sampling (project pattern)
    - already_rotated stage-choice knob isolating rotation vs BallTree embedding bugs
key_files:
  created:
    - master_thesis_code_test/fixtures/coordinate.py
  modified: []
decisions:
  - "Use NamedTuple (EclipticCoords) for astropy result — clean unpacking in tests, self-documenting field names"
  - "already_rotated=False stores positions in degrees (equatorial) to exercise missing rotation; already_rotated=True stores radians (ecliptic) to isolate BallTree embedding bug independently"
  - "Delegate build_balltree to real GalaxyCatalogueHandler.setup_galaxy_catalog_balltree via object.__new__ shim so tests exercise the actual (buggy or fixed) production embedding without disk I/O"
  - "Preserve STELLAR_MASS_ABSOULTE_ERROR typo verbatim from InternalCatalogColumns — do not silently correct it"
metrics:
  duration_seconds: 119
  completed_date: "2026-04-21"
  tasks_completed: 1
  files_created: 1
  files_modified: 0
---

# Phase 35 Plan 01: Coordinate Fixtures Module Summary

**One-liner:** Shared fixtures module with astropy BarycentricTrueEcliptic ground-truth wrapper, stage-switchable synthetic catalog builder, and GalaxyCatalogueHandler BallTree shim for Phase 35/36 coordinate-frame tests.

## What Was Built

`master_thesis_code_test/fixtures/coordinate.py` — 207 lines, zero runtime side effects on import, no GPU dependencies.

### Exported API

| Symbol | Kind | Purpose |
|--------|------|---------|
| `EclipticCoords` | `NamedTuple` | Holds `(lambda_rad, beta_rad, theta_polar_rad)` from astropy conversion |
| `equatorial_to_ecliptic_astropy(ra_deg, dec_deg)` | function | Ground-truth ICRS→BarycentricTrueEcliptic(equinox="J2000") conversion via astropy; returns `EclipticCoords` |
| `synthetic_catalog_builder(n, sky_band, *, seed, already_rotated)` | function | Builds `pd.DataFrame` with `InternalCatalogColumns` schema; three sky bands: `"ecliptic_equator"`, `"north_pole"`, `"uniform"` |
| `build_balltree(catalog)` | function | Delegates to real `GalaxyCatalogueHandler.setup_galaxy_catalog_balltree` via `object.__new__` shim; returns `BallTree` |

### The `already_rotated` Stage-Choice Knob

Per PATTERNS.md §"Caveat — the synthetic builder must choose a stage":

- `already_rotated=False` (default): positions stored in **degrees**, equatorial (RA/Dec), matching how GLADE delivers data. Tests that inject this DataFrame must call `_map_angles_to_spherical_coordinates` — exercising the missing equatorial→ecliptic rotation bug (COORD-03).
- `already_rotated=True`: positions stored in **radians**, ecliptic (phi=lambda, theta=polar angle). Tests that inject this DataFrame bypass the rotation stage entirely and directly exercise the buggy BallTree Cartesian embedding (COORD-02: `cos(theta)` used as latitude instead of `sin(theta)` for the z-component).

This knob lets Plans 02 and 03 isolate each bug independently in separate test functions.

### Verification Results

| Check | Result |
|-------|--------|
| `equatorial_to_ecliptic_astropy(0.0, 0.0).beta_rad ≈ 0` | PASS (vernal equinox on both equators) |
| `synthetic_catalog_builder(10, 'ecliptic_equator')` returns 10-row DataFrame with RIGHT_ASCENSION + DECLINATION | PASS |
| Module imports cleanly with no side effects | PASS |
| `ruff check` | PASS (all checks passed) |
| `mypy` | PASS (no issues) |
| No `from __future__ import annotations` | PASS |
| No `import cupy` | PASS |
| `object.__new__(GalaxyCatalogueHandler)` shim pattern | PASS |

## Deviations from Plan

None — plan executed exactly as written.

The `# type: ignore[return-value]` comment on `return instance.catalog_ball_tree` was added then removed: mypy correctly inferred the `BallTree` return type without needing the ignore comment, so it was flagged as unused and removed (Rule 1 auto-fix during verification).

## Known Stubs

None. This is a pure infrastructure module — no data flows to UI rendering, no placeholder text.

## Threat Flags

No new security-relevant surface introduced. All four threats in the plan's STRIDE register were reviewed and accepted:
- T-35-01: DataFrame constructed in-memory only.
- T-35-02: IERS auto-download; astropy falls back to bundled predictive table.
- T-35-03: `object.__new__` bypass is test-only, already used at `test_catalog_only_diagnostic.py:37-73`.
- T-35-04: Import pinned to local package.

## Self-Check: PASSED

- File exists: `master_thesis_code_test/fixtures/coordinate.py` — FOUND
- Commit d9a7f86 exists in git log — FOUND
- No unexpected file deletions in commit — CONFIRMED
