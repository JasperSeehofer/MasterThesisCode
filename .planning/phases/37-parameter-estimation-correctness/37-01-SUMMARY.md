---
phase: 37-parameter-estimation-correctness
plan: "01"
subsystem: hygiene
tags: [hygiene, constants, snr-threshold, coordinate-frame, idempotency]
dependency_graph:
  requires: ["36-coordinate-frame-fix (36-05-SUMMARY.md committed, 9 roundtrip tests GREEN)"]
  provides: ["stable software baseline for PE-01/PE-02 physics tasks in 37-02"]
  affects: ["master_thesis_code/constants.py", "master_thesis_code/cosmological_model.py", "master_thesis_code/galaxy_catalogue/handler.py", "master_thesis_code/main.py", "master_thesis_code/plotting/paper_figures.py"]
tech_stack:
  added: []
  patterns: ["constant derivation (C/1000 instead of literal)", "idempotency guard via boolean flag", "single-source-of-truth constant import pattern"]
key_files:
  created: []
  modified:
    - master_thesis_code/constants.py
    - master_thesis_code/cosmological_model.py
    - master_thesis_code/galaxy_catalogue/handler.py
    - master_thesis_code/main.py
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code_test/test_coordinate_roundtrip.py
decisions:
  - "PE-03: Swap Omega_m limits to lower=0.04, upper=0.5 (data-entry error in LamCDMScenario)"
  - "PE-04: Add PRE_SCREEN_SNR_FACTOR=0.3 constant; SNR_THRESHOLD is now the sole source of truth for snr_threshold defaults"
  - "PE-05: SPEED_OF_LIGHT_KM_S derived as C/1000 from astropy, eliminating hardcoded 300000.0 literal (value identical to float64 precision)"
  - "COORD-05: _angles_mapped_to_ecliptic bool flag + AssertionError guard prevents silent double-call corruption in GalaxyCatalogueHandler"
  - "Rule 1 auto-fix: test_coordinate_roundtrip.py instances created via object.__new__() needed _angles_mapped_to_ecliptic=False initialisation to work with the new guard"
metrics:
  duration: "18m"
  completed: "2026-04-22T12:45:00Z"
  tasks_completed: 5
  files_modified: 6
---

# Phase 37 Plan 01: Software Hygiene — Omega_m, SNR Unification, C/1000, Idempotency Guard

Four pure-software hygiene fixes landed as a single atomic `chore(37)` commit with no physics formula changes. This is the GSD Wave 1 baseline for Phase 37 before the physics tasks (PE-01 h-threading, PE-02 per-parameter epsilon) land in Plan 37-02 under the `/physics-change` protocol.

## What Was Built

- **PE-03**: Fixed swapped `Omega_m` lower/upper limits in `LamCDMScenario` — `lower_limit=0.04, upper_limit=0.5` (was `lower=0.5, upper=0.04`)
- **PE-04**: Added `PRE_SCREEN_SNR_FACTOR: float = 0.3` to `constants.py`; replaced `snr_threshold: int = 15` in `cosmological_model.py` and `snr_threshold: float = 15.0` in `paper_figures.py` with `SNR_THRESHOLD`; replaced literal `0.3` in `main.py` pre-screen check with the named constant
- **PE-05**: `SPEED_OF_LIGHT_KM_S: float = C / 1000` — derived from astropy `C`, eliminating the `300000.0` hardcoded literal (value 299792.458 km/s, matching `C/1000` to float64 precision)
- **COORD-05**: `_angles_mapped_to_ecliptic: bool = False` field on `GalaxyCatalogueHandler.__init__`; `AssertionError` guard at top of `_map_angles_to_spherical_coordinates` prevents silent double-call corruption

## Commits

| Hash | Description |
|------|-------------|
| `35cc79d` | `chore(37): software hygiene — Omega_m limits, SNR unification, C/1000, idempotency guard` |

## Verification Results

All contract acceptance tests passed:

| Test | Result |
|------|--------|
| test-omega-limits: `lower_limit=0.04 < upper_limit=0.5` | PASS |
| test-speed-precision: `SPEED_OF_LIGHT_KM_S == C/1000` to 1e-6 | PASS (299792.458) |
| test-snr-grep: no stale literal 15 adjacent to snr tokens | PASS |
| test-double-call: second call to `_map_angles_to_spherical_coordinates` raises `AssertionError` | PASS |
| Phase 36 roundtrip tests (9/9) | PASS |
| Full test suite: `pytest -m "not gpu and not slow"` | 517 passed, 0 failures |
| mypy: `master_thesis_code/ master_thesis_code_test/` | Success: no issues |
| ruff check: all source files | Clean |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Initialize `_angles_mapped_to_ecliptic` on `object.__new__` instances in tests**
- **Found during:** Task 4 verification
- **Issue:** Four tests in `test_coordinate_roundtrip.py` construct `GalaxyCatalogueHandler` via `object.__new__(GalaxyCatalogueHandler)` to bypass catalog file loading. The new `_angles_mapped_to_ecliptic` guard requires the flag to exist before `_map_angles_to_spherical_coordinates` is called; these instances bypassed `__init__` so the attribute was missing, causing `AttributeError` on all four tests.
- **Fix:** Added `instance._angles_mapped_to_ecliptic = False` immediately after each `object.__new__` call in the test file.
- **Files modified:** `master_thesis_code_test/test_coordinate_roundtrip.py`
- **Commit:** `35cc79d` (included in the same atomic commit)

## Known Stubs

None — no stub patterns introduced in this plan.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes introduced.

## Self-Check

- [x] `master_thesis_code/constants.py` — modified, exists
- [x] `master_thesis_code/cosmological_model.py` — modified, exists
- [x] `master_thesis_code/galaxy_catalogue/handler.py` — modified, exists
- [x] `master_thesis_code/main.py` — modified, exists
- [x] `master_thesis_code/plotting/paper_figures.py` — modified, exists
- [x] `master_thesis_code_test/test_coordinate_roundtrip.py` — modified, exists
- [x] Commit `35cc79d` — exists in git log

## Self-Check: PASSED
