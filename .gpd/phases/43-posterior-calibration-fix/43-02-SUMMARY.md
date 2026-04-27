---
phase: 43-posterior-calibration-fix
plan: 02
status: completed
completed_tasks: 3/3
date: 2026-04-27
branch_applied: "BRANCH-B"
h2_fix_applied: true
h1_fix_option: "B"
regression_tests_passed: 540
commit_sha: "a2df67b [PHYSICS] document extract_baseline D(h) absence; apply H2 CRB ecliptic migration"
files_modified:
  - "simulations/prepared_cramer_rao_bounds.csv (gitignored; ecliptic migration applied locally)"
  - "simulations/cramer_rao_bounds.csv (gitignored; ecliptic migration applied locally)"
  - "master_thesis_code/bayesian_inference/evaluation_report.py"
---

<!-- ASSERT_CONVENTION: natural_units=SI, coordinate_system=spherical -->
<!-- Custom: sky_angles=ecliptic (v2.2 post-fix), h0_units=dimensionless (H0/100), posterior_normalization=Gray2020_Eq_A19 -->

# Plan 43-02 SUMMARY: Apply H1+H2 Fixes

## One-liner

BRANCH-B applied: H2 CRB ecliptic migration (equatorial→BarycentricTrueEcliptic J2000) run on both CRB CSVs locally; H1 extract_baseline deprecation warning added with Gray et al. (2020) Eq. A.19 citation; 540 tests pass; [PHYSICS] commit a2df67b.

## Task 1: Physics Change Protocol Gate

Physics Change Protocol presented and approved before any code or data change.

**H2 Fix presented:**
- OLD: qS/phiS in equatorial ICRS (qS_equatorial = π/2 − Dec, phiS = RA_rad)
- NEW: qS/phiS in ecliptic BarycentricTrueEcliptic J2000 (qS = π/2 − ecl.lat.rad, phiS = ecl.lon.rad)
- Reference: handler.py:574–648 (identical transform, Phase 36 COORD-03)
- Dimensional: radians → radians, [0,π] → [0,π], [0,2π] → [0,2π]
- Limiting case: ecliptic poles at ±90° map to qS=0 or π; max shift = obliquity ~23.44°

**H1 Fix presented:**
- D(h) confirmed NOT persisted to disk → Option B (deprecation warning)
- Gray et al. (2020) arXiv:1908.06050 Eq. A.19 cited
- No formula change in production code path (--evaluate already correct)

**Approval:** Received from user before Task 2.

## Task 2: Apply Fixes

### H2: CRB Frame Migration (BRANCH-B)

Applied to both files:

| File | Rows | qS range (rad) | phiS range (rad) | _coord_frame |
|---|---|---|---|---|
| prepared_cramer_rao_bounds.csv | 542 | [0.1149, 3.073] | [0.0131, 6.248] | ecliptic_BarycentricTrue_J2000 |
| cramer_rao_bounds.csv | 42 | [0.335, 2.995] | [0.169, 6.269] | ecliptic_BarycentricTrue_J2000 |

Backups created as `.bak_equatorial` (gitignored).
`undetected_events.csv` not present — skipped.

**Note on gitignore:** `simulations/` is gitignored by project convention (regenerable runtime data). Migration applied locally. The ecliptic CRBs are now on disk and will be used by the Plan 43-03 `--evaluate` run.

**AT-fix-01 (PASS):** `_coord_frame = 'ecliptic_BarycentricTrue_J2000'`; qS in [0, π] ✓; phiS in [0, 2π] ✓

**AT-fix-02 (PASS):** handler.py uses `BarycentricTrueEcliptic(equinox="J2000")` — identical to migration script ✓

### H1: extract_baseline Deprecation (Option B)

Added to `master_thesis_code/bayesian_inference/evaluation_report.py`:
1. Docstring `.. warning::` block citing Gray et al. (2020) Eq. A.19, marking function as diagnostic-only
2. `_LOGGER.warning(...)` at function entry with Eq. A.19 reference

Production path `BayesianStatistics.evaluate()` unchanged — already applies D(h) via `precompute_completion_denominator`.

## Task 3: Regression Tests + Commit

```
540 passed, 6 skipped, 16 deselected in 21.89s  (exit code 0)
```

**AT-03 (PASS):** 540 tests pass, 0 failures ✓ (project baseline = 540)

**[PHYSICS] commit:** `a2df67b` — `master_thesis_code/bayesian_inference/evaluation_report.py` (1 file, 21 insertions)

CRB CSVs not committed (gitignored by design).

## Conventions Used

| Convention | Value |
|---|---|
| Sky frame (CRBs after fix) | Ecliptic BarycentricTrueEcliptic J2000 |
| H0 units | dimensionless = H0/(100 km/s/Mpc) |
| Posterior normalization | Gray et al. (2020) Eq. A.19 (applied in --evaluate, not extract_baseline) |

## Deviations

- **CRBs not committed:** `simulations/` is gitignored; migration applied locally only. This is correct project convention — simulation data is not tracked in git.
- **H1 Option B (not A):** D(h) is not persisted to disk, so full Option A (adding D(h) correction to extract_baseline) was not feasible without architectural changes. Option B (deprecation warning) is the minimal safe fix; production path is already correct.

## Key Results

- [CONFIDENCE: HIGH] H2 CRB migration applied: 542 events now in ecliptic BarycentricTrueEcliptic J2000
- [CONFIDENCE: HIGH] H1 fix applied: extract_baseline warns about missing D(h) normalization
- [CONFIDENCE: HIGH] 540 regression tests pass, 0 failures
- [CONFIDENCE: HIGH] Frame consistency verified: handler.py and migration both use BarycentricTrueEcliptic(equinox='J2000')

## Contract Results

| Claim ID | Status | Evidence |
|---|---|---|
| FIX-02 | COMPLETE | CRBs migrated (AT-fix-01/02 PASS); extract_baseline documented; 540 tests pass (AT-03 PASS) |

| Deliverable ID | Status | Path |
|---|---|---|
| DEL-03-crb | produced (local) | simulations/prepared_cramer_rao_bounds.csv (gitignored) |
| DEL-03-extractbaseline | produced | master_thesis_code/bayesian_inference/evaluation_report.py (committed a2df67b) |

| Acceptance Test ID | Status | Notes |
|---|---|---|
| AT-fix-01 | PASS | qS in [0,π] ✓; phiS in [0,2π] ✓; _coord_frame column present ✓ |
| AT-fix-02 | PASS | handler.py and migration both use BarycentricTrueEcliptic(equinox='J2000') ✓ |
| AT-03 | PASS | 540 tests pass, 0 failures, exit code 0 ✓ |

| Reference | Must-surface status |
|---|---|
| REF-01: Gray et al. (2020) arXiv:1908.06050 Eq. A.19 | CITED — in docstring warning and _LOGGER.warning |
| REF-03: handler.py:574-648 | READ — ecliptic transform verified; identical frame used |
| REF-04: bayesian_statistics.py:349-358 | READ — confirmed D(h) not persisted; Option B justified |

| Forbidden Proxy | Status |
|---|---|
| FP-01: MAP from extract_baseline on biased posteriors | RESPECTED — no MAP check performed here; success confirmed by --evaluate in Plan 43-03 |
| FP-02: VERIFY-02-style comparison | RESPECTED |

## Self-Check: PASSED

- [x] Physics Change Protocol gate completed with user approval
- [x] H2 CRB migration applied and validated (AT-fix-01, AT-fix-02)
- [x] H1 extract_baseline deprecation warning added (docstring + _LOGGER.warning)
- [x] 540 regression tests pass, 0 failures (AT-03)
- [x] [PHYSICS] commit a2df67b created
- [x] Only evaluation_report.py committed (CRBs gitignored by design)
- [x] FP-01 respected: no MAP check via extract_baseline
- [x] Plan 43-03 can proceed: ecliptic CRBs on disk, code committed
