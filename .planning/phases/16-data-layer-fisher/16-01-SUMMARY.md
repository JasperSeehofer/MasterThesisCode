---
phase: 16-data-layer-fisher
plan: 01
subsystem: plotting
tags: [numpy, pandas, covariance, fisher-matrix, crb]

# Dependency graph
requires:
  - phase: 15-style-infra
    provides: plotting module structure, _labels.py LABELS constants
provides:
  - "CRB data layer: PARAMETER_NAMES, INTRINSIC, EXTRINSIC constants"
  - "reconstruct_covariance() for 14x14 matrix from CSV rows"
  - "label_key() mapping CSV param names to LABELS keys"
  - "sample_crb_row and sample_crb_dataframe test fixtures"
affects: [16-02, 17-fisher-plots, 18-corner-plots]

# Tech tracking
tech-stack:
  added: []
  patterns: ["CRB CSV -> covariance matrix reconstruction via lower-triangle column naming"]

key-files:
  created:
    - master_thesis_code/plotting/_data.py
    - master_thesis_code_test/plotting/test_data.py
  modified:
    - master_thesis_code_test/plotting/conftest.py

key-decisions:
  - "PARAMETER_NAMES order matches ParameterSpace._parameters_to_dict() exactly (14 params)"
  - "Only two label key mappings needed: luminosity_distance->d_L, x0->Y0"

patterns-established:
  - "CRB data access: always use reconstruct_covariance(row) rather than manual column parsing"
  - "Label mapping: use label_key(param) to bridge CSV names and _labels.py LABELS keys"

requirements-completed: [FISH-01]

# Metrics
duration: 2min
completed: 2026-04-02
---

# Phase 16 Plan 01: CRB Data Layer Summary

**CRB data layer with 14x14 covariance reconstruction from CSV rows, parameter name constants, and label key mapping for Fisher-based visualizations**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-02T11:33:52Z
- **Completed:** 2026-04-02T11:36:10Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created `_data.py` module with 6 public exports: PARAMETER_NAMES, INTRINSIC, EXTRINSIC, PARAM_TO_LABEL_KEY, label_key(), reconstruct_covariance()
- Full TDD workflow: 15 unit tests written first (red), then implementation (green), all passing
- Round-trip test verifies covariance matrix reconstruction is lossless

## Task Commits

Each task was committed atomically:

1. **Task 1: Create test_data.py and conftest fixtures (TDD red)** - `303bd39` (test)
2. **Task 2: Create _data.py with constants and reconstruct_covariance (TDD green)** - `7bc329b` (feat)

_TDD workflow: test commit first, then implementation commit._

## Files Created/Modified
- `master_thesis_code/plotting/_data.py` - CRB data layer: constants, covariance reconstruction, label mapping
- `master_thesis_code_test/plotting/test_data.py` - 15 unit tests for _data.py (shape, symmetry, diagonal, roundtrip, constants, partitioning, label mapping)
- `master_thesis_code_test/plotting/conftest.py` - Added sample_crb_row and sample_crb_dataframe fixtures

## Decisions Made
- PARAMETER_NAMES order follows _parameters_to_dict() exactly: M, mu, a, p0, e0, x0, luminosity_distance, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
- Only two params need label key mapping (luminosity_distance -> d_L, x0 -> Y0); all others are identity

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully implemented with no placeholders.

## Next Phase Readiness
- _data.py ready for Plan 02 (Fisher visualization functions) to import
- reconstruct_covariance() provides the matrix data for error ellipses and uncertainty distributions
- label_key() bridges CSV parameter names to display labels

## Self-Check: PASSED

- FOUND: master_thesis_code/plotting/_data.py
- FOUND: master_thesis_code_test/plotting/test_data.py
- FOUND: commit 303bd39
- FOUND: commit 7bc329b

---
*Phase: 16-data-layer-fisher*
*Completed: 2026-04-02*
