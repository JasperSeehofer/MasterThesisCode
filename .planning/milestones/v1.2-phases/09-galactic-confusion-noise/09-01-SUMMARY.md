---
phase: 09-galactic-confusion-noise
plan: 01
subsystem: physics
tags: [lisa, psd, confusion-noise, galactic-foreground, tdi]

# Dependency graph
requires: []
provides:
  - "LisaTdiConfiguration with galactic confusion noise S_c(f) in A/E-channel PSD"
  - "include_confusion_noise toggle for backward-compatible instrumental-only PSD"
  - "t_obs_years field for observation-time-dependent confusion noise level"
affects: [parameter-estimation, snr-computation, detection-yield]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conditional noise component addition via dataclass field toggle"
    - "Instance method for observation-time-dependent noise (not static)"

key-files:
  created: []
  modified:
    - master_thesis_code/LISA_configuration.py
    - master_thesis_code_test/LISA_configuration_test.py

key-decisions:
  - "T_obs power-law coefficients (a1,b1,ak,bk) expect years, not seconds -- plan erroneously used YEAR_IN_SEC conversion"
  - "Confusion noise test range limited to 0.1--3 mHz where S_c is physically relevant (underflows to 0 above ~5 mHz)"
  - "Reference comment cites Cornish & Robson (2017) arXiv:1703.09858 as primary source, not arXiv:2303.15929"

patterns-established:
  - "Dataclass field toggle for optional noise components in PSD"

requirements-completed: [PHYS-02]

# Metrics
duration: 4min
completed: 2026-03-29
---

# Phase 09 Plan 01: Galactic Confusion Noise Summary

**Galactic confusion noise S_c(f) added to LISA A/E-channel PSD using LDC parameterization of Cornish & Robson (2017), with T_obs-dependent foreground subtraction level**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-29T13:19:39Z
- **Completed:** 2026-03-29T13:23:54Z
- **Tasks:** 1 auto + 1 checkpoint (pending)
- **Files modified:** 2

## Accomplishments
- Added `_confusion_noise()` method to `LisaTdiConfiguration` computing S_c(f) with all 7 LDC constants from `constants.py`
- Added `t_obs_years` (default 4.0) and `include_confusion_noise` (default True) fields to dataclass
- A/E-channel PSD now includes galactic binary foreground; T-channel unchanged
- 6 new CPU tests validate confusion noise behavior (increases PSD at 1 mHz, negligible above 10 mHz, backward compat, positivity, T_obs dependence, T-channel isolation)
- All 192 CPU tests pass with no regressions; mypy and ruff clean

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `b7cb4cd` (test)
2. **Task 1 GREEN: Implementation + test fix** - `3bed9fc` ([PHYSICS] feat)

_TDD approach: RED phase committed separately, then GREEN with implementation._

## Files Created/Modified
- `master_thesis_code/LISA_configuration.py` - Added confusion noise method, dataclass fields, conditional PSD addition
- `master_thesis_code_test/LISA_configuration_test.py` - 6 new CPU confusion noise tests

## Decisions Made
- **T_obs units correction:** The plan specified `t_obs_sec = self.t_obs_years * YEAR_IN_SEC` for the log10 argument in the power-law fits for f1 and fk. This produced f1 ~ 19 microHz and fk ~ 22 microHz, making S_c(f) zero across the entire LISA band. The coefficients (a1, b1, ak, bk) from the LDC parameterization expect T_obs in **years**, producing f1 ~ 1.4 mHz and fk ~ 2.3 mHz, consistent with Robson et al. (2019).
- **Citation:** Reference comment cites Cornish & Robson (2017) arXiv:1703.09858 Eq. (3) as the primary source. The constants.py comment citing arXiv:2303.15929 Eq. 17 is noted as incorrect (that paper does not contain a confusion noise formula).
- **Test range:** `test_confusion_noise_positive` uses 0.1--3 mHz range instead of the plan's 0.1 mHz--0.1 Hz, since S_c underflows to 0.0 above ~5 mHz due to the steep exponential suppression.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] T_obs unit error in confusion noise formula**
- **Found during:** Task 1 GREEN (implementation verification)
- **Issue:** Plan specified `t_obs_sec = self.t_obs_years * YEAR_IN_SEC` then `xp.log10(t_obs_sec)` for the power-law fits. With T_obs in seconds (1.26e8), the transition frequencies f1 and fk were ~20 microHz, making S_c effectively zero across the entire LISA band. The LDC power-law coefficients expect T_obs in years.
- **Fix:** Changed to `xp.log10(self.t_obs_years)` directly. Removed unused `YEAR_IN_SEC` import.
- **Files modified:** `master_thesis_code/LISA_configuration.py`
- **Verification:** S_c(1 mHz) = 6.65e-38 Hz^-1 with T_obs in years, consistent with Robson et al. (2019) value of 6.66e-38.
- **Committed in:** `3bed9fc`

**2. [Rule 1 - Bug] Test range for confusion noise positivity**
- **Found during:** Task 1 GREEN (test verification)
- **Issue:** Plan specified `np.logspace(-4, -1, 100)` (0.1 mHz to 0.1 Hz) for positivity test, but S_c underflows to exactly 0.0 above ~5 mHz due to `exp(-(f/f1)^1.8)` suppression.
- **Fix:** Narrowed test range to `np.logspace(-4, log10(3e-3), 50)` covering the physically relevant 0.1--3 mHz band.
- **Files modified:** `master_thesis_code_test/LISA_configuration_test.py`
- **Verification:** All values strictly positive in narrowed range; underflow above is physically correct behavior.
- **Committed in:** `3bed9fc`

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for physical correctness. No scope creep.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully wired.

## Next Phase Readiness
- Confusion noise is active by default; EMRI SNR estimates are now more realistic in the 0.1--3 mHz band
- Existing callers of `LisaTdiConfiguration()` automatically get confusion noise (backward compatible via `include_confusion_noise=False`)
- Detection yield will decrease with realistic noise; campaign size calibration needed (Phase 11)
- Fisher matrix stencil upgrade (Phase 10) is independent and can proceed

---
*Phase: 09-galactic-confusion-noise*
*Completed: 2026-03-29*
