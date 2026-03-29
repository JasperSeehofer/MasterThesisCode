---
phase: 09-galactic-confusion-noise
verified: 2026-03-29T14:00:00Z
status: human_needed
score: 5/5 must-haves verified
re_verification: false
human_verification:
  - test: "Confirm that the citation discrepancy is intentionally resolved"
    expected: >
      The ROADMAP success criterion #4 requires a reference comment citing
      "Babak et al. (2023) arXiv:2303.15929", but the implementation cites
      Cornish & Robson (2017) arXiv:1703.09858 as the primary source for the
      _confusion_noise method.  The SUMMARY documents this was a deliberate
      correction: arXiv:2303.15929 does not contain the confusion noise formula.
      Babak et al. is present in the class docstring and the power_spectral_density_a_channel
      docstring for the instrumental noise component.  Human should confirm this
      deviation from the ROADMAP criterion is accepted, and optionally update the
      ROADMAP success criterion to reflect the correct citation.
    why_human: >
      The ROADMAP recorded an incorrect citation as a success criterion before the
      research phase discovered the actual source.  The code is physically correct.
      Only a human can decide whether the ROADMAP criterion should be updated or
      left as-is, and whether the current citation arrangement is acceptable for
      the thesis.
---

# Phase 9: Galactic Confusion Noise Verification Report

**Phase Goal:** LISA PSD includes the galactic confusion foreground, producing physically correct SNR estimates
**Verified:** 2026-03-29T14:00:00Z
**Status:** human_needed (automated checks all pass; one citation question for human)
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PSD at 1 mHz with confusion noise is measurably larger than PSD without | VERIFIED | Spot-check: 6.654e-38 vs 8.844e-43 Hz^-1 (ratio 75,237x); test_confusion_noise_increases_psd_at_1mhz passes |
| 2 | PSD at 10 mHz is negligibly affected by confusion noise (< 1% change) | VERIFIED | Spot-check: ratio = 1.000000 at 10 mHz; test_confusion_noise_negligible_above_10mhz passes |
| 3 | Setting include_confusion_noise=False reproduces the old instrumental-only PSD exactly | VERIFIED | test_confusion_noise_toggle_backward_compat passes; LisaTdiConfiguration(include_confusion_noise=False) gives all-positive PSD |
| 4 | T-channel PSD is unaffected by the confusion noise toggle | VERIFIED | Spot-check: np.allclose True; test_t_channel_unaffected_by_confusion_noise passes; _confusion_noise absent from power_spectral_density_t_channel |
| 5 | All existing CPU tests pass without modification | VERIFIED | Full suite: 192 passed, 18 deselected (14 GPU + 2 slow + 2 integration), 0 failures; coverage 39.86% (gate 25%) |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `master_thesis_code/LISA_configuration.py` | LisaTdiConfiguration with _confusion_noise method and t_obs_years/include_confusion_noise fields | VERIFIED | File exists, contains all required symbols; 173 lines total |
| `master_thesis_code_test/LISA_configuration_test.py` | CPU tests for confusion noise behavior | VERIFIED | File exists, contains all 6 required test functions |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| power_spectral_density_a_channel | _confusion_noise | conditional addition when self.include_confusion_noise is True | WIRED | Line 135: `if self.include_confusion_noise:` followed by `instrumental = instrumental + self._confusion_noise(frequencies)` at line 136 |
| _confusion_noise | constants.py | imports LISA_PSD_A, LISA_PSD_ALPHA, LISA_PSD_F2, LISA_PSD_A1, LISA_PSD_B1, LISA_PSD_AK, LISA_PSD_BK | WIRED | All 7 constants imported at lines 28-36 and used in the formula at lines 101-109 |

**Note on key link deviation:** The PLAN specified importing `YEAR_IN_SEC` from constants. The implementation omits it -- the SUMMARY documents this as an intentional bug fix: the power-law coefficients (a1, b1, ak, bk) were fitted with T_obs in years, so `xp.log10(self.t_obs_years)` is used directly. Using seconds produced transition frequencies ~20 microHz, making S_c effectively zero across the LISA band. The link is functionally correct; the import was never needed.

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `power_spectral_density_a_channel` | `instrumental` (PSD array) | Formula over input `frequencies` + 7 constants from constants.py | Yes -- numerical computation with real constants | FLOWING |
| `_confusion_noise` | return value (S_c array) | `self.t_obs_years`, 7 PSD constants, input `frequencies` | Yes -- spot-check S_c(1 mHz) = 6.654e-38 matches Robson et al. (2019) reference of 6.66e-38 | FLOWING |

### Behavioral Spot-Checks

| Behavior | Result | Status |
|----------|--------|--------|
| S_c(1 mHz, t_obs=4yr) physically correct | 6.654e-38 Hz^-1 vs Robson+2019 reference 6.66e-38 (0.09% error) | PASS |
| PSD ratio (with/without) at 1 mHz | 75,236.8x (dominates by 4.5 orders of magnitude as expected) | PASS |
| PSD ratio (with/without) at 10 mHz | 1.000000 (confusion noise fully suppressed) | PASS |
| T-channel identical regardless of toggle | np.allclose = True | PASS |
| t_obs=1yr vs t_obs=4yr produces different PSD | 8.544e-38 vs 6.654e-38 Hz^-1 (differ: True) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PHYS-02 | 09-01-PLAN.md | LISA PSD includes galactic confusion noise term (Babak et al. 2023) | SATISFIED | `power_spectral_density_a_channel` now adds S_c(f) when `include_confusion_noise=True` (default); confirmed by tests and spot-checks |

**REQUIREMENTS.md traceability:** PHYS-02 mapped to Phase 9 in the traceability table. No orphaned requirements for this phase.

**ROADMAP success criteria cross-check:**

| Criterion | Status | Notes |
|-----------|--------|-------|
| 1. power_spectral_density_a_channel() includes galactic confusion noise term | SATISFIED | S_c(f) added conditionally; active by default |
| 2. PSD at 1 mHz with confusion noise is measurably larger than without | SATISFIED | Ratio 75,237x verified by test and spot-check |
| 3. Existing CPU tests pass with updated PSD (no regression) | SATISFIED | 192/192 CPU tests pass |
| 4. Reference comment citing Babak et al. (2023) arXiv:2303.15929 appears above confusion noise implementation | PARTIAL -- see human verification | Primary citation above _confusion_noise is Cornish & Robson (2017) arXiv:1703.09858 (correct source). Babak et al. arXiv:2303.15929 appears in class docstring and a_channel method docstring for instrumental noise. SUMMARY documents that arXiv:2303.15929 does not contain the confusion noise formula. |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `master_thesis_code/constants.py` | 74 | Comment `# LISA galactic confusion noise PSD coefficients (arXiv:2303.15929, Eq. 17)` -- incorrect citation (paper does not contain confusion noise formula) | Info | No code impact; misleading comment only. Pre-existing issue noted in Phase 9 PLAN. Not introduced by this phase. |

No stub patterns, empty implementations, or TODO/FIXME markers found in the modified files.

### Human Verification Required

#### 1. Citation arrangement accepted for thesis

**Test:** Review the citation layout in `master_thesis_code/LISA_configuration.py`:
- Class docstring (lines 62-64): lists Babak et al. (2023) arXiv:2303.15929, Cornish & Robson (2017) arXiv:1703.09858, and Robson, Cornish & Liu (2019) arXiv:1803.01944
- Above `_confusion_noise` (lines 83-85): cites Cornish & Robson (2017) arXiv:1703.09858 Eq. (3) as primary, and Robson et al. (2019) arXiv:1803.01944 Eq. (14)
- In `power_spectral_density_a_channel` docstring (lines 117-118): instrumental noise cites Babak et al. (2023), confusion noise cites Cornish & Robson (2017)

**Expected:** You confirm that citing Cornish & Robson (2017) / Robson et al. (2019) as the primary source for the confusion noise formula is correct for the thesis, and that the ROADMAP success criterion #4 (which asked for Babak et al. arXiv:2303.15929 above the implementation) was based on an incorrect assumption that has been appropriately corrected.

**Why human:** The ROADMAP recorded an incorrect citation target before the research phase discovered the actual source of the formula. The code is physically correct (S_c(1 mHz) within 0.09% of published value). Only you can decide whether to update the ROADMAP criterion text or accept the current citation arrangement.

**Optional cleanup:** If you want the `constants.py` comment at line 74 corrected from `(arXiv:2303.15929, Eq. 17)` to `(Cornish & Robson 2017, arXiv:1703.09858, Eq. 3)`, that is a safe doc-only change that can be done in a follow-up commit.

### Gaps Summary

No blocking gaps. All 5 observable truths are verified, both artifacts are substantive and wired, data flows through the formula producing physically correct values (S_c within 0.09% of reference), all key links are wired, PHYS-02 is satisfied, and 192 CPU tests pass with 39.86% coverage.

The sole outstanding item is a citation question: the ROADMAP's success criterion #4 named arXiv:2303.15929 as the required citation for the confusion noise implementation, but the implementation correctly uses Cornish & Robson (2017) arXiv:1703.09858 because the former paper does not contain the formula. This needs human sign-off, not code changes.

---

_Verified: 2026-03-29T14:00:00Z_
_Verifier: Claude (gsd-verifier)_
