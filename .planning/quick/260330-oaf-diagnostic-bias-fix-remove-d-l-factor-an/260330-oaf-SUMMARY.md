---
phase: quick
plan: 01
subsystem: bayesian-inference
tags: [diagnostic, bias-audit, pipeline-b]
dependency_graph:
  requires: []
  provides: [diagnostic-likelihood-mode]
  affects: [bayesian_statistics.py]
tech_stack:
  added: []
  patterns: [debug-flag-bypass]
key_files:
  created: []
  modified:
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
decisions:
  - "Used module-level bool flag for P_det bypass (simple, no runtime overhead when False)"
  - "Removed /d_L from both numerator integrands as suspected bias source"
metrics:
  duration: 85s
  completed: "2026-03-30T15:34:00Z"
---

# Quick Task 260330-oaf: Diagnostic Bias Fix Summary

Diagnostic mode for Pipeline B likelihood: removed spurious d_L divisor from both numerator integrands and added a flag to bypass detection probability (P_det=1.0) in all four integrand functions.

## Changes Made

### Task 1: Add debug flag, remove /d_L, bypass P_det

**Commit:** `ae118d4`

Three changes to `bayesian_statistics.py`:

1. **Debug flag + warning:** Added `_DEBUG_DISABLE_DETECTION_PROBABILITY = True` at module level (line 60). Emits a `DIAGNOSTIC MODE` warning at import time so it is impossible to accidentally run production code with the flag on.

2. **Removed `/d_L` from numerator integrands:**
   - `numerator_integrant_without_bh_mass`: removed trailing `/ d_L` from return expression
   - `numerator_integrant_with_bh_mass`: changed `/ (d_L * (1 + z))` to `/ (1 + z)`, preserving the mass Jacobian while removing the suspect d_L divisor. Removed the `# TODO: check if this is correct` comment.

3. **P_det bypass:** All four integrand functions now use a `p_det` local variable that is `1.0` when the flag is True, otherwise calls the original `detection_probability.*_interpolated(...)` method. The four locations:
   - `numerator_integrant_without_bh_mass`
   - `denominator_integrant_without_bh_mass`
   - `numerator_integrant_with_bh_mass`
   - `denominator_integrant_with_bh_mass_vectorized`

### Task 2: Run existing tests

198 passed, 18 deselected (GPU/slow), 0 failures. Coverage 40.80% (gate: 25%).

## Deviations from Plan

None -- plan executed exactly as written.

## Known Stubs

- `_DEBUG_DISABLE_DETECTION_PROBABILITY = True` is intentionally a diagnostic stub. It must be set back to `False` (or removed entirely) after the bias audit confirms which factor causes the posterior shift. Tracked by the `TODO(bias-audit)` comment.

## Verification

1. `grep "_DEBUG_DISABLE_DETECTION_PROBABILITY"` shows 8 occurrences: 1 definition, 1 warning check, 1 warning message, 4 usage sites in integrands, plus the TODO comment.
2. `grep "/ d_L"` in the integrands returns only comments (fraction explanation), not code.
3. Import triggers the DIAGNOSTIC MODE warning.
4. ruff, mypy, and pytest all pass.

## Self-Check: PASSED
