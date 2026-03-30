---
phase: quick
plan: 260330-twe
title: Re-run h-value sweep evaluation with BH mass Gaussian index fix
status: complete
completed: 2026-03-30
duration: ~24 min
tasks_completed: 3
tasks_total: 3
key_files:
  created:
    - evaluation/run_v12_bhmass_fix/ (evaluation directory with all outputs)
    - evaluation/run_v12_bhmass_fix/simulations/posteriors/ (11 JSON files)
    - evaluation/run_v12_bhmass_fix/simulations/posteriors_with_bh_mass/ (11 JSON files)
    - evaluation/run_v12_bhmass_fix/comparison_report.md
decisions:
  - "BH mass Gaussian index fix (ab77e70) has zero effect due to delta-function M_z approximation"
  - "'With BH mass' path needs structural changes (integrate over M_z) to contribute information"
---

# Quick Task 260330-twe: Re-run H-Value Sweep with BH Mass Fix -- Summary

BH mass Gaussian index fix (ab77e70) re-evaluated across 11 h-values; fix produced zero change in posteriors due to delta-function M_z approximation.

## Tasks Completed

| # | Task | Result |
|---|------|--------|
| 1 | Set up evaluation directory | Created run_v12_bhmass_fix with CRB data, galaxy catalog symlink |
| 2 | Run h-value sweep (11 values) | All 22 posterior JSON files generated (~11s each) |
| 3 | Generate comparison report | Report at evaluation/run_v12_bhmass_fix/comparison_report.md |

## Key Finding

The fix (using 4D Gaussian `[1]` instead of 3D `[0]`, adding `redshifted_mass_fraction` coordinate) is mathematically correct but has **no practical effect** because:

- `redshifted_mass_fraction = M_z / detection.M = 1.0` always (delta-function approximation)
- Evaluating the 4D Gaussian at its mean in the M_z dimension produces a constant factor
- This constant cancels between numerator and denominator

## Bias Status (unchanged from previous diagnostic run)

| Path | Peak h | Offset from h_true=0.73 |
|------|--------|-------------------------|
| Without BH mass | 0.678 | -0.052 |
| With BH mass | 0.600 | -0.130 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Missing undetected_events.csv**
- **Found during:** Task 2
- **Issue:** Pipeline B requires `simulations/undetected_events.csv` which was not listed in the plan's copy commands
- **Fix:** Copied from run_v12_diagnostic alongside the other CSV files

## Known Stubs

None.

## Self-Check: PASSED

- evaluation/run_v12_bhmass_fix/simulations/posteriors/ contains 11 JSON files
- evaluation/run_v12_bhmass_fix/simulations/posteriors_with_bh_mass/ contains 11 JSON files
- evaluation/run_v12_bhmass_fix/comparison_report.md exists with full analysis
