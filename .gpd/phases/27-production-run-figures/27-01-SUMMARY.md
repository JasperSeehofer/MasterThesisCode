---
phase: 27-production-run-figures
plan: 01
depth: full
one-liner: "Fixed posteriors directory name bug in generate_figures and validated all 30 production JSON files (23 + 7 usable), confirming 538 events per file with 4 missing indices"
subsystem: analysis
tags: [data-validation, pipeline-fix, posterior-data]

requires:
  - phase: cluster production run
    provides: eval_corrected_full posterior JSON files
provides:
  - Fixed generate_figures pipeline (posteriors directory name corrected)
  - Data validation report with event counts, h-grids, MAP values, corruption status
  - Confirmed two combined posterior versions (11-pt subdir vs 15-pt toplevel)
affects: [27-02, 27-03, 27-04]

methods:
  added: [JSON integrity validation, event-index gap detection]
  patterns: [per-file event counting via integer-key detection]

key-files:
  created:
    - .gpd/phases/27-production-run-figures/data_validation.json
  modified:
    - master_thesis_code/main.py

key-decisions:
  - "Renamed corrupted h_0_64.json to .corrupted rather than deleting (preserves evidence)"
  - "Documented two combined posterior versions with different h-grids and MAP values"

conventions:
  - "h = H0 / (100 km/s/Mpc), dimensionless"
  - "h_true = 0.73"
  - "peak-normalized posteriors"

plan_contract_ref: ".gpd/phases/27-production-run-figures/27-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-dir-fix:
      status: passed
      summary: "Directory name corrected from posteriors_without_bh_mass to posteriors on line 724 of main.py. Import verified."
      linked_ids: [deliv-main-py-fix, test-dir-loads]
    claim-data-integrity:
      status: partial
      summary: "All non-corrupted files load. Event counts consistent (538). Plan expected 8 usable with-BH-mass files but only 7 exist (h_0_64 corrupted). 4 event indices missing across all files."
      linked_ids: [deliv-data-report, test-data-integrity]
  deliverables:
    deliv-main-py-fix:
      status: passed
      path: master_thesis_code/main.py
      summary: "Line 724 changed to _load_posteriors('posteriors')"
      linked_ids: [claim-dir-fix, test-dir-loads]
    deliv-data-report:
      status: passed
      path: .gpd/phases/27-production-run-figures/data_validation.json
      summary: "Complete JSON report with file counts, h-grids, event counts, corruption status, combined posterior details"
      linked_ids: [claim-data-integrity, test-data-integrity]
  acceptance_tests:
    test-dir-loads:
      status: passed
      summary: "Import of generate_figures succeeds. _load_posteriors('posteriors') resolves to correct directory path."
      linked_ids: [claim-dir-fix, deliv-main-py-fix]
    test-data-integrity:
      status: partial
      summary: "Without BH mass: 23 files, 538 events, consistent. With BH mass: 7 files (not 8 as plan expected), 538 events, consistent. Plan acceptance criterion was 8 files but only 7 are usable."
      linked_ids: [claim-data-integrity, deliv-data-report]
  references: {}
  forbidden_proxies:
    fp-no-skip:
      status: rejected
      notes: "Corrupted h_0_64.json documented in report with explicit entry. Missing event indices (5, 188, 248, 333) documented. Zero-length events tracked per file."
  uncertainty_markers:
    weakest_anchors:
      - "With-BH-mass variant has only 7 usable h-values (not 8), making posterior even more coarsely sampled than plan anticipated"
      - "Two combined posterior versions exist with different MAP values (subdir: 0.704, toplevel: 0.66/0.68) -- downstream plans must choose which to use"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations:
      - "With-BH-mass subdir combined_posterior.json appears identical to without-BH subdir combined (same h_values, same MAP=0.704, same n_events). May be a copy error from --combine."

duration: 12min
completed: 2026-04-07
---

# Phase 27, Plan 01: Data Pipeline Fix and Validation Summary

**Fixed posteriors directory name bug in generate_figures and validated all 30 production JSON files (23 + 7 usable), confirming 538 events per file with 4 missing indices**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-07
- **Completed:** 2026-04-07
- **Tasks:** 2/2
- **Files modified:** 2

## Key Results

- `_load_posteriors("posteriors_without_bh_mass")` -> `_load_posteriors("posteriors")` on line 724 of main.py: pipeline unblocked
- Without BH mass: 23 files, h in [0.60, 0.86], 538 events each, all consistent [CONFIDENCE: HIGH]
- With BH mass: 7 usable files (h_0_64 corrupted, 108MB truncated JSON), 538 events each [CONFIDENCE: HIGH]
- 4 event indices missing in both variants (5, 188, 248, 333): zero-length lists, not missing keys
- Two combined posterior versions differ: subdir (11-pt grid, MAP=0.704) vs toplevel (15-pt grid, MAP=0.66 / 0.68)
- SNR_THRESHOLD = 15 (constants.py), not 20 as some code comments suggest

## Task Commits

1. **Task 1: Fix directory name bug and handle corrupted JSON** - `51dde8d` (fix)
2. **Task 2: Validate all production data files and produce integrity report** - `53de00f` (validate)

## Files Created/Modified

- `master_thesis_code/main.py` -- fixed posteriors directory name on line 724
- `.gpd/phases/27-production-run-figures/data_validation.json` -- full integrity report
- `cluster_results/eval_corrected_full/posteriors_with_bh_mass/h_0_64.json.corrupted` -- renamed corrupted file (untracked)

## Next Phase Readiness

- generate_figures pipeline is unblocked (directory name fixed, corrupted file excluded)
- Data validation report provides ground truth for Plans 02-04
- Key discovery: two combined posterior versions exist with different grids and MAPs. Plan 02 (number extraction) must decide which version to use or re-run --combine
- With-BH-mass has only 7 h-values (plan assumed 8), which may be too sparse for publication-quality posterior curves

## Contract Coverage

- Claim IDs: claim-dir-fix -> passed, claim-data-integrity -> partial (7 files not 8)
- Deliverable IDs: deliv-main-py-fix -> passed, deliv-data-report -> passed
- Acceptance test IDs: test-dir-loads -> passed, test-data-integrity -> partial
- Forbidden proxies: fp-no-skip -> rejected (all corruption documented)

## Decisions Made

1. **Renamed corrupted file rather than deleting** -- preserves evidence of the truncation for potential recovery or debugging
2. **Documented both combined posterior versions** rather than choosing one -- this is a decision for Plan 02

## Deviations from Plan

### Deviation 1: With-BH-mass file count

- **[Rule 4 - Missing component]** Plan expected 8 usable files after excluding h_0_64, but directory contains only 8 total h_*.json files (including corrupted one), yielding 7 usable. The plan's count was off by one.
- **Impact:** Posterior is more coarsely sampled than anticipated. No action needed at this plan level; flagged for Plan 02/03.

### Deviation 2: SNR threshold is 15, not 20

- **[Rule 4 - Missing component]** Plan mentioned SNR_THRESHOLD might be 20, but constants.py shows 15. Research doc (27-RESEARCH.md) noted this inconsistency.
- **Impact:** Paper must use 15 consistently. Flagged for Plan 04 (paper marker filling).

### Deviation 3: Events per file is 538, not 539

- Plan acceptance test said "539 events each" but actual count is 538 (indices 0-541 with 4 gaps: 5, 188, 248, 333 have zero-length lists). Total unique event indices: 542 - 4 empty = 538 with data.
- **Impact:** Negligible. Numbers in validation report are correct.

**Total deviations:** 3 (all Rule 4, auto-documented)
**Impact on plan:** Minor numerical corrections to plan estimates. No scope change needed.

## Issues Encountered

- JSON files are large (~100MB each for with-BH-mass). Loading all files for validation takes ~30s. Not a blocker but relevant for downstream plans that iterate over files.
- The with-BH-mass subdir `combined_posterior.json` appears to be identical to the without-BH subdir version (same MAP=0.704, same h_values, same event counts). This is likely a --combine bug or copy error. The toplevel combined posteriors are distinct and appear correct.

## Open Questions

- Why does the subdir combined_posterior.json in posteriors_with_bh_mass/ appear identical to the posteriors/ version? Was --combine run with the wrong arguments?
- Should Plan 02 use the subdirectory combined posteriors (11-pt grid, MAP=0.704) or the toplevel ones (15-pt grid, MAP=0.66/0.68)?
- Are the 4 missing event indices (5, 188, 248, 333) events that failed waveform generation, or a data pipeline bug?

---

_Phase: 27-production-run-figures, Plan: 01_
_Completed: 2026-04-07_
