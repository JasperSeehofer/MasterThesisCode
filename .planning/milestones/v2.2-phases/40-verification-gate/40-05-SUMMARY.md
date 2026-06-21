---
phase: 40-verification-gate
plan: "05"
subsystem: bayesian-inference
status: COMPLETE
date: 2026-04-24
requirements: [VERIFY-05]
requirements-completed: [VERIFY-05]
verdict: PHASE-41-TRIGGER-BORDERLINE
phase_41_trigger: PHASE-41-TRIGGER-BORDERLINE
phase_41_trigger_bool: false
mean_h_073_numerator_lb: 0.0409
tags: [wave-3, VERIFY-05, quadrature-weight, pdet-diagnostic, phase-41-borderline]

requires:
  - phase: 40-verification-gate/40-03
    provides: Per-h sweep logs (verify03_sweep log with 37 h-section markers)
  - phase: 40-verification-gate/40-02
    provides: h=0.73 pre-captured STAT-04 WARNINGs (verify02_quadrature_warnings log)

provides:
  - VERIFY-05 P_det quadrature-weight-outside-grid diagnostic across all 38 h-values
  - Phase 41 borderline trigger flag for Plan 40-07 routing decision
  - Raw CSV (990010 rows), per-h summary table, all-h aggregate, 10-bin histogram

key-files:
  created:
    - .planning/debug/pdet_quadrature_raw_20260423T172607Z.csv
    - .planning/debug/pdet_quadrature_parser_20260423T172607Z.py
    - .planning/debug/pdet_quadrature_aggregator_20260423T172607Z.py
    - .planning/debug/pdet_quadrature_summary_20260423T172607Z.md
    - .planning/debug/pdet_quadrature_summary_20260423T172607Z.json

key-decisions:
  - "VERIFY-05 PHASE-41-TRIGGER-BORDERLINE: mean_{h=0.73}(per-event max numerator) lb = 0.0409, inside borderline band [0.03, 0.05]"
  - "23 unique events at h=0.73 have max_numerator > 5%; 19 events are 100% off-grid (numerator=1.000)"
  - "Event 2 confirmed: max_numerator=1.000, 100% off-grid across all host-galaxy calls"
  - "Aggregation unit corrected (Rule 1): per-event max, not raw row sum (raw sum/N_events = 15.09 >> 1, meaningless)"
  - "Data source deviation (Rule 3): WARNINGs in verify03_sweep log (section-delimited), not per-h master_thesis_code_*.log files"

duration: ~20min
completed: 2026-04-24
---

# Phase 40 Plan 05: VERIFY-05 — Summary

**One-liner:** Grepped 990010 STAT-04 WARNING rows from 38 h-value sources, corrected per-event max aggregation, computed Phase 41 trigger at h=0.73 as BORDERLINE (mean_lb=0.0409, band [0.03, 0.05]).

## Verdict

**VERIFY-05: PHASE-41-TRIGGER-BORDERLINE**

D-18 rule: `mean_{h=0.73}(per-event max quadrature_weight_outside_grid_numerator) > 0.05` triggers Phase 41.

Observed:
- `mean_h_073_numerator_lb = 0.0409` (lower bound — unlogged events treated as 0.0; per-event max aggregation)
- Inside borderline band [0.03, 0.05] — **user decision required** before Phase 41 routing

Key signal: 23 of 542 events (4.24%) have at least one host-galaxy call with numerator > 5% at h=0.73. Of those, **19 events are 100% off-grid** (max_numerator = 1.000), including Event 2 which was flagged in Phase 38.

## Phase 41 Trigger Signal

| Metric | Value |
|--------|-------|
| mean_h_073_numerator_lb (per-event max) | 0.0409 |
| Borderline band | [0.03, 0.05] |
| Verdict | PHASE-41-TRIGGER-BORDERLINE |
| N events above 5% at h=0.73 | 23 / 542 (4.24%) |
| N events 100% off-grid at h=0.73 | 19 |
| Event 2 max_numerator | 1.000 (confirmed 100% off-grid) |

The borderline verdict means: the D-18 threshold of 0.05 was NOT exceeded, but the signal is close enough that Plan 40-07 must present the data to the user before routing Phase 41.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Data source: STAT-04 WARNINGs not in per-h master_thesis_code_*.log files**
- **Found during:** Task 1 execution
- **Issue:** The plan assumed WARNINGs would appear in `simulations/master_thesis_code_*_h_0_*.log`.
  Inspection showed those files do NOT contain WARNING lines — the application logger uses a
  `%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s` formatter on the file
  handler, but the STAT-04 WARNINGs emitted during the 40-03 sweep were captured in the
  consolidated sweep shell log (`verify03_sweep_20260423T172607Z.log`) via section markers
  `=== BEGIN h=X ===` / `=== END h=X ===`.
- **Fix:** Parser reads sweep log (section-delimited, 37 h-values) + verify02_quadrature_warnings
  (h=0.73 pre-captured file). 40-03 PASS gate bypassed (same precedent as 40-04: SC-3 FAIL is
  an investigation item, not a blocker for VERIFY-05).
- **Files modified:** `.planning/debug/pdet_quadrature_parser_20260423T172607Z.py`

**2. [Rule 1 - Bug] Aggregation unit: per-row sum inflated by multi-host-galaxy counting**
- **Found during:** Task 2 — first aggregator run returned mean_h_073_numerator_lb = 15.09
- **Issue:** The STAT-04 WARNING fires once per **host-galaxy call** (not per event). One event
  with 500 potential host galaxies generates ~500 WARNING rows. Summing all rows / N_events_total
  produces a value >> 1, which is meaningless as a "fraction outside grid."
- **Fix:** Aggregator corrected to first collapse each event to its **max(numerator)** across
  all host-galaxy calls, then compute `sum(per-event max) / N_events_total`. This is the
  correct lower-bound mean: unlogged events (all host calls <= 5%) contribute 0.0.
- **Corrected value:** mean_h_073_numerator_lb = 0.0409 (was 15.09)
- **Files modified:** `.planning/debug/pdet_quadrature_aggregator_20260423T172607Z.py`

## Artifacts

| Artifact | Description |
|----------|-------------|
| `.planning/debug/pdet_quadrature_raw_20260423T172607Z.csv` | 990010 raw WARNING rows (per host-galaxy call), 38 h-values |
| `.planning/debug/pdet_quadrature_summary_20260423T172607Z.md` | Human-readable report: per-h table, all-h aggregate, histogram, dominant events |
| `.planning/debug/pdet_quadrature_summary_20260423T172607Z.json` | Machine-readable (consumed by 40-07 index) — `phase_41_trigger` key present |
| `.planning/debug/pdet_quadrature_parser_20260423T172607Z.py` | Parser driver |
| `.planning/debug/pdet_quadrature_aggregator_20260423T172607Z.py` | Aggregator driver |

## Commits

| Hash | Message |
|------|---------|
| `e8d5cd9` | feat(40-05): Task 1 — parse STAT-04 WARNINGs into raw CSV (990010 rows, 38 h-values) |
| `16d7c6e` | feat(40-05): Task 2 — VERIFY-05 quadrature summary PHASE-41-TRIGGER-BORDERLINE |
| `(this commit)` | docs(40-05): VERIFY-05 quadrature summary — PHASE-41-TRIGGER-BORDERLINE [ts=20260423T172607Z] |

## Next

- Plan 40-07 (phase close) reads this SUMMARY's `phase_41_trigger: PHASE-41-TRIGGER-BORDERLINE`
  frontmatter to present the borderline decision to the user in the verify_gate index
- Phase 41 (Stage 1 Injection Campaign) — routing decision in 40-07 based on this verdict
- Event 2 and the 19 other 100%-off-grid events should be prioritized if Phase 41 runs
