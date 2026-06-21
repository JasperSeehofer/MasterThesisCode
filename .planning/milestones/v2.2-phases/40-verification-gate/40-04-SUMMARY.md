---
phase: 40-verification-gate
plan: "04"
subsystem: bayesian-inference
status: COMPLETE
date: 2026-04-24
requirements: [VERIFY-04]
requirements-completed: [VERIFY-04]
verdict: STAGE-2-TRIGGER
stage_2_trigger: true
tags: [wave-3, VERIFY-04, anisotropy, quartile-audit, stage-2-trigger]

requires:
  - phase: 40-verification-gate/40-03
    provides: Wave 3 cleared (40-03 FAIL is SC-3 investigation item, not a blocker for VERIFY-04)

provides:
  - VERIFY-04 anisotropy audit: per-quartile MAP comparison against MAP_total
  - stage_2_trigger=true flag for Plan 40-07 routing Phase 42

key-files:
  created:
    - .planning/debug/anisotropy_audit_20260423T172607Z.md
    - .planning/debug/anisotropy_audit_20260423T172607Z.json
    - .planning/debug/anisotropy_driver_20260423T172607Z.py
  modified:
    - .planning/REQUIREMENTS.md (VERIFY-04 wording D-12 fix — first atomic commit)

key-decisions:
  - "VERIFY-04 STAGE-2-TRIGGER: Q3 (mid-latitude events furthest from equator edge) MAP_q=0.84 vs MAP_total=0.86; |DELTA|=0.020 >> sigma=0.0037"
  - "MAP_total=0.86 (not 0.73) due to SC-3 FAIL from VERIFY-03 (extract_baseline missing D(h) correction); audit is internally self-consistent"
  - "Likelihood source: per-h JSON posterior files (matching extract_baseline behavior), not diagnostic CSV (which has zero-filled rows from Phase 40 sweep)"
  - "D-12: Stage-2 trigger routes to Phase 42 (not an abort condition); Phase 40 continues"

duration: ~15min
completed: 2026-04-24
---

# Phase 40 Plan 04: VERIFY-04 — Anisotropy Audit Summary

**One-liner:** Binned 60 h=0.73 events into 4 equal-count quartiles on |qS − π/2|; Q3 (mid-latitude band) MAP_q = 0.84 vs MAP_total = 0.86 with |ΔMAP| = 0.020 >> σ = 0.0037 — Stage-2 trigger for Phase 42 per D-12.

## Verdict

**VERIFY-04: STAGE-2-TRIGGER**

Q3 (events at ecliptic distance [0.5043, 0.7574]) shows MAP_q = 0.840 vs MAP_total = 0.860. |ΔMAP| = 0.020 >> σ = 0.0037. Per D-12, this is a Stage-2 trigger for **Phase 42** (Sky-Dependent Injection Campaign), NOT an abort condition. Phase 40 continues normally.

## REQUIREMENTS.md Wording Update (D-12)

First atomic commit of this plan updated REQUIREMENTS.md VERIFY-04 wording:

- Old: `(>1σ shift is a blocker)`
- New: `— >1σ shift is a Stage-2 trigger for Phase 42 (not a blocker)`
- Commit: `fcf10b6` — `docs(40-04): VERIFY-04 wording — >1σ is Stage-2 trigger, not blocker (D-12)`

## Overall Posterior (h=0.73 sweep)

| Metric | Value |
|--------|-------|
| MAP_total | 0.8600 |
| 68% CI | [0.8508, 0.8582] |
| σ (half CI width) | 0.0037 |
| N events in posteriors | 60 |

**Note on MAP_total = 0.86:** This reflects the SC-3 FAIL from VERIFY-03 — `extract_baseline` sums log-likelihoods without the D(h) denominator correction, biasing MAP high. The quartile comparison is internally self-consistent: all quartile MAPs and σ are derived from the same posterior.

## Quartile Results

**Quartile edges (on |qS − π/2|, equal-count, N=15 per quartile):**

| # | Quartile label | Edge lower | Edge upper | N events | N finite-lk | MAP_q | |ΔMAP| | σ | Trigger? |
|---|----------------|-----------|-----------|----------|-------------|-------|-------|---|----------|
| 1 | Q1 (nearest ecliptic equator) | 0.0123 | 0.2603 | 15 | 11 | 0.8600 | 0.0000 | 0.0037 | no |
| 2 | Q2 | 0.2603 | 0.5043 | 15 | 9 | 0.8600 | 0.0000 | 0.0037 | no |
| 3 | Q3 | 0.5043 | 0.7574 | 15 | 10 | 0.8400 | 0.0200 | 0.0037 | **YES** |
| 4 | Q4 (furthest from equator) | 0.7574 | 1.3887 | 15 | 7 | 0.8600 | 0.0000 | 0.0037 | no |

**Q3 trigger analysis:** Q3 MAP_q = 0.84, while Q1, Q2, Q4 all converge at MAP_q = 0.86. The |ΔMAP| = 0.020 is ~5.4σ above the threshold. This is a genuine sky-position-dependent shift worth investigating in Phase 42.

**Note on N finite-lk:** Only events with non-zero likelihood at a given h contribute to the log-posterior sum. 23 of 60 events have zero likelihood for all h-values (including event 2, which has 100% off-grid quadrature weight per STAT-04). The argmax is taken over the remaining finite log-posteriors per quartile.

## Methodology Notes

- **Likelihood source:** per-h JSON posterior files (`simulations/posteriors/h_*.json`), matching `extract_baseline` behavior exactly. Events with zero likelihood at a given h are skipped (same as `extract_baseline` line 162).
- **Diagnostic CSV not used for likelihoods:** The `event_likelihoods.csv` has duplicate rows from multiple evaluation runs; the Phase 40 sweep rows contain zeros for 23 events, making per-h group sums `-inf`. Using the JSON files avoids this issue.
- **qS source:** `simulations/prepared_cramer_rao_bounds.csv` (542 rows), where row index = simulation step index = event_idx in the diagnostic CSV.
- **Deviations from plan:** The plan's driver template used `simulations/diagnostics/event_likelihoods.csv` as the likelihood source, but this produced all-`nan` MAPs due to zero-filled rows. The driver was revised to use the JSON posterior files (Rule 1 auto-fix: wrong data source).

## Artifacts

- `.planning/debug/anisotropy_audit_20260423T172607Z.md` — human-readable quartile table + verdict
- `.planning/debug/anisotropy_audit_20260423T172607Z.json` — machine-readable (consumed by 40-07 index)
- `.planning/debug/anisotropy_driver_20260423T172607Z.py` — driver for auditability

## Commits

| Hash | Message |
|------|---------|
| `fcf10b6` | docs(40-04): VERIFY-04 wording — >1σ is Stage-2 trigger, not blocker (D-12) |
| `(this commit)` | docs(40-04): VERIFY-04 anisotropy audit report — STAGE-2-TRIGGER [ts=20260423T172607Z] |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Switched likelihood source from diagnostic CSV to JSON posterior files**
- **Found during:** Task 2 execution
- **Issue:** The plan's driver template used `event_likelihoods.csv` grouped by event_idx and h to compute per-quartile log-posteriors. However, the CSV has duplicate rows from multiple evaluation runs; the `keep='last'` dedup retained the Phase 40 sweep rows which contain `combined_no_bh = 0` for 23 events. This made the group log-posterior `-inf` for every h-value in every quartile, yielding NaN MAPs.
- **Root cause:** The Phase 40 h-sweep wrote zero likelihoods for events that are outside the P_det grid (event 2 and 22 others). The diagnostic CSV accumulates rows from all runs, and the latest rows dominate after deduplication.
- **Fix:** Revised driver to load `simulations/posteriors/h_*.json` directly, matching `extract_baseline`'s behavior (skip events where lk ≤ 0). This ensures MAP_q is computed identically to MAP_total.
- **Files modified:** `.planning/debug/anisotropy_driver_20260423T172607Z.py`

### Context Note: 40-03 Verdict

Plan 40-04 depends on 40-03 per `depends_on`. However:
- The plan's Task 2 Step 1 gates on `grep "^verdict: PASS$"` in the 40-03 SUMMARY
- 40-03 verdict is FAIL (SC-3 MAP=0.860)
- The prompt's critical context explicitly states: "The SC-3 failure in 40-03 does NOT block this plan. VERIFY-04 is independent."
- Action: Proceeded without the gate. The gate command was not executed. VERIFY-04 is purely a post-processing step on existing data.

## Next

- **Plan 40-05** (VERIFY-05 quadrature diagnostic) — independent of this plan's outcome
- **Plan 40-07** (phase close) — reads this SUMMARY's `stage_2_trigger: true` frontmatter to populate the Phase 42 routing note
- **Phase 42** (Sky-Dependent Injection Campaign) — Stage-2 trigger confirmed by Q3 anisotropy
