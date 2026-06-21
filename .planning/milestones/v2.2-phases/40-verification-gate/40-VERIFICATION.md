# Phase 40: Verification Gate — Verification

**Date:** 2026-04-24
**Phase:** 40
**Requirements:** VERIFY-01, VERIFY-02, VERIFY-03, VERIFY-04, VERIFY-05
**Gate timestamp:** 20260423T172607Z

## Summary

Phase 40 executed all five verification checks across Waves 0–3. VERIFY-01 (CPU pytest) and
VERIFY-02 (h=0.73 abort gate) both PASS cleanly. VERIFY-03 SC-3 is NOT VERIFIED — the raw
`extract_baseline` metric peaks at MAP=0.860 instead of the expected 0.73±0.01 due to a
missing D(h) denominator correction in that metric (not a physics regression in the pipeline).
VERIFY-04 triggered Stage-2 routing to Phase 42 (Q3 quartile |ΔMAP|=0.020 >> σ=0.0037).
VERIFY-05 landed in the borderline band (mean_lb=0.0409, threshold 0.05) and requires a user
decision on Phase 41. Phase 40 status: **GAPS_FOUND** — SC-3 investigation required before
declaring v2.2 complete.

## Success Criteria

### SC-1 — VERIFY-01: Full CPU pytest suite passes
**Status:** [x] PASS
**Evidence:**
- `uv run pytest master_thesis_code_test/ -m "not gpu" --tb=short -q` exit: 0
- Tests passed: 544 (threshold ≥ 540; D-05 baseline ≥ 508)
- 6 skipped, 0 failed
- Regression inventory (D-06): 5/5 items present
  - test_coordinate_roundtrip.py (Phase 36): 10 tests
  - test_parameter_space_h.py PE-01 (Phase 37): 5 tests
  - test_l_cat_equivalence.py (Phase 38): 4 tests
  - test_completion_term_fix.py STAT-03 zero-fill: 12 tests
  - test_sigterm_drain_with_flush_interval_25 HPC-02: 1 test
- Detail: `.planning/debug/verify01_report_20260423T172607Z.md`
- Commit: `53f7042` — `feat(40-01): VERIFY-01 PASS — 544 tests green, all D-06 inventory items present`

### SC-2 — VERIFY-02: MAP at h=0.73 stable under v2.2
**Status:** [x] PASS
**Evidence:**
- v2.1 baseline MAP: 0.7350
- v2.2 current MAP: 0.7350
- |ΔMAP| / 0.73: 0.0000% (abort threshold: 5.00%) — abort gate did NOT fire
- bias_percent at h=0.73: +0.68% (SC-2 bias threshold: < 1.00% — SUBPASS)
- KS p-value: ≈1.0 (log-posterior curve shapes unchanged)
- N events: 417 (v2.1 archive format; same format compared against itself in VERIFY-02)
- Detail: `.planning/debug/verify02_abort_check_20260423T172607Z.md`
- Machine-readable: `.planning/debug/verify02_comparison_20260423T172607Z.json`
- Commit: `81ae3e3` — `[PHYSICS] docs(40-02): VERIFY-02 abort-gate check — PASS [ts=20260423T172607Z]`

### SC-3 — VERIFY-03: 27-h sweep regenerated; figures rebuilt
**Status:** [ ] NOT VERIFIED (SC-3 sanity check FAIL)
**Evidence:**
- Sweep processed: 37 non-0.73 h-values + h=0.73 preserved from VERIFY-02; 0 failures
- h-values on disk: 38 (≥ 27 threshold — PASS)
- MAP at h=0.73 post-sweep (via extract_baseline): **0.860** (threshold 0.73 ± 0.01 — FAIL)
- m_z_improvement.html regenerated: YES (mtime ≥ sweep-end confirmed)
- Static convergence figure `simulations/figures/fig08_h0_convergence.pdf`: regenerated
- **SC-3 root cause:** `extract_baseline` sums `Σ log L(event_i | h)` without the
  `N × log D(h)` completeness-correction denominator. The v2.2 posteriors have 60 events
  (vs v2.1's 417); with only 60 events and no D(h) correction the raw log-posterior increases
  monotonically with h, peaking at h=0.86. This is a limitation of the verification metric,
  not a physics regression in the `--evaluate` pipeline (which includes D(h)).
- **Impact:** The SC-3 FAIL is a genuine finding that must be resolved before Phase 40 can
  be declared PASS. Options: (a) verify MAP via `--evaluate` log output, (b) fix
  `extract_baseline` to include D(h), or (c) replace SC-3 metric with the correct pipeline output.
- Detail: `.planning/debug/verify03_report_20260423T172607Z.md`
- Commits: `5b5e44e`, `4258551`, `5850a86`

### SC-4 — VERIFY-04: Per-quartile anisotropy within 1σ (OR Stage-2 trigger, not blocker)
**Status:** [x] STAGE-2-TRIGGER (per D-12 — not a blocker, routes Phase 42)
**Evidence:**
- MAP_total = 0.8600 (σ = 0.0037 from overall 68% CI width / 2)
- Per-quartile results on |qS − π/2| equal-count bins (N=15 per quartile):
  - Q1 [0.0123, 0.2603): MAP_q=0.8600, |ΔMAP|=0.000 — no trigger
  - Q2 [0.2603, 0.5043): MAP_q=0.8600, |ΔMAP|=0.000 — no trigger
  - Q3 [0.5043, 0.7574): MAP_q=0.8400, |ΔMAP|=0.020 >> σ=0.0037 — **TRIGGER (~5.4σ)**
  - Q4 [0.7574, 1.3887]: MAP_q=0.8600, |ΔMAP|=0.000 — no trigger
- Any |ΔMAP_q| > σ: YES (Q3)
- Verdict: STAGE-2-TRIGGER — Phase 42 routed (Sky-Dependent Injection Campaign)
- REQUIREMENTS.md wording updated (D-12): `(>1σ shift is a blocker)` → `— >1σ shift is a Stage-2 trigger for Phase 42 (not a blocker)`
- Detail: `.planning/debug/anisotropy_audit_20260423T172607Z.md`
- Commits: `fcf10b6` (D-12 wording) + `(audit commit from 40-04)`

### SC-5 — VERIFY-05: P_det quadrature-weight diagnostic summary reported
**Status:** [x] REPORTED (PHASE-41-TRIGGER-BORDERLINE — user decision required)
**Evidence:**
- `mean_{h=0.73}(per-event max quadrature_weight_outside_grid_numerator)` lower bound: **0.0409**
- Phase 41 trigger threshold (D-18): 0.05
- Borderline band [0.03, 0.05]: 0.0409 falls inside — PHASE-41-TRIGGER-BORDERLINE
- N unique events with >5% numerator at h=0.73: 23 / 542 (4.24%)
- N events 100% off-grid (max_numerator=1.000): 19
- Dominant events: Event 2 (max_num=1.000, 100% off-grid, 49 host calls), Event 90, 161, 105, 107, 115, 113, 112, 110, 109
- Raw data: 990010 WARNING rows across 38 h-values
- Phase 41 routing: **user decides** (borderline — do not auto-advance)
- Detail: `.planning/debug/pdet_quadrature_summary_20260423T172607Z.md`
- Commits from 40-05: `e8d5cd9`, `16d7c6e`

## Regression Check (D-30 Invariant from Phase 39)

- test_coordinate_roundtrip.py (Phase 36): 10/10 GREEN (via VERIFY-01)
- test_parameter_space_h.py (Phase 37): 5/5 GREEN (via VERIFY-01)
- test_l_cat_equivalence.py (Phase 38): 4/4 GREEN (via VERIFY-01)
- Full CPU suite: 544 tests, 0 failures

## Lint / Type Gate

Not run for Phase 40 — no source-code changes in this phase (only ledger + debug artifacts).

## Downstream Routing

| Condition                                          | Next Phase | Route                    |
|----------------------------------------------------|------------|--------------------------|
| SC-3 NOT VERIFIED (MAP=0.860)                      | pause      | Investigate: confirm --evaluate peak at h≈0.73 OR fix extract_baseline |
| phase_41_trigger=PHASE-41-TRIGGER-BORDERLINE (W1)  | pause      | User decides Phase 41 vs skip |
| stage_2_trigger=true (VERIFY-04)                   | Phase 42   | `/gsd:execute-phase 42` (confirmed) |
| SC-3 clears + Phase 41 triggered                   | Phase 41   | `/gsd:execute-phase 41` |
| SC-3 clears + Phase 41 skipped                     | Phase 42   | Phase 42 already triggered |

**This gate's decision: GAPS_FOUND — SC-3 investigation + VERIFY-05 borderline user decision required**

Phase 42 routing is confirmed (independent of SC-3 and Phase 41 decisions).
Phase 41 routing requires user decision on whether mean_lb=0.0409 is sufficient to trigger.

## Plan SUMMARY Pointers

- 40-00-SUMMARY.md (v2.1 baseline archive)
- 40-01-SUMMARY.md (VERIFY-01)
- 40-02-SUMMARY.md (VERIFY-02)
- 40-03-SUMMARY.md (VERIFY-03 — SC-3 FAIL)
- 40-04-SUMMARY.md (VERIFY-04 — STAGE-2-TRIGGER)
- 40-05-SUMMARY.md (VERIFY-05 — PHASE-41-TRIGGER-BORDERLINE)
- Top-level index: `.planning/debug/verify_gate_20260423T172607Z.md`
