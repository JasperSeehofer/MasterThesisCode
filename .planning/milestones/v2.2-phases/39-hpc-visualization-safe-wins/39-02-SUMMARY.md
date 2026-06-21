---
phase: 39-hpc-visualization-safe-wins
plan: 02
subsystem: hpc-io
tags: [hpc, io-batching, lustre, regression-test, sigterm-drain]
requirements: [HPC-02]
requirements_addressed: [HPC-02]
dependency-graph:
  requires: [39-01]
  provides: ["batched CRB writes (1 write per 25 detections)", "SIGTERM-drain regression test"]
  affects: [main.py:_sigterm_handler]
tech-stack:
  added: []
  patterns: ["instance-attribute override for back-compat tests"]
key-files:
  created: []
  modified:
    - master_thesis_code/parameter_estimation/parameter_estimation.py
    - master_thesis_code_test/parameter_estimation/parameter_estimation_test.py
key-decisions:
  - "Instance default raised to 25 (D-06); no CLI flag added"
  - "Test factory _make_minimal_pe still pins instance value to 1 — back-compat with existing tests"
  - "SIGTERM drain test exercises auto-flush at row 25 + manual flush of remaining 5 (D-07)"
metrics:
  duration_min: 2.2
  duration_note: "pure agent work; excludes uv sync time"
  tasks_completed: 2
  files_changed: 2
  source_lines_changed: 1
  test_lines_added: 35
  completed_date: "2026-04-22"
commits:
  - 555304f: "test(39-02): HPC-02 add SIGTERM-drain regression for flush_interval=25"
  - 815ac4a: "perf(39-02): HPC-02 batch CRB writes — flush_interval default 1 -> 25"
---

# Phase 39 Plan 02: HPC-02 Batch CRB Writes Summary

**One-liner:** Raised `_crb_flush_interval` default from 1 to 25 to reduce Lustre I/O by ~25x per SLURM array task; added SIGTERM-drain regression test exercising the `main.py:_sigterm_handler` → `flush_pending_results()` contract.

## Outcome

HPC-02 (ROADMAP SC-2) satisfied. CRB writes are now batched in groups of 25 by default. The SIGTERM handler at `main.py:351` continues to drain the buffer on clean SLURM termination, and the new regression test (`test_sigterm_drain_with_flush_interval_25`) proves the drain contract holds. Worst-case data loss on hard crash (SIGKILL/OOM) is 24 buffered rows — accepted per 39-CONTEXT.md D-06 / threat T-39-02-01.

## Changes

### Source change (1 line)

`master_thesis_code/parameter_estimation/parameter_estimation.py:128`

Before:
```python
self._crb_flush_interval: int = 1
```

After:
```python
self._crb_flush_interval: int = 25  # ROADMAP SC-2: batch CRB writes to reduce Lustre I/O
```

### Test addition (35 lines)

`master_thesis_code_test/parameter_estimation/parameter_estimation_test.py` — appended `test_sigterm_drain_with_flush_interval_25`:

- Sets `pe._crb_flush_interval = 25` at instance level
- Calls `save_cramer_rao_bound` 30 times
- Asserts CSV has 25 rows (auto-flush at row 25) and `_crb_buffer` has 5 rows
- Calls `pe.flush_pending_results()` (mirrors SIGTERM handler path)
- Asserts CSV has 30 rows and `_crb_buffer == []`

The existing `test_crb_buffer_auto_flushes_at_interval` (which overrides interval to 2 at instance level) is unchanged and still passes.

## Verification

| Check | Command | Result |
|-------|---------|--------|
| New default present | `rg "_crb_flush_interval: int = 25" master_thesis_code/parameter_estimation/parameter_estimation.py` | 1 match (line 128) |
| Old default removed | `rg "_crb_flush_interval: int = 1$" master_thesis_code/parameter_estimation/parameter_estimation.py` | 0 matches |
| New test exists | `rg "def test_sigterm_drain_with_flush_interval_25" master_thesis_code_test/parameter_estimation/parameter_estimation_test.py` | 1 match |
| New test passes | `uv run pytest .../test_sigterm_drain_with_flush_interval_25 -v` | 1 passed |
| Existing flush tests pass | targeted run of all `test_*flush*` and `test_*crb_buffer*` | 4/4 passed |
| Parameter-estimation CPU suite | `uv run pytest master_thesis_code_test/parameter_estimation/ -m "not gpu"` | 19 passed, 6 deselected |
| Full CPU regression suite | `uv run pytest master_thesis_code_test/ -m "not gpu"` | **536 passed**, 6 skipped, 12 deselected |
| Ruff (source) | `uv run ruff check master_thesis_code/parameter_estimation/parameter_estimation.py` | clean |
| Ruff (test) | `uv run ruff check master_thesis_code_test/parameter_estimation/parameter_estimation_test.py` | clean |
| Mypy (source) | `uv run mypy master_thesis_code/parameter_estimation/parameter_estimation.py` | clean |

**Suite size:** 536 passed (well above the ≥525 floor specified in 39-02-PLAN.md acceptance criteria — Wave 1 / 39-01 added more tests than the baseline of 524 anticipated).

## Success Criteria

- [x] HPC-02 / ROADMAP SC-2: `_crb_flush_interval = 25` at parameter_estimation.py:128
- [x] SIGTERM handler at main.py:351 still calls `flush_pending_results()` (untouched per D-07)
- [x] New regression test verifies batches-of-25 + manual-drain semantics
- [x] Phase 36/37/38 regressions intact (D-30) — full CPU suite green at 536 passed

## Decisions Made

- **D-06 (already locked):** Default value is 25 exactly; no CLI flag exposed. Followed verbatim — the planner already justified this in 39-CONTEXT.md.
- **D-07 (already locked):** SIGTERM handler path validated by unit test, not modified. Followed verbatim.
- **Implementation choice:** Test fixture `_make_minimal_pe` was NOT changed to use 25 — keeps existing tests deterministic (they rely on flush-on-every-call). The new test overrides at instance level. This is the same pattern `test_crb_buffer_auto_flushes_at_interval` uses (overrides to 2).

## Deviations from Plan

None — plan executed exactly as written. Both tasks landed verbatim per the PLAN's task table; verification commands all returned the expected results; no auto-fixes (Rules 1-3) were triggered; no architectural decisions (Rule 4) were needed.

## Performance Impact (HPC)

**Expected (not measured here — observational on cluster per CONTEXT D-06):**

- Lustre I/O write count: 1 per detection → 1 per 25 detections (25× reduction)
- Estimated wall-clock saving: 5–20 s per SLURM array task (depends on per-task detection count and Lustre contention)
- Per-detection memory overhead: ~24 row dicts buffered in worst case (negligible vs simulation memory footprint)

## Risk Tradeoffs

| Risk | Mitigation | Status |
|------|------------|--------|
| Lost rows on hard crash (SIGKILL/OOM) | Up to 24 rows lost out of thousands per SLURM job — accepted per D-06 | Documented in 39-CONTEXT.md threat register |
| Lost rows on clean SLURM timeout (SIGTERM) | `_sigterm_handler` calls `flush_pending_results()` | Verified by `test_sigterm_drain_with_flush_interval_25` |
| Test factory regression | `_make_minimal_pe` still sets instance value to 1 | All 19 CPU parameter-estimation tests still pass |

## Self-Check: PASSED

Files exist:
- FOUND: master_thesis_code/parameter_estimation/parameter_estimation.py (modified line 128)
- FOUND: master_thesis_code_test/parameter_estimation/parameter_estimation_test.py (test appended)

Commits exist:
- FOUND: 555304f (test commit)
- FOUND: 815ac4a (perf commit)
