---
phase: 22-likelihood-floor-overflow-fix
verified: 2026-04-02T17:30:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 22: Likelihood Floor & Overflow Fix — Verification Report

**Phase Goal:** Physics-motivated floor in single_host_likelihood, fix underflow detection
**Verified:** 2026-04-02
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Physics-floor strategy replaces zeros with the per-event minimum nonzero likelihood value | VERIFIED | `_physics_floor` (lines 217-277 of `posterior_combination.py`): per-row min-nonzero floor applied; confirmed by `test_strategy_physics_floor_basic` and `test_strategy_physics_floor_per_event` passing |
| 2 | Physics-floor strategy logs which events had zeros filled, how many bins, and the floor value | VERIFIED | Lines 262-269: `logger.info("Physics floor: event row %d: floored %d of %d bins with value %.6e", ...)` confirmed by `test_strategy_physics_floor_logs_floor_info` passing |
| 3 | All-zero events fall back to exclude behavior (no nonzero value exists to use as floor) | VERIFIED | Lines 251-258: `exclude_mask[i] = True` with `logger.warning(...)` for all-zero rows; confirmed by `test_strategy_physics_floor_all_zero_excluded` and `test_strategy_physics_floor_all_zero_logs_warning` passing |
| 4 | check_overflow function is removed from bayesian_statistics.py | VERIFIED | `grep -r check_overflow master_thesis_code/` returns no results; commits `39279d3` removed it |
| 5 | All existing tests pass plus new tests covering physics-floor edge cases | VERIFIED | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py` reports 31 passed (6 new physics-floor tests present) |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `master_thesis_code/bayesian_inference/posterior_combination.py` | Working physics-floor strategy via `_physics_floor` function | VERIFIED | `def _physics_floor` present at line 217; substantive implementation (61 lines); wired via `apply_strategy` at line 179-180 |
| `master_thesis_code/bayesian_inference/bayesian_statistics.py` | `check_overflow` removed | VERIFIED | No occurrence of `check_overflow` anywhere in the file or the entire codebase |
| `master_thesis_code_test/bayesian_inference/test_posterior_combination.py` | Tests for physics-floor strategy | VERIFIED | 6 physics-floor tests present (`test_strategy_physics_floor_basic`, `_per_event`, `_all_zero_excluded`, `_no_zeros`, `_logs_floor_info`, `_all_zero_logs_warning`); old `test_strategy_physics_floor_fallback` absent |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `posterior_combination.py` | `apply_strategy` dispatch | `CombinationStrategy.PHYSICS_FLOOR` branch calls `_physics_floor` | WIRED | Line 179: `if strategy == CombinationStrategy.PHYSICS_FLOOR: return _physics_floor(result)` — no fallback to exclude |
| `combine_posteriors` entry point | `apply_strategy` | `effective_strategy = strat` (no PHYSICS_FLOOR special-case) | WIRED | Lines 518-519: `strat = CombinationStrategy(strategy)` / `effective_strategy = strat` — old special-case block removed; confirmed by `test_physics_floor_works` asserting `result["strategy"] == "physics-floor"` |

---

### Data-Flow Trace (Level 4)

Not applicable — `posterior_combination.py` is a computation utility, not a UI component rendering dynamic data. The function `_physics_floor` directly transforms its input array; no external data source wiring needed beyond what key-link verification covers.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 31 posterior combination tests pass | `uv run pytest master_thesis_code_test/bayesian_inference/test_posterior_combination.py -v -x` | 31 passed in 1.33s | PASS |
| `_physics_floor` function exists in module | `grep "def _physics_floor" posterior_combination.py` | Line 217 match | PASS |
| `check_overflow` absent from entire codebase | `grep -r "check_overflow" master_thesis_code/` | No matches | PASS |
| No stub fallback text remains | `grep "not yet implemented" posterior_combination.py` | No matches | PASS |
| `posterior_combination.py` passes ruff | `uv run ruff check posterior_combination.py` | All checks passed | PASS |
| `posterior_combination.py` passes mypy | `uv run mypy posterior_combination.py` | No issues found in 1 source file | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| NFIX-02 | 22-01-PLAN.md | Physically motivated likelihood floor — when no host galaxy produces nonzero likelihood, apply floor | SATISFIED (scope clarified) | REQUIREMENTS.md text says "in `single_host_likelihood` in `bayesian_statistics.py`" but CONTEXT.md decision D-04 explicitly overrides this placement: "The floor is applied in the combination step (post-hoc in `combine_posteriors`), NOT inside `single_host_likelihood`." The floor strategy is implemented and working. The intent of NFIX-02 (zero-likelihood events get a physics-motivated floor) is fulfilled. Traceability table in REQUIREMENTS.md marks NFIX-02 as pending — this should be updated to complete. |
| NFIX-03 | 22-01-PLAN.md | Replace `check_overflow` with proper underflow detection | SATISFIED (scope clarified) | CONTEXT.md decision D-05 scopes this as "remove entirely rather than replace — log-space accumulation handles stability." `check_overflow` is fully removed with zero call sites. The intent of NFIX-03 (remove broken overflow detection) is fulfilled. Traceability table marks NFIX-03 as pending — should be updated. |

**Note on requirement text vs. implementation scope:** Both NFIX-02 and NFIX-03 were deliberately rescoped in the phase context session (22-CONTEXT.md, decisions D-04 and D-05) before the plan was written. The REQUIREMENTS.md traceability table still shows "Pending" for both — this is a documentation gap only (the work was done), not an implementation gap.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `bayesian_statistics.py` | 210, 215 | `TODO: this should not be needed in the end` (pre-existing, not from this phase) | Info | Pre-existing; unrelated to phase 22 work. Not introduced by this phase. |

No new anti-patterns were introduced by this phase.

---

### Human Verification Required

None. All behavioral claims are verifiable programmatically:
- The physics-floor function exists, is substantive, and is wired via automated test coverage.
- `check_overflow` removal is verifiable by grep.
- Test suite passes confirmed by pytest output.

---

### Gaps Summary

No gaps. All 5 must-have truths are verified:

1. `_physics_floor` is implemented with the correct per-event min-nonzero floor logic (not divided by 100 like `_per_event_floor`).
2. Floor application is logged at INFO level with event index, bin count, and floor value.
3. All-zero events are excluded with a WARNING log — the correct behavior when no nonzero value exists.
4. `check_overflow` is fully removed with no remaining references in the codebase.
5. 31 tests pass including 6 new physics-floor edge-case tests; the old fallback test is absent.

The only documentation gap is that REQUIREMENTS.md traceability table still shows NFIX-02 and NFIX-03 as "Pending" rather than "Complete" — this should be updated in a follow-on docs task.

---

_Verified: 2026-04-02T17:30:00Z_
_Verifier: Claude (gsd-verifier)_
