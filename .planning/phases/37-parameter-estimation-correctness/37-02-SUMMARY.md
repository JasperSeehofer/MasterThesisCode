---
phase: 37-parameter-estimation-correctness
plan: "02"
subsystem: parameter-estimation
tags: [physics, h-threading, fisher-matrix, PE-01, partial]
dependency_graph:
  requires: [37-01-SUMMARY.md]
  provides: [PE-01 h-threading complete]
  affects: [parameter_space.py, main.py, test_parameter_space_h.py]
tech_stack:
  added: []
  patterns: [required-kwarg-no-default, h-threading-end-to-end]
key_files:
  created:
    - master_thesis_code_test/test_parameter_space_h.py
  modified:
    - master_thesis_code/datamodels/parameter_space.py
    - master_thesis_code/main.py
    - master_thesis_code_test/datamodels/parameter_space_test.py
decisions:
  - "h required kwarg (no default) in set_host_galaxy_parameters — TypeError on omission satisfies SC-2 (D-01)"
  - "h_value added to data_simulation() signature and threaded from CLI arguments.h_value"
  - "Plan acceptance test ratio spec was inverted (d_L ∝ 1/h, so d_L(h=0.5)/d_L(h=1.0)=2.0 not d_L(h=1.0)/d_L(h=0.5)=2.0) — corrected to match physics (Rule 1)"
  - "PE-02 not implemented — delegated to GPD executor per plan spec"
metrics:
  duration: "~18 minutes"
  completed_date: "2026-04-22"
  tasks_completed: 4
  tasks_total: 5
  files_changed: 4
status: partial
pending: PE-02 per-parameter derivative_epsilon (GPD delegation)
---

# Phase 37 Plan 02: PE-01 h-threading Summary (Partial)

**One-liner:** Thread injected Hubble constant h_inj end-to-end into set_host_galaxy_parameters so Fisher CRBs are computed at the correct luminosity distance for any h_inj, not just H=0.73.

## Status

PE-01 complete (Tasks 2-5). PE-02 pending GPD delegation (Task 6 — orchestrator handles separately).

## What Was Done (PE-01)

`set_host_galaxy_parameters` in `parameter_space.py` previously called `dist(host_galaxy.z)` which used
the hardcoded `H=0.73` default from `physical_relations.py`. For the standard campaign (h_inj=0.73) this
was harmless, but for injection sweeps at other h values the Fisher matrix was computed at the wrong
luminosity distance.

**Changes:**

1. `parameter_space.py:144` — signature changed from `set_host_galaxy_parameters(self, host_galaxy)` to
   `set_host_galaxy_parameters(self, host_galaxy, h: float)` with no default (D-01).
   Body: `dist(host_galaxy.z, h=h)` replaces bare `dist(host_galaxy.z)`.

2. `main.py` — `data_simulation()` gains `h_value: float` parameter; `main()` passes `arguments.h_value`;
   the `set_host_galaxy_parameters` call at line 399 now passes `h=h_value`.
   Injection pipeline workaround comment replaced with factual comment (root cause fixed).

3. `test_parameter_space_h.py` (new) — two tests:
   - `test_set_host_galaxy_parameters_h_ratio`: d_L(h=0.5)/d_L(h=1.0) == 2.0, rtol=1e-10 (SC-1).
   - `test_set_host_galaxy_parameters_requires_h`: TypeError raised when h is omitted (SC-2).

4. `parameter_space_test.py` — two existing tests updated to pass `h=0.73` explicitly.

## Commits

| Hash | Message |
|------|---------|
| 55a6d99 | [PHYSICS] PE-01: thread h_inj into set_host_galaxy_parameters |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan acceptance test ratio was inverted**
- **Found during:** Task 4 (test execution)
- **Issue:** Plan specified `ratio = ps_one.luminosity_distance.value / ps_half.luminosity_distance.value`
  (h=1.0 / h=0.5) should equal 2.0. But d_L ∝ c/H_0 ∝ 1/h (Hogg 1999 Eq. 16), so the actual ratio
  is 0.5 not 2.0. The correct ratio is d_L(h=0.5) / d_L(h=1.0) = 2.0.
- **Fix:** Corrected test to use `ratio = ps_half.luminosity_distance.value / ps_one.luminosity_distance.value`
  with the comment `# d_L ∝ 1/h`.
- **Files modified:** master_thesis_code_test/test_parameter_space_h.py
- **Commit:** 55a6d99 (same commit)

**2. [Rule 2 - Missing correctness] Existing tests broke on required h kwarg**
- **Found during:** Task 2 (signature change)
- **Issue:** Two tests in `parameter_space_test.py` called `set_host_galaxy_parameters(host)` without `h`.
  Adding the required kwarg would break them.
- **Fix:** Updated both tests to pass `h=0.73` explicitly; expected_dist updated to `dist(z, h=0.73)`.
- **Files modified:** master_thesis_code_test/datamodels/parameter_space_test.py
- **Commit:** 55a6d99 (same commit)

**3. [Rule 3 - Blocking] data_simulation() lacked h_value in scope**
- **Found during:** Task 3 (reading main.py)
- **Issue:** The plan said to change the call at "~line 394" to pass `h=h_value`, but `h_value` was not
  in `data_simulation`'s signature. It is available in `main()` as `arguments.h_value`.
- **Fix:** Added `h_value: float` to `data_simulation()` signature and passed `arguments.h_value` at the
  call site in `main()`.
- **Files modified:** master_thesis_code/main.py
- **Commit:** 55a6d99 (same commit)

## Pending: PE-02

Per plan spec, PE-02 (per-parameter derivative_epsilon for 14 EMRI parameters) is delegated to a GPD
executor under `/gpd:quick` with the `/physics-change` protocol citing Vallisneri (2008)
arXiv:gr-qc/0703086. The GSD executor (this agent) does not implement PE-02.

## Verification Results

- SC-1: d_L(h=0.5) / d_L(h=1.0) = 2.0 to rtol=1e-10 — PASS
- SC-2: TypeError raised when h omitted — PASS
- Full suite: 519 passed, 0 failed, 6 skipped (not gpu/slow) — PASS
- Pre-commit hooks (ruff, mypy) on changed files — PASS
- No `dist(host_galaxy.z)` bare call remains in parameter_space.py — PASS

## Self-Check: PASSED

- [x] 55a6d99 exists in git log
- [x] master_thesis_code/datamodels/parameter_space.py modified
- [x] master_thesis_code/main.py modified
- [x] master_thesis_code_test/test_parameter_space_h.py created
- [x] 519 tests pass
