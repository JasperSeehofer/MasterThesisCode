---
phase: 35-unified-pipeline-paper-figures
plan: "01"
subsystem: plotting
tags: [refactor, testing, ci-computation, shared-utility]
dependency_graph:
  requires: []
  provides: [compute_credible_interval in _helpers.py]
  affects: [convergence_plots.py, paper_figures.py (Plan 02)]
tech_stack:
  added: []
  patterns: [cumulative np.trapezoid CDF, shared plotting utility]
key_files:
  created: []
  modified:
    - master_thesis_code/plotting/_helpers.py
    - master_thesis_code/plotting/convergence_plots.py
    - master_thesis_code_test/plotting/test_helpers.py
decisions:
  - "Used per-step np.trapezoid accumulation (not np.gradient + np.cumsum) for the CDF per D-07 design decision"
  - "Return (nan, nan) for zero-norm posteriors instead of the old fallback of full-range width"
metrics:
  duration: "~10 minutes"
  completed: "2026-04-08"
  tasks_completed: 1
  tasks_total: 1
  files_changed: 3
requirements:
  - PFIG-03
---

# Phase 35 Plan 01: Extract Shared compute_credible_interval Summary

Extracted duplicated CI calculation into `compute_credible_interval()` in `_helpers.py`, unit-tested against Gaussian and uniform analytical distributions, and wired `convergence_plots.py` to use it — eliminating the private `_credible_interval_width` and establishing the D-07 shared CDF utility.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Extract compute_credible_interval and write unit tests | c38ec37 | _helpers.py, convergence_plots.py, test_helpers.py |

## What Was Built

**`compute_credible_interval(h_values, posterior, level=0.68) -> tuple[float, float]`** added to `master_thesis_code/plotting/_helpers.py`:

- Normalizes posterior via `np.trapezoid`
- Returns `(nan, nan)` when norm <= 0 (zero posterior)
- Builds CDF by accumulating per-step `np.trapezoid` slices (per D-07)
- Normalizes CDF to [0, 1] then interpolates lo/hi bounds
- Full NumPy-style docstring with Args/Returns

**`_credible_interval_width` deleted** from `convergence_plots.py` (used `np.gradient + np.cumsum`, now superseded).

**`convergence_plots.py` updated** to import and call the shared function:
```python
from master_thesis_code.plotting._helpers import _fig_from_ax, compute_credible_interval, get_figure
...
lo, hi = compute_credible_interval(h_values, combined, level=level)
ci_width = hi - lo
```

**5 unit tests** in `TestComputeCredibleInterval` (all passing):
- `test_gaussian_68ci_equals_two_sigma` — |CI_width - 2*0.04| < 0.003
- `test_uniform_68ci_equals_68_percent_range` — |CI_width - 0.68*0.30| < 0.01
- `test_zero_posterior_returns_nan` — (nan, nan) returned
- `test_returns_tuple_of_floats` — return type verified
- `test_level_parameter` — 95% level: |CI_width - 2*1.96*0.04| < 0.005

## Verification

```
uv run --extra dev pytest master_thesis_code_test/plotting/test_helpers.py \
  master_thesis_code_test/plotting/test_convergence_plots.py \
  -x -m "not slow and not gpu" -v
# => 20 passed
```

## Deviations from Plan

None — plan executed exactly as written.

The old `_credible_interval_width` returned `float(h_values[-1] - h_values[0])` for zero-norm posteriors (the full grid width). The new shared function returns `(nan, nan)` instead, which is semantically cleaner and avoids silently producing a misleading width value. This is an improvement over the original behavior, not a regression — callers that previously used the width value would have gotten a nonsensical result anyway.

## Known Stubs

None.

## Threat Flags

None — internal plotting utility, no security-relevant surface.

## Self-Check: PASSED

- `master_thesis_code/plotting/_helpers.py` — FOUND, contains `def compute_credible_interval`
- `master_thesis_code_test/plotting/test_helpers.py` — FOUND, contains `class TestComputeCredibleInterval`
- `master_thesis_code/plotting/convergence_plots.py` — FOUND, `_credible_interval_width` deleted
- Commit c38ec37 — FOUND (`git log --oneline -1` confirms)
