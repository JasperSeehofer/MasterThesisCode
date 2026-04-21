---
phase: 30-baseline-evaluation-infrastructure
reviewed: 2026-04-08T10:23:08Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - master_thesis_code/arguments.py
  - master_thesis_code/bayesian_inference/evaluation_report.py
  - master_thesis_code/main.py
  - master_thesis_code_test/bayesian_inference/test_evaluation_report.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 30: Code Review Report

**Reviewed:** 2026-04-08T10:23:08Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Phase 30 introduces `evaluation_report.py` (baseline extraction, credible interval computation, comparison reports), two CLI flags (`--save_baseline`, `--compare_baseline`) in `arguments.py`, their wiring in `main.py`, and comprehensive tests. The code is well-structured with good test coverage. Four warnings were identified: silent miscount of `n_events`, missing directory existence checks in CLI wrappers, a type-annotation workaround that bypasses mypy, and redundant timestamp generation. Three info-level items cover dead code, a non-vectorised pandas pattern, and overwrite-prone output file naming.

## Warnings

### WR-01: `n_events` always taken from first posterior file -- may silently miscount

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:254`
**Issue:** `n_events = int(posteriors[0]["n_detections"])` reads from the file with the smallest `h` value (since `posteriors` is sorted by `h`). If different `h_*.json` files have different detection counts (e.g., after a partial re-run), `n_events` will be incorrect without any warning. The MAP-h file is the natural choice, since that is the file corresponding to the reported MAP value.
**Fix:**
```python
# Use the MAP-h file instead of the first file
n_events = int(posteriors[map_idx]["n_detections"])

# Optionally warn on inconsistency
counts = {int(r["n_detections"]) for r in posteriors}
if len(counts) > 1:
    _LOGGER.warning("Inconsistent n_detections across h files: %s. Using MAP-h count.", counts)
```

### WR-02: `_save_baseline` and `_compare_baseline` raise unhandled errors when posteriors directory is absent

**File:** `master_thesis_code/main.py:136` and `master_thesis_code/main.py:173`
**Issue:** Neither `_save_baseline` nor `_compare_baseline` checks whether `posteriors_dir` exists before calling `extract_baseline`. If the user runs `--save_baseline` before an h-sweep has been executed, `posteriors_dir.glob(...)` returns an empty iterator on Linux, which then hits the `len(posteriors) < 3` check in `extract_baseline` and raises a `ValueError` with a traceback. On Windows, `Path.glob` on a non-existent directory raises `OSError`. Both produce poor user experience. A pre-flight check with an actionable log message is better.
**Fix:**
```python
def _save_baseline(working_directory: str) -> None:
    from pathlib import Path
    from master_thesis_code.bayesian_inference.evaluation_report import extract_baseline

    posteriors_dir = Path(working_directory) / "simulations" / "posteriors"
    if not posteriors_dir.is_dir():
        _ROOT_LOGGER.error(
            "--save_baseline requires posteriors at %s. Run an h-sweep first.",
            posteriors_dir,
        )
        return
    # ... rest unchanged
```

Apply the same pattern to `_compare_baseline` at line 173.

### WR-03: `from_json` imports `Any` inside method body -- bypasses mypy visibility

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:96-98`
**Issue:**
```python
from typing import Any
d: dict[str, Any] = data
```
The local import of `Any` inside the method is a workaround to silence mypy about subscript access on `object` values. Mypy may not fully resolve the annotation since `Any` is imported at function scope. The correct approach is to declare the parameter type directly or import `Any` at module level.
**Fix:**
```python
# At module top level, add:
from typing import Any

# Then change the method signature:
@classmethod
def from_json(cls, data: dict[str, Any]) -> "BaselineSnapshot":
    return cls(
        map_h=float(data["map_h"]),
        # ... no need for re-annotation trick
    )
```

### WR-04: `extract_baseline` redundantly generates timestamp and git commit -- default_factory is misleading

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:271-272`
**Issue:** `BaselineSnapshot` defines `default_factory` for `created_at` and `git_commit`, but `extract_baseline` explicitly passes both fields at lines 271-272, making the defaults dead code. This is not a bug, but makes the dataclass contract misleading -- the defaults suggest the fields are self-populating, while the only real call site always overrides them.
**Fix:** Remove the explicit arguments from `extract_baseline` and rely on the defaults:
```python
return BaselineSnapshot(
    map_h=map_h,
    ci_lower=ci_lower,
    ci_upper=ci_upper,
    ci_width=ci_width,
    bias_percent=bias_percent,
    n_events=n_events,
    h_values=h_values,
    log_posteriors=log_posts,
    per_event_summaries=per_event_summaries,
    # created_at and git_commit use default_factory
)
```

## Info

### IN-01: `validate()` stores `_simulation_steps` as dead instance attribute

**File:** `master_thesis_code/arguments.py:149`
**Issue:** `self._simulation_steps = int(self._parsed_arguments.simulation_steps)` assigns a new instance attribute that is never read. The `simulation_steps` property reads from `self._parsed_arguments` directly. This is dead code.
**Fix:** Remove the assignment. The `try/except ValueError` can still raise `ArgumentsError` without storing the result:
```python
try:
    int(self._parsed_arguments.simulation_steps)
except ValueError as original_error:
    raise ArgumentsError(...) from original_error
```

### IN-02: `_extract_per_event_summaries` uses `df.iterrows()` -- could use vectorised extraction

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:291`
**Issue:** `df.iterrows()` is the slowest pandas row-iteration pattern. For the expected sizes (tens to hundreds of rows) this has no practical impact, but the vectorised alternative is both shorter and idiomatic:
**Fix:**
```python
keep = [c for c in ["d_L", "SNR", "sigma_d_L_over_d_L", "condition_number", "quality_pass"]
        if c in df.columns]
return df[keep].to_dict(orient="records")
```

### IN-03: `_compare_baseline` output always overwrites `comparison_current.md`

**File:** `master_thesis_code/main.py:184`
**Issue:** `generate_comparison_report(baseline, current, output_dir)` uses the default `label="current"`, so every invocation writes to `comparison_current.md`. Multiple sequential comparisons (e.g., after Phase 31 and Phase 32 fixes) silently overwrite earlier reports.
**Fix:** Derive the label from the baseline file name or include a timestamp:
```python
label = Path(baseline_path).stem  # e.g. "baseline" -> comparison_baseline.md
report_path = generate_comparison_report(baseline, current, output_dir, label=label)
```

---

_Reviewed: 2026-04-08T10:23:08Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
