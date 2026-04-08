---
phase: 30-baseline-evaluation-infrastructure
reviewed: 2026-04-08T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - master_thesis_code/bayesian_inference/evaluation_report.py
  - master_thesis_code_test/bayesian_inference/test_evaluation_report.py
  - master_thesis_code/arguments.py
  - master_thesis_code/main.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 30: Code Review Report

**Reviewed:** 2026-04-08T00:00:00Z
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Phase 30 introduces `evaluation_report.py`, two CLI flags (`--save_baseline`, `--compare_baseline`),
and their wiring in `main.py`. The code is generally well-structured and well-tested.
Four warnings stand out: an incorrect `n_events` extraction that silently reads the first file
instead of the consistent maximum, a `_save_baseline` that raises unhandled `ValueError`/`FileNotFoundError`
when the posteriors directory is absent, a `from_json` type import inside a method body that
circumvents mypy, and inconsistent duplicate isoformat timestamp generation in `extract_baseline`.
Three lower-priority info items follow.

---

## Warnings

### WR-01: `n_events` always taken from first posterior file — may silently miscount

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:254`

**Issue:** `n_events` is set from `posteriors[0]["n_detections"]`, where `posteriors` is sorted
by ascending `h` value. There is no guarantee that every `h_*.json` file in an h-sweep records
the same number of detection events. If different h-value files contain different event sets (e.g.,
after a partial re-run where some files were regenerated), `n_events` will silently reflect
whichever file has the smallest `h` value, not the actual count for the MAP value or a stable
aggregate. A more robust approach is to verify consistency across files or use the MAP-h file:

```python
# Take n_events from the MAP-h file, not arbitrarily the first
n_events = int(posteriors[map_idx]["n_detections"])
```

If the intent is to assert all files agree, add an assertion and surface a warning on mismatch:
```python
counts = {int(r["n_detections"]) for r in posteriors}
if len(counts) > 1:
    _LOGGER.warning("Inconsistent n_detections across h files: %s. Using MAP-h count.", counts)
n_events = int(posteriors[map_idx]["n_detections"])
```

---

### WR-02: `_save_baseline` in `main.py` propagates `ValueError`/`FileNotFoundError` to the user without context

**File:** `master_thesis_code/main.py:136-156`

**Issue:** `_save_baseline` calls `extract_baseline(posteriors_dir=...)` without checking whether
`posteriors_dir` exists first. If the user runs `--save_baseline` before an h-sweep has been
executed, `load_posteriors` calls `posteriors_dir.glob(...)` on a non-existent path, which raises
`FileNotFoundError` on some platforms (Python `Path.glob` on a non-existent directory raises
`OSError` on Windows; on Linux it returns an empty iterator, which then raises the `ValueError`
from `extract_baseline`). Either path produces an unformatted traceback instead of an actionable
error message. The fix is a pre-flight existence check:

```python
def _save_baseline(working_directory: str) -> None:
    from pathlib import Path
    from master_thesis_code.bayesian_inference.evaluation_report import extract_baseline

    posteriors_dir = Path(working_directory) / "simulations" / "posteriors"
    if not posteriors_dir.is_dir():
        _ROOT_LOGGER.error(
            "--save_baseline requires a posteriors directory at %s. "
            "Run a full h-sweep (--evaluate with multiple --h_value calls) first.",
            posteriors_dir,
        )
        return
    ...
```

The same applies to `_compare_baseline` at line 173 for the same `posteriors_dir`.

---

### WR-03: `from_json` imports `Any` inside method body — bypasses mypy

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:96-98`

**Issue:**
```python
@classmethod
def from_json(cls, data: dict[str, object]) -> "BaselineSnapshot":
    from typing import Any
    d: dict[str, Any] = data
```

The local `from typing import Any` is used solely to re-annotate `data` as `dict[str, Any]`
to silence mypy complaints about subscript access on `object` values. This is a workaround that
hides a genuine type narrowing gap. Mypy cannot see the annotation because `Any` is imported
inside the function. The proper fix is to declare the parameter as `dict[str, Any]` directly:

```python
# At module top level (already has: from dataclasses import dataclass, field)
from typing import Any  # add this

@classmethod
def from_json(cls, data: dict[str, Any]) -> "BaselineSnapshot":
    return cls(
        map_h=float(data["map_h"]),
        ...
    )
```

This lets mypy correctly type-check the entire method rather than silently widening to `Any`.

---

### WR-04: `extract_baseline` constructs timestamp twice — `created_at` field is always overwritten

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:261-273`

**Issue:** `BaselineSnapshot` has a `default_factory` for `created_at` that generates an
ISO-8601 timestamp. However, `extract_baseline` explicitly re-generates the timestamp at line 271
and passes it to the constructor, effectively ignoring the default and running
`datetime.datetime.now(datetime.UTC)` twice (once via the default factory if the field were not
overridden, but here it is overridden). While not a bug per se, the pattern is inconsistent —
the constructor is called with `created_at=...` and `git_commit=...` explicitly, making the
`default_factory` declarations on those fields misleading. They appear to be self-contained
defaults but are always externally supplied by `extract_baseline`.

Consider either:
1. Removing the `default_factory` for `created_at` and `git_commit` (use `field(default="")`)
   and documenting that callers are expected to populate them, or
2. Removing the explicit `created_at=` and `git_commit=` arguments from the `extract_baseline`
   call and relying on the defaults — which is cleaner and avoids the dual generation.

Option 2 is preferred for minimising call-site boilerplate:
```python
# In extract_baseline — remove these two lines:
#     created_at=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
#     git_commit=_get_git_commit_safe(),
# The dataclass default_factory already handles them.
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
)
```

---

## Info

### IN-01: `_extract_per_event_summaries` iterates with `df.iterrows()` — consider vectorised extraction

**File:** `master_thesis_code/bayesian_inference/evaluation_report.py:291`

**Issue:** `df.iterrows()` is the slowest pandas row iteration method. For the sizes expected
(tens to hundreds of rows) this is not a performance issue, but the pattern is flagged as a
code-quality note because the project conventions call out vectorised operations for array code.
The columns of interest are all numeric; a vectorised extraction is both idiomatic and avoids
the pandas `Series.index` attribute used on line 295:

```python
keep = [c for c in ["d_L", "SNR", "sigma_d_L_over_d_L", "condition_number", "quality_pass"]
        if c in df.columns]
return df[keep].to_dict(orient="records")
```

Note: the return type would then be `list[dict[str, float]]` only if `quality_pass` is numeric
(boolean stored as 0/1), which should be verified against the CSV schema.

---

### IN-02: `arguments.py` — `validate()` stores `_simulation_steps` as instance attribute but never uses it

**File:** `master_thesis_code/arguments.py:149`

**Issue:** Inside `validate()`:
```python
self._simulation_steps = int(self._parsed_arguments.simulation_steps)
```
This assigns a new instance attribute `_simulation_steps` that shadows nothing and is never
read again. The `simulation_steps` property already reads from `self._parsed_arguments` directly.
The assignment is dead code left over from a refactor. It should be removed.

---

### IN-03: `_compare_baseline` in `main.py` calls `generate_comparison_report` without a `label` argument — output file name is always `comparison_current.md`

**File:** `master_thesis_code/main.py:184`

**Issue:**
```python
report_path = generate_comparison_report(baseline, current, output_dir)
```
The `label` parameter defaults to `"current"`, so every invocation of `--compare_baseline`
overwrites the same file `comparison_current.md`. If the user runs multiple comparisons
(e.g., after Phase 31 and Phase 32), earlier reports are silently lost. Consider deriving
the label from the baseline file name or a timestamp:

```python
label = Path(baseline_path).stem  # e.g. "baseline" → "comparison_baseline.md"
report_path = generate_comparison_report(baseline, current, output_dir, label=label)
```

---

_Reviewed: 2026-04-08T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
