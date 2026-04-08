---
phase: 35-unified-pipeline-paper-figures
reviewed: 2026-04-08T12:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - master_thesis_code/main.py
  - master_thesis_code/plotting/convergence_plots.py
  - master_thesis_code/plotting/_helpers.py
  - master_thesis_code/plotting/paper_figures.py
  - master_thesis_code_test/plotting/test_helpers.py
  - master_thesis_code_test/plotting/test_paper_figures.py
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 35: Code Review Report

**Reviewed:** 2026-04-08
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Reviewed the unified figure pipeline added in phase 35: `paper_figures.py` (5 figure factory functions plus data loaders), `_helpers.py` (shared `compute_credible_interval` and figure utilities), `convergence_plots.py`, the `generate_figures` manifest additions in `main.py`, and corresponding tests. The code is generally well-structured with consistent factory-function conventions and good graceful degradation when data is missing.

Key concerns: a regex in `_load_per_event_with_mass_scalars` silently drops events whose likelihood values lack a decimal point (e.g. integer-valued entries), and `_select_representative_events` can return duplicate event IDs when the event pool is small. The `open()` calls in data loaders lack explicit encoding, relying on platform defaults.

## Warnings

### WR-01: Regex in _load_per_event_with_mass_scalars drops integer-valued likelihoods

**File:** `master_thesis_code/plotting/paper_figures.py:165`
**Issue:** The regex pattern `r'"(\d+)": \[(\d+\.\d+(?:e[+-]?\d+)?)\]'` requires a decimal point in the numeric value (`\d+\.\d+`). If any aggregated scalar likelihood is an integer (e.g. `"42": [0]` or `"42": [1]`), the match will silently fail and that event will be missing from the result (defaulting to 0.0 via `raw[h].get(eid, 0.0)` at line 186). This is a silent data-loss bug that would produce incorrect figure output without any error or warning.
**Fix:**
```python
pattern = re.compile(r'"(\d+)": \[(\d+(?:\.\d+)?(?:e[+-]?\d+)?)\]')
```

### WR-02: _select_representative_events can return duplicate event IDs

**File:** `master_thesis_code/plotting/paper_figures.py:307-313`
**Issue:** When the number of valid events is small (e.g. fewer than 20), the percentile-based index selection (`n // 20`, `n // 4`, `n // 2`, `int(0.95 * n)`) can map to the same index, producing duplicate event IDs in the returned list. This would cause `plot_single_event_likelihoods` to plot the same event multiple times instead of showing 4 distinct events.
**Fix:** Deduplicate the selection, falling back to adjacent indices when duplicates occur:
```python
candidates = [
    max(1, n // 20),
    n // 4,
    n // 2,
    int(0.95 * n),
]
seen: set[int] = set()
selected: list[str] = []
for idx in candidates:
    while idx in seen and idx < n - 1:
        idx += 1
    seen.add(idx)
    selected.append(stats[idx][0])
return selected
```

### WR-03: Missing signal.alarm(0) cancellation in data_simulation exception handlers

**File:** `master_thesis_code/main.py:412-457`
**Issue:** In the `data_simulation` function, when the first `signal.alarm(90)` fires and an exception other than `TimeoutError` is caught (e.g. `ParameterOutOfBoundsError`, `AssertionError`, `RuntimeError`, etc. at lines 421-453), the alarm is not cancelled. The alarm continues to count down, and if the next iteration's computation happens to be fast, the stale alarm can fire during the *following* event's computation, causing a spurious timeout. By contrast, the `injection_campaign` function correctly calls `signal.alarm(0)` in every exception handler.
**Fix:** Add `signal.alarm(0)` at the top of each exception handler in `data_simulation`, or restructure with a `try/finally` that always cancels the alarm:
```python
try:
    signal.alarm(90)
    # ... computation ...
    signal.alarm(0)
except TimeoutError:
    ...
except ParameterOutOfBoundsError as e:
    signal.alarm(0)  # cancel stale alarm
    ...
```

### WR-04: open() calls without explicit encoding in data loaders

**File:** `master_thesis_code/plotting/paper_figures.py:97,121,171`
**Issue:** `open(path)` at line 97 and `open(base / f)` at line 121 rely on `locale.getpreferredencoding()` for text decoding. On some systems (especially cluster nodes with minimal locale configuration), this may not be UTF-8, causing `UnicodeDecodeError` on JSON files containing non-ASCII characters. Line 171 opens in binary mode and manually decodes as UTF-8, which is correct.
**Fix:**
```python
with open(path, encoding="utf-8") as f:
```

## Info

### IN-01: Hardcoded truth value 0.73 repeated across paper figure functions

**File:** `master_thesis_code/plotting/paper_figures.py:255,377-378,800`
**Issue:** The injected Hubble constant value `0.73` appears as a magic number in `plot_h0_posterior_comparison` (line 255), `plot_single_event_likelihoods` (lines 377-378), and `plot_h0_posterior_kde` (line 800). While this matches `constants.py:H=0.73`, importing the constant would prevent silent divergence if the injection value ever changes.
**Fix:** Import and use `from master_thesis_code.constants import H` or accept `true_h` as a parameter (as `plot_h0_convergence` already does).

### IN-02: Identical list-conversion logic in plot_h0_convergence

**File:** `master_thesis_code/plotting/convergence_plots.py:64-68`
**Issue:** Both branches of the `isinstance` check produce `list(event_posteriors)`, making the conditional a no-op. This appears to be leftover from an earlier design where one branch might have done something different.
**Fix:**
```python
posteriors_list: list[npt.NDArray[np.float64]] = list(event_posteriors)
```

### IN-03: Unused import `math` could be replaced by `np.isnan` in test file

**File:** `master_thesis_code_test/plotting/test_helpers.py:3`
**Issue:** `import math` is used only for `math.isnan` at lines 110-111. The rest of the test file uses NumPy. This is a minor style inconsistency.
**Fix:** Use `np.isnan(lo)` and `np.isnan(hi)` instead, and remove the `math` import. Alternatively, keep as-is since `math.isnan` works fine on Python floats.

---

_Reviewed: 2026-04-08_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
