# Phase 34: Fisher Matrix Quality - Research

**Researched:** 2026-04-09
**Domain:** Numerical linear algebra — covariance matrix conditioning, exclusion policy, diagnostic CSV/plot, CLI extension
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Check condition number of both 3D and 4D covariance matrices at evaluation time, in `bayesian_statistics.py` where `pinv()` is currently called (~lines 427, 433).

**D-02:** Use `np.linalg.cond()` as the detection metric — standard, easy to threshold, already used upstream in `parameter_estimation.py:391`.

**D-03:** Check 3D and 4D independently. An event is flagged if either matrix exceeds the threshold.

**D-04:** Exclude flagged events from the posterior likelihood product entirely. Do not regularize or downweight — degenerate events should not contribute unreliable likelihoods.

**D-05:** The primary goal is to understand *why* singularity occurs (it shouldn't physically). Exclusion is the safe default while investigating the root cause.

**D-06:** Generate a two-panel diagnostic plot for all flagged events every evaluation run:
- Panel 1: Eigenvalue spectrum (bar chart per flagged event, shows which direction is degenerate)
- Panel 2: Parameter scatter of flagged events in (d_L, SNR, M) space (shows correlation with physical parameters)

**D-07:** Debug plot is always generated (not opt-in). Cheap to produce since it only covers flagged events. Ensures regressions are never missed.

**D-08:** Determine threshold empirically from the current data — run evaluation, collect all condition numbers, identify the gap between well-conditioned and degenerate events.

**D-09:** Make threshold configurable via `--fisher_cond_threshold` CLI flag with the empirically determined default. Follows Phase 33 pattern (`--pdet_dl_bins`).

**D-10:** Start with same threshold for both 3D and 4D matrices. If empirical data shows they need different thresholds, deviate to separate values (but start unified).

**D-11:** Per-run log summary: total events, flagged count, excluded count, top-5 worst condition numbers. INFO level.

**D-12:** Write `fisher_quality.csv` alongside posteriors with columns: detection_index, cond_3d, cond_4d, excluded (bool). Used as input for the debug plot and post-hoc analysis.

**D-13:** Add a "Fisher Quality" section to the Phase 30 comparison report showing: events excluded before vs after, condition number distribution shift, impact on MAP h.

### Claude's Discretion

- Exact eigenvalue visualization style (grouped bars, stacked, or per-event subplots)
- Whether to use `apply_style()` theming on the debug plot or keep it plain diagnostic style
- Module placement for Fisher quality utilities (inline in `bayesian_statistics.py` vs separate module)
- How to compute empirical threshold (gap detection, percentile, or manual inspection)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FISH-01 | Degenerate Fisher matrices detected and handled (regularization or exclusion) instead of `allow_singular=True` | Confirmed: pinv() at lines 427/433 is the target; exclusion-via-mask pattern is well-supported by existing pre-allocation structure |
| FISH-02 | Events with near-singular covariance flagged in diagnostic output with condition number | Confirmed: `np.linalg.cond()` already present at `parameter_estimation.py:391`; CSV + log pattern matches existing diagnostics convention |
</phase_requirements>

---

## Summary

Phase 34 replaces the implicit `pinv()` / `allow_singular=True` approach to near-singular covariance matrices in `BayesianStatistics.__init__()` with an explicit condition-number gate. Events whose 3D or 4D covariance matrix exceeds a configurable threshold are excluded from the posterior likelihood product and written to a diagnostic CSV. A two-panel debug plot is always generated to help identify the physical origin of singularity.

The implementation is concentrated in three files: `bayesian_statistics.py` (condition-number check and exclusion mask in the Gaussian pre-computation loop), `arguments.py` (new `--fisher_cond_threshold` CLI flag), and `evaluation_report.py` (Fisher Quality section in the comparison report). A new diagnostic plot function (file to be determined — inline or separate module per Claude's discretion) produces the eigenvalue-spectrum and parameter-scatter panels.

The prerequisite sub-task is an empirical threshold calibration run: execute evaluation on the current dataset, collect all condition numbers from the new CSV, inspect the distribution, and set the default threshold where the gap between well-conditioned and degenerate events is largest.

**Primary recommendation:** Implement the condition-number gate as an in-loop mask computed during the existing `for index, row in self.cramer_rao_bounds.iterrows()` loop at `bayesian_statistics.py:363`. Store excluded slots in a boolean array `_excluded_mask` (shape `(n_det,)`) that propagates cleanly to `child_process_init` and causes workers to skip those detection slots.

---

## Standard Stack

### Core (all already in project dependencies)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | project-pinned | `np.linalg.cond()`, `np.linalg.eigh()` for eigenvalue decomposition | Already used everywhere; `cond()` already called upstream |
| pandas | project-pinned | DataFrame for `fisher_quality.csv` output via `.to_csv()` | Existing CSV output pattern in diagnostics |
| matplotlib | project-pinned | Two-panel diagnostic plot | Existing plotting infrastructure |

No new dependencies are required. [VERIFIED: codebase grep]

### Existing Project Utilities (reuse, do not reinvent)

| Utility | Location | Purpose in Phase 34 |
|---------|----------|---------------------|
| `np.linalg.cond()` | already at `parameter_estimation.py:391` | Same call for covariance matrices at `bayesian_statistics.py:427/433` |
| `np.linalg.eigh()` | numpy standard | Eigenvalue decomposition for Panel 1 of debug plot; use `eigh` (not `eig`) since covariance is symmetric |
| `np.linalg.slogdet()` | already at lines 428/434 | Sign already checked; reuse for free |
| `apply_style()` | `plotting/_style.py` | Theming (Claude's discretion whether to apply to diagnostic plot) |
| `save_figure()` | `plotting/_helpers.py` | File output with parent-dir creation |
| `get_figure()` | `plotting/_helpers.py` | Figure/axes creation with optional preset sizing |
| `generate_comparison_report()` | `evaluation_report.py` | Extend with Fisher Quality section |

---

## Architecture Patterns

### Recommended Project Structure (no new directories needed)

```
master_thesis_code/
├── bayesian_inference/
│   ├── bayesian_statistics.py   # condition-number check + exclusion mask (main changes)
│   ├── evaluation_report.py     # add Fisher Quality section
│   └── fisher_diagnostics.py   # [optional] separate module for plot + CSV logic
├── plotting/
│   └── (fisher_diagnostics plot can live here if separate module chosen)
master_thesis_code_test/
└── bayesian_inference/
    └── test_fisher_quality.py   # new test file for condition-number gate logic
```

### Pattern 1: In-Loop Condition-Number Gate with Exclusion Mask

The existing Gaussian pre-computation loop at `bayesian_statistics.py:363` already builds per-slot arrays. The cleanest implementation adds condition-number computation inside this loop and records exclusions in a boolean mask:

```python
# Source: codebase reading, bayesian_statistics.py:363-444 pattern
_excluded_mask = np.zeros(n_det, dtype=bool)  # pre-allocate alongside existing arrays
_cond_3d = np.zeros(n_det)  # for fisher_quality.csv
_cond_4d = np.zeros(n_det)

for index, row in self.cramer_rao_bounds.iterrows():
    det = Detection(row)
    slot = _det_index_to_slot[int(index)]

    # ... existing covariance construction (unchanged) ...

    cond_3d = float(np.linalg.cond(cov_3d))
    cond_4d = float(np.linalg.cond(cov_4d))
    _cond_3d[slot] = cond_3d
    _cond_4d[slot] = cond_4d

    if cond_3d > fisher_cond_threshold or cond_4d > fisher_cond_threshold:
        _excluded_mask[slot] = True
        # leave _cov_inv_3d[slot], _log_norm_3d[slot], etc. at zero (safe sentinel)
        continue  # skip pinv computation for this slot

    _cov_inv_3d[slot] = np.linalg.pinv(cov_3d)
    _sign, logdet = np.linalg.slogdet(cov_3d)
    _log_norm_3d[slot] = -0.5 * (3 * np.log(2 * np.pi) + logdet)
    # ... rest of existing computation ...
```

**Note on zero sentinels:** When a slot is excluded, `_cov_inv_3d[slot]` stays at the zero-initialized value. Workers must check `_excluded_mask` before reading these arrays. The mask must be passed to `child_process_init` alongside the other arrays.

### Pattern 2: Mask Propagation to Workers

The existing `child_process_init` (line 1487) receives ~17 arrays as positional args. Add `_excluded_mask` as a new positional parameter:

```python
# Source: bayesian_statistics.py:1487-1544 pattern
def child_process_init(
    ...
    current_excluded_mask: npt.NDArray[np.bool_],
    current_D_h_table: dict[float, float] | None = None,
) -> None:
    global excluded_mask
    ...
    excluded_mask = current_excluded_mask
```

Worker functions (`single_host_likelihood`) must check `excluded_mask[slot]` and return `(0.0, 0.0)` or skip-sentinel for excluded slots. Alternatively, excluded detection indices can be removed from the work queue before `pool.starmap()` — cleaner, avoids mask lookup in hot path.

**Recommendation (preferred):** Filter excluded indices from the work list before pool dispatch. This is cleaner than checking a mask inside the worker. The existing `det_indices` list is already built from `self.cramer_rao_bounds.index` — simply filter it:

```python
active_det_indices = [
    idx for idx, slot in _det_index_to_slot.items()
    if not _excluded_mask[slot]
]
# pass active_det_indices to pool.starmap instead of all det_indices
```

### Pattern 3: fisher_quality.csv Output

Follows the existing diagnostic CSV pattern (`_diagnostic_rows` / pandas `.to_csv()`):

```python
# Source: evaluation_report.py and bayesian_statistics.py diagnostic patterns
import pandas as pd
fisher_quality_rows = [
    {
        "detection_index": int(idx),
        "cond_3d": float(_cond_3d[slot]),
        "cond_4d": float(_cond_4d[slot]),
        "excluded": bool(_excluded_mask[slot]),
    }
    for idx, slot in _det_index_to_slot.items()
]
pd.DataFrame(fisher_quality_rows).to_csv(
    os.path.join(output_dir, "fisher_quality.csv"), index=False
)
```

### Pattern 4: Two-Panel Debug Plot

Panel 1 (eigenvalue spectrum) uses `np.linalg.eigh()` — appropriate for symmetric positive semi-definite matrices. Panel 2 (parameter scatter) uses per-event d_L, SNR, M scalars already stored in `_det_d_L`, `_det_M` arrays.

```python
# Source: plotting/_helpers.py get_figure() and save_figure() pattern
fig, axes = get_figure(nrows=1, ncols=2, preset="double")
ax_eigen, ax_scatter = axes

# Panel 1: eigenvalue spectrum for each flagged event
for slot in flagged_slots:
    eigenvalues_3d = np.linalg.eigh(cov_3d_arr[slot])[0]  # sorted ascending
    eigenvalues_4d = np.linalg.eigh(cov_4d_arr[slot])[0]
    # bar chart: position = parameter index, height = eigenvalue magnitude
    ax_eigen.bar(...)  # style: Claude's discretion

# Panel 2: scatter of flagged events in (d_L, SNR, M) space
ax_scatter.scatter(flagged_d_L, flagged_SNR, c=flagged_M, ...)
ax_scatter.set_xlabel("d_L")
ax_scatter.set_ylabel("SNR")

save_figure(fig, os.path.join(output_dir, "fisher_quality_diagnostic"))
```

**Note on storing covariance matrices for plot:** The covariance arrays `cov_3d` and `cov_4d` are currently local loop variables. To plot eigenvalues post-loop, either (a) stash them in an `(n_det, 3, 3)` and `(n_det, 4, 4)` array for flagged slots only, or (b) compute eigenvalues inside the loop and store `_eigen_3d` / `_eigen_4d` arrays. Option (b) is preferred (avoids large memory allocation for the full set of unflagged events).

### Pattern 5: CLI Flag (following Phase 33 `--pdet_dl_bins` pattern)

In `arguments.py`, add as an integer/float property on `Arguments` and as a `parser.add_argument` call:

```python
# Source: arguments.py:120-128 (pdet_dl_bins pattern)
@property
def fisher_cond_threshold(self) -> float:
    """Condition number threshold for flagging near-singular covariance matrices."""
    return float(self._parsed_arguments.fisher_cond_threshold)

# In _parse_arguments():
parser.add_argument(
    "--fisher_cond_threshold",
    type=float,
    default=1e10,  # placeholder — replace with empirically determined value
    help="Condition number threshold above which a covariance matrix is flagged and the event excluded. "
         "Default: 1e10 (empirically determined).",
)
```

The default `1e10` is a placeholder until the empirical calibration sub-task is complete. [ASSUMED — actual default must be set after running evaluation with the new CSV logging]

Thread through `main.py:evaluate()` signature: add `fisher_cond_threshold: float = 1e10` parameter and pass `arguments.fisher_cond_threshold` at the call site in `main()`.

### Pattern 6: Fisher Quality Section in Comparison Report

`evaluation_report.py:generate_comparison_report()` currently builds a Markdown table from `BaselineSnapshot` fields. Add a "Fisher Quality" section by:
1. Extending `BaselineSnapshot` with optional fields: `n_excluded_fisher: int = 0`, `fisher_quality_csv_path: str | None = None`
2. Loading `fisher_quality.csv` from the posteriors directory during `extract_baseline()` if it exists
3. Appending the Fisher Quality section to the Markdown lines in `generate_comparison_report()`

```python
# Markdown section to append
lines += [
    "",
    "## Fisher Quality",
    "",
    f"| Metric | Baseline | Current | Delta |",
    f"|--------|----------|---------|-------|",
    f"| Events excluded (Fisher) | {baseline.n_excluded_fisher} | "
    f"{current.n_excluded_fisher} | {current.n_excluded_fisher - baseline.n_excluded_fisher:+d} |",
]
```

### Anti-Patterns to Avoid

- **Do not check condition number after `pinv()`:** `pinv()` silently handles singular matrices. The check must happen *before* calling `pinv()`, replacing it for excluded slots. [VERIFIED: codebase reading]
- **Do not pass `allow_singular=True` anywhere in the production path:** The testing path at line 1347-1348 also uses `allow_singular=True` — remove or add condition-number guard there too if it covers real events.
- **Do not store full `(n_det, 3, 3)` covariance array for all events just for the plot:** Only flagged events need eigenvalue data for the debug plot. Store eigenvalues during the loop for flagged slots only.
- **Do not use `np.linalg.eig()` for symmetric matrices:** Use `np.linalg.eigh()` — it guarantees real eigenvalues and is faster for symmetric/Hermitian matrices. [VERIFIED: numpy docs assumption — standard practice]
- **Do not hardcode the threshold:** It must come from `fisher_cond_threshold` parameter threaded from CLI, not from a module-level constant.
- **Do not skip the diagnostic plot when zero events are flagged:** Generate an empty/placeholder plot (or a "no degenerate events" annotation) so the output directory is always consistent.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Condition number | Custom ratio of max/min singular values | `np.linalg.cond()` | Already in numpy, consistent with upstream usage at `parameter_estimation.py:391` |
| Eigenvalue decomposition | Manual power iteration | `np.linalg.eigh()` | Symmetric matrix, guaranteed real eigenvalues, numerically stable |
| CSV output | Custom string writing | `pandas.DataFrame.to_csv()` | Existing diagnostic CSV pattern throughout codebase |
| Figure creation/saving | `plt.figure()` + manual `os.makedirs` | `get_figure()` + `save_figure()` | Existing plotting utilities with parent-dir creation and format control |

---

## Common Pitfalls

### Pitfall 1: `pinv()` Does Not Raise on Singular Matrices

**What goes wrong:** `np.linalg.pinv()` never raises — it silently returns a pseudo-inverse for singular inputs. The condition-number check must happen before calling `pinv()`, and `pinv()` must be skipped (or removed) for excluded slots.
**Why it happens:** `pinv()` uses SVD with a tolerance cutoff — it is designed to handle singularity gracefully.
**How to avoid:** Structure as: compute condition number → if exceeds threshold, set mask and `continue`; else call `pinv()`.
**Warning signs:** If excluded events still contribute to posteriors, the exclusion is silently bypassed.

### Pitfall 2: Worker Hot Path Sees Excluded Slots

**What goes wrong:** If excluded slots remain in the work queue, workers compute a likelihood using `cov_inv = 0` (the zero sentinel), producing `exp(-0.5 * x^T 0 x) = 1.0` — a flat likelihood that corrupts the posterior.
**Why it happens:** Zero-initialized arrays pass silently through the `_mvn_pdf` computation without raising.
**How to avoid:** Filter excluded indices from the work list before `pool.starmap()` (preferred) or add a sentinel check in `single_host_likelihood`.
**Warning signs:** Likelihood values of exactly 1.0 for every h-value for a given detection.

### Pitfall 3: `_excluded_mask` Not Pickle-Safe

**What goes wrong:** `child_process_init` receives arrays via `pool.starmap()` / `initializer`. NumPy boolean arrays are pickle-safe, but must be passed as a named argument or correctly positioned in the signature.
**Why it happens:** Signature changes to `child_process_init` require matching changes at every call site.
**How to avoid:** Add `current_excluded_mask: npt.NDArray[np.bool_]` as a new positional parameter before the existing optional `current_D_h_table` parameter, and update all call sites together.
**Warning signs:** `TypeError: child_process_init() takes N positional arguments but M were given`.

### Pitfall 4: Testing Path `allow_singular=True` at Line 1347

**What goes wrong:** The testing/validation path at `bayesian_statistics.py:1347-1348` also constructs `multivariate_normal(..., allow_singular=True)`. If this path runs on events that are excluded in the main path, it produces inconsistent results.
**Why it happens:** The testing path was written independently and duplicates the same singularity workaround.
**How to avoid:** Apply the same exclusion logic or condition-number guard to this path, or verify it only runs on the same active (non-excluded) events.
**Warning signs:** Log shows different event counts between main and testing paths.

### Pitfall 5: Empirical Threshold Must Be Set Before Default Is Committed

**What goes wrong:** If the placeholder `1e10` is committed as the final default, events that are genuinely degenerate may fall below it (condition numbers can be much larger: `1e12`, `1e15`, `inf`). Or the threshold may be too tight and exclude well-conditioned events.
**Why it happens:** Condition number distribution is data-dependent and not known a priori.
**How to avoid:** The empirical calibration sub-task (Wave 0 or early Wave 1) must run `--evaluate` with the CSV logging active, inspect the condition-number distribution, and update the default before the threshold is used in any comparison runs.
**Warning signs:** All events excluded (threshold too low) or zero events excluded but `slogdet` still returns negative sign (threshold too high).

### Pitfall 6: `slogdet` Sign Already Provides Partial Information

**What goes wrong:** The existing `slogdet` at lines 428/434 already detects numerically singular matrices (sign == 0 indicates zero determinant). Relying solely on `cond()` can miss cases where `cond()` is finite but the matrix is numerically not positive definite.
**Why it happens:** `cond()` measures the ratio of extreme singular values; a matrix can have all positive singular values but still fail Cholesky due to floating-point roundoff.
**How to avoid:** After the condition-number check passes, also verify `sign > 0` from `slogdet`. Log a warning if sign <= 0 for an otherwise well-conditioned matrix (indicates a different class of numerical problem).

---

## Code Examples

### Condition-Number Gate (core logic)

```python
# Source: codebase reading, bayesian_statistics.py:363-444 + parameter_estimation.py:391
# Pre-allocate alongside existing arrays
_excluded_mask = np.zeros(n_det, dtype=bool)
_cond_3d = np.zeros(n_det, dtype=np.float64)
_cond_4d = np.zeros(n_det, dtype=np.float64)

for index, row in self.cramer_rao_bounds.iterrows():
    det = Detection(row)
    slot = _det_index_to_slot[int(index)]

    # ... existing covariance construction unchanged ...

    cond_3d = float(np.linalg.cond(cov_3d))
    cond_4d = float(np.linalg.cond(cov_4d))
    _cond_3d[slot] = cond_3d
    _cond_4d[slot] = cond_4d

    if cond_3d > fisher_cond_threshold or cond_4d > fisher_cond_threshold:
        _excluded_mask[slot] = True
        continue  # slots remain zero-initialized (safe sentinel for workers that check mask)

    # Existing pinv / slogdet logic (unchanged for non-excluded events)
    _means_3d[slot] = [det.phi, det.theta, 1]
    _cov_inv_3d[slot] = np.linalg.pinv(cov_3d)
    _sign, logdet = np.linalg.slogdet(cov_3d)
    _log_norm_3d[slot] = -0.5 * (3 * np.log(2 * np.pi) + logdet)
    # ... rest unchanged ...
```

### Log Summary (D-11)

```python
# Source: existing _LOGGER.info pattern in bayesian_statistics.py
n_flagged = int(_excluded_mask.sum())
n_total = n_det
top5_worst = sorted(
    [(int(idx), float(_cond_3d[slot]), float(_cond_4d[slot]))
     for idx, slot in _det_index_to_slot.items()],
    key=lambda t: max(t[1], t[2]),
    reverse=True,
)[:5]
_LOGGER.info(
    "Fisher quality: %d total, %d flagged/excluded (%.1f%%). "
    "Top-5 worst cond: %s",
    n_total,
    n_flagged,
    100 * n_flagged / max(n_total, 1),
    [(idx, f"{c3:.2e}", f"{c4:.2e}") for idx, c3, c4 in top5_worst],
)
```

### Eigenvalue Computation for Debug Plot

```python
# Source: numpy standard; use eigh for symmetric matrices
import numpy as np
# For each flagged slot, compute eigenvalues during the loop:
_eigen_3d: dict[int, npt.NDArray[np.float64]] = {}  # slot -> sorted eigenvalues
_eigen_4d: dict[int, npt.NDArray[np.float64]] = {}

if _excluded_mask[slot]:
    _eigen_3d[slot] = np.linalg.eigh(cov_3d)[0]  # returns (eigenvalues, eigenvectors)
    _eigen_4d[slot] = np.linalg.eigh(cov_4d)[0]
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| `allow_singular=True` + `pinv()` | Explicit condition-number gate + exclusion | This phase |
| No condition-number logging at eval time | `fisher_quality.csv` with cond_3d, cond_4d per event | This phase |
| Silent degenerate events in posterior | Counted, logged, excluded | This phase |

**Relevant upstream context:** `np.linalg.cond()` is already called at `parameter_estimation.py:391` on the 14x14 Fisher matrix before inversion. That check flags events with negative CRB diagonals and raises `ParameterEstimationError` — the same spirit, but at simulation time. Phase 34 applies the same discipline at evaluation time on the smaller 3x3/4x4 projected covariance matrices.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Default threshold placeholder `1e10` will be replaced after empirical calibration run | Standard Stack / CLI Flag | If not replaced: either too tight (excludes good events) or too loose (passes degenerate ones) |
| A2 | Filtering excluded indices from the work list before `pool.starmap()` is cleaner than mask lookup in workers | Architecture Patterns | If wrong: worker-side mask check is also viable but adds per-call overhead |
| A3 | `np.linalg.eigh()` is the correct decomposition for covariance matrices | Code Examples | Covariance is symmetric by construction; `eigh` is correct; risk is negligible |

---

## Open Questions

1. **What is the actual condition-number distribution of the current 531-event dataset?**
   - What we know: Some events produce singular matrices (else `allow_singular=True` would not have been needed).
   - What's unclear: How many events? What condition numbers? Is there a clear gap?
   - Recommendation: Make the empirical calibration run the first task in Wave 1. Add CSV logging with a sentinel threshold (`1e30`) so all condition numbers are collected without exclusion. Inspect distribution, then set final default.

2. **Does the testing path at `bayesian_statistics.py:1347-1348` run on real evaluation events or only in unit tests?**
   - What we know: It exists at line 1347 and uses `allow_singular=True`.
   - What's unclear: Whether it is exercised during `--evaluate` runs (not apparent from reading ~40 lines of context around it).
   - Recommendation: Read the surrounding function context before implementation to determine if this path needs the same condition-number guard or can be left for now.

3. **Should `fisher_quality.csv` be written per-h-value or once per evaluation session?**
   - What we know: Condition numbers are computed once in `__init__()`, not per h-value. The CSV should be written once.
   - What's unclear: Whether the output directory is available in `__init__()` or only in `evaluate()`.
   - Recommendation: Write CSV at end of `evaluate()` after `output_dir` is established. Pass condition-number arrays as instance attributes (`self._cond_3d`, `self._cond_4d`, `self._excluded_mask`) from `__init__()` to `evaluate()`.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 34 is purely code changes within the existing Python environment. No new external tools, services, CLIs, or databases are required.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` |
| Quick run command | `uv run pytest master_thesis_code_test/bayesian_inference/test_fisher_quality.py -x -q` |
| Full suite command | `uv run pytest -m "not gpu and not slow" --tb=short -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| FISH-01 | Excluded events produce zero contribution to posterior (not `1.0`) | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_fisher_quality.py::test_excluded_event_zero_contribution -x` | No — Wave 0 |
| FISH-01 | Active (non-excluded) events still produce correct `_cov_inv` and `_log_norm` | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_fisher_quality.py::test_active_event_unchanged -x` | No — Wave 0 |
| FISH-02 | `fisher_quality.csv` written with correct columns (detection_index, cond_3d, cond_4d, excluded) | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_fisher_quality.py::test_fisher_quality_csv_columns -x` | No — Wave 0 |
| FISH-02 | Events exceeding threshold have `excluded=True` in CSV | unit | `uv run pytest master_thesis_code_test/bayesian_inference/test_fisher_quality.py::test_csv_excluded_flag -x` | No — Wave 0 |
| FISH-01 | `--fisher_cond_threshold` CLI flag parsed and threaded to `BayesianStatistics.evaluate()` | unit | `uv run pytest master_thesis_code_test/test_arguments.py::test_fisher_cond_threshold -x` | No — Wave 0 (extend existing file) |

### Sampling Rate
- **Per task commit:** `uv run pytest master_thesis_code_test/bayesian_inference/test_fisher_quality.py -x -q`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow" --tb=short -q`
- **Phase gate:** Full suite green before verification

### Wave 0 Gaps
- [ ] `master_thesis_code_test/bayesian_inference/test_fisher_quality.py` — covers FISH-01 and FISH-02 unit tests
- [ ] Extend `master_thesis_code_test/test_arguments.py` with `test_fisher_cond_threshold` — covers CLI flag parsing

---

## Security Domain

Step skipped — this phase modifies numerical filtering logic with no authentication, session management, access control, or external input surfaces. No ASVS categories apply.

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase 34 |
|-----------|-------------------|
| All public and private functions must have complete type annotations | New functions (`_check_condition_number`, plot function, CSV writer) need full annotations |
| Use `npt.NDArray[np.float64]` for typed arrays, never bare `np.ndarray` | `_excluded_mask` must be `npt.NDArray[np.bool_]`, condition arrays `npt.NDArray[np.float64]` |
| `field(default_factory=...)` for mutable dataclass defaults | If `BaselineSnapshot` gains new list/dict fields for Fisher quality, wrap in `field()` |
| NumPy-style docstrings for new code | All new functions need Args/Returns/Notes sections |
| `snake_case` for functions, `SCREAMING_SNAKE_CASE` for constants | `fisher_cond_threshold` (parameter name), `_FISHER_COND_THRESHOLD_DEFAULT` if a module constant is needed |
| Pre-commit hooks run ruff + mypy on every commit | Run `/check` before each commit; `np.bool_` annotation requires numpy stubs |
| Physics-change protocol NOT triggered | Phase 34 is a software change — exclusion policy, logging, and CSV output have no effect on computed formula values. The covariance matrices themselves are unchanged. |
| GSD workflow enforcement | Execute via `/gsd:execute-phase 34` |

---

## Sources

### Primary (HIGH confidence)
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` lines 363-462 — exact structure of Gaussian pre-computation loop, `pinv()` call sites, `slogdet` usage [VERIFIED: Read tool]
- `master_thesis_code/parameter_estimation/parameter_estimation.py` lines 388-395 — `np.linalg.cond()` usage pattern [VERIFIED: Read tool]
- `master_thesis_code/arguments.py` — CLI argument structure, `pdet_dl_bins` pattern to follow [VERIFIED: Read tool]
- `master_thesis_code/main.py:728-749` — `evaluate()` function signature and threading pattern [VERIFIED: Read tool]
- `master_thesis_code/bayesian_inference/evaluation_report.py` — `generate_comparison_report()` structure, Markdown generation pattern [VERIFIED: Read tool]
- `master_thesis_code/bayesian_inference/bayesian_statistics.py:1487-1544` — `child_process_init` signature and global variable pattern [VERIFIED: Read tool]

### Secondary (MEDIUM confidence)
- `.gpd/research-map/CONCERNS.md` lines 96-103 — historical context for `allow_singular=True` issue [VERIFIED: Read tool]
- `.gpd/research-map/CONVENTIONS.md` lines 158-163 — Fisher matrix approximation and singularity risk [VERIFIED: Read tool]

### Tertiary (LOW confidence / ASSUMED)
- Default threshold `1e10` as placeholder — [ASSUMED] actual value requires empirical calibration run
- `np.linalg.eigh()` preferred over `eig()` for symmetric matrices — [ASSUMED from training; standard numpy practice, low risk]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in project; no new dependencies
- Architecture: HIGH — implementation sites verified by direct codebase reading
- Pitfalls: HIGH — all identified from direct code inspection (zero-sentinel, worker hot path, slogdet sign)
- CLI threading pattern: HIGH — Phase 33 pattern read directly from arguments.py

**Research date:** 2026-04-09
**Valid until:** 2026-05-09 (stable domain; numpy linalg API does not change)
