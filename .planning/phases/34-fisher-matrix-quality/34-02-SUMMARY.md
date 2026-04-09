---
phase: 34-fisher-matrix-quality
plan: "02"
subsystem: bayesian-inference
tags: [fisher-matrix, diagnostics, plotting, comparison-report, condition-number]
dependency_graph:
  requires: [34-01]
  provides: [fisher-quality-plot, fisher-quality-comparison-report]
  affects: [bayesian_statistics, evaluation_report, fisher_plots]
tech_stack:
  added: []
  patterns: [two-panel-diagnostic-plot, backward-compat-serialization, deferred-import]
key_files:
  created: []
  modified:
    - master_thesis_code/bayesian_inference/evaluation_report.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
    - master_thesis_code/plotting/fisher_plots.py
    - master_thesis_code_test/bayesian_inference/test_evaluation_report.py
decisions:
  - "plot_fisher_diagnostics() added to existing fisher_plots.py (not a new module) to keep Fisher visualizations co-located"
  - "Panel 1 eigenvalue bars use eigen_3d only (3D covariance) since cond_4d is well-understood to be large due to scale mixing"
  - "allow_singular=True replaced with try/except returning [0.0] to match list[float] return type of the testing function"
  - "Fisher Quality section in comparison report gated on n_excluded_fisher > 0 (either snapshot) to avoid noise in normal runs"
  - "Backward compat in BaselineSnapshot.from_json() via .get() fallbacks — old baseline.json files missing the three new fields load cleanly"
metrics:
  duration_minutes: 25
  completed_date: "2026-04-09"
  tasks_completed: 2
  tasks_total: 3
  files_created: 0
  files_modified: 4
---

# Phase 34 Plan 02: Fisher Quality Plot and Comparison Report Integration Summary

**One-liner:** Two-panel Fisher quality diagnostic plot (`plot_fisher_diagnostics`) wired into every evaluation run, Fisher Quality section added to comparison reports, `allow_singular=True` removed from bayesian_statistics.py.

## What Was Built

Tasks 1 and 2 are complete. Task 3 is a `checkpoint:human-verify` — paused for manual verification.

### Task 1: BaselineSnapshot extension + comparison report Fisher Quality section

**`master_thesis_code/bayesian_inference/evaluation_report.py`**

- **Three new fields on `BaselineSnapshot`**: `n_excluded_fisher: int = 0`, `median_cond_3d: float = 0.0`, `median_cond_4d: float = 0.0` (all optional with defaults for backward compatibility).
- **`to_json()`** updated to include the three new fields.
- **`from_json()`** updated with `.get(..., default)` fallbacks so existing `baseline.json` files that predate these fields deserialize without error.
- **`extract_baseline()`** now checks for `fisher_quality.csv` at `posteriors_dir.parent / "fisher_quality.csv"`. If present, reads `excluded` count and median condition numbers and sets them on the returned snapshot.
- **`generate_comparison_report()`** appends a `## Fisher Quality` section to the Markdown report (and adds `n_excluded_fisher` / `median_cond_*` to the JSON sidecar) when either snapshot has `n_excluded_fisher > 0`.

**`master_thesis_code_test/bayesian_inference/test_evaluation_report.py`**

Eight new tests added:
- `test_baseline_snapshot_fisher_fields_default_zero` — defaults to 0
- `test_baseline_snapshot_fisher_fields_roundtrip` — serialize/deserialize with n_excluded_fisher=5
- `test_baseline_snapshot_fisher_fields_backward_compat` — old JSON without keys loads cleanly
- `test_generate_comparison_report_fisher_quality_section` — section appears with correct delta
- `test_generate_comparison_report_no_fisher_section_when_zero` — section absent when both 0
- `test_generate_comparison_report_json_has_fisher_fields` — JSON sidecar contains all new fields
- `test_extract_baseline_reads_fisher_quality_csv` — reads CSV and sets n_excluded=2
- `test_extract_baseline_zero_fisher_when_no_csv` — defaults to 0 when CSV absent

### Task 2: Two-panel diagnostic plot + allow_singular cleanup + plot call wiring

**`master_thesis_code/plotting/fisher_plots.py`**

- Added `plot_fisher_diagnostics()` (120 lines) to the existing module. Takes `cond_3d`, `cond_4d`, `excluded_mask`, `eigen_3d`, `eigen_4d`, `det_d_L`, `det_M`, `det_index_to_slot`, `threshold`, `output_dir`.
- **Panel 1 (left)**: Eigenvalue spectrum — grouped bars (3 bars per flagged event, one per eigenvalue from `eigen_3d`), log-scale y-axis. If zero flagged events: centered "No degenerate events detected" annotation with no axes.
- **Panel 2 (right)**: Parameter scatter in (d_L, M) space — all events as small gray dots, flagged events as larger colored markers with colormap = log10(max(cond_3d, cond_4d)), colorbar labeled. If zero flagged: annotation "No flagged events".
- Uses `apply_style()`, `get_figure(nrows=1, ncols=2, preset="double")`, `save_figure()` to `fisher_quality_diagnostic.pdf`.

**`master_thesis_code/bayesian_inference/bayesian_statistics.py`**

- Deferred import + call to `plot_fisher_diagnostics()` added at end of `evaluate()`, after `_write_fisher_quality_csv()`.
- `allow_singular=True` **removed** from the testing-path `multivariate_normal()` call (~line 1472). Replaced with `try/except np.linalg.LinAlgError` that logs a WARNING and returns `[0.0]` (matching the `list[float]` return type of `single_host_likelihood_integration_testing`).

## Verification Results

| Check | Result |
|-------|--------|
| `pytest test_evaluation_report.py` | 28/28 passed |
| `pytest -m "not gpu and not slow"` | 494 passed, 6 skipped |
| `ruff check` | clean |
| `mypy` | clean (3 source files) |

## Task 3 Status: Checkpoint Pending

**Task 3: End-to-end verification** is a `checkpoint:human-verify` task. Requires running:

```bash
uv run python -m master_thesis_code simulations --evaluate --h_value 0.73 --compare_baseline .planning/debug/baseline.json
```

Then verifying:
1. `simulations/fisher_quality.csv` exists with correct columns
2. `simulations/fisher_quality_diagnostic.pdf` (or `.png`) exists with two panels
3. Comparison report contains "Fisher Quality" section (only if events were excluded)
4. MAP h is reported; note any change from baseline

## Deviations from Plan

**1. [Rule 1 - Bug] `allow_singular` removal needed `list[float]` return type fix**
- **Found during:** Task 2 (mypy)
- **Issue:** `single_host_likelihood_integration_testing` returns `list[float]`, but the early-exit `return 0.0` was a bare float.
- **Fix:** Changed to `return [0.0]` to match return type.
- **Files modified:** `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- **Commit:** 2bded1a

**2. [Rule 1 - Bug] Unused `type: ignore` comments in plot function**
- **Found during:** Task 2 (mypy)
- **Issue:** `type: ignore[assignment]` / `type: ignore[index]` annotations on `ax_eig` and `ax_scatter` lines were flagged as unused by mypy (types resolved correctly).
- **Fix:** Removed the ignore comments entirely.
- **Files modified:** `master_thesis_code/plotting/fisher_plots.py`
- **Commit:** 2bded1a

## Known Stubs

None. The `plot_fisher_diagnostics()` function handles the zero-flagged case gracefully (annotation instead of empty plot). The Fisher Quality section in comparison reports is gated on actual exclusion counts — it will appear only when the threshold is calibrated below the empirical cond_4d values (which are all in the 2.5e8–5.2e14 range; current default threshold is 1e10).

## Threat Flags

None found. The new CSV read in `extract_baseline()` is from the same local `simulations/` directory as existing posteriors — same trust boundary. Extreme eigenvalues are handled via log-scale y-axis (T-34-06 mitigated).

## Self-Check

- [x] `master_thesis_code/bayesian_inference/evaluation_report.py` — modified (verified by commit `2bded1a`)
- [x] `master_thesis_code/bayesian_inference/bayesian_statistics.py` — modified
- [x] `master_thesis_code/plotting/fisher_plots.py` — modified (min_lines: 60 satisfied — 120 lines added)
- [x] `master_thesis_code_test/bayesian_inference/test_evaluation_report.py` — modified
- [x] Commit `2bded1a` exists

## Self-Check: PASSED
