---
phase: 35-unified-pipeline-paper-figures
plan: "03"
subsystem: plotting
tags: [paper-figures, kde, smoothing, polish, testing]
dependency_graph:
  requires: [35-02 paper figure manifest entries 16-19, 35-01 compute_credible_interval]
  provides: [_kde_smooth_posterior, plot_h0_posterior_kde, manifest entry 20]
  affects: [master_thesis_code/plotting/paper_figures.py, master_thesis_code/main.py]
tech_stack:
  added: [scipy.stats.gaussian_kde (lazy import inside function)]
  patterns: [Scott bandwidth KDE smoothing, auto-detected grid spacing via np.diff, TDD RED-GREEN cycle]
key_files:
  created: []
  modified:
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code/main.py
    - master_thesis_code_test/plotting/test_paper_figures.py
decisions:
  - "KDE import kept inside function body to avoid module-level scipy dependency"
  - "MAP preservation check emits warning rather than raising — allows downstream callers to decide"
  - "tight_layout() called inside figure functions rather than at call site — consistent with project convention"
  - "fontsize kwargs removed from legend/title/xlabel calls so emri_thesis.mplstyle controls font sizes"
metrics:
  duration: "~15 minutes"
  completed: "2026-04-08"
  tasks_completed: 1
  tasks_total: 2
  files_changed: 3
requirements:
  - PFIG-04
---

# Phase 35 Plan 03: KDE Smoothing + Figure Polish Summary

KDE-smoothed H0 posterior variant added with Scott's-rule bandwidth and auto-detected grid resolution via `np.diff`; all 4 existing paper figures polished to inherit font sizes from the mplstyle sheet; manifest extended to 20 entries.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (RED) | Add failing KDE smoothing tests | 21061f1 | test_paper_figures.py |
| 1 (GREEN) | Implement KDE smoothing + polish + manifest entry | af52eb4 | paper_figures.py, main.py |

## Task 2 Status: Awaiting Human Verification

Task 2 is a `checkpoint:human-verify` gate and was intentionally not executed by this agent. It requires:

1. Running the figure generation pipeline:
   ```bash
   uv run python -m master_thesis_code cluster_results/eval_corrected_full --generate_figures cluster_results/eval_corrected_full
   ```
2. Opening the 5 generated PDFs in `cluster_results/eval_corrected_full/figures/`:
   - `paper_h0_posterior.pdf`
   - `paper_single_event.pdf`
   - `paper_convergence.pdf`
   - `paper_snr_distribution.pdf`
   - `paper_h0_posterior_kde.pdf`
3. Verifying no Type 3 fonts: `pdffonts paper_h0_posterior.pdf`
4. Checking font sizes are appropriate for REVTeX single-column width (~3.375 inches)
5. Typing "approved" to confirm publication quality or describing specific issues.

## What Was Built

**`_kde_smooth_posterior(h_values, posterior, n_fine=500)`** (Task 1):
- Computes weight array from posterior (`weights = posterior / norm`)
- Calls `scipy.stats.gaussian_kde` with Scott's bandwidth rule
- Returns `(h_fine, kde_fine)` on a 500-point fine grid
- Early return with `.copy()` arrays when `norm <= 0`

**`plot_h0_posterior_kde(data_dir)`** (Task 1):
- Loads both posterior JSONs via `_load_combined_posterior`
- Auto-detects grid spacing: `grid_spacing = float(np.diff(h_values).mean())` (D-06)
- Verifies MAP preservation — logs warning if KDE MAP drifts more than one grid spacing
- Plots discrete markers (alpha=0.4) + KDE smooth lines (full alpha) for each variant
- 68% CI shading on KDE-smoothed values using `compute_credible_interval`
- Truth line at h=0.73; `get_figure(preset="single")` layout

**Manifest entry 20** added to `generate_figures()` in `main.py`:
```python
manifest.append(("paper_h0_posterior_kde", _gen_paper_h0_posterior_kde))
```

**Polished existing figures** — removed all hardcoded `fontsize=` kwargs from:
- `plot_h0_posterior_comparison`: `ax.legend()` (was `fontsize=9`), added `fig.tight_layout()`
- `plot_single_event_likelihoods`: `ax_no.set_ylabel()`, column titles, xlabel (all were `fontsize=9`), added `fig.tight_layout(h_pad=0.3)`
- `plot_posterior_convergence`: `ax.legend()` (was `fontsize=9`), added `ax.minorticks_on()`, `fig.tight_layout()`
- `plot_snr_distribution`: `ax_hist.legend()` (was `fontsize=8`), added `fig.tight_layout(w_pad=1.0)`

**6 new tests** in `TestKDESmoothing`:
- `test_kde_smooth_returns_correct_shape` — 500-point output
- `test_kde_map_within_grid_spacing` — MAP preserved within grid spacing
- `test_kde_zero_posterior_returns_copy` — zero input returns copy
- `test_auto_detect_grid_spacing_15pt` — np.diff mean ~0.0214
- `test_auto_detect_grid_spacing_31pt` — np.diff mean ~0.0100
- `test_plot_h0_posterior_kde_returns_fig_axes` — synthetic data, returns (Figure, Axes)

## Verification

```
uv run pytest master_thesis_code_test/plotting/test_paper_figures.py -x -m "not slow and not gpu" -v --no-cov
# => 15 passed

uv run pytest -m "not gpu and not slow" -x --no-cov -q
# => 465 passed, 6 skipped, 18 deselected

grep -c "def _kde_smooth_posterior" master_thesis_code/plotting/paper_figures.py  # => 1
grep -c "def plot_h0_posterior_kde" master_thesis_code/plotting/paper_figures.py   # => 1
grep -c "gaussian_kde" master_thesis_code/plotting/paper_figures.py                # => 2 (import + call)
grep -c "np.diff" master_thesis_code/plotting/paper_figures.py                     # => 2
grep -c "paper_h0_posterior_kde" master_thesis_code/main.py                       # => 2
grep "fontsize=9" master_thesis_code/plotting/paper_figures.py | grep "legend"    # => (empty)
```

## Deviations from Plan

None — plan executed exactly as written. The `tight_layout` UserWarning from matplotlib is cosmetic (an internal matplotlib sequencing note) and does not affect figure output.

## Known Stubs

None. All functions accept `data_dir` explicitly; no hardcoded paths.

## Threat Flags

None — internal plotting code; no network access, user input, or auth.

## Self-Check: PASSED

- `master_thesis_code/plotting/paper_figures.py` — FOUND, contains `_kde_smooth_posterior`, `plot_h0_posterior_kde`, `gaussian_kde`, `np.diff`
- `master_thesis_code/main.py` — FOUND, contains `paper_h0_posterior_kde` (2 occurrences)
- `master_thesis_code_test/plotting/test_paper_figures.py` — FOUND, contains `TestKDESmoothing`, `test_kde_map_within_grid_spacing`
- Commit 21061f1 (RED tests) — FOUND
- Commit af52eb4 (GREEN implementation) — FOUND
