---
phase: 35-unified-pipeline-paper-figures
verified: 2026-04-08T00:00:00Z
status: human_needed
score: 5/6 must-haves verified
human_verification:
  - test: "Generate all figures by running: uv run python -m master_thesis_code cluster_results/eval_corrected_full --generate_figures cluster_results/eval_corrected_full"
    expected: "5 paper PDFs produced in cluster_results/eval_corrected_full/figures/: paper_h0_posterior.pdf, paper_single_event.pdf, paper_convergence.pdf, paper_snr_distribution.pdf, paper_h0_posterior_kde.pdf. All look publication-ready: markers visible, CI shading, truth lines, no clipped labels, row labels readable, no Type 3 fonts."
    why_human: "Visual publication quality — aesthetics, label legibility, font sizes at REVTeX single-column width (~3.375 inches), Type 3 font check via pdffonts — cannot be verified programmatically. This was a blocking checkpoint:human-verify gate in Plan 03, Task 2."
---

# Phase 35: Unified Figure Pipeline & Paper Figures — Verification Report

**Phase Goal:** Merge the two disconnected figure pipelines and deliver polished paper figures with new style
**Verified:** 2026-04-08
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Single `--generate_figures <dir>` command generates all figures (paper + thesis + galaxy-level) | VERIFIED | `main.py` manifest has 20 entries (15 thesis + 5 paper) with paper entries 16-20 at lines 1044-1099; all use lazy imports + graceful degradation |
| 2 | `paper_figures.py` functions integrated into unified manifest | VERIFIED | Entries `paper_h0_posterior`, `paper_single_event`, `paper_convergence`, `paper_snr_distribution`, `paper_h0_posterior_kde` in manifest; `from master_thesis_code.plotting.paper_figures import` in each closure |
| 3 | CI calculation uses trapezoidal CDF everywhere (unit test) | VERIFIED | `compute_credible_interval` in `_helpers.py` uses `np.trapezoid`; `convergence_plots.py` imports and calls it; `_credible_interval_width` deleted; 5 unit tests in `TestComputeCredibleInterval` — 27/27 tests pass |
| 4 | 4 existing paper figures polished with new style | VERIFIED (code) | `tight_layout()` present in all 4 functions; hardcoded `fontsize=9`/`fontsize=8` removed from all legend/label calls; one `fontsize=8` remains in SNR placeholder text element (not a legend call — acceptable per plan) |
| 5 | Contour-smoothed H0 posterior added as new variant (KDE, preserves MAP within grid spacing) | VERIFIED | `_kde_smooth_posterior` and `plot_h0_posterior_kde` exist in `paper_figures.py`; `gaussian_kde` with Scott's rule; MAP preservation check with `abs(kde_map - discrete_map) >= grid_spacing` warning; `test_kde_map_within_grid_spacing` passes |
| 6 | Auto-detect h-grid resolution (works with 15-pt and future finer grids) | VERIFIED | `float(np.diff(h_no).mean())` and `float(np.diff(h_with).mean())` in `plot_h0_posterior_kde`; `test_auto_detect_grid_spacing_15pt` and `test_auto_detect_grid_spacing_31pt` both pass |

**Score:** 5/6 truths verified programmatically (SC4 verified at code level; publication quality requires human)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `master_thesis_code/plotting/_helpers.py` | `compute_credible_interval` function | VERIFIED | `def compute_credible_interval(h_values, posterior, level=0.68)` at line 22; uses `np.trapezoid`; NumPy-style docstring; proper type annotations |
| `master_thesis_code_test/plotting/test_helpers.py` | CI unit tests | VERIFIED | `class TestComputeCredibleInterval` with 5 test methods: Gaussian, uniform, zero, return type, level parameter — all pass |
| `master_thesis_code/main.py` | Manifest entries 16-19 (+ 20) for paper figures | VERIFIED | All 5 entries present at lines 1046-1099 |
| `master_thesis_code/plotting/paper_figures.py` | Integrated paper figure functions without standalone main() | VERIFIED | No `_DATA_ROOT`, no `def main()`, no `if __name__`, no `_ci_width_from_log_posteriors`, no `from __future__ import annotations` |
| `master_thesis_code/plotting/paper_figures.py` | KDE smoothing function | VERIFIED | `def _kde_smooth_posterior(` at line 37; `def plot_h0_posterior_kde(` at line 682 |
| `master_thesis_code_test/plotting/test_paper_figures.py` | Smoke tests + KDE tests | VERIFIED | 15 tests across `TestPaperFigureImports`, `TestPaperFigureGracefulDegradation`, `TestNoStandaloneMain`, `TestKDESmoothing` — all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `convergence_plots.py` | `_helpers.py` | `import compute_credible_interval` | VERIFIED | Line 19: `from master_thesis_code.plotting._helpers import _fig_from_ax, compute_credible_interval, get_figure` |
| `main.py` | `paper_figures.py` | manifest import in generator closures | VERIFIED | 5 lazy `from master_thesis_code.plotting.paper_figures import ...` statements inside generator closures |
| `paper_figures.py` | `_helpers.py` | `import compute_credible_interval` | VERIFIED | Line 30: `from master_thesis_code.plotting._helpers import compute_credible_interval, get_figure` |
| `paper_figures.py` | `scipy.stats.gaussian_kde` | lazy import inside `_kde_smooth_posterior` | VERIFIED | Line 58: `from scipy.stats import gaussian_kde` inside function body (avoids module-level scipy dependency) |

### Data-Flow Trace (Level 4)

Paper figure functions are plotting factories that consume JSON data from disk. They do not render dynamic state from React-style stores. The data flow is:

- `_load_combined_posterior(variant, data_dir)` reads `combined_posterior.json` / `combined_posterior_with_bh_mass.json` from `data_dir`
- `data_dir` is threaded in from the manifest closure as `Path(output_dir)` (the `--generate_figures` CLI argument)
- No hardcoded `_DATA_ROOT` or module-level constant remains

The `test_plot_h0_posterior_kde_returns_fig_axes` test in `TestKDESmoothing` writes synthetic JSON files to `tmp_path` and calls the function with real data flowing through — confirming the data path is wired and not hollow.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `compute_credible_interval` importable from `_helpers` | `pytest test_helpers.py -q` | 12 passed | PASS |
| Paper figure functions importable with `data_dir` signature | `pytest test_paper_figures.py -q` | 15 passed | PASS |
| `_kde_smooth_posterior` returns 500-point output | `pytest -k kde_smooth` | passes | PASS |
| Convergence tests not broken by `_credible_interval_width` removal | `pytest test_convergence_plots.py -q` | 8 passed | PASS |
| Full test suite with no regression | `pytest -m "not gpu and not slow" -q` | 465 passed, 6 skipped (per SUMMARY) | PASS |

### Requirements Coverage

| Requirement ID | Source Plan | Description | Status | Evidence |
|----------------|-------------|-------------|--------|---------|
| PFIG-01 | 35-02-PLAN.md | Single `--generate_figures <dir>` generates all figures | SATISFIED | 20-entry manifest in `main.py`; entries 16-20 for paper figures |
| PFIG-02 | 35-02-PLAN.md | `paper_figures.py` functions integrated into unified manifest | SATISFIED | All 4 public functions wired; `data_dir` threaded; no `_DATA_ROOT` |
| PFIG-03 | 35-01-PLAN.md | CI calculation uses trapezoidal CDF everywhere (unit tested) | SATISFIED | `compute_credible_interval` in `_helpers.py` with 5 unit tests passing |
| PFIG-04 | 35-03-PLAN.md | 4 paper figures polished; KDE variant added; auto-detect grid resolution | SATISFIED (code) / NEEDS HUMAN (visual quality) | Code evidence: `tight_layout`, fontsize removed, `_kde_smooth_posterior`, `np.diff` grid detection. Visual quality requires human review. |

**Note on requirements traceability:** PFIG-01 through PFIG-04 are defined as phase-specific informal IDs within ROADMAP.md phase 35's requirements field and 35-RESEARCH.md. They do not appear in `.planning/REQUIREMENTS.md`, which tracks the v2.1 H0 Bias Resolution milestone requirements (DIAG-xx, COMP-xx, etc.). This is a documentation gap — phase 35 belongs to the v2.1 Publication Figures milestone, not the H0 Bias Resolution milestone, and no separate REQUIREMENTS section was created for it. The PFIG IDs are internally consistent within the phase planning artifacts.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `paper_figures.py` | 668 | `fontsize=8` in placeholder `ax.text()` | Info | In the SNR placeholder (no-data path only). This is a static text element in a placeholder figure, not a legend or axis label. Acceptable — the plan only specified removing `fontsize` from `legend()` calls. |

No blockers found. No TODOs, FIXMEs, stub returns, or orphaned artifacts detected.

### Human Verification Required

#### 1. Publication Quality of Generated Paper Figures

**Test:** With cluster data available at `cluster_results/eval_corrected_full/`, run:
```bash
uv run python -m master_thesis_code cluster_results/eval_corrected_full --generate_figures cluster_results/eval_corrected_full
```

Then open each PDF in `cluster_results/eval_corrected_full/figures/`:
- `paper_h0_posterior.pdf` — verify: markers visible, CI shading, truth line at h=0.73, no clipped labels, legend readable
- `paper_single_event.pdf` — verify: 4x2 grid, row labels readable, panels aligned, both column titles visible
- `paper_convergence.pdf` — verify: error bars visible, N^{-1/2} reference line, log-log axes with minor ticks
- `paper_snr_distribution.pdf` — verify: threshold line, colorbar if redshift available, panels not crowded
- `paper_h0_posterior_kde.pdf` — verify: smooth KDE curves visible over discrete markers, CI shading, truth line

Also run: `pdffonts paper_h0_posterior.pdf` and verify zero Type 3 font entries.

**Expected:** All 5 PDFs render correctly at REVTeX single-column width (~3.375 inches). Font sizes look appropriate. No Type 3 fonts. Figures ready for paper submission.

**Why human:** Visual aesthetics, label legibility at publication size, Type 3 font detection — cannot be verified programmatically. This was explicitly designated as a blocking `checkpoint:human-verify` gate in Plan 03 Task 2 and was intentionally not executed by the automated agent.

### Gaps Summary

No functional gaps. The single open item is the blocking human verification gate (Plan 03 Task 2) for publication-quality visual inspection of generated PDFs. All automated checks — 27 unit and smoke tests, full suite of 465 tests, key link verification, data-flow trace — pass completely.

---

_Verified: 2026-04-08_
_Verifier: Claude (gsd-verifier)_
