---
phase: 39-hpc-visualization-safe-wins
plan: 04
subsystem: visualization
tags: [viz-01, viz-02, latex, bootstrap-hdi, convergence-plot, generate-figures]
requirements_addressed: [VIZ-01, VIZ-02]
dependency_graph:
  requires: [39-03]
  provides:
    - "main.py:generate_figures auto-detects local LaTeX install and routes to apply_style(use_latex=True) / apply_style()"
    - "plot_h0_convergence accepts bootstrap_bank: ImprovementBank | None kwarg, drawing 16/84 HDI band on right panel"
    - "_gen_h0_convergence wires the cached compute_m_z_improvement_bank result through to the plot factory"
  affects:
    - "Phase 40 verification SC-7 (HDI band visible on the convergence figure)"
tech_stack:
  added: []
  patterns:
    - "TYPE_CHECKING + string-literal annotation to break a runtime circular import while satisfying mypy"
    - "Module-top stdlib import (shutil) + monkeypatch on the importer module's attribute for testability"
    - "Inline `from X import y` rebinding pattern: monkeypatch the source module's attribute (master_thesis_code.plotting._style.apply_style) so the function-scope re-import picks up the spy"
key_files:
  created: []
  modified:
    - master_thesis_code/main.py
    - master_thesis_code/plotting/convergence_plots.py
    - master_thesis_code_test/plotting/test_convergence_plots.py
    - master_thesis_code_test/test_generate_figures.py
decisions:
  - "Honored D-19 (LaTeX gating only at generate_figures, not the three other apply_style sites)."
  - "Honored D-22/D-27 (alpha=0.2, zorder=2, right panel only — no band on ax_post)."
  - "Honored D-24 (string-literal annotation; TYPE_CHECKING for the type-only import — circular import avoided without `from __future__ import annotations`)."
  - "Honored D-25 (reuse cached compute_m_z_improvement_bank result — JSON read, not bootstrap recomputation)."
metrics:
  duration_min: 6
  tasks_completed: 2
  commits: 4
  tests_added: 4
  tests_passing: 540
  tests_skipped: 6
  tests_deselected: 16
  completed_date: 2026-04-23
---

# Phase 39 Plan 04: VIZ-01 + VIZ-02 LaTeX auto-detect + bootstrap HDI band — Summary

LaTeX auto-detection routes `generate_figures` to `apply_style(use_latex=True)` whenever a local `latex` binary is on PATH; `plot_h0_convergence` now accepts an optional `bootstrap_bank: ImprovementBank | None` kwarg that draws a 16/84-percentile HDI band on the right panel, wired through `_gen_h0_convergence` from the cached `compute_m_z_improvement_bank` result.

## Tasks executed

| # | Task | Type | Commits |
|---|------|------|---------|
| 1 | VIZ-02 — `bootstrap_bank` kwarg + HDI band in `plot_h0_convergence` | TDD | `345aa66` (RED) → `fd1953c` (GREEN) |
| 2 | VIZ-01 LaTeX gating + VIZ-02 wiring in `_gen_h0_convergence` + smoke tests | TDD | `f88b4d3` (RED) → `c1cbaac` (GREEN) |

## VIZ-01 — LaTeX auto-detection (`main.py:~803`)

**Before** (`master_thesis_code/main.py:801-803`):
```python
from master_thesis_code.plotting._style import apply_style

apply_style()
```

**After**:
```python
from master_thesis_code.plotting._style import apply_style

# VIZ-01: auto-detect a local LaTeX install and route to the matching style.
if shutil.which("latex"):
    apply_style(use_latex=True)
    _ROOT_LOGGER.info("LaTeX detected; rendering figures with text.usetex=True")
else:
    apply_style()
    _ROOT_LOGGER.info("LaTeX not detected; using mathtext fallback")
```

`import shutil` was added to the module-top import block (sorted alphabetically after `os`). The other three `apply_style()` call sites (`main.py:35`, `plotting/fisher_plots.py:509`, `bayesian_inference/bayesian_inference_mwe.py:67`) are intentionally untouched per D-21.

## VIZ-02 — `plot_h0_convergence` signature delta (`plotting/convergence_plots.py`)

**New imports at top of file**:
```python
from typing import TYPE_CHECKING
...
if TYPE_CHECKING:
    # Type-only import avoids a circular dep with convergence_analysis at runtime.
    from master_thesis_code.plotting.convergence_analysis import ImprovementBank
```

**Signature delta** (one new keyword-only parameter inserted before `ax`):
```python
def plot_h0_convergence(
    ...
    color: str | None = None,
    color_alt: str | None = None,
    bootstrap_bank: "ImprovementBank | None" = None,   # NEW
    ax: None = None,
) -> tuple[Figure, npt.NDArray[np.object_]]:
```

**New band block** (inserted after the existing right-panel `ax_ci.plot(...)` calls, before the `1/sqrt(N)` reference curve):
```python
# --- Optional bootstrap HDI band on the right panel (VIZ-02) ---
if bootstrap_bank is not None:
    b_sizes = np.asarray(bootstrap_bank.sizes, dtype=np.float64)
    # Primary variant (no mass)
    w_no_lo = np.asarray(bootstrap_bank.metrics_no_mass["hdi68_width"]["p16"], dtype=np.float64)
    w_no_hi = np.asarray(bootstrap_bank.metrics_no_mass["hdi68_width"]["p84"], dtype=np.float64)
    ax_ci.fill_between(b_sizes, w_no_lo, w_no_hi, color=color, alpha=0.2, zorder=2)
    # Alt variant (with mass) — only if alt posteriors were provided
    if event_posteriors_alt is not None:
        w_with_lo = np.asarray(bootstrap_bank.metrics_with_mass["hdi68_width"]["p16"], dtype=np.float64)
        w_with_hi = np.asarray(bootstrap_bank.metrics_with_mass["hdi68_width"]["p84"], dtype=np.float64)
        ax_ci.fill_between(b_sizes, w_with_lo, w_with_hi, color=color_alt, alpha=0.2, zorder=2)
```

`alpha=0.2`, `zorder=2`, right panel only (no band on `ax_post` — D-27). Backward-compatible: `bootstrap_bank=None` reproduces the pre-VIZ-02 figure with zero `PolyCollection` artists on the right panel.

## VIZ-02 wiring delta (`main.py:_gen_h0_convergence`)

**Before**:
```python
def _gen_h0_convergence() -> tuple[object, object] | None:
    if post_data is None:
        return None
    from master_thesis_code.plotting.convergence_plots import plot_h0_convergence

    h_vals, event_posts = post_data
    h_alt, ep_alt = post_data_with if post_data_with is not None else (None, None)
    return plot_h0_convergence(
        h_vals,
        event_posts,
        true_h=0.73,
        h_values_alt=h_alt,
        event_posteriors_alt=ep_alt,
    )
```

**After**:
```python
def _gen_h0_convergence() -> tuple[object, object] | None:
    if post_data is None:
        return None
    from master_thesis_code.constants import H as TRUE_H
    from master_thesis_code.plotting.convergence_analysis import (
        compute_m_z_improvement_bank,
    )
    from master_thesis_code.plotting.convergence_plots import plot_h0_convergence

    h_vals, event_posts = post_data
    h_alt, ep_alt = post_data_with if post_data_with is not None else (None, None)
    # VIZ-02: try to load the cached improvement bank for the right-panel band.
    # Cached on disk by compute_m_z_improvement_bank — one JSON read per call.
    try:
        bootstrap_bank = compute_m_z_improvement_bank(Path(output_dir), h_true=float(TRUE_H))
    except (FileNotFoundError, ValueError, KeyError):
        bootstrap_bank = None
    return plot_h0_convergence(
        h_vals,
        event_posts,
        true_h=float(TRUE_H),
        h_values_alt=h_alt,
        event_posteriors_alt=ep_alt,
        bootstrap_bank=bootstrap_bank,
    )
```

Hardcoded `0.73` was also replaced with `float(TRUE_H)` (sourced from `master_thesis_code.constants.H`) so the truth marker tracks the project-wide constant.

## Smoke tests added

| Test | File | Branch covered |
|------|------|----------------|
| `TestPlotH0Convergence.test_plot_h0_convergence_without_bootstrap_bank_has_no_fill_between` | `master_thesis_code_test/plotting/test_convergence_plots.py` | VIZ-02 backward-compat: zero PolyCollection on right panel when kwarg None |
| `TestPlotH0Convergence.test_plot_h0_convergence_with_bootstrap_bank_adds_fill_between`     | `master_thesis_code_test/plotting/test_convergence_plots.py` | VIZ-02 active path: synthetic 14-field `ImprovementBank` produces ≥1 PolyCollection |
| `TestApplyStyleLatexGating.test_latex_branch_called_when_which_returns_path`               | `master_thesis_code_test/test_generate_figures.py`           | VIZ-01 `shutil.which → /usr/bin/latex` ⇒ `apply_style(use_latex=True)` |
| `TestApplyStyleLatexGating.test_mathtext_branch_called_when_which_returns_none`            | `master_thesis_code_test/test_generate_figures.py`           | VIZ-01 `shutil.which → None` ⇒ `apply_style()` (mathtext fallback) |

All 4 fail pre-implementation, all pass post-implementation (TDD RED→GREEN).

## Verification gate results

| Gate | Expected | Observed |
|------|----------|----------|
| `rg "shutil\.which\(\"latex\"\)" main.py` | 1 | **1** |
| `rg "apply_style\(use_latex=True\)" main.py` | 1 | **1** |
| `rg "apply_style\(\)" main.py` | ≥1 | **2** (line 35 module-load smoke + line 808 mathtext fallback) |
| `rg "apply_style\(\)" fisher_plots.py` | 1 (untouched) | **1** |
| `rg "apply_style\(\)" bayesian_inference_mwe.py` | 1 (untouched) | **1** |
| `rg "bootstrap_bank=bootstrap_bank" main.py` | 1 | **1** |
| `rg "compute_m_z_improvement_bank\(Path\(output_dir\)" main.py` | 2 | **2** (new VIZ-02 site + existing `_gen_paper_m_z_improvement`) |
| `rg "bootstrap_bank" convergence_plots.py` | ≥6 | **8** |
| `rg "fill_between" convergence_plots.py` | ≥2 | **3** (2 new VIZ-02 + 1 pre-existing in `plot_detection_efficiency`) |
| `pytest convergence_plots + generate_figures -m "not gpu"` | all green | **16 passed** |
| `pytest -m "not gpu and not slow"` (regression) | all green, ≥4 new | **540 passed**, 6 skipped, 16 deselected, 14 warnings (all pre-existing) |
| `mypy main.py + convergence_plots.py` | exit 0 | **Success: no issues found in 2 source files** |
| `ruff check` (4 files) | exit 0 | **All checks passed!** |

## Commits

| Hash | Type | Description |
|------|------|-------------|
| `345aa66` | test | RED — failing tests for VIZ-02 `bootstrap_bank` kwarg + HDI band |
| `fd1953c` | feat | GREEN — `bootstrap_bank` kwarg, TYPE_CHECKING import, band block |
| `f88b4d3` | test | RED — failing tests for VIZ-01 LaTeX auto-detection branches |
| `c1cbaac` | feat | GREEN — module-top `import shutil`, gated `apply_style`, VIZ-02 wiring in `_gen_h0_convergence` |

## Deviations from Plan

None — plan executed exactly as written. The single inline reformat (collapsing the `compute_m_z_improvement_bank(Path(output_dir), ...)` call from a 3-line wrap onto a single line) was made to satisfy the literal acceptance regex `compute_m_z_improvement_bank\(Path\(output_dir\)` which expects no whitespace between `bank(` and `Path(`. Functionally identical; line stays under the project's 100-character ruff limit.

## Threat model adherence

- **T-39-04-01 (Tampering — LaTeX subprocess)**: accepted; all figure text is project-controlled, no user-supplied strings reach the LaTeX subprocess. `shutil.which("latex")` provides a graceful fallback when LaTeX is absent.
- **T-39-04-02 (DoS — `compute_m_z_improvement_bank` exception)**: mitigated; `try/except (FileNotFoundError, ValueError, KeyError)` wraps the call and falls back to `bootstrap_bank=None`, preserving the pre-VIZ-02 figure.
- **T-39-04-03 (Info Disclosure — log line)**: accepted; logged only locally to SLURM stdout, no secrets, no network exposure.

No new threat surface introduced; no unmitigated threats remain.

## Self-Check: PASSED

- File `master_thesis_code/main.py` modified: **FOUND** (commit c1cbaac).
- File `master_thesis_code/plotting/convergence_plots.py` modified: **FOUND** (commit fd1953c).
- File `master_thesis_code_test/test_generate_figures.py` modified: **FOUND** (commit f88b4d3).
- File `master_thesis_code_test/plotting/test_convergence_plots.py` modified: **FOUND** (commit 345aa66).
- Commit 345aa66 in git log: **FOUND**.
- Commit fd1953c in git log: **FOUND**.
- Commit f88b4d3 in git log: **FOUND**.
- Commit c1cbaac in git log: **FOUND**.
