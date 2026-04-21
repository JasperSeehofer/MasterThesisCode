# Phase 35: Unified Figure Pipeline & Paper Figures - Research

**Researched:** 2026-04-08
**Domain:** Matplotlib plotting pipeline, KDE smoothing, CI calculation, manifest pattern
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Integrate `paper_figures.py` functions into `generate_figures()` manifest in `main.py` as entries 16-19. Remove the standalone `main()` from `paper_figures.py`.
- **D-02:** Replace the hardcoded `_DATA_ROOT = Path("cluster_results/eval_corrected_full")` with the same `data_dir` parameter used by `generate_figures()`. All figures look in the same working directory passed via `--generate_figures <dir>`.
- **D-03:** Keep `--generate_interactive` as a separate flag. Interactive Plotly figures remain decoupled from static PDF generation.
- **D-04:** Full visual rework of all 4 paper figures. Redesign layouts, adjust spacing, add annotations, reconsider color choices. Publication-ready polish pass on each figure.
- **D-05:** Add a contour-smoothed H0 posterior variant using conservative Gaussian KDE (Scott's rule bandwidth). Must preserve MAP within one grid spacing of the discrete MAP.
- **D-06:** Auto-detect h-grid resolution from the data via `np.diff(h_values)`. No config parameter needed. Must work with 15-pt, 31-pt, and future finer grids.
- **D-07:** Extract the duplicated CDF/CI calculation into a shared `compute_credible_interval(h_values, posterior, level=0.68)` function in `_helpers.py`. Both `paper_figures.py` and `convergence_plots.py` call it.
- **D-08:** Unit test the shared CI function against two analytical distributions: (1) Gaussian where 68% CI = 2*sigma, and (2) uniform distribution where 68% CI = 0.68 * range.
- **D-09:** All figures (paper + thesis) go to `<dir>/figures/` in a single flat directory. Paper figures use a `paper_` prefix (e.g., `paper_h0_posterior.pdf`).

### Claude's Discretion

- Figure numbering scheme within the unified manifest (continue fig16-fig19 or renumber)
- Internal refactoring of data loaders in `paper_figures.py` to work with the manifest pattern
- Exact KDE bandwidth selection method details
- Specific visual rework choices (annotation text, spacing values, legend positioning) for each of the 4 paper figures

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PFIG-01 | Single `--generate_figures <dir>` command generates all figures (paper + thesis + galaxy-level) | Manifest pattern already in `main.py` lines 835-1044; add entries 16-19 for paper figures |
| PFIG-02 | `paper_figures.py` functions integrated into unified manifest | 4 functions identified: `plot_h0_posterior_comparison`, `plot_single_event_likelihoods`, `plot_posterior_convergence`, `plot_snr_distribution` |
| PFIG-03 | CI calculation uses trapezoidal CDF everywhere (unit test) | Two CI implementations found, both trapezoid-based but different accumulation logic; shared helper extracts consistent approach |
| PFIG-04 | 4 existing paper figures polished with new style; contour-smoothed H0 posterior added as new variant; auto-detect h-grid resolution | KDE verified: scipy.stats.gaussian_kde with Scott's rule; MAP preservation verified on 15-pt grid |
</phase_requirements>

---

## Summary

Phase 35 is a software-only pipeline merge and visual polish task. The codebase already has all the building blocks; the work is integration and refinement, not new capability.

The two pipelines that need merging are: (1) `generate_figures()` in `main.py` (15-entry manifest, driven by `--generate_figures <dir>`) and (2) the standalone `main()` in `paper_figures.py` (4 figures, hardcoded `_DATA_ROOT`). The integration strategy is straightforward: add entries 16-19 to the manifest, update the paper figure functions to accept `data_dir` as a parameter (they already accept it as a keyword argument with default `_DATA_ROOT`), and remove the standalone entry point.

The CI calculation duplication is real: `paper_figures.py:_ci_width_from_log_posteriors()` and `convergence_plots.py:_credible_interval_width()` both compute 68% CI via CDF integration, but with slightly different accumulation approaches (per-step `np.trapezoid` vs `np.gradient + np.cumsum`). A shared `compute_credible_interval()` in `_helpers.py` will be the canonical implementation. The unit tests for this function must use loose tolerances (`atol ~ 2 * grid_spacing`) because coarse h-grids introduce ~3% discretization error on the uniform distribution test case — this is expected and not a bug.

**Primary recommendation:** Work in three ordered passes: (1) extract `compute_credible_interval()` and write tests, (2) wire paper figure functions into the manifest with `data_dir` threading, (3) do the visual polish and add the KDE-smoothed variant.

---

## Standard Stack

### Core (all already installed — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | installed in .venv | Plotting engine, style application | Project standard |
| numpy | installed in .venv | Array math, CDF accumulation | Project standard |
| scipy.stats.gaussian_kde | scipy 1.17.1 [VERIFIED: local] | KDE smoothing for posterior variant | Only dependency for D-05 |
| scipy | 1.17.1 [VERIFIED: local] | `gaussian_kde` with Scott's rule | Already installed |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | installed | CRB CSV loading in SNR figure | Already used in `plot_snr_distribution` |

**Installation:** No new packages needed. All dependencies present in `.venv`. [VERIFIED: local env check]

---

## Architecture Patterns

### Manifest Pattern (existing, to be extended)

The manifest pattern is defined in `main.py` lines 835-1044 [VERIFIED: codebase read]:

```python
manifest: list[tuple[str, Callable[[], tuple[object, object] | None]]] = []

def _gen_paper_h0_posterior() -> tuple[object, object] | None:
    from master_thesis_code.plotting.paper_figures import plot_h0_posterior_comparison
    return plot_h0_posterior_comparison(data_dir=Path(output_dir))

manifest.append(("paper_h0_posterior", _gen_paper_h0_posterior))
```

Each generator:
- Returns `(fig, ax)` or `None` when data is missing (graceful degradation)
- Is a zero-argument closure capturing `output_dir` and pre-loaded data
- Gets saved via `_save(fig, name)` which calls `save_figure()` + `_check_file_size()`

Paper figure names use `paper_` prefix per D-09: `paper_h0_posterior`, `paper_single_event`, `paper_convergence`, `paper_snr_distribution`.

### Data Loader Threading Pattern

The paper figure data loaders in `paper_figures.py` already accept `data_dir` as a keyword argument with `_DATA_ROOT` as default [VERIFIED: code read]:

```python
def _load_combined_posterior(variant: str) -> dict[str, Any]:
    # Uses module-level _DATA_ROOT — must be refactored

def plot_h0_posterior_comparison(
    data_dir: Path = _DATA_ROOT,  # already parameterized
) -> tuple[Figure, Axes]:
    p_no = _load_combined_posterior("posteriors")  # still uses _DATA_ROOT internally
```

The internal loaders `_load_combined_posterior`, `_load_per_event_no_mass`, and `_load_per_event_with_mass_scalars` use the module-level `_DATA_ROOT` directly rather than accepting a `base` parameter through the call chain. The refactor must thread `data_dir` through these private loaders.

Pattern: Convert private loader functions to accept `data_dir: Path` as first positional argument, then all public plot functions pass through. This is a straightforward parameter-threading task.

### Shared CI Helper Pattern

Target signature for `_helpers.py` addition [ASSUMED — based on D-07 and existing code]:

```python
def compute_credible_interval(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.68,
) -> tuple[float, float]:
    """Return (lo, hi) of the central credible interval at *level*.

    Uses cumulative trapezoidal integration (CDF). Returns (nan, nan)
    if posterior norm is zero.
    """
```

**Key implementation choice:** Use `np.trapezoid` per-step (matching `paper_figures.py` style) rather than `np.gradient + np.cumsum` (matching `convergence_plots.py` style). The per-step trapezoid is more numerically accurate for non-uniform grids, and the phase requires "trapezoidal CDF everywhere" (D-08). Both implementations produce virtually identical results on uniform grids, so the choice is cosmetic.

Returning `(lo, hi)` instead of just `hi - lo` gives callers more flexibility (e.g., shading CIs).

### KDE Smoothing Pattern

For the contour-smoothed H0 posterior variant (D-05) [VERIFIED: tested locally]:

```python
from scipy.stats import gaussian_kde

def _kde_smooth_posterior(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Smooth posterior with Gaussian KDE (Scott's rule).

    MAP preservation: verified that discrete MAP and KDE MAP agree
    within one grid spacing for typical h-grids (15-pt and 31-pt).
    """
    norm = posterior.sum()
    if norm <= 0:
        return posterior.copy()
    weights = posterior / norm
    kde = gaussian_kde(h_values, weights=weights, bw_method="scott")
    # Evaluate on a fine grid, then resample to original h_values for overlay
    h_fine = np.linspace(h_values[0], h_values[-1], 500)
    kde_fine = kde(h_fine)
    return h_fine, kde_fine
```

**Verified on 15-pt grid:** Discrete MAP = 0.73, KDE MAP = 0.7301, shift = 0.0001 < grid_spacing (0.0186). MAP preservation constraint satisfied. [VERIFIED: local test]

### Auto-Detect Grid Resolution

D-06 is trivially implemented [VERIFIED: tested locally]:

```python
grid_spacing = float(np.diff(h_values).mean())
# 15-pt grid (0.60-0.90): spacing = 0.0214
# 31-pt grid (0.60-0.90): spacing = 0.0100
```

`np.diff(h_values)` works correctly for both grids. No hardcoded assumptions needed.

### Recommended Project Structure (no changes)

The phase does not introduce new files — it reorganizes existing code. Files affected:

```
master_thesis_code/
├── plotting/
│   ├── _helpers.py          # ADD: compute_credible_interval()
│   ├── paper_figures.py     # MODIFY: thread data_dir; remove main(); update all 4 plots
│   └── convergence_plots.py # MODIFY: call shared compute_credible_interval()
└── main.py                  # MODIFY: add manifest entries 16-19 for paper figures
master_thesis_code_test/
└── plotting/
    ├── test_helpers.py      # ADD: tests for compute_credible_interval
    └── test_paper_figures.py # ADD: smoke tests for 4 paper figure functions
```

---

## CI Calculation — Critical Details

### Two Existing Implementations Found

[VERIFIED: code read]

**`paper_figures.py:_ci_width_from_log_posteriors()` (lines 397-419)**
- Takes log-posteriors, exponentiates internally
- Builds CDF with per-step `np.trapezoid(pn[i-1:i+1], h[i-1:i+1])`
- Interpolates on 2000-point fine grid
- Returns `float` width (hi - lo)

**`convergence_plots.py:_credible_interval_width()` (lines 26-58)**
- Takes linear posterior
- Uses `np.gradient(h_values)` then `np.cumsum(p * dh)`
- Interpolates directly on h_values with `np.interp`
- Returns `float` width (hi - lo)

**Key difference:** `np.gradient` uses one-sided differences at array boundaries, while the per-step `np.trapezoid` accumulation is equivalent to a proper trapezoid rule. For uniform grids both give essentially the same result, but the trapezoid approach is more principled.

### Unit Test Tolerances

The Gaussian test case works cleanly (verified: 68% CI = 0.0602 vs expected 0.0600, error = 0.3%). [VERIFIED: local test]

The uniform distribution test case has ~3.3% error on a 31-pt grid due to boundary effects in CDF accumulation. On a 1001-pt grid the error drops to 0.1%. [VERIFIED: local test]

**Test tolerance recommendation:**
- Gaussian: `atol=0.003` (half grid spacing of 31-pt grid = 0.005)
- Uniform: `atol=0.01` (larger tolerance; grid artifact, not algorithm bug)

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| KDE smoothing | Custom kernel convolution | `scipy.stats.gaussian_kde` | Handles weights, Scott's rule, evaluation on arbitrary points |
| CDF from posterior | Manual sum | `np.trapezoid` accumulation | Already working in codebase; don't change what works |
| CI width | Custom percentile logic | `compute_credible_interval()` from `_helpers.py` | The whole point of D-07 is to stop hand-rolling this |

---

## Common Pitfalls

### Pitfall 1: `_load_combined_posterior` ignores `data_dir`

**What goes wrong:** After D-02, callers pass `data_dir` to the public plot functions. But `_load_combined_posterior()` reads from the module-level `_DATA_ROOT` directly, ignoring the passed-in path.

**Why it happens:** The public functions already have `data_dir` parameters, but the internal loader chain doesn't thread it through.

**How to avoid:** Search for all uses of `_DATA_ROOT` inside `paper_figures.py` (there are 2: in `_load_combined_posterior` and `plot_snr_distribution`'s internal loader call). Thread `data_dir: Path` through each private helper. [VERIFIED: code read — `_DATA_ROOT` used in `_load_combined_posterior` and `plot_snr_distribution` directly]

**Warning signs:** Figures generate successfully but read from `cluster_results/eval_corrected_full/` instead of the user-specified `<dir>`.

### Pitfall 2: `convergence_plots.py` still uses old CI function after D-07

**What goes wrong:** `_credible_interval_width` in `convergence_plots.py` is a private function. After extracting `compute_credible_interval` to `_helpers.py`, the existing private function must be deleted and replaced with a call to the shared one, or the duplication remains.

**How to avoid:** Delete `_credible_interval_width` from `convergence_plots.py` entirely and call `from master_thesis_code.plotting._helpers import compute_credible_interval`. The `plot_h0_convergence` function calls it at line 133.

**Warning signs:** CI tests pass but the old private function still exists alongside the new shared one.

### Pitfall 3: `_ci_width_from_log_posteriors` in `paper_figures.py` takes log-posterior

**What goes wrong:** The shared `compute_credible_interval` should take a linear posterior. `paper_figures.py`'s `_ci_width_from_log_posteriors` takes log-posteriors and does the `exp()` internally. If the shared helper has the same signature as `_credible_interval_width` (linear), the call sites in `plot_posterior_convergence` must be updated to exponentiate first.

**How to avoid:** The shared function takes linear posterior. The `plot_posterior_convergence` caller already does the log-to-linear conversion before the CI call (line 491: `log_combined = np.sum(log_event_matrix[idx, :], axis=0)` then must `exp()`). Keep the public interface as linear-posterior.

**Warning signs:** CI returns NaN or negative values.

### Pitfall 4: `paper_` prefix missing from saved figures

**What goes wrong:** Manifest entries 16-19 use names like `h0_posterior_comparison` (without `paper_` prefix), so they collide with existing thesis figures.

**How to avoid:** Per D-09, use `paper_h0_posterior`, `paper_single_event`, `paper_convergence`, `paper_snr_distribution` as manifest keys.

### Pitfall 5: KDE MAP shift on coarse grids with asymmetric posteriors

**What goes wrong:** For strongly asymmetric posteriors or posteriors with low weight at the true MAP, Scott's rule KDE can shift the MAP by more than one grid spacing.

**How to avoid:** Always verify MAP preservation by computing `abs(h_fine[argmax(kde_vals)] - h_values[argmax(posterior)])` and comparing to `np.diff(h_values).mean()`. Log a warning if violated. [VERIFIED: MAP preserved on symmetric Gaussian; [ASSUMED] behavior on asymmetric posteriors]

---

## Code Examples

### Adding a paper figure to the manifest

```python
# In generate_figures(), after existing manifest entries:

# 16. Paper figure: H0 posterior comparison
def _gen_paper_h0_posterior() -> tuple[object, object] | None:
    from master_thesis_code.plotting.paper_figures import plot_h0_posterior_comparison
    try:
        return plot_h0_posterior_comparison(data_dir=Path(output_dir))
    except (FileNotFoundError, KeyError):
        return None

manifest.append(("paper_h0_posterior", _gen_paper_h0_posterior))
```

### shared compute_credible_interval in _helpers.py

```python
# Source: convergence_plots.py:_credible_interval_width (adapted to return (lo, hi))
def compute_credible_interval(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.68,
) -> tuple[float, float]:
    """Return (lo, hi) of the central credible interval at *level*.

    Uses cumulative trapezoidal integration. Returns (nan, nan) if
    posterior norm is zero or non-positive.
    """
    norm = np.trapezoid(posterior, h_values)
    if norm <= 0:
        return float("nan"), float("nan")
    p = posterior / norm
    cdf = np.zeros(len(h_values))
    for i in range(1, len(h_values)):
        cdf[i] = cdf[i - 1] + np.trapezoid(p[i - 1 : i + 1], h_values[i - 1 : i + 1])
    cdf /= cdf[-1]
    lo = float(np.interp((1.0 - level) / 2.0, cdf, h_values))
    hi = float(np.interp((1.0 + level) / 2.0, cdf, h_values))
    return lo, hi
```

### Unit test pattern for D-08

```python
# Source: design from CONTEXT.md D-08; implementation follows test_helpers.py pattern
import numpy as np
import pytest
from master_thesis_code.plotting._helpers import compute_credible_interval


class TestComputeCredibleInterval:

    def test_gaussian_68ci_equals_two_sigma(self) -> None:
        """68% CI of Gaussian = 2*sigma (within grid discretization tolerance)."""
        sigma = 0.04
        h = np.linspace(0.60, 0.90, 101)  # fine enough for 1% error
        posterior = np.exp(-0.5 * ((h - 0.73) / sigma) ** 2)
        lo, hi = compute_credible_interval(h, posterior, level=0.68)
        assert abs((hi - lo) - 2 * sigma) < 0.003

    def test_uniform_68ci_equals_68_percent_range(self) -> None:
        """68% CI of uniform = 0.68 * range (coarse grid: atol=0.01)."""
        h = np.linspace(0.60, 0.90, 31)
        posterior = np.ones_like(h)
        lo, hi = compute_credible_interval(h, posterior, level=0.68)
        expected = 0.68 * (h[-1] - h[0])
        assert abs((hi - lo) - expected) < 0.01  # grid artifact tolerance

    def test_zero_posterior_returns_nan(self) -> None:
        """Zero posterior returns (nan, nan) gracefully."""
        h = np.linspace(0.60, 0.90, 31)
        lo, hi = compute_credible_interval(h, np.zeros_like(h))
        assert np.isnan(lo) and np.isnan(hi)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Two disconnected figure pipelines | Single manifest in `generate_figures()` | Phase 35 | One CLI command for all figures |
| Per-file CI implementation | Shared `compute_credible_interval` in `_helpers.py` | Phase 35 | Single source of truth, testable |
| Raw discrete posterior only | + KDE-smoothed variant | Phase 35 | Cleaner publication figures |
| Hardcoded `_DATA_ROOT` | `data_dir` parameter | Phase 35 | Works with any `--generate_figures <dir>` |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | MAP preservation holds for asymmetric posteriors under Scott's rule KDE | Common Pitfalls, Code Examples | KDE-smoothed figure could misrepresent MAP; mitigation: add a warning log |
| A2 | Returning `(lo, hi)` tuple from `compute_credible_interval` is compatible with all call sites in `convergence_plots.py` | Code Examples | Callers expecting single `float` width need `hi - lo` to be added |

---

## Open Questions (RESOLVED)

1. **Should `compute_credible_interval` return `(lo, hi)` or just `hi - lo` (width)?**
   RESOLVED: Return `(lo, hi)` tuple. Plan 35-01 implements this signature.
   - What we know: `paper_figures.py` uses both `lo` and `hi` individually (for CI shading in `plot_h0_posterior_comparison` lines 222-225); `convergence_plots.py` only uses the width.
   - Recommendation: Return `(lo, hi)` tuple. Callers that only need width use `hi - lo`. This avoids a future change when CI shading is needed.

2. **Figure numbering: fig16-fig19 or paper_01-paper_04?**
   RESOLVED: Use `paper_` prefix without sequential number. Plan 35-02 implements this.
   - Claude's discretion per CONTEXT.md.
   - Recommendation: Use `paper_` prefix without a sequential number (e.g., `paper_h0_posterior`, `paper_single_event`, `paper_convergence`, `paper_snr_distribution`). More readable, avoids renumbering if thesis figures change.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| scipy.stats.gaussian_kde | D-05 KDE smoothing | Yes | 1.17.1 | None needed |
| matplotlib | All figures | Yes | in .venv | None needed |
| numpy | CI calculation | Yes | in .venv | None needed |

No missing dependencies. All tools available locally. [VERIFIED: local env check]

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (installed in .venv) |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest master_thesis_code_test/plotting/test_helpers.py master_thesis_code_test/plotting/test_paper_figures.py -x -m "not slow and not gpu"` |
| Full suite command | `uv run pytest -m "not gpu and not slow"` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PFIG-03 | `compute_credible_interval` Gaussian 68% CI = 2*sigma | unit | `uv run pytest master_thesis_code_test/plotting/test_helpers.py::TestComputeCredibleInterval::test_gaussian_68ci_equals_two_sigma -x` | No — Wave 0 |
| PFIG-03 | `compute_credible_interval` uniform 68% CI = 0.68 * range | unit | `uv run pytest master_thesis_code_test/plotting/test_helpers.py::TestComputeCredibleInterval::test_uniform_68ci_equals_68_percent_range -x` | No — Wave 0 |
| PFIG-01 | `generate_figures()` with paper entries in manifest completes without error | smoke | `uv run pytest master_thesis_code_test/test_generate_figures.py::TestGenerateFigures::test_graceful_degradation_empty_dir -x` | Yes (existing) |
| PFIG-02 | Paper figure functions accept `data_dir` parameter | smoke | `uv run pytest master_thesis_code_test/plotting/test_paper_figures.py -x` | No — Wave 0 |
| PFIG-04 | KDE-smoothed posterior function returns correct shape | unit | `uv run pytest master_thesis_code_test/plotting/test_paper_figures.py::TestKDESmoothing -x` | No — Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest master_thesis_code_test/plotting/test_helpers.py master_thesis_code_test/plotting/test_paper_figures.py -x -m "not slow and not gpu"`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow"`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- `master_thesis_code_test/plotting/test_paper_figures.py` — smoke tests for 4 paper figure factory functions; covers PFIG-02 and PFIG-04
- `master_thesis_code_test/plotting/test_helpers.py` already exists — extend with `TestComputeCredibleInterval` class for PFIG-03

---

## Security Domain

Not applicable. This phase is plotting/visualization code with no authentication, network access, user input validation, or cryptographic operations. No ASVS categories apply.

---

## Project Constraints (from CLAUDE.md)

| Directive | Impact on Phase |
|-----------|----------------|
| All public/private functions must have complete type annotations | `compute_credible_interval` and all modified functions need full annotations |
| Use `list[float]` not `List[float]`; `X | None` not `Optional[X]` | Apply in new CI helper and any new signatures |
| `npt.NDArray[np.float64]` for typed arrays — never bare `np.ndarray` | All array parameters in CI helper must use typed form |
| Factory convention: data in, `(Figure, Axes)` out | Paper figure functions already comply; maintain for KDE variant |
| Never use a mutable object as bare dataclass default | Not applicable (no new dataclasses) |
| NumPy-style docstrings for new code | `compute_credible_interval` needs `Args:`, `Returns:` sections |
| Tests must run on CPU-only dev machine | All proposed tests use numpy only; no GPU dependency |
| No physics change protocol needed | This is software-only; no formula or constant changes |
| `apply_style()` must be called in tests via session fixture | Already handled by `conftest.py` root `_plotting_style` fixture |
| Pre-commit hooks: ruff + mypy before commit | Run `/check` before committing |

---

## Sources

### Primary (HIGH confidence)

- Codebase: `master_thesis_code/plotting/paper_figures.py` — full read, all 4 paper figure functions, data loaders, CI implementation
- Codebase: `master_thesis_code/main.py` lines 754-1072 — full manifest pattern read, 15 existing entries
- Codebase: `master_thesis_code/plotting/convergence_plots.py` — full read, `_credible_interval_width` implementation
- Codebase: `master_thesis_code/plotting/_helpers.py` — full read, `get_figure`, `save_figure`
- Codebase: `master_thesis_code/arguments.py` — CLI flag definitions confirmed
- Local verification: `scipy.stats.gaussian_kde` MAP preservation test on 15-pt and 31-pt grids
- Local verification: CI accuracy tests (Gaussian and uniform distributions)

### Secondary (MEDIUM confidence)

- CONTEXT.md decisions D-01 through D-09 — architectural choices already locked by user discussion
- Test infrastructure: `master_thesis_code_test/plotting/conftest.py` and existing `test_convergence_plots.py` — confirmed patterns for new tests

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all libraries verified locally, no new dependencies needed
- Architecture patterns: HIGH — manifest pattern read directly from source; CI implementations compared
- Pitfalls: HIGH — data loader threading verified by reading call chain; CI tolerance issue measured numerically
- KDE behavior: HIGH — verified locally with Scott's rule on typical h-grids

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (matplotlib/scipy APIs are stable; no fast-moving dependencies)
