# Phase 14: Test Infrastructure & Safety Net - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

A safety net of plot smoke tests so that style and infrastructure changes in later phases (15-19) cannot silently break existing thesis-critical figures. This phase adds tests only — no production code changes.

</domain>

<decisions>
## Implementation Decisions

### Test Data Strategy
- **D-01:** Use shared pytest fixtures in `master_thesis_code_test/plotting/conftest.py` for reusable test data (fake DataFrames, arrays, galaxy catalogs). Extend the existing conftest infrastructure rather than inlining data in every test.

### Test Granularity
- **D-02:** One smoke test per plot factory function (22 individual tests). Each test calls the factory with minimal valid data and asserts a `(Figure, Axes)` return without error. No content checks (labels, legends, titles) — pure smoke tests.
- **D-03:** The rcParams regression test pins all 16 settings from `emri_thesis.mplstyle` — the test mirrors the style file exactly, catching any unintentional drift.

### Test Organization
- **D-04:** One test file per plot module, mirroring the source layout:
  - `test_bayesian_plots.py` (5 factories)
  - `test_catalog_plots.py` (4 factories)
  - `test_evaluation_plots.py` (5 factories)
  - `test_model_plots.py` (4 factories)
  - `test_physical_relations_plots.py` (1 factory)
  - `test_simulation_plots.py` (4 factories)
- **D-05:** rcParams regression test added to existing `test_style.py` (already covers `apply_style`, `get_figure`, `save_figure`).
- **D-06:** Plotting-specific fixtures live in `master_thesis_code_test/plotting/conftest.py`. Session-scoped `apply_style()` fixture already exists per project convention.

### Claude's Discretion
- Exact synthetic data shapes and values for each factory's fixtures — Claude determines what minimal valid input each function needs by reading the source.
- Whether to use `@pytest.mark.parametrize` within a module's test file for functions with similar signatures.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Plot modules (source of truth for factory signatures)
- `master_thesis_code/plotting/bayesian_plots.py` — 5 factory functions
- `master_thesis_code/plotting/catalog_plots.py` — 4 factory functions
- `master_thesis_code/plotting/evaluation_plots.py` — 5 factory functions
- `master_thesis_code/plotting/model_plots.py` — 4 factory functions
- `master_thesis_code/plotting/physical_relations_plots.py` — 1 factory function
- `master_thesis_code/plotting/simulation_plots.py` — 4 factory functions (includes `_fig_from_ax`)

### Style infrastructure
- `master_thesis_code/plotting/emri_thesis.mplstyle` — 16 rcParams to pin in regression test
- `master_thesis_code/plotting/_style.py` — `apply_style()` implementation
- `master_thesis_code/plotting/_helpers.py` — `get_figure()`, `save_figure()`, `make_colorbar()`

### Existing tests
- `master_thesis_code_test/plotting/test_style.py` — 11 existing tests for style/helpers; add rcParams regression here
- `master_thesis_code_test/conftest.py` — root conftest with `xp` fixture and test markers

### Requirements
- `.planning/REQUIREMENTS.md` §Test Infrastructure — TEST-01, TEST-02

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `master_thesis_code_test/conftest.py` — root conftest with session fixtures and pytest markers
- `master_thesis_code_test/plotting/test_style.py` — existing pattern for plot tests (apply_style + plt.close)
- `apply_style()` fixture convention — already established in project (feedback memory)

### Established Patterns
- All plot factories follow `data in, (fig, ax) out` pattern
- Tests use `plt.close(fig)` in teardown to prevent memory leaks
- `@pytest.mark.gpu` and `@pytest.mark.slow` markers exclude tests from fast suite

### Integration Points
- New test files register in `master_thesis_code_test/plotting/` alongside `test_style.py`
- All new tests must pass `uv run pytest -m "not gpu and not slow"` (CI gate)
- No GPU markers needed — plot functions are CPU-only

</code_context>

<specifics>
## Specific Ideas

No specific requirements — standard smoke test approach with the project's existing conventions.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 14-test-infrastructure*
*Context gathered: 2026-04-01*
