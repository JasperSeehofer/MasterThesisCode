---
phase: 16
slug: data-layer-fisher
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 16 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (via uv run) |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest master_thesis_code_test/plotting/ -m "not gpu and not slow" -x` |
| **Full suite command** | `uv run pytest -m "not gpu and not slow"` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest master_thesis_code_test/plotting/ -m "not gpu and not slow" -x`
- **After every plan wave:** Run `uv run pytest -m "not gpu and not slow"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 16-01-01 | 01 | 1 | FISH-01 | unit | `uv run pytest master_thesis_code_test/plotting/test_data.py -x` | ❌ W0 | ⬜ pending |
| 16-01-02 | 01 | 1 | FISH-01 | unit | `uv run pytest master_thesis_code_test/plotting/test_data.py::test_diagonal_positive -x` | ❌ W0 | ⬜ pending |
| 16-01-03 | 01 | 1 | FISH-01 | unit | `uv run pytest master_thesis_code_test/plotting/test_data.py::test_roundtrip -x` | ❌ W0 | ⬜ pending |
| 16-02-01 | 02 | 2 | FISH-02 | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_fisher_ellipses -x` | ❌ W0 | ⬜ pending |
| 16-02-02 | 02 | 2 | FISH-02 | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_fisher_ellipses_multi -x` | ❌ W0 | ⬜ pending |
| 16-03-01 | 03 | 2 | FISH-04 | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_characteristic_strain -x` | ❌ W0 | ⬜ pending |
| 16-03-02 | 03 | 2 | FISH-04 | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_strain_three_curves -x` | ❌ W0 | ⬜ pending |
| 16-04-01 | 04 | 2 | FISH-05 | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_uncertainties_violin -x` | ❌ W0 | ⬜ pending |
| 16-04-02 | 04 | 2 | FISH-05 | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_uncertainties_bar -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `master_thesis_code_test/plotting/test_data.py` — stubs for FISH-01 (covariance reconstruction, constants)
- [ ] `master_thesis_code_test/plotting/test_fisher_plots.py` — stubs for FISH-02, FISH-04, FISH-05
- [ ] Fixtures in `conftest.py`: `sample_crb_row` (pd.Series with 105 delta columns + 14 params + metadata), `sample_crb_dataframe` (5-10 rows)

*Existing infrastructure covers test framework requirements (pytest already installed).*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Ellipses visually correct orientation | FISH-02 | Requires visual inspection | Generate plot with known covariance, compare ellipse tilt to expected angle |
| Strain plot log-log scale readable | FISH-04 | Visual quality check | Verify axis labels, legend, line styles render correctly |
| Violin bodies don't extend beyond physical bounds | FISH-05 | Visual quality check | Generate plot with real CRB data, check for unreasonable tails |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
