---
phase: 35
slug: unified-pipeline-paper-figures
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-08
---

# Phase 35 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest -m "not gpu and not slow" -x` |
| **Full suite command** | `uv run pytest -m "not gpu and not slow"` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -m "not gpu and not slow" -x`
- **After every plan wave:** Run `uv run pytest -m "not gpu and not slow"`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 35-01-01 | 01 | 1 | PFIG-03 | — | N/A | unit | `uv run pytest master_thesis_code_test/plotting/test_helpers.py -k credible_interval` | ❌ W0 | ⬜ pending |
| 35-01-02 | 01 | 1 | PFIG-01 | — | N/A | unit | `uv run pytest master_thesis_code_test/plotting/ -k paper` | ✅ | ⬜ pending |
| 35-02-01 | 02 | 2 | PFIG-02 | — | N/A | smoke | `uv run pytest master_thesis_code_test/plotting/test_paper_figures.py` | ✅ | ⬜ pending |
| 35-02-02 | 02 | 2 | PFIG-04 | — | N/A | unit | `uv run pytest -k kde_smooth` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `master_thesis_code_test/plotting/test_helpers.py` — add `test_compute_credible_interval_*` stubs
- [ ] `master_thesis_code_test/plotting/test_paper_figures.py` — add KDE smoothing test stubs

*Existing test infrastructure (conftest.py, fixtures) covers all phase requirements.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visual polish of paper figures | PFIG-02 | Aesthetic judgment | Open generated PDFs, verify layout/spacing/annotations match publication quality |
| pdffonts Type 3 check | PFIG-02 | Requires generated PDFs | `pdffonts <figure>.pdf` — verify zero Type 3 fonts |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
