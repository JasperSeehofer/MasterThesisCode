---
phase: 29
slug: style-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-07
---

# Phase 29 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `uv run pytest master_thesis_code_test/test_style.py -v` |
| **Full suite command** | `uv run pytest -m "not gpu and not slow" -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest master_thesis_code_test/test_style.py -v`
- **After every plan wave:** Run `uv run pytest -m "not gpu and not slow" -v`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 29-01-01 | 01 | 1 | STYL-01 | — | N/A | unit | `uv run pytest master_thesis_code_test/test_style.py::test_rcparams_snapshot -v` | ✅ | ⬜ pending |
| 29-01-02 | 01 | 1 | STYL-01 | — | N/A | unit | `uv run pytest master_thesis_code_test/test_style.py::test_apply_style_default_unchanged -v` | ✅ | ⬜ pending |
| 29-01-03 | 01 | 1 | STYL-02 | — | N/A | unit | `uv run pytest master_thesis_code_test/test_style.py -v` | ✅ | ⬜ pending |
| 29-01-04 | 01 | 1 | STYL-03 | — | N/A | unit | `uv run pytest master_thesis_code_test/test_style.py -v` | ✅ | ⬜ pending |
| 29-01-05 | 01 | 1 | STYL-01 | — | No Type 3 fonts in PDF | manual | `pdffonts output.pdf` | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements. `test_style.py` already exists with regression guard tests that will be updated alongside the style changes.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| No Type 3 fonts in PDF output | STYL-01 | Requires `pdffonts` CLI tool | Generate a test PDF, run `pdffonts test.pdf`, verify no "Type 3" entries |
| Visual appearance of figures | STYL-01/02/03 | Subjective visual quality | Generate sample figures at REVTeX widths, inspect font sizes and color contrast |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
