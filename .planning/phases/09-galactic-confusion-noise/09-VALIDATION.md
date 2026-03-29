---
phase: 9
slug: galactic-confusion-noise
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest master_thesis_code_test/LISA_configuration_test.py -m "not gpu and not slow" -x` |
| **Full suite command** | `uv run pytest -m "not gpu and not slow"` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest master_thesis_code_test/LISA_configuration_test.py -m "not gpu and not slow" -x`
- **After every plan wave:** Run `uv run pytest -m "not gpu and not slow"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | PHYS-02 | unit | `uv run pytest master_thesis_code_test/LISA_configuration_test.py -x` | ✅ | ⬜ pending |
| 09-01-02 | 01 | 1 | PHYS-02 | unit | `uv run pytest master_thesis_code_test/LISA_configuration_test.py -x` | ✅ | ⬜ pending |
| 09-01-03 | 01 | 1 | PHYS-02 | regression | `uv run pytest -m "not gpu and not slow"` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.*

Existing test file `master_thesis_code_test/LISA_configuration_test.py` has 7 CPU tests covering PSD positivity and basic properties. New confusion noise tests will be added to the same file.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PSD plot shows confusion noise bump at 0.1-3 mHz | PHYS-02 | Visual verification of spectral shape | Generate PSD plot with and without confusion noise; verify visible bump in mHz band |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
