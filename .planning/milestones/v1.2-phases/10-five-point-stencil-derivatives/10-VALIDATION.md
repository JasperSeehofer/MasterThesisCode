---
phase: 10
slug: five-point-stencil-derivatives
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` `[tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest -m "not gpu and not slow" -x -q` |
| **Full suite command** | `uv run pytest -m "not gpu and not slow"` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -m "not gpu and not slow" -x -q`
- **After every plan wave:** Run `uv run pytest -m "not gpu and not slow"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | PHYS-01 | unit | `uv run pytest master_thesis_code_test/parameter_estimation/ -x -q` | ❌ W0 | ⬜ pending |
| 10-01-02 | 01 | 1 | PHYS-01 | unit | `uv run pytest master_thesis_code_test/parameter_estimation/ -x -q` | ❌ W0 | ⬜ pending |
| 10-01-03 | 01 | 1 | PHYS-03 | integration | `uv run pytest master_thesis_code_test/ -x -q -k "timeout or crb"` | ❌ W0 | ⬜ pending |
| 10-01-04 | 01 | 1 | PHYS-01 | unit | `uv run pytest master_thesis_code_test/parameter_estimation/ -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `master_thesis_code_test/parameter_estimation/test_five_point_stencil.py` — stubs for PHYS-01 (derivative toggle, API match, condition number logging)
- [ ] `master_thesis_code_test/test_timeout.py` — stubs for PHYS-03 (timeout value >= 120s)

*Existing test infrastructure (conftest.py, fixtures) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU waveform generation with 5-point stencil | PHYS-01 | Requires CUDA GPU | Run on cluster: `uv run pytest -m gpu -k stencil` |
| CRB wall time under SLURM limits | PHYS-03 | Requires cluster job | Phase 11 validation campaign |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
