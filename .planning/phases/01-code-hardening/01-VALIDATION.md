---
phase: 1
slug: code-hardening
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (with pytest-cov, pytest-benchmark) |
| **Config file** | `pyproject.toml` [tool.pytest.ini_options] |
| **Quick run command** | `uv run pytest -m "not gpu and not slow" --tb=short -q` |
| **Full suite command** | `uv run pytest --tb=short -q` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -m "not gpu and not slow" --tb=short -q`
- **After every plan wave:** Run `uv run pytest -m "not gpu and not slow" --tb=short -q` + `uv run mypy master_thesis_code/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | CODE-02 | smoke | `uv run python -c "from master_thesis_code.memory_management import MemoryManagement"` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | CODE-02 | unit | `uv run pytest master_thesis_code_test/test_memory_management.py -x` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | CODE-01 | unit | `uv run pytest master_thesis_code_test/test_arguments.py -x` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | CODE-01 | smoke | `uv run python -m master_thesis_code --help \| grep -E "use_gpu\|num_workers"` | ❌ W0 | ⬜ pending |
| 01-02-03 | 02 | 1 | CODE-03 | unit | `uv run pytest master_thesis_code_test/test_arguments.py::test_num_workers_default -x` | ❌ W0 | ⬜ pending |
| REGR | - | - | - | regression | `uv run pytest -m "not gpu and not slow" --tb=short -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `master_thesis_code_test/test_memory_management.py` — stubs for CODE-02 (CPU instantiation, no-op methods)
- [ ] `master_thesis_code_test/test_arguments.py` — stubs for CODE-01 (`--use_gpu` flag), CODE-03 (`--num_workers` flag, default logic)

*Existing test infrastructure covers regression requirement.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| GPU acceleration works on cluster | CODE-01 | Requires CUDA hardware | Run `python -m master_thesis_code <dir> --simulation_steps 1 --use_gpu` on GPU node |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
