---
phase: 21
slug: analysis-post-processing
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-02
---

# Phase 21 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pyproject.toml |
| **Quick run command** | `uv run pytest -m "not gpu and not slow" -x` |
| **Full suite command** | `uv run pytest -m "not gpu and not slow"` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -m "not gpu and not slow" -x`
- **After every plan wave:** Run `uv run pytest -m "not gpu and not slow"`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 21-01-01 | 01 | 1 | ANAL-01 | integration | `uv run pytest master_thesis_code_test/ -k "diagnostic"` | ❌ W0 | ⬜ pending |
| 21-01-02 | 01 | 1 | ANAL-02 | integration | `uv run pytest master_thesis_code_test/ -k "comparison"` | ❌ W0 | ⬜ pending |
| 21-02-01 | 02 | 1 | POST-01, NFIX-01 | unit | `uv run pytest master_thesis_code_test/ -k "combine"` | ❌ W0 | ⬜ pending |
| 21-02-02 | 02 | 1 | POST-01 | integration | `uv run python -m master_thesis_code results/h_sweep_20260401 --combine` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `master_thesis_code_test/bayesian_inference/test_combine_posteriors.py` — stubs for POST-01, NFIX-01
- [ ] `master_thesis_code_test/bayesian_inference/test_diagnostic_report.py` — stubs for ANAL-01, ANAL-02

*Existing test infrastructure (conftest.py, fixtures) covers framework needs.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Diagnostic report readable | ANAL-01 | Report quality is subjective | Read generated diagnostic_report.md, verify tables render correctly |
| Comparison table completeness | ANAL-02 | Requires human review of MAP values | Verify MAP values match known baselines (0.72/0.86 naive) |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
