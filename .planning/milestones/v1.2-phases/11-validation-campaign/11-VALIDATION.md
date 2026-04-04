---
phase: 11
slug: validation-campaign
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-29
---

# Phase 11 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest -m "not gpu and not slow"` |
| **Full suite command** | `uv run pytest` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -m "not gpu and not slow"` (only if code is changed, e.g., D-08/D-09 fixes)
- **After every plan wave:** N/A (cluster operations, not code changes)
- **Before `/gsd:verify-work`:** Comparison report produced and pass/fail criteria from D-04 met
- **Max feedback latency:** ~15 seconds (CPU tests); cluster run ~2 hours

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| TBD | 01 | 1 | SIM-01 | manual-only | N/A — requires GPU cluster | N/A | ⬜ pending |
| TBD | 01 | 1 | SIM-03 | manual-only | N/A — analysis of cluster output | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. This phase validates existing code by running it on the cluster and analyzing outputs. No new test infrastructure needed. If D-08 or D-09 trigger code changes, those changes are to existing tested code (timeout values, epsilon values) that do not need new test files.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Validation campaign completes with corrected physics | SIM-01 | Requires GPU cluster (bwUniCluster H100) | Submit via `cluster/submit_pipeline.sh --tasks 3 --steps 10 --seed 100`, monitor with `sacct`, verify non-zero detections |
| d_L threshold recalibrated from validation data | SIM-03 | Analysis of cluster output CSV | Compare delta_d_L/d_L distribution from v1.2 run against v1.1 baseline, report percentiles and recommended threshold |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
