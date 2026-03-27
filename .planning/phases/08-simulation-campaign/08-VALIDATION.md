---
phase: 8
slug: simulation-campaign
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-28
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml [tool.pytest.ini_options]` |
| **Quick run command** | `uv run pytest -m "not gpu and not slow"` |
| **Full suite command** | `uv run pytest` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -m "not gpu and not slow"` (only if code is changed)
- **After every plan wave:** N/A (single-wave phase expected)
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 08-01-01 | 01 | 1 | SIM-01 | manual-only | N/A -- cluster execution | N/A | ⬜ pending |
| 08-01-02 | 01 | 1 | SIM-02 | manual-only | N/A -- cluster execution | N/A | ⬜ pending |
| 08-01-03 | 01 | 1 | SIM-03 | manual-only | N/A -- cluster validation | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. This phase validates existing code by running it on the cluster — no new tests are needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Simulation tasks complete with timing data | SIM-01 | Requires GPU cluster execution | `sacct -j <job_id>` shows all tasks COMPLETED; `run_metadata_*.json` files exist |
| Evaluation produces H0 posterior | SIM-02 | Requires cluster-generated Cramér-Rao bounds | `ls $RUN_DIR/simulations/posteriors/h_0_73.json` exists and is non-empty |
| Results pass sanity checks | SIM-03 | Validation is on cluster output data | SNR > 0, detection rate 1-30%, H0 posterior peak in [0.6, 0.9] |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
