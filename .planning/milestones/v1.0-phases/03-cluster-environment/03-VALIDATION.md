---
phase: 3
slug: cluster-environment
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-26
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | bash syntax checking + manual cluster verification |
| **Config file** | none — shell scripts, not Python tests |
| **Quick run command** | `bash -n cluster/modules.sh && bash -n cluster/setup.sh` |
| **Full suite command** | `source cluster/modules.sh` on bwUniCluster login node |
| **Estimated runtime** | ~2 seconds (syntax check); minutes (cluster verification) |

---

## Sampling Rate

- **After every task commit:** Run `bash -n cluster/*.sh` (syntax validation, runs on dev machine)
- **After every plan wave:** N/A (full validation requires cluster access)
- **Before `/gsd:verify-work`:** Full suite on bwUniCluster login node
- **Max feedback latency:** 2 seconds (syntax check)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | ENV-01 | syntax | `bash -n cluster/modules.sh` | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | ENV-03 | syntax+grep | `grep -q 'WORKSPACE' cluster/modules.sh` | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | ENV-02 | syntax | `bash -n cluster/setup.sh` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `cluster/modules.sh` — created by first task
- [ ] `cluster/setup.sh` — created by first task

*Shell scripts are greenfield — Wave 0 is the creation of the files themselves.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| `source cluster/modules.sh` loads all modules | ENV-01 | Requires bwUniCluster login node with Lmod | SSH to cluster, `source cluster/modules.sh`, check exit code 0 |
| `cluster/setup.sh` produces working `.venv` | ENV-02 | Requires cluster with CUDA, Python modules | Run `cluster/setup.sh`, then `uv run python -c "import master_thesis_code"` |
| `$WORKSPACE` resolves to bwHPC workspace path | ENV-03 | `ws_find` only available on bwHPC | After sourcing `modules.sh`, verify `echo $WORKSPACE` shows `/pfs/work9/...` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 2s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
