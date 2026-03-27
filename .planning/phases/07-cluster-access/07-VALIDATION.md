---
phase: 7
slug: cluster-access
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-27
---

# Phase 7 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Manual verification via SSH commands |
| **Config file** | N/A (infrastructure phase, not code) |
| **Quick run command** | `ssh bwunicluster 'echo ok'` |
| **Full suite command** | Preflight check sequence (5 SSH commands) |
| **Estimated runtime** | ~30 seconds (SSH round-trips) |

---

## Sampling Rate

- **After every task commit:** Run `ssh bwunicluster 'echo ok'` (if SSH is configured)
- **After every plan wave:** Run full preflight check sequence
- **Before `/gsd:verify-work`:** All preflight checks must pass
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 07-01-01 | 01 | 1 | ACCESS-01 | manual-only | N/A — human action on bwIDM portal | N/A | ⬜ pending |
| 07-01-02 | 01 | 1 | ACCESS-02 | smoke | `ssh bwunicluster 'echo ok'` | N/A — infra | ⬜ pending |
| 07-01-03 | 01 | 1 | ACCESS-03 | smoke | `ssh bwunicluster 'source cluster/modules.sh && sinfo -p gpu_h100 --noheader'` | N/A — infra | ⬜ pending |
| 07-01-04 | 01 | 1 | ACCESS-03 | smoke | `ssh bwunicluster 'source .venv/bin/activate && python -c "import few; import cupy"'` | N/A — infra | ⬜ pending |
| 07-01-05 | 01 | 1 | ACCESS-04 | smoke | Claude runs `ssh bwunicluster 'hostname'` and receives output | N/A — infra | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. This is an infrastructure/connectivity phase, not a code phase requiring test file stubs.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| SSH key registered on bwIDM portal | ACCESS-01 | Requires human web browser interaction with bwIDM portal | 1. Navigate to https://login.bwidm.de/user/ssh-keys.xhtml 2. Upload `~/.ssh/id_ed25519.pub` 3. Confirm key appears in portal list |
| Initial 2FA-unlocked SSH login | ACCESS-02 | Requires interactive TOTP entry | 1. Run `ssh bwunicluster` 2. Enter TOTP code when prompted 3. Verify shell access |

*All subsequent verifications (ACCESS-03, ACCESS-04) are automated via SSH commands over the established ControlMaster session.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
