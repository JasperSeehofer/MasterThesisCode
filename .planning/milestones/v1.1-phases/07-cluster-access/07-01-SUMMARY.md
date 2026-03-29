---
phase: 07-cluster-access
plan: 01
subsystem: infra
tags: [ssh, bwunicluster, hpc, controlmaster, slurm]

requires:
  - phase: 03-cluster-environment
    provides: cluster venv with few+cupy, modules.sh, workspace setup
provides:
  - SSH config with ControlMaster for bwUniCluster 3.0
  - Verified cluster environment (modules, GPU partition, workspace, venv+imports)
  - Claude can execute cluster commands via `ssh bwunicluster '<cmd>'`
affects: [08-simulation-campaign]

tech-stack:
  added: []
  patterns: [ssh-controlmaster-2fa-reuse]

key-files:
  created:
    - ~/.ssh/config
    - ~/.ssh/sockets/
  modified: []

key-decisions:
  - "Used scontrol instead of sinfo for GPU partition check (sinfo returns 'Access/permission denied' on login nodes)"
  - "ControlPersist 8h matches bwUniCluster 2FA token validity window"

patterns-established:
  - "SSH command pattern: ssh bwunicluster '<cmd>' for all cluster interactions"
  - "Module loading: source ~/MasterThesisCode/cluster/modules.sh before any cluster work"

requirements-completed: [ACCESS-01, ACCESS-02, ACCESS-03, ACCESS-04]

duration: 10min
completed: 2026-03-28
---

# Phase 7: Cluster Access Summary

**SSH ControlMaster to bwUniCluster 3.0 with 2FA session reuse, all 5 preflight checks passing (modules, GPU partition, workspace, venv+imports)**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-27T23:50:00Z
- **Completed:** 2026-03-28T00:00:00Z
- **Tasks:** 3
- **Files modified:** 2 (outside repo: ~/.ssh/config, ~/.ssh/sockets/)

## Accomplishments
- SSH config created with `bwunicluster` alias and ControlMaster (8h persistence)
- User registered SSH key on bwIDM portal and completed initial 2FA login
- All 5 preflight checks pass: SSH connectivity, module loading, GPU partition (gpu_h100, 12 nodes), workspace, venv with few+cupy 13.6.0

## Task Commits

No repo commits — all changes were to ~/.ssh/ (outside repository).

## Files Created/Modified
- `~/.ssh/config` — SSH config with bwunicluster alias, ControlMaster, ControlPersist 8h
- `~/.ssh/sockets/` — ControlMaster socket directory (700 permissions)

## Decisions Made
- `sinfo` returns "Access/permission denied" on login nodes — used `scontrol show partition gpu_h100` instead, which confirmed partition exists and is UP
- SSH hostname `bwunicluster.scc.kit.edu` resolves correctly (same host as `uc3.scc.kit.edu`)

## Deviations from Plan
None — plan executed as specified. Minor adaptation: scontrol used instead of sinfo for GPU partition verification.

## Issues Encountered
- `sinfo -p gpu_h100` returned "Access/permission denied" on login nodes. Resolved by using `scontrol show partition gpu_h100` which showed full partition details including State=UP, 12 nodes, H100 GPUs.

## Next Phase Readiness
- SSH connectivity established — Claude can execute `ssh bwunicluster '<cmd>'` via Bash tool
- ControlMaster session valid for 8 hours after 2FA login
- All cluster prerequisites verified — ready for Phase 8 (Simulation Campaign)
- Note: User must re-authenticate (`ssh bwunicluster` interactively) after ControlPersist expires (8h)

---
*Phase: 07-cluster-access*
*Completed: 2026-03-28*
