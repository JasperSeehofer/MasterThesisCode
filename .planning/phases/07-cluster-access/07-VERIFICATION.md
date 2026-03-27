---
status: passed
phase: 07-cluster-access
verified: 2026-03-28
score: 4/4
---

# Phase 7: Cluster Access — Verification

## Goal
Establish SSH connectivity from local machine to bwUniCluster 3.0 and verify the cluster environment is ready for simulation.

## Must-Haves Verification

### ACCESS-01: SSH key registered via bwUniCluster user portal
- **Status:** PASS
- **Evidence:** User registered ed25519 key on bwIDM portal. SSH authentication succeeded — `ssh bwunicluster 'hostname'` returned `uc3n990.localdomain` without password prompt.

### ACCESS-02: ~/.ssh/config entry configured for direct `ssh bwunicluster` access
- **Status:** PASS
- **Evidence:** `~/.ssh/config` contains:
  - `Host bwunicluster` alias
  - `HostName bwunicluster.scc.kit.edu`
  - `User st_ac147838`
  - `ControlMaster auto` with `ControlPersist 8h`
  - `ControlPath ~/.ssh/sockets/%r@%h-%p`
  - `IdentityFile ~/.ssh/id_ed25519`
  - Permissions: config 600, sockets dir 700

### ACCESS-03: Environment preflight verified
- **Status:** PASS
- **Evidence:** 5/5 preflight checks passed:
  1. SSH connectivity: `hostname` returned `uc3n990.localdomain`
  2. Module loading: gnu/14.2, cuda/12.8, python/3.13.3-gnu-14.2 all loaded
  3. GPU partition: `scontrol show partition gpu_h100` — 12 nodes, State=UP, H100 GPUs
  4. Workspace: `ws_find emri` returned `/pfs/work9/workspace/scratch/st_ac147838-emri`
  5. Venv+imports: `import few; import cupy` succeeded, cupy 13.6.0

### ACCESS-04: Claude SSH integration configured for direct cluster command execution
- **Status:** PASS
- **Evidence:** All preflight checks were executed by Claude via `ssh bwunicluster '<cmd>'` through the Bash tool. ControlMaster session reuse eliminated 2FA prompts for subsequent commands.

## Score: 4/4 must-haves verified

## Notes
- `sinfo` command returns "Access/permission denied" on login nodes, but `scontrol show partition` works. Job submission (`sbatch`) is unaffected — this is a display permission only.
- ControlMaster session expires after 8 hours. User must re-authenticate interactively before Phase 8 if the session has expired.
- No code changes were made to the repository — all modifications were to `~/.ssh/` (outside repo).

## Human Verification Items
None — all verification was performed during execution.
