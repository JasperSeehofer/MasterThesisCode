# Phase 7: Cluster Access - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish SSH connectivity from the local dev machine to bwUniCluster 3.0 and verify the cluster environment is ready for simulation. Delivers: SSH key registration (manual), SSH config entry, and an automated preflight verification that confirms modules, venv, and GPU partition are operational.

</domain>

<decisions>
## Implementation Decisions

### Claude's Cluster Access Method
- **D-01:** Use `ssh bwunicluster '<cmd>'` via Bash tool. No MCP SSH server or manual copy-paste workflow. This requires ACCESS-02 (SSH config alias) to be working first. Claude can then autonomously run preflight checks and later submit jobs in Phase 8.

### Preflight Verification Scope
- **D-02:** Preflight passes when ALL of the following succeed via SSH:
  1. `module load` for all required modules (compiler/gnu, cuda, gsl, python)
  2. `sinfo -p gpu_h100` shows the GPU partition
  3. `ws_find emri` returns a workspace path
  4. Venv activation + `python -c "import few; import cupy"` succeeds
- **D-03:** No smoke test job submission in this phase — that overlaps with Phase 8's test run.

### VPN/Network
- **D-04:** VPN is handled by the user outside this phase's scope. Claude assumes SSH connectivity is available once ACCESS-01 and ACCESS-02 are complete.

### SSH Setup (Human Tasks)
- **D-05:** ACCESS-01 (SSH key registration via bwUniCluster portal) and ACCESS-02 (SSH config entry) are manual human-verify tasks. The user has not yet registered their SSH key — the plan must include clear instructions for both steps.
- **D-06:** SSH config alias should be `bwunicluster` (matching `cluster/README.md` references). Full hostname: `bwunicluster.scc.kit.edu`.

### Claude's Discretion
- Exact SSH config entry format (ProxyJump, IdentityFile path, etc.)
- Preflight script structure (inline checks vs dedicated script)
- Error reporting format for failed preflight steps

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` -- ACCESS-01, ACCESS-02, ACCESS-03, ACCESS-04

### Prior Phase Context
- `.planning/phases/03-cluster-environment/03-CONTEXT.md` -- Module names (D-01), workspace integration (D-04, D-05), uv installation (D-03)
- `.planning/phases/03-cluster-environment/03-01-SUMMARY.md` -- What was actually built for cluster environment

### Codebase References
- `cluster/modules.sh` -- Module load sequence, $WORKSPACE export
- `cluster/setup.sh` -- One-time setup (uv, workspace, venv)
- `cluster/README.md` -- Quickstart guide with SSH hostname and rsync examples
- `~/.ssh/config` -- User's SSH configuration (to be created/updated)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cluster/modules.sh` -- Already exports `$WORKSPACE`, `$PROJECT_ROOT`, `$VENV_PATH`; can be sourced in preflight checks
- `cluster/setup.sh` -- Already runs `uv sync --extra gpu`; preflight can verify its output
- `cluster/README.md` -- Documents `bwunicluster.scc.kit.edu` as the hostname

### Established Patterns
- All cluster scripts `source cluster/modules.sh` as first step (Phase 3/4 convention)
- `$WORKSPACE` is single source of truth for output paths
- Module names: `compiler/gnu/14.2`, `devel/cuda/12.8`, `numlib/gsl`, `devel/python/3.13.3-gnu-14.2`

### Integration Points
- SSH config alias `bwunicluster` used in `cluster/README.md` rsync examples
- Preflight verification feeds confidence into Phase 8 job submission
- `ssh bwunicluster '<cmd>'` pattern will be reused throughout Phase 8

</code_context>

<specifics>
## Specific Ideas

- User's VPN is already running but SSH key not yet registered on the portal
- No existing `~/.ssh/config` entry for bwunicluster
- Phase 3 flagged "GSL module name unconfirmed" -- preflight will resolve this

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 07-cluster-access*
*Context gathered: 2026-03-27*
