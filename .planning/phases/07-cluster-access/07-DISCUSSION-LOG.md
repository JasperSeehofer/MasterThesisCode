# Phase 7: Cluster Access - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-27
**Phase:** 07-cluster-access
**Areas discussed:** Claude's cluster access method, Preflight verification scope, VPN/network requirements

---

## Claude's Cluster Access Method

| Option | Description | Selected |
|--------|-------------|----------|
| ssh via Bash tool | Claude runs `ssh bwunicluster '<cmd>'` directly via Bash tool | Yes |
| MCP SSH server | Dedicated MCP server managing SSH connection | |
| Manual copy-paste | Claude tells user what to run, user pastes output back | |

**User's choice:** ssh via Bash tool
**Notes:** Simplest approach, no extra tooling needed

---

## Preflight Verification Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Basic | Modules load, sinfo shows gpu_h100, ws_find returns path | |
| Venv check | Basic + activate venv, python imports few and cupy | Yes |
| Smoke test job | Venv check + submit 1-task 1-step simulation | |

**User's choice:** Venv check (Recommended)
**Notes:** Smoke test overlaps with Phase 8

---

## VPN/Network Requirements

| Option | Description | Selected |
|--------|-------------|----------|
| VPN already running | User has VPN sorted, Claude just needs SSH | Yes |
| VPN needed first | Need to connect VPN before SSH works | |
| No VPN required | Direct access without VPN | |

**User's choice:** VPN already running
**Notes:** User started VPN during discussion session. No shell restart needed.

---

## Additional Context

- SSH key not yet registered on bwUniCluster portal (ACCESS-01 pending)
- No ~/.ssh/config entry exists yet (ACCESS-02 pending)
- Both are human-verify tasks in the plan

## Claude's Discretion

- SSH config entry format details
- Preflight script structure
- Error reporting format

## Deferred Ideas

None
