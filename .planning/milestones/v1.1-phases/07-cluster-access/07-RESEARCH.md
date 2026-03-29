# Phase 7: Cluster Access - Research

**Researched:** 2026-03-27
**Domain:** SSH connectivity, bwHPC authentication, cluster environment verification
**Confidence:** HIGH

## Summary

Phase 7 establishes SSH connectivity from the local dev machine to bwUniCluster 3.0 and verifies the cluster environment is ready for simulation. The phase involves three categories of work: (1) human-performed SSH key registration via the bwIDM portal, (2) local SSH config creation with ControlMaster for session multiplexing, and (3) automated preflight verification that Claude can run via `ssh bwunicluster '<cmd>'`.

Critical discovery: bwUniCluster 3.0 uses **two-factor authentication** (TOTP + service password) even with SSH keys. Interactive SSH keys must be "unlocked" by a 2FA login, after which they remain valid for **8 hours**. SSH ControlMaster multiplexing is essential -- it keeps the authenticated session alive so Claude can execute multiple commands without re-authenticating. The cluster hostname is `uc3.scc.kit.edu` (not `bwunicluster.scc.kit.edu` as used in cluster/README.md, though the old name may still resolve).

**Primary recommendation:** User performs 2FA-unlocked SSH login with ControlMaster, then Claude runs all preflight checks over the persistent connection within the 8-hour window.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Use `ssh bwunicluster '<cmd>'` via Bash tool. No MCP SSH server or manual copy-paste workflow.
- **D-02:** Preflight passes when ALL of: modules load, `sinfo -p gpu_h100` shows GPU partition, `ws_find emri` returns workspace, venv activation + `import few; import cupy` succeeds.
- **D-03:** No smoke test job submission -- that is Phase 8 scope.
- **D-04:** VPN handled by user outside phase scope. Claude assumes SSH connectivity available.
- **D-05:** ACCESS-01 and ACCESS-02 are manual human-verify tasks. User has not yet registered SSH key.
- **D-06:** SSH config alias should be `bwunicluster`. Full hostname: `bwunicluster.scc.kit.edu`.

### Claude's Discretion
- Exact SSH config entry format (ProxyJump, IdentityFile path, etc.)
- Preflight script structure (inline checks vs dedicated script)
- Error reporting format for failed preflight steps

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ACCESS-01 | SSH key registered via bwUniCluster user portal for passwordless authentication | bwIDM portal at https://login.bwidm.de/user/ssh-keys.xhtml; ed25519 key already exists at ~/.ssh/id_ed25519; keys expire after 180 days; interactive keys require 2FA unlock (8h validity) |
| ACCESS-02 | ~/.ssh/config entry configured for direct `ssh bwunicluster` access | Config format documented below; ControlMaster essential for 2FA session reuse; user has no existing config file |
| ACCESS-03 | Environment preflight verified (modules load, venv exists, GPU partition accessible) | cluster/modules.sh verified on cluster (Phase 3); all checks can run via `ssh bwunicluster '<cmd>'`; specific commands documented below |
| ACCESS-04 | Claude SSH integration configured for direct cluster command execution | Bash tool with `ssh bwunicluster '<cmd>'` pattern; ControlMaster keeps session alive for 8h after user's 2FA login |
</phase_requirements>

## Architecture Patterns

### Authentication Flow (bwUniCluster 3.0)

bwUniCluster 3.0 uses centrally-managed SSH keys via bwIDM. The authentication model is:

1. **Upload public key** to https://login.bwidm.de/user/ssh-keys.xhtml
2. **Register as Interactive key** for bwUniCluster 3.0 service
3. **2FA unlock**: First SSH login requires TOTP + service password. After successful 2FA, the interactive key is valid for **8 hours**.
4. **Subsequent connections** within the 8h window authenticate via key only (no 2FA re-prompt).

Self-managed `~/.ssh/authorized_keys` files are **ignored** on bwUniCluster 3.0. All keys must go through the portal.

**Key expiration:** Registered keys expire after **180 days** and must be re-registered.

### Recommended SSH Config

```
Host bwunicluster
    HostName bwunicluster.scc.kit.edu
    User st_ac147838
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

**Why ControlMaster is essential:** Without it, every `ssh bwunicluster '<cmd>'` would trigger a new SSH handshake. With 2FA, the first connection in any 8-hour window requires interactive TOTP + password entry. ControlMaster reuses the authenticated connection for all subsequent commands, which is what enables Claude to run commands non-interactively via the Bash tool.

**ControlPath directory:** `~/.ssh/sockets/` must be created with mode 700.

**ControlPersist 8h:** Matches the bwUniCluster 2FA validity window. The master connection stays alive for 8 hours even after the initial terminal is closed.

**ServerAliveInterval:** Prevents idle timeouts from firewalls or the cluster's SSH daemon.

### Hostname Note

The bwHPC wiki documents `uc3.scc.kit.edu` as the primary hostname for bwUniCluster 3.0. However, `bwunicluster.scc.kit.edu` is referenced in the project's `cluster/README.md` and was chosen in CONTEXT.md (D-06). Both likely resolve to the same login nodes. The config should use `bwunicluster.scc.kit.edu` per user decision.

### Username

From `cluster/vpn.sh`, the user's university account is `ac147838@uni-stuttgart.de`. The bwHPC username prefix for University of Stuttgart is `st_`, making the cluster username `st_ac147838`.

### Preflight Verification Pattern

Each check should be a separate SSH command that returns a clear pass/fail:

```bash
# Check 1: Modules load
ssh bwunicluster 'source ~/MasterThesisCode/cluster/modules.sh'

# Check 2: GPU partition visible
ssh bwunicluster 'sinfo -p gpu_h100 --noheader | head -1'

# Check 3: Workspace exists
ssh bwunicluster 'ws_find emri'

# Check 4: Venv + imports work
ssh bwunicluster 'source ~/MasterThesisCode/cluster/modules.sh && source ~/MasterThesisCode/.venv/bin/activate && python -c "import few; import cupy; print(f\"few OK, cupy {cupy.__version__}\")"'
```

**Confidence: HIGH** -- These commands are directly derived from the verified Phase 3 cluster setup (03-01-SUMMARY.md confirms all modules load and imports work).

### Anti-Patterns to Avoid

- **Running SSH without ControlMaster:** Each command would prompt for 2FA. Claude cannot enter TOTP interactively.
- **Hardcoding workspace path:** Use `ws_find emri` dynamically. The path (`/pfs/work9/workspace/scratch/st_ac147838-emri`) could change if workspace is re-allocated.
- **Assuming SSH works without user action:** ACCESS-01 (key registration) and ACCESS-02 (config file) are prerequisites that only the user can complete.
- **Using command keys for Claude access:** Command keys require admin approval, IP restriction, and are limited to a single command with full path. Interactive keys + ControlMaster are simpler and sufficient.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SSH session management | Custom SSH wrapper script | SSH ControlMaster in ~/.ssh/config | Built-in OpenSSH feature, battle-tested, no code to maintain |
| Module environment setup | New module loader | Existing `cluster/modules.sh` | Already verified on cluster (Phase 3), exports all needed vars |
| Environment verification | Ad-hoc SSH commands | Structured preflight script or inline checks with clear reporting | Reusable for Phase 8 and future sessions |

## Common Pitfalls

### Pitfall 1: 2FA Blocks Non-Interactive SSH
**What goes wrong:** Claude tries `ssh bwunicluster '<cmd>'` but gets a TOTP prompt that hangs.
**Why it happens:** No prior 2FA-unlocked session exists, or ControlMaster socket expired.
**How to avoid:** User must perform one interactive `ssh bwunicluster` login first (enters TOTP + password). ControlMaster then keeps the session alive for 8 hours.
**Warning signs:** SSH command hangs with no output, or returns "Permission denied".

### Pitfall 2: Missing ControlPath Directory
**What goes wrong:** SSH fails with "ControlPath ... too long" or "No such file or directory".
**Why it happens:** `~/.ssh/sockets/` directory doesn't exist.
**How to avoid:** Create it: `mkdir -p ~/.ssh/sockets && chmod 700 ~/.ssh/sockets`.
**Warning signs:** First SSH connection fails immediately after config is written.

### Pitfall 3: Key Not Yet Active on Portal
**What goes wrong:** SSH key authentication rejected even though key is uploaded.
**Why it happens:** Key must be registered as Interactive for the specific bwUniCluster 3.0 service, not just uploaded. There may also be a propagation delay.
**How to avoid:** Verify in bwIDM portal that key shows as "Active" for bwUniCluster 3.0 under "Registered Services".
**Warning signs:** "Permission denied (publickey,keyboard-interactive)" after providing correct OTP + password.

### Pitfall 4: Stale ControlMaster Socket
**What goes wrong:** SSH commands fail with "Connection refused" or similar after the 8h window expires.
**Why it happens:** ControlPersist timeout reached; master socket is stale.
**How to avoid:** User re-authenticates with interactive login. Can also `ssh -O exit bwunicluster` to clean up stale socket before reconnecting.
**Warning signs:** Commands that worked earlier suddenly fail.

### Pitfall 5: Hostname Mismatch
**What goes wrong:** `ssh bwunicluster` works but `ssh bwunicluster.scc.kit.edu` creates a separate connection (no ControlMaster reuse).
**Why it happens:** ControlPath is per-hostname; alias and FQDN create different sockets.
**How to avoid:** Always use the alias `bwunicluster` everywhere (config, scripts, Claude commands).
**Warning signs:** Being prompted for 2FA when you already authenticated.

## Code Examples

### SSH Config Entry
```
# ~/.ssh/config -- bwUniCluster 3.0 access
# Source: bwHPC wiki + ControlMaster for 2FA session reuse

Host bwunicluster
    HostName bwunicluster.scc.kit.edu
    User st_ac147838
    IdentityFile ~/.ssh/id_ed25519
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### Preflight Check Script (inline approach)
```bash
#!/usr/bin/env bash
# Preflight verification for bwUniCluster 3.0
# Run from local machine after SSH is configured and 2FA-unlocked

set -euo pipefail
REMOTE="bwunicluster"
PROJECT="~/MasterThesisCode"
PASS=0
FAIL=0

check() {
    local name="$1"; shift
    if ssh "$REMOTE" "$@" &>/dev/null; then
        echo "  PASS: $name"
        ((PASS++))
    else
        echo "  FAIL: $name"
        ((FAIL++))
    fi
}

echo "=== bwUniCluster Preflight ==="
check "SSH connectivity" "echo ok"
check "Module loading" "source $PROJECT/cluster/modules.sh"
check "GPU partition" "sinfo -p gpu_h100 --noheader | grep -q ."
check "Workspace 'emri'" "ws_find emri"
check "Venv + imports" "source $PROJECT/cluster/modules.sh && source $PROJECT/.venv/bin/activate && python -c 'import few; import cupy'"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]] && echo "Preflight PASSED" || echo "Preflight FAILED"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Self-managed ~/.ssh/authorized_keys | Centrally-managed keys via bwIDM portal | bwUniCluster 3.0 launch | authorized_keys files are ignored; must use portal |
| Hostname bwunicluster.scc.kit.edu | Primary hostname uc3.scc.kit.edu | bwUniCluster 3.0 | Old hostname may still work but new hostname is canonical |
| Password-only auth | 2FA (TOTP + service password) | bwUniCluster 3.0 | All logins require TOTP; SSH keys need 2FA unlock |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| OpenSSH client | SSH connectivity | Yes | 10.2p1 | -- |
| ssh-keygen | Key generation | Yes | (bundled) | -- |
| ed25519 key pair | ACCESS-01 | Yes | ~/.ssh/id_ed25519 | -- |
| ~/.ssh/config | ACCESS-02 | No (to be created) | -- | -- |
| openconnect | VPN (user-managed) | Yes | installed | -- |
| bwIDM account | ACCESS-01 | Assumed | -- | -- |

**Missing dependencies with no fallback:**
- `~/.ssh/config` file does not exist -- must be created (ACCESS-02)
- `~/.ssh/sockets/` directory does not exist -- must be created for ControlMaster

**Missing dependencies with fallback:**
- None

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Manual verification via SSH commands |
| Config file | N/A (infrastructure phase, not code) |
| Quick run command | `ssh bwunicluster 'echo ok'` |
| Full suite command | Preflight check sequence (5 SSH commands) |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ACCESS-01 | SSH key registered on bwIDM portal | manual-only | N/A -- human action on web portal | N/A |
| ACCESS-02 | SSH config allows `ssh bwunicluster` | smoke | `ssh bwunicluster 'echo ok'` | N/A -- infra |
| ACCESS-03 | Environment preflight (modules, venv, GPU partition) | smoke | `ssh bwunicluster 'source ~/MasterThesisCode/cluster/modules.sh && sinfo -p gpu_h100 --noheader && ws_find emri && source ~/MasterThesisCode/.venv/bin/activate && python -c "import few; import cupy"'` | N/A -- infra |
| ACCESS-04 | Claude can execute cluster commands via Bash tool | smoke | Claude runs `ssh bwunicluster 'hostname'` and receives output | N/A -- infra |

### Sampling Rate
- **Per task:** Run individual SSH check after each setup step
- **Phase gate:** All 5 preflight checks pass in sequence

### Wave 0 Gaps
None -- this is an infrastructure phase with manual verification, not a code phase requiring test files.

## Open Questions

1. **Hostname resolution**
   - What we know: bwHPC wiki says `uc3.scc.kit.edu` is primary; project uses `bwunicluster.scc.kit.edu`
   - What's unclear: Whether `bwunicluster.scc.kit.edu` still resolves correctly for bwUniCluster 3.0
   - Recommendation: Use `bwunicluster.scc.kit.edu` per CONTEXT.md D-06; if it fails, fall back to `uc3.scc.kit.edu`

2. **2FA unlock duration**
   - What we know: bwHPC wiki says 8 hours for bwUniCluster 3.0
   - What's unclear: Whether this is 8h from first login or 8h rolling from last activity
   - Recommendation: Assume 8h from first login; ControlPersist 8h matches this

3. **Key propagation delay**
   - What we know: Keys are registered via portal and then activated
   - What's unclear: How long after portal registration the key becomes usable
   - Recommendation: Plan for possible delay (minutes to hours); include retry guidance

## Sources

### Primary (HIGH confidence)
- [bwHPC Wiki: bwUniCluster 3.0 Login](https://wiki.bwhpc.de/e/BwUniCluster3.0/Login) -- hostname, authentication flow, login nodes
- [bwHPC Wiki: SSH Key Registration](https://wiki.bwhpc.de/e/Registration/SSH) -- portal URL, key types, interactive vs command keys, 2FA unlock validity (8h), 180-day expiration
- Phase 3 execution summary (`03-01-SUMMARY.md`) -- verified module names, workspace path, import smoke tests
- `cluster/modules.sh`, `cluster/setup.sh` -- existing verified cluster scripts

### Secondary (MEDIUM confidence)
- [bwHPC Wiki: Login Examples](https://wiki.bwhpc.de/e/Login/Examples) -- username format with prefix
- [bwHPC Wiki: Username Registration](https://wiki.bwhpc.de/e/Registration/Login/Username) -- prefix convention (st_ for Stuttgart)

### Tertiary (LOW confidence)
- ControlPersist 8h recommendation -- derived from 2FA validity window, not officially documented by bwHPC as a recommended config

## Metadata

**Confidence breakdown:**
- SSH authentication model: HIGH -- verified against bwHPC wiki (multiple pages)
- SSH config format: HIGH -- standard OpenSSH config, ControlMaster is well-documented
- Preflight commands: HIGH -- derived from Phase 3 verified cluster setup
- 2FA timing: MEDIUM -- 8h stated in wiki, but edge cases (renewal, idle timeout) unclear
- Username (st_ac147838): MEDIUM -- inferred from vpn.sh (ac147838) + bwHPC prefix convention

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (SSH infrastructure is stable; key expiration is 180 days)
