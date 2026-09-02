# Archive script fix record — row #288 defect

**Date:** 2026-09-03. **Authorization:** row #325 autonomy grant (safe-hygiene ops fix, flagged)
+ the fix being explicitly "owed" since row #288 (`RUNBOOK_NEXT_SESSION_41.md` §5, "New this
session"). **Effort:** medium, as directed. **No cluster access this session** (SSH expired —
see `LENS3_INFRA_DATA.md` header, "no SSH — cluster access expired tonight"); everything below
is designed and self-tested locally, not integration-tested against the real cluster.

## The defect

`results/_archive/archive_run_wave2.sh`'s existence check (old line 42) was:

```bash
if ! ssh -o BatchMode=yes bwunicluster "test -d $WS/$item"; then
  echo "SKIP: $WS/$item not found on cluster" >> "$LOG"; ok=2; break
fi
```

This conflates two distinct outcomes into one boolean failure: `ssh` itself failing (transport
refused, auth expired, ControlMaster session timed out — `ssh` exits 255 for these) is
indistinguishable from `ssh` succeeding and the remote `test -d` legitimately returning "not a
directory / doesn't exist" (exit 1). Both paths hit the same `if !`, print the same "not found on
cluster" message, and mark the item absent.

**Verified impact (row #288 / LENS3 finding 3):** the 2026-09-01T03:56 log shows all 8 registered
items SKIPPED as "not found" in one run; the same items succeeded 14 hours later once the
ControlMaster session was re-authenticated. That was an auth-expiry false-negative, not real
absence — a textbook hit of this exact defect, and the reason the fix has been "owed" since row
#288.

## The fix

### `results/_archive/remote_exists.sh` (new, reusable helper)

A small sourced library, independent of any one archive script, providing:

- **`remote_probe_reachable <host>`** — a fast session-start check (`ssh -o BatchMode=yes
  -o ConnectTimeout=10 <host> true`). Meant to be called once, loudly, before any per-item work
  starts, so a dead ControlMaster session is caught up front instead of surfacing as N misleading
  "not found" results.
- **`remote_exists <host> <path>`** — prints exactly one of `PRESENT` / `ABSENT` / `UNREACHABLE`
  and returns 0 / 1 / 2 respectively. Distinguishes the ssh transport outcome from the remote
  test's outcome two ways at once:
  1. The remote command is `test -e '<path>' && echo REMOTE_EXISTS_YES || echo REMOTE_EXISTS_NO`
     — the `||` guarantees the remote command's own exit status is always 0 whenever ssh
     successfully ran anything at all, so ssh's own exit code is never aliased by the remote
     test's result.
  2. `ssh`'s exit code is checked directly for `255` (ssh's own transport/auth failure code) and
     mapped straight to `UNREACHABLE`.
  3. Even when ssh returns non-255, the function still requires a positive sentinel
     (`REMOTE_EXISTS_YES`/`REMOTE_EXISTS_NO`) in stdout before reporting `PRESENT`/`ABSENT`. Any
     other outcome (garbled output, an unexpected non-255 error, empty stdout) falls through to
     `UNREACHABLE` — the function never reports `ABSENT` unless it positively observed the remote
     side say so. This closes the failure mode one layer deeper than "just check for 255": it
     also protects against a transport that "succeeds" in some silent/truncated way.
  4. Every call to `ssh` inside the helper goes through the overridable variable
     `REMOTE_EXISTS_SSH_CMD` (default `ssh`), which is how the self-test fakes all three outcomes
     without touching the network, `PATH`, or a real `ssh` binary.

### `results/_archive/archive_run_wave2.sh` (fixed)

- Sources `remote_exists.sh` at the top (`SCRIPT_DIR`-relative, so it works regardless of the
  caller's cwd).
- Calls `remote_probe_reachable "$HOST"` once, before the item loop. On failure it prints an
  explicit `ABORT: ... re-auth needed (run 'ssh $HOST true' — OTP-gated)` message to stderr and
  the log, and exits 3 — no items are attempted at all if the session is already dead.
- Replaces the old boolean existence check with `remote_exists "$HOST" "$WS/$item"` and branches
  on all three values:
  - `PRESENT` → falls through to the existing `rsync --partial` retry logic (unchanged).
  - `ABSENT` → `SKIP: ... not found on cluster (confirmed absent, not an unreachable host)` —
    same `ok=2` bookkeeping as before, but the log line is now honest about which outcome it is.
  - `UNREACHABLE` → **aborts the whole run immediately** (`exit 3`) with a loud message telling
    the operator to treat every prior SKIP in the run as suspect and re-auth before re-running.
    This is the load-bearing behavior change: an expired session can no longer cause the loop to
    quietly mislabel the rest of the item list as absent.
- `--self-test` flag: running `archive_run_wave2.sh --self-test` (or `remote_exists.sh
  --self-test` directly) runs the local self-test suite and exits — no cluster or network
  required.

## Self-test output (verbatim, this session)

```
$ bash -n results/_archive/archive_run_wave2.sh && echo "SYNTAX OK"
SYNTAX OK
$ bash -n results/_archive/remote_exists.sh && echo "SYNTAX OK (helper)"
SYNTAX OK (helper)
$ bash results/_archive/archive_run_wave2.sh --self-test
[present]    out=PRESENT rc=0
[absent]     out=ABSENT rc=1
[unreachable/255] out=UNREACHABLE rc=2
[garbled/no-sentinel] out=UNREACHABLE rc=2
[probe-reachable]     rc=0 (as expected)
[probe-unreachable]   rc!=0 (as expected)
SELF-TEST: ALL PASS (6/6)
exit=0
```

The six cases: ssh succeeds + remote path present; ssh succeeds + remote path absent; ssh itself
returns 255 (the row #288 case — expired ControlMaster); ssh "succeeds" (rc 0) but returns no
recognizable sentinel (fail-safe check — must resolve `UNREACHABLE`, not `ABSENT`); the
reachability probe succeeding; the reachability probe failing. All 6 pass.

`shellcheck` was checked for (`which shellcheck`) and is **not installed** on this dev box, so no
shellcheck pass was run this session — `bash -n` syntax checks passed on both files as the
available substitute. If shellcheck becomes available, both files should get a pass before their
next real cluster use (they use plain POSIX-ish constructs — `local`, `case`, arrays, `$(...)`,
`[ ]` tests — expected to be clean, but unverified).

## What remains untestable until SSH returns

- **The actual `ssh -o BatchMode=yes ... true` reachability probe against `bwunicluster`** —
  self-test only exercises the function contract via a faked `ssh`, not real OpenSSH exit-code
  behavior (255 on auth/transport failure is documented OpenSSH behavior, not something this
  session independently reproduced against the real host).
- **A real UNREACHABLE hit mid-loop** — e.g. a ControlMaster session that dies partway through
  the 8-item wave-2 list (the exact failure the row #288 defect exhibited) rather than at the
  very first probe. The logic is symmetric (the same `remote_exists` call sits inside the loop
  for every item), but it has not been exercised against a live, degrading SSH session.
  - **Open question, unverified this session (no author statement, no test — flag only):**
    whether re-authenticating (`ssh bwunicluster true`, OTP-gated) from one shell actually
    refreshes a `ControlMaster` session shared by *other* concurrent consumers (e.g. a
    background monitor loop started earlier), or only benefits new connections opened after the
    re-auth. If it's the latter, an `UNREACHABLE` abort caught by this script mid-run might need
    more than "run `ssh bwunicluster true` once" to actually clear before a re-run. Worth
    confirming empirically next session with cluster access; not something this session could
    check or should assert either way.
- **Confirmation that the fixed script correctly resumes/archives the one still-open item**
  (`c1`, currently a legitimate SKIP per runbook 41 — "never launched, not a defect hit") once
  the session is live again; this record does not claim that item's status changed.
- **A real rsync retry/`--partial` interaction with the new existence check** — the retry loop
  around `rsync` itself is unchanged from the pre-fix script and was already presumed working
  (7/8 items succeeded in the 2026-09-01 evening re-run); only the existence-check branch
  upstream of it was touched.

## Files changed

- `results/_archive/remote_exists.sh` (new)
- `results/_archive/archive_run_wave2.sh` (modified: sources the helper, adds the session-start
  probe, replaces the two-valued existence check with the three-valued one, adds `--self-test`)
- `results/campaign51_20260728/realistic_20260729/graph1_20260901/health_scan/ARCHIVE_FIX_RECORD.md`
  (this file)

No commits made (per instructions).
