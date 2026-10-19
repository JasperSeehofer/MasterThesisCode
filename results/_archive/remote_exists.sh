#!/usr/bin/env bash
# remote_exists.sh — three-valued remote-path existence check (PRESENT / ABSENT / UNREACHABLE).
#
# Fixes the row #288 defect: archive_run_wave2.sh's bare `ssh ... test -d ...` conflated
# "SSH transport/auth failure" with "file not found on the cluster" — a single expired
# ControlMaster session made an entire run SKIP every item as "not found" (2026-09-01 03:56
# log; the same 8 items succeeded 14h later once re-authenticated). See
# results/campaign51_20260728/realistic_20260729/graph1_20260901/health_scan/LENS3_INFRA_DATA.md
# finding 3 and ledger row #288 ("Fix owed: distinguish present / absent / unreachable").
#
# Contract:
#   remote_probe_reachable <host>
#     Fast session-start check. Returns 0 if the host answers a trivial command, 1 otherwise.
#     Intended use: call once at the top of a run and abort loudly on failure (re-auth needed)
#     rather than discovering unreachability item-by-item and silently skipping each one.
#
#   remote_exists <host> <path>
#     Prints exactly one of PRESENT / ABSENT / UNREACHABLE to stdout and returns:
#       0 = PRESENT     (remote `test -e` succeeded)
#       1 = ABSENT      (remote `test -e` failed AND we positively saw its sentinel — i.e. the
#                        ssh session itself worked, the remote host ran the test and reported
#                        "no")
#       2 = UNREACHABLE (ssh itself failed — exit 255 is ssh's own transport/auth failure code
#                        — OR the expected sentinel never appeared in stdout for any other
#                        reason). UNREACHABLE is the safe default: this function never reports
#                        ABSENT unless it positively observed the remote side say so.
#
# Both functions consult $REMOTE_EXISTS_SSH_CMD (default: "ssh") so a caller's --self-test mode
# can substitute a fake ssh implementation without touching PATH or real ssh.

REMOTE_EXISTS_SSH_CMD="${REMOTE_EXISTS_SSH_CMD:-ssh}"

remote_probe_reachable() {
  # $1 = host. Session-start reachability probe — meant to be called once, loudly, before any
  # per-item work starts.
  local host="$1"
  $REMOTE_EXISTS_SSH_CMD -o BatchMode=yes -o ConnectTimeout=10 "$host" true >/dev/null 2>&1
}

remote_exists() {
  # $1 = host, $2 = remote path. Prints PRESENT/ABSENT/UNREACHABLE; return code per the
  # contract above.
  local host="$1"
  local path="$2"
  local out rc

  # Sentinel tokens (not just the remote test's own exit code) so a truncated/garbled
  # transport still fails safe into UNREACHABLE instead of being misread as ABSENT: the
  # single `||` chain below guarantees the remote command's own exit status is always 0 when
  # ssh successfully ran anything at all, so ssh's exit code is reserved for ssh's own
  # transport/auth outcome (255 = failure) and never aliases the remote test result.
  out=$($REMOTE_EXISTS_SSH_CMD -o BatchMode=yes -o ConnectTimeout=10 "$host" \
    "test -e '$path' && echo REMOTE_EXISTS_YES || echo REMOTE_EXISTS_NO" 2>/dev/null)
  rc=$?

  if [ "$rc" -eq 255 ]; then
    echo "UNREACHABLE"
    return 2
  fi

  case "$out" in
    *REMOTE_EXISTS_YES*)
      echo "PRESENT"
      return 0
      ;;
    *REMOTE_EXISTS_NO*)
      echo "ABSENT"
      return 1
      ;;
    *)
      # No sentinel seen for any reason (killed mid-transfer, non-255 ssh error, garbled
      # output, empty output). Never default to ABSENT here — that is exactly the row #288
      # failure mode, just relocated.
      echo "UNREACHABLE"
      return 2
      ;;
  esac
}

# ---------------------------------------------------------------------------------------------
# Self-test: exercise remote_exists()/remote_probe_reachable() against faked ssh outcomes, with
# no network access and no real cluster. Run directly: `bash remote_exists.sh --self-test`.
# ---------------------------------------------------------------------------------------------
_remote_exists_self_test() {
  local failures=0

  # --- Fake 1: remote path PRESENT (ssh succeeds, remote test -e succeeds) ---
  fake_ssh_present() { echo "REMOTE_EXISTS_YES"; return 0; }
  REMOTE_EXISTS_SSH_CMD=fake_ssh_present
  out=$(remote_exists host /some/path); rc=$?
  echo "[present]    out=$out rc=$rc"
  [ "$out" = "PRESENT" ] && [ "$rc" -eq 0 ] || { echo "  FAIL: expected PRESENT/0"; failures=$((failures + 1)); }

  # --- Fake 2: remote path ABSENT (ssh succeeds, remote test -e fails, sentinel seen) ---
  fake_ssh_absent() { echo "REMOTE_EXISTS_NO"; return 0; }
  REMOTE_EXISTS_SSH_CMD=fake_ssh_absent
  out=$(remote_exists host /some/path); rc=$?
  echo "[absent]     out=$out rc=$rc"
  [ "$out" = "ABSENT" ] && [ "$rc" -eq 1 ] || { echo "  FAIL: expected ABSENT/1"; failures=$((failures + 1)); }

  # --- Fake 3: SSH transport/auth failure (expired ControlMaster) — the row #288 case ---
  fake_ssh_unreachable() { return 255; }
  REMOTE_EXISTS_SSH_CMD=fake_ssh_unreachable
  out=$(remote_exists host /some/path); rc=$?
  echo "[unreachable/255] out=$out rc=$rc"
  [ "$out" = "UNREACHABLE" ] && [ "$rc" -eq 2 ] || { echo "  FAIL: expected UNREACHABLE/2"; failures=$((failures + 1)); }

  # --- Fake 4: ssh "succeeds" (rc=0) but no sentinel in stdout — must fail SAFE, not ABSENT ---
  fake_ssh_garbled() { echo "some unrelated banner / MOTD noise"; return 0; }
  REMOTE_EXISTS_SSH_CMD=fake_ssh_garbled
  out=$(remote_exists host /some/path); rc=$?
  echo "[garbled/no-sentinel] out=$out rc=$rc"
  [ "$out" = "UNREACHABLE" ] && [ "$rc" -eq 2 ] || { echo "  FAIL: expected UNREACHABLE/2 (fail-safe, not ABSENT)"; failures=$((failures + 1)); }

  # --- Fake 5: reachability probe succeeds ---
  fake_ssh_probe_ok() { return 0; }
  REMOTE_EXISTS_SSH_CMD=fake_ssh_probe_ok
  if remote_probe_reachable host; then
    echo "[probe-reachable]     rc=0 (as expected)"
  else
    echo "[probe-reachable]     FAIL: expected success"
    failures=$((failures + 1))
  fi

  # --- Fake 6: reachability probe fails (expired session) ---
  fake_ssh_probe_fail() { return 255; }
  REMOTE_EXISTS_SSH_CMD=fake_ssh_probe_fail
  if remote_probe_reachable host; then
    echo "[probe-unreachable]   FAIL: expected failure"
    failures=$((failures + 1))
  else
    echo "[probe-unreachable]   rc!=0 (as expected)"
  fi

  unset REMOTE_EXISTS_SSH_CMD

  if [ "$failures" -eq 0 ]; then
    echo "SELF-TEST: ALL PASS (6/6)"
    return 0
  else
    echo "SELF-TEST: $failures FAILURE(S)"
    return 1
  fi
}

# Allow `bash remote_exists.sh --self-test` directly, in addition to being sourced.
if [ "${1:-}" = "--self-test" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  _remote_exists_self_test
  exit $?
fi
