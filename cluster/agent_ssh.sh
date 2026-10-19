#!/usr/bin/env bash
# agent_ssh.sh — the ONLY way agents talk to bwunicluster (ledger row #357 incident).
#   cluster/agent_ssh.sh run  '<remote command>'      # one short remote command (default timeout 600 s)
#   cluster/agent_ssh.sh poll <jobid[,jobid]> [secs]  # sacct until every job is terminal; sleeps LOCALLY
# Guarantees:
#   (1) at most 3 concurrent agent sessions on the shared ControlMaster (flock semaphore), so the
#       server-side sshd MaxSessions cap (10) is never reached together with the keepalive + author;
#   (2) "Session open refused by peer" is retried with backoff and NEVER treated as a dead master;
#   (3) never closes the master; if the socket is gone it says so and exits 3 (author OTP needed).
set -u
HOST=bwunicluster
SEM_DIR=/tmp/claude-1000/agent_ssh_sem
mkdir -p "$SEM_DIR"

acquire() {  # prints the fd that holds one of 3 slots; returns 1 if none free
  local fd i
  for i in 0 1 2; do
    exec {fd}>"$SEM_DIR/slot$i"
    if flock -n "$fd"; then echo "$fd"; return 0; fi
    exec {fd}>&-
  done
  return 1
}

run_once() {
  local out rc
  out=$(timeout "${AGENT_SSH_TIMEOUT:-600}" ssh -o BatchMode=yes "$HOST" "$1" 2>&1); rc=$?
  printf '%s\n' "$out"
  return $rc
}

cmd=${1:-}; shift || true
case "$cmd" in
  run)
    if ! ssh -O check "$HOST" >/dev/null 2>&1; then
      echo "agent_ssh: ControlMaster socket absent — the author must log in once (OTP). Do NOT re-auth, do NOT clean up. Report and stop." >&2
      exit 3
    fi
    for attempt in 1 2 3 4 5 6; do
      if fd=$(acquire); then
        out=$(run_once "$1"); rc=$?
        exec {fd}>&-
        if printf '%s' "$out" | grep -q "Session open refused by peer\|mux_client_request_session"; then
          echo "agent_ssh: mux session refused (server cap full) — backing off $((30*attempt)) s (attempt $attempt/6)" >&2
          sleep $((30*attempt)); continue
        fi
        printf '%s\n' "$out"; exit $rc
      fi
      sleep 10
    done
    echo "agent_ssh: no session slot / mux refused after 6 attempts — do NOT close the master; report and stop." >&2
    exit 4 ;;
  poll)
    jobs=${1:?jobids}; every=${2:-120}
    for i in $(seq 1 200); do
      line=$("$0" run "sacct -j $jobs -X --format=JobID,State,Elapsed --noheader") || exit $?
      printf '%s\n' "$line"
      if ! printf '%s' "$line" | grep -qE "PENDING|RUNNING|COMPLETING|CONFIGURING"; then exit 0; fi
      sleep "$every"   # LOCAL sleep — no remote session is held while waiting
    done ;;
  *) echo "usage: $0 run '<remote cmd>' | poll <jobids> [secs]" >&2; exit 2 ;;
esac
