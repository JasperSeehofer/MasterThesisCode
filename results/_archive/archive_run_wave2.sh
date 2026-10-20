#!/usr/bin/env bash
# Option-A MUST-ARCHIVE rsync program, wave 2 (launched under rows #222/#223 — charter node:
# wave-2 GAP-CLOSURE archive/notes worker, GAP 6). Same form as archive_run_20260828.sh.
# This script runs AFTER retrieval of the wave-2 cluster out-roots from the workspace — it is
# NOT run by this node. --partial so a dropped SSH connection resumes instead of restarting;
# 2 retries per item (kept consistent with the 2026-08-28 script's loop, despite the header
# comment there saying "2 retries" while looping 3 attempts).
#
# Row #288 fix (2026-09-03): the existence check used to be a bare `ssh ... test -d ...`, which
# conflates "SSH unreachable/auth-expired" with "file not found on cluster" — an expired
# ControlMaster session made an entire 8-item run SKIP everything as "not found" (2026-09-01
# 03:56 log; the same items succeeded 14h later once re-authenticated). Fixed by delegating to
# remote_exists.sh's three-valued PRESENT/ABSENT/UNREACHABLE check: UNREACHABLE now aborts the
# whole run loudly at first occurrence instead of silently marking items absent, and a
# session-start reachability probe fails fast before any item work starts. See
# results/campaign51_20260728/realistic_20260729/graph1_20260901/health_scan/
# ARCHIVE_FIX_RECORD.md.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./remote_exists.sh
source "$SCRIPT_DIR/remote_exists.sh"

# --self-test: run remote_exists.sh's local self-test (no cluster/network needed) and exit.
if [ "${1:-}" = "--self-test" ]; then
  _remote_exists_self_test
  exit $?
fi

WS=/pfs/work9/workspace/scratch/st_ac147838-emri
DEST=results/_archive
LOG=$DEST/archive_run_wave2.log
HOST=bwunicluster

ITEMS=(
  run_20260829_wave2_c0_iiib
  run_20260829_wave2_c1_iiib
  run_20260829_wave2_c3_iiib
  run_20260829_wave2_c4_iiib

  # --- wave 3 block (launched under rows #222/#223 — charter wave 3 / node B7.3 readout,
  # appended 2026-08-30 by the wave-3 sbatch builder pass). The ONE blind HEAD readout, both
  # venues, full H_GRID_41 (41 tasks each). See cluster/WAVE3_SUBMISSION_NOTE_20260830.md. ---
  run_20260830_wave3_headreadout_iiib
  run_20260830_wave3_headreadout_joint_r1

  # --- wave 3 C0' off-gate block (launched under rows #222/#223 — charter wave 3 / node B7.3
  # readout, appended 2026-08-30 by the same builder pass). The A14 falsifier BASELINE gate,
  # both venues, h=0.730 only (2 tasks total). See
  # cluster/WAVE3_SUBMISSION_NOTE_20260830.md §1a ("C0' off-gate"). ---
  run_20260830_wave3_c0prime_off_iiib
  run_20260830_wave3_c0prime_off_joint_r1
)

echo "=== archive run start $(date -Is) ===" >> "$LOG"

# Session-start reachability probe: fail fast and loud if the ControlMaster session has expired
# (ControlMaster/OTP sessions expire ~8h — see runbook 41 §5). This is deliberately separate from
# per-item existence checks so a dead session is caught once, up front, instead of surfacing as
# 8 misleading "not found" results.
if ! remote_probe_reachable "$HOST"; then
  echo "ABORT: $HOST unreachable at session start (re-auth needed: run 'ssh $HOST true' — OTP-gated) $(date -Is)" | tee -a "$LOG" >&2
  exit 3
fi

fail=0
for item in "${ITEMS[@]}"; do
  ok=1
  for attempt in 1 2 3; do
    echo "--- $item attempt $attempt $(date -Is)" >> "$LOG"
    # existence check (out-root names are the wave-2 GAP-CLOSURE convention; verify against the
    # actual sbatch out-dir naming before running, since C0-C4 registrations name arms, not
    # necessarily these exact directory names). Three-valued: PRESENT / ABSENT / UNREACHABLE —
    # see remote_exists.sh and the row #288 fix note above.
    existence=$(remote_exists "$HOST" "$WS/$item")
    case "$existence" in
      PRESENT)
        : # fall through to rsync below
        ;;
      ABSENT)
        echo "SKIP: $WS/$item not found on cluster (confirmed absent, not an unreachable host)" >> "$LOG"
        ok=2
        break
        ;;
      UNREACHABLE)
        # Never silently skip on UNREACHABLE — abort the whole run loudly instead of mislabeling
        # an auth/transport failure as "not found" (the exact row #288 defect).
        echo "ABORT: $HOST unreachable while checking $WS/$item — treat all prior SKIPs in a run with an UNREACHABLE as suspect, re-auth and re-run $(date -Is)" | tee -a "$LOG" >&2
        exit 3
        ;;
    esac
    if rsync -a --partial --timeout=120 "$HOST:$WS/$item" "$DEST/" >> "$LOG" 2>&1; then
      echo "OK: $item $(date -Is)" >> "$LOG"; break
    fi
    ok=0; sleep 30
  done
  [ "$ok" = 0 ] && { echo "FAILED after retries: $item" >> "$LOG"; fail=1; }
done
echo "=== archive run end $(date -Is) fail=$fail ===" >> "$LOG"
exit $fail
