#!/usr/bin/env bash
# cluster/resubmit_failed.sh -- Resubmit only failed simulation array tasks.
#
# Queries sacct for FAILED/NODE_FAIL/OUT_OF_MEMORY tasks (TIMEOUT only with
# --include-timeout: TIMEOUT is the EXPECTED terminal state on gpu_h100_short,
# where tasks are time-capped by design and their partial output is valid),
# deletes the failed tasks' partial output files (D-13), and resubmits only
# those indices.
#
# Usage:
#   resubmit_failed.sh [--include-timeout] [--force] <job_id> <run_dir> <base_seed> <sim_steps> [h_value]
#
#   job_id            SLURM array job ID from the simulate step
#   run_dir           Path to the campaign run directory
#   base_seed         Base random seed (same as original submission)
#   sim_steps         Simulation steps per task (same as original submission)
#   h_value           Injected true Hubble constant (optional). If omitted it is
#                     recovered from run_metadata_*.json in run_dir; only if
#                     neither source exists does it fall back to 0.73 (warned).
#   --include-timeout Also resubmit TIMEOUT tasks (see warning above).
#   --force           Bypass the already-merged guard (see below).
#
# GUARD: refuses to run if $RUN_DIR/simulations/cramer_rao_bounds.csv exists.
# The merge job auto-chains with afterany + --delete-sources, and
# merge_cramer_rao_bounds.py APPENDS to an existing merged CSV — resubmitting
# an already-merged task would duplicate its events.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: resubmit_failed.sh [--include-timeout] [--force] <job_id> <run_dir> <base_seed> <sim_steps> [h_value]
  job_id            SLURM array job ID from the simulate step
  run_dir           Path to the campaign run directory
  base_seed         Base random seed (same as original submission)
  sim_steps         Simulation steps per task (same as original submission)
  h_value           Injected truth (optional; recovered from run_metadata_*.json
                    if omitted; falls back to 0.73 with a warning if neither
                    source exists)
  --include-timeout Also resubmit TIMEOUT tasks. WARNING: TIMEOUT is the
                    expected terminal state on gpu_h100_short (time-capped by
                    design) — only use for tasks that hung/died early.
  --force           Bypass the already-merged guard.
EOF
    exit 1
}

INCLUDE_TIMEOUT=0
FORCE=0
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-timeout) INCLUDE_TIMEOUT=1; shift ;;
        --force)           FORCE=1;           shift ;;
        -h|--help) usage ;;
        -*) echo "ERROR: Unknown flag: $1" >&2; usage ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL[@]} -lt 4 || ${#POSITIONAL[@]} -gt 5 ]]; then
    echo "ERROR: Expected 4-5 positional arguments, got ${#POSITIONAL[@]}." >&2
    usage
fi

JOB_ID="${POSITIONAL[0]}"
RUN_DIR="${POSITIONAL[1]}"
BASE_SEED="${POSITIONAL[2]}"
SIM_STEPS="${POSITIONAL[3]}"
H_VALUE_ARG="${POSITIONAL[4]:-}"

# ---------------------------------------------------------------------------
# Guard: refuse if the run has already been merged (unless --force)
#
# merge auto-chains with afterany + --delete-sources, and the merge entry point
# APPENDS to an existing merged CSV — resubmitting an already-merged task would
# DUPLICATE its events in the merged output.
# ---------------------------------------------------------------------------

MERGED_CSV="$RUN_DIR/simulations/cramer_rao_bounds.csv"
if [[ -f "$MERGED_CSV" && "$FORCE" -ne 1 ]]; then
    cat >&2 <<EOF
ERROR: Merged CSV already exists: $MERGED_CSV
The merge job auto-chains with afterany + --delete-sources, and the merge step
APPENDS to an existing merged CSV — resubmitting an already-merged task would
DUPLICATE its events. Before resubmitting:
  * archive/remove the merged CSV (and prepared_cramer_rao_bounds.csv), or
  * scancel the pending merge job for this run,
then re-run this script. Use --force only if you have handled this manually.
EOF
    exit 1
fi

# ---------------------------------------------------------------------------
# Recover H_VALUE (BEFORE the cleanup loop deletes any run_metadata files)
#
# run_metadata_*.json records "h_value" under cli_args (main.py:_write_run_metadata).
# ---------------------------------------------------------------------------

RECOVERED_H=""
if compgen -G "$RUN_DIR/run_metadata_*.json" > /dev/null 2>&1; then
    RECOVERED_H=$(grep -hoE '"h_value":[[:space:]]*[0-9.]+' "$RUN_DIR"/run_metadata_*.json 2>/dev/null \
        | grep -oE '[0-9]+\.?[0-9]*$' | sort -u || true)
fi
N_RECOVERED=0
if [[ -n "$RECOVERED_H" ]]; then
    N_RECOVERED=$(echo "$RECOVERED_H" | wc -l | tr -d ' ')
fi

if [[ -n "$H_VALUE_ARG" ]]; then
    if [[ "$N_RECOVERED" -ge 1 ]] && ! echo "$RECOVERED_H" | grep -qx "$H_VALUE_ARG"; then
        echo "ERROR: explicit h_value=$H_VALUE_ARG conflicts with run_metadata_*.json" >&2
        echo "       (recorded: $(echo "$RECOVERED_H" | tr '\n' ' '))." >&2
        echo "       Aborting — resubmitting tasks at a different truth would mix populations." >&2
        exit 1
    fi
    H_VALUE="$H_VALUE_ARG"
    echo "H_VALUE: $H_VALUE (explicit argument)"
elif [[ "$N_RECOVERED" -eq 1 ]]; then
    H_VALUE="$RECOVERED_H"
    echo "H_VALUE: $H_VALUE (recovered from $RUN_DIR/run_metadata_*.json)"
elif [[ "$N_RECOVERED" -gt 1 ]]; then
    echo "ERROR: multiple distinct h_value entries in $RUN_DIR/run_metadata_*.json:" >&2
    echo "$RECOVERED_H" >&2
    echo "       Pass h_value explicitly to disambiguate." >&2
    exit 1
else
    H_VALUE="0.73"
    echo "################################################################"
    echo "# WARNING: no h_value argument and no surviving run_metadata_* #"
    echo "# json to recover it from. Falling back to H_VALUE=0.73.       #"
    echo "# VERIFY this matches the original submission before trusting  #"
    echo "# the resubmitted data (closure runs use a different truth!).  #"
    echo "################################################################"
fi

# ---------------------------------------------------------------------------
# Query failed tasks
# ---------------------------------------------------------------------------

STATES="FAILED,NODE_FAIL,OUT_OF_MEMORY"
if [[ "$INCLUDE_TIMEOUT" -eq 1 ]]; then
    STATES="$STATES,TIMEOUT"
    echo ""
    echo "WARNING: --include-timeout given. TIMEOUT is the EXPECTED terminal state"
    echo "         on gpu_h100_short (tasks are time-capped by design and their"
    echo "         partial output is valid). Only resubmit TIMEOUT tasks that hung"
    echo "         or died abnormally early."
fi

echo ""
echo "Querying failed tasks for job $JOB_ID (states: $STATES)..."

FAILED_TASKS=$(sacct --array --jobs="$JOB_ID" \
    --state="$STATES" \
    --format=JobID%30 \
    --noheader --parsable2 \
    | grep -oP '^\d+_\K\d+' \
    | sort -n | uniq)

if [[ -z "$FAILED_TASKS" ]]; then
    echo "No failed tasks found for job $JOB_ID."
    exit 0
fi

FAILED_COUNT=$(echo "$FAILED_TASKS" | wc -l)
echo "Found $FAILED_COUNT failed task(s)."
echo ""

# ---------------------------------------------------------------------------
# Clean up partial output (D-13)
# ---------------------------------------------------------------------------

echo "Cleaning up partial output files..."

while IFS= read -r IDX; do
    for FILE in \
        "$RUN_DIR/simulations/cramer_rao_bounds_simulation_${IDX}.csv" \
        "$RUN_DIR/simulations/undetected_events_simulation_${IDX}.csv" \
        "$RUN_DIR/run_metadata_${IDX}.json"; do
        if [[ -f "$FILE" ]]; then
            echo "  Removing: $FILE"
        fi
        rm -f "$FILE"
    done
done <<< "$FAILED_TASKS"

echo ""

# ---------------------------------------------------------------------------
# Resubmit failed tasks
# ---------------------------------------------------------------------------

FAILED_ARRAY=$(echo "$FAILED_TASKS" | tr '\n' ',' | sed 's/,$//')
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Resubmitting failed tasks: $FAILED_ARRAY"

RESUB_JOB=$(sbatch --parsable \
    --array="$FAILED_ARRAY" \
    --output="$RUN_DIR/logs/simulate_%A_%a.out" \
    --error="$RUN_DIR/logs/simulate_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$BASE_SEED",SIM_STEPS="$SIM_STEPS",H_VALUE="$H_VALUE" \
    "$CLUSTER_DIR/simulate.sbatch")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "Resubmitted: $RESUB_JOB (tasks: $FAILED_ARRAY, H_VALUE=$H_VALUE)"
echo ""
echo "After completion, resubmit merge+evaluate:"
echo "  sbatch --dependency=afterok:$RESUB_JOB \\"
echo "    --output=$RUN_DIR/logs/merge_%j.out \\"
echo "    --error=$RUN_DIR/logs/merge_%j.err \\"
echo "    --export=ALL,RUN_DIR=$RUN_DIR \\"
echo "    $CLUSTER_DIR/merge.sbatch"
