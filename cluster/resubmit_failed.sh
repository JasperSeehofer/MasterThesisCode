#!/usr/bin/env bash
# cluster/resubmit_failed.sh -- Resubmit only failed simulation array tasks.
#
# Queries sacct for FAILED/TIMEOUT/NODE_FAIL/OUT_OF_MEMORY tasks, deletes
# their partial output files (D-13), and resubmits only those indices.
#
# Usage:
#   resubmit_failed.sh <job_id> <run_dir> <base_seed> <sim_steps>
#
#   job_id      SLURM array job ID from the simulate step
#   run_dir     Path to the campaign run directory
#   base_seed   Base random seed (same as original submission)
#   sim_steps   Simulation steps per task (same as original submission)

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: resubmit_failed.sh <job_id> <run_dir> <base_seed> <sim_steps>
  job_id      SLURM array job ID from the simulate step
  run_dir     Path to the campaign run directory
  base_seed   Base random seed (same as original submission)
  sim_steps   Simulation steps per task (same as original submission)
EOF
    exit 1
}

if [[ $# -lt 4 ]]; then
    echo "ERROR: Expected 4 arguments, got $#." >&2
    usage
fi

JOB_ID="$1"
RUN_DIR="$2"
BASE_SEED="$3"
SIM_STEPS="$4"

# ---------------------------------------------------------------------------
# Query failed tasks
# ---------------------------------------------------------------------------

echo "Querying failed tasks for job $JOB_ID..."

FAILED_TASKS=$(sacct --array --jobs="$JOB_ID" \
    --state=FAILED,TIMEOUT,NODE_FAIL,OUT_OF_MEMORY \
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
    --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$BASE_SEED",SIM_STEPS="$SIM_STEPS" \
    "$CLUSTER_DIR/simulate.sbatch")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "Resubmitted: $RESUB_JOB (tasks: $FAILED_ARRAY)"
echo ""
echo "After completion, resubmit merge+evaluate:"
echo "  sbatch --dependency=afterok:$RESUB_JOB \\"
echo "    --output=$RUN_DIR/logs/merge_%j.out \\"
echo "    --error=$RUN_DIR/logs/merge_%j.err \\"
echo "    --export=ALL,RUN_DIR=$RUN_DIR \\"
echo "    $CLUSTER_DIR/merge.sbatch"
