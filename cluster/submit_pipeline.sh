#!/usr/bin/env bash
# cluster/submit_pipeline.sh -- Submit the full simulate-merge-evaluate pipeline.
#
# Chains three SLURM jobs via --dependency=afterok:
#   1. simulate (GPU array job) -- one task per EMRI simulation
#   2. merge (CPU job)          -- combines per-task CSVs, prepares detections
#   3. evaluate (CPU job)       -- Bayesian inference for H0 posterior
#
# Usage:
#   submit_pipeline.sh --tasks N --steps S --seed SEED
#
# All three flags are required. No defaults.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: submit_pipeline.sh --tasks N --steps S --seed SEED
  --tasks   Number of array tasks (simulation jobs)
  --steps   Simulation steps per task
  --seed    Base random seed (per-task seed = seed + task_id)
EOF
    exit 1
}

TASKS=""
STEPS=""
SEED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)  TASKS="$2";  shift 2 ;;
        --steps)  STEPS="$2";  shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        -h|--help) usage ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$TASKS" ]]; then
    echo "ERROR: --tasks is required." >&2
    usage
fi

if [[ -z "$STEPS" ]]; then
    echo "ERROR: --steps is required." >&2
    usage
fi

if [[ -z "$SEED" ]]; then
    echo "ERROR: --seed is required." >&2
    usage
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLUSTER_DIR/modules.sh"

if [[ -z "${WORKSPACE:-}" ]]; then
    echo "ERROR: \$WORKSPACE is not set. Run cluster/setup.sh first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run directory creation (D-08)
# ---------------------------------------------------------------------------

DATESTAMP=$(date +%Y%m%d)
RUN_DIR="$WORKSPACE/run_${DATESTAMP}_seed${SEED}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/simulations"

echo ""
echo "=== EMRI Pipeline Submission ==="
echo "  Tasks:     $TASKS"
echo "  Steps:     $STEPS"
echo "  Seed:      $SEED"
echo "  Run dir:   $RUN_DIR"
echo ""

# ---------------------------------------------------------------------------
# Job submission chain (SLURM-04)
# ---------------------------------------------------------------------------

# 1. Simulate array job (GPU)
SIM_JOB=$(sbatch --parsable \
    --array="0-$((TASKS - 1))" \
    --output="$RUN_DIR/logs/simulate_%A_%a.out" \
    --error="$RUN_DIR/logs/simulate_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$SEED",SIM_STEPS="$STEPS" \
    "$CLUSTER_DIR/simulate.sbatch")

# 2. Merge job (CPU, after all simulate tasks finish — afterany tolerates timeouts)
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterany:$SIM_JOB" \
    --output="$RUN_DIR/logs/merge_%j.out" \
    --error="$RUN_DIR/logs/merge_%j.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",PREPARE_SEED="$((SEED + 999999))" \
    "$CLUSTER_DIR/merge.sbatch")

# 3. Evaluate array job (CPU, after merge completes)
#    Array tasks map to h values defined in evaluate.sbatch (27 values: 0.60–0.86, step 0.01)
EVAL_JOB=$(sbatch --parsable \
    --array="0-26" \
    --dependency="afterok:$MERGE_JOB" \
    --output="$RUN_DIR/logs/evaluate_%A_%a.out" \
    --error="$RUN_DIR/logs/evaluate_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR" \
    "$CLUSTER_DIR/evaluate.sbatch")

# 4. Combine posteriors (CPU, after all evaluate tasks finish)
COMBINE_JOB=$(sbatch --parsable \
    --dependency="afterok:$EVAL_JOB" \
    --output="$RUN_DIR/logs/combine_%j.out" \
    --error="$RUN_DIR/logs/combine_%j.err" \
    --export=ALL,RUN_DIR="$RUN_DIR" \
    "$CLUSTER_DIR/combine.sbatch")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "Pipeline submitted:"
echo "  Run directory: $RUN_DIR"
echo "  Simulate:  $SIM_JOB (array 0-$((TASKS - 1)))"
echo "  Merge:     $MERGE_JOB (after simulate)"
echo "  Evaluate:  $EVAL_JOB (array 0-26, h sweep 0.60–0.86, step 0.01)"
echo "  Combine:   $COMBINE_JOB (after evaluate)"
echo ""
echo "Monitor: sacct -j $SIM_JOB,$MERGE_JOB,$EVAL_JOB,$COMBINE_JOB"
