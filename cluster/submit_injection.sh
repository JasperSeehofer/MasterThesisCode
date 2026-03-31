#!/usr/bin/env bash
# cluster/submit_injection.sh -- Submit injection campaign array jobs.
#
# Submits GPU array jobs for multiple h values to build a simulation-based
# detection probability grid P_det(z, h). Each h value gets its own batch
# of array tasks with isolated seed ranges.
#
# Usage:
#   submit_injection.sh --tasks_per_h 20 --steps 500 --seed 12345
#   submit_injection.sh --tasks_per_h 20 --steps 500 --seed 12345 \
#       --h_values "0.60,0.65,0.70,0.73,0.80,0.85,0.90"

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: submit_injection.sh --tasks_per_h N --steps S --seed SEED [--h_values "h1,h2,..."]

  --tasks_per_h  Number of array tasks per h value (required)
  --steps        Successful injection events per task (required)
  --seed         Base random seed (required)
  --h_values     Comma-separated h values (default: 0.60,0.65,0.70,0.73,0.80,0.85,0.90)

Example:
  submit_injection.sh --tasks_per_h 20 --steps 500 --seed 12345
  # => 20 tasks * 500 events = 10,000 events per h value
  # => 7 h values * 20 tasks = 140 total SLURM tasks
EOF
    exit 1
}

TASKS_PER_H=""
STEPS=""
SEED=""
H_VALUES="0.60,0.65,0.70,0.73,0.80,0.85,0.90"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks_per_h)  TASKS_PER_H="$2";  shift 2 ;;
        --steps)        STEPS="$2";        shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --h_values)     H_VALUES="$2";     shift 2 ;;
        -h|--help)      usage ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$TASKS_PER_H" ]]; then
    echo "ERROR: --tasks_per_h is required." >&2
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
    echo "WARNING: \$WORKSPACE not set. Using fallback directory." >&2
    WORKSPACE="$HOME/emri-runs"
fi

# ---------------------------------------------------------------------------
# Run directory creation
# ---------------------------------------------------------------------------

DATESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$WORKSPACE/injection_${DATESTAMP}_seed${SEED}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/simulations/injections"

# ---------------------------------------------------------------------------
# Parse h values into array
# ---------------------------------------------------------------------------

IFS=',' read -ra H_ARRAY <<< "$H_VALUES"
NUM_H=${#H_ARRAY[@]}

echo ""
echo "=== EMRI Injection Campaign Submission ==="
echo "  H values:       ${H_ARRAY[*]} ($NUM_H values)"
echo "  Tasks per h:    $TASKS_PER_H"
echo "  Steps per task: $STEPS"
echo "  Base seed:      $SEED"
echo "  Run directory:  $RUN_DIR"
echo ""

# ---------------------------------------------------------------------------
# Submit array jobs for each h value
# ---------------------------------------------------------------------------

TOTAL_TASKS=0
JOB_IDS=()

for h_index in "${!H_ARRAY[@]}"; do
    h="${H_ARRAY[$h_index]}"

    # Seed isolation: offset by h_index * 10000 so different h values get
    # non-overlapping seed ranges. E.g., for tasks_per_h=20:
    #   h=0.60 (index 0): seeds SEED+0     to SEED+19
    #   h=0.65 (index 1): seeds SEED+10000 to SEED+10019
    #   h=0.70 (index 2): seeds SEED+20000 to SEED+20019
    H_BASE_SEED=$((SEED + h_index * 10000))

    # Label for log files: replace '.' with '_' (e.g., 0.73 -> 0_73)
    h_label="${h//./_}"

    JOB_ID=$(sbatch --parsable \
        --array="0-$((TASKS_PER_H - 1))" \
        --output="$RUN_DIR/logs/inject_h_${h_label}_%A_%a.out" \
        --error="$RUN_DIR/logs/inject_h_${h_label}_%A_%a.err" \
        --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$H_BASE_SEED",INJ_STEPS="$STEPS",H_VALUE="$h" \
        "$CLUSTER_DIR/inject.sbatch")

    JOB_IDS+=("$JOB_ID")
    TOTAL_TASKS=$((TOTAL_TASKS + TASKS_PER_H))
    echo "  Submitted h=$h: job $JOB_ID (array 0-$((TASKS_PER_H - 1)), base_seed=$H_BASE_SEED)"
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

TOTAL_EVENTS=$((TOTAL_TASKS * STEPS))
ALL_JOBS=$(IFS=','; echo "${JOB_IDS[*]}")

echo ""
echo "=== Injection Campaign Summary ==="
echo "  H values:        $NUM_H"
echo "  Tasks per h:     $TASKS_PER_H"
echo "  Total tasks:     $TOTAL_TASKS"
echo "  Events per task: $STEPS"
echo "  Total events:    $TOTAL_EVENTS"
echo "  Run directory:   $RUN_DIR"
echo ""
echo "Monitor: sacct -j $ALL_JOBS"
