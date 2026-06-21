#!/usr/bin/env bash
# cluster/submit_injection.sh -- Submit injection campaign array jobs.
#
# Builds the simulation-based detection-probability pool consumed by
# SimulationDetectionProbability (the detection-horizon SURVIVAL function).
#
# IMPORTANT — a SINGLE h suffices (default 0.73):
#   The survival estimator uses the per-injection horizon d_hor = SNR*d_L/thr,
#   which is h-INVARIANT (the 1/d_L amplitude scaling and the d_L assignment
#   cancel) and pools ALL injections regardless of h_inj. The injected (M,z) are
#   drawn from the h-independent rate model dN/dz*R(M), and M_z = M*(1+z) is
#   h-free, so every h-node samples the SAME d_hor/M_z distribution. Running
#   multiple h-values therefore adds NO per-h structure — it only accumulates
#   more independent samples (equivalent to more --tasks_per_h at one h).
#   The legacy multi-h default existed for the old per-h KDE estimator (Phase
#   11.1) and is no longer needed. Total pooled samples = tasks_per_h * steps
#   * (number of h-values); set --tasks_per_h to your sample budget. Each task
#   is one emcee chain (~50% unique (M,z) due to MCMC autocorrelation — that is
#   correct density representation, not a bug; do NOT deduplicate).
#
# Usage (single-h default, recommended):
#   submit_injection.sh --tasks_per_h 80 --steps 900 --seed 12345
# Multi-h (optional; only for more samples or the legacy per-h estimator):
#   submit_injection.sh --tasks_per_h 80 --steps 900 --seed 12345 \
#       --h_values "0.60,0.70,0.80,0.90"

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
  --h_values     Comma-separated h values (default: 0.73 — a single h; the
                 survival p_det is h-invariant so multi-h only adds samples)

Example (single-h default):
  submit_injection.sh --tasks_per_h 80 --steps 900 --seed 12345
  # => 80 tasks * 900 events = 72,000 pooled injection samples at h=0.73
EOF
    exit 1
}

TASKS_PER_H=""
STEPS=""
SEED=""
# Single h by default: the detection-horizon survival p_det is h-invariant
# (see header). Multi-h only accumulates more pooled samples, not per-h grids.
H_VALUES="0.73"

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
