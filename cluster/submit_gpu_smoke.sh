#!/usr/bin/env bash
# cluster/submit_gpu_smoke.sh -- Submit a single 1-event GPU smoke test.
#
# Verifies that --use_gpu actually drives the GPU on the cluster (not a
# silent fallback to numpy). Captures pre/post nvidia-smi snapshots and a
# 1-Hz utilization trace under $RUN_DIR/gpu_smoke/.
#
# Usage:
#   submit_gpu_smoke.sh                    # uses default seed 42, h=0.73
#   submit_gpu_smoke.sh --seed 12345
#   submit_gpu_smoke.sh --seed 12345 --h 0.70

set -euo pipefail

SEED=42
H_VALUE=0.73

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="$2"; shift 2 ;;
        --h)    H_VALUE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: submit_gpu_smoke.sh [--seed N] [--h H_VALUE]"
            echo "  Submits a single 1-event GPU smoke job (~5 min wall)."
            exit 0
            ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLUSTER_DIR/modules.sh"

if [[ -z "${WORKSPACE:-}" ]]; then
    echo "WARNING: \$WORKSPACE not set. Using fallback directory." >&2
    WORKSPACE="$HOME/emri-runs"
fi

DATESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_DIR="$WORKSPACE/gpu_smoke_${DATESTAMP}"
mkdir -p "$RUN_DIR/logs"

echo "=== GPU Smoke Submission ==="
echo "  Seed:          $SEED"
echo "  H value:       $H_VALUE"
echo "  Run dir:       $RUN_DIR"
echo ""

JOB_ID=$(sbatch --parsable \
    --output="$RUN_DIR/logs/gpu_smoke_%A.out" \
    --error="$RUN_DIR/logs/gpu_smoke_%A.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$SEED",H_VALUE="$H_VALUE" \
    "$CLUSTER_DIR/gpu_smoke.sbatch")

echo "  Submitted job: $JOB_ID"
echo ""
echo "Monitor:        sacct -j $JOB_ID"
echo "Stdout:         $RUN_DIR/logs/gpu_smoke_${JOB_ID}.out"
echo "GPU artifacts:  $RUN_DIR/gpu_smoke/"
