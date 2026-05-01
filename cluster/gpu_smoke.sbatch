#!/usr/bin/env bash
# cluster/gpu_smoke.sbatch -- 1-event GPU smoke test for the injection path.
#
# Purpose: produce concrete evidence that --use_gpu actually executes on the GPU
# (rather than silently falling back to numpy). Captures nvidia-smi state
# pre/post and a 1-Hz utilization trace for the duration of the python call.
#
# Required environment variables (set by submit_gpu_smoke.sh):
#   RUN_DIR        -- workspace directory for output (must already exist)
#   BASE_SEED      -- random seed
#   H_VALUE        -- Hubble constant (default 0.73 from submit_gpu_smoke.sh)

#SBATCH --job-name=emri-gpu-smoke
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set." >&2
    exit 1
fi
if [[ -z "${BASE_SEED:-}" ]]; then
    echo "ERROR: BASE_SEED is not set." >&2
    exit 1
fi

H_VALUE="${H_VALUE:-0.73}"

mkdir -p "$RUN_DIR/simulations/injections" "$RUN_DIR/gpu_smoke"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

PRE_SNAPSHOT="$RUN_DIR/gpu_smoke/nvidia_smi_pre.txt"
POST_SNAPSHOT="$RUN_DIR/gpu_smoke/nvidia_smi_post.txt"
TRACE_CSV="$RUN_DIR/gpu_smoke/nvidia_smi_trace.csv"

echo "=== GPU Smoke Test ==="
echo "  Node:           $(hostname)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  H value:        $H_VALUE"
echo "  Seed:           $BASE_SEED"
echo "  Run dir:        $RUN_DIR"
echo "  Pre snapshot:   $PRE_SNAPSHOT"
echo "  Trace CSV:      $TRACE_CSV"
echo "  Post snapshot:  $POST_SNAPSHOT"
echo ""

nvidia-smi -q -d MEMORY,UTILIZATION,POWER,CLOCK > "$PRE_SNAPSHOT"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,temperature.gpu \
           --format=csv,nounits -l 1 > "$TRACE_CSV" &
TRACE_PID=$!

# Stop the trace cleanly even if python errors out.
trap 'kill -TERM "$TRACE_PID" 2>/dev/null || true; wait "$TRACE_PID" 2>/dev/null || true' EXIT

cd "$PROJECT_ROOT"
python -m master_thesis_code "$RUN_DIR" \
    --injection_campaign \
    --h_value "$H_VALUE" \
    --simulation_steps 1 \
    --simulation_index 0 \
    --seed "$BASE_SEED" \
    --use_gpu \
    --log_level DEBUG

# Give nvidia-smi one final tick before stopping it.
sleep 1
kill -TERM "$TRACE_PID" 2>/dev/null || true
wait "$TRACE_PID" 2>/dev/null || true
trap - EXIT

nvidia-smi -q -d MEMORY,UTILIZATION,POWER,CLOCK > "$POST_SNAPSHOT"

echo ""
echo "=== Smoke test complete ==="
echo ""
echo "Inspect:"
echo "  awk -F, 'NR>1 {print \$2}' $TRACE_CSV | sort -n | uniq -c"
echo "  # Should show non-zero utilization.gpu values during the run."
echo ""
echo "  diff <(grep -E 'Used|Free|Utilization' $PRE_SNAPSHOT) \\"
echo "       <(grep -E 'Used|Free|Utilization' $POST_SNAPSHOT)"
