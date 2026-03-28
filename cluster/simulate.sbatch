#!/usr/bin/env bash
# cluster/simulate.sbatch -- GPU array job for EMRI simulation.
#
# This script is submitted by submit_pipeline.sh, not run directly.
# Each array task runs one EMRI simulation with a unique simulation_index
# mapped from SLURM_ARRAY_TASK_ID.
#
# Seed derivation: Per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID (TRACE-02)
# This ensures reproducibility while giving each task a different random state.
#
# Required environment variables (set by submit_pipeline.sh):
#   RUN_DIR    -- workspace directory for simulation output
#   BASE_SEED  -- base random seed (per-task seed derived from this)
#   SIM_STEPS  -- number of simulation iterations per task
#
# Note: --output and --error are NOT set here; submit_pipeline.sh passes them
# via command line so that shell variables (e.g. %A, %a) are expanded correctly.

#SBATCH --job-name=emri-simulate
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# SLURM copies sbatch scripts to /var/spool/slurmd/ before execution, so
# BASH_SOURCE[0] points to the spool copy. Use PROJECT_ROOT (exported by
# submit_pipeline.sh via --export=ALL) or fall back to ~/MasterThesisCode.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set. Set it before submitting this job." >&2
    exit 1
fi

if [[ -z "${BASE_SEED:-}" ]]; then
    echo "ERROR: BASE_SEED is not set. Set it before submitting this job." >&2
    exit 1
fi

if [[ -z "${SIM_STEPS:-}" ]]; then
    echo "ERROR: SIM_STEPS is not set. Set it before submitting this job." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Compute per-task seed (TRACE-02)
# Per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------------

TASK_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

# ---------------------------------------------------------------------------
# Run EMRI simulation
# ---------------------------------------------------------------------------

echo "=== EMRI Simulation ==="
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Task Seed:     $TASK_SEED (BASE_SEED=$BASE_SEED + $SLURM_ARRAY_TASK_ID)"
echo "  Sim Steps:     $SIM_STEPS"
echo "  Run Dir:       $RUN_DIR"
echo "  Node:          $(hostname)"
echo "  GPU:           ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# Change to run directory so relative paths in constants.py resolve correctly.
# Without this, simulation CSVs land in the SLURM submission directory instead
# of $RUN_DIR/simulations/ and the merge step finds nothing.
cd "$RUN_DIR"

python -m master_thesis_code "$RUN_DIR" \
    --simulation_steps "$SIM_STEPS" \
    --simulation_index "$SLURM_ARRAY_TASK_ID" \
    --seed "$TASK_SEED" \
    --use_gpu \
    --log_level INFO

echo ""
echo "=== Task $SLURM_ARRAY_TASK_ID completed (seed=$TASK_SEED) ==="
