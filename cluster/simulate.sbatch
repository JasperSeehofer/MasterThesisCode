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
# Optional environment variables:
#   H_VALUE    -- injected true Hubble constant for the simulated events
#                 (default 0.73). Set != 0.73 for closure runs at a different
#                 truth; threaded to --h_value so set_host_galaxy_parameters
#                 places each event at d_L(z, H_VALUE).
#
# Note: --output and --error are NOT set here; submit_pipeline.sh passes them
# via command line so that shell variables (e.g. %A, %a) are expanded correctly.

#SBATCH --job-name=emri-simulate
# gpu_h100_short (28 H100 GPUs, 30-min wall): tasks are TIME-capped (they run to the
# wall and merge tolerates the timeout via afterany), so a 30-min wall just halves
# each task's event yield -- compensated by doubling --tasks. Short jobs backfill
# constantly and are never blocked behind multi-day jobs, so they start far sooner
# than the contended gpu_h100 partition. Same H100 hardware -> identical per-event
# CRBs; only the per-task event count changes.
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
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

# Injected true Hubble constant (default 0.73; overridden for closure runs).
H_VALUE="${H_VALUE:-0.73}"

# ---------------------------------------------------------------------------
# Run EMRI simulation
# ---------------------------------------------------------------------------

echo "=== EMRI Simulation ==="
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Task Seed:     $TASK_SEED (BASE_SEED=$BASE_SEED + $SLURM_ARRAY_TASK_ID)"
echo "  Sim Steps:     $SIM_STEPS"
echo "  H_VALUE (truth): $H_VALUE"
echo "  Run Dir:       $RUN_DIR"
echo "  Node:          $(hostname)"
echo "  GPU:           ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# Simulation code uses relative paths (e.g. simulations/cramer_rao_bounds_*.csv)
# from CWD, but also expects the galaxy catalog at ./master_thesis_code/galaxy_catalogue/.
# Solution: run from the project root and symlink simulations/ → $RUN_DIR/simulations/
# so output lands in the run directory.
cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

python -m master_thesis_code "$RUN_DIR" \
    --simulation_steps "$SIM_STEPS" \
    --simulation_index "$SLURM_ARRAY_TASK_ID" \
    --seed "$TASK_SEED" \
    --h_value "$H_VALUE" \
    --use_gpu \
    --log_level INFO

echo ""
echo "=== Task $SLURM_ARRAY_TASK_ID completed (seed=$TASK_SEED) ==="
