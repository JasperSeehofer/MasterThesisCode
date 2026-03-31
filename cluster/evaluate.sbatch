#!/usr/bin/env bash
# cluster/evaluate.sbatch -- CPU array job for Bayesian inference (H0 posterior).
#
# This script is submitted by submit_pipeline.sh after the merge job completes.
# It runs Bayesian inference on the merged Cramer-Rao bounds to produce the
# Hubble constant posterior at multiple h values.
#
# Array tasks map to h values via the H_VALUES array below:
#   task 0 → h=0.600, task 1 → h=0.626, ..., task 5 → h=0.730, ..., task 10 → h=0.860
#
# Worker count: --num_workers is NOT passed explicitly. The default in
# arguments.py uses os.sched_getaffinity(0) - 2, which automatically respects
# SLURM's cgroup CPU limit (16 CPUs allocated -> 14 workers). This is more
# robust than hardcoding a value.
#
# Required environment variables (set by submit_pipeline.sh):
#   RUN_DIR -- workspace directory containing prepared Cramer-Rao bounds
#
# Note: --output and --error are NOT set here; submit_pipeline.sh passes them
# via command line so that shell variables are expanded correctly.

#SBATCH --job-name=emri-evaluate
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# H-value sweep: 0.6 to 0.86 in 11 steps (includes 0.73)
# ---------------------------------------------------------------------------

H_VALUES=(0.600 0.626 0.652 0.678 0.704 0.730 0.756 0.782 0.808 0.834 0.860)

H_VALUE="${H_VALUES[$SLURM_ARRAY_TASK_ID]}"

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

# ---------------------------------------------------------------------------
# Run Bayesian inference
# ---------------------------------------------------------------------------

echo "=== Bayesian Inference (H0 Posterior) ==="
echo "  Run Dir:    $RUN_DIR"
echo "  H value:    $H_VALUE (task $SLURM_ARRAY_TASK_ID of ${#H_VALUES[@]})"
echo "  Node:       $(hostname)"
echo "  CPUs alloc: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "  Workers:    auto-detected from cgroup (SLURM_CPUS_PER_TASK - 2)"
echo ""

# Evaluation code reads simulations/prepared_cramer_rao_bounds.csv relative to CWD.
# Symlink simulations/ → $RUN_DIR/simulations/ so paths resolve correctly.
cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

python -m master_thesis_code "$RUN_DIR" \
    --evaluate \
    --h_value "$H_VALUE" \
    --log_level INFO

echo ""
echo "=== Evaluation Complete (h=$H_VALUE) ==="
