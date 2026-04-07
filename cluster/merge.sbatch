#!/usr/bin/env bash
# cluster/merge.sbatch -- CPU job to merge per-task CSVs and prepare detections.
#
# This script is submitted by submit_pipeline.sh after all simulation array
# tasks complete. It runs two steps:
#   1. emri-merge: combines per-task Cramer-Rao bounds CSVs into one file
#      and deletes the per-task sources
#   2. emri-prepare: filters and prepares detections for Bayesian inference
#
# Output files:
#   $RUN_DIR/simulations/cramer_rao_bounds.csv         (merged)
#   $RUN_DIR/simulations/prepared_cramer_rao_bounds.csv (prepared)
#
# Required environment variables (set by submit_pipeline.sh):
#   RUN_DIR -- workspace directory containing simulation output
#
# Note: --output and --error are NOT set here; submit_pipeline.sh passes them
# via command line so that shell variables are expanded correctly.

#SBATCH --job-name=emri-merge
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

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

# ---------------------------------------------------------------------------
# Step 1: Merge per-task Cramer-Rao bounds CSVs
# ---------------------------------------------------------------------------

echo "=== Merge Cramer-Rao Bounds ==="
echo "  Run Dir: $RUN_DIR"
echo "  Node:    $(hostname)"
echo ""

emri-merge --workdir "$RUN_DIR" --delete-sources

# ---------------------------------------------------------------------------
# Step 2: Prepare detections for Bayesian inference
# ---------------------------------------------------------------------------

echo ""
echo "=== Prepare Detections ==="

emri-prepare --workdir "$RUN_DIR" --seed "${PREPARE_SEED:-42}" --force

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "=== Merge & Prepare Complete ==="
echo "  Merged:   $RUN_DIR/simulations/cramer_rao_bounds.csv"
echo "  Prepared: $RUN_DIR/simulations/prepared_cramer_rao_bounds.csv"
