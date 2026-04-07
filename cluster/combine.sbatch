#!/usr/bin/env bash
# cluster/combine.sbatch -- Combine per-h posteriors into joint H0 posterior.
#
# This script is submitted by submit_pipeline.sh after the evaluate array
# completes. It combines individual h_*.json files into combined_posterior.json
# for both posterior variants (with and without BH mass).
#
# Required environment variables (set by submit_pipeline.sh):
#   RUN_DIR -- workspace directory containing evaluation posteriors
#
# Note: --output and --error are NOT set here; submit_pipeline.sh passes them
# via command line so that shell variables are expanded correctly.

#SBATCH --job-name=emri-combine
#SBATCH --partition=cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

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
# Symlink simulations/ so relative paths resolve correctly
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

# ---------------------------------------------------------------------------
# Combine posteriors
# ---------------------------------------------------------------------------

echo "=== Posterior Combination ==="
echo "  Run Dir:  $RUN_DIR"
echo "  Node:     $(hostname)"
echo "  Strategy: physics-floor"
echo ""

python -m master_thesis_code "$RUN_DIR/simulations" \
    --combine \
    --strategy physics-floor \
    --log_level INFO

echo ""
echo "=== Combination Complete ==="
echo ""
echo "Output files:"
ls -la "$RUN_DIR/simulations/posteriors/combined_posterior.json" 2>/dev/null || echo "  (no posteriors/combined_posterior.json)"
ls -la "$RUN_DIR/simulations/posteriors_with_bh_mass/combined_posterior.json" 2>/dev/null || echo "  (no posteriors_with_bh_mass/combined_posterior.json)"
