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

# ---------------------------------------------------------------------------
# Save baseline snapshot and compare against previous baseline
# ---------------------------------------------------------------------------

echo ""
echo "=== Saving Baseline & Comparison ==="
python -m master_thesis_code "$RUN_DIR/simulations" \
    --save_baseline \
    --compare_baseline "$PROJECT_ROOT/.planning/debug/baseline.json" \
    --log_level INFO

# Copy baseline into RUN_DIR so it's included in rsync
cp "$PROJECT_ROOT/.planning/debug/baseline.json" "$RUN_DIR/baseline.json" 2>/dev/null || true
# Copy comparison report too
cp "$PROJECT_ROOT/.planning/debug/comparison_"*.md "$RUN_DIR/" 2>/dev/null || true
echo "Baseline + comparison report copied to $RUN_DIR/"

# ---------------------------------------------------------------------------
# Archive old figures (if any exist from a previous run)
# ---------------------------------------------------------------------------

FIGURES_DIR="$RUN_DIR/simulations/figures"

_has_old_figures() {
    [[ -d "$FIGURES_DIR" ]] && compgen -G "$FIGURES_DIR/*.pdf" > /dev/null 2>&1
}

if _has_old_figures; then
    ARCHIVE_BASE="$RUN_DIR/simulations/archive"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVE_DIR="$ARCHIVE_BASE/figures_${TIMESTAMP}"
    mkdir -p "$ARCHIVE_DIR"

    cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by_job": "$SLURM_JOB_ID",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_branch": "$(cd "$PROJECT_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-figure-generation cleanup: archiving old figures before regeneration",
  "source_run_dir": "$RUN_DIR"
}
METAEOF

    mv "$FIGURES_DIR"/*.pdf "$ARCHIVE_DIR/" 2>/dev/null || true
    echo "Archived old figures to $ARCHIVE_DIR/"
else
    echo "No old figures to archive."
fi

# ---------------------------------------------------------------------------
# Generate figures on-cluster (avoids rsyncing ~600 MB/h of raw posteriors)
# ---------------------------------------------------------------------------

echo ""
echo "=== Generating Figures ==="
python -m master_thesis_code "$RUN_DIR/simulations" \
    --generate_figures "$RUN_DIR/simulations" \
    --log_level INFO

echo ""
echo "Figures saved to $RUN_DIR/simulations/figures/"
ls -la "$RUN_DIR/simulations/figures/"*.pdf 2>/dev/null || echo "  (no PDF figures found)"
