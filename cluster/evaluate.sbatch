#!/usr/bin/env bash
# cluster/evaluate.sbatch -- CPU array job for Bayesian inference (H0 posterior).
#
# This script is submitted by submit_pipeline.sh after the merge job completes.
# It runs Bayesian inference on the merged Cramer-Rao bounds to produce the
# Hubble constant posterior at multiple h values.
#
# Array tasks map to h values via the H_VALUES array below:
#   38 values: 0.010 step in wings (0.60–0.67, 0.79–0.86), 0.005 step in peak (0.68–0.785)
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
#SBATCH --partition=cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:15:00

set -euo pipefail

# ---------------------------------------------------------------------------
# H-value sweep: 0.60 to 0.86, 38 values (hybrid grid)
#   Wings (0.60–0.67, 0.79–0.86): step 0.010 — low posterior weight, coarse is fine
#   Peak  (0.68–0.785):           step 0.005 — dense where MAP shifts matter
# Upper bound 0.86 matches LamCDMScenario.h.upper_limit in cosmological_model.py
# ---------------------------------------------------------------------------

H_VALUES=(0.600 0.610 0.620 0.630 0.640 0.650 0.660 0.670 0.680 0.685 0.690 0.695 0.700 0.705 0.710 0.715 0.720 0.725 0.730 0.735 0.740 0.745 0.750 0.755 0.760 0.765 0.770 0.775 0.780 0.785 0.790 0.800 0.810 0.820 0.830 0.840 0.850 0.860)

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
# Archive old posteriors (task 0 only — runs before other tasks start writing)
# ---------------------------------------------------------------------------

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    POSTERIORS_DIR="$RUN_DIR/simulations/posteriors"
    POSTERIORS_BH_DIR="$RUN_DIR/simulations/posteriors_with_bh_mass"
    ARCHIVE_BASE="$RUN_DIR/simulations/archive"

    _has_old_files() {
        local dir="$1"
        [[ -d "$dir" ]] && compgen -G "$dir/h_*.json" > /dev/null 2>&1
    }

    if _has_old_files "$POSTERIORS_DIR" || _has_old_files "$POSTERIORS_BH_DIR"; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ARCHIVE_DIR="$ARCHIVE_BASE/eval_${TIMESTAMP}"
        mkdir -p "$ARCHIVE_DIR"

        # Save metadata for traceability
        cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by_job": "$SLURM_JOB_ID",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_branch": "$(cd "$PROJECT_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-evaluation cleanup: archiving old posteriors before new evaluation run",
  "h_grid": "$(echo "${H_VALUES[@]}")",
  "source_run_dir": "$RUN_DIR"
}
METAEOF

        # Move old posteriors to archive
        if _has_old_files "$POSTERIORS_DIR"; then
            mkdir -p "$ARCHIVE_DIR/posteriors"
            mv "$POSTERIORS_DIR"/h_*.json "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
            mv "$POSTERIORS_DIR"/combined_posterior.json "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
            mv "$POSTERIORS_DIR"/diagnostic_report.md "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
            mv "$POSTERIORS_DIR"/comparison_table.md "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
            echo "Archived old posteriors/ to $ARCHIVE_DIR/posteriors/"
        fi

        if _has_old_files "$POSTERIORS_BH_DIR"; then
            mkdir -p "$ARCHIVE_DIR/posteriors_with_bh_mass"
            mv "$POSTERIORS_BH_DIR"/h_*.json "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
            mv "$POSTERIORS_BH_DIR"/combined_posterior.json "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
            mv "$POSTERIORS_BH_DIR"/diagnostic_report.md "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
            mv "$POSTERIORS_BH_DIR"/comparison_table.md "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
            echo "Archived old posteriors_with_bh_mass/ to $ARCHIVE_DIR/posteriors_with_bh_mass/"
        fi
    else
        echo "No old posteriors to archive."
    fi
fi

# Brief pause so task 0 archive completes before other tasks start writing
sleep 3

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

# P_det grid resolution: explicit flags for reproducibility (Phase 33)
python -m master_thesis_code "$RUN_DIR" \
    --evaluate \
    --h_value "$H_VALUE" \
    --pdet_dl_bins 60 \
    --pdet_mass_bins 40 \
    --log_level INFO

echo ""
echo "=== Evaluation Complete (h=$H_VALUE) ==="
