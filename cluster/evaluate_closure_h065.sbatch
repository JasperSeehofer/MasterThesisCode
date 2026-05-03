#!/usr/bin/env bash
# cluster/evaluate_closure_h065.sbatch -- Audit A7 (lean) closure test at h_true=0.65.
#
# Runs Bayesian inference on the rescaled Cramer-Rao bounds (h_true=0.65) at a
# focused h-grid bracketing 0.65. The CRB CSV under $RUN_DIR/simulations/ is
# expected to have been produced by scripts/bias_investigation/test_17_rescale_crb_to_h065.py
# followed by scripts/prepare_detections.py (or equivalent).
#
# Closure-test acceptance gates:
#   G7a  MAP ∈ [h_true ± 1.5·σ_boot] = [0.635, 0.665] → pipeline unbiased.
#   G7b  MAP ≈ 0.665–0.680 (additive +0.025 like h=0.73 case) → h-independent
#                                                                   additive systematic.
#   G7c  MAP ≈ 0.680 (relative +3.4% bias matching h=0.73)    → multiplicative bias.
#   G7d  MAP ≈ 0.730 → pipeline TUNED to 0.73; HALT paper.
#
# Required environment variables (set via --export when submitting):
#   RUN_DIR -- workspace directory containing rescaled prepared CRB.

#SBATCH --job-name=emri-closure-h065
#SBATCH --partition=dev_cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Closure-test h-grid: 10 values bracketing h_true=0.65
#   Wings (0.55, 0.58, 0.62, 0.68, 0.70, 0.73): coarse
#   Peak  (0.62, 0.64, 0.65, 0.66, 0.68): fine around h_true
# ---------------------------------------------------------------------------

H_VALUES=(0.600 0.650 0.700 0.750)

H_VALUE="${H_VALUES[$SLURM_ARRAY_TASK_ID]}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Archive old closure posteriors (task 0 only)
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
        ARCHIVE_DIR="$ARCHIVE_BASE/closure_${TIMESTAMP}"
        mkdir -p "$ARCHIVE_DIR/posteriors" "$ARCHIVE_DIR/posteriors_with_bh_mass"

        cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by_job": "$SLURM_JOB_ID",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-closure-test cleanup",
  "h_grid": "${H_VALUES[*]}",
  "h_true": "0.65",
  "source_run_dir": "$RUN_DIR"
}
METAEOF

        if _has_old_files "$POSTERIORS_DIR"; then
            mv "$POSTERIORS_DIR"/h_*.json "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
            mv "$POSTERIORS_DIR"/combined_posterior.json "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
        fi
        if _has_old_files "$POSTERIORS_BH_DIR"; then
            mv "$POSTERIORS_BH_DIR"/h_*.json "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
            mv "$POSTERIORS_BH_DIR"/combined_posterior.json "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
        fi
        echo "Archived old closure posteriors to $ARCHIVE_DIR/"
    fi
fi

sleep 3

# ---------------------------------------------------------------------------
# Run Bayesian inference at this h-value
# ---------------------------------------------------------------------------

echo "=== Closure Test (Audit A7) — Bayesian Inference at h=$H_VALUE ==="
echo "  RUN_DIR (h_true=0.65):  $RUN_DIR"
echo "  H value (trial):        $H_VALUE (task $SLURM_ARRAY_TASK_ID of ${#H_VALUES[@]})"
echo "  Node:                   $(hostname)"
echo "  CPUs:                   ${SLURM_CPUS_PER_TASK:-?}"
echo ""

cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

python -m master_thesis_code "$RUN_DIR" \
    --evaluate \
    --h_value "$H_VALUE" \
    --pdet_dl_bins 60 \
    --pdet_mass_bins 40 \
    --log_level INFO

echo ""
echo "=== Closure-test eval complete (h=$H_VALUE) ==="
