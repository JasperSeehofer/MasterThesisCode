#!/usr/bin/env bash
# cluster/evaluate_closure_h_true_finegrid.sbatch -- Multi-truth fine-grid closure test.
#
# Generalized successor to evaluate_closure_h065_finegrid.sbatch.  Evaluates
# Bayesian inference on a rescaled CRB at 11 h-values centred on H_TRUE
# (Δh=0.005), enabling a multi-truth bias-vs-h_true sweep to test whether the
# post-Tier-3 residual bias is purely statistical.
#
# Required environment variables (set via --export when submitting):
#   RUN_DIR -- workspace directory containing rescaled prepared CRB.
#   H_TRUE  -- the rescaled injection truth (e.g. 0.60, 0.65, 0.70, 0.73, 0.75, 0.80, 0.85).
#
# Strategy: dev_cpu_il caps at 1 running + 3 queued.  Submit --array=0-3 and
# stride-slice 11 h-values across the 4 tasks.  At ~3 min/h on dev_cpu_il,
# each task does ≤3 h × 3 min ≈ 9 min wall, well inside 30-min walltime.

#SBATCH --job-name=emri-closure-multi
#SBATCH --partition=dev_cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate environment variables
# ---------------------------------------------------------------------------

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set." >&2
    exit 1
fi
if [[ -z "${H_TRUE:-}" ]]; then
    echo "ERROR: H_TRUE is not set (e.g. H_TRUE=0.70)." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the 11-point fine grid centred on H_TRUE (Δh=0.005, half-width 0.025)
# ---------------------------------------------------------------------------

# Use awk for portable float arithmetic (bash can't do floats natively).
# Generate H_TRUE-0.025, H_TRUE-0.020, ..., H_TRUE+0.025 rounded to 4 decimals.
# LC_ALL=C forces "." as the decimal separator regardless of cluster locale —
# without it, locales like de_DE.UTF-8 emit "," and break the H_VALUES array.
mapfile -t H_VALUES_ALL < <(LC_ALL=C awk -v h="$H_TRUE" 'BEGIN{
    for (i = -5; i <= 5; i++) printf "%.4f\n", h + i*0.005
}')

if [[ ${#H_VALUES_ALL[@]} -ne 11 ]]; then
    echo "ERROR: expected 11 h-values, got ${#H_VALUES_ALL[@]}" >&2
    exit 1
fi

# Stride-slice across 4 tasks: task k → indices k, k+4, k+8 (3, 3, 3, 2 values)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_H_VALUES=()
for ((i = TASK_ID; i < ${#H_VALUES_ALL[@]}; i += 4)); do
    TASK_H_VALUES+=("${H_VALUES_ALL[$i]}")
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Archive old fine-grid posteriors (task 0 only, before any task starts writing)
# ---------------------------------------------------------------------------

if [[ "$TASK_ID" -eq 0 ]]; then
    POSTERIORS_DIR="$RUN_DIR/simulations/posteriors"
    POSTERIORS_BH_DIR="$RUN_DIR/simulations/posteriors_with_bh_mass"
    ARCHIVE_BASE="$RUN_DIR/simulations/archive"

    _has_old_files() {
        local dir="$1"
        [[ -d "$dir" ]] && compgen -G "$dir/h_*.json" > /dev/null 2>&1
    }

    if _has_old_files "$POSTERIORS_DIR" || _has_old_files "$POSTERIORS_BH_DIR"; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ARCHIVE_DIR="$ARCHIVE_BASE/closure_h${H_TRUE}_${TIMESTAMP}"
        mkdir -p "$ARCHIVE_DIR/posteriors" "$ARCHIVE_DIR/posteriors_with_bh_mass"

        cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by_job": "$SLURM_JOB_ID",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-multi-truth-closure cleanup",
  "h_grid": "${H_VALUES_ALL[*]}",
  "h_true": "$H_TRUE",
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
# Run Bayesian inference for each h-value assigned to this task
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

echo "=== Multi-truth Closure Test (h_true=$H_TRUE) ==="
echo "  RUN_DIR:                $RUN_DIR"
echo "  H_TRUE (injection):     $H_TRUE"
echo "  Task:                   $TASK_ID of 4"
echo "  H values this task:     ${TASK_H_VALUES[*]}"
echo "  Full grid (11 values):  ${H_VALUES_ALL[*]}"
echo "  Node:                   $(hostname)"
echo ""

for H_VALUE in "${TASK_H_VALUES[@]}"; do
    echo "--- Bayesian Inference at h=$H_VALUE ---"
    python -m master_thesis_code "$RUN_DIR" \
        --evaluate \
        --h_value "$H_VALUE" \
        --pdet_dl_bins 60 \
        --pdet_mass_bins 40 \
        --log_level INFO
    echo "--- Done h=$H_VALUE ---"
    echo ""
done

echo "=== Multi-truth closure eval complete (h_true=$H_TRUE, task $TASK_ID) ==="
