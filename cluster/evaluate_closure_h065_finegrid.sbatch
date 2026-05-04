#!/usr/bin/env bash
# cluster/evaluate_closure_h065_finegrid.sbatch -- Audit A7 fine-grid closure test at h_true=0.65.
#
# Evaluates Bayesian inference on the rescaled Cramer-Rao bounds at 11 h-values
# spanning a tight Δh=0.005 window around h_true=0.65 — mirrors the production
# h=0.73 grid resolution (cluster/evaluate.sbatch:37, peak Δh=0.005) so that
# any +0.025-magnitude bias of the sort observed at h=0.73 is resolvable on
# this grid.
#
# Dev_cpu_il caps queued tasks at 4. Strategy: run --array=0-3 and have each
# task evaluate a stride-of-4 slice of the 11-value grid (3, 3, 3, 2 values
# per task). At ~3 min per h on dev_cpu_il, ≤3 h-values per task ≤ 9 min
# wall-clock, well inside the 30-min walltime envelope.
#
# Required environment variables (set via --export when submitting):
#   RUN_DIR -- workspace directory containing rescaled prepared CRB.

#SBATCH --job-name=emri-closure-finegrid
#SBATCH --partition=dev_cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Fine-grid h-values: 11 points at Δh=0.005 in [0.625, 0.675] around h_true=0.65
# ---------------------------------------------------------------------------

H_VALUES_ALL=(0.625 0.630 0.635 0.640 0.645 0.650 0.655 0.660 0.665 0.670 0.675)

# Task ID picks every 4th value starting at SLURM_ARRAY_TASK_ID:
#   task 0 → 0.625, 0.645, 0.665   (3 values)
#   task 1 → 0.630, 0.650, 0.670   (3 values)
#   task 2 → 0.635, 0.655, 0.675   (3 values)
#   task 3 → 0.640, 0.660           (2 values)
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

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set." >&2
    exit 1
fi

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
        ARCHIVE_DIR="$ARCHIVE_BASE/closure_finegrid_${TIMESTAMP}"
        mkdir -p "$ARCHIVE_DIR/posteriors" "$ARCHIVE_DIR/posteriors_with_bh_mass"

        cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by_job": "$SLURM_JOB_ID",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-fine-grid-closure cleanup",
  "h_grid": "${H_VALUES_ALL[*]}",
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

# Brief pause so task 0 archive completes before other tasks start writing
sleep 3

# ---------------------------------------------------------------------------
# Run Bayesian inference for each h-value assigned to this task
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

echo "=== Closure Test Fine Grid (Audit A7 redux) ==="
echo "  RUN_DIR (h_true=0.65):  $RUN_DIR"
echo "  Task:                   $TASK_ID of 4"
echo "  H values this task:     ${TASK_H_VALUES[*]}"
echo "  Node:                   $(hostname)"
echo "  CPUs:                   ${SLURM_CPUS_PER_TASK:-?}"
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

echo "=== Closure-test fine-grid eval complete (task $TASK_ID) ==="
