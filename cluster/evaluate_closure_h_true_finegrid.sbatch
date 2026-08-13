#!/usr/bin/env bash
# cluster/evaluate_closure_h_true_finegrid.sbatch -- Multi-truth fine-grid closure test.
#
# Generalized successor to evaluate_closure_h065_finegrid.sbatch.  Evaluates
# Bayesian inference on a rescaled CRB at 21 h-values centred on H_TRUE
# (Δh=0.005, half-width 0.05), enabling a multi-truth bias-vs-h_true sweep
# to test whether the post-Tier-3 residual bias is purely statistical.
#
# Grid widened from ±0.025 to ±0.05 so the panel-extreme truths (0.60, 0.85)
# can resolve a posterior peak that drifts toward the production injection
# truth h=0.73 without railing at the grid boundary.
#
# Required environment variables (set via --export when submitting):
#   RUN_DIR -- workspace directory containing rescaled prepared CRB.
#   H_TRUE  -- the rescaled injection truth (e.g. 0.60, 0.65, 0.70, 0.73, 0.75, 0.80, 0.85).
#
# Strategy: cpu_il queue with --array=0-6 (7 tasks), stride-slicing 21
# h-values across the 7 tasks (3 h-values each).  At ~1.5 min/h on cpu_il
# (128 cores), each task does 3 × 1.5 min ≈ 5 min wall, well inside the
# 30-min walltime.

#SBATCH --job-name=emri-closure-multi
#SBATCH --partition=cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
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
# Build the fine grid centred on H_TRUE (Δh=0.005, half-width 0.05).
# Clamp to the LamCDMScenario h prior support [0.60, 0.86]
# (cosmological_model.py:303-305) so the inference is never asked for an
# h-value outside the prior.  At panel extremes the grid is asymmetric:
#   h_true = 0.60 → 11 points (0.6000 .. 0.6500)
#   h_true = 0.65 → 21 points (0.6000 .. 0.7000)
#   h_true = 0.85 → 13 points (0.8000 .. 0.8600)
# parabolic_refine in test_24 handles arbitrary grids transparently, but the
# panel verdict will FLAG truths whose MAP rails at the (clamped) edge.
# ---------------------------------------------------------------------------

# Use awk for portable float arithmetic (bash can't do floats natively).
# Generate H_TRUE-0.05, H_TRUE-0.045, ..., H_TRUE+0.05 rounded to 4 decimals
# and drop values outside the LamCDM prior.
# LC_ALL=C forces "." as the decimal separator regardless of cluster locale —
# without it, locales like de_DE.UTF-8 emit "," and break the H_VALUES array.
mapfile -t H_VALUES_ALL < <(LC_ALL=C awk -v h="$H_TRUE" 'BEGIN{
    for (i = -10; i <= 10; i++) {
        v = h + i*0.005
        if (v + 1e-9 >= 0.60 && v - 1e-9 <= 0.86) printf "%.4f\n", v
    }
}')

# Need at least ~11 points to support parabolic-refine + bootstrap σ_boot.
if [[ ${#H_VALUES_ALL[@]} -lt 11 ]]; then
    echo "ERROR: clamped grid has only ${#H_VALUES_ALL[@]} points; minimum 11 required" >&2
    echo "       (H_TRUE=$H_TRUE pushes the ±0.05 window outside [0.60, 0.86])" >&2
    exit 1
fi

# Stride-slice across N tasks: task k → indices k, k+N, k+2N, ...
# N is taken from SLURM_ARRAY_TASK_COUNT so the same sbatch handles cpu_il
# (--array=0-6, N=7) and dev_cpu_il (--array=0-3, N=4) without edits.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-7}"
TASK_H_VALUES=()
for ((i = TASK_ID; i < ${#H_VALUES_ALL[@]}; i += TASK_COUNT)); do
    TASK_H_VALUES+=("${H_VALUES_ALL[$i]}")
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
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
echo "  Task:                   $TASK_ID of $TASK_COUNT"
echo "  H values this task:     ${TASK_H_VALUES[*]}"
echo "  Full grid (${#H_VALUES_ALL[@]} values):   ${H_VALUES_ALL[*]}"
echo "  Node:                   $(hostname)"
echo ""

for H_VALUE in "${TASK_H_VALUES[@]}"; do
    echo "--- Bayesian Inference at h=$H_VALUE ---"
    python -m darksiren_emri "$RUN_DIR" \
        --evaluate \
        --h_value "$H_VALUE" \
        --pdet_dl_bins 60 \
        --pdet_mass_bins 40 \
        --log_level INFO
    echo "--- Done h=$H_VALUE ---"
    echo ""
done

echo "=== Multi-truth closure eval complete (h_true=$H_TRUE, task $TASK_ID) ==="
