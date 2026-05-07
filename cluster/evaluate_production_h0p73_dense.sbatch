#!/usr/bin/env bash
# cluster/evaluate_production_h0p73_dense.sbatch -- Production fine-grid sweep at h=0.73.
#
# Adapted from evaluate_closure_h_true_finegrid.sbatch.  Single-truth
# (H_TRUE=0.73 hardcoded) production sweep with a non-uniform 63-point
# h-grid: dense Δh=0.001 across [0.710, 0.750] (truth ± 0.02 ≈ ±5σ_boot
# post-H3-fix), coarse Δh=0.010 wings spanning [0.600, 0.700] and
# [0.760, 0.860] for posterior-tail visualisation.
#
# Goal: produce paper-grade MAP and σ_boot where Δh is not the
# resolution-limiting factor.  σ_boot post-H3-fix is ~0.0037 (2D) /
# ~0.0047 (1D); Δh_core = 0.001 ≈ σ_boot/4 keeps parabolic refine and
# bootstrap distributions well-resolved.  See Phase 48 plan at
# ~/.claude/plans/please-look-at-the-velvety-quail.md.
#
# Required environment variables (set via --export when submitting):
#   RUN_DIR -- workspace directory containing rescaled prepared CRB.
#
# Strategy: cpu_il queue with --array=0-6 (7 tasks), stride-slicing 63
# h-values across the 7 tasks (~9 each).  At ~2 min/h on cpu_il
# (128 cores), each task does 9 × 2 min ≈ 18 min wall, well inside the
# 30-min walltime.

#SBATCH --job-name=emri-prod-h0p73
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

# ---------------------------------------------------------------------------
# Build the 63-point non-uniform production grid.  H_TRUE=0.73 hardcoded.
# Tier 1 (dense core): Δh=0.001 across [0.710, 0.750] → 41 points.
# Tier 2 (left wing):  Δh=0.010 across [0.600, 0.700] → 11 points.
# Tier 3 (right wing): Δh=0.010 across [0.760, 0.860] → 11 points.
# Truth h=0.730 lies on the dense core (index 20: 0.710 + 20*0.001).
# Span = LamCDM prior support [0.60, 0.86] per cosmological_model.py:303-305.
# ---------------------------------------------------------------------------

H_TRUE=0.73

mapfile -t H_VALUES_ALL < <(LC_ALL=C python3 -c "
import numpy as np
core = np.round(np.arange(0.710, 0.750 + 0.0005, 0.001), 4)
left = np.round(np.arange(0.600, 0.710, 0.010), 4)
right = np.round(np.arange(0.760, 0.860 + 0.005, 0.010), 4)
for v in sorted(set(np.concatenate([left, core, right]).tolist())):
    print(f'{v:.4f}')
")

if [[ ${#H_VALUES_ALL[@]} -ne 63 ]]; then
    echo "ERROR: expected 63 grid points, got ${#H_VALUES_ALL[@]}" >&2
    exit 1
fi

# Stride-slice across N tasks: task k → indices k, k+N, k+2N, ...
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-7}"
TASK_H_VALUES=()
for ((i = TASK_ID; i < ${#H_VALUES_ALL[@]}; i += TASK_COUNT)); do
    TASK_H_VALUES+=("${H_VALUES_ALL[$i]}")
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Archive old production-sweep posteriors (task 0 only, before any task writes)
#
# Gated on ARCHIVE_OLD=yes (default no) so re-running this sbatch over an
# in-progress RUN_DIR (idempotent recovery from TIMEOUT) does NOT wipe the
# already-completed posteriors.  Pass --export=ALL,RUN_DIR=...,ARCHIVE_OLD=yes
# only when a true clean restart of an existing RUN_DIR is intended.
# ---------------------------------------------------------------------------

if [[ "$TASK_ID" -eq 0 && "${ARCHIVE_OLD:-no}" == "yes" ]]; then
    POSTERIORS_DIR="$RUN_DIR/simulations/posteriors"
    POSTERIORS_BH_DIR="$RUN_DIR/simulations/posteriors_with_bh_mass"
    ARCHIVE_BASE="$RUN_DIR/simulations/archive"

    _has_old_files() {
        local dir="$1"
        [[ -d "$dir" ]] && compgen -G "$dir/h_*.json" > /dev/null 2>&1
    }

    if _has_old_files "$POSTERIORS_DIR" || _has_old_files "$POSTERIORS_BH_DIR"; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ARCHIVE_DIR="$ARCHIVE_BASE/production_h${H_TRUE}_${TIMESTAMP}"
        mkdir -p "$ARCHIVE_DIR/posteriors" "$ARCHIVE_DIR/posteriors_with_bh_mass"

        cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by_job": "$SLURM_JOB_ID",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-production-sweep cleanup",
  "h_grid_n": "${#H_VALUES_ALL[@]}",
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
        echo "Archived old production posteriors to $ARCHIVE_DIR/"
    fi
fi

sleep 3

# ---------------------------------------------------------------------------
# Run Bayesian inference for each h-value assigned to this task
# ---------------------------------------------------------------------------

cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

echo "=== Production Fine-Grid Sweep (h_true=$H_TRUE) ==="
echo "  RUN_DIR:                $RUN_DIR"
echo "  H_TRUE (injection):     $H_TRUE"
echo "  Task:                   $TASK_ID of $TASK_COUNT"
echo "  H values this task:     ${TASK_H_VALUES[*]}"
echo "  Full grid (${#H_VALUES_ALL[@]} values, non-uniform: 41 dense @Δh=0.001 in [0.710,0.750] + 22 wings @Δh=0.010)"
echo "  Node:                   $(hostname)"
echo ""

for H_VALUE in "${TASK_H_VALUES[@]}"; do
    # Idempotent skip: posterior file naming mirrors Python's
    # str(float(h)).replace(".", "_"), which strips trailing zeros
    # (0.7300 -> "h_0_73", 0.7340 -> "h_0_734", 0.6000 -> "h_0_6").
    # Use python3 to compute the same fragment to avoid bash float foot-guns.
    H_FILE_FRAG=$(LC_ALL=C python3 -c "print(str(float('$H_VALUE')).replace('.', '_'))")
    POSTERIOR_FILE="$RUN_DIR/simulations/posteriors/h_${H_FILE_FRAG}.json"
    POSTERIOR_FILE_BH="$RUN_DIR/simulations/posteriors_with_bh_mass/h_${H_FILE_FRAG}.json"
    if [[ -f "$POSTERIOR_FILE" && -f "$POSTERIOR_FILE_BH" ]]; then
        echo "--- Skipping h=$H_VALUE (both 1D + 2D posteriors already exist) ---"
        echo ""
        continue
    fi
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

echo "=== Production sweep complete (h_true=$H_TRUE, task $TASK_ID) ==="
