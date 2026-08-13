#!/usr/bin/env bash
# cluster/evaluate_production_h0p73_superdense.sbatch -- Phase 50 superdense production sweep at h=0.73.
#
# Extends evaluate_production_h0p73_dense.sbatch (63-point) by adding a
# 20-point super-dense band at Δh=0.0005 across [0.720, 0.740].  Total
# 83 h-values:
#   Tier 0 (super-dense): Δh=0.0005 mid-points in (0.720, 0.740) → 20 points
#   Tier 1 (dense core):  Δh=0.001  across [0.710, 0.750]        → 41 points
#   Tier 2 (left wing):   Δh=0.010  across [0.600, 0.700]        → 11 points
#   Tier 3 (right wing):  Δh=0.010  across [0.760, 0.860]        → 11 points
#
# Rationale: the dense Δh=0.001 grid hit the HDI68=0.001 grid-resolution
# floor in `paper_m_z_improvement.pdf` at N=1473.  σ_boot post-H3-fix is
# ~0.0037 (2D); Δh_super = 0.0005 ≈ σ_boot/7 lets the 2D channel's true
# tightening show below 0.001.  See handoff §B in
# .planning/HANDOFF-PLOTTING-OVERHAUL-20260516.md.
#
# Required environment variables (set via --export when submitting):
#   RUN_DIR -- workspace directory containing rescaled prepared CRB.
#
# Strategy: cpu_il queue with --array=0-8 (9 tasks), stride-slicing 83
# h-values across the 9 tasks (~9-10 each).  At ~2 min/h on cpu_il
# (128 cores), each task does 10 × 2 min ≈ 20 min wall, inside 45:00.

#SBATCH --job-name=emri-prod-superdense
#SBATCH --partition=cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:45:00

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate environment variables
# ---------------------------------------------------------------------------

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup (must precede any python3 -c invocation — the bare
# system python3 on compute nodes does not have numpy)
# ---------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Build the 83-point non-uniform production grid.  H_TRUE=0.73 hardcoded.
#   Tier 0: Δh=0.0005 mid-points in (0.720, 0.740) → {0.7205, 0.7215, ..., 0.7395} = 20 pts
#   Tier 1: Δh=0.001  across [0.710, 0.750]        → 41 pts (includes truth h=0.730)
#   Tier 2: Δh=0.010  across [0.600, 0.700]        → 11 pts
#   Tier 3: Δh=0.010  across [0.760, 0.860]        → 11 pts
# Span = LamCDM prior support [0.60, 0.86] per cosmological_model.py:303-305.
# ---------------------------------------------------------------------------

H_TRUE=0.73

mapfile -t H_VALUES_ALL < <(LC_ALL=C python3 -c "
import numpy as np
core = np.round(np.arange(0.710, 0.750 + 0.0005, 0.001), 4)
super_dense = np.round(np.arange(0.7205, 0.7400, 0.001), 4)  # 20 mid-points filling gaps in [0.7205, 0.7395]
left = np.round(np.arange(0.600, 0.710, 0.010), 4)
right = np.round(np.arange(0.760, 0.860 + 0.005, 0.010), 4)
for v in sorted(set(np.concatenate([left, core, super_dense, right]).tolist())):
    print(f'{v:.4f}')
")

if [[ ${#H_VALUES_ALL[@]} -ne 83 ]]; then
    echo "ERROR: expected 83 grid points, got ${#H_VALUES_ALL[@]}" >&2
    exit 1
fi

# Stride-slice across N tasks: task k → indices k, k+N, k+2N, ...
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-9}"
TASK_H_VALUES=()
for ((i = TASK_ID; i < ${#H_VALUES_ALL[@]}; i += TASK_COUNT)); do
    TASK_H_VALUES+=("${H_VALUES_ALL[$i]}")
done

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
        ARCHIVE_DIR="$ARCHIVE_BASE/production_h${H_TRUE}_superdense_${TIMESTAMP}"
        mkdir -p "$ARCHIVE_DIR/posteriors" "$ARCHIVE_DIR/posteriors_with_bh_mass"

        cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by_job": "$SLURM_JOB_ID",
  "git_commit": "$(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-superdense-sweep cleanup",
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

echo "=== Phase 50 Superdense Production Sweep (h_true=$H_TRUE) ==="
echo "  RUN_DIR:                $RUN_DIR"
echo "  H_TRUE (injection):     $H_TRUE"
echo "  Task:                   $TASK_ID of $TASK_COUNT"
echo "  H values this task:     ${TASK_H_VALUES[*]}"
echo "  Full grid (${#H_VALUES_ALL[@]} values: 41 dense Δh=0.001 + 20 super-dense Δh=0.0005 + 22 wings Δh=0.010)"
echo "  Node:                   $(hostname)"
echo ""

for H_VALUE in "${TASK_H_VALUES[@]}"; do
    H_FILE_FRAG=$(LC_ALL=C python3 -c "print(str(float('$H_VALUE')).replace('.', '_'))")
    POSTERIOR_FILE="$RUN_DIR/simulations/posteriors/h_${H_FILE_FRAG}.json"
    POSTERIOR_FILE_BH="$RUN_DIR/simulations/posteriors_with_bh_mass/h_${H_FILE_FRAG}.json"
    if [[ -f "$POSTERIOR_FILE" && -f "$POSTERIOR_FILE_BH" ]]; then
        echo "--- Skipping h=$H_VALUE (both 1D + 2D posteriors already exist) ---"
        echo ""
        continue
    fi
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

echo "=== Phase 50 superdense sweep complete (h_true=$H_TRUE, task $TASK_ID) ==="
