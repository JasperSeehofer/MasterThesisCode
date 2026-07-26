#!/usr/bin/env bash
# pp_fullpower.sbatch — P-P impostor-harness full-power campaign (15 cells).
#
# One array task per cell: 3 modes x 3 truths at the primary operating point
# (z_support=0.43), plus the B_num-characterization axis (generator_marginal
# at the two high-impostor points) x 3 truths. n_realizations=2000 per cell.
# See ppwork_plan.md (same directory) for the calibration provenance.
#
# PREREQUISITE: the cluster checkout must be on branch feat/pp-impostor-harness
# (commit 44ee912) — catalogue mode does not exist on main.
#
# Self-contained synthetic harness: NO GPU, NO galaxy-catalog file, NO
# simulations dir/symlink (verified from a bare worktree).
#
# Submit via a wrapper that creates RUN_DIR and passes --output/--error/--export
# on the command line (do NOT hardcode %A/%a paths here), e.g.:
#   RUN_DIR=$WORKSPACE/pp_fullpower_20260727
#   mkdir -p "$RUN_DIR/logs"
#   sbatch --array=0-8 \
#     --output="$RUN_DIR/logs/%x_%A_%a.out" --error="$RUN_DIR/logs/%x_%A_%a.err" \
#     --export=ALL,RUN_DIR="$RUN_DIR",PROJECT_ROOT="$HOME/MasterThesisCode" \
#     pp_fullpower.sbatch

# ---- 1. resources -------------------------------------------------------------
#SBATCH --job-name=pp-fullpower
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
# Measured: ~10 min/cell at n=2000 on the dev machine (0.30 s/realization);
# x1.5 margin plus cluster-core headroom -> 1 h (cap 72 h not remotely needed).
#SBATCH --ntasks=1
#SBATCH --array=0-8

set -euo pipefail

# ---- 2. Environment (identical in every job) ----------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"   # loads modules; exports VENV_PATH
source "$VENV_PATH/bin/activate"

# ---- 3. Validate inputs --------------------------------------------------------
: "${RUN_DIR:?RUN_DIR not set — the submit wrapper must export it}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?array job required (submit with --array=0-8)}"

# ---- 3a. Cell tables (index = SLURM_ARRAY_TASK_ID) -----------------------------
MODES=(  lcat lcat lcat  absolute absolute absolute  generator_marginal generator_marginal generator_marginal )
ZSUPS=(  0.30 0.30 0.30  0.30 0.30 0.30  0.30 0.30 0.30 )
SKYFRACS=( 2e-4 2e-4 2e-4  2e-4 2e-4 2e-4  2e-4 2e-4 2e-4 )
TRUTHS=( 0.62 0.72 0.84  0.62 0.72 0.84  0.62 0.72 0.84 )

MODE="${MODES[$TASK_ID]}"
ZSUP="${ZSUPS[$TASK_ID]}"
SKYFRAC="${SKYFRACS[$TASK_ID]}"
TRUTH="${TRUTHS[$TASK_ID]}"
SEED=$(( 20260727100 + TASK_ID ))

MODE_TAG="$MODE"
[[ "$MODE" == "generator_marginal" ]] && MODE_TAG="genmarg"
OUT="$RUN_DIR/pp_cat_${MODE_TAG}_zs${ZSUP}_sky${SKYFRAC}_h${TRUTH}.json"

cd "$PROJECT_ROOT"

# ---- 3b. Idempotency ------------------------------------------------------------
[[ -f "$OUT" ]] && { echo "SKIP: $OUT exists"; exit 0; }

# ---- 4. the actual work ----------------------------------------------------------
echo "=== $SLURM_JOB_NAME task=$TASK_ID mode=$MODE zs=$ZSUP sky=$SKYFRAC h=$TRUTH seed=$SEED node=$(hostname) ==="
echo "=== branch: $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD) @ $(git -C "$PROJECT_ROOT" rev-parse --short HEAD) ==="

python -m master_thesis_code.validation.pp_coverage \
    --catalogue-mode \
    --mixture-mode "$MODE" \
    --z-support "$ZSUP" \
    --sky-frac "$SKYFRAC" \
    --truths "$TRUTH" \
    --n-realizations 2000 \
    --n-events 120 \
    --seed "$SEED" \
    --output "$OUT"

echo "=== done: task=$TASK_ID -> $OUT ==="
