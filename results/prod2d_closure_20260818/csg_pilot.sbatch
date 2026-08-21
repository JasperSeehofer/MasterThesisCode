#!/usr/bin/env bash
# results/prod2d_closure_20260818/csg_pilot.sbatch -- C-SG v3 MANDATORY 4-seed
# C-SG-F pilot (PREREGISTRATION_SELFGEN_CONTROL.md section 6: "Its only outputs
# are sigma_hat_seed and per-seed sigma_h. N and every band are set from it,
# and the false-fail table is published, before the remaining seeds run.")
#
# One (arm=csgf, seed) task per array index via
# `python -m darksiren_emri.validation.selfgen_control` (v3 module, commit
# 7ab5f001). Seeds 910101-910104 (registry start, prereg section 4).
#
# Walltime: banked-fleet anchor = 0.478 CPU-h/seed (~29 min wall,
# single-process, n=200) + C-SG's one-time selection-table sweep (~2 min,
# lru_cached per task). --time=03:00:00 gives >4x margin (gotcha 9:
# JSON-at-end instruments lose everything on a walltime kill).
#
# Idempotent: run_csg_arm_seed skips if $OUT_DIR/<arm>_seed<seed>.json exists.
#
# Required env (export at submit):
#   OUT_DIR  -- e.g. $(ws_find emri)/csg_pilot_20260821/out
# Optional:
#   PROJECT_ROOT (default ~/darksiren-emri), WORK_ROOT_BASE (default OUT_DIR/../work)

#SBATCH --job-name=csg_pilot
#SBATCH --partition=cpu,cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-3

set -euo pipefail

SEEDS=(910101 910102 910103 910104)
ARM="csgf"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set (submit with --array=0-3)." >&2
    exit 1
fi
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

: "${OUT_DIR:?OUT_DIR not set -- export it (e.g. OUT_DIR=\$(ws_find emri)/csg_pilot_20260821/out)}"
mkdir -p "$OUT_DIR"
WORK_ROOT_BASE="${WORK_ROOT_BASE:-$OUT_DIR/../work}"
WORK_ROOT="$WORK_ROOT_BASE/${ARM}_seed${SEED}"
mkdir -p "$WORK_ROOT"

echo "=== $SLURM_JOB_NAME job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$(hostname) arm=$ARM seed=$SEED ==="

cd "$PROJECT_ROOT"
python -m darksiren_emri.validation.selfgen_control \
    --arm "$ARM" \
    --seed "$SEED" \
    --work-root "$WORK_ROOT" \
    --out-dir "$OUT_DIR"

echo "=== done: arm=$ARM seed=$SEED job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID ==="
