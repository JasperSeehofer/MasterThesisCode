#!/usr/bin/env bash
# results/prod2d_closure_20260818/csg_fleet.sbatch -- C-SG v3 remaining fleet
# (42 tasks), submitted ONLY after csg_pilot_bands.py published the bands of
# record with fleet_may_launch=true (prereg section 6 pilot mandate).
#
#   csgf   seeds 910105-910115 (11)   h_gen=0.73, sigma fixed
#   csge   seeds 910101-910115 (15)   h_gen=0.73, sigma empirical
#   csgdm  seeds 910101-910108 (8)    h_gen=0.68, sigma fixed
#   csgdp  seeds 910101-910108 (8)    h_gen=0.78, sigma fixed
# Total: 11 + 15 + 8 + 8 = 42 tasks -> --array=0-41
#
# Walltime/idempotency conventions identical to csg_pilot.sbatch.
# Required env: OUT_DIR (same directory as the pilot's, so the scorer sees all
# 46 JSONs in one place).

#SBATCH --job-name=csg_fleet
#SBATCH --partition=cpu,cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-41

set -euo pipefail

TASK_ARMS=()
TASK_SEEDS=()
for s in $(seq 910105 910115); do TASK_ARMS+=("csgf");  TASK_SEEDS+=("$s"); done  # 11
for s in $(seq 910101 910115); do TASK_ARMS+=("csge");  TASK_SEEDS+=("$s"); done  # 15
for s in $(seq 910101 910108); do TASK_ARMS+=("csgdm"); TASK_SEEDS+=("$s"); done  # 8
for s in $(seq 910101 910108); do TASK_ARMS+=("csgdp"); TASK_SEEDS+=("$s"); done  # 8

N_TASKS=${#TASK_ARMS[@]}
if [[ "$N_TASKS" -ne 42 ]]; then
    echo "ERROR: task-list arithmetic drift -- built $N_TASKS tasks, expected 42." >&2
    exit 1
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set (submit with --array=0-41)." >&2
    exit 1
fi
ARM="${TASK_ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${TASK_SEEDS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

: "${OUT_DIR:?OUT_DIR not set}"
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
