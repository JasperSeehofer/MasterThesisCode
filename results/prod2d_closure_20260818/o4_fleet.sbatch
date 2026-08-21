#!/usr/bin/env bash
# results/prod2d_closure_20260818/o4_fleet.sbatch -- O4 pairing-test shards on
# the cluster (recovery from the local OOM: each Arm-P evaluate() needs ~9GB;
# the 30GB dev box cannot run them wide). One seed per array task, scorer
# committed at bfe4d09c, per-seed shard JSONs merged locally by
# o4_merge_shards.py.
#
# Required env: OUT_DIR (shard JSONs + logs), e.g.
#   $(ws_find emri)/o4_shards_20260821
#SBATCH --job-name=o4_shards
#SBATCH --partition=cpu,cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-10

set -euo pipefail

SEEDS=(910104 910105 910106 910107 910108 910109 910110 910111 910112 910113 910115)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

: "${OUT_DIR:?OUT_DIR not set}"
mkdir -p "$OUT_DIR"
WORK_ROOT="$OUT_DIR/work"
mkdir -p "$WORK_ROOT"

echo "=== o4 shard seed=$SEED job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID node=$(hostname) ==="
cd "$PROJECT_ROOT"
python results/prod2d_closure_20260818/o4_pairing_test.py \
    --seeds "$SEED" \
    --out "$OUT_DIR/shard_${SEED}.json" \
    --work-root "$WORK_ROOT"
echo "=== done seed=$SEED ==="
