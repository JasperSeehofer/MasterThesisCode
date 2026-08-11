#!/usr/bin/env bash
# cluster/calibration_gate_v2.sbatch — stage-4 calibration-gate v2 registered campaign
# (PREREGISTRATION_CALIBRATION_GATE_V2.md, commit 065e7f58) as a 10-task CPU array.
#
# Array task -> registered job map (matches the prereg §5 program exactly):
#   0: R0 retro-read (zero-compute)   1: A 0.690   2: A 0.730   3: A 0.770
#   4: B0 0.730   5: B1 0.730   6: B2 0.690   7: B2 0.730   8: B2 0.770
#   9: V1 0.730
# Seeds are pinned INSIDE the instrument by the prereg seed plan (base 20260808 +
# registered offsets); SLURM_ARRAY_TASK_ID selects the job, never the seed.
# Submit (from ~/MasterThesisCode on the cluster):
#   sbatch --array=0-9 --partition=dev_cpu_il --cpus-per-task=64 --time=00:25:00 \
#     --output="$HOME/MasterThesisCode/results/calibration_gate_v2_20260810/logs/%x_%a.out" \
#     --error="$HOME/MasterThesisCode/results/calibration_gate_v2_20260810/logs/%x_%a.err" \
#     cluster/calibration_gate_v2.sbatch
# Fallback partition for re-runs without the 30-min cap: cpu_il / cpu (128 cpus).

#SBATCH --job-name=calgate-v2
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-64}"
TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-9)}"

case "$TID" in
  0) ARGS=(--cell R0) ;;
  1) ARGS=(--cell A  --truth 0.690) ;;
  2) ARGS=(--cell A  --truth 0.730) ;;
  3) ARGS=(--cell A  --truth 0.770) ;;
  4) ARGS=(--cell B0 --truth 0.730) ;;
  5) ARGS=(--cell B1 --truth 0.730) ;;
  6) ARGS=(--cell B2 --truth 0.690) ;;
  7) ARGS=(--cell B2 --truth 0.730) ;;
  8) ARGS=(--cell B2 --truth 0.770) ;;
  9) ARGS=(--cell V1 --truth 0.730) ;;
  *) echo "unknown array task $TID" >&2; exit 64 ;;
esac

echo "calgate-v2 task=$TID args=${ARGS[*]} workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
python -m darksiren_emri.validation.calibration_gate "${ARGS[@]}" --workers "$WORKERS"
