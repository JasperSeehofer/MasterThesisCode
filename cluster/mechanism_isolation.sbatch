#!/usr/bin/env bash
# cluster/mechanism_isolation.sbatch — mechanism-isolation arm E1 (split dose)
# Prereg: results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md
# Arms:   results/mechanism_study_20260813/ARMS.md
#
# Array task -> arm map (15 seeds each; seeds pinned INSIDE the instrument by
# the cell registry, so SLURM_ARRAY_TASK_ID selects the ARM, never the seed):
#   0: MN0  null      dose_target=all         seeds 20310808..20310822
#   1: MEH  E1-host   dose_target=host        seeds 20310908..20310922
#   2: MEI  E1-imp    dose_target=impostors   seeds 20311008..20311022
#
# Registered prediction (prereg DS-M5): MEI bias >= 0.030 AND MEH bias <= 0.012
# confirms M5'; MN0 must reproduce the campaign's +0.037 +- 0.002 or the study
# is STUDY-CONFOUNDED and nothing else may be read.
#
# Submit (from ~/MasterThesisCode on the cluster; consult /cluster + preflight
# first, require VERDICT: READY, probe with sbatch --test-only):
#   mkdir -p results/mechanism_study_20260813/logs
#   sbatch --array=0-2 --partition=cpu_il --cpus-per-task=15 --time=12:00:00 \
#     --output="$HOME/MasterThesisCode/results/mechanism_study_20260813/logs/%x_%a.out" \
#     --error="$HOME/MasterThesisCode/results/mechanism_study_20260813/logs/%x_%a.err" \
#     cluster/mechanism_isolation.sbatch
#
# Budget: MN0/MEI are heavy (all impostors dosed => the sigma_z>0 quadrature
# path) at the campaign's measured 3.79 CPU-h/seed => ~57 CPU-h/arm, ~3.8 h
# wall at 15 workers UNCONTENDED. Per the venue-transfer §11 addendum-2 lesson,
# margins are computed against CONTENDED timing (packed tasks ran 1.6-1.9x
# slower): 3.8 h x 1.9 ~ 7.2 h => 12 h requested. MEH is light: only the 982
# hosts carry a kernel, the ~1.19M impostors take the sigma=0 point-evaluation
# branch, so expect well under an hour.
#
# --grain seed deliberately (the campaign's mode) so the N-0 arm is compared to
# the campaign under the identical execution path. Do NOT switch to --grain h
# for these three arms even though it is certified byte-identical: keeping the
# path identical removes a disclosure item from the readout for no benefit.

#SBATCH --job-name=mech-isolation
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-15}"
TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-2)}"
OUT_DIR="results/mechanism_study_20260813"

case "$TID" in
  0) CELL=MN0 ;;
  1) CELL=MEH ;;
  2) CELL=MEI ;;
  *) echo "unknown array task $TID" >&2; exit 64 ;;
esac

echo "mech-isolation task=$TID cell=$CELL truth=0.730 seeds=0:15 workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
python -m darksiren_emri.validation.venue_transfer \
  --cell "$CELL" --truth 0.730 --seed-range "0:15" --workers "$WORKERS" \
  --out "$OUT_DIR/${CELL}_h0p730_results_seeds0_15.json"
