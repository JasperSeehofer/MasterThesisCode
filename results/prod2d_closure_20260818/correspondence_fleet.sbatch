#!/usr/bin/env bash
# results/prod2d_closure_20260818/correspondence_fleet.sbatch -- Option-B
# cluster fleet execution (task spec item 2).
#
# Runs one (arm, seed) task per array index via
# `darksiren_emri.validation.correspondence_1d --stage arm`
# (darksiren_emri/validation/correspondence_1d.py). Arm -> (sigma_z_scale,
# area_scale) mapping and per-arm seed lists are OWNED by the harness's
# ARM_SPECS/ARM_SEEDS registry -- this script's TASK_ARMS/TASK_SEEDS arrays
# are built from the SAME registered seed-list arithmetic
# (PREREGISTRATION_1D_CORRESPONDENCE.md v2 + AMENDMENT A-1 + AMENDMENT A-2)
# so the two cannot silently drift:
#   b0       (production-mapped, sigma_z=1.00, area=1.00): seeds 900101-900125 (N=25)
#   bsig005  (starvation dose,    sigma_z=0.05, area=1.00): seeds 900101-900125 (N=25)
#   bsig025  (starvation dose,    sigma_z=0.25, area=1.00): seeds 900101-900110 (N=10)
#   eden05   (E-DEN, exploratory, sigma_z=1.00, area=0.50): seeds 900101-900110 (N=10)
#   eden2    (E-DEN, exploratory, sigma_z=1.00, area=2.00): seeds 900101-900110 (N=10)
#   bout     (AMENDMENT A-2, population-model host draw, out-of-catalogue,
#             ARM_HOST_MODE="population"):                  seeds 900101-900115 (N=15)
#   bf1      (AMENDMENT A-2, B-0 config + f=1 completeness control,
#             ARM_UNITY_COMPLETENESS=True):                 seeds 900101-900102 (N=2)
#   bsel     (AMENDMENT A-3, population x (1-completeness) x survival host
#             draw -- the true isolation test,
#             ARM_HOST_MODE="population_selected"):          seeds 900101-900115 (N=15)
#   bself    (AMENDMENT A-4, B-SEL configuration verbatim + first bisection
#             step -- selection_in_completion_numerator="fused" in place of
#             B-SEL's "off", ARM_SELECTION_CELL["bself"]="fused"):
#                                                              seeds 900101-900115 (N=15)
# Total: 25 + 25 + 10 + 10 + 10 + 15 + 2 + 15 + 15 = 127 tasks -> --array=0-126
# (80 pre-A-2 tasks + 17 AMENDMENT A-2 tasks + 15 AMENDMENT A-3 tasks + 15
# AMENDMENT A-4 tasks, matching the prereg's AMENDMENT A-3 "15 + 3 ~= 18
# seed-runs ~= 20 CPU-h" budget line -- the +3 TIMEOUT resubmits are handled
# by re-running this same idempotent script for the affected (arm, seed)
# pairs, not by a separate task-list entry -- and AMENDMENT A-4's "15 seeds
# ~= 11 CPU-h" budget line).
#
# Walltime: pilot anchor (G-2, prereg §1 D-D) = 0.478 CPU-h/seed-run (~29 min
# wall, single-process). --time=02:00:00 gives >4x margin per task (cluster-
# skill convention: generous per-task margin on an array job, since a single
# slow/contended node should not blow the whole fleet).
#
# Idempotent (walltime-kill safety): the harness's run_arm_seed() skips
# entirely (no catalogue build, no evaluate() call) if
# $OUT_DIR/<arm>_seed<seed>.json already exists -- a resubmitted array picks
# up only the missing tasks.
#
# Validation-only: this script and correspondence_1d.py are the ONLY
# artifacts this task touches; bayesian_inference/* is untouched (D-A calls
# it wholesale, unmodified).
#
# Required environment variables (export at submit time):
#   OUT_DIR -- fleet output directory (one JSON per task; e.g.
#     $(ws_find emri)/correspondence_1d_fleet_<date>/out)
# Optional:
#   PROJECT_ROOT -- repo root; falls back to ~/darksiren-emri (cluster/
#     modules.sh convention, matches cluster/evaluate.sbatch).
#   WORK_ROOT_BASE -- per-task scratch parent; falls back to
#     $OUT_DIR/../work.

#SBATCH --job-name=corr1d_fleet
#SBATCH --partition=cpu,cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-126

set -euo pipefail

# ---------------------------------------------------------------------------
# Flat (arm, seed) task list -- built from the SAME registered per-arm seed
# ranges as darksiren_emri/validation/correspondence_1d.py's ARM_SEEDS, in
# the SAME arm order, so TASK_ARMS[i]/TASK_SEEDS[i] and SLURM_ARRAY_TASK_ID=i
# always name the identical task the harness would build from ARM_SEEDS
# directly.
# ---------------------------------------------------------------------------

TASK_ARMS=()
TASK_SEEDS=()

for s in $(seq 900101 900125); do TASK_ARMS+=("b0");      TASK_SEEDS+=("$s"); done  # 25
for s in $(seq 900101 900125); do TASK_ARMS+=("bsig005"); TASK_SEEDS+=("$s"); done  # 25
for s in $(seq 900101 900110); do TASK_ARMS+=("bsig025"); TASK_SEEDS+=("$s"); done  # 10
for s in $(seq 900101 900110); do TASK_ARMS+=("eden05");  TASK_SEEDS+=("$s"); done  # 10
for s in $(seq 900101 900110); do TASK_ARMS+=("eden2");   TASK_SEEDS+=("$s"); done  # 10
for s in $(seq 900101 900115); do TASK_ARMS+=("bout");    TASK_SEEDS+=("$s"); done  # 15 (AMENDMENT A-2)
for s in $(seq 900101 900102); do TASK_ARMS+=("bf1");     TASK_SEEDS+=("$s"); done  # 2  (AMENDMENT A-2)
for s in $(seq 900101 900115); do TASK_ARMS+=("bsel");    TASK_SEEDS+=("$s"); done  # 15 (AMENDMENT A-3)
for s in $(seq 900101 900115); do TASK_ARMS+=("bself");   TASK_SEEDS+=("$s"); done  # 15 (AMENDMENT A-4)

N_TASKS=${#TASK_ARMS[@]}
if [[ "$N_TASKS" -ne 127 ]]; then
    echo "ERROR: task-list arithmetic drift -- built $N_TASKS tasks, expected 127." >&2
    exit 1
fi

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is not set (submit with --array=0-126)." >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= N_TASKS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0, $((N_TASKS - 1))]." >&2
    exit 1
fi

ARM="${TASK_ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${TASK_SEEDS[$SLURM_ARRAY_TASK_ID]}"

# ---------------------------------------------------------------------------
# Environment setup (cluster/evaluate.sbatch header pattern).
# ---------------------------------------------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

: "${OUT_DIR:?OUT_DIR not set -- export it (e.g. OUT_DIR=\$(ws_find emri)/correspondence_1d_fleet_20260819/out)}"
mkdir -p "$OUT_DIR"

WORK_ROOT_BASE="${WORK_ROOT_BASE:-$OUT_DIR/../work}"
WORK_ROOT="$WORK_ROOT_BASE/${ARM}_seed${SEED}"
mkdir -p "$WORK_ROOT"

echo "=== $SLURM_JOB_NAME  job=$SLURM_JOB_ID  array_task=$SLURM_ARRAY_TASK_ID/$((N_TASKS - 1))  node=$(hostname) ==="
echo "  Arm:        $ARM"
echo "  Seed:       $SEED"
echo "  OUT_DIR:    $OUT_DIR"
echo "  WORK_ROOT:  $WORK_ROOT"
echo "  CPUs alloc: ${SLURM_CPUS_PER_TASK:-unknown}"

cd "$PROJECT_ROOT"
python -m darksiren_emri.validation.correspondence_1d \
    --stage arm \
    --arm "$ARM" \
    --seed "$SEED" \
    --work-root "$WORK_ROOT" \
    --out-dir "$OUT_DIR"

echo "=== done: arm=$ARM seed=$SEED job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID ==="
