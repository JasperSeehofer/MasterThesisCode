#!/usr/bin/env bash
# cluster/graph1_s0b_truth_hbounds060.sbatch — Research Graph 1, Branch D
# forensics follow-up: re-run the S0-B TRUTH node only, under
# --h-bounds 0.60,0.86 (production's class default), to test
# DRIVER_VENUE_FORENSICS.md's decisive finding that hier_s0_driver.py's
# module-constant H_BOUNDS=(0.50, 0.86) — not production's (0.60, 0.86) —
# is the sole discriminator behind the driver/production candidate-window
# mismatch (results/campaign51_20260728/realistic_20260729/graph1_20260901/
# exec/d-photoz-leverage/DRIVER_VENUE_FORENSICS.md section 5, "discriminating
# single-job test").
#
# Exact copy of cluster/graph1_m_s0b_production.sbatch's driver invocation
# (node selection, resources, ancestor pin pattern), restricted to the truth
# node only (NODES=(truth), --array=0-0 — the production sbatch's own
# per-array-task NODES[$SLURM_ARRAY_TASK_ID] node-selection mechanism, here
# with a single-element array instead of running all 5 nodes) and with
# --h-bounds 0.60,0.86 appended. exec/d-photoz-leverage/
# HBOUNDS_FLAG_BUILD_RECORD.md documents the flag's build (byte-identical
# default, zero-behaviour-change proof) this sbatch depends on.
#
# NOT SUBMITTED by this build task — builder scope only (no cluster access).
# d9e50179 below is a placeholder: the ops agent submitting this job must
# replace it with the short commit hash of the commit that lands the
# --h-bounds flag (this sbatch's own commit or a descendant), per the
# ancestor-check pattern already used by graph1_m_s0b_production.sbatch
# (row #331 fix — ancestor check, not strict equality, since docs-only
# commits land constantly on this branch).
#
# Required env (set by the submitter):
#   PROJECT_ROOT -- optional; falls back to $HOME/darksiren-emri

#SBATCH --job-name=graph1-s0b-hb060
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --array=0-0

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

EXPECTED_COMMIT="d9e50179"  # placeholder -- ops agent fills in after the --h-bounds commit lands (ancestor check, not strict equality; see header)
ACTUAL_COMMIT="$(git rev-parse --short=8 HEAD)"
if ! git merge-base --is-ancestor "$EXPECTED_COMMIT" HEAD; then
    echo "STOP: HEAD ($ACTUAL_COMMIT) does not contain required commit $EXPECTED_COMMIT -- re-sync before running" >&2
    exit 1
fi

NODES=(truth)
NODE="${NODES[$SLURM_ARRAY_TASK_ID]}"

OUT_ROOT="$PROJECT_ROOT/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/d-photoz-leverage/graph1_s0b_truth_hbounds060_20260904"
mkdir -p "$OUT_ROOT/logs"

LOG_BASE="$OUT_ROOT/logs/s0b_truth_hbounds060_task_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-s0b-hb060 job=${SLURM_JOB_ID:-NA} array_task=${SLURM_ARRAY_TASK_ID} node=$NODE host=$(hostname) commit=$ACTUAL_COMMIT ==="

source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$OUT_ROOT" "graph1 s0b-hb060 (d-photoz-leverage forensics follow-up): registered S0-B truth node only, iiib/CoR-P, theta_sites=2.2, smear=off, h-bounds=0.60,0.86 (production default, vs driver's as-built 0.50,0.86), seed=900101, h=0.73"

python "$PROJECT_ROOT/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py" \
    --arm S0-A \
    --seeds 900101 \
    --nodes "$NODE" \
    --jobs 1 \
    --out-root "$OUT_ROOT" \
    --theta-sites 2.2 \
    --smear off \
    --config iiib \
    --b-half-width 0.033 \
    --h-bounds 0.60,0.86

echo "=== done: s0b-hb060 node=$NODE ==="
