#!/bin/bash
#SBATCH --job-name=prod2d_closure
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --output=prod2d_closure_%j.out
#SBATCH --error=prod2d_closure_%j.err

# Production-2D closure + catalog-quality landscape (prereg
# PREREGISTRATION_PROD2D_CLOSURE_LANDSCAPE.md Sec 3, T1+T2, cluster job;
# verifier Part VII P7-1 amendment: 18 registered cells, applied verbatim).
# One node, 18 workers = one worker per registered cell (cluster-skill
# gotcha 7: match the reservation to the actual work count). Pure-CPU numpy
# harness -- no GPU, no galaxy catalogue download. Seeds live inside
# run_prod2d.py (venue-shared master seeds 20280411/20280511); no SLURM
# seed derivation on purpose: the registered scheme must not be layered
# over.
#
# Walltime: Sec 2 sizing estimate (P7-6) ~74-79 CPU-h total across 18 cells
# / 18 workers ~= 5-6h wall; --time=14:00:00 gives >= 2x margin against the
# contended-anchor cluster gotcha (6/9). Idempotent: run_prod2d.py skips
# cells whose JSON already exists, so a resubmit resumes.
#
# RUN_DIR must be exported by the submit wrapper (workspace run dir, e.g.
# $(ws_find emri)/run_prod2d_20260818); this script does not invent one.

set -euo pipefail
cd "$HOME/darksiren-emri"
source cluster/modules.sh

: "${RUN_DIR:?RUN_DIR not set -- export it (e.g. RUN_DIR=$(ws_find emri)/run_prod2d_20260818)}"
mkdir -p "$RUN_DIR"

echo "=== $SLURM_JOB_NAME  job=$SLURM_JOB_ID  node=$(hostname)  RUN_DIR=$RUN_DIR ==="

cd results/prod2d_closure_20260818
RUN_DIR="$RUN_DIR" ../../.venv/bin/python -u run_prod2d.py

echo "=== done: job=$SLURM_JOB_ID ==="
