#!/usr/bin/env bash
# cluster/mechanism_null_n100.sbatch — mechanism-study null replication at N=100
# Prereg: results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md
# Arms:   results/mechanism_study_20260813/ARMS.md
#
# Single-job (non-array) run of cell MN0X (the N=100 extension of the split-dose
# null arm MN0) at truth 0.730, seed-range 0:100, dose_target=all (fully-dosed
# path, same as MN0 in cluster/mechanism_isolation.sbatch).
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight
# first, require VERDICT: READY, probe with sbatch --test-only):
#   mkdir -p results/mechanism_study_20260813/logs
#   sbatch --partition=cpu_il --cpus-per-task=15 --time=13:00:00 \
#     --output="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%j.out" \
#     --error="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%j.err" \
#     cluster/mechanism_null_n100.sbatch
#
# Budget: array 6303086's split-dose arms measured MN0 (fully-dosed, all
# candidates on the sigma_z>0 quadrature path) at 15 seeds in 00:58:08 wall at
# --cpus-per-task=15 on cpu_il => ~0.97 CPU-h/seed (~4x faster than the
# venue-transfer campaign's 3.79 CPU-h/seed anchor, plausibly the ratified
# Route 1 adaptive Gauss-Hermite change). At N=100 seeds that is ~97 CPU-h
# total. We deliberately keep --cpus-per-task=15, matching the measured
# configuration exactly rather than extrapolating pool-over-seeds scaling to
# an unmeasured worker count: uncontended wall = 100 seeds x 0.97 CPU-h/seed /
# 15 workers ~= 6.47 h. Per the venue-transfer prereg §11 addendum-2 lesson,
# margins are sized against CONTENDED timing, not uncontended (packed tasks
# ran 1.6-1.9x slower): 6.47 h x 1.9 ~= 12.3 h => 13 h requested.
#
# Per EXP-61 (twice confirmed): `sbatch --test-only` ignores backfill and its
# predicted start time is wrong by orders of magnitude for short wide jobs (it
# predicted a 4-days-out start for a job that started in ~30 min). Still run
# and log the probe as a probe-vs-actual pair for the record, but do not
# believe the predicted start time.

#SBATCH --job-name=mech-null-n100
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-15}"
CELL=MN0X
OUT_DIR="results/mechanism_study_20260813"

echo "mech-null-n100 cell=$CELL truth=0.730 seeds=0:100 workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
python -m darksiren_emri.validation.venue_transfer \
  --cell "$CELL" --truth 0.730 --seed-range "0:100" --workers "$WORKERS" \
  --out "$OUT_DIR/${CELL}_h0p730_results_seeds0_100.json"
