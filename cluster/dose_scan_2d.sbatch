#!/usr/bin/env bash
# cluster/dose_scan_2d.sbatch — mechanism-study 2-D dose scan (host x impostor)
# Prereg: results/mechanism_study_20260813/PREREGISTRATION_MECHANISM_ISOLATION.md
# Arms:   results/mechanism_study_20260813/ARMS.md
#
# Array task -> cell map (16 tasks, 4x4 grid over host-dose index h and
# impostor-dose index i, each in {0,1,2,3} => dose fractions of sigma_z
# {0.0, 0.25, 0.5, 1.0}). Cell name S{h}{i}, task = 4*h + i, computed
# ARITHMETICALLY below (not a 16-branch case statement):
#   task  0: S00   task  4: S10   task  8: S20   task 12: S30
#   task  1: S01   task  5: S11   task  9: S21   task 13: S31
#   task  2: S02   task  6: S12   task 10: S22   task 14: S32
#   task  3: S03   task  7: S13   task 11: S23*  task 15: S33
# * S23 is the sole cell that can discriminate the two competing hypotheses
#   under study and is registered (darksiren_emri/validation/venue_transfer.py
#   SCAN_HIGH_N_CELL/SCAN_HIGH_N) at N=100 seeds instead of the uniform N=15
#   — it gets the statistics. It is therefore SPLIT OUT of the 16-task array
#   into its own single-task submission with its own wall budget (see below).
#   The per-task seed count is derived from the cell name at runtime (never
#   hardcoded), so this one script serves both submissions correctly.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight
# first, require VERDICT: READY, probe with sbatch --test-only):
#
#   mkdir -p results/mechanism_study_20260813/logs
#
#   # (1) the fifteen N=15 cells — everything except task 11 (S23):
#   sbatch --array=0-10,12-15 --partition=cpu_il --cpus-per-task=15 --time=02:00:00 \
#     --output="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%a.out" \
#     --error="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%a.err" \
#     cluster/dose_scan_2d.sbatch
#
#   # (2) S23 alone (task 11), N=100, its own wider wall:
#   sbatch --array=11-11 --partition=cpu_il --cpus-per-task=15 --time=13:00:00 \
#     --output="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%a.out" \
#     --error="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%a.err" \
#     cluster/dose_scan_2d.sbatch
#
# Budget: i=0 cells (impostor dose fraction 0) take the sigma=0
# point-evaluation branch and are near-free (per array 6303086's MEH arm: 15
# seeds in 00:02:20 at --cpus-per-task=15). Cells with i>0 pay the full
# quadrature path at the measured ~0.97 CPU-h/seed (array 6303086's MN0/MEI
# arms, ~4x faster than the venue-transfer 3.79 CPU-h/seed anchor, plausibly
# the ratified Route 1 adaptive Gauss-Hermite change) => ~15 CPU-h for a
# 15-seed cell.
#
# N=15 cells: --time is sized for the WORST such cell (S33-like, h=i=3, both
# fully dosed): uncontended wall = 15 seeds x 0.97 CPU-h/seed / 15 workers
# ~= 0.97 h. Per the venue-transfer prereg §11 addendum-2 lesson, margins are
# sized against CONTENDED timing, not uncontended (packed tasks ran 1.6-1.9x
# slower, and submission (1) packs 15 tasks at once): 0.97 h x 1.9 ~= 1.84 h
# => 2 h requested, applied uniformly across all 15 tasks in submission (1)
# (the light i=0 cells finish in minutes and cost nothing extra against the
# same budget).
#
# S23 (N=100, submission (2) alone): uncontended wall = 100 seeds x 0.969
# CPU-h/seed / 15 workers ~= 6.46 h. Contended margin (submission (2) is a
# single task, but the cluster-wide contended-timing lesson is applied
# regardless — it is not self-contention, it is queue-mate contention):
# 6.46 h x 1.9 ~= 12.27 h => 13 h requested.
#
# Per EXP-61 (twice confirmed): `sbatch --test-only` ignores backfill and its
# predicted start time is wrong by orders of magnitude for short wide jobs (it
# predicted a 4-days-out start for a job that started in ~30 min). Still run
# and log the probe as a probe-vs-actual pair for the record, but do not
# believe the predicted start time.

#SBATCH --job-name=dose-scan-2d
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-15}"
TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-10,12-15 and, separately, --array=11-11)}"
OUT_DIR="results/mechanism_study_20260813"

if (( TID < 0 || TID > 15 )); then
  echo "unknown array task $TID" >&2
  exit 64
fi

H=$((TID / 4))
I=$((TID % 4))
CELL="S${H}${I}"

# Per-task seed count derived from the cell name (S23 -> 100, else 15), so
# this one script serves both the 15-cell array and the S23-only submission.
if [[ "$CELL" == "S23" ]]; then
  N_SEEDS=100
else
  N_SEEDS=15
fi

echo "dose-scan-2d task=$TID cell=$CELL truth=0.730 seeds=0:${N_SEEDS} workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
python -m darksiren_emri.validation.venue_transfer \
  --cell "$CELL" --truth 0.730 --seed-range "0:${N_SEEDS}" --workers "$WORKERS" \
  --out "$OUT_DIR/${CELL}_h0p730_results_seeds0_${N_SEEDS}.json"
