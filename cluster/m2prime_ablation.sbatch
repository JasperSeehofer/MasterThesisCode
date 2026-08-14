#!/usr/bin/env bash
# cluster/m2prime_ablation.sbatch — stage-2 estimator-variant arms A-M2'/A-NULL
# Prereg: results/mechanism_study_20260813/PREREGISTRATION_M2PRIME_ABLATION.md
# Arms:   results/mechanism_study_20260813/ARMS.md ("Stage-2 arms" section)
#
# Array task -> arm map (worker geometry mirrors MN0X's single-job pool-over-
# seeds pattern, cluster/mechanism_null_n100.sbatch — SLURM_ARRAY_TASK_ID
# selects the ARM, never the seed; seeds are pinned inside the instrument by
# the M2P_CELL_SPECS registry):
#   0: AM2P   A-M2'    estimator_variant=m2prime_jacobian   25 seeds  base+53000..53024
#   1: ANULL  A-NULL   estimator_variant=null_scale_1p7     15 seeds  base+50000..50014
#
# ANULL's seed block is a DELIBERATE PAIRING with MN0X's first 15 seeds
# (prereg §3 DS-N1: exact per-seed MAP grid equality against MN0X's stored
# records) — not an error, not re-derived seeds.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight
# first, require VERDICT: READY, probe with sbatch --test-only):
#   mkdir -p results/mechanism_study_20260813/logs
#   sbatch --array=0-1 --partition=cpu_il --cpus-per-task=15 --time=05:00:00 \
#     --output="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%a.out" \
#     --error="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%a.err" \
#     cluster/m2prime_ablation.sbatch
#
# Budget (realized 0.969 CPU-h/seed anchor, cluster/mechanism_null_n100.sbatch
# / runbook §4, measured on the BASE estimator):
#   AM2P: 25 seeds x 0.969 CPU-h/seed = 24.2 CPU-h (matches prereg §3's own
#   "A-M2' ~= 24 CPU-h" estimate) => 1.62 h uncontended at 15 workers. The
#   Jacobian variant does TWO EXTRA dist_vectorized calls per kernel-branch
#   quadrature node (central difference, prereg-registered eps=1e-6) on top
#   of the one the base path already makes — a real per-seed cost increase
#   the 0.969 anchor does NOT capture, since that anchor was measured on the
#   base estimator (MN0/MN0X). Budgeted at 1.5x on top of the anchor for this
#   extra work: 1.62 h x 1.5 ~= 2.43 h. Per the venue-transfer prereg §11
#   addendum-2 lesson, margins are sized against CONTENDED timing, not
#   uncontended (packed tasks ran 1.6-1.9x slower): 2.43 h x 1.9 ~= 4.6 h.
#   ANULL: 15 seeds x 0.969 CPU-h/seed = 14.5 CPU-h; the x1.7 scalar multiply
#   is negligible extra cost (no extra dist_vectorized calls) => 0.97 h
#   uncontended at 15 workers, x1.9 contended ~= 1.84 h, plus fixed per-job
#   catalogue/pin-loading overhead the seed count does not scale with.
#   A single --time is requested for the whole array (the light ANULL task
#   accepts headroom it does not need, matching cluster/mechanism_isolation.sbatch's
#   convention for its light MEH arm): 05:00:00 covers AM2P's ~4.6 h estimate
#   with margin and is generous for ANULL's ~1.84-2.5 h estimate.
#
# Per EXP-61 (twice confirmed): `sbatch --test-only` ignores backfill and its
# predicted start time is wrong by orders of magnitude for short wide jobs.
# Still run and log the probe as a probe-vs-actual pair for the record, but
# do not believe the predicted start time.
#
# --grain seed deliberately (the campaign's/MN0X's mode), matching
# cluster/mechanism_isolation.sbatch's rationale: keeping the execution path
# identical to the arm being compared against (MN0X for ANULL) removes a
# disclosure item from the readout for no benefit, even though --grain h is
# certified byte-identical.

#SBATCH --job-name=m2p-ablation
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-15}"
TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-1)}"
OUT_DIR="results/mechanism_study_20260813"

case "$TID" in
  0) CELL=AM2P; SEED_RANGE="0:25" ;;
  1) CELL=ANULL; SEED_RANGE="0:15" ;;
  *) echo "unknown array task $TID" >&2; exit 64 ;;
esac

N=$(echo "$SEED_RANGE" | cut -d: -f2)
OUT_JSON="$OUT_DIR/${CELL}_h0p730_results_seeds0_${N}.json"

if [ -e "$OUT_JSON" ]; then
  echo "abort: $OUT_JSON already exists -- refusing to overwrite a registered result." >&2
  echo "Move or remove it first if a rerun is genuinely intended." >&2
  exit 65
fi

echo "m2p-ablation task=$TID cell=$CELL truth=0.730 seeds=$SEED_RANGE workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
python -m darksiren_emri.validation.venue_transfer \
  --cell "$CELL" --truth 0.730 --seed-range "$SEED_RANGE" --workers "$WORKERS" \
  --out "$OUT_JSON"
