#!/usr/bin/env bash
# cluster/venue_transfer.sbatch — venue-transfer registered campaign
# (PREREGISTRATION_VENUE_TRANSFER.md, results/venue_transfer_20260811/) as a
# 49-task CPU array (prereg §5: heavy cells chunked <= 25 seeds/task, 64 workers).
#
# Array task -> registered job map (cell / truth / --seed-range START:COUNT):
#    0- 7: T0 0.730  chunks 0:25 .. 175:25   (200 seeds, sigma_z = 0 anchor)
#       8: Ta 0.730  0:200                    (Poisson lambda=4 arm, light)
#    9-16: Tb 0.730  chunks 0:25 .. 175:25   (real K, flat 0.035)
#   17-24: Tc 0.690  chunks 0:25 .. 175:25   (decision wings, N=200)
#   25-40: Tc 0.730  chunks 0:25 .. 375:25   (DECISION cell, N=400)
#   41-48: Tc 0.770  chunks 0:25 .. 175:25   (decision wings, N=200)
# Seeds are pinned INSIDE the instrument by the prereg seed plan (base 20260808 +
# v3 offsets +40000..+45399); SLURM_ARRAY_TASK_ID selects the job, never the seed.
# Reserved blocks +46000 (W1) / +47000 (O2) are NOT built and have no task here.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight first,
# require VERDICT: READY, probe with sbatch --test-only — prereg §5 venue rule):
#   sbatch --array=0-48 --partition=cpu_il --cpus-per-task=64 --time=04:00:00 \
#     --output="$HOME/darksiren-emri/results/venue_transfer_20260811/logs/%x_%a.out" \
#     --error="$HOME/darksiren-emri/results/venue_transfer_20260811/logs/%x_%a.err" \
#     cluster/venue_transfer.sbatch
# Prereg budget: heavy seed ~4.33 CPU-h derived => ~<=2 h wall/task predicted at 64
# workers (<=4 h requested). Smoke-measured per-seed CPU is the abort-(a) input —
# if >2x the derived anchor, apply the §5 N-floor fallback stages (author-gated).
#
# Each task writes results/venue_transfer_20260811/<cell>_h<truth>_results_seedsSTART_COUNT.json
# (per-chunk documents, combined at readout; every JSON embeds the clean-rule
# provenance + the recomputed V-T3 pin block).
#
# ── OPT-IN intra-seed h-grain mode (venue_transfer.py divergence 11) ──────────
# NOT the campaign default — do NOT edit the python line below or resubmit the
# running array; the registered campaign stays at --grain seed. For follow-on
# campaigns or straggler re-runs, --grain h forks each seed's 41 estimator
# h-points instead of pooling over seeds: per-seed wall time drops from hours
# to minutes at --workers 41, and per-seed records are BYTE-IDENTICAL to seed
# grain for any worker count (unit-tested cross-mode, incl. real-K capped-g
# multi-chunk cases). Seeds then run serially per task, so size the chunk
# accordingly (e.g. fewer seeds per task, more array tasks). Example:
#   python -m darksiren_emri.validation.venue_transfer \
#     --cell Tc --truth 0.730 --seed-range "$START:$COUNT" \
#     --workers 41 --grain h \
#     --out "results/<NEW_RUN_DIR>/Tc_h0p730_results_seeds${START}_${COUNT}.json"
# Keep the BLAS threading env (OMP_NUM_THREADS / OPENBLAS_NUM_THREADS, from
# modules.sh) identical to the seed-grain runs — both modes fork workers from
# the same parent env, and byte-for-byte comparability assumes the same
# kernels. Memory note: h-grain workers share the parent context copy-on-write
# but each materialises its own per-chunk intermediates (bounded by
# --chunk-pairs x 50 nodes, same as a seed-grain worker).

#SBATCH --job-name=venue-transfer
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-64}"
TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-48)}"

case "$TID" in
  [0-7])
    CELL=T0; TRUTH=0.730; START=$((TID * 25)); COUNT=25 ;;
  8)
    CELL=Ta; TRUTH=0.730; START=0; COUNT=200 ;;
  9|1[0-6])
    CELL=Tb; TRUTH=0.730; START=$(((TID - 9) * 25)); COUNT=25 ;;
  1[7-9]|2[0-4])
    CELL=Tc; TRUTH=0.690; START=$(((TID - 17) * 25)); COUNT=25 ;;
  2[5-9]|3[0-9]|40)
    CELL=Tc; TRUTH=0.730; START=$(((TID - 25) * 25)); COUNT=25 ;;
  4[1-8])
    CELL=Tc; TRUTH=0.770; START=$(((TID - 41) * 25)); COUNT=25 ;;
  *)
    echo "unknown array task $TID" >&2; exit 64 ;;
esac

echo "venue-transfer task=$TID cell=$CELL truth=$TRUTH seed-range=$START:$COUNT workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"

# Provenance stamp (cluster/SKILL.md gotcha #12): `python -m
# darksiren_emri.validation.venue_transfer` is a submodule invocation, not the
# package entry point `python -m darksiren_emri` — main.py's run_metadata
# writer never fires for it.
PROV_DIR="$PROJECT_ROOT/results/venue_transfer_20260811/logs"
mkdir -p "$PROV_DIR"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$PROV_DIR" "venue_transfer cell=$CELL truth=$TRUTH seed-range=$START:$COUNT"

python -m darksiren_emri.validation.venue_transfer \
  --cell "$CELL" --truth "$TRUTH" --seed-range "$START:$COUNT" --workers "$WORKERS"
