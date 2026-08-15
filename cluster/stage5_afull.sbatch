#!/usr/bin/env bash
# cluster/stage5_afull.sbatch — stage-5 estimator-variant arm A-FULL (FULL-F)
# Prereg: results/mechanism_study_20260813/PREREGISTRATION_A_FULL_STAGE5.md
# Ledger: row #110 ("all approved": FULL-F definition, registration + run, bands
# seeded from the mirror pre-measurement).
#
# The arm: cell AFULL, estimator_variant="a_full" (d_obs-density GW factor x
# selected-population prior w_pop*S_phi (alpha retained as its normalization) x
# leave-one-out impostor weight; NO Jacobian, NO kernel renormalization),
# h_true=0.730, N=25 fresh seeds (offset block +54200...+54224), full dose.
#
# Pre-submission gates (prereg §6): (1) the installed instrument variant was
# cross-checked bit-exactly against the pre-measurement mirror on seed
# 20310808 (|dln1| = 0.0 at both h-neighbours, paired tilt +29.998 vs pin
# +30.0); (2) score_stage5.py dry-run on the committed AJREN JSON reproduced
# its three references; (3) preflight VERDICT: READY required at submission.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight
# first, require VERDICT: READY, probe with sbatch --test-only):
#   mkdir -p results/mechanism_study_20260813/logs
#   sbatch --partition=cpu_il --cpus-per-task=15 --time=05:00:00 \
#     --output="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%j.out" \
#     --error="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%j.err" \
#     cluster/stage5_afull.sbatch
#
# Budget (0.969 CPU-h/seed base anchor; stage3_ajren.sbatch's margin reasoning
# adapted): 25 seeds x 0.969 = 24.2 CPU-h => 1.62 h uncontended at 15 workers.
# a_full's cost delta over base: the density-form GW factor is one extra
# elementwise divide, w_pop + S_phi are one closed-form eval + one np.interp
# per node (comparable to AJREN's norm.cdf additions, cheaper than AM2P's two
# extra dist_vectorized calls), and the LOO weight is an h-independent
# construction recomputed per h-point (~1-2 s x 41 points ~ 1 min/seed, a
# known, accepted inefficiency). Budget 1.5x => 2.43 h; contended x1.9
# (venue-transfer prereg §11 addendum-2 lesson) => 4.6 h. --time=05:00:00
# covers with margin, matching the AJREN/AM2P precedent budgets.
#
# EXP-61 (twice confirmed): sbatch --test-only ignores backfill; log the probe
# as probe-vs-actual for the record, do not believe its predicted start.
#
# --grain seed deliberately, matching MN0X/AM2P/AJREN (identical execution
# path to the arms this one is compared against).

#SBATCH --job-name=stage5-afull
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-15}"
OUT_DIR="results/mechanism_study_20260813"

CELL=AFULL
SEED_RANGE="0:25"
N=$(echo "$SEED_RANGE" | cut -d: -f2)
OUT_JSON="$OUT_DIR/${CELL}_h0p730_results_seeds0_${N}.json"

if [ -e "$OUT_JSON" ]; then
  echo "abort: $OUT_JSON already exists -- refusing to overwrite a registered result." >&2
  echo "Move or remove it first if a rerun is genuinely intended." >&2
  exit 65
fi

echo "stage5-afull cell=$CELL truth=0.730 seeds=$SEED_RANGE workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
python -m darksiren_emri.validation.venue_transfer \
  --cell "$CELL" --truth 0.730 --seed-range "$SEED_RANGE" --workers "$WORKERS" \
  --out "$OUT_JSON"
