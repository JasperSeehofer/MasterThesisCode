#!/usr/bin/env bash
# cluster/stage3_ajren.sbatch — stage-3 estimator-variant arm A-JREN (primary)
# Prereg: results/mechanism_study_20260813/PREREGISTRATION_A_JREN_STAGE3.md
# Arms:   results/mechanism_study_20260813/ARMS.md ("Stage-3 arms" section)
#
# Per the registration finalization block F1 (A-JREN's trigger fired,
# L0-REN-B read R3 = BUDGET-TENSION): A-JREN is this stage's PRIMARY arm,
# submitted first and unconditionally. A-REN converts to CONDITIONAL — it
# runs only after a post-readout author [RULE] (F1: "runs only if A-JREN's
# readout leaves single-term attribution needed... a post-readout author
# [RULE]; otherwise A-REN is withdrawn by that same [RULE], never silently").
# Its stanza below is present (registered code form + seed block are fixed
# at registration regardless of submission) but commented out and marked
# CONDITIONAL — do not enable without that [RULE].
#
# Single task (no array): only AJREN is wired for unattended submission.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight
# first, require VERDICT: READY, probe with sbatch --test-only):
#   mkdir -p results/mechanism_study_20260813/logs
#   sbatch --partition=cpu_il --cpus-per-task=15 --time=05:00:00 \
#     --output="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%j.out" \
#     --error="$HOME/darksiren-emri/results/mechanism_study_20260813/logs/%x_%j.err" \
#     cluster/stage3_ajren.sbatch
#
# Budget (realized 0.969 CPU-h/seed anchor, cluster/mechanism_null_n100.sbatch
# / runbook §4, measured on the BASE estimator; cluster/m2prime_ablation.sbatch's
# AM2P margin reasoning carried verbatim — the Jacobian cost is identical, and
# A-JREN adds TWO norm.cdf evaluations per kernel-branch quadrature node on
# top of that, a smaller relative addition than the Jacobian's two extra
# dist_vectorized calls):
#   AJREN: 25 seeds x 0.969 CPU-h/seed = 24.2 CPU-h => 1.62 h uncontended at
#   15 workers. Budgeted at 1.5x on top of the anchor for the Jacobian's two
#   extra dist_vectorized calls PLUS the renormalization's two norm.cdf calls
#   per candidate (the latter is O(n_rows), not O(n_rows * n_quad) like the
#   Jacobian, so it is a small addition on top of AM2P's own 1.5x budget, not
#   a second independent 1.5x): 1.62 h x 1.5 ~= 2.43 h. Per the venue-transfer
#   prereg §11 addendum-2 lesson, margins are sized against CONTENDED timing,
#   not uncontended (packed tasks ran 1.6-1.9x slower): 2.43 h x 1.9 ~= 4.6 h.
#   --time=05:00:00 covers this estimate with margin, matching AM2P's own
#   05:00:00 budget in cluster/m2prime_ablation.sbatch (same order of
#   estimator-cost increase over the base anchor).
#
# Per EXP-61 (twice confirmed): `sbatch --test-only` ignores backfill and its
# predicted start time is wrong by orders of magnitude for short wide jobs.
# Still run and log the probe as a probe-vs-actual pair for the record, but
# do not believe the predicted start time.
#
# --grain seed deliberately (the campaign's/MN0X's mode), matching
# cluster/m2prime_ablation.sbatch's rationale: keeping the execution path
# identical to the arm being compared against (MN0X / AM2P) removes a
# disclosure item from the readout for no benefit, even though --grain h is
# certified byte-identical.

#SBATCH --job-name=stage3-ajren
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

WORKERS="${SLURM_CPUS_PER_TASK:-15}"
OUT_DIR="results/mechanism_study_20260813"

CELL=AJREN
SEED_RANGE="0:25"
N=$(echo "$SEED_RANGE" | cut -d: -f2)
OUT_JSON="$OUT_DIR/${CELL}_h0p730_results_seeds0_${N}.json"

if [ -e "$OUT_JSON" ]; then
  echo "abort: $OUT_JSON already exists -- refusing to overwrite a registered result." >&2
  echo "Move or remove it first if a rerun is genuinely intended." >&2
  exit 65
fi

echo "stage3-ajren cell=$CELL truth=0.730 seeds=$SEED_RANGE workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
python -m darksiren_emri.validation.venue_transfer \
  --cell "$CELL" --truth 0.730 --seed-range "$SEED_RANGE" --workers "$WORKERS" \
  --out "$OUT_JSON"

# ── CONDITIONAL: A-REN (results/mechanism_study_20260813/PREREGISTRATION_A_JREN_STAGE3.md
# F1) — DO NOT ENABLE without a post-readout author [RULE] confirming A-JREN's
# readout leaves single-term attribution needed (TERM-PARTIAL or OTHER and the
# interaction decomposition requires the single-REN datum). Uncomment and
# submit as its OWN separate job only after that [RULE] is granted; it is
# NOT part of this script's unattended submission by design.
#
# CELL=AREN
# SEED_RANGE="0:25"
# N=$(echo "$SEED_RANGE" | cut -d: -f2)
# OUT_JSON="$OUT_DIR/${CELL}_h0p730_results_seeds0_${N}.json"
# if [ -e "$OUT_JSON" ]; then
#   echo "abort: $OUT_JSON already exists -- refusing to overwrite a registered result." >&2
#   exit 65
# fi
# echo "stage3-aren (CONDITIONAL) cell=$CELL truth=0.730 seeds=$SEED_RANGE workers=$WORKERS host=$(hostname) commit=$(git rev-parse HEAD)"
# python -m darksiren_emri.validation.venue_transfer \
#   --cell "$CELL" --truth 0.730 --seed-range "$SEED_RANGE" --workers "$WORKERS" \
#   --out "$OUT_JSON"
