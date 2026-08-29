#!/usr/bin/env bash
# cluster/wave2_c4_twin_mz_sel.sbatch — wave-2 charter node C4: PROD-CF-2D, the with-BH
# catalogue-leg twin `catalogue_numerator_survival_2d=mz_sel` (center=eff), arm T only
# (charter node B7.2). Registration:
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md
#   §6.2 (registered form) + §13.3 (appended, confirmed-unchanged final form)
#
# **Launched under rows #222/#223 — charter nodes C0/B5.2/B7.2/B1.2.**
#
# Baseline B (`catalogue_numerator_survival_2d=off`) is REUSED at zero compute from the banked
# HEAD readout, conditional on the shared C0 baseline-gate task PASSing (§6.2 "Code state", L5).
# This sbatch runs ONLY arm T over the H4 grid {0.660, 0.665, 0.670, 0.730}. Task index 0 is
# h=0.730 -- the STEP-2 smoke (§13.2) that pins the ASSUMED-not-measured 1.0-1.3x mz_sel
# overhead at production candidate counts (mirror-venue evidence was NOT slower at 200 events,
# p3_2d_rhs2.sbatch:15-16, but production candidate counts are ~1e3x larger). Per
# cluster/submit_wave2.sh, task 0 is submitted FIRST as a single-task job; tasks 1-3 are
# submitted as a SEPARATE, DEPENDENT array (--dependency=afterok) once task 0 completes, so the
# smoke's wall-time is observed before committing the other three tasks' walltime budget.
#
# WALLTIME RESUBMIT RULE (WAVE2_REGISTRATION_CHECK_20260829.md §1.5 "STEP-2 smoke" GAP-minor):
# if task 0 exceeds this script's --time (i.e. overhead > ~2.1x, not the assumed 1.0-1.3x), that
# is a WALLTIME RESUBMIT (resize --time and resubmit the array), NOT a materiality-band change --
# do not reinterpret an overrun as an H0 finding.
#
# F4 COST ESTIMATE: 4 H4 nodes x 14.93-20.27 CPU-h/node (instructed anchor, proposal §6.2) x
# 1.0-1.3 assumed mz_sel overhead = 59.7-105 CPU-h (ceiling at the 1.3x overhead upper bound;
# COMPUTE_LEDGER.md row C4 / proposal §6.2 table). --time=03:00:00 is ~2.1x the HEAD-config
# slowest observed iiib task (1:25:52), per the proposal's own sizing (§6.2).
#
# Required env (set by cluster/submit_wave2.sh via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260829_wave2_c4_iiib (out-root name fixed by
#                    results/_archive/archive_run_wave2.sh -- do not rename)
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=wave2-c4-twin
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
# NOTE: --array is set on the sbatch command line by submit_wave2.sh (0-0 for the STEP-2 smoke,
# then 1-3 as a dependent array) rather than hardcoded here, so the same script serves both
# submissions without editing #SBATCH pragmas between them.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set -- submit_wave2.sh must export it}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-0 then --array=1-3, see header)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/wave2_c4_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== wave2-c4-twin task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

CRB_SRC="$WORKSPACE/run_20260729_seed61000"
ln -sfn "$CRB_SRC/simulations/cramer_rao_bounds.csv"          "$RUN_DIR/simulations/cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/injections"                     "$RUN_DIR/simulations/injections"

EXPECTED_CRB_MD5="9a1f2a14384a9281c97ca3be312ddaab"
EXPECTED_CATALOGUE_MD5="c52c13b5cab61f6b3f04bbe202550969"
ACTUAL_CRB_MD5=$(md5sum "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv" | awk '{print $1}')
if [[ "$ACTUAL_CRB_MD5" != "$EXPECTED_CRB_MD5" ]]; then
    echo "STOP: prepared_cramer_rao_bounds.csv md5 mismatch (expected $EXPECTED_CRB_MD5, got $ACTUAL_CRB_MD5)" >&2
    exit 1
fi
ACTUAL_CATALOGUE_MD5=$(md5sum "$PROJECT_ROOT/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv" | awk '{print $1}')
if [[ "$ACTUAL_CATALOGUE_MD5" != "$EXPECTED_CATALOGUE_MD5" ]]; then
    echo "STOP: reduced_galaxy_catalogue.csv md5 mismatch (expected $EXPECTED_CATALOGUE_MD5, got $ACTUAL_CATALOGUE_MD5)" >&2
    exit 1
fi
echo "dataset pins OK: CRB=$ACTUAL_CRB_MD5 catalogue=$ACTUAL_CATALOGUE_MD5"

# ---------------------------------------------------------------------------
# H4 grid, task-0-first ordering so the STEP-2 smoke (h=0.730) is index 0 -- the value used
# for both the standalone --array=0-0 smoke submission and (unused) if the whole array were
# ever submitted in one shot. H41_INDEX keeps EVAL_SEED+index consistent with evaluate.sbatch's
# canonical per-h seeding convention (cluster/SKILL.md gotcha 4).
# ---------------------------------------------------------------------------
H_VALUES=(0.730 0.660 0.665 0.670)
H41_INDEX=(21 7 8 9)
H_VALUE="${H_VALUES[$TID]}"
H41_IDX="${H41_INDEX[$TID]}"
EVAL_SEED=777000
TASK_SEED=$((EVAL_SEED + H41_IDX))

H_LABEL=$(echo "$H_VALUE" | sed -E 's/0+$//; s/\.$//; s/\./_/')
POSTERIOR_JSON="$RUN_DIR/simulations/posteriors/h_${H_LABEL}.json"
POSTERIOR_BH_JSON="$RUN_DIR/simulations/posteriors_with_bh_mass/h_${H_LABEL}.json"
if [[ -f "$POSTERIOR_JSON" && -f "$POSTERIOR_BH_JSON" ]]; then
    echo "Posteriors for h=$H_VALUE already exist -- skipping (idempotency guard):"
    echo "  $POSTERIOR_JSON"
    echo "  $POSTERIOR_BH_JSON"
    exit 0
fi

TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "wave2 C4 PROD-CF-2D mz_sel/eff arm T, h=$H_VALUE, launched under rows #222/#223 charter node B7.2"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

# ---------------------------------------------------------------------------
# Arm T: CoR-P CLI verbatim EXCEPT catalogue_numerator_survival_2d/_center, which are this
# arm's own counterfactual values (proposal §13.3).
# ---------------------------------------------------------------------------
echo "=== wave2-c4-twin: task=$TID h=$H_VALUE seed=$TASK_SEED (H41 idx $H41_IDX) $( [[ $TID -eq 0 ]] && echo '[STEP-2 SMOKE]' ) ==="

python -m darksiren_emri "$RUN_DIR" \
    --evaluate \
    --h_value "$H_VALUE" \
    --seed "$TASK_SEED" \
    --simulation_index "$H41_IDX" \
    --strategy physics-floor \
    --pdet_dl_bins 60 \
    --pdet_mass_bins 40 \
    --pdet_estimator local_linear \
    --pdet_z_resolved \
    --fisher_cond_threshold 1e16 \
    --host_z_kernel volume_deconv \
    --host_mass_kernel auto \
    --normalization_mode absolute_marginal \
    --selection_in_completion_numerator fused \
    --catalogue_mass_overlap production \
    --catalogue_mass_error_scale 1.0 \
    --completion_b_scale derived \
    --eddington_m on \
    --sigma4d_mass_kernel point \
    --completion_event_measure ratio \
    --catalogue_numerator_survival_2d mz_sel \
    --catalogue_numerator_survival_2d_center eff \
    --catalogue_global_selection phi \
    --mass_filter_geometry linear \
    --mass_filter_k 1.5 \
    --log_level INFO

echo "=== done: wave2-c4-twin task=$TID h=$H_VALUE ==="
