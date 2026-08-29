#!/usr/bin/env bash
# cluster/wave2_c3_win_k3.sbatch — wave-2 charter node C3: log k=3 mass-window geometry
# counterfactual, arm T only (charter node B5.2). Registration:
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md
#   (esp. §1 "The run", §2 design matrix, §11 cost)
#
# **Launched under rows #222/#223 — charter nodes C0/B5.2/B7.2/B1.2.**
#
# Baseline B (`mass_filter_geometry=linear, mass_filter_k=1.5`) is REUSED at zero compute from
# the banked HEAD readout (results/.../headreadout_20260827/iiib/), conditional on the shared
# C0 baseline-gate task PASSing (§1 of the registration, L5). This sbatch runs ONLY arm T:
# `mass_filter_geometry=log, mass_filter_k=3.0`, everything else CoR-P, over the H4 grid
# {0.660, 0.665, 0.670, 0.730} (registration §1 "H4 grid"). One h-value per array task
# (backfill-friendly, gotcha 7).
#
# F4 COST ESTIMATE: 4 H4 nodes x 14.93-22.9 CPU-h/node x 0.73-1.5 (measured p95 candidate-growth
# factor, PHYSICS_CHANGE_MASS_WINDOW_GEOMETRY_20260829.md §7: aggregate ratio 0.726, p95
# per-event ratio 1.498) = 44-137 CPU-h (COMPUTE_LEDGER.md row C3). --time sized off the p95
# growth factor (not the mean, per B5.1's recommendation) against the CONTENDED HEAD-readout
# off-arm slowest task anchor 1:25:52 (5152 s): 5152 * 1.498 = 7718 s = 2:08:38 -- --time=03:00:00
# below keeps ~40% margin over that product.
#
# Required env (set by cluster/submit_wave2.sh via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260829_wave2_c3_iiib (out-root name fixed by
#                    results/_archive/archive_run_wave2.sh -- do not rename)
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=wave2-c3-wink3
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --array=0-3

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set -- submit_wave2.sh must export it}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-3)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/wave2_c3_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== wave2-c3-wink3 task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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
# H4 grid, task index -> (h_value, canonical H_GRID_41 index for the seed convention).
# The H_GRID_41 index keeps EVAL_SEED+index consistent with evaluate.sbatch's own
# per-h seeding convention (cluster/SKILL.md gotcha 4) rather than this array's local index.
# ---------------------------------------------------------------------------
H_VALUES=(0.660 0.665 0.670 0.730)
H41_INDEX=(7 8 9 21)
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
write_provenance "$RUN_DIR/logs" "wave2 C3 log-k3 window counterfactual arm T, h=$H_VALUE, launched under rows #222/#223 charter node B5.2"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

# ---------------------------------------------------------------------------
# Arm T: CoR-P CLI verbatim (headreadout_20260827/iiib/run_metadata_21.json:cli_args) EXCEPT
# mass_filter_geometry/mass_filter_k, which are the arm's own counterfactual values
# (mass_filter_sigma stays at the production "symmetric" default in BOTH arms -- not varied,
# registration §1/§7).
# ---------------------------------------------------------------------------
echo "=== wave2-c3-wink3: task=$TID h=$H_VALUE seed=$TASK_SEED (H41 idx $H41_IDX) ==="

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
    --catalogue_numerator_survival_2d off \
    --catalogue_global_selection phi \
    --mass_filter_geometry log \
    --mass_filter_k 3.0 \
    --log_level INFO

echo "=== done: wave2-c3-wink3 task=$TID h=$H_VALUE ==="
