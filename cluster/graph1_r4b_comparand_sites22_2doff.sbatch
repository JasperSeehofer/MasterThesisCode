#!/usr/bin/env bash
# cluster/graph1_r4b_comparand_sites22_2doff.sbatch — R4b discriminating test: ONE
# evaluation of the production HEAD on venue iiib at h=0.73, theta_sites="2.2",
# theta=(0,1) (b=0,s=1), smear_global_selection OFF, catalogue_leg_1d_mass_aware
# OFF, PLUS the S0-B driver's pinned 2D catalogue-numerator-survival counterfactual.
#
# Copied verbatim from cluster/graph1_r4_comparand_sites22.sbatch (job 6790708, the
# R4 comparand) with EXACTLY one additional deviation:
#   (no flag, defaults to production-adopted mz_sel/eff)
#     -> --catalogue_numerator_survival_2d off
#     -> --catalogue_numerator_survival_2d_center unset
#   This is the driver's pinned pre-[P3-2D]-adoption counterfactual, copied exactly
#   from hier_s0_driver.py's iiib call site (results/campaign51_20260728/
#   realistic_20260729/fanout1_20260829/hier_s0_driver.py:1289-1290:
#     catalogue_numerator_survival_2d="off",
#     catalogue_numerator_survival_2d_center="unset",
#   ). Per the driver's own inline comment (lines 1287-1288), this pin exists "so
#   this banked Stage-0/KW-Q1 comparand stays byte-identical" -- i.e. it is a
#   deliberate pre-adoption counterfactual, never a production posterior value.
#   Flag names/choices confirmed via `uv run python -m darksiren_emri --help`:
#     --catalogue_numerator_survival_2d {off,mz_sel}
#     --catalogue_numerator_survival_2d_center {unset,raw,eff}
#
# Purpose (exec/d-photoz-leverage/R4_COMPARAND_READ.md CHAIR CORRECTION, 2026-09-04):
# if this run's event_likelihoods.csv is byte-identical to the S0-B truth node
# (job 6779532, node_truth_iiib_sites2.2_nosmear), the S0-B measurement was taken
# on a PRE-ADOPTION counterfactual of the production 2D setting, not on the
# production HEAD -- discriminates the CHAIR CORRECTION's hypothesis.
#
# All other flags/resources/dataset pins identical to graph1_r4_comparand_sites22.sbatch:
#   cpu_il, 16 cpus, 00:45:00
#   CRB set = run_20260729_seed61000 (md5 9a1f2a14384a9281c97ca3be312ddaab)
#   reduced galaxy catalogue (md5 c52c13b5cab61f6b3f04bbe202550969)
#
# Required env (set by the submitter via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/graph1_r4b_comparand_sites22_2doff_20260904
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)
#
# NOT interpreted by this job or its submitter -- output is retrieved and left
# for the author/orchestrator to read.

#SBATCH --job-name=graph1-r4b
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=00:45:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/graph1_r4b_comparand_sites22_2doff_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-r4b job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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

H_VALUE="0.730"
EVAL_SEED=777021
TASK_SEED="$EVAL_SEED"
TASK_INDEX=0

POSTERIOR_JSON="$RUN_DIR/simulations/posteriors/h_0_73.json"
POSTERIOR_BH_JSON="$RUN_DIR/simulations/posteriors_with_bh_mass/h_0_73.json"
if [[ -f "$POSTERIOR_JSON" && -f "$POSTERIOR_BH_JSON" ]]; then
    echo "Posteriors for h=$H_VALUE already exist -- skipping (idempotency guard):"
    echo "  $POSTERIOR_JSON"
    echo "  $POSTERIOR_BH_JSON"
    exit 0
fi

source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "graph1 R4b discriminating test, venue=iiib, h=$H_VALUE, theta_sites=2.2, theta=(0,1), smear_global_selection=off, catalogue_leg_1d_mass_aware=off, catalogue_numerator_survival_2d=off (driver-pinned counterfactual) -- one additional deviation from graph1_r4_comparand_sites22.sbatch (job 6790708)"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-r4b: h=$H_VALUE seed=$TASK_SEED ==="

python -m darksiren_emri "$RUN_DIR" \
    --evaluate \
    --h_value "$H_VALUE" \
    --seed "$TASK_SEED" \
    --simulation_index "$TASK_INDEX" \
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
    --catalogue_global_selection phi \
    --mass_filter_geometry linear \
    --mass_filter_k 1.5 \
    --theta_b 0.0 \
    --theta_s 1.0 \
    --theta_sites 2.2 \
    --catalogue_leg_1d_mass_aware off \
    --catalogue_numerator_survival_2d off \
    --catalogue_numerator_survival_2d_center unset \
    --log_level INFO

echo "=== done: graph1-r4b h=$H_VALUE ==="
