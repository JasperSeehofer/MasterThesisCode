#!/usr/bin/env bash
# cluster/graph1_r4_comparand_sites22.sbatch — R4 comparand: ONE evaluation of the
# production HEAD on venue iiib at h=0.73, theta_sites="2.2", theta=(0,1)
# (b=0,s=1), smear_global_selection OFF, catalogue_leg_1d_mass_aware OFF.
#
# CLI verbatim from cluster/graph1_headrebaseline_iiib.sbatch (the registered
# CoR-P production CLI used by the m-head-rebaseline iiib run,
# exec/m-head-rebaseline/LAUNCH_RECORD.md), with exactly two deviations:
#   --theta_sites all  -> --theta_sites 2.2   (task requirement)
#   (no flag)          -> --catalogue_leg_1d_mass_aware off   (task requirement;
#                          production default is 'auto', which resolves 'on'
#                          post-flip -- row #284-#286 -- so this is an explicit
#                          override, not a no-op)
# --theta_b 0.0 / --theta_s 1.0 / no --smear_global_selection are already the
# headrebaseline baseline's values (matches task requirement as-is).
# REGISTERED_RESOLVED_FLAGS reference: darksiren_emri/validation/correspondence_1d.py:3154.
#
# Single h-value, single task (not the H_GRID_41 array) -- one evaluate() call
# at h=0.73. Same per-task resources as one graph1_headrebaseline_iiib task
# (cpu_il, 16 cpus, ~7 min/h-point anchor -> 00:45:00 keeps the same ~5x margin).
#
# Dataset pins (CLAUDE.md pinning rule), identical to graph1_headrebaseline_iiib.sbatch:
#   CRB set = run_20260729_seed61000 (md5 9a1f2a14384a9281c97ca3be312ddaab)
#   reduced galaxy catalogue (md5 c52c13b5cab61f6b3f04bbe202550969)
#
# Required env (set by the submitter via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/graph1_r4_comparand_sites22_20260904
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)
#
# NOT interpreted by this job or its submitter -- output is retrieved and left
# for the author/orchestrator to read.

#SBATCH --job-name=graph1-r4-comparand-sites22
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

LOG_BASE="$RUN_DIR/logs/graph1_r4_comparand_sites22_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-r4-comparand-sites22 job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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
write_provenance "$RUN_DIR/logs" "graph1 R4 comparand, venue=iiib, h=$H_VALUE, theta_sites=2.2, theta=(0,1), smear_global_selection=off, catalogue_leg_1d_mass_aware=off -- deviations from m-head-rebaseline iiib CLI"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-r4-comparand-sites22: h=$H_VALUE seed=$TASK_SEED ==="

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
    --log_level INFO

echo "=== done: graph1-r4-comparand-sites22 h=$H_VALUE ==="
