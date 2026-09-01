#!/usr/bin/env bash
# cluster/graph1_headrebaseline_joint_r1.sbatch — Research Graph 1, Branch B node
# m-head-rebaseline: the new banked HEAD readout under the post-flip production default,
# joint_r1 venue (observed-catalogue realization).
#
# **Launched under ledger row #290 (Research Graph 1 RATIFIED), decisions row 4
# (m-head-rebaseline, Branch B, [DO] Approved).**
#
# Same object as graph1_headrebaseline_iiib.sbatch (see that script's header for the full
# rationale) run against the joint_r1 venue: the reduced GLADE+ catalogue replaced, for host
# lookups, by the observed-catalogue realization (--observed_catalogue), matching the wave-3
# joint_r1 CLI exactly except for run-dir/commit.
#
# BLINDNESS: identical to the iiib sibling -- no "--catalogue_numerator_survival_2d" /
# "--catalogue_numerator_survival_2d_center" flag passed; catalogue_leg_1d_mass_aware left at its
# own "auto" default (engages post-flip per row #286).
#
# Observed-catalogue realization + dataset pins (CLAUDE.md pinning rule):
#   path:   $WORKSPACE/realizations_20260729/observed_catalogue_seed900001.csv
#   sha256: e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751
# STOP-gated below, on top of the shared CRB/catalogue md5 gates.
#
# Realization-sidecar gotcha (cluster/SKILL.md gotcha #10): verify the sidecar's "parent_csv"
# still resolves before this array is trusted -- verified as part of this launch's pre-flight
# (see LAUNCH_RECORD.md).
#
# --time=01:30:00: identical sizing rationale to wave3_headreadout_joint_r1.sbatch (joint_r1
# historically >= 2.2x the iiib per-task anchor).
#
# Required env (set by the submitter via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260902_graph1_headrebaseline_joint_r1
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=graph1-headrebaseline-joint_r1
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-40
#SBATCH --time=01:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-40)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/graph1_headrebaseline_joint_r1_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-headrebaseline-joint_r1 task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

CRB_SRC="$WORKSPACE/run_20260729_seed61000"
ln -sfn "$CRB_SRC/simulations/cramer_rao_bounds.csv"          "$RUN_DIR/simulations/cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/injections"                     "$RUN_DIR/simulations/injections"

EXPECTED_CRB_MD5="9a1f2a14384a9281c97ca3be312ddaab"
EXPECTED_CATALOGUE_MD5="c52c13b5cab61f6b3f04bbe202550969"
EXPECTED_OBS_CATALOGUE_SHA256="e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751"
OBS_CATALOGUE="$WORKSPACE/realizations_20260729/observed_catalogue_seed900001.csv"

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
if [[ ! -f "$OBS_CATALOGUE" ]]; then
    echo "STOP: observed-catalogue realization not found: $OBS_CATALOGUE" >&2
    exit 1
fi
ACTUAL_OBS_CATALOGUE_SHA256=$(sha256sum "$OBS_CATALOGUE" | awk '{print $1}')
if [[ "$ACTUAL_OBS_CATALOGUE_SHA256" != "$EXPECTED_OBS_CATALOGUE_SHA256" ]]; then
    echo "STOP: observed_catalogue_seed900001.csv sha256 mismatch (expected $EXPECTED_OBS_CATALOGUE_SHA256, got $ACTUAL_OBS_CATALOGUE_SHA256)" >&2
    exit 1
fi
echo "dataset pins OK: CRB=$ACTUAL_CRB_MD5 catalogue=$ACTUAL_CATALOGUE_MD5 observed_catalogue=$ACTUAL_OBS_CATALOGUE_SHA256"

H_VALUES=(0.600 0.610 0.620 0.630 0.640 0.650 0.655 0.660 0.665 0.670 0.675 0.680 0.685 0.690 0.695 0.700 0.705 0.710 0.715 0.720 0.725 0.730 0.735 0.740 0.745 0.750 0.755 0.760 0.765 0.770 0.775 0.780 0.785 0.790 0.800 0.810 0.820 0.830 0.840 0.850 0.860)
if [[ "$TID" -ge "${#H_VALUES[@]}" ]]; then
    echo "STOP: array task id $TID out of range for H_GRID_41 (${#H_VALUES[@]} nodes)" >&2
    exit 1
fi
H_VALUE="${H_VALUES[$TID]}"
EVAL_SEED=777000
TASK_SEED=$((EVAL_SEED + TID))
TASK_INDEX="$TID"

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
write_provenance "$RUN_DIR/logs" "graph1 m-head-rebaseline blind HEAD readout, venue=joint_r1, h=$H_VALUE, task=$TID -- BLIND to any post-flip flag (no explicit --catalogue_numerator_survival_2d/_center; catalogue_leg_1d_mass_aware left at its own 'auto' default); launched under ledger row #290 decisions row 4"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-headrebaseline-joint_r1: task=$TID h=$H_VALUE seed=$TASK_SEED observed_catalogue=$OBS_CATALOGUE ==="

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
    --theta_sites all \
    --observed_catalogue "$OBS_CATALOGUE" \
    --log_level INFO

echo "=== done: graph1-headrebaseline-joint_r1 task=$TID h=$H_VALUE ==="
