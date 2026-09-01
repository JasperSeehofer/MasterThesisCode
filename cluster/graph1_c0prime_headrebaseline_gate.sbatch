#!/usr/bin/env bash
# cluster/graph1_c0prime_headrebaseline_gate.sbatch — Research Graph 1, Branch B node
# m-head-rebaseline: the C0-prime ingredient gate that must PASS before the blind HEAD arrays are
# trusted as the new comparand (g-c0-baseline: max_abs = 0 on shared columns, md5 match).
#
# **Launched under ledger row #290 (Research Graph 1 RATIFIED), decisions row 4
# (m-head-rebaseline, Branch B, [DO] Approved).**
#
# PURPOSE: reproduce the wave-3 blind HEAD readout's own h=0.730 row
# (run_20260830_wave3_headreadout_{iiib,joint_r1}, task 21) bit-for-bit at the CURRENT commit
# (post-flip, 5e7fda16 an ancestor of HEAD) by passing the SAME explicit flags that wave-3's
# blind resolution used at ITS commit (pre-flip: --catalogue_numerator_survival_2d off
# --catalogue_numerator_survival_2d_center unset) -- i.e. this is the wave3_c0prime_off_gate.sbatch
# pattern (rows #279/#281/#283), re-run at the post-flip commit, comparand = the wave-3 banked
# blind HEAD row instead of the 2026-08-27 banked row. A PASS certifies that nothing outside the
# 1D mass-aware flip moved behaviour at this explicit flag value, so the wave-3 banked blind
# readout remains a valid zero-compute baseline object wherever this graph node's own new blind
# HEAD readout needs one. This script does NOT itself compute the diff -- g-c0-baseline is
# evaluated at readout by the orchestrator (max_abs = 0 on shared columns; md5 match).
#
# Array shape: 2 tasks. Task 0 = iiib, task 1 = joint_r1. Single h-point (h = 0.730, H41 index 21,
# seed 777000 + 21 = 777021), same economy as wave3_c0prime_off_gate.sbatch.
#
# Dataset pins (CLAUDE.md pinning rule) -- identical to the wave-3 scripts: CRB set
# run_20260729_seed61000 (md5 9a1f2a14384a9281c97ca3be312ddaab); reduced galaxy catalogue (md5
# c52c13b5cab61f6b3f04bbe202550969); joint_r1 additionally pins the observed-catalogue realization
# (sha256 e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751). STOP-gated below.
#
# Comparand of record for the readout's diff: results/campaign51_20260728/realistic_20260729/
#   wave3_20260830/{iiib,joint_r1}/simulations/{posteriors,posteriors_with_bh_mass}/h_0_73.json and
#   diagnostics/event_likelihoods.csv row for h=0.730 (task 21 of job 6746275/6746276).
#
# Required env: PROJECT_ROOT (optional; falls back to $HOME/darksiren-emri). No RUN_DIR export --
# this script computes its own per-task out-root, matching wave3_c0prime_off_gate.sbatch.

#SBATCH --job-name=graph1-c0prime-headrebaseline
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --time=01:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-1)}"

case "$TID" in
    0) VENUE="iiib" ;;
    1) VENUE="joint_r1" ;;
    *)
        echo "STOP: array task id $TID out of range (expected 0=iiib, 1=joint_r1)" >&2
        exit 1
        ;;
esac

RUN_DIR="$WORKSPACE/run_20260902_graph1_c0prime_headrebaseline_${VENUE}"
mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/graph1_c0prime_headrebaseline_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-c0prime-headrebaseline task=$TID venue=$VENUE job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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

OBS_CATALOGUE_FLAGS=()
if [[ "$VENUE" == "joint_r1" ]]; then
    EXPECTED_OBS_CATALOGUE_SHA256="e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751"
    OBS_CATALOGUE="$WORKSPACE/realizations_20260729/observed_catalogue_seed900001.csv"
    if [[ ! -f "$OBS_CATALOGUE" ]]; then
        echo "STOP: observed-catalogue realization not found: $OBS_CATALOGUE" >&2
        exit 1
    fi
    ACTUAL_OBS_CATALOGUE_SHA256=$(sha256sum "$OBS_CATALOGUE" | awk '{print $1}')
    if [[ "$ACTUAL_OBS_CATALOGUE_SHA256" != "$EXPECTED_OBS_CATALOGUE_SHA256" ]]; then
        echo "STOP: observed_catalogue_seed900001.csv sha256 mismatch (expected $EXPECTED_OBS_CATALOGUE_SHA256, got $ACTUAL_OBS_CATALOGUE_SHA256)" >&2
        exit 1
    fi
    echo "observed-catalogue pin OK: $ACTUAL_OBS_CATALOGUE_SHA256"
    OBS_CATALOGUE_FLAGS=(--observed_catalogue "$OBS_CATALOGUE")
fi

H_VALUE="0.730"
EVAL_SEED=777000
TASK_INDEX=21
TASK_SEED=$((EVAL_SEED + TASK_INDEX))

POSTERIOR_JSON="$RUN_DIR/simulations/posteriors/h_0_73.json"
POSTERIOR_BH_JSON="$RUN_DIR/simulations/posteriors_with_bh_mass/h_0_73.json"
if [[ -f "$POSTERIOR_JSON" && -f "$POSTERIOR_BH_JSON" ]]; then
    echo "Posteriors for h=$H_VALUE already exist -- skipping (idempotency guard):"
    echo "  $POSTERIOR_JSON"
    echo "  $POSTERIOR_BH_JSON"
    exit 0
fi

TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "graph1 m-head-rebaseline C0-prime gate, venue=$VENUE, h=$H_VALUE, task=$TID -- explicit --catalogue_numerator_survival_2d off/_center unset at the post-flip commit, certifying the wave-3 banked blind HEAD row (task 21) reproduces bit-for-bit; launched under ledger row #290 decisions row 4"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-c0prime-headrebaseline: task=$TID venue=$VENUE h=$H_VALUE seed=$TASK_SEED ==="

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
    --catalogue_numerator_survival_2d off \
    --catalogue_numerator_survival_2d_center unset \
    ${OBS_CATALOGUE_FLAGS[@]+"${OBS_CATALOGUE_FLAGS[@]}"} \
    --log_level INFO

echo "=== done: graph1-c0prime-headrebaseline task=$TID venue=$VENUE h=$H_VALUE ==="
