#!/usr/bin/env bash
# cluster/graph1_t5_armR_c0prime.sbatch -- Research Graph 1, Branch F node
# m-t5-armR-c0prime: Arm R's C0-prime-equivalent INGREDIENT check (row #284(4a): "Arm R launch
# strictly behind its own C0-prime-equivalent gate").
#
# **Launched under ledger row #290 (Research Graph 1 RATIFIED), decisions row 8 ("Arm R launch
# strictly behind its own C0-prime-equivalent gate", row #284(4a)) + row #301 (wave-2 launch;
# g-c0-baseline re-stamped GREEN-AS-CORRECTED per docket item 5(A)).**
#
# PURPOSE: Arm R (PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md sec 6.2, T5.2 of the charter) reads
# joint_r1 at mass_filter_geometry=log / mass_filter_k=3.0 against "the banked joint_r1 HEAD
# readout at the same nodes (zero compute)" as its baseline -- sec 6.2 explicitly requires "a
# joint_r1 C0-prime task, approx 1-2 CPU-h ... because no joint_r1 baseline has been re-run at
# the current HEAD" before that baseline can be trusted. This script IS that ingredient check: it
# reproduces, at the CURRENT commit, the single h=0.730 (H41 index 21) row of the already-banked
# m-head-rebaseline blind joint_r1 HEAD readout (cluster/graph1_headrebaseline_joint_r1.sbatch,
# task 21; out-root run_20260902_graph1_headrebaseline_joint_r1) -- i.e. the joint_r1 HEAD-readout
# CLI at production baseline flags (mass_filter_geometry linear / mass_filter_k 1.5; BLIND to
# catalogue_numerator_survival_2d, matching the m-head-rebaseline convention). This is NOT Arm R's
# own log/k=3 measurement -- Arm R itself does not launch until this gate's bit-identity read is
# evaluated green (that evaluation is a later read, not this launch).
#
# CoR-P CLI verbatim source: cluster/graph1_headrebaseline_joint_r1.sbatch (byte-identical CLI,
# single task instead of the 41-task H_GRID_41 array). Comparand row: the banked
# run_20260902_graph1_headrebaseline_joint_r1 task-21 output (h_0_73.json,
# posteriors_with_bh_mass/h_0_73.json, diagnostics/event_likelihoods.csv), itself the state
# candidate this graph's Branch B node banked in wave-1.
#
# Array shape: single task (--array=0-0) -- only the one h=0.730 node is needed for an ingredient
# check (matches the wave-3 C0-prime / graph1_c0prime_headrebaseline_gate single-h economy).
#
# --time=00:45:00: the iiib-anchor value from graph1_c0prime_headrebaseline_gate.sbatch would be
# too tight for joint_r1 (>=2.2x iiib historically); using the same 01:30:00 ceiling as
# graph1_headrebaseline_joint_r1.sbatch / wave3_c0prime_off_gate.sbatch for consistency and
# margin.
#
# Dataset pins (CLAUDE.md pinning rule) -- identical to graph1_headrebaseline_joint_r1.sbatch:
# CRB set run_20260729_seed61000 (md5 9a1f2a14384a9281c97ca3be312ddaab); reduced galaxy catalogue
# (md5 c52c13b5cab61f6b3f04bbe202550969); observed-catalogue realization seed900001 (sha256
# e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751). All STOP-gated below.
#
# Required env (set by the submitter via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260902_graph1_t5_armR_c0prime_joint_r1
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=graph1-t5-armR-c0prime
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-0
#SBATCH --time=01:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-0)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/graph1_t5_armR_c0prime_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-t5-armR-c0prime task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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

# Single h-point: h = 0.730 only, canonical H41 index 21 -- bit-identical to the banked
# run_20260902_graph1_headrebaseline_joint_r1 task-21 seed, modulo the code-state difference this
# gate exists to certify away.
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
write_provenance "$RUN_DIR/logs" "graph1 m-t5-armR-c0prime ingredient check, venue=joint_r1, h=$H_VALUE, task=$TID -- reproduces the banked m-head-rebaseline joint_r1 task-21 row bit-for-bit at the current commit (baseline mass_filter_geometry=linear/k=1.5, blind to catalogue_numerator_survival_2d), the C0-prime-equivalent gate Arm R's own log/k=3 measurement launches behind; launched under ledger row #290 decisions row 8 (row #284(4a)) + row #301"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-t5-armR-c0prime: task=$TID h=$H_VALUE seed=$TASK_SEED observed_catalogue=$OBS_CATALOGUE ==="

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

echo "=== done: graph1-t5-armR-c0prime task=$TID h=$H_VALUE ==="
