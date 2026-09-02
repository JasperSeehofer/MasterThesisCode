#!/usr/bin/env bash
# cluster/graph1_t5_armR.sbatch -- Research Graph 1, Branch F node m-t5-armR: Arm R's own
# measurement (T5.2 of the charter) -- log-symmetric mass window k=3.0 vs the linear k=1.5
# baseline, joint_r1 venue, H4 grid.
#
# **Launched under ledger row #290 (Research Graph 1 RATIFIED), decisions row 8 ("Arm R launch
# strictly behind its own C0-prime-equivalent gate", row #284(4a)) + row #301 (wave-2 launch
# authorization). Gate: exec/m-t5-armR-c0prime/eval/GATE_RECORD.md -- GREEN
# ("Bit-identity holds on every shared column of every named comparand artifact ... GREEN. No
# numbers are banked as a delta because there is no delta to bank."), sacct job 6767465_0
# COMPLETED 0:0, 0 mismatches / 0 missing / 1549/1549 files verified against the banked
# run_20260902_graph1_headrebaseline_joint_r1 task-21 comparand.**
#
# Registration: results/campaign51_20260728/realistic_20260729/tree2_20260830/
#   PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md section 6.2 (Arm R design):
#   "the joint_r1 HEAD-readout CLI (headreadout_20260827/joint_r1/run_metadata_21.json:cli_args,
#   observed_catalogue seed 900001) with mass_filter_geometry = log, mass_filter_k = 3.0, on the
#   H4 grid; baseline = the banked joint_r1 HEAD readout at the same nodes (zero compute)."
#   CLI verbatim source (byte-identical except geometry/k and the H4 subset instead of the H41
#   array): cluster/graph1_headrebaseline_joint_r1.sbatch (= the joint_r1 HEAD-readout CLI the
#   design cites) / cluster/graph1_t5_armR_c0prime.sbatch (the gate script that reproduced its
#   task-21 row bit-for-bit at the current commit).
#
# Baseline (NOT re-run here, per the design's "zero compute" instruction): the banked
# run_20260902_graph1_headrebaseline_joint_r1 task-21/{7,8,9} rows (H4 grid), certified
# reproducible at the current code state by the GATE_RECORD above (bit-identity at task 21;
# tasks 7/8/9 share the identical CLI/code path, not independently re-gated -- the gate's own
# scope, per its "Verdict statement", is code-state identity, not per-h numerics).
#
# Array shape: 4 tasks, H4 grid {0.660, 0.665, 0.670, 0.730} = H41 indices {7, 8, 9, 21} --
# identical H4 convention to cluster/graph1_t5_armS_iiib.sbatch.
#
# Cost: design section 6.2 -- "approx 11-15 CPU-h (docket 2 section 7 rank 5; joint_r1 HEAD cost
# >= 2.2x iiib) + the C0-prime [already spent: sacct-measured 6:37 wall / 16 cpus =~ 1.8 CPU-h]."
# This script is the 11-15 CPU-h leg only.
#
# --time=01:30:00: identical ceiling to graph1_headrebaseline_joint_r1.sbatch / the C0-prime gate
# (same venue, same per-task cost class; C0-prime measured 6:37 for one task at linear k=1.5 --
# log k=3.0 exercises the same window-evaluation code path at comparable cost).
#
# Dataset pins (CLAUDE.md pinning rule) -- identical to graph1_headrebaseline_joint_r1.sbatch /
# graph1_t5_armR_c0prime.sbatch: CRB set run_20260729_seed61000 (md5
# 9a1f2a14384a9281c97ca3be312ddaab); reduced galaxy catalogue (md5
# c52c13b5cab61f6b3f04bbe202550969); observed-catalogue realization seed900001 (sha256
# e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751). All STOP-gated below.
#
# Required env (set by the submitter via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260902_graph1_t5_armR_joint_r1
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=graph1-t5-armR
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --time=01:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-3)}"

H_VALUES=(0.660 0.665 0.670 0.730)
H41_INDEX=(7 8 9 21)
if [[ "$TID" -ge "${#H_VALUES[@]}" ]]; then
    echo "STOP: array task id $TID out of range (expected 0-3: H4 grid)" >&2
    exit 1
fi
H_VALUE="${H_VALUES[$TID]}"
H41_IDX="${H41_INDEX[$TID]}"
EVAL_SEED=777000
TASK_SEED=$((EVAL_SEED + H41_IDX))
TASK_INDEX="$H41_IDX"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/graph1_t5_armR_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-t5-armR task=$TID h=$H_VALUE job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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
write_provenance "$RUN_DIR/logs" "graph1 m-t5-armR: log-symmetric mass window k=3.0 vs linear k=1.5 baseline (design PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md section 6.2), venue=joint_r1, h=$H_VALUE, task=$TID (H41 idx $H41_IDX) -- launched under ledger row #290 decisions row 8 (row #284(4a)), C0-prime-equivalent gate GREEN per exec/m-t5-armR-c0prime/eval/GATE_RECORD.md"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-t5-armR: task=$TID h=$H_VALUE seed=$TASK_SEED (H41 idx $H41_IDX) observed_catalogue=$OBS_CATALOGUE ==="

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
    --mass_filter_geometry log \
    --mass_filter_k 3.0 \
    --theta_b 0.0 \
    --theta_s 1.0 \
    --theta_sites all \
    --observed_catalogue "$OBS_CATALOGUE" \
    --log_level INFO

echo "=== done: graph1-t5-armR task=$TID h=$H_VALUE ==="
