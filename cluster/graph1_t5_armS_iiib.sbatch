#!/usr/bin/env bash
# cluster/graph1_t5_armS_iiib.sbatch — Research Graph 1, Branch F node m-t5-armS: log-symmetric
# mass window vs linear k=1.5, iiib venue, k in {2.0, 2.5, 3.5} (+ optional k = infinity anchor).
#
# **Launched under ledger row #290 (Research Graph 1 RATIFIED), decisions row 8
# (m-t5-armS + m-t5-armR, Branch F, [DO] Approved -- ARM S ONLY; Arm R is gated behind its own
# C0-prime-equivalent check, NOT launched here).**
#
# Registration: results/campaign51_20260728/realistic_20260729/tree2_20260830/
#   PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md section 6.1 (Arm S design, ratified rows
#   #278(2)/#284(4)).
#
# Form: verbatim the C3 form (cluster/wave2_c3_win_k3.sbatch CLI; CRB_SRC run_20260729_seed61000;
# catalogue md5 c52c13b5..., CRB md5 9a1f2a14..., EVAL_SEED 777000 + H41 index {7, 8, 9, 21};
# H4 grid {0.660, 0.665, 0.670, 0.730}) with mass_filter_geometry=log and mass_filter_k varied
# over {2.0, 2.5, 3.5} (+ optional k=infinity, implemented as mass_filter_k=1000000.0, which
# converges both geometries to the all-True mask per the registration's invariant 2).
# mass_filter_sigma stays at the production "symmetric" default (not varied).
# catalogue_numerator_survival_2d = off is HELD EXPLICIT at the C3 value (section 6.1: "so that
# the banked k=3 point (+0.003523) is a valid fourth point on the same curve" -- k=3.0 itself is
# NOT re-run here; it is already banked from wave-2 job C3 / cluster/wave2_c3_win_k3.sbatch).
# This is a DISCLOSED CHOICE against the B7.3/row #286 production default (mz_sel/eff) -- a
# second scan at the new default is a separate registration if warranted later.
#
# Array shape: 16 tasks (4 k-values x 4 H4 nodes). TID = k_index*4 + h_index.
#   k_index 0 -> k=2.0, 1 -> k=2.5, 2 -> k=3.5, 3 -> k=infinity (1000000.0, optional anchor)
#   h_index 0..3 -> H4 grid {0.660, 0.665, 0.670, 0.730} = H41 indices {7, 8, 9, 21}
#
# --time=03:00:00: identical sizing rationale to wave2_c3_win_k3.sbatch (p95 growth-factor anchor
# against the contended HEAD-readout off-arm slowest task; the k=infinity anchor is a strict
# subset of the linear-window candidate pool -- expected NOT slower than k=3.0's measured
# 4.97 CPU-h / 4 tasks).
#
# Cost: 4 k-sets x ~5 CPU-h/set = ~20 CPU-h (registration section 6.1: "3 (or 4) x approx 5
# CPU-h per k-node set ... expected total 15-20 CPU-h").
#
# Required env (set by the submitter via --export=ALL,...):
#   BASE_RUN_DIR -- $WORKSPACE/run_20260902_graph1_t5_armS_iiib (per-k subdirectory computed below)
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=graph1-t5-armS-iiib
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-15
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${BASE_RUN_DIR:?BASE_RUN_DIR not set}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-15)}"

K_VALUES=(2.0 2.5 3.5 1000000.0)
K_LABELS=(k2_0 k2_5 k3_5 kinf)
H_VALUES=(0.660 0.665 0.670 0.730)
H41_INDEX=(7 8 9 21)

K_IDX=$((TID / 4))
H_IDX=$((TID % 4))
if [[ "$K_IDX" -ge "${#K_VALUES[@]}" || "$H_IDX" -ge "${#H_VALUES[@]}" ]]; then
    echo "STOP: array task id $TID out of range (expected 0-15: 4 k-values x 4 H4 nodes)" >&2
    exit 1
fi
K_VALUE="${K_VALUES[$K_IDX]}"
K_LABEL="${K_LABELS[$K_IDX]}"
H_VALUE="${H_VALUES[$H_IDX]}"
H41_IDX="${H41_INDEX[$H_IDX]}"
EVAL_SEED=777000
TASK_SEED=$((EVAL_SEED + H41_IDX))

RUN_DIR="$BASE_RUN_DIR/$K_LABEL"
mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/graph1_t5_armS_iiib_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-t5-armS-iiib task=$TID k=$K_VALUE ($K_LABEL) h=$H_VALUE job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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
write_provenance "$RUN_DIR/logs" "graph1 m-t5-armS: log-symmetric mass window k=$K_VALUE ($K_LABEL), venue=iiib, h=$H_VALUE, task=$TID -- ratified design PROPOSAL_MASS_LAW_KEYED_WINDOW_20260830.md section 6.1; launched under ledger row #290 decisions row 8 (Arm S only)"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-t5-armS-iiib: task=$TID k=$K_VALUE h=$H_VALUE seed=$TASK_SEED (H41 idx $H41_IDX) ==="

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
    --mass_filter_k "$K_VALUE" \
    --log_level INFO

echo "=== done: graph1-t5-armS-iiib task=$TID k=$K_VALUE h=$H_VALUE ==="
