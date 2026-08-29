#!/usr/bin/env bash
# cluster/wave2_c1_s0b_TEMPLATE.sbatch — wave-2 charter node C1: S0-B production theta-score
# at truth (charter node B1.2). TEMPLATE ONLY.
#
# *** DO NOT SUBMIT until BOTH of the following have landed: ***
#   1. PA-HIER-31 is authored and ratified (skeleton:
#      results/campaign51_20260728/realistic_20260729/fanout1_20260829/WAVE2_REGISTRATION_CHECK_20260829.md
#      §2, to be appended to PREREGISTRATION_HIER_HTHETA_20260826.md).
#   2. P6 (theta CLI plumbing) is COMMITTED, not merely implemented. Status at authoring time:
#      results/campaign51_20260728/realistic_20260729/fanout1_20260829/P6_THETA_CLI_PLUMBING_RECORD.md
#      says "IMPLEMENTED, not committed" -- --theta_b/--theta_s/--theta_sites do not exist on
#      the CLI of any committed HEAD as of this template's authoring. This script's theta flags
#      are therefore GUARDED (commented out below) and must be uncommented only once P6 lands
#      and PA-HIER-31 authorizes the launch.
#
# Registration (skeleton, item numbers below refer to it):
#   WAVE2_REGISTRATION_CHECK_20260829.md §2 (PA-HIER-31 skeleton), items 1 (venue/CLI), 2 (theta
#   form / F-A), 4 (theta nodes), 11 (cost).
#
# **Launched under rows #222/#223 — charter nodes C0/B5.2/B7.2/B1.2.** This template is
# delivered under those rows' authorization to BUILD, not to submit; submission additionally
# requires PA-HIER-31 ratification per this header.
#
# theta form (F-A, registration item 2): theta_sites="2.2" (the CoR-P-faithful, unsmeared
# per-host host-z kernel) with smear_global_selection=False (default -- NOT passed). Site 2.3
# is OUT OF SCOPE (requires smear_global_selection=True, which is non-CoR-P and was measured to
# move alpha_G_phi by -12.0% / combined_no_bh by up to 7.45e-3 relative -- WAVE2_REGISTRATION_CHECK
# §0 finding F-A). h is fixed at 0.730 (truth venue node, matching C0/wave2_c0_baseline.sbatch's
# seed=777021 for a PAIRED deterministic comparison against the truth node -- A15 "paired
# recomputation" pattern used throughout wave 2).
#
# theta NODES (registration item 4, 4 array tasks, one per node):
#   task 0: theta_b=+0.033, theta_s=1.0            (b_max=0.0661 re-derived, PA-HIER-29)
#   task 1: theta_b=-0.033, theta_s=1.0
#   task 2: theta_b=0.0,    theta_s=0.7071067811865476  (1/sqrt(2))
#   task 3: theta_b=0.0,    theta_s=1.4142135623730951  (sqrt(2))
#
# F4 COST ESTIMATE: 4 theta-nodes x 14.93-22.9 CPU-h/node = 60-92 CPU-h (unsmeared/"2.2" form
# only; the 81-113 CPU-h smeared band is WITHDRAWN per WAVE2_REGISTRATION_CHECK_20260829.md §0
# F-A / COMPUTE_LEDGER.md row C1 revision). Plus the shared C0 gate task (not counted again).
#
# Required env (set by cluster/submit_wave2.sh via --export=ALL,...; C1 is NOT wired into
# submit_wave2.sh's default submission set -- it is commented out there):
#   RUN_DIR      -- $WORKSPACE/run_20260829_wave2_c1_iiib (out-root name fixed by
#                    results/_archive/archive_run_wave2.sh -- do not rename)
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=wave2-c1-s0b
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

LOG_BASE="$RUN_DIR/logs/wave2_c1_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== wave2-c1-s0b task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
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
# theta nodes (registration item 4). Fixed h=0.730, seed=777021 (paired against C0's truth
# node -- H41 index 21, same as wave2_c0_baseline.sbatch), so only theta differs across tasks.
# ---------------------------------------------------------------------------
H_VALUE="0.73"
H41_IDX=21
EVAL_SEED=777000
TASK_SEED=$((EVAL_SEED + H41_IDX))

THETA_B=(0.033 -0.033 0.0 0.0)
THETA_S=(1.0 1.0 0.7071067811865476 1.4142135623730951)
NODE_LABEL=("bplus" "bminus" "shalf" "sdouble")
TB="${THETA_B[$TID]}"
TS="${THETA_S[$TID]}"
NLABEL="${NODE_LABEL[$TID]}"

# Idempotency: this template writes per-node output under a node-labeled subdirectory rather
# than the standard h_<label>.json (four DIFFERENT theta nodes at the SAME h would otherwise
# collide on one posterior filename) -- skip-if-output guards on that subdirectory instead.
NODE_OUT_DIR="$RUN_DIR/simulations/theta_nodes/$NLABEL"
if [[ -f "$NODE_OUT_DIR/posteriors/h_0_73.json" && -f "$NODE_OUT_DIR/posteriors_with_bh_mass/h_0_73.json" ]]; then
    echo "theta node $NLABEL (task $TID) already has output -- skipping (idempotency guard): $NODE_OUT_DIR"
    exit 0
fi
mkdir -p "$NODE_OUT_DIR"

TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "wave2 C1 S0-B theta-score node=$NLABEL (b=$TB,s=$TS), h=$H_VALUE, launched under rows #222/#223 charter node B1.2 -- PA-HIER-31 + P6 required before submit"

JOB_CWD="$RUN_DIR/cwd_$NLABEL"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== wave2-c1-s0b: task=$TID node=$NLABEL theta_b=$TB theta_s=$TS h=$H_VALUE seed=$TASK_SEED ==="

# ---------------------------------------------------------------------------
# CoR-P CLI verbatim + explicit mass-window defaults (same as C0/wave2_c0_baseline.sbatch),
# PLUS --theta_b/--theta_s/--theta_sites (P6 plumbing) and NO --smear_global_selection
# (defaults False, matching the CoR-P-faithful "2.2"/unsmeared form -- site 2.3 out of scope).
#
# *** GUARDED: uncomment the --theta_* flags ONLY once P6 is committed (see header) and
# PA-HIER-31 is ratified. Until then this template is not runnable as-is (theta_b/theta_s/
# theta_sites are not on any committed CLI, so passing them would fail argparse). ***
# ---------------------------------------------------------------------------
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
    --mass_filter_geometry linear \
    --mass_filter_k 1.5 \
    --log_level INFO
    # --- GUARDED theta flags (uncomment once P6 is committed + PA-HIER-31 ratified), and move
    # the trailing "--log_level INFO" line above to end in "\" so these continue the same
    # invocation:
    #   --theta_b "$TB" \
    #   --theta_s "$TS" \
    #   --theta_sites 2.2
    # (smear_global_selection NOT passed -> defaults False, matching CoR-P / F-A's "2.2"/unsmeared form)

echo "=== done: wave2-c1-s0b task=$TID node=$NLABEL ==="
