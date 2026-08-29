#!/usr/bin/env bash
# cluster/wave2_c0_baseline.sbatch — wave-2 charter node C0: shared baseline gate task
# (serves C1/C3/C4). Registration:
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_C0_BASELINE_GATE_20260829.md
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/WAVE2_REGISTRATION_CHECK_20260829.md §1.1
#
# **Launched under rows #222/#223 — charter nodes C0/B5.2/B7.2/B1.2.**
#
# PURPOSE: reproduce, at production scale and default flag values, the four estimator commits
# landed since the banked HEAD readout (d40fe5c8, 1f003da6, 0b308828, 901653a1) at venue iiib,
# h=0.730 only (the `evaluate.sbatch --array=21` node). PASS (all 18 numeric event_likelihoods.csv
# columns + posteriors <= 1e-12 relative vs the banked d04d9dc9 rows) makes the banked HEAD
# readout the zero-compute baseline for C1 (truth node)/C3/C4. This job does NOT itself perform
# the diff — it produces the comparison inputs (event_likelihoods.csv, posteriors/*.json,
# posteriors_with_bh_mass/*.json) at RUN_DIR; the column diff is a separate, local, zero-compute
# read against results/campaign51_20260728/realistic_20260729/headreadout_20260827/iiib/.
#
# F4 COST ESTIMATE: 15-23 CPU-h (1 task x 16 cpus x up to ~1:25:52 contended anchor from the
# HEAD-readout off-arm slowest task; production per-h-point anchor 14.93-22.9 CPU-h,
# cluster/LAUNCHING_JOBS.md:47 / MEASUREMENT_HEAD_READOUT_20260827.md). Single task, no packing.
#
# Required env (set by cluster/submit_wave2.sh via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260829_wave2_c0_iiib (out-root name fixed by
#                    results/_archive/archive_run_wave2.sh — do not rename)
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=wave2-c0-baseline
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set -- submit_wave2.sh must export it}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

# #SBATCH --output/--error cannot reference $WORKSPACE-derived paths (pragmas are not
# shell-expanded), so redirect once RUN_DIR is known, matching cluster/p3_2d_rhs2.sbatch.
LOG_BASE="$RUN_DIR/logs/wave2_c0_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== wave2-c0-baseline job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

# ---------------------------------------------------------------------------
# CRB set wiring -- identical to the HEAD readout (MEASUREMENT_HEAD_READOUT_20260827.md
# STEP 0/STEP 1): symlink this run's simulations/ onto the seed61000 prepared CRB set.
# ---------------------------------------------------------------------------
CRB_SRC="$WORKSPACE/run_20260729_seed61000"
ln -sfn "$CRB_SRC/simulations/cramer_rao_bounds.csv"          "$RUN_DIR/simulations/cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/injections"                     "$RUN_DIR/simulations/injections"

# ---------------------------------------------------------------------------
# Dataset pinning STOP-gate (CLAUDE.md: any multi-GB input not in version control carries a
# checksum pin at each consumer, STOP-gated on mismatch).
# ---------------------------------------------------------------------------
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
# h / seed: this arm is a SINGLE node, h=0.730, matching the banked headreadout_20260827
# array task 21 (run_metadata_21.json: random_seed=777021). Seed MUST match for a
# bit-reproduction gate.
# ---------------------------------------------------------------------------
H_VALUE="0.73"
TASK_SEED=777021
TASK_INDEX=21

H_LABEL=$(echo "$H_VALUE" | sed -E 's/0+$//; s/\.$//; s/\./_/')
POSTERIOR_JSON="$RUN_DIR/simulations/posteriors/h_${H_LABEL}.json"
POSTERIOR_BH_JSON="$RUN_DIR/simulations/posteriors_with_bh_mass/h_${H_LABEL}.json"
if [[ -f "$POSTERIOR_JSON" && -f "$POSTERIOR_BH_JSON" ]]; then
    echo "Posteriors for h=$H_VALUE already exist -- skipping (idempotency guard):"
    echo "  $POSTERIOR_JSON"
    echo "  $POSTERIOR_BH_JSON"
    exit 0
fi

# ---------------------------------------------------------------------------
# Provenance stamp (cluster/SKILL.md gotcha #12) -- fail-soft, cheap.
# ---------------------------------------------------------------------------
TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "wave2 C0 baseline gate, h=$H_VALUE, launched under rows #222/#223 charter node C0"

# ---------------------------------------------------------------------------
# Private per-run CWD (TC-03), identical pattern to evaluate.sbatch.
# ---------------------------------------------------------------------------
JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

# ---------------------------------------------------------------------------
# CoR-P CLI verbatim (headreadout_20260827/iiib/run_metadata_21.json:cli_args) PLUS the
# explicit new-commit defaults (byte-identical to pre-flag behaviour, but stated explicitly
# per REGISTRATION_C0_BASELINE_GATE_20260829.md §2): --mass_filter_geometry linear
# --mass_filter_k 1.5. theta_b/theta_s/theta_sites have no CLI flag on this commit (F-C in the
# registration check) -- they apply as the code's own defaults (0.0, 1.0, "all") regardless.
# ---------------------------------------------------------------------------
echo "=== wave2-c0-baseline: h=$H_VALUE seed=$TASK_SEED task_index=$TASK_INDEX ==="

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
    --catalogue_numerator_survival_2d off \
    --catalogue_global_selection phi \
    --mass_filter_geometry linear \
    --mass_filter_k 1.5 \
    --log_level INFO

echo "=== done: wave2-c0-baseline h=$H_VALUE ==="
