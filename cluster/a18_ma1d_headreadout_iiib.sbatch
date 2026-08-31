#!/usr/bin/env bash
# cluster/a18_ma1d_headreadout_iiib.sbatch — A18 production arm: mass-aware 1D catalogue leg,
# iiib venue, EXTENDED 55-node grid.
#
# **Launched under rows #278/#282/#284 — A18 ratified; AMENDMENT G-EXT 55-node grid; measured
# band MAP 0.66 [0.65, 0.67] per BAND_REDERIVATION_20260831.md; flip rule reads MAP AND mean_h
# per the gate doc section 6.3.**
#
# This is a copy-adaptation of cluster/wave3_headreadout_iiib.sbatch (the wave-3 blind HEAD
# readout template) with EXACTLY the following changes, and no others:
#   (a) H_VALUES extended from the template's H_GRID_41 (41 nodes, 0.600-0.860) by appending
#       0.870 0.880 0.890 0.900 0.910 0.920 0.930 0.940 0.950 0.960 0.970 0.980 0.990 1.000
#       (14 nodes at 0.010 spacing) -- 55 nodes total, array 0-54. Task-to-h mapping mechanism
#       (SLURM_ARRAY_TASK_ID as the direct grid index) is unchanged, only extended in range. See
#       AMENDMENT G-EXT (results/campaign51_20260728/realistic_20260729/tree2_20260830/
#       PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md, appended 2026-08-31 under row #284): the
#       on-arm matched-class read railed at the H_GRID_41 ceiling (0.86) in 4/4 mirror seeds;
#       without the extension the production arm's most important diagnostic sub-read would be
#       censored by construction.
#   (b) --catalogue_leg_1d_mass_aware on added to the python invocation. Every other flag is
#       byte-identical to the template, INCLUDING the blind-to-adoption convention: no
#       --catalogue_numerator_survival_2d / --catalogue_numerator_survival_2d_center flag is
#       passed (absent in the template, absent here) -- the row #223 adopted default
#       ("mz_sel"/"eff") now applies at HEAD and that is intended, not an oversight.
#       --catalogue_leg_1d_mass_aware="on" requires catalogue_global_selection to resolve to
#       "phi" (already passed below, unchanged) and theta_phi_divisor="off" (the unchanged
#       default, not passed) -- see arguments.py:1334-1350 and bayesian_statistics.py:4020-4256.
#   (c) out-root renamed run_20260831_a18_ma1d_iiib everywhere the template names its out-root
#       (was run_20260830_wave3_headreadout_iiib).
#   (d) job name and log-base names updated to a18_ma1d (were wave3-headreadout-iiib /
#       wave3_headreadout_iiib).
#   (e) dataset-pin STOP-gate, write_provenance line, module/venv/symlink dance, --time and
#       --partition are EXACTLY as the template (same per-task walltime is correct: one h-point
#       per task, same per-task cost model).
#
# PURPOSE: reproduce, at the A18 (mass-aware 1D catalogue leg, "on") code path, the full 55-node
# H0 posterior (H_GRID_41 + G-EXT wing, both channels) that is the production arm of record for
# the A18 flip decision (map_h AND mean_h against the measured band MAP 0.66 [0.65, 0.67],
# mean_h 0.652-0.673 -- BAND_REDERIVATION_20260831.md §4, per row #284). This job does NOT itself
# adjudicate the flip: it produces posteriors/*.json, posteriors_with_bh_mass/*.json, and
# diagnostics/event_likelihoods.csv at RUN_DIR; the flip-rule read is performed elsewhere, by the
# orchestrator, after this job completes. Scorer: the frozen T0 gradient-weighted convention (see
# the 2026-08-31 correction note in WAVE3_A14_DELTA_READ_20260831.md).
#
# CoR-P CLI verbatim source: cluster/wave3_headreadout_iiib.sbatch (headreadout_20260827/iiib/
# run_metadata_21.json:cli_args, commit d04d9dc9, 2026-08-27) PLUS --catalogue_leg_1d_mass_aware
# on (this arm's own instrument flag, row #255 tree 2 node T2.3).
#
# Grid / seeding: H_GRID_41 (41 nodes, 0.600-0.860) + G-EXT wing (14 nodes, 0.870-1.000) = 55
# nodes, one h per array task. SLURM_ARRAY_TASK_ID doubles as the grid index (evaluate.sbatch's
# own convention, extended): task 21 = h 0.730, seed 777000 + 21 = 777021 (bit-identical to the
# wave-3 template's task-21 seed for the shared low-wing nodes); tasks 41-54 are new nodes with
# seeds 777041-777054.
#
# --time=00:45:00 sizing: unchanged from the template (same per-task walltime is correct -- one
# h-point per task, same per-task cost model; AMENDMENT G-EXT's added cost is purely from the 14
# extra array tasks, not from any per-task slowdown).
#
# Dataset pins (CLAUDE.md pinning rule): CRB set = run_20260729_seed61000 (md5
# 9a1f2a14384a9281c97ca3be312ddaab); reduced galaxy catalogue (md5 c52c13b5cab61f6b3f04bbe202550969).
# STOP-gated below; iiib carries no observed-catalogue realization (unchanged from the template).
#
# Required env (set by cluster/submit_a18.sh via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260831_a18_ma1d_iiib
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=a18-ma1d-headreadout-iiib
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-54
#SBATCH --time=00:45:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set -- submit_a18.sh must export it}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-54)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

# #SBATCH --output/--error cannot reference $WORKSPACE-derived paths (pragmas are not
# shell-expanded), so redirect once RUN_DIR is known, matching cluster/p3_2d_rhs2.sbatch and the
# wave-3 scripts.
LOG_BASE="$RUN_DIR/logs/a18_ma1d_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== a18-ma1d-headreadout-iiib task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

# ---------------------------------------------------------------------------
# CRB set wiring -- identical to the HEAD readout (MEASUREMENT_HEAD_READOUT_20260827.md
# STEP 0/STEP 1) and to the wave-3 scripts: symlink this run's simulations/ onto the
# seed61000 prepared CRB set.
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
# Extended grid: H_GRID_41 (verbatim from evaluate.sbatch / correspondence_1d.py:H_GRID_41)
# PLUS the AMENDMENT G-EXT wing (14 nodes, 0.870-1.000 at 0.010 spacing) -- 55 nodes total.
# Task index IS the grid index (evaluate.sbatch's own seeding convention, extended in range).
# ---------------------------------------------------------------------------
H_VALUES=(0.600 0.610 0.620 0.630 0.640 0.650 0.655 0.660 0.665 0.670 0.675 0.680 0.685 0.690 0.695 0.700 0.705 0.710 0.715 0.720 0.725 0.730 0.735 0.740 0.745 0.750 0.755 0.760 0.765 0.770 0.775 0.780 0.785 0.790 0.800 0.810 0.820 0.830 0.840 0.850 0.860 0.870 0.880 0.890 0.900 0.910 0.920 0.930 0.940 0.950 0.960 0.970 0.980 0.990 1.000)
if [[ "$TID" -ge "${#H_VALUES[@]}" ]]; then
    echo "STOP: array task id $TID out of range for the extended 55-node grid (${#H_VALUES[@]} nodes)" >&2
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

# ---------------------------------------------------------------------------
# Provenance stamp (cluster/SKILL.md gotcha #12) -- fail-soft, cheap.
# ---------------------------------------------------------------------------
TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "A18 production arm, mass-aware 1D catalogue leg ON, venue=iiib, h=$H_VALUE, task=$TID -- extended 55-node grid (AMENDMENT G-EXT); launched under rows #278/#282/#284"

# ---------------------------------------------------------------------------
# Private per-run CWD (TC-03), identical pattern to evaluate.sbatch / wave-3 scripts. Shared
# across the 55 array tasks of this RUN_DIR -- safe, because every task writes the exact same
# symlink targets (idempotent `ln -sfn`).
# ---------------------------------------------------------------------------
JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

# ---------------------------------------------------------------------------
# CoR-P CLI verbatim (headreadout_20260827/iiib/run_metadata_21.json:cli_args via the wave-3
# template) PLUS --catalogue_leg_1d_mass_aware on (A18's own instrument flag), MINUS
# catalogue_numerator_survival_2d/_center (deliberately omitted, unchanged from the template --
# see the header note above: the row #223 adopted default now applies at HEAD, intentionally).
# theta_b/theta_s/theta_sites passed explicitly at their literal-skip-identity values, matching
# the template.
# ---------------------------------------------------------------------------
echo "=== a18-ma1d-headreadout-iiib: task=$TID h=$H_VALUE seed=$TASK_SEED ==="

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
    --catalogue_leg_1d_mass_aware on \
    --log_level INFO

echo "=== done: a18-ma1d-headreadout-iiib task=$TID h=$H_VALUE ==="
