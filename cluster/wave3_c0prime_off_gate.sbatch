#!/usr/bin/env bash
# cluster/wave3_c0prime_off_gate.sbatch — wave-3 charter: the A14 falsifier BASELINE gate
# ("C0'"), both venues, h = 0.730 only.
#
# **Launched under rows #222/#223 — charter wave 3 / node B7.3 readout.**
#
# PURPOSE: certify, at PRODUCTION SCALE and at the wave-3 commit, that passing the
# PRE-ADOPTION flag pair EXPLICITLY --
#   --catalogue_numerator_survival_2d off --catalogue_numerator_survival_2d_center unset
# -- reproduces the banked 2026-08-27 headreadout row bit-for-bit (both venues, h = 0.730,
# task 21) for the 14 diagnostic event_likelihoods.csv columns AND both posterior JSONs
# (md5-identical), per REGISTRATION_C0_BASELINE_GATE_20260829.md §3 (gate band, column set,
# per-arm consumption) / §13 (RESULT RECORD form: bit-diff table + md5 comparison). This is the
# reproduction gate that makes the ALREADY-BANKED 2026-08-27 readouts usable as the pre-adoption
# baseline for the wave-3 blind HEAD readout's A14 falsifier delta read (T_mat = 0.008,
# PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §8) WITHOUT running a second full 82-task
# "--catalogue_numerator_survival_2d off" array across all 41 H_GRID_41 nodes x 2 venues (F4:
# size the falsifier-baseline check against what is actually needed, not the full grid).
#
# PASS (both venues, band satisfied) => the banked headreadout_20260827/{iiib,joint_r1} readouts
# ARE the pre-adoption baseline for the T_mat = 0.008 delta read; no further baseline compute is
# needed.
# FAIL (either venue) => the explicit-off CLI at the wave-3 commit does NOT reproduce the banked
# rows, meaning something in the four post-d04d9dc9 commits (or the wave-3 commit itself) moved
# behaviour even at the flag's own pre-adoption value; the full 82-task
# "--catalogue_numerator_survival_2d off" array (all 41 H_GRID_41 nodes, both venues) becomes
# necessary, and the h=0.730 diff produced by THIS gate is the first artifact used to diagnose
# which commit owns it, before that larger array is launched.
#
# WHY THIS IS NOT REDUNDANT WITH THE WAVE-3 BLIND HEAD READOUT (wave3_headreadout_{iiib,
# joint_r1}.sbatch): those two scripts pass NO catalogue_numerator_survival_2d flag at all (F2
# blindness -- they read whatever the wave-3 commit's default resolves to, i.e. the ADOPTED
# "mz_sel"/"eff" post row-#223). This script passes the flag EXPLICITLY at its PRE-adoption
# value ("off"/"unset") -- a different code path (the explicit-off branch, not the
# default-resolution branch) -- and exists ONLY to certify that explicit path still reproduces
# the 2026-08-27 bytes at the wave-3 commit. The two objects answer different questions and are
# both required before the delta read is trustworthy.
#
# CoR-P CLI verbatim source: identical to wave3_headreadout_{iiib,joint_r1}.sbatch (same
# headreadout_20260827/<venue>/run_metadata_21.json:cli_args + the same explicit post-wave-2
# defaults: --mass_filter_geometry linear --mass_filter_k 1.5, --theta_b 0.0 --theta_s 1.0
# --theta_sites all) PLUS the explicit pre-adoption pair
# --catalogue_numerator_survival_2d off --catalogue_numerator_survival_2d_center unset (this is
# the one deliberate CLI difference from the two blind-readout scripts; "off"/"unset" is a valid
# explicit combination -- the mz_sel+unset guard in arguments.py:522-527 does not fire for
# "off").
#
# Array shape: 2 tasks, NOT 2 x 41. Task 0 = iiib, task 1 = joint_r1. Each task runs ONE h-point
# (h = 0.730, canonical H41 index 21, seed 777000 + 21 = 777021) -- this gate needs only the
# single node the wave-3 blind readout's falsifier check anchors on (REGISTRATION_C0_BASELINE_
# GATE_20260829.md's own C0 gate was likewise single-h; this is the "C0-prime" / off-arm sibling
# of that same gate, same single-h economy, applied to the pre-adoption flag value instead of
# the post-d40fe5c8 commit set). RUN_DIR is computed from TID, not passed in -- a plain sbatch
# array shares one #SBATCH block (and one --export=ALL environment) across all tasks, so the two
# venues' distinct out-roots and distinct CoR-P CLIs (joint_r1 adds --observed_catalogue) are
# selected by a case-on-TID below rather than by two separate submissions' worth of exported
# RUN_DIR values.
#
# --time=01:30:00 (builder deviation, documented): the task brief calls for
# "--time=00:45:00 / 01:30:00 per venue" matching the two blind-readout scripts' own per-script
# --time values, but #SBATCH --time is ONE directive for the whole array in plain sbatch (no
# per-task-index walltime in vanilla SLURM) -- so this script uses the LARGER of the two
# venue-specific times (joint_r1's 01:30:00) as a uniform ceiling for both tasks. iiib (task 0)
# is measured at ~7 min/h-point (wave-2 C0 anchor, RESULT RECORD §13: 00:06:28 elapsed) and so
# finishes at a small fraction of the shared budget; the ceiling costs nothing but unused
# walltime headroom, not correctness. See wave3_headreadout_iiib.sbatch / _joint_r1.sbatch
# headers for the per-venue anchor derivations this ceiling is built from.
#
# Dataset pins (CLAUDE.md pinning rule) -- identical to the two blind-readout scripts: CRB set
# run_20260729_seed61000 (md5 9a1f2a14384a9281c97ca3be312ddaab); reduced galaxy catalogue (md5
# c52c13b5cab61f6b3f04bbe202550969); joint_r1 additionally pins the observed-catalogue
# realization (sha256 e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751). All
# STOP-gated below, per-task (a task only gates the pins its own venue actually consumes).
#
# Required env (set by cluster/submit_wave3.sh via --export=ALL,...):
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)
# WORKSPACE is resolved by cluster/modules.sh (ws_find emri); RUN_DIR is NOT an input -- it is
# computed below from WORKSPACE + the per-task venue name (out-root names fixed by
# results/_archive/archive_run_wave2.sh's "wave 3" ITEMS block -- do not rename):
#   task 0 (iiib):     $WORKSPACE/run_20260830_wave3_c0prime_off_iiib
#   task 1 (joint_r1): $WORKSPACE/run_20260830_wave3_c0prime_off_joint_r1

#SBATCH --job-name=wave3-c0prime-off-gate
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

RUN_DIR="$WORKSPACE/run_20260830_wave3_c0prime_off_${VENUE}"
mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

# #SBATCH --output/--error cannot reference $WORKSPACE-derived paths (pragmas are not
# shell-expanded), so redirect once RUN_DIR is known, matching the wave-3 headreadout scripts.
LOG_BASE="$RUN_DIR/logs/wave3_c0prime_off_gate_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== wave3-c0prime-off-gate task=$TID venue=$VENUE job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

# ---------------------------------------------------------------------------
# CRB set wiring -- identical to both wave-3 headreadout scripts and the banked HEAD readout.
# ---------------------------------------------------------------------------
CRB_SRC="$WORKSPACE/run_20260729_seed61000"
ln -sfn "$CRB_SRC/simulations/cramer_rao_bounds.csv"          "$RUN_DIR/simulations/cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/injections"                     "$RUN_DIR/simulations/injections"

# ---------------------------------------------------------------------------
# Dataset pinning STOP-gate (CLAUDE.md: any multi-GB input not in version control carries a
# checksum pin at each consumer, STOP-gated on mismatch). joint_r1 carries one extra pin: the
# observed-catalogue realization's sha256.
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

# ---------------------------------------------------------------------------
# Single h-point: h = 0.730 only, canonical H41 index 21 (evaluate.sbatch's own convention),
# seed 777000 + 21 = 777021 -- bit-identical to the banked headreadout_20260827/<venue> task 21
# seed, modulo the code-state difference this gate exists to certify away.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Provenance stamp (cluster/SKILL.md gotcha #12) -- fail-soft, cheap.
# ---------------------------------------------------------------------------
TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "wave3 A14 falsifier BASELINE gate (C0'), venue=$VENUE, h=$H_VALUE, task=$TID -- explicit --catalogue_numerator_survival_2d off/_center unset at the wave-3 commit, certifying the banked 2026-08-27 readouts as the pre-adoption baseline; launched under rows #222/#223 charter wave 3 / node B7.3 readout"

# ---------------------------------------------------------------------------
# Private per-run CWD (TC-03), identical pattern to evaluate.sbatch / the wave-3 headreadout
# scripts.
# ---------------------------------------------------------------------------
JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

# ---------------------------------------------------------------------------
# CoR-P CLI verbatim (headreadout_20260827/<venue>/run_metadata_21.json:cli_args) PLUS the
# explicit post-wave-2 defaults, PLUS the explicit pre-adoption
# --catalogue_numerator_survival_2d off --catalogue_numerator_survival_2d_center unset pair (the
# one deliberate difference from the wave-3 blind-readout scripts -- see the header note above).
# joint_r1 additionally gets --observed_catalogue.
# ---------------------------------------------------------------------------
echo "=== wave3-c0prime-off-gate: task=$TID venue=$VENUE h=$H_VALUE seed=$TASK_SEED ==="

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

echo "=== done: wave3-c0prime-off-gate task=$TID venue=$VENUE h=$H_VALUE ==="
