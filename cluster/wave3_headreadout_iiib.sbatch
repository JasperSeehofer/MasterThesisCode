#!/usr/bin/env bash
# cluster/wave3_headreadout_iiib.sbatch — wave-3 charter: the ONE blind HEAD readout, iiib venue.
#
# **Launched under rows #222/#223 — charter wave 3 / node B7.3 readout.**
#
# PURPOSE: reproduce, at the wave-3 commit and PRODUCTION DEFAULTS ONLY, the full 41-node H0
# posterior (H_GRID_41, both channels) that the F2 amendment (charter, ratified row #222 with the
# F5 substitution) designates as "the ONE wave-3 blind HEAD readout" — see
# results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md
# §0.2 (F2 statement) and §9 (ledger rows / what this readout is). This job does NOT itself
# adjudicate anything: it produces posteriors/*.json, posteriors_with_bh_mass/*.json, and
# diagnostics/event_likelihoods.csv at RUN_DIR; the delta read (against the per-change
# "--catalogue_numerator_survival_2d off" counterfactual arm, which is a SEPARATE object not built
# by this script — see §8) and the falsifier check against T_mat = 0.008 (§8 of the same document)
# are performed elsewhere, by the orchestrator, after this job and its counterfactual sibling both
# complete.
#
# BLINDNESS (deliberate, do not "fix"): this sbatch passes NO
# "--catalogue_numerator_survival_2d" / "--catalogue_numerator_survival_2d_center" flag at all. At
# the wave-2 commit (ff230621) that default resolves to "off"/"unset" (pre-adoption). Once the
# [PHYSICS] adoption commit for row #223 charter node B7.3 lands (literal default flip "off" ->
# "mz_sel", "unset" -> "eff" — PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §0.3/§9.3), this exact
# same invocation resolves to "mz_sel"/"eff" with zero edits to this script. That is the point of
# F2's "one blind HEAD readout, no per-change attribution from the composed delta"
# (MEASUREMENT_HEAD_READOUT_20260827.md's own registered structural blindness §4.2 is the
# precedent this mirrors). GIT_COMMIT_AT_RUN.txt below records which commit actually ran.
#
# CoR-P CLI verbatim source: headreadout_20260827/iiib/run_metadata_21.json:cli_args (commit
# d04d9dc9, 2026-08-27) PLUS the explicit post-wave-2 defaults stated in the wave-3 task brief:
# --mass_filter_geometry linear --mass_filter_k 1.5 (byte-identical to pre-flag behaviour, cf.
# wave2_c0_baseline.sbatch) and --theta_b 0.0 --theta_s 1.0 --theta_sites all (the theta-hook
# literal-skip identity, now that P6's CLI plumbing has landed — arguments.py:1143-1179).
#
# Grid / seeding: full H_GRID_41 (41 nodes, 0.600-0.860), one h per array task.
# SLURM_ARRAY_TASK_ID doubles as the canonical H41 index (evaluate.sbatch's own convention):
# task 21 = h 0.730, seed 777000 + 21 = 777021 (bit-identical to the banked
# headreadout_20260827/iiib task 21 seed, modulo the code-state difference documented above).
#
# --time=00:45:00 sizing (wave-2 MEASURED anchors on iiib, 16 cpus, this task brief): C0 6:28,
# C3 4:34-4:50, C4 6:10-6:38 per h-point — call it ~7 min at the slow end. 7 min x 1.3 assumed
# packing/contention overhead ~= 9.1 min; 45:00 is a generous ~5x margin over that, well clear of
# gotcha 6 (size against the contended anchor) and gotcha 9 (no partial-progress checkpoint, a
# walltime kill loses the whole task).
#
# Dataset pins (CLAUDE.md pinning rule): CRB set = run_20260729_seed61000 (md5
# 9a1f2a14384a9281c97ca3be312ddaab); reduced galaxy catalogue (md5 c52c13b5cab61f6b3f04bbe202550969).
# STOP-gated below; iiib carries no observed-catalogue realization (that is the joint_r1 sibling
# script only).
#
# Required env (set by cluster/submit_wave3.sh via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260830_wave3_headreadout_iiib (out-root name fixed by
#                    results/_archive/archive_run_wave2.sh's "wave 3" ITEMS block — do not rename)
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=wave3-headreadout-iiib
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --array=0-40
#SBATCH --time=00:45:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${RUN_DIR:?RUN_DIR not set -- submit_wave3.sh must export it}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-40)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

# #SBATCH --output/--error cannot reference $WORKSPACE-derived paths (pragmas are not
# shell-expanded), so redirect once RUN_DIR is known, matching cluster/p3_2d_rhs2.sbatch and the
# wave-2 scripts.
LOG_BASE="$RUN_DIR/logs/wave3_headreadout_iiib_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== wave3-headreadout-iiib task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

# ---------------------------------------------------------------------------
# CRB set wiring -- identical to the HEAD readout (MEASUREMENT_HEAD_READOUT_20260827.md
# STEP 0/STEP 1) and to the wave-2 scripts: symlink this run's simulations/ onto the
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
# Full H_GRID_41 (verbatim from evaluate.sbatch / correspondence_1d.py:H_GRID_41). Task index
# IS the canonical H41 index (evaluate.sbatch's own seeding convention): task 21 = h 0.730.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Provenance stamp (cluster/SKILL.md gotcha #12) -- fail-soft, cheap.
# ---------------------------------------------------------------------------
TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "wave3 blind HEAD readout, venue=iiib, h=$H_VALUE, task=$TID -- BLIND to the 2D-twin adoption (no --catalogue_numerator_survival_2d flag passed); launched under rows #222/#223 charter wave 3 / node B7.3 readout"

# ---------------------------------------------------------------------------
# Private per-run CWD (TC-03), identical pattern to evaluate.sbatch / wave-2 scripts. Shared
# across the 41 array tasks of this RUN_DIR -- safe, because every task writes the exact same
# symlink targets (idempotent `ln -sfn`).
# ---------------------------------------------------------------------------
JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

# ---------------------------------------------------------------------------
# CoR-P CLI verbatim (headreadout_20260827/iiib/run_metadata_21.json:cli_args) PLUS the explicit
# post-wave-2 defaults, MINUS catalogue_numerator_survival_2d/_center (deliberately omitted --
# see the BLINDNESS note above). theta_b/theta_s/theta_sites have a CLI surface as of P6
# (arguments.py:1143-1179); passed explicitly at their literal-skip-identity values so the
# readout states them rather than relying on an implicit default.
# ---------------------------------------------------------------------------
echo "=== wave3-headreadout-iiib: task=$TID h=$H_VALUE seed=$TASK_SEED ==="

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
    --log_level INFO

echo "=== done: wave3-headreadout-iiib task=$TID h=$H_VALUE ==="
