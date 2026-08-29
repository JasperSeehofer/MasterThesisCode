#!/usr/bin/env bash
# cluster/wave3_headreadout_joint_r1.sbatch — wave-3 charter: the ONE blind HEAD readout,
# joint_r1 venue (observed-catalogue realization).
#
# **Launched under rows #222/#223 — charter wave 3 / node B7.3 readout.**
#
# PURPOSE: same object as wave3_headreadout_iiib.sbatch (see that script's header for the full
# rationale) run against the joint_r1 venue: the reduced GLADE+ catalogue replaced, for host
# lookups, by an OBSERVED-CATALOGUE REALIZATION (--observed_catalogue), matching the
# headreadout_20260827/joint_r1 run exactly except for the code-state difference this readout is
# designed to surface. Registration:
# results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md
# §0.2 (F2) / §9. This job produces posteriors/*.json, posteriors_with_bh_mass/*.json, and
# diagnostics/event_likelihoods.csv at RUN_DIR only; the per-change delta read and the T_mat =
# 0.008 falsifier check (§8) happen elsewhere, against a separate counterfactual-arm run this
# script does not build.
#
# BLINDNESS (deliberate, do not "fix"): identical to the iiib sibling -- no
# "--catalogue_numerator_survival_2d" / "--catalogue_numerator_survival_2d_center" flag is passed.
# GIT_COMMIT_AT_RUN.txt records which commit actually ran; the default resolves per that commit's
# own code (pre-adoption "off"/"unset", post-adoption "mz_sel"/"eff" -- F2, no per-change
# attribution from the composed delta).
#
# CoR-P CLI verbatim source: headreadout_20260827/joint_r1/run_metadata_21.json:cli_args (commit
# d04d9dc9, 2026-08-27) -- identical to the iiib sibling's cli_args in every field except
# "observed_catalogue" -- PLUS the same explicit post-wave-2 defaults (mass_filter_geometry
# linear/k=1.5, theta_b/theta_s/theta_sites literal-skip identity) and PLUS
# "--observed_catalogue" pointing at the realization used by that same banked run.
#
# Observed-catalogue realization + dataset pins (CLAUDE.md pinning rule; cluster/datasets.yaml
# entry "run_20260827_headreadout_joint_r1"):
#   path:   $WORKSPACE/realizations_20260729/observed_catalogue_seed900001.csv
#   sha256: e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751
# STOP-gated below, on top of the shared CRB/catalogue md5 gates.
#
# Realization-sidecar gotcha (cluster/SKILL.md gotcha #10): the observed-catalogue CSV's
# ".meta.json" sidecar records "parent_csv" as an ABSOLUTE path, which goes stale if the repo
# checkout ever moves on the cluster; production code reads that sidecar and hash-verifies it
# before using the realization (evaluate.sbatch's OBSERVED_CATALOGUE comment: "its .meta.json
# sidecar is read and hash-verified"), so a stale sidecar FAILS the job rather than silently
# mis-scoring it -- but the fix (rewrite "parent_csv", keep a ".bak") requires cluster access this
# builder pass does not have. The pre-launch checklist in cluster/submit_wave3.sh calls this out
# explicitly; verify/repair the sidecar BEFORE flipping DRY_RUN=0 for this script.
#
# Grid / seeding / --time sizing: identical H_GRID_41 / EVAL_SEED=777000+task-index convention as
# the iiib sibling. --time=01:30:00: joint_r1 is HISTORICALLY >= 2.2x the iiib per-task anchor
# (cluster/datasets.yaml / MEASUREMENT_HEAD_READOUT_20260827.md §9/§F); iiib's own ~7 min x 1.3
# overhead anchor x a 2.2-3x joint_r1 multiplier puts the expected slow end at roughly
# 9.1 min x 3 ~= 27.3 min; 01:30:00 keeps a >3x margin over that, well inside gotcha 6/9's
# "size against the contended anchor, no partial-progress checkpoint" convention.
#
# Required env (set by cluster/submit_wave3.sh via --export=ALL,...):
#   RUN_DIR      -- $WORKSPACE/run_20260830_wave3_headreadout_joint_r1 (out-root name fixed by
#                    results/_archive/archive_run_wave2.sh's "wave 3" ITEMS block -- do not rename)
#   PROJECT_ROOT -- (optional; falls back to $HOME/darksiren-emri)

#SBATCH --job-name=wave3-headreadout-joint_r1
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

: "${RUN_DIR:?RUN_DIR not set -- submit_wave3.sh must export it}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-40)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/wave3_headreadout_joint_r1_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== wave3-headreadout-joint_r1 task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

# ---------------------------------------------------------------------------
# CRB set wiring -- identical to the iiib sibling and to the banked HEAD readout.
# ---------------------------------------------------------------------------
CRB_SRC="$WORKSPACE/run_20260729_seed61000"
ln -sfn "$CRB_SRC/simulations/cramer_rao_bounds.csv"          "$RUN_DIR/simulations/cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/injections"                     "$RUN_DIR/simulations/injections"

# ---------------------------------------------------------------------------
# Dataset pinning STOP-gates (CLAUDE.md: any multi-GB input not in version control carries a
# checksum pin at each consumer, STOP-gated on mismatch). joint_r1 carries one extra pin: the
# observed-catalogue realization's sha256.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Full H_GRID_41 (verbatim from evaluate.sbatch / correspondence_1d.py:H_GRID_41). Task index
# IS the canonical H41 index: task 21 = h 0.730.
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
write_provenance "$RUN_DIR/logs" "wave3 blind HEAD readout, venue=joint_r1, h=$H_VALUE, task=$TID -- BLIND to the 2D-twin adoption (no --catalogue_numerator_survival_2d flag passed); launched under rows #222/#223 charter wave 3 / node B7.3 readout"

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
# CoR-P CLI verbatim (headreadout_20260827/joint_r1/run_metadata_21.json:cli_args) PLUS the
# explicit post-wave-2 defaults, PLUS --observed_catalogue, MINUS
# catalogue_numerator_survival_2d/_center (deliberately omitted -- see the BLINDNESS note above).
# ---------------------------------------------------------------------------
echo "=== wave3-headreadout-joint_r1: task=$TID h=$H_VALUE seed=$TASK_SEED observed_catalogue=$OBS_CATALOGUE ==="

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

echo "=== done: wave3-headreadout-joint_r1 task=$TID h=$H_VALUE ==="
