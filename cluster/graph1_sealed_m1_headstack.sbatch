#!/usr/bin/env bash
# cluster/graph1_sealed_m1_headstack.sbatch — Research Graph 1 ADDENDUM, Branch M [SEAL],
# node m-closure067-headstack, stage (m1): the EXISTING, UNSEALED 0.67 closure pool
# (run_20260729_seed64000_h0p67, truth h=0.67, jobs 6090909-6090912, commit 7b30d1f) re-scored
# on the CURRENT production default (post-flip, post-A14) over the full H_GRID_41.
#
# Registration of record: results/campaign51_20260728/realistic_20260729/graph1_20260901/
#   exec/r-sealed-mock/REGISTRATION_DRAFT.md (sec 6-8) + D_SEALED_REGISTER_DOSSIER.md (sec 2).
# Authorization: row #349 (batch-2 grant) item (2) "sealed-mock stage m1 -- HEAD re-score of
#   the existing closure_seed64000_h0p67 pool ... (52-72 core-h/venue, cap 75/venue), behind a
#   d-sealed-register dossier ratified-by-chair (flagged)"; addendum sec 3 row A-M2. The
#   docket-2.2 STANDING requires a GREEN design gate before this is submitted (dossier sec 4).
#
# This is NOT the sealed T-1 verdict (the truth 0.67 is public); it is the binary TUNED vs
# NOT-TUNED mechanism check on the post-flip stack. The TRANSFER/UNBIASED split is NOT
# evaluable on H_GRID_41 (draft sec 7) -- register only the binary read.
#
# CLI: VERBATIM cluster/graph1_headrebaseline_iiib.sbatch (the registered CoR-P production set,
# = REGISTERED_RESOLVED_FLAGS in darksiren_emri/validation/correspondence_1d.py), BLIND to the
# post-flip flags exactly as that script (no --catalogue_numerator_survival_2d*, no
# --catalogue_leg_1d_mass_aware -> 'auto' resolves 'on' post-flip; run_metadata_*.json records
# the resolved values). Array shape copied: 41 tasks x 16 cores x ~6 min; task N = H41 index N;
# seed EVAL_SEED 777000 + N (gotcha 4, same convention as the re-baseline).
#
# DATA THIS JOB READS (all on the cluster; nothing exists locally except posteriors):
#   CRB:   $WORKSPACE/run_20260729_seed64000_h0p67/simulations/{prepared_,}cramer_rao_bounds.csv
#          md5 pin = EXPECTED_CRB67_MD5, taken at FIRST TOUCH by the submitter (dossier sec 2/5;
#          no md5 of this file exists in any record) -- STOP if unset or mismatched.
#   pool:  $WORKSPACE/injection_pool_mix200k_20260728 (PIN CORRECTION, chair 2026-09-04: the
#          registration's expectation of injection_pool_depth15_50k was a factual error; the
#          0.67 run's own injections link resolves to this pool -- 707 files, measured at
#          first touch), STOP if the file count != 707, and STOP if
#          the 0.67 run dir's own simulations/injections link resolves elsewhere.
#   cat.:  darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv md5 c52c13b5...
#
# VENUE switch: VENUE=iiib (default) | VENUE=joint_r1 (adds --observed_catalogue with the
# realization's sha256 pin + sidecar check, verbatim graph1_headrebaseline_joint_r1.sbatch;
# submit with --time=01:30:00). joint_r1 is submitted ONLY if the iiib array's sacct cost
# lands <= 60 core-h (dossier sec 1 item 6).
#
# HEAD pin: ancestor check on 081b1f28 (row #331 pattern) + darksiren_emri/ diff-quiet vs
# 081b1f28 (physics freeze; the STRICT-equality form is reserved for the sealed (m2) run).
# GIT_COMMIT_AT_RUN.txt records H_code.
#
# Required env (set by the submitter via --export=ALL,...):
#   RUN_DIR            -- $WORKSPACE/graph1_sealed_m1_iiib_20260904  (or ..._joint_r1_20260904)
#   EXPECTED_CRB67_MD5 -- md5 of the 0.67 pool's prepared_cramer_rao_bounds.csv (first-touch pin)
#   VENUE              -- optional, iiib (default) | joint_r1
#   PROJECT_ROOT       -- optional; falls back to $HOME/darksiren-emri

#SBATCH --job-name=graph1-sealed-m1-headstack
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

: "${RUN_DIR:?RUN_DIR not set}"
: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"
: "${EXPECTED_CRB67_MD5:?EXPECTED_CRB67_MD5 not set -- take the first-touch pin (dossier sec 2) and pass it on --export}"
VENUE="${VENUE:-iiib}"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-40)}"

mkdir -p "$RUN_DIR/simulations" "$RUN_DIR/logs"

LOG_BASE="$RUN_DIR/logs/graph1_sealed_m1_${VENUE}_task${TID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

EXPECTED_COMMIT="081b1f28"
ACTUAL_COMMIT="$(git rev-parse --short=8 HEAD)"
if ! git merge-base --is-ancestor "$EXPECTED_COMMIT" HEAD; then
    echo "STOP: HEAD ($ACTUAL_COMMIT) does not contain required commit $EXPECTED_COMMIT -- re-sync before running" >&2
    exit 1
fi
if ! git diff --quiet "$EXPECTED_COMMIT" HEAD -- darksiren_emri/; then
    echo "STOP: darksiren_emri/ differs from $EXPECTED_COMMIT -- the physics freeze of the batch-2 registration is void; do not run" >&2
    git diff --stat "$EXPECTED_COMMIT" HEAD -- darksiren_emri/ >&2
    exit 1
fi

echo "=== graph1-sealed-m1-headstack venue=$VENUE task=$TID job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$(git rev-parse HEAD) ==="
git rev-parse HEAD > "$RUN_DIR/GIT_COMMIT_AT_RUN.txt"

# --- data staging: the 0.67 pool's CRB + the canonical 0.73 p_det pool ----------------------
CRB_SRC="$WORKSPACE/run_20260729_seed64000_h0p67"
POOL="$WORKSPACE/injection_pool_mix200k_20260728"

for f in cramer_rao_bounds.csv prepared_cramer_rao_bounds.csv; do
    if [[ ! -f "$CRB_SRC/simulations/$f" ]]; then
        echo "STOP: $CRB_SRC/simulations/$f not found (if only cramer_rao_bounds.csv exists, prepare_detections.py was never run on this pool -- report, do not improvise)" >&2
        exit 1
    fi
done
ln -sfn "$CRB_SRC/simulations/cramer_rao_bounds.csv"          "$RUN_DIR/simulations/cramer_rao_bounds.csv"
ln -sfn "$CRB_SRC/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv"

if [[ ! -d "$POOL" ]]; then
    echo "STOP: canonical injection pool not found: $POOL" >&2
    exit 1
fi
POOL_COUNT=$(ls "$POOL"/injection_h_*.csv 2>/dev/null | wc -l)
if [[ "$POOL_COUNT" -ne 707 ]]; then
    echo "STOP: injection pool file count $POOL_COUNT != 707 (PIN CORRECTION, chair 2026-09-04: injection_pool_mix200k_20260728 measured count)" >&2
    exit 1
fi
if [[ -e "$CRB_SRC/simulations/injections" ]]; then
    RUN67_POOL="$(readlink -f "$CRB_SRC/simulations/injections")"
    if [[ "$RUN67_POOL" != "$(readlink -f "$POOL")" ]]; then
        echo "STOP: the 0.67 run's own injections link resolves to $RUN67_POOL, not the canonical pool $POOL -- pool mismatch, report" >&2
        exit 1
    fi
fi
ln -sfn "$POOL" "$RUN_DIR/simulations/injections"

EXPECTED_CATALOGUE_MD5="c52c13b5cab61f6b3f04bbe202550969"
ACTUAL_CRB_MD5=$(md5sum "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv" | awk '{print $1}')
if [[ "$ACTUAL_CRB_MD5" != "$EXPECTED_CRB67_MD5" ]]; then
    echo "STOP: 0.67 prepared_cramer_rao_bounds.csv md5 mismatch (expected $EXPECTED_CRB67_MD5, got $ACTUAL_CRB_MD5)" >&2
    exit 1
fi
ACTUAL_CATALOGUE_MD5=$(md5sum "$PROJECT_ROOT/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv" | awk '{print $1}')
if [[ "$ACTUAL_CATALOGUE_MD5" != "$EXPECTED_CATALOGUE_MD5" ]]; then
    echo "STOP: reduced_galaxy_catalogue.csv md5 mismatch (expected $EXPECTED_CATALOGUE_MD5, got $ACTUAL_CATALOGUE_MD5)" >&2
    exit 1
fi
CRB_ROWS=$(( $(wc -l < "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv") - 1 ))
echo "dataset pins OK: CRB67=$ACTUAL_CRB_MD5 rows=$CRB_ROWS (expected 1343; g-population read-time disclosure) catalogue=$ACTUAL_CATALOGUE_MD5 pool=$POOL ($POOL_COUNT files)"

OBS_ARGS=()
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
    # gotcha 10: the sidecar's parent_csv must resolve (verified by the head-rebaseline launch, re-checked here)
    SIDECAR="${OBS_CATALOGUE%.csv}.meta.json"
    if [[ -f "$SIDECAR" ]]; then
        PARENT_CSV=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['parent_csv'])" "$SIDECAR")
        if [[ ! -f "$PARENT_CSV" ]]; then
            echo "STOP: realization sidecar parent_csv does not resolve: $PARENT_CSV (cluster/SKILL.md gotcha 10)" >&2
            exit 1
        fi
    fi
    OBS_ARGS=(--observed_catalogue "$OBS_CATALOGUE")
    echo "observed_catalogue pin OK: $ACTUAL_OBS_CATALOGUE_SHA256"
elif [[ "$VENUE" != "iiib" ]]; then
    echo "STOP: VENUE must be iiib or joint_r1 (got $VENUE)" >&2
    exit 1
fi

# --- H_GRID_41, verbatim graph1_headrebaseline_iiib.sbatch ---------------------------------
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

TASK_SEED="$TASK_SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$RUN_DIR/logs" "graph1 sealed-mock stage (m1) m-closure067-headstack: HEAD re-score of the UNSEALED 0.67 pool run_20260729_seed64000_h0p67 (CRB md5 $ACTUAL_CRB_MD5, rows $CRB_ROWS), venue=$VENUE, h=$H_VALUE, task=$TID -- production CoR-P CLI verbatim, BLIND to post-flip flags (catalogue_leg_1d_mass_aware left at 'auto'); row #349 item (2), addendum A-M2, dossier D_SEALED_REGISTER_DOSSIER.md"

JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/darksiren_emri" "$JOB_CWD/darksiren_emri"
cd "$JOB_CWD"

echo "=== graph1-sealed-m1-headstack: venue=$VENUE task=$TID h=$H_VALUE seed=$TASK_SEED ==="

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
    ${OBS_ARGS[@]+"${OBS_ARGS[@]}"} \
    --log_level INFO

echo "=== done: graph1-sealed-m1-headstack venue=$VENUE task=$TID h=$H_VALUE ==="
