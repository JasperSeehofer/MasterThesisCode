#!/usr/bin/env bash
# Launcher for PREREGISTRATION_PROD_COUNTERFACTUAL.md v2 (run ON the cluster login node).
# Creates the 8 registered run dirs (cloning the run_20260817_fusioncf_* symlink pattern)
# and submits the evaluate.sbatch arrays: V0 probes (h=0.72 idx19, h=0.78 idx31) +
# {V1' neutralized, V2 k=0.5, V2 k=2.0} x {iiib, joint_r1} full 41-h arrays.
# Total tasks: 4 + 6*41 = 250. Config of record: fusion-counterfactual off basis
# (absolute_marginal, volume_deconv, pdet_z_resolved, EVAL_SEED=777000,
# --selection_in_completion_numerator off).
set -euo pipefail

PROJECT_ROOT="$HOME/darksiren-emri"
WS=$(ws_find emri)
SEED_SRC="$WS/run_20260729_seed61000/simulations"
OBS_CAT="$WS/realizations_20260729/observed_catalogue_seed900001.csv"
STAMP=20260819

[[ -f "$SEED_SRC/prepared_cramer_rao_bounds.csv" ]] || { echo "ERROR: seed61000 CRB missing" >&2; exit 1; }
[[ -f "$OBS_CAT" ]] || { echo "ERROR: observed catalogue missing" >&2; exit 1; }

make_run_dir() {
    local dir="$1"
    mkdir -p "$dir/simulations"
    ln -sfn "$SEED_SRC/cramer_rao_bounds.csv" "$dir/simulations/cramer_rao_bounds.csv"
    ln -sfn "$SEED_SRC/prepared_cramer_rao_bounds.csv" "$dir/simulations/prepared_cramer_rao_bounds.csv"
    ln -sfn "$SEED_SRC/injections" "$dir/simulations/injections"
}

submit_cell() {
    local cell="$1" venue="$2" array_spec="$3" extra_eval="$4"
    local run_dir="$WS/run_${STAMP}_cf_${cell}_${venue}"
    make_run_dir "$run_dir"
    local env="ALL,PROJECT_ROOT=$PROJECT_ROOT,RUN_DIR=$run_dir,EVAL_SEED=777000"
    env+=",NORMALIZATION_MODE=absolute_marginal,HOST_Z_KERNEL=volume_deconv,PDET_Z_RESOLVED=yes"
    env+=",EXTRA_EVAL_ARGS=--selection_in_completion_numerator off ${extra_eval}"
    if [[ "$venue" == "joint_r1" ]]; then
        env+=",OBSERVED_CATALOGUE=$OBS_CAT"
    fi
    sbatch --array="$array_spec" \
        --export="$env" \
        --output="$run_dir/slurm_%A_%a.out" \
        --error="$run_dir/slurm_%A_%a.err" \
        --job-name="cf_${cell}_${venue}" \
        "$PROJECT_ROOT/cluster/evaluate.sbatch"
}

for venue in iiib joint_r1; do
    # V0 continuity probes (mode=production explicit): h=0.72 (idx 19), h=0.78 (idx 31)
    submit_cell v0 "$venue" "19,31" "--catalogue_mass_overlap production"
    # V1' neutralized
    submit_cell v1 "$venue" "0-40" "--catalogue_mass_overlap neutralized"
    # V2 pure-width ladder
    submit_cell v2k05 "$venue" "0-40" "--catalogue_mass_overlap inflated --catalogue_mass_error_scale 0.5"
    submit_cell v2k2 "$venue" "0-40" "--catalogue_mass_overlap inflated --catalogue_mass_error_scale 2.0"
done
echo "All cells submitted."
