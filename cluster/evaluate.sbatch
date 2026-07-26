#!/usr/bin/env bash
# cluster/evaluate.sbatch -- CPU array job for Bayesian inference (H0 posterior).
#
# This script is submitted by submit_pipeline.sh after the merge job completes.
# It runs Bayesian inference on the merged Cramer-Rao bounds to produce the
# Hubble constant posterior at multiple h values.
#
# Array tasks map to h values via the H_VALUES array below:
#   41 values: 0.010 step in wings (0.60–0.65, 0.79–0.86), 0.005 step in peak (0.655–0.785)
#   NOTE: submit_pipeline.sh derives its --array range by parsing the H_VALUES
#   line below — keep the array definition on a single line (TC-14).
#
# Worker count: --num_workers is NOT passed explicitly. The default in
# arguments.py uses os.sched_getaffinity(0) - 2, which automatically respects
# SLURM's cgroup CPU limit (16 CPUs allocated -> 14 workers). This is more
# robust than hardcoding a value.
#
# Required environment variables (set by submit_pipeline.sh):
#   RUN_DIR -- workspace directory containing prepared Cramer-Rao bounds
#
# Optional environment variables:
#   MAX_REDSHIFT -- analysis-depth truncation override (issue #30 depth-
#     truncation study), passed through as --max_redshift. Unset/empty means
#     the pipeline's built-in default (1.5, a no-op for the current
#     population) -- byte-identical to before this variable existed.
#
# Note: --output and --error are NOT set here; submit_pipeline.sh passes them
# via command line so that shell variables are expanded correctly.

#SBATCH --job-name=emri-evaluate
#SBATCH --partition=cpu,cpu_il
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# Walltime: pre-smoke PLACEHOLDER. Measured anchor (2026-07-03, jobs 5732036):
# 56–76 min per h-value at 3355 events / 16 CPUs (volume_deconv). Handoff §5
# rule: budget 6h @ 16 cpus to cover a 2–4x campaign event yield. Re-size from
# the smoke test before the full campaign submission.
#SBATCH --time=06:00:00

set -euo pipefail

# ---------------------------------------------------------------------------
# H-value sweep: 0.60 to 0.86, 41 values (hybrid grid)
#   Wings (0.60–0.65, 0.79–0.86): step 0.010 — low posterior weight, coarse is fine
#   Peak  (0.655–0.785):          step 0.005 — dense where MAP shifts matter;
#                                 covers the 0.67 closure truth (TC-09)
# Upper bound 0.86 matches LamCDMScenario.h.upper_limit in cosmological_model.py
# ---------------------------------------------------------------------------

H_VALUES=(0.600 0.610 0.620 0.630 0.640 0.650 0.655 0.660 0.665 0.670 0.675 0.680 0.685 0.690 0.695 0.700 0.705 0.710 0.715 0.720 0.725 0.730 0.735 0.740 0.745 0.750 0.755 0.760 0.765 0.770 0.775 0.780 0.785 0.790 0.800 0.810 0.820 0.830 0.840 0.850 0.860)

H_VALUE="${H_VALUES[$SLURM_ARRAY_TASK_ID]}"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# SLURM copies sbatch scripts to /var/spool/slurmd/ before execution, so
# BASH_SOURCE[0] points to the spool copy. Use PROJECT_ROOT (exported by
# submit_pipeline.sh via --export=ALL) or fall back to ~/MasterThesisCode.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set. Set it before submitting this job." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Idempotency guard (TC-06): skip if this task's posterior output already exists.
# Archiving of stale posteriors now happens at SUBMIT time in submit_pipeline.sh
# (login node) — the old task-0 in-job archive raced with sibling array tasks.
# Posterior filename convention (bayesian_statistics.py:1272):
#   h_label = str(np.round(h, 4)).replace(".", "_") — trailing zeros stripped,
#   e.g. 0.600 -> h_0_6.json, 0.655 -> h_0_655.json.
# ---------------------------------------------------------------------------

H_LABEL=$(echo "$H_VALUE" | sed -E 's/0+$//; s/\.$//; s/\./_/')
POSTERIOR_JSON="$RUN_DIR/simulations/posteriors/h_${H_LABEL}.json"
POSTERIOR_BH_JSON="$RUN_DIR/simulations/posteriors_with_bh_mass/h_${H_LABEL}.json"

if [[ -f "$POSTERIOR_JSON" && -f "$POSTERIOR_BH_JSON" ]]; then
    echo "Posteriors for h=$H_VALUE already exist:"
    echo "  $POSTERIOR_JSON"
    echo "  $POSTERIOR_BH_JSON"
    echo "Skipping (idempotency guard) — archive or remove them to force a re-run."
    exit 0
fi

# ---------------------------------------------------------------------------
# Run Bayesian inference
# ---------------------------------------------------------------------------

echo "=== Bayesian Inference (H0 Posterior) ==="
echo "  Run Dir:    $RUN_DIR"
echo "  H value:    $H_VALUE (task $SLURM_ARRAY_TASK_ID of ${#H_VALUES[@]})"
echo "  Node:       $(hostname)"
echo "  CPUs alloc: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "  Workers:    auto-detected from cgroup (SLURM_CPUS_PER_TASK - 2)"
echo "  Max z:      ${MAX_REDSHIFT:-<default>} (issue #30 depth-truncation study)"
echo ""

# Evaluation code reads simulations/prepared_cramer_rao_bounds.csv relative to
# CWD, and the galaxy catalogue via ./master_thesis_code/galaxy_catalogue/
# (handler.py:24). Use a PRIVATE per-run CWD (TC-03) instead of re-pointing the
# shared $PROJECT_ROOT/simulations symlink, so concurrent runs with different
# RUN_DIRs cannot cross-contaminate.
JOB_CWD="$RUN_DIR/cwd"
mkdir -p "$JOB_CWD"
ln -sfn "$RUN_DIR/simulations" "$JOB_CWD/simulations"
ln -sfn "$PROJECT_ROOT/master_thesis_code" "$JOB_CWD/master_thesis_code"
cd "$JOB_CWD"

# Optional --max_redshift pass-through (issue #30 depth-truncation study):
# unset/empty MAX_REDSHIFT leaves the array empty, so the python invocation
# below is byte-identical to before this option existed.
EXTRA_ARGS=()
if [[ -n "${MAX_REDSHIFT:-}" ]]; then
    EXTRA_ARGS+=(--max_redshift "$MAX_REDSHIFT")
fi

# P_det grid resolution: explicit flags for reproducibility (Phase 33).
# Seeding (TC-05): per-task seed = EVAL_SEED + task id (EVAL_SEED exported by
# submit_pipeline.sh from the campaign seed; standalone fallback 777000).
# --simulation_index gives each task its own run_metadata_<task>.json.
python -m master_thesis_code "$RUN_DIR" \
    --evaluate \
    --h_value "$H_VALUE" \
    --seed "$((${EVAL_SEED:-777000} + SLURM_ARRAY_TASK_ID))" \
    --simulation_index "$SLURM_ARRAY_TASK_ID" \
    --pdet_dl_bins 60 \
    --pdet_mass_bins 40 \
    --log_level INFO \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Evaluation Complete (h=$H_VALUE) ==="
