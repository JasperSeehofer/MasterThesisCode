#!/usr/bin/env bash
# cluster/submit_resimulate_phase50.sh -- Phase 50 fresh simulation campaign.
#
# Chains the full pipeline with two divergences from submit_pipeline.sh:
#   1. Uses the well-mixed M1 sampler (commit 991333a, nwalkers=50, burn_in=10000)
#      to produce a single homogeneous CRB campaign that replaces the heterogeneous
#      seed200⊕seed300 production set diagnosed in handoff §A.
#   2. Substitutes evaluate_production_h0p73_superdense.sbatch (83-point grid)
#      for the default evaluate.sbatch (38-point grid), bundling handoff §B
#      h-grid refinement into the same campaign.
#
# F4 (Nadaraya-Watson p_det, commit d1087f1) is on main and will be used by the
# evaluate stage — this campaign therefore answers handoff §F as a side-effect.
#
# Usage:
#   submit_resimulate_phase50.sh  [defaults: --tasks 50 --steps 35 --seed 400]
#
# Override defaults by passing flags identical to submit_pipeline.sh.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing (defaults chosen for Phase 50)
# ---------------------------------------------------------------------------

TASKS="50"
STEPS="35"
SEED="400"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)  TASKS="$2";  shift 2 ;;
        --steps)  STEPS="$2";  shift 2 ;;
        --seed)   SEED="$2";   shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: submit_resimulate_phase50.sh [--tasks N] [--steps S] [--seed SEED]
  --tasks   Number of array tasks (simulation jobs)        [default: 50]
  --steps   Simulation steps per task                      [default: 35]
  --seed    Base random seed (per-task seed = seed + tid)  [default: 400]
EOF
            exit 0
            ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLUSTER_DIR/modules.sh"

if [[ -z "${WORKSPACE:-}" ]]; then
    echo "ERROR: \$WORKSPACE is not set. Run cluster/setup.sh first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run directory creation
# ---------------------------------------------------------------------------

DATESTAMP=$(date +%Y%m%d)
RUN_DIR="$WORKSPACE/run_${DATESTAMP}_seed${SEED}_phase50"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/simulations"

# Capture campaign metadata at submission time for posterity
cat > "$RUN_DIR/campaign_metadata.json" <<META
{
  "campaign": "phase50_resimulate",
  "submitted_at": "$(date -Iseconds)",
  "tasks": $TASKS,
  "steps": $STEPS,
  "base_seed": $SEED,
  "h_grid_points": 83,
  "h_grid_description": "63 dense + 20 super-dense Δh=0.0005 in [0.720, 0.740] + wings",
  "git_commit_at_submit": "$(cd "$CLUSTER_DIR/.." && git rev-parse HEAD)",
  "purpose": "Replace heterogeneous seed200⊕seed300 production CRB with single homogeneous campaign using well-mixed sampler (commit 991333a); bundles handoff §B superdense h-grid + §F F4 estimator transition.",
  "reference_handoff": ".planning/HANDOFF-PLOTTING-OVERHAUL-20260516.md",
  "bias_resolution_section": "docs/H0_BIAS_RESOLUTION.md §3.16"
}
META

echo ""
echo "=== Phase 50 Re-simulation Campaign ==="
echo "  Tasks:     $TASKS"
echo "  Steps:     $STEPS"
echo "  Seed:      $SEED"
echo "  Run dir:   $RUN_DIR"
echo "  Evaluate:  83-point superdense h-grid (handoff §B)"
echo "  Sampler:   nwalkers=50, burn_in=10000 (commit 991333a, handoff §A fix)"
echo ""

# ---------------------------------------------------------------------------
# Job submission chain
# ---------------------------------------------------------------------------

# 1. Simulate array job (GPU) — uses the well-mixed M1 sampler from main
SIM_JOB=$(sbatch --parsable \
    --array="0-$((TASKS - 1))" \
    --output="$RUN_DIR/logs/simulate_%A_%a.out" \
    --error="$RUN_DIR/logs/simulate_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$SEED",SIM_STEPS="$STEPS" \
    "$CLUSTER_DIR/simulate.sbatch")

# 2. Merge job (CPU, after all simulate tasks finish — afterany tolerates timeouts)
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterany:$SIM_JOB" \
    --output="$RUN_DIR/logs/merge_%j.out" \
    --error="$RUN_DIR/logs/merge_%j.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",PREPARE_SEED="$((SEED + 999999))" \
    "$CLUSTER_DIR/merge.sbatch")

# 3. Evaluate array job (CPU) — 83-point superdense h-grid, 9 tasks
EVAL_JOB=$(sbatch --parsable \
    --array="0-8" \
    --dependency="afterok:$MERGE_JOB" \
    --output="$RUN_DIR/logs/evaluate_%A_%a.out" \
    --error="$RUN_DIR/logs/evaluate_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR" \
    "$CLUSTER_DIR/evaluate_production_h0p73_superdense.sbatch")

# 4. Combine posteriors (CPU, after all evaluate tasks finish)
COMBINE_JOB=$(sbatch --parsable \
    --dependency="afterok:$EVAL_JOB" \
    --output="$RUN_DIR/logs/combine_%j.out" \
    --error="$RUN_DIR/logs/combine_%j.err" \
    --export=ALL,RUN_DIR="$RUN_DIR" \
    "$CLUSTER_DIR/combine.sbatch")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "Pipeline submitted:"
echo "  Run directory: $RUN_DIR"
echo "  Simulate:  $SIM_JOB (array 0-$((TASKS - 1)), gpu_h100)"
echo "  Merge:     $MERGE_JOB (afterany simulate, cpu)"
echo "  Evaluate:  $EVAL_JOB (array 0-8, afterok merge, cpu_il, 83-pt superdense grid)"
echo "  Combine:   $COMBINE_JOB (afterok evaluate, cpu)"
echo ""
echo "Monitor: sacct -j $SIM_JOB,$MERGE_JOB,$EVAL_JOB,$COMBINE_JOB"
echo ""
echo "After campaign completes, rsync results local with:"
echo "  rsync -avz bwunicluster:$RUN_DIR/simulations/ ./simulations/cluster_run_phase50_${DATESTAMP}/"
