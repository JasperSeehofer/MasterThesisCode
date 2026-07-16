#!/usr/bin/env bash
# cluster/submit_pipeline.sh -- Submit the full simulate-merge-evaluate pipeline.
#
# Chains three SLURM jobs via --dependency=afterok:
#   1. simulate (GPU array job) -- one task per EMRI simulation
#   2. merge (CPU job)          -- combines per-task CSVs, prepares detections
#   3. evaluate (CPU job)       -- Bayesian inference for H0 posterior
#
# Usage:
#   submit_pipeline.sh --tasks N --steps S --seed SEED --injection_pool PATH [options]
#
# --tasks/--steps/--seed are required. --injection_pool is required unless
# --no_injections is given.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: submit_pipeline.sh --tasks N --steps S --seed SEED --injection_pool PATH [options]
  --tasks           Number of array tasks (simulation jobs)
  --steps           Simulation steps per task
  --seed            Base random seed (per-task seed = seed + task_id)
  --injection_pool  Directory holding injection_h_*.csv files (P_det pool).
                    Linked into RUN_DIR/simulations/injections/ at submit time
                    so evaluate uses exactly the intended pool.
                    REQUIRED unless --no_injections is given.
  --no_injections   Skip injection-pool staging (evaluate will fail unless a
                    pool is staged into RUN_DIR/simulations/injections/ by
                    other means).
  --h_true V        Injected true Hubble constant for the simulate stage
                    (default 0.73). Embedded in the run-dir name when != 0.73
                    (e.g. run_YYYYMMDD_seedS_h0p67).
EOF
    exit 1
}

TASKS=""
STEPS=""
SEED=""
H_TRUE="0.73"
INJECTION_POOL=""
NO_INJECTIONS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tasks)          TASKS="$2";          shift 2 ;;
        --steps)          STEPS="$2";          shift 2 ;;
        --seed)           SEED="$2";           shift 2 ;;
        --h_true)         H_TRUE="$2";         shift 2 ;;
        --injection_pool) INJECTION_POOL="$2"; shift 2 ;;
        --no_injections)  NO_INJECTIONS=1;     shift ;;
        -h|--help) usage ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage ;;
    esac
done

if [[ -z "$TASKS" ]]; then
    echo "ERROR: --tasks is required." >&2
    usage
fi

if [[ -z "$STEPS" ]]; then
    echo "ERROR: --steps is required." >&2
    usage
fi

if [[ -z "$SEED" ]]; then
    echo "ERROR: --seed is required." >&2
    usage
fi

if [[ "$NO_INJECTIONS" -eq 0 && -z "$INJECTION_POOL" ]]; then
    echo "ERROR: --injection_pool is required (or pass --no_injections explicitly)." >&2
    echo "       Evaluate's p_det grid is built from RUN_DIR/simulations/injections/;" >&2
    echo "       see cluster/datasets.yaml for pool provenance/retirement status." >&2
    usage
fi

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
# H-grid: single source of truth is the H_VALUES line in evaluate.sbatch (TC-14).
# Derive the evaluate array size from it — never hardcode the count here.
# ---------------------------------------------------------------------------

EVAL_SBATCH="$CLUSTER_DIR/evaluate.sbatch"
# `|| true`: a grep miss under `set -euo pipefail` would abort before the diagnostic
# guard below fires (review finding CLU-03).
H_GRID=$(grep -m1 '^H_VALUES=(' "$EVAL_SBATCH" | sed -E 's/^H_VALUES=\(//; s/\)[[:space:]]*$//' || true)
N_H=$(echo "$H_GRID" | grep -oE '[0-9]+\.[0-9]+' | wc -l | tr -d ' ' || true)

if [[ -z "$N_H" || "$N_H" -eq 0 ]]; then
    echo "ERROR: Could not parse H_VALUES from $EVAL_SBATCH (got 0 values)." >&2
    echo "       The H_VALUES=(...) definition must be a single line." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Run directory creation (D-08). Non-default truth is embedded in the name
# (TC-09) so closure runs can't be mistaken for production runs.
# ---------------------------------------------------------------------------

DATESTAMP=$(date +%Y%m%d)
RUN_SUFFIX=""
if [[ "$H_TRUE" != "0.73" ]]; then
    RUN_SUFFIX="_h$(echo "$H_TRUE" | tr '.' 'p')"
fi
RUN_DIR="$WORKSPACE/run_${DATESTAMP}_seed${SEED}${RUN_SUFFIX}"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/simulations"

# ---------------------------------------------------------------------------
# Injection pool staging (TC-09): link the pool's CSVs into this run's
# simulations/injections/ so evaluate's p_det grid uses exactly this pool.
# ---------------------------------------------------------------------------

N_POOL=0
if [[ "$NO_INJECTIONS" -eq 0 ]]; then
    if ! compgen -G "$INJECTION_POOL/injection_h_*.csv" > /dev/null; then
        echo "ERROR: --injection_pool '$INJECTION_POOL' contains no injection_h_*.csv files." >&2
        exit 1
    fi
    mkdir -p "$RUN_DIR/simulations/injections"
    for f in "$INJECTION_POOL"/injection_h_*.csv; do
        ln -sfn "$f" "$RUN_DIR/simulations/injections/$(basename "$f")"
        N_POOL=$((N_POOL + 1))
    done
fi

# ---------------------------------------------------------------------------
# Archive old posteriors at SUBMIT time (TC-06 — moved out of evaluate.sbatch,
# where the task-0 in-job archive raced with sibling array tasks). Runs on the
# login node before the evaluate array is submitted.
# ---------------------------------------------------------------------------

POSTERIORS_DIR="$RUN_DIR/simulations/posteriors"
POSTERIORS_BH_DIR="$RUN_DIR/simulations/posteriors_with_bh_mass"
ARCHIVE_BASE="$RUN_DIR/simulations/archive"

_has_old_files() {
    local dir="$1"
    [[ -d "$dir" ]] && compgen -G "$dir/h_*.json" > /dev/null 2>&1
}

if _has_old_files "$POSTERIORS_DIR" || _has_old_files "$POSTERIORS_BH_DIR"; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVE_DIR="$ARCHIVE_BASE/eval_${TIMESTAMP}"
    mkdir -p "$ARCHIVE_DIR"

    # Save metadata for traceability
    cat > "$ARCHIVE_DIR/archive_metadata.json" <<METAEOF
{
  "archived_at": "$(date -Iseconds)",
  "archived_by": "submit_pipeline.sh (login node, at submit time)",
  "git_commit": "$(cd "$CLUSTER_DIR/.." && git rev-parse HEAD 2>/dev/null || echo unknown)",
  "git_branch": "$(cd "$CLUSTER_DIR/.." && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)",
  "reason": "Pre-evaluation cleanup: archiving old posteriors before new evaluation run",
  "h_grid": "$H_GRID",
  "source_run_dir": "$RUN_DIR"
}
METAEOF

    # Move old posteriors to archive
    if _has_old_files "$POSTERIORS_DIR"; then
        mkdir -p "$ARCHIVE_DIR/posteriors"
        mv "$POSTERIORS_DIR"/h_*.json "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
        mv "$POSTERIORS_DIR"/combined_posterior.json "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
        mv "$POSTERIORS_DIR"/diagnostic_report.md "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
        mv "$POSTERIORS_DIR"/comparison_table.md "$ARCHIVE_DIR/posteriors/" 2>/dev/null || true
        echo "Archived old posteriors/ to $ARCHIVE_DIR/posteriors/"
    fi

    if _has_old_files "$POSTERIORS_BH_DIR"; then
        mkdir -p "$ARCHIVE_DIR/posteriors_with_bh_mass"
        mv "$POSTERIORS_BH_DIR"/h_*.json "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
        mv "$POSTERIORS_BH_DIR"/combined_posterior.json "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
        mv "$POSTERIORS_BH_DIR"/diagnostic_report.md "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
        mv "$POSTERIORS_BH_DIR"/comparison_table.md "$ARCHIVE_DIR/posteriors_with_bh_mass/" 2>/dev/null || true
        echo "Archived old posteriors_with_bh_mass/ to $ARCHIVE_DIR/posteriors_with_bh_mass/"
    fi
else
    echo "No old posteriors to archive."
fi

# Per-task evaluate seed base (TC-05): derived from the campaign seed so the
# whole campaign remains reproducible from --seed alone.
EVAL_SEED=$((SEED * 1000))

echo ""
echo "=== EMRI Pipeline Submission ==="
echo "  Tasks:      $TASKS"
echo "  Steps:      $STEPS"
echo "  Seed:       $SEED"
echo "  H true:     $H_TRUE"
if [[ "$NO_INJECTIONS" -eq 0 ]]; then
    echo "  Inj. pool:  $INJECTION_POOL ($N_POOL csv linked)"
else
    echo "  Inj. pool:  (none — --no_injections)"
fi
echo "  H grid:     $N_H values (parsed from evaluate.sbatch)"
echo "  Eval seed:  $EVAL_SEED (+ task id)"
echo "  Run dir:    $RUN_DIR"
echo ""

# ---------------------------------------------------------------------------
# Job submission chain (SLURM-04)
# ---------------------------------------------------------------------------

# 1. Simulate array job (GPU) — H_VALUE threads the injected truth (TC-09)
SIM_JOB=$(sbatch --parsable \
    --array="0-$((TASKS - 1))" \
    --output="$RUN_DIR/logs/simulate_%A_%a.out" \
    --error="$RUN_DIR/logs/simulate_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$SEED",SIM_STEPS="$STEPS",H_VALUE="$H_TRUE" \
    "$CLUSTER_DIR/simulate.sbatch")

# 2. Merge job (CPU, after all simulate tasks finish — afterany tolerates timeouts)
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterany:$SIM_JOB" \
    --output="$RUN_DIR/logs/merge_%j.out" \
    --error="$RUN_DIR/logs/merge_%j.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",PREPARE_SEED="$((SEED + 999999))" \
    "$CLUSTER_DIR/merge.sbatch")

# 3. Evaluate array job (CPU, after merge completes)
#    Array size derived from the H_VALUES grid in evaluate.sbatch (TC-14)
EVAL_JOB=$(sbatch --parsable \
    --array="0-$((N_H - 1))" \
    --dependency="afterok:$MERGE_JOB" \
    --output="$RUN_DIR/logs/evaluate_%A_%a.out" \
    --error="$RUN_DIR/logs/evaluate_%A_%a.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",EVAL_SEED="$EVAL_SEED" \
    "$CLUSTER_DIR/evaluate.sbatch")

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
echo "  H true:    $H_TRUE"
if [[ "$NO_INJECTIONS" -eq 0 ]]; then
    echo "  Inj. pool: $INJECTION_POOL ($N_POOL csv linked)"
else
    echo "  Inj. pool: (none — --no_injections)"
fi
echo "  Simulate:  $SIM_JOB (array 0-$((TASKS - 1)))"
echo "  Merge:     $MERGE_JOB (after simulate)"
echo "  Evaluate:  $EVAL_JOB (array 0-$((N_H - 1)), ${N_H}-point hybrid h-grid 0.60–0.86, EVAL_SEED=$EVAL_SEED)"
echo "  Combine:   $COMBINE_JOB (after evaluate)"
echo ""
echo "Monitor: sacct -j $SIM_JOB,$MERGE_JOB,$EVAL_JOB,$COMBINE_JOB"
