#!/usr/bin/env bash
# Multi-truth bias-vs-h_true sweep orchestrator (post-Tier-3 verification).
#
# For each h_true in the panel:
#   1. Rescale local CRB to that h_true                     (test_23)
#   2. Run prepare_detections                               (fresh seed)
#   3. Push CRBs to a per-truth cluster RUN_DIR
#   4. Submit the parameterized sbatch
# Once all jobs complete, pull posteriors and run the analyzer (test_24).
#
# Submission runs on a configurable partition (default cpu_il, 128 cpus,
# --array=0-6).  Override via env to run on dev_cpu_il for the smoke or to
# tune the production sweep:
#   SBATCH_PARTITION (default cpu_il)
#   SBATCH_CPUS      (default 128)
#   SBATCH_ARRAY     (default 0-6)
# The script submits one truth at a time, polling for completion before
# moving to the next, so we never have more than one job in queue.
#
# Usage:
#   bash scripts/bias_investigation/run_multi_truth_sweep.sh
#
# Override panel via TRUTHS env var:
#   TRUTHS="0.65 0.70 0.75" bash scripts/bias_investigation/run_multi_truth_sweep.sh

set -euo pipefail

# Force "." as decimal separator so printf "%.2f" "$h_truth" works regardless
# of the user's locale (de_DE.UTF-8 etc. otherwise emit "," and break the
# orchestrator's filename construction).  Mirrors the LC_ALL=C usage inside
# cluster/evaluate_closure_h_true_finegrid.sbatch.
export LC_ALL=C

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Truth panel — default mirrors the verification plan
TRUTHS=${TRUTHS:-"0.60 0.65 0.70 0.73 0.75 0.80 0.85"}

# Source CRB for rescaling.  Default is the phase46-merged dataset
# (Phase 45 seed=200 + seed=300 extension), 924 SNR>=20 events.  Override
# via INPUT_CRB env var to use a different CRB (e.g. raw Phase 45 only).
INPUT_CRB=${INPUT_CRB:-"$PROJECT_ROOT/simulations/cluster_run_phase46_merged_20260504/cramer_rao_bounds.csv"}

CLUSTER_HOST="bwunicluster"
CLUSTER_WS="/pfs/work9/workspace/scratch/st_ac147838-emri"
SEED_BASE=200  # bumped per truth to ensure fresh noise draws

# Cluster job parameters (env-overridable for smoke / different partitions)
SBATCH_PARTITION=${SBATCH_PARTITION:-cpu_il}
SBATCH_CPUS=${SBATCH_CPUS:-128}
SBATCH_ARRAY=${SBATCH_ARRAY:-0-6}

echo "=== Multi-truth closure sweep ==="
echo "Truths:    $TRUTHS"
echo "Input CRB: $INPUT_CRB"
echo "Partition: $SBATCH_PARTITION (cpus=$SBATCH_CPUS, array=$SBATCH_ARRAY)"
echo ""

if [[ ! -f "$INPUT_CRB" ]]; then
    echo "ERROR: INPUT_CRB does not exist: $INPUT_CRB" >&2
    exit 1
fi

# ssh helper with retries (dev_cpu_il connections sometimes drop)
ssh_retry() {
    local cmd="$1"
    for i in 1 2 3 4 5; do
        if out=$(ssh -o ConnectTimeout=15 -o ConnectionAttempts=2 "$CLUSTER_HOST" "$cmd" 2>&1); then
            echo "$out"
            return 0
        fi
        echo "  ssh attempt $i failed: $(echo "$out" | head -1)" >&2
        sleep 8
    done
    return 1
}

# Push the parameterized sbatch (only needs to happen once)
echo "[setup] Pushing evaluate_closure_h_true_finegrid.sbatch to cluster..."
for i in 1 2 3 4 5; do
    if scp -o ConnectTimeout=20 -o ConnectionAttempts=3 \
        "$PROJECT_ROOT/cluster/evaluate_closure_h_true_finegrid.sbatch" \
        "$CLUSTER_HOST:~/darksiren-emri/cluster/" 2>&1; then
        echo "  pushed."
        break
    fi
    sleep 8
done

idx=0
for h_truth in $TRUTHS; do
    idx=$((idx + 1))
    h_short=$(printf "%.2f" "$h_truth" | tr '.' 'p')
    workdir="simulations/closure_h${h_short}"
    cluster_run_dir="${CLUSTER_WS}/run_closure_h${h_short}_$(date +%Y%m%d)"
    seed=$((SEED_BASE + idx))

    echo ""
    echo "================================================================"
    echo "[$idx] h_truth=$h_truth  (workdir=$workdir, cluster=$cluster_run_dir)"
    echo "================================================================"

    # 1. Local rescale
    echo "[1/5] Rescaling CRB locally..."
    uv run python scripts/bias_investigation/test_23_rescale_crb_to_h_true.py \
        --h-true "$h_truth" \
        --workdir "$workdir" \
        --input-crb "$INPUT_CRB"

    # 2. prepare_detections
    echo "[2/5] Running prepare_detections (seed=$seed)..."
    uv run python scripts/prepare_detections.py \
        --workdir "$workdir" --seed "$seed" --force

    # 3. Push CRBs + setup cluster RUN_DIR
    echo "[3/5] Setting up cluster RUN_DIR and pushing CRBs..."
    ssh_retry "mkdir -p $cluster_run_dir/simulations/logs && \
        ln -sfn $CLUSTER_WS/run_phase45_20260501/simulations/injections \
            $cluster_run_dir/simulations/injections"
    rsync -avz -e "ssh -o ConnectTimeout=20 -o ConnectionAttempts=3" \
        "$PROJECT_ROOT/$workdir/simulations/cramer_rao_bounds.csv" \
        "$PROJECT_ROOT/$workdir/simulations/prepared_cramer_rao_bounds.csv" \
        "$CLUSTER_HOST:$cluster_run_dir/simulations/" | tail -3

    # 4. Submit — array/partition/cpus chosen at orchestrator entry; sbatch
    #    stride logic auto-adapts via SLURM_ARRAY_TASK_COUNT.
    echo "[4/5] Submitting sbatch on $SBATCH_PARTITION (array=$SBATCH_ARRAY, cpus=$SBATCH_CPUS)..."
    submit_out=$(ssh_retry "cd ~/darksiren-emri && \
        sbatch --array=$SBATCH_ARRAY \
            --partition=$SBATCH_PARTITION \
            --cpus-per-task=$SBATCH_CPUS \
            --export=ALL,RUN_DIR=$cluster_run_dir,H_TRUE=$h_truth,PROJECT_ROOT=\$HOME/darksiren-emri \
            --output=$cluster_run_dir/simulations/logs/closure_multi_%A_%a.out \
            --error=$cluster_run_dir/simulations/logs/closure_multi_%A_%a.err \
            cluster/evaluate_closure_h_true_finegrid.sbatch")
    echo "  $submit_out"
    job_id=$(echo "$submit_out" | grep -oP 'Submitted batch job \K\d+')
    if [[ -z "$job_id" ]]; then
        echo "  ERROR: could not parse job id from sbatch output" >&2
        exit 1
    fi
    echo "  Job ID: $job_id"

    # 5. Wait for completion (cpu_il runs the 7 tasks in parallel; ~5 min each)
    echo "[5/5] Waiting for job $job_id to complete..."
    while true; do
        sleep 60
        n_remaining=$(ssh_retry "squeue -u \$USER -j $job_id -h 2>/dev/null | wc -l" || echo "?")
        if [[ "$n_remaining" == "0" ]]; then
            echo "  Job $job_id done."
            break
        fi
        echo "    [$(date +%H:%M)] $n_remaining task(s) still in queue/running..."
    done

    # Pull posteriors
    local_finegrid_dir="$PROJECT_ROOT/simulations/cluster_run_closure_h${h_short}_finegrid"
    mkdir -p "$local_finegrid_dir"
    rsync -avz -e "ssh -o ConnectTimeout=20 -o ConnectionAttempts=3" \
        "$CLUSTER_HOST:$cluster_run_dir/simulations/posteriors/" \
        "$local_finegrid_dir/posteriors/" | tail -3
    rsync -avz -e "ssh -o ConnectTimeout=20 -o ConnectionAttempts=3" \
        "$CLUSTER_HOST:$cluster_run_dir/simulations/posteriors_with_bh_mass/" \
        "$local_finegrid_dir/posteriors_with_bh_mass/" | tail -3
    echo "  Posteriors rsynced to $local_finegrid_dir"
done

echo ""
echo "================================================================"
echo "All truths complete. Running analyzer..."
echo "================================================================"
uv run python scripts/bias_investigation/test_24_multi_truth_bias_sweep.py \
    --truths $TRUTHS

echo ""
echo "Sweep complete. Output:"
echo "  scripts/bias_investigation/outputs/phase45/multi_truth_sweep.json"
echo "  scripts/bias_investigation/outputs/phase45/multi_truth_sweep.png"
