#!/usr/bin/env bash
# Re-evaluate production CRBs (run_20260401_seed200, ecliptic-migrated)
# under the Phase 45 P_det first-bin asymptote anchor (commit 09ee262).
# Same data + same script structure as Phase 44; expects MAP shift from
# 0.7650 (Phase 44 post-fix) to ~0.73 (Phase 45 post-anchor-lift).
# Acceptance window: MAP ∈ [0.72, 0.74]; 68% bootstrap interval ∋ 0.73.

set -euo pipefail
CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLUSTER_DIR/modules.sh"

[[ -n "${WORKSPACE:-}" ]] || { echo "ERROR: \$WORKSPACE not set." >&2; exit 1; }

SRC_RUN="/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260401_seed200"
NEW_RUN="$WORKSPACE/run_phase45_$(date +%Y%m%d)"
NEW_SIM="$NEW_RUN/simulations"

echo "=== Phase 45 post-anchor-lift re-evaluation ==="
echo "  Source CRBs: $SRC_RUN/simulations  (post-Phase-43 ecliptic-migrated)"
echo "  Run dir:     $NEW_RUN"
echo "  Expected:    MAP shifts from 0.7650 (Phase 44 post-fix) to ∈ [0.72, 0.74] (Phase 45 post-anchor)"
echo ""

mkdir -p "$NEW_SIM" "$NEW_RUN/logs"

# Copy the migrated CRB CSVs (NOT .bak_equatorial — we want post-migration)
for f in cramer_rao_bounds.csv prepared_cramer_rao_bounds.csv; do
    cp "$SRC_RUN/simulations/$f" "$NEW_SIM/$f"
    echo "  copied $f"
done

# Symlink injections from the source run (or wherever they live)
if [[ -d "$SRC_RUN/simulations/injections" ]]; then
    ln -sf "$SRC_RUN/simulations/injections" "$NEW_SIM/injections"
    echo "  linked injections"
fi

# Submit evaluate array (38 h-values, 0.60–0.86)
EVAL_JOB=$(sbatch --parsable \
    --array="0-37" \
    --output="$NEW_RUN/logs/evaluate_%A_%a.out" \
    --error="$NEW_RUN/logs/evaluate_%A_%a.err" \
    --export=ALL,RUN_DIR="$NEW_RUN" \
    "$CLUSTER_DIR/evaluate.sbatch")
echo "Evaluate job: $EVAL_JOB"

COMBINE_JOB=$(sbatch --parsable \
    --dependency="afterok:$EVAL_JOB" \
    --output="$NEW_RUN/logs/combine_%j.out" \
    --error="$NEW_RUN/logs/combine_%j.err" \
    --export=ALL,RUN_DIR="$NEW_RUN" \
    "$CLUSTER_DIR/combine.sbatch")
echo "Combine job:  $COMBINE_JOB"

echo ""
echo "Monitor:  sacct -j $EVAL_JOB,$COMBINE_JOB --format=JobID,State,Elapsed,ExitCode"
echo "Rsync:    rsync -avz bwunicluster:$NEW_RUN/simulations/posteriors/ ./results/phase45_posteriors/"
