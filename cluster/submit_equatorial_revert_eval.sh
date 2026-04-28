#!/usr/bin/env bash
# cluster/submit_equatorial_revert_eval.sh
#
# Diagnostic: re-run --evaluate + --combine using the EQUATORIAL CRBs
# (*.bak_equatorial files) to determine whether the MAP=0.86 bias was
# introduced by the ecliptic migration or was already present before it.
#
# Run this from the cluster AFTER git pull:
#   bash cluster/submit_equatorial_revert_eval.sh
#
# Results land in:
#   $WORKSPACE/run_equatorial_revert_YYYYMMDD/simulations/posteriors/
#
# Rsync back (from local machine):
#   rsync -avz bwunicluster:$(ssh bwunicluster 'ws_find emri')/run_equatorial_revert_*/simulations/posteriors/ \
#         ./results/equatorial_revert_posteriors/

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CLUSTER_DIR/modules.sh"

if [[ -z "${WORKSPACE:-}" ]]; then
    echo "ERROR: \$WORKSPACE is not set. Run cluster/setup.sh first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Production seed200 run — source of the migrated CSVs and their backups
SRC_RUN="/pfs/work9/workspace/scratch/st_ac147838-emri/run_20260401_seed200"
SRC_SIM="$SRC_RUN/simulations"

# Fresh diagnostic run directory
NEW_RUN="$WORKSPACE/run_equatorial_revert_$(date +%Y%m%d)"
NEW_SIM="$NEW_RUN/simulations"

echo ""
echo "=== Equatorial CRB Revert Diagnostic ==="
echo "  Source run:  $SRC_RUN"
echo "  Revert run:  $NEW_RUN"
echo "  Purpose:     Test if MAP=0.86 predates the ecliptic migration"
echo ""

# ---------------------------------------------------------------------------
# Verify backup files exist
# ---------------------------------------------------------------------------

if ! compgen -G "$SRC_SIM/*.bak_equatorial" > /dev/null 2>&1; then
    echo "ERROR: No .bak_equatorial files found in $SRC_SIM" >&2
    echo "       The migration script must have been run first to create backups." >&2
    exit 1
fi

echo "Found .bak_equatorial backup files:"
ls "$SRC_SIM"/*.bak_equatorial
echo ""

# ---------------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------------

mkdir -p "$NEW_RUN/logs"
mkdir -p "$NEW_SIM/posteriors"
mkdir -p "$NEW_SIM/posteriors_with_bh_mass"

# ---------------------------------------------------------------------------
# Symlink all non-CRB files from the source simulations/ directory.
# This covers undetected_events.csv, P_det cache, diagnostics, etc.
# Sky angles don't affect these files — only the CRB CSVs matter.
# ---------------------------------------------------------------------------

echo "Linking non-CRB files from source simulations/..."
for f in "$SRC_SIM"/*; do
    fname="$(basename "$f")"
    case "$fname" in
        # Skip: CRBs (we'll restore equatorial versions below)
        *cramer_rao_bounds*.csv) continue ;;
        # Skip: posterior output dirs and other run-specific dirs
        posteriors|posteriors_with_bh_mass|archive|figures) continue ;;
    esac
    ln -sfn "$f" "$NEW_SIM/$fname"
    echo "  linked: $fname"
done
echo ""

# ---------------------------------------------------------------------------
# Restore equatorial CRBs from *.bak_equatorial backups
# ---------------------------------------------------------------------------

echo "Restoring equatorial CRB CSVs:"
for bak in "$SRC_SIM"/*.bak_equatorial; do
    original_name="$(basename "$bak" .bak_equatorial)"
    cp "$bak" "$NEW_SIM/$original_name"
    echo "  restored: $original_name"
done
echo ""

# Sanity-check: confirm sky-angle columns are equatorial (no ecliptic marker)
echo "Spot-checking coordinate frame in restored CSVs:"
for csv in "$NEW_SIM"/*cramer_rao_bounds*.csv; do
    frame=$(head -1 "$csv" | tr ',' '\n' | grep -i "ecliptic\|equatorial\|frame" | head -1 || echo "(no frame column found)")
    echo "  $(basename "$csv"): $frame"
done
echo ""

# ---------------------------------------------------------------------------
# Submit evaluate array job (38 h-values, 0.60–0.86, hybrid grid)
# ---------------------------------------------------------------------------

echo "Submitting evaluate job (38-point h sweep)..."
EVAL_JOB=$(sbatch --parsable \
    --array="0-37" \
    --output="$NEW_RUN/logs/evaluate_%A_%a.out" \
    --error="$NEW_RUN/logs/evaluate_%A_%a.err" \
    --export=ALL,RUN_DIR="$NEW_RUN" \
    "$CLUSTER_DIR/evaluate.sbatch")
echo "  Evaluate job ID: $EVAL_JOB"

# ---------------------------------------------------------------------------
# Submit combine job (runs after all evaluate tasks complete)
# ---------------------------------------------------------------------------

COMBINE_JOB=$(sbatch --parsable \
    --dependency="afterok:$EVAL_JOB" \
    --output="$NEW_RUN/logs/combine_%j.out" \
    --error="$NEW_RUN/logs/combine_%j.err" \
    --export=ALL,RUN_DIR="$NEW_RUN" \
    "$CLUSTER_DIR/combine.sbatch")
echo "  Combine job ID:  $COMBINE_JOB"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "=== Jobs Submitted ==="
echo ""
echo "Monitor:"
echo "  sacct -j $EVAL_JOB,$COMBINE_JOB --format=JobID,State,Elapsed,ExitCode"
echo "  squeue -j $EVAL_JOB,$COMBINE_JOB"
echo ""
echo "Expected runtime: ~10 min (evaluate) + ~5 min (combine)"
echo ""
echo "Results will appear in:"
echo "  $NEW_RUN/simulations/posteriors/combined_posterior.json"
echo ""
echo "Rsync back to local machine (after completion):"
echo "  rsync -avz bwunicluster:$NEW_RUN/simulations/posteriors/ \\"
echo "        ./results/equatorial_revert_posteriors/"
echo ""
echo "Interpretation:"
echo "  MAP ≈ 0.73  →  bias introduced by ecliptic migration (frame mismatch bug)"
echo "  MAP ≈ 0.86  →  bias predates migration (latent L_cat / P_det bug)"
