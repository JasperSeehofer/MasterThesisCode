#!/usr/bin/env bash
set -euo pipefail

# Archive the v2.1-era posteriors to an immutable snapshot before Phase 40 re-evaluation.
# Per Phase 40 D-01/D-02: simulations/ state is already pre-v2.2 because --evaluate has
# not been run since Phase 36. This script is run once; exits 1 if archive already exists.

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SRC_DIR="simulations"
DEST_DIR="simulations/_archive_v2_1_baseline"

# D-09: fail loudly if archive already exists — never silently overwrite
if [[ -e "$DEST_DIR" ]]; then
    echo "ERROR: Archive already exists at $DEST_DIR" >&2
    echo "  The v2.1 baseline is captured once and only once." >&2
    echo "  If you need to re-archive, manually move the old archive aside" >&2
    echo "  (e.g., mv $DEST_DIR simulations/_archive_v2_1_baseline_SUPERSEDED_\$(date -u +%Y%m%dT%H%M%SZ))" >&2
    echo "  and re-run this script." >&2
    exit 1
fi

# Verify required source files exist
for required in \
    "$SRC_DIR/combined_posterior.json" \
    "$SRC_DIR/combined_posterior_with_bh_mass.json" \
    "$SRC_DIR/posteriors" \
    "$SRC_DIR/posteriors_with_bh_mass"; do
    if [[ ! -e "$required" ]]; then
        echo "ERROR: Required source path missing: $required" >&2
        exit 2
    fi
done

# Create archive and copy (cp -a preserves mtimes and perms)
mkdir -p "$DEST_DIR"
cp -a "$SRC_DIR/combined_posterior.json"              "$DEST_DIR/"
cp -a "$SRC_DIR/combined_posterior_with_bh_mass.json" "$DEST_DIR/"
cp -a "$SRC_DIR/posteriors"                           "$DEST_DIR/"
cp -a "$SRC_DIR/posteriors_with_bh_mass"              "$DEST_DIR/"

# Write provenance manifest
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --porcelain | wc -l | tr -d ' ')"
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

MANIFEST="$DEST_DIR/ARCHIVE_MANIFEST.md"
{
    echo "# v2.1 Baseline Archive Manifest"
    echo ""
    echo "**Archived:** $TS_UTC"
    echo "**git_commit:** $GIT_COMMIT"
    echo "**git_dirty_files:** $GIT_STATUS"
    echo "**source_dir:** $SRC_DIR"
    echo ""
    echo "## Rationale"
    echo ""
    echo "The posteriors under \`$SRC_DIR\` are the v2.1-era result because"
    echo "\`--evaluate\` has not been run since Phase 36 landed (2026-04-22)."
    echo "Captured verbatim per Phase 40 D-01 / D-02 before any v2.2 re-evaluation."
    echo ""
    echo "## Files (sha256)"
    echo ""
    echo '```'
    cd "$DEST_DIR"
    find . -type f -name '*.json' | sort | xargs sha256sum
    echo '```'
} > "$MANIFEST"

echo "Archive created at $DEST_DIR"
echo "Manifest: $MANIFEST"
