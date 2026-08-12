#!/usr/bin/env bash
# migrate_cluster_rename.sh — Rebrand Migration Checklist §2 (+ the §3
# cluster-path lines folded into §2 per the checklist).
#
# This script has two independent parts:
#
#   PART A — cluster-side rename. NOT executed by this script. It only
#   PRINTS the commands the author runs by hand, interactively, ON THE
#   CLUSTER (bwUniCluster login node). No ssh call is made from here.
#
#   PART B — local docs pass. Executed ONLY when this script is invoked
#   with --docs-pass, and only AFTER Part A has completed on the cluster
#   (repo renamed, remote updated, venv rebuilt). Applies the checklist
#   §2.4 sed fixes across the git-tracked cluster docs/scripts, plus the
#   three scripts/bias_investigation cluster-path lines carried over from
#   §3, then commits.
#
# Usage:
#   bash scripts/migrate_cluster_rename.sh              # prints Part A, does nothing else
#   bash scripts/migrate_cluster_rename.sh --docs-pass   # runs Part B (local sed + commit)
#
# See docs/REBRAND_MIGRATION_CHECKLIST.md §2 and §3 for the source plan.

set -euo pipefail

MODE="${1:-}"

# =========================================================================
# PART A — cluster-side rename (author runs these BY HAND on the cluster)
# =========================================================================
print_part_a() {
    cat <<'CLUSTER_COMMANDS'
=========================================================================
PART A — run these commands YOURSELF on the cluster (login node), in
order. This script does NOT ssh anywhere or execute any of this — it is
printed for you to copy/paste or read before acting.
=========================================================================

--- A0. Queue guard: abort if ANY job is RUNNING or PENDING -------------
# The ONE-repo rule means an in-flight job is still reading the old repo
# path. In particular, venue-transfer array 6259842 (see RUNBOOK-9 /
# 2026-08-11 addendum) must be fully DRAINED and its results RETRIEVED to
# the dev box before you touch the cluster checkout. Do not proceed if
# this shows anything:

    squeue -u "$USER"

# Expected: empty (or only COMPLETED/FAILED entries, which squeue won't
# even show — if squeue -u $USER returns any row, STOP and either wait
# for it to finish or scancel it deliberately, then re-check).

--- A1. Rename the cluster checkout -------------------------------------

    mv ~/MasterThesisCode ~/darksiren-emri
    cd ~/darksiren-emri

--- A2. Point the remote at the renamed GitHub repo ---------------------
# The GitHub-side rename (Phase 3) redirects the old remote URL, so this
# step is technically optional — do it explicitly anyway so `git remote
# -v` reads correctly and nobody has to rely on the redirect surviving.

    git remote set-url origin git@github.com:JasperSeehofer/darksiren-emri.git
    git remote -v   # verify it now points at darksiren-emri

--- A3. Pull to confirm the renamed remote resolves ---------------------

    git pull

--- A4. Rebuild the venv -------------------------------------------------
# The cluster venv was built against the old package name (pre Phase-1
# package rename) or is now sitting under a path that no longer matches
# its recorded interpreter symlinks. Rebuild it — do not reuse it.

    source cluster/modules.sh
    uv sync --extra gpu

--- A5. Hand-audit: workspace symlinks / DATA_INVENTORY paths -----------
# (checklist §2.5) These live on the cluster filesystem or in workspace-
# relative manifests, NOT in git — a blind find-replace across tracked
# files (Part B, below) will not reach them. Check by hand:
#   - Any symlink under $WORKSPACE (ws_find emri) that embeds
#     "MasterThesisCode" as a literal path component.
#   - DATA_INVENTORY.md entries that reference cluster-absolute paths
#     containing "MasterThesisCode".
# Fix any hits found; there is no script for this step by design.

--- A6. Smoke-test before trusting the rename ----------------------------
# Submit one small job from the renamed checkout to confirm PROJECT_ROOT,
# module loads, and the rebuilt venv all resolve correctly, e.g.:
#   sbatch --array=1-2 cluster/gpu_smoke.sbatch
# Confirm it completes, THEN run the docs pass (Part B) below and tick
# the §2 verification-gate box in docs/REBRAND_MIGRATION_CHECKLIST.md.

=========================================================================
Once Part A above is done and verified, run this script again with
--docs-pass to apply the §2.4 / §3 local documentation fixes and commit.
=========================================================================
CLUSTER_COMMANDS
}

# =========================================================================
# PART B — local docs pass (only runs with --docs-pass)
# =========================================================================
run_part_b() {
    echo "=== migrate_cluster_rename.sh --docs-pass — checklist §2.4 / §3 ==="
    echo
    echo "PRECONDITION (not verified by this script — you must confirm it"
    echo "yourself): Part A has completed on the cluster — the repo is"
    echo "renamed to ~/darksiren-emri, the remote points at the renamed"
    echo "GitHub repo, and the venv has been rebuilt."
    echo

    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "$REPO_ROOT"

    OLD_NAME="MasterThesisCode"
    NEW_NAME="darksiren-emri"
    OLD_LOCAL="/home/jasper/Repositories/MasterThesisCode"
    NEW_LOCAL="/home/jasper/Repositories/darksiren-emri"

    TOUCHED=()

    # --- §2.4: cluster-path references (~/MasterThesisCode etc.) --------
    # Word-level replace: these files use the bare directory name in
    # ~/MasterThesisCode, $HOME/MasterThesisCode, and (SKILL.md:46) the
    # rsync target bwunicluster:MasterThesisCode/darksiren_emri/... .
    CLUSTER_FILES=(
        "cluster/JOB_TEMPLATE.sbatch"
        "cluster/evaluate_production_h0p73_superdense.sbatch"
        "cluster/calibration_gate_v2.sbatch"
        "cluster/combine.sbatch"
        "cluster/evaluate_densecore.sbatch"
        "cluster/evaluate_closure_h_true_finegrid.sbatch"
        "cluster/merge.sbatch"
        "cluster/evaluate_closure_h065_finegrid.sbatch"
        "cluster/gpu_smoke.sbatch"
        "cluster/README.md"
        "cluster/evaluate_closure_h065.sbatch"
        "cluster/campaign_orchestrator.sh"
        "cluster/venue_transfer.sbatch"
        "cluster/evaluate.sbatch"
        "cluster/preflight.sh"
        "cluster/inject.sbatch"
        "cluster/simulate.sbatch"
        "cluster/evaluate_production_h0p73_dense.sbatch"
        "cluster/LAUNCHING_JOBS.md"
        "cluster/cluster.env"
        ".claude/rules/hpc-gpu.md"
        ".claude/skills/cluster/SKILL.md"
    )
    for f in "${CLUSTER_FILES[@]}"; do
        if [ -f "$f" ]; then
            if grep -q "$OLD_NAME" "$f"; then
                sed -i "s|$OLD_NAME|$NEW_NAME|g" "$f"
                echo "  fixed: $f"
                TOUCHED+=("$f")
            else
                echo "  WARNING: $f has no occurrences of $OLD_NAME — skipping (already fixed?)."
            fi
        else
            echo "  WARNING: expected file not found, skipping: $f"
        fi
    done

    # --- §2.4: cluster/datasets.yaml dev_box_repo ------------------------
    # This one is the LOCAL dev-box path, not a cluster path — it depends
    # on §1 (local rename) having also happened. Only touch it if the new
    # local path actually exists; otherwise warn and leave it for a later
    # pass once §1 has landed.
    DATASETS="cluster/datasets.yaml"
    if [ -f "$DATASETS" ]; then
        if [ -d "$NEW_LOCAL" ]; then
            if grep -q "$OLD_LOCAL" "$DATASETS"; then
                sed -i "s|$OLD_LOCAL|$NEW_LOCAL|g" "$DATASETS"
                echo "  fixed: $DATASETS (dev_box_repo)"
                TOUCHED+=("$DATASETS")
            else
                echo "  WARNING: $DATASETS has no occurrences of $OLD_LOCAL — skipping."
            fi
        else
            echo "  SKIPPED: $DATASETS dev_box_repo left as-is — $NEW_LOCAL does not"
            echo "           exist yet, meaning §1 (local rename) hasn't landed."
            echo "           Re-run --docs-pass after scripts/migrate_local_rename.sh."
        fi
    else
        echo "  WARNING: expected file not found, skipping: $DATASETS"
    fi

    # --- §3 carry-over: bias_investigation cluster-path lines ------------
    SWEEP="scripts/bias_investigation/run_multi_truth_sweep.sh"
    if [ -f "$SWEEP" ]; then
        if grep -q "$OLD_NAME" "$SWEEP"; then
            sed -i "s|$OLD_NAME|$NEW_NAME|g" "$SWEEP"
            echo "  fixed: $SWEEP"
            TOUCHED+=("$SWEEP")
        else
            echo "  WARNING: $SWEEP has no occurrences of $OLD_NAME — skipping."
        fi
    else
        echo "  WARNING: expected file not found, skipping: $SWEEP"
    fi
    echo

    # --- commit -----------------------------------------------------------
    if [ "${#TOUCHED[@]}" -eq 0 ]; then
        echo "No files were touched — nothing to commit."
        return 0
    fi

    git add "${TOUCHED[@]}"
    if git diff --cached --quiet; then
        echo "Staged files produced no diff — nothing to commit."
        return 0
    fi

    git commit -m "$(cat <<'EOF'
docs(cluster): cluster paths -> ~/darksiren-emri (migration §2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Vf92KFbg1F213zhYZKAKR2
EOF
)"
    echo "committed."
    echo
    echo "Next: tick the §2 verification-gate box in"
    echo "docs/REBRAND_MIGRATION_CHECKLIST.md once a cluster job submitted"
    echo "from ~/darksiren-emri with the rebuilt venv has completed."
}

if [ "$MODE" = "--docs-pass" ]; then
    run_part_b
else
    print_part_a
fi
