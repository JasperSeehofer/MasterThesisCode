#!/usr/bin/env bash
# cluster/submit_a18.sh — prints (does not execute) the exact sbatch submission line for the A18
# production arm: mass-aware 1D catalogue leg ("on"), iiib venue, extended 55-node grid.
#
# **Launched under rows #278/#282/#284 — A18 ratified; AMENDMENT G-EXT 55-node grid.**
# BUILDER NOTE: this script is a DRY-RUN printer, not a submitter, exactly like
# cluster/submit_wave3.sh. It does not call sbatch. No ssh/sbatch was run by this builder pass;
# the orchestrator reviews the printed command, works through the pre-launch checklist below, and
# only then removes the DRY_RUN guard (or copies the printed line by hand) to actually submit.
#
# Registrations:
#   results/campaign51_20260728/realistic_20260729/tree2_20260830/
#     PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md
#     §2 (instrument spec), §6.1/§6.3 (flip rule: map_h AND mean_h), AMENDMENT G-EXT (55-node
#     grid, appended 2026-08-31 under row #284)
#   BAND_REDERIVATION_20260831.md §4 (measured band: MAP 0.66 [0.65, 0.67], mean_h 0.652-0.673)
#   WAVE3_A14_DELTA_READ_20260831.md (2026-08-31 correction note: frozen T0 gradient-weighted
#     scorer convention)
#
# Script delivered (under cluster/):
#   a18_ma1d_headreadout_iiib.sbatch  -- 55-task array (H_GRID_41 + G-EXT wing), production
#                                         defaults + --catalogue_leg_1d_mass_aware on, iiib venue.
#
# Out-root: run_20260831_a18_ma1d_iiib -- name fixed by this delivery; if an archive-schedule
# ITEMS block needs it added, that is a separate step (see checklist item 4 below).

set -euo pipefail

# DRY_RUN=1 (default): print the sbatch line without executing it. Set DRY_RUN=0 in the
# environment to actually submit (after the orchestrator has completed the pre-launch checklist
# below). This script never submits by accident.
DRY_RUN="${DRY_RUN:-1}"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
cd "$PROJECT_ROOT"

if [[ -z "${WORKSPACE:-}" ]]; then
    echo "NOTE: \$WORKSPACE not set in this (local) shell — run this script on the cluster after"
    echo "      'source cluster/modules.sh', or export WORKSPACE manually for a dry-run preview."
    WORKSPACE='$WORKSPACE'  # left as a literal placeholder for a local dry-run preview
fi

IIIB_RUN_DIR="$WORKSPACE/run_20260831_a18_ma1d_iiib"

run_or_print() {
    # $1 = human label, remaining args = the sbatch command
    local label="$1"; shift
    echo ""
    echo "# --- $label ---"
    echo "$*"
    if [[ "$DRY_RUN" != "1" ]]; then
        "$@"
    fi
}

echo "=== A18 production-arm submission set (DRY_RUN=$DRY_RUN) — launched under rows #278/#282/#284 ==="
echo ""
echo "Pre-launch checklist (orchestrator, before flipping DRY_RUN=0) — every item is a hard"
echo "precondition, not a suggestion:"
echo "  1. Cluster HEAD matches local: ssh bwunicluster 'git -C ~/darksiren-emri rev-parse HEAD'"
echo "     equals \$(git rev-parse HEAD) here, byte-for-byte, and git status --porcelain is empty"
echo "     on both sides (A22 dirty-state stamp). The A18 instrument (row #255 tree 2 node T2.3,"
echo "     --catalogue_leg_1d_mass_aware) must be present on the cluster checkout."
echo "  2. ssh bwunicluster 'bash -s' < cluster/preflight.sh  ->  VERDICT: READY ✓"
echo "  3. Fresh out-root verified absent on the cluster this session (no idempotency collision):"
echo "     ssh bwunicluster \"test -e \$WORKSPACE/run_20260831_a18_ma1d_iiib && echo EXISTS ||"
echo "     echo ABSENT\" -- must print ABSENT before submitting."
echo "  4. Dataset pins (also re-checked by the sbatch's own STOP-gate at run start): CRB md5"
echo "     9a1f2a14384a9281c97ca3be312ddaab; catalogue md5 c52c13b5cab61f6b3f04bbe202550969."
echo "  5. Archive-schedule note: this run is NOT YET listed in"
echo "     results/_archive/archive_run_wave2.sh's ITEMS block (or its successor archive script)."
echo "     Add run_20260831_a18_ma1d_iiib before submission, or confirm a manual retrieval +"
echo "     archive plan with the orchestrator -- workspace expiry / extension policy per"
echo "     cluster/SKILL.md gotcha list."
echo "  6. Flip-rule band of record for this readout's eventual use: map_h AND mean_h against the"
echo "     MEASURED band MAP 0.66 [0.65, 0.67], mean_h 0.652-0.673"
echo "     (BAND_REDERIVATION_20260831.md §4, per row #284) -- section 6.3 of"
echo "     PHYSICS_CHANGE_MASS_AWARE_1D_LEG_20260830.md. Scorer: the frozen T0 gradient-weighted"
echo "     convention (2026-08-31 correction note, WAVE3_A14_DELTA_READ_20260831.md). State this"
echo "     band when the readout is reported; do not let a downstream reader infer a different"
echo "     threshold or a mean_h-only / map_h-only rule."

mkdir -p "$IIIB_RUN_DIR/logs" 2>/dev/null || true

# --- iiib : A18 production arm, extended 55-node grid array ---
run_or_print "A18 production arm, iiib, mass-aware 1D catalogue leg ON (array 0-54)" \
    sbatch --parsable \
    --array=0-54 \
    --export=ALL,RUN_DIR="$IIIB_RUN_DIR" \
    cluster/a18_ma1d_headreadout_iiib.sbatch

echo ""
echo "=== monitor ==="
echo "squeue -u \$USER"
echo "sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode"

echo ""
echo "=== retrieve (after completion) ==="
echo "mkdir -p results/campaign51_20260728/realistic_20260729/tree2_20260830/a18_prod_arm/"
echo "rsync -avz bwunicluster:$IIIB_RUN_DIR/ results/campaign51_20260728/realistic_20260729/tree2_20260830/a18_prod_arm/"
