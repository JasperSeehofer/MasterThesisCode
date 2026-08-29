#!/usr/bin/env bash
# cluster/submit_wave3.sh — prints (does not execute) the exact sbatch submission lines for
# wave-3 charter node B7.3: the ONE blind HEAD readout, both venues (iiib, joint_r1).
#
# **Launched under rows #222/#223 — charter wave 3 / node B7.3 readout.**
# BUILDER NOTE: this script is a DRY-RUN printer, not a submitter, exactly like
# cluster/submit_wave2.sh. It does not call sbatch. No ssh/sbatch was run by this builder pass
# (cluster access is down); the orchestrator reviews the printed commands (and
# cluster/WAVE3_SUBMISSION_NOTE_20260830.md), works through the pre-launch checklist below, and
# only then removes the DRY_RUN guard (or copies the printed lines by hand) to actually submit.
#
# Registrations:
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md
#     §0.2 (F2 statement), §8 (A14 falsifier, T_mat = 0.008), §9 (ledger rows)
#   results/campaign51_20260728/realistic_20260729/MEASUREMENT_HEAD_READOUT_20260827.md
#     (grid/seeding convention, dataset pins, the structural-blindness precedent this readout
#     mirrors)
#   results/campaign51_20260728/realistic_20260729/headreadout_20260827/{iiib,joint_r1}/run_metadata_21.json
#     (CoR-P CLI source)
#
# Scripts delivered (both under cluster/):
#   wave3_headreadout_iiib.sbatch      -- 41-task array (H_GRID_41), production defaults, BLIND
#                                          to the 2D-twin adoption (no
#                                          --catalogue_numerator_survival_2d flag).
#   wave3_headreadout_joint_r1.sbatch  -- same, joint_r1 venue (+ --observed_catalogue realization,
#                                          + its own sha256 STOP-gate).
#
# Out-roots use the exact names results/_archive/archive_run_wave2.sh's "wave 3" ITEMS block
# already expects (run_20260830_wave3_headreadout_{iiib,joint_r1}) -- do not rename.
#
# WHAT THIS READOUT IS AND IS NOT (repeat of the sbatch headers, stated here for the reviewer):
# this is the ONE blind full-grid HEAD readout. It does NOT itself compute the per-change delta
# or the A14 falsifier verdict against T_mat = 0.008 -- that requires a SEPARATE counterfactual
# arm run with an explicit "--catalogue_numerator_survival_2d off" at the same wave-3 commit,
# which is NOT part of this delivery. Submitting only these two scripts answers "what does HEAD
# say" (F2); it does not by itself answer "was the adoption material".

set -euo pipefail

# DRY_RUN=1 (default): print every sbatch line without executing it. Set DRY_RUN=0 in the
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

IIIB_RUN_DIR="$WORKSPACE/run_20260830_wave3_headreadout_iiib"
JOINT_R1_RUN_DIR="$WORKSPACE/run_20260830_wave3_headreadout_joint_r1"

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

echo "=== wave-3 submission set (DRY_RUN=$DRY_RUN) — launched under rows #222/#223, charter wave 3 / node B7.3 readout ==="
echo ""
echo "Pre-launch checklist (orchestrator, before flipping DRY_RUN=0) — every item is a hard"
echo "precondition, not a suggestion:"
echo "  1. The row-#223 [PHYSICS] adoption commit (catalogue_numerator_survival_2d default"
echo "     'off'->'mz_sel', center 'unset'->'eff', PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md"
echo "     §9.3) is HEAD on this branch AND has been pushed. Verify: git log -1 --oneline"
echo "     (subject should start '[PHYSICS] adopt the with-BH catalogue-leg twin') and"
echo "     git status --porcelain is empty (A22 dirty-state stamp)."
echo "  2. The cluster checkout has PULLED that exact commit -- ssh bwunicluster"
echo "     'git -C ~/darksiren-emri rev-parse HEAD' matches the local HEAD from item 1 byte-for-"
echo "     byte. Do not submit against a cluster checkout that predates the adoption commit: the"
echo "     whole point of this readout (F2) is to read the adopted default, not the pre-adoption"
echo "     one."
echo "  3. ssh bwunicluster 'bash -s' < cluster/preflight.sh  ->  VERDICT: READY ✓"
echo "  4. This run is ARCHIVE-SCHEDULED: results/_archive/archive_run_wave2.sh's 'wave 3' ITEMS"
echo "     block (appended this pass) lists both out-roots below; confirm the archive step will"
echo "     actually run post-retrieval (Option A, MUST-ARCHIVE tier, workspace expires 2026-09-23,"
echo "     0 extensions)."
echo "  5. Dataset pins (also re-checked by each sbatch's own STOP-gate at run start):"
echo "     CRB md5 9a1f2a14384a9281c97ca3be312ddaab; catalogue md5"
echo "     c52c13b5cab61f6b3f04bbe202550969; joint_r1 observed-catalogue sha256"
echo "     e8f7ab310ea70ddfdd3b81970dc99ad943808e6b6c128777bb085db01b4f6751."
echo "  6. gotcha #10 (realization sidecar staleness): verify"
echo "     \$WS/realizations_20260729/observed_catalogue_seed900001.csv.meta.json's 'parent_csv'"
echo "     path still resolves on the cluster (repair per gotcha #10 if the repo has moved since"
echo "     it was written) BEFORE submitting the joint_r1 array -- a stale sidecar fails every"
echo "     joint_r1 task at run start, not just one."
echo "  7. Falsifier band of record for this readout's eventual use: A14, T_mat = 0.008 on"
echo "     |Delta-mean_h| (2D channel), BOTH venues, evaluated against the separate"
echo "     '--catalogue_numerator_survival_2d off' counterfactual arm (not built by this"
echo "     script) -- PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §8. State this band when the"
echo "     readout is reported; do not let a downstream reader infer a different threshold."
echo "  8. Fresh out-roots verified absent on the cluster this session (no idempotency"
echo "     collision) -- re-check immediately before submitting, since this checklist was"
echo "     authored without cluster access."

mkdir -p "$IIIB_RUN_DIR/logs" "$JOINT_R1_RUN_DIR/logs" 2>/dev/null || true

# --- 1. iiib : blind HEAD readout, full H_GRID_41 array ---
run_or_print "wave-3 blind HEAD readout, iiib (array 0-40)" \
    sbatch --parsable \
    --array=0-40 \
    --export=ALL,RUN_DIR="$IIIB_RUN_DIR" \
    cluster/wave3_headreadout_iiib.sbatch

# --- 2. joint_r1 : blind HEAD readout, full H_GRID_41 array ---
run_or_print "wave-3 blind HEAD readout, joint_r1 (array 0-40)" \
    sbatch --parsable \
    --array=0-40 \
    --export=ALL,RUN_DIR="$JOINT_R1_RUN_DIR" \
    cluster/wave3_headreadout_joint_r1.sbatch

echo ""
echo "=== monitor ==="
echo "squeue -u \$USER"
echo "sacct -j <jobids> --format=JobID,State,Elapsed,MaxRSS,ExitCode"

echo ""
echo "=== retrieve (after completion) ==="
echo "mkdir -p results/campaign51_20260728/realistic_20260729/wave3_20260830/{iiib,joint_r1}"
echo "rsync -avz bwunicluster:$IIIB_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave3_20260830/iiib/"
echo "rsync -avz bwunicluster:$JOINT_R1_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave3_20260830/joint_r1/"
echo "(see cluster/WAVE3_SUBMISSION_NOTE_20260830.md for the full combine + dataset-registration plan)"
