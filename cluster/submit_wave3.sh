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
# Scripts delivered (all under cluster/):
#   wave3_c0prime_off_gate.sbatch      -- 2-task array (task 0=iiib, task 1=joint_r1), h=0.730
#                                          only, explicit "--catalogue_numerator_survival_2d off
#                                          --catalogue_numerator_survival_2d_center unset" -- the
#                                          A14 falsifier BASELINE gate ("C0'"): certifies the
#                                          banked 2026-08-27 readouts as the pre-adoption baseline
#                                          without a full 82-task off-array. Submitted FIRST.
#   wave3_headreadout_iiib.sbatch      -- 41-task array (H_GRID_41), production defaults, BLIND
#                                          to the 2D-twin adoption (no
#                                          --catalogue_numerator_survival_2d flag).
#   wave3_headreadout_joint_r1.sbatch  -- same, joint_r1 venue (+ --observed_catalogue realization,
#                                          + its own sha256 STOP-gate).
#
# Out-roots use the exact names results/_archive/archive_run_wave2.sh's "wave 3" ITEMS block
# already expects (run_20260830_wave3_headreadout_{iiib,joint_r1},
# run_20260830_wave3_c0prime_off_{iiib,joint_r1}) -- do not rename.
#
# WHAT THIS READOUT IS AND IS NOT (repeat of the sbatch headers, stated here for the reviewer):
# the blind HEAD readout (headreadout_{iiib,joint_r1}) is the ONE blind full-grid readout; it
# does NOT itself compute the per-change delta or the A14 falsifier verdict against T_mat = 0.008.
# The c0prime_off_gate is NOT that missing counterfactual arm either -- it is a much cheaper
# single-h (h=0.730) reproduction gate that certifies the ALREADY-BANKED 2026-08-27 readouts as
# usable in place of a full 82-task off-array for that same delta read (see the gate's own
# PASS/FAIL consequence in its script header and in WAVE3_SUBMISSION_NOTE_20260830.md's
# "C0' off-gate" section). Submitting all three scripts here still does not by itself compute
# the delta or the falsifier verdict -- it produces the two ingredients (blind HEAD posteriors +
# gate verdict) the orchestrator combines to answer "was the adoption material".

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
# c0prime_off_gate computes its own per-task RUN_DIR from WORKSPACE + venue name internally
# (see that script's header) -- no RUN_DIR export needed for it. Named here only for the
# pre-launch "fresh out-roots absent" check (item 8) and the retrieval section below.
C0PRIME_IIIB_RUN_DIR="$WORKSPACE/run_20260830_wave3_c0prime_off_iiib"
C0PRIME_JOINT_R1_RUN_DIR="$WORKSPACE/run_20260830_wave3_c0prime_off_joint_r1"

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
echo "     block (appended this pass) lists all four out-roots below (2 blind-readout +"
echo "     2 c0prime_off_gate); confirm the archive step will actually run post-retrieval"
echo "     (Option A, MUST-ARCHIVE tier, workspace expires 2026-09-23, 0 extensions)."
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
echo "     |Delta-mean_h| (2D channel), BOTH venues, evaluated against the pre-adoption baseline"
echo "     -- the banked 2026-08-27 readouts IF the c0prime_off_gate PASSES (item 9), else the"
echo "     full '--catalogue_numerator_survival_2d off' counterfactual array (not built by this"
echo "     delivery) -- PHYSICS_CHANGE_2D_TWIN_ADOPTION_20260829.md §8. State this band when the"
echo "     readout is reported; do not let a downstream reader infer a different threshold."
echo "  8. Fresh out-roots verified absent on the cluster this session (no idempotency"
echo "     collision) -- re-check immediately before submitting, since this checklist was"
echo "     authored without cluster access."
echo "  9. c0prime_off_gate band (its own PASS/FAIL, checked AFTER it completes, before reading"
echo "     the blind HEAD readout against the banked baseline): max |relative difference| <= 1e-12"
echo "     on the 14 diagnostic event_likelihoods.csv columns + md5-identical posterior JSONs, at"
echo "     h=0.730, both venues, vs. headreadout_20260827/{iiib,joint_r1} task 21 --"
echo "     REGISTRATION_C0_BASELINE_GATE_20260829.md §3/§13 (gate band + RESULT RECORD form)."
echo "     FAIL on either venue means the full 82-task off-array becomes necessary; do not"
echo "     proceed to the A14 delta read on that venue until diagnosed."

mkdir -p "$IIIB_RUN_DIR/logs" "$JOINT_R1_RUN_DIR/logs" \
         "$C0PRIME_IIIB_RUN_DIR/logs" "$C0PRIME_JOINT_R1_RUN_DIR/logs" 2>/dev/null || true

# --- 0. c0prime_off_gate : A14 falsifier BASELINE gate, both venues, h=0.730 only (array 0-1).
#        Submitted FIRST, ahead of the two blind arrays -- cheap (cost ~2+4 CPU-h) and its
#        PASS/FAIL result is a precondition for trusting the blind readout's eventual delta read
#        against the banked 2026-08-27 rows (see WAVE3_SUBMISSION_NOTE_20260830.md "C0' off-gate"
#        section). No RUN_DIR export -- the script computes its own per-task out-root. ---
run_or_print "wave-3 A14 falsifier baseline gate, c0prime_off (array 0-1: iiib, joint_r1)" \
    sbatch --parsable \
    --array=0-1 \
    cluster/wave3_c0prime_off_gate.sbatch

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
echo "mkdir -p results/campaign51_20260728/realistic_20260729/wave3_20260830/{iiib,joint_r1,c0prime_off_iiib,c0prime_off_joint_r1}"
echo "rsync -avz bwunicluster:$IIIB_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave3_20260830/iiib/"
echo "rsync -avz bwunicluster:$JOINT_R1_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave3_20260830/joint_r1/"
echo "rsync -avz bwunicluster:$C0PRIME_IIIB_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave3_20260830/c0prime_off_iiib/"
echo "rsync -avz bwunicluster:$C0PRIME_JOINT_R1_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave3_20260830/c0prime_off_joint_r1/"
echo "(see cluster/WAVE3_SUBMISSION_NOTE_20260830.md for the full combine + dataset-registration plan, and its \"C0' off-gate\" section for the gate band/verdict procedure)"
