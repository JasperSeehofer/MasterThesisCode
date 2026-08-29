#!/usr/bin/env bash
# cluster/submit_wave2.sh — prints (does not execute) the exact sbatch submission lines for
# wave-2 charter nodes C0/C3/C4 (charter node C1 is commented out — NOT ready, see
# cluster/wave2_c1_s0b_TEMPLATE.sbatch header).
#
# **Launched under rows #222/#223 — charter nodes C0/B5.2/B7.2/B1.2.**
# BUILDER NOTE: this script is a DRY-RUN printer, not a submitter. It does not call sbatch. The
# orchestrator reviews the printed commands (and cluster/WAVE2_SUBMISSION_NOTE_20260829.md),
# confirms the wave-2 commit hash + A22 dirty-state-clean stamp, confirms preflight
# `VERDICT: READY ✓`, confirms COMPUTE_LEDGER.md's archive-scheduled cells read "yes" (they do,
# per COMPUTE_LEDGER.md:99-102, GAP-6 closure), and only then removes the DRY_RUN guard (or
# copies the printed lines by hand) to actually submit.
#
# Registrations:
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/REGISTRATION_C0_BASELINE_GATE_20260829.md
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/PREREGISTRATION_WIN_K3_COUNTERFACTUAL_20260829.md
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/PROPOSAL_2D_TWIN_ADOPTION_20260829.md §6.2/§13.3
#   results/campaign51_20260728/realistic_20260729/fanout1_20260829/WAVE2_REGISTRATION_CHECK_20260829.md
#
# Submission ORDER (per the orchestrator path decision of record, WAVE2_REGISTRATION_CHECK
# §4 item 9, and the C4 STEP-2 smoke pattern, proposal §13.2):
#   1. C0 (single task, baseline gate) — must run/PASS-check before C3/C4's baseline reuse is
#      trusted, but C3/C4's own arm-T tasks do NOT wait on it (they measure their own arm
#      regardless; only the BASELINE-REUSE decision depends on C0).
#   2. C3 arm-T array (4 tasks, 0-3) — submitted in the same set as C0/C4's smoke, no dependency
#      on C0 (per task-brief instruction: "C0 + C3 + C4 in one submission set").
#   3. C4 arm-T, task 0 ONLY (h=0.730, the STEP-2 smoke) — submitted in the same set.
#   4. C4 arm-T, tasks 1-3 — submitted as a DEPENDENT array, --dependency=afterok on the job ID
#      from step 3, so the smoke's wall-time is observed before committing the other three
#      tasks' walltime budget (task-brief instruction).
#   5. C1 (4 theta-node array) — COMMENTED OUT. Not submitted this wave (PA-HIER-31 unauthored,
#      P6 not committed — see wave2_c1_s0b_TEMPLATE.sbatch header).
#
# All four RUN_DIRs use the exact out-root names cluster/../results/_archive/archive_run_wave2.sh
# expects (run_20260829_wave2_c{0,1,3,4}_iiib) — do not rename.

set -euo pipefail

# DRY_RUN=1 (default): print every sbatch line without executing it. Set DRY_RUN=0 in the
# environment to actually submit (after the orchestrator has verified preflight + A22 + the
# ledger's archive-scheduled cells, per the header note above). This script never submits by
# accident.
DRY_RUN="${DRY_RUN:-1}"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
cd "$PROJECT_ROOT"

if [[ -z "${WORKSPACE:-}" ]]; then
    echo "NOTE: \$WORKSPACE not set in this (local) shell — run this script on the cluster after"
    echo "      'source cluster/modules.sh', or export WORKSPACE manually for a dry-run preview."
    WORKSPACE='$WORKSPACE'  # left as a literal placeholder for a local dry-run preview
fi

C0_RUN_DIR="$WORKSPACE/run_20260829_wave2_c0_iiib"
C1_RUN_DIR="$WORKSPACE/run_20260829_wave2_c1_iiib"
C3_RUN_DIR="$WORKSPACE/run_20260829_wave2_c3_iiib"
C4_RUN_DIR="$WORKSPACE/run_20260829_wave2_c4_iiib"

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

echo "=== wave-2 submission set (DRY_RUN=$DRY_RUN) — launched under rows #222/#223 ==="
echo "Pre-launch checklist (orchestrator, before flipping DRY_RUN=0):"
echo "  1. ssh bwunicluster 'bash -s' < cluster/preflight.sh  ->  VERDICT: READY (check)"
echo "  2. wave-2 commit exists; git status clean (A22 dirty-state stamp)"
echo "  3. COMPUTE_LEDGER.md archive-scheduled cells (C0/C3/C4) read yes -- confirmed at line 99-102"
echo "  4. Dataset pins: CRB md5 9a1f2a14384a9281c97ca3be312ddaab; catalogue md5 c52c13b5cab61f6b3f04bbe202550969"
echo "     (also re-checked by each sbatch's own STOP-gate at run start)"

mkdir -p "$C0_RUN_DIR/logs" "$C3_RUN_DIR/logs" "$C4_RUN_DIR/logs" 2>/dev/null || true

# --- 1. C0 : shared baseline gate task (single task, no array) ---
run_or_print "C0 baseline gate (single task)" \
    sbatch --parsable \
    --export=ALL,RUN_DIR="$C0_RUN_DIR" \
    cluster/wave2_c0_baseline.sbatch

# --- 2. C3 : log k=3 window counterfactual, arm T, H4 grid (4 tasks) ---
run_or_print "C3 log-k3 window counterfactual, arm T (array 0-3)" \
    sbatch --parsable \
    --array=0-3 \
    --export=ALL,RUN_DIR="$C3_RUN_DIR" \
    cluster/wave2_c3_win_k3.sbatch

# --- 3. C4 : PROD-CF-2D mz_sel/eff, arm T, task 0 ONLY = STEP-2 smoke (h=0.730) ---
echo ""
echo "# --- C4 PROD-CF-2D STEP-2 smoke (h=0.730 only, array 0-0) ---"
echo "sbatch --parsable --array=0-0 --export=ALL,RUN_DIR=$C4_RUN_DIR cluster/wave2_c4_twin_mz_sel.sbatch"
if [[ "$DRY_RUN" == "1" ]]; then
    C4_SMOKE_JOBID="<jobid-c4-smoke>"
    echo "# -> job id: $C4_SMOKE_JOBID (DRY_RUN=1, not submitted)"
else
    C4_SMOKE_JOBID=$(sbatch --parsable \
        --array=0-0 \
        --export=ALL,RUN_DIR="$C4_RUN_DIR" \
        cluster/wave2_c4_twin_mz_sel.sbatch)
    echo "# -> job id: $C4_SMOKE_JOBID"
fi

# --- 4. C4 : remaining 3 tasks, DEPENDENT array (afterok on the smoke) ---
run_or_print "C4 PROD-CF-2D remaining tasks (array 1-3, dependency=afterok:$C4_SMOKE_JOBID)" \
    sbatch --parsable \
    --array=1-3 \
    --dependency="afterok:$C4_SMOKE_JOBID" \
    --export=ALL,RUN_DIR="$C4_RUN_DIR" \
    cluster/wave2_c4_twin_mz_sel.sbatch

echo ""
echo "=== C1 (S0-B theta-score) — COMMENTED OUT, NOT submitted this wave ==="
echo "# Blocked on: PA-HIER-31 ratification + P6 (theta CLI plumbing) commit."
echo "# Once both land, the equivalent line is:"
echo "#   sbatch --parsable --array=0-3 --export=ALL,RUN_DIR=$C1_RUN_DIR cluster/wave2_c1_s0b_TEMPLATE.sbatch"

echo ""
echo "=== monitor ==="
echo "squeue -u \$USER"
echo "sacct -j <jobids> --format=JobID,State,Elapsed,MaxRSS,ExitCode"

echo ""
echo "=== retrieve (after completion) ==="
echo "rsync -avz bwunicluster:$C0_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave2_20260829/c0/"
echo "rsync -avz bwunicluster:$C3_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave2_20260829/c3/"
echo "rsync -avz bwunicluster:$C4_RUN_DIR/ results/campaign51_20260728/realistic_20260729/wave2_20260829/c4/"
echo "(see cluster/WAVE2_SUBMISSION_NOTE_20260829.md for the full retrieval + dataset-registration plan)"
