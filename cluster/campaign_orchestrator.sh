#!/usr/bin/env bash
# cluster/campaign_orchestrator.sh — Phase-2 staggered pipeline submitter.
#
# Runs detached on the LOGIN node (nohup; ~zero CPU: 5-minute polls) so the
# campaign submission chain does NOT depend on any dev-box SSH connection.
# Submits the remaining campaign pipelines one at a time whenever the user's
# queue depth drops below MAX_PENDING (the per-user submit cap rejected ~544
# pending jobs and accepted ~294 on 2026-07-03, so 250 + one 143-job pipeline
# stays safely under it).
#
# Idempotent: a seed whose run directory already exists is skipped, so the
# script can be re-run (or restarted after a login-node purge) without
# double-submitting. Progress: $WS/campaign_orchestrator.log; raw submission
# output: $WS/campaign_orchestrator_submissions.log.
#
# Launch:
#   cd ~/MasterThesisCode && nohup bash cluster/campaign_orchestrator.sh \
#       > /dev/null 2>&1 & disown
# Status:  tail $(ws_find emri)/campaign_orchestrator.log
# Stop:    pkill -f campaign_orchestrator.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

WS="${WORKSPACE:-$(ws_find emri 2>/dev/null)}"
WS="${WS:-/pfs/work9/workspace/scratch/st_ac147838-emri}"
POOL="$WS/injection_pool_depth15_50k"
MAIN_LOG="$WS/campaign_orchestrator.log"
SUB_LOG="$WS/campaign_orchestrator_submissions.log"

# Remaining campaign pipelines, submitted in order: "BASE_SEED:h_true"
# (seed1000 @ 0.73 was submitted manually 2026-07-03, jobs 5743694-97).
PIPELINES=("2000:0.73" "3000:0.73" "4000:0.73" "5000:0.67" "6000:0.77")

# Submit the next 143-job pipeline only below this expanded-array queue depth.
MAX_PENDING=250
POLL_S=300

log() { echo "[$(date '+%F %T')] $*" >> "$MAIN_LOG"; }

log "orchestrator started (pid $$, host $(hostname)); pipelines: ${PIPELINES[*]}"

if [[ ! -d "$POOL" ]]; then
    log "FATAL: injection pool $POOL missing — aborting"
    exit 1
fi

for spec in "${PIPELINES[@]}"; do
    seed="${spec%%:*}"
    ht="${spec##*:}"

    # Idempotency: any existing run dir for this seed means it was submitted.
    if compgen -G "$WS/run_*_seed${seed}" > /dev/null \
        || compgen -G "$WS/run_*_seed${seed}_*" > /dev/null; then
        log "seed $seed: run dir already exists — skipping"
        continue
    fi

    # Wait for queue headroom (squeue -r expands array elements, matching how
    # the submit cap counts).
    while :; do
        n=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)
        [[ "$n" -lt "$MAX_PENDING" ]] && break
        sleep "$POLL_S"
    done

    log "seed $seed (h_true=$ht): submitting at queue depth $n"
    if bash cluster/submit_pipeline.sh \
            --tasks 100 --steps 40 --seed "$seed" \
            --h_true "$ht" --injection_pool "$POOL" >> "$SUB_LOG" 2>&1; then
        log "seed $seed: submitted OK"
    else
        log "seed $seed: SUBMISSION FAILED (see $SUB_LOG) — continuing to next after one retry window"
        sleep "$POLL_S"
        if bash cluster/submit_pipeline.sh \
                --tasks 100 --steps 40 --seed "$seed" \
                --h_true "$ht" --injection_pool "$POOL" >> "$SUB_LOG" 2>&1; then
            log "seed $seed: retry submitted OK"
        else
            log "seed $seed: retry FAILED — manual submission needed"
        fi
    fi
    sleep 60
done

log "orchestrator finished: all pipelines submitted or skipped"
