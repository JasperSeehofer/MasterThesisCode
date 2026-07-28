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
# Stop:    pkill -f 'campaign_orchestrator[.]sh'   # bracket idiom: don't match the ssh wrapper shell

set -u

# PROJECT_ROOT can be overridden so the script may run from a copy outside
# the repo (e.g. staged on the workspace when $HOME git sync is unavailable).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"
cd "$PROJECT_ROOT"

WS="${WORKSPACE:-$(ws_find emri 2>/dev/null)}"
WS="${WS:-/pfs/work9/workspace/scratch/st_ac147838-emri}"
# Pool + pipeline list are env-overridable (campaign #51, 2026-07-28) so new
# campaigns don't require editing this script:
#   POOL_OVERRIDE=$WS/injection_pool_mix200k_20260728 \
#   PIPELINES_SPEC="61000:0.73 62000:0.73 63000:0.73 64000:0.67 65000:0.77" \
#       bash cluster/campaign_orchestrator.sh
POOL="${POOL_OVERRIDE:-$WS/injection_pool_depth15_50k}"
MAIN_LOG="$WS/campaign_orchestrator.log"
SUB_LOG="$WS/campaign_orchestrator_submissions.log"

# Campaign pipelines, submitted in order: "BASE_SEED:h_true".
# Default = the phase-2 list (seed1000 @ 0.73 was submitted manually
# 2026-07-03, jobs 5743694-97); override via PIPELINES_SPEC.
read -ra PIPELINES <<< "${PIPELINES_SPEC:-2000:0.73 3000:0.73 4000:0.73 5000:0.67 6000:0.77}"

# Submit the next 143-job pipeline only below this expanded-array queue depth.
# Conservative: the per-user cap sits somewhere in (294, 544]; 150 + 143 stays
# clear even at the low end.
MAX_PENDING=150
POLL_S=300
# 2026-07-03: the $HOME Lustre filesystem intermittently returns EIO on reads
# of recently-written files (git pull unpack failures, one failed read of
# submit_pipeline.sh). Probe readability before each attempt and retry the
# SAME seed with backoff — never advance past a failed seed.
MAX_ATTEMPTS_PER_SEED=500

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

    attempt=0
    while :; do
        attempt=$((attempt + 1))
        if [[ "$attempt" -gt "$MAX_ATTEMPTS_PER_SEED" ]]; then
            log "seed $seed: gave up after $MAX_ATTEMPTS_PER_SEED attempts — manual submission needed"
            break
        fi

        # Wait for queue headroom (squeue -r expands array elements, matching
        # how the submit cap counts).
        n=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)
        if [[ -z "$n" || "$n" -ge "$MAX_PENDING" ]]; then
            sleep "$POLL_S"
            continue
        fi

        # $HOME EIO probe: all files the submission path reads.
        if ! cat cluster/submit_pipeline.sh cluster/simulate.sbatch \
                 cluster/merge.sbatch cluster/evaluate.sbatch \
                 cluster/combine.sbatch cluster/modules.sh \
                 > /dev/null 2>&1; then
            log "seed $seed: \$HOME read probe FAILED (attempt $attempt) — backing off"
            sleep "$POLL_S"
            continue
        fi

        log "seed $seed (h_true=$ht): submitting at queue depth $n (attempt $attempt)"
        if bash cluster/submit_pipeline.sh \
                --tasks 100 --steps 40 --seed "$seed" \
                --h_true "$ht" --injection_pool "$POOL" >> "$SUB_LOG" 2>&1; then
            log "seed $seed: submitted OK"
            break
        fi

        log "seed $seed: submission attempt $attempt FAILED (see $SUB_LOG) — cleaning any empty run dir, retrying"
        # Defensive: a failure after submit_pipeline's mkdir would poison the
        # idempotency check on restart. Remove run dirs for this seed that
        # contain no job logs yet.
        for d in "$WS"/run_*_seed"${seed}" "$WS"/run_*_seed"${seed}"_*; do
            [[ -d "$d" ]] || continue
            if ! compgen -G "$d/logs/*" > /dev/null; then
                log "seed $seed: removing unsubmitted run dir $d"
                rm -rf "$d"
            fi
        done
        sleep "$POLL_S"
    done
    sleep 60
done

log "orchestrator finished: all pipelines submitted or skipped"
