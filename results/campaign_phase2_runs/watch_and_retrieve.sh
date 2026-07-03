#!/usr/bin/env bash
# Weekend watcher (2026-07-03): every 30 min, snapshot cluster status and
# rsync campaign run artifacts back to the dev box. Detached via setsid;
# survives session death. Stop: pkill -f "watch_and_retrieve[.]sh"
#
# Pulls: run dirs for smoke (900), campaign seeds (1000..6000) — everything
# except the per-run cwd/ symlink dirs. Also mirrors the orchestrator logs.

WS=/pfs/work9/workspace/scratch/st_ac147838-emri
DEST=/home/jasper/Repositories/MasterThesisCode/results/campaign_phase2_runs
mkdir -p "$DEST"

while :; do
    ts=$(date '+%F %T')
    {
        echo "=== $ts ==="
        ssh -o ConnectTimeout=30 -o BatchMode=yes bwunicluster '
            echo "queue depth (expanded): $(squeue -u $USER -h -r | wc -l)"
            squeue -u $USER -h --format="%.12i %.14j %.8T %.10M" | head -12
            echo "-- orchestrator --"
            tail -3 /pfs/work9/workspace/scratch/st_ac147838-emri/campaign_orchestrator.log 2>/dev/null
        ' 2>&1
    } >> "$DEST/status.log"

    ssh -o ConnectTimeout=30 -o BatchMode=yes bwunicluster \
        "ls -d $WS/run_20260703_seed* $WS/run_20260704_seed* $WS/run_20260705_seed* $WS/run_20260706_seed* 2>/dev/null" \
        2>/dev/null | while read -r d; do
        [ -n "$d" ] || continue
        name=$(basename "$d")
        rsync -az --timeout=120 --exclude=cwd \
            "bwunicluster:$d/" "$DEST/$name/" >> "$DEST/rsync.log" 2>&1
    done

    rsync -az --timeout=60 \
        "bwunicluster:$WS/campaign_orchestrator.log" \
        "bwunicluster:$WS/campaign_orchestrator_submissions.log" \
        "$DEST/" >> "$DEST/rsync.log" 2>&1

    sleep 1800
done
