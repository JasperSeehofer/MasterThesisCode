#!/usr/bin/env bash
# cluster/write_provenance.sh — single shared definition of "what a provenance
# stamp is" for cluster jobs that bypass `python -m darksiren_emri` (and
# therefore never hit main.py:_write_run_metadata, the only place that writes
# run_metadata*.json today).
#
# WHY: run_metadata*.json ties a result to the exact code that produced it
# (git_commit, seed, timestamp, args). It is written ONLY by the package entry
# point. Bespoke harness drivers invoked directly from bespoke sbatch scripts
# (p3_2d_fleet.sbatch, csg_pilot.sbatch, o4_fleet.sbatch, ...) never call that
# entry point and so never got it — see cluster/SKILL.md gotcha #12.
#
# USAGE (source, then call once near the top of the job, after RUN_DIR/OUT_DIR
# is known but before the heavy work starts):
#   source "$PROJECT_ROOT/cluster/write_provenance.sh"
#   write_provenance "$RUN_DIR" "seed=$TASK_SEED arm=$ARM"   # 2nd arg optional
#
# Writes "$1/provenance_${SLURM_JOB_ID:-nojob}_${SLURM_ARRAY_TASK_ID:-0}.json".
# Fails SOFT: git unavailable / weird output dir => log a warning, do not
# fail the job (a missing stamp must never be worse than a job that didn't run).

write_provenance() {
    local out_dir="${1:?write_provenance: output dir required}"
    local extra="${2:-}"
    local cmd="${3:-${BASH_SOURCE[1]:-unknown} ${SLURM_JOB_NAME:-unknown}}"

    if [[ ! -d "$out_dir" ]]; then
        mkdir -p "$out_dir" 2>/dev/null || {
            echo "WARNING: write_provenance: cannot create/access '$out_dir' — skipping stamp" >&2
            return 0
        }
    fi

    local proj_root="${PROJECT_ROOT:-$HOME/darksiren-emri}"
    local commit="unknown"
    local dirty="unknown"
    local branch="unknown"

    if command -v git >/dev/null 2>&1 && git -C "$proj_root" rev-parse --git-dir >/dev/null 2>&1; then
        commit=$(git -C "$proj_root" rev-parse HEAD 2>/dev/null || echo "unknown")
        dirty=$(git -C "$proj_root" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
        branch=$(git -C "$proj_root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
        # If HEAD is detached at a tag, prefer the tag name (branch will read "HEAD").
        if [[ "$branch" == "HEAD" ]]; then
            branch=$(git -C "$proj_root" describe --tags --exact-match 2>/dev/null || echo "HEAD-detached")
        fi
    else
        echo "WARNING: write_provenance: git unavailable or '$proj_root' is not a git repo — commit/dirty/branch will read 'unknown'" >&2
    fi

    local job_id="${SLURM_JOB_ID:-none}"
    local array_task_id="${SLURM_ARRAY_TASK_ID:-none}"
    local array_job_id="${SLURM_ARRAY_JOB_ID:-none}"
    local seed="${TASK_SEED:-${BASE_SEED:-none}}"
    local host
    host=$(hostname 2>/dev/null || echo "unknown")
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")

    local out_file="$out_dir/provenance_${job_id}_${array_task_id}.json"

    # Best-effort JSON (no jq dependency); values are shell-controlled except
    # $extra/$cmd, which we escape minimally (backslash + double-quote).
    local extra_esc cmd_esc
    extra_esc=$(printf '%s' "$extra" | sed 's/\\/\\\\/g; s/"/\\"/g')
    cmd_esc=$(printf '%s' "$cmd" | sed 's/\\/\\\\/g; s/"/\\"/g')

    if ! cat > "$out_file" <<EOF
{
  "git_commit": "$commit",
  "git_branch": "$branch",
  "tree_dirty_file_count": "$dirty",
  "slurm_job_id": "$job_id",
  "slurm_array_job_id": "$array_job_id",
  "slurm_array_task_id": "$array_task_id",
  "seed": "$seed",
  "hostname": "$host",
  "start_timestamp_utc": "$ts",
  "command": "$cmd_esc",
  "note": "$extra_esc"
}
EOF
    then
        echo "WARNING: write_provenance: failed to write '$out_file' — continuing without a stamp" >&2
        return 0
    fi

    echo "provenance stamp: $out_file (commit=$commit dirty=$dirty job=$job_id/$array_task_id)"
    return 0
}
