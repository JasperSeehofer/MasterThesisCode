#!/usr/bin/env bash
# cluster/p3_2d_rhs2.sbatch — [P3-2D] RHS2 array job
# (PREREGISTRATION_P3_2D_20260825.md §2 instrument (iii) + PA-2D-1 F7-F16 + PA-2D-4 item 1).
#
# 16-task array (SLURM_ARRAY_TASK_ID 0-15); per-task base seed = 980001 + TASK_ID
# (results/.../ca_rhs_scorer.py's own RHS2_BASE_SEED_DEFAULT=980001, disjoint from every
# registered campaign seed range and from stage_score's 960001/stage_acceptance's 970001 -- the
# repo-wide BASE_SEED + SLURM_ARRAY_TASK_ID convention, cluster/SKILL.md gotcha 4). Each task
# runs an INDEPENDENT chunked --stage rhs2 pass (its own seed stream, its own out-root) and
# writes ONE task JSON; a final local `--stage rhs2-combine` pass (NOT part of this array — run
# after retrieval, see below) pools every task JSON at the chunk level into the fleet RHS2 +/- SE.
#
# ── Costing (justify from the b0i2d fleet's OWN measured anchor; ~40 CPU-h cap, PA-2D-1 F15) ──
#
# Measured anchor: p3_2d_work/b{c,t}_900101_meta.json wall_time_s = 64.996 / 62.944 s for a
# SINGLE-arm, 200-event run_mirror_seed_inprocess (evaluate()) call at --cpus-per-task=16 (the
# fleet's OWN sbatch cpu reservation, p3_2d_fleet.sbatch) — no independent "32 min/task" figure
# for THIS instrument was found anywhere in the repo at authoring time (grepped
# results/.../*.md, cluster/*.md — none); this sbatch's sizing is derived from the fleet's own
# measured number instead, disclosed here rather than asserted.
#   ~64 s/200-events/arm * 16 cpus         = 1024 cpu-s / (200-event, 1-arm) chunk
#   RHS2 scores BOTH arrangements per chunk = 2048 cpu-s / (200-event, both-arm) chunk
#                                            = 10.24 cpu-s / drawn event (both arms)
# Budget: ~40 CPU-h total (144000 cpu-s) / 16 tasks = 9000 cpu-s/task target (with margin).
#   9000 / 10.24 ~= 878 draws/task -> sized DOWN to --n-draws 800 (chunk-pinned at 200, so
#   4 chunks/task, 800 events drawn exactly) for margin:
#   800 * 10.24 = 8192 cpu-s/task = 2.28 CPU-h/task * 16 tasks = 36.4 CPU-h total (~9% margin
#   under the 40 CPU-h cap).
#
# DISCLOSED SHORTFALL (not silently resolved): PA-2D-1 F15's registered fallback SE target
# (<= eps2/3 = 6.38e-4) needs N ~= 24k with-BH-scored draws. This sbatch's total N = 16*800 =
# 12800 (~53% of that). Expected SE inflates by sqrt(24000/12800) ~= 1.37x relative to the
# fallback-target SE, i.e. ~8.7e-4 > eps2/3 -- if that is what lands, F15 requires an A21 STOP +
# a fresh costing line, NOT a silently widened band. This sbatch does not attempt to reach N~=24k
# within the stated cap; a wider cap needs the author's fresh costing line before re-sizing.
#
# Time: 4 chunks/task * 2 arms * ~64s/arm ~= 512 s (~8.5 min) uncontended, plus host-pool/
# catalogue build overhead (~10-30s once per task, gotcha 6) + the F10(c) reference pool
# (--f10c-reference-n 2000, zero-evaluate — cheap, seconds) + gotcha-6 cross-task
# memory-bandwidth contention when packed on cpu_il. Sized generously (well above the ~10 min
# uncontended estimate) since gotcha 9 (JSON-only-at-end instruments lose everything on a
# walltime kill) makes under-sizing expensive to recover from — THIS instrument checkpoints its
# accumulator JSON every chunk specifically to blunt that risk, but a killed task still loses
# whatever chunks had not yet completed.
#
# Idempotency: --stage rhs2 REFUSES (PA-CA-11) a non-empty out-root (a stale
# rhs2_chunk*_work dir) — a resubmit MUST use a fresh --out-root per task (this script's
# $OUT_ROOT/task$TID), never a shared one across resubmits.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight first, require
# VERDICT: READY):
#   sbatch cluster/p3_2d_rhs2.sbatch
# Combine (after retrieval, local, NOT part of this array job):
#   uv run python results/campaign51_20260728/realistic_20260729/ca_rhs_scorer.py \
#       --stage rhs2-combine --out-root <retrieved_out_root> \
#       --rhs2-task-glob 'task*/ca_rhs_rhs2_output.json'
# (no --export/--output overrides needed — WORKSPACE is resolved at runtime via modules.sh's
# ws_find, and this script redirects its own stdout/stderr into the out-root once WORKSPACE is
# known, since #SBATCH pragmas cannot reference it before submission.)

#SBATCH --job-name=p3-2d-rhs2
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-31

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-31)}"
BASE_SEED=980001
# Verifier F8 fix (2026-08-26): stride task seeds past the per-task chunk range
# (chunk i uses SEED+i inside the task) -- adjacent-TID tasks previously shared
# seeds (64 chunks / 19 distinct seeds, double-counted draws, invalid SE).
# Max seed = 980001 + 100*31 + 3 = 983104 < 990001 (F10c reference disjointness).
SEED=$((BASE_SEED + 100 * TID))
N_DRAWS=800
F10C_REFERENCE_N=2000

: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) — run cluster/setup.sh first}"
OUT_ROOT="$WORKSPACE/p3_2d_rhs2_20260826/task${TID}"
LOG_DIR="$WORKSPACE/p3_2d_rhs2_20260826/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

# Redirect stdout/stderr into the log dir now that WORKSPACE is known (the static
# #SBATCH --output/--error pragmas cannot reference $WORKSPACE).
LOG_BASE="$LOG_DIR/task${TID}_seed${SEED}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== p3-2d-rhs2 task=$TID seed=$SEED out_root=$OUT_ROOT host=$(hostname) commit=$(git rev-parse HEAD) ==="

# Provenance stamp (cluster/SKILL.md gotcha #12): bare `python <driver>.py`
# invocation bypasses main.py's run_metadata writer. Stamped into LOG_DIR, not
# OUT_ROOT — `--stage rhs2` REFUSES a non-empty out-root (PA-CA-11 idempotency
# check above), so OUT_ROOT must stay empty until the driver itself writes to it.
TASK_SEED="$SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$LOG_DIR" "p3_2d_rhs2 task$TID"

DRIVER="results/campaign51_20260728/realistic_20260729/ca_rhs_scorer.py"

python "$DRIVER" --stage rhs2 --wbh-center eff \
    --n-draws "$N_DRAWS" --seed "$SEED" \
    --f10c-reference-n "$F10C_REFERENCE_N" \
    --out-root "$OUT_ROOT" \
    --out "$OUT_ROOT/ca_rhs_rhs2_output.json"

echo "=== done: task=$TID seed=$SEED ==="
