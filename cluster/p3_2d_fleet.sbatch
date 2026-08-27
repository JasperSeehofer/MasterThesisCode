#!/usr/bin/env bash
# cluster/p3_2d_fleet.sbatch — [P3-2D] b0i2d fleet array job
# (PREREGISTRATION_P3_2D_20260825.md §2/§7 + PA-2D-1 F14 + PA-2D-4/5/6).
#
# 24-task array (SLURM_ARRAY_TASK_ID 0-23); per-task seed = 900101 + TASK_ID
# (PA-2D-1 F14's 24-seed, 900101-900124 power decision — the repo-wide
# BASE_SEED + SLURM_ARRAY_TASK_ID convention, cluster/SKILL.md gotcha 4).
# Each task runs BOTH arms sequentially (B2-C "bc" then B2-T "bt") for its
# one seed, via results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py
# --stage fleet (commit a75515e7 or later — the PA-2D-6 six-stamp flag fix;
# an earlier commit silently stamps the STALE catalogue_numerator_survival
# pin, PA-2D-6).
#
# Driver internals (checked before sizing, not assumed): p3_2d_fleet.py itself
# runs single-process/sequential (no subprocess or process-pool fan-out across
# seeds — c1d.run_mirror_seed_inprocess is called in a plain for-loop). BUT
# BayesianStatistics.evaluate() (bayesian_statistics.py:4387-4392), called
# once per arm, spins its OWN multiprocessing.Pool sized to
# len(os.sched_getaffinity(0)) - 2 by default — i.e. it auto-scales to
# whatever --cpus-per-task reserves. The per-event likelihood pool has real
# work up to n_events=200 (correspondence_1d.py:643, the registered
# n_events=200 default) per seed, so this is NOT the "1-4 core" single-thread
# case hypothesized in the launch task; it is genuine per-event parallelism.
# --cpus-per-task=16 below matches the closest measured production anchor for
# this same evaluate() pool (cluster/LAUNCHING_JOBS.md §2: "evaluate: 56-76
# min per h-value @ 3355 events / 16 cpus") rather than an unmeasured guess —
# our per-seed workload (200 events, single h) is far smaller, so 16 cpus
# should comfortably keep the pool busy without idling a big chunk of a
# 128-core cpu_il node per task (gotcha 7).
#
# Time: prereg anchor is ~64 s/arm (single-h) uncontended; 2 arms/task ~=
# 128 s, plus host-pool/catalogue build overhead per task and gotcha-6
# cross-task memory-bandwidth contention when packed >2 tasks/node on
# cpu_il. Sized generously (>= the 1 h floor the launch task requires) since
# gotcha 9 (JSON-at-end instruments lose everything on a walltime kill, no
# `scontrol` extension for regular users) makes under-sizing expensive to
# recover from.
#
# Idempotency: p3_2d_fleet.py's own stage_fleet()/​_run_b0i2d_arm_seed()
# already skip-and-reuse per (arm, seed) when `<out_root>/<arm>_<seed>_meta
# .json` exists (PA-CA-11) — a resubmit of this array is a safe no-op replay
# for any task whose meta already landed.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight
# first, require VERDICT: READY):
#   sbatch cluster/p3_2d_fleet.sbatch
# (no --export/--output overrides needed — WORKSPACE is resolved at runtime
# via modules.sh's ws_find, and this script redirects its own stdout/stderr
# into the out-root once WORKSPACE is known, since #SBATCH pragmas cannot
# reference it before submission.)

#SBATCH --job-name=p3-2d-fleet
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-23

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-23)}"
BASE_SEED=900101
SEED=$((BASE_SEED + TID))

: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) — run cluster/setup.sh first}"
# Out-root is overridable via `sbatch --export=ALL,OUT_ROOT=...` (PA-2DR, 2026-08-27).
# The default is the ORIGINAL banked 2026-08-25 fleet. A re-run MUST pass its own
# OUT_ROOT: PA-CA-11's idempotency guard skips any (arm, seed) whose meta already
# exists, so re-submitting against the banked default is a silent no-op that also
# risks contaminating the frozen comparand.
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/p3_2d_fleet_20260825}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

# Redirect stdout/stderr into the out-root now that WORKSPACE is known (the
# static #SBATCH --output/--error pragmas cannot reference $WORKSPACE).
LOG_BASE="$LOG_DIR/task${TID}_seed${SEED}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== p3-2d-fleet task=$TID seed=$SEED out_root=$OUT_ROOT host=$(hostname) commit=$(git rev-parse HEAD) ==="

# Provenance stamp (cluster/SKILL.md gotcha #12): this driver is invoked as a
# bare `python <driver>.py`, not `python -m darksiren_emri`, so main.py's
# run_metadata writer never fires. write_provenance() is the one-line adopter.
TASK_SEED="$SEED"
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$OUT_ROOT" "p3_2d_fleet arm loop"

DRIVER="results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py"

for ARM in bc bt; do
    echo "--- arm=$ARM seed=$SEED ---"
    python "$DRIVER" --stage fleet --arm "$ARM" --seeds "$SEED" --out-root "$OUT_ROOT"
done

echo "=== done: task=$TID seed=$SEED ==="
