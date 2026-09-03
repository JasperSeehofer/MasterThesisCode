#!/usr/bin/env bash
# cluster/graph1_gateacc_relaunch.sbatch — GATE-ACC reporting-only relaunch
# (v-falsifier-ii-classG ADJUDICATING_READ_RECORD.md §6/§8: the 2026-09-02
# login-node `--stage gates` run died — libpython3.13 not found because
# cluster/modules.sh was never sourced (interactive/bare invocation, not an
# sbatch job — cluster/SKILL.md gotcha 1). This resubmits the SAME two
# invocations as a proper batch job so modules are loaded first.)
#
# Single-node CPU job (NOT an array — `--stage gates` takes the full 33-seed
# set in one call and processes seeds internally, unlike `--stage fleet`
# which is one-seed-per-task). Chains arm=bt then arm=bc against the same
# out-root, per the launch task's "chain bt then bc" instruction and per
# ADJUDICATING_READ_RECORD.md §6 which reports bt first, bc second.
#
# Resources modeled on cluster/p3_2d_fleet.sbatch (closest working comparand
# for this driver): cpu_il partition, 16 cpus-per-task (BayesianStatistics
# .evaluate()'s internal multiprocessing.Pool auto-scales to
# os.sched_getaffinity(0) - 2, so this reservation keeps that pool busy
# without idling a big chunk of a 128-core cpu_il node per gotcha 7).
# Time: the 2026-09-02 attempt was still inside its initial
# `build_b0i_2d_selection_objects` phase at >2h16m CPU for ONE arm on a
# login node (no batch memory-bandwidth contention data available yet for
# this stage) — sized generously per gotcha 9 (JSON-at-end instrument, no
# walltime-extension recovery path) to 08:00:00 for both arms chained.
#
# Out-root: continues the existing p3_2d_fleet_aprime_20260902 out-root
# (registered in the read record) — `stage_fleet`'s per-(arm,seed) meta
# idempotency does not apply to `stage_gates` output filenames
# (gates_bt.json / gates_bc.json), so this is a fresh compute either way,
# not a resume.
#
# Submit (from ~/darksiren-emri on the cluster; consult /cluster + preflight
# first, require VERDICT: READY):
#   sbatch cluster/graph1_gateacc_relaunch.sbatch

#SBATCH --job-name=graph1-gateacc-relaunch
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --ntasks=1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) — run cluster/setup.sh first}"

OUT_ROOT="${OUT_ROOT:-$WORKSPACE/p3_2d_fleet_aprime_20260902}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

exec > "$LOG_DIR/gateacc_relaunch.out" 2> "$LOG_DIR/gateacc_relaunch.err"

echo "=== graph1-gateacc-relaunch out_root=$OUT_ROOT host=$(hostname) commit=$(git rev-parse HEAD) ==="

TASK_SEED=900101
source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$OUT_ROOT" "graph1_gateacc_relaunch --stage gates bt+bc"

DRIVER="results/campaign51_20260728/realistic_20260729/p3_2d_fleet.py"
SEEDS="900101,900102,900103,900104,900105,900106,900107,900108,900109,900110,900111,900112,900113,900114,900115,900116,900117,900118,900119,900120,900121,900122,900123,900124,900125,900126,900127,900128,900129,900130,900131,900132,900133"

for ARM in bt bc; do
    echo "--- stage gates arm=$ARM ---"
    python "$DRIVER" --stage gates --arm "$ARM" --seeds "$SEEDS" --out-root "$OUT_ROOT"
done

echo "=== done ==="
