#!/usr/bin/env bash
# cluster/graph1_s0c_hgrid.sbatch — Research Graph 1, Branch D follow-on, node
# r-s0c-hgrid / m-s0c-hgrid: the registered S0-B theta-grid at the two FRESH h-values
# {0.665, 0.780}. The h=0.730 cells are job 6779532 (S0-B of record) and are NOT re-run
# (PA-HIER-34 sec 3.2: byte-identical config, driver + darksiren_emri/ unchanged since
# 081b1f28, --h-nodes 0.73 == the driver's default (H_GEN,)).
#
# Registration of record: results/campaign51_20260728/realistic_20260729/graph1_20260901/
#   exec/r-s0c-hgrid/REGISTRATION_ADDENDUM_PA-HIER-34.md  (PROPOSED; fresh RULE)
# Authorization chain: row #349 (batch-2 grant, "feel free to also use the cluster"),
#   chair item (3) "S0-C -- S0-B theta-grid at h in {0.665, 0.73, 0.78} behind a PA-HIER
#   addendum"; d-photoz-leverage DOSSIER_20260903.md item 2. Cap 10 CPU-h ORCHESTRATOR-
#   DERIVED (row #332 basis) -- see PA-HIER-34 sec 9 for the two cost bases; the chair must
#   confirm the basis BEFORE submitting (sec 12 item 1).
#
# CLI: VERBATIM from cluster/graph1_m_s0b_production.sbatch (job 6779532) plus exactly ONE
# new flag, --h-nodes <h> (single value per task; hier_s0_driver.py:2926-2933). Everything
# else identical: --arm S0-A --config iiib --theta-sites 2.2 --smear off --seeds 900101
# --b-half-width 0.033 --jobs 1, mirror instruments at driver defaults (divisor off, zwindow
# off, z_window_k 1.0, sky_cone_k 1.5, catalogue_leg_1d_mass_aware off).
#
# Array design: 10 tasks = 2 h-values x 5 theta-nodes, one node per task (GATE TABLE-FRESH,
# PA-HIER-31(j)). TID -> h = H_LIST[TID/5], node = NODES[TID%5]. Per-h out-roots because the
# driver's node-dir suffix carries NO h (a second h at one out-root would overwrite the
# first): $WORKSPACE/graph1_s0c_hgrid_20260904/h_0p665 and .../h_0p780.
#
# Resources: identical to 6779532 (cpu_il, 16 cpus/task -> 14 workers). NOT reduced: the
# worker count changes summation order (row #337 showed the L_cat_no_bh==0 label flips on
# float noise), which would break byte-comparability with the reused h=0.73 cells.
# --time=00:45:00: 6779532 measured 7:21-7:36 per cell; ~5x margin (head-rebaseline convention).
#
# HEAD pin: ancestor check (row #331 pattern) on 081b1f28 (the commit 6779532 ran at) PLUS a
# physics-freeze guard: darksiren_emri/ and the driver must be diff-quiet vs 081b1f28 -- this
# is what makes the h=0.73 reuse valid at RUN time, not only at authoring time.
#
# Dataset pins: build_iiib_venue STOP-gates on c1d.CRB_CSV_MD5 (9a1f2a14384a9281c97ca3be312ddaab)
# and c1d.REDUCED_CATALOGUE_MD5 (c52c13b5cab61f6b3f04bbe202550969) itself (row #295).
#
# Submit (from the cluster, after preflight READY and git pull --ff-only):
#   source cluster/modules.sh && sbatch cluster/graph1_s0c_hgrid.sbatch
# Required env: PROJECT_ROOT optional (falls back to $HOME/darksiren-emri); WORKSPACE from
# modules.sh (ws_find emri).

#SBATCH --job-name=graph1-s0c-hgrid
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --array=0-9

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

: "${WORKSPACE:?WORKSPACE not resolved by modules.sh (ws_find emri) -- run cluster/setup.sh first}"

DRIVER="$PROJECT_ROOT/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py"

EXPECTED_COMMIT="081b1f28"  # the S0-B-of-record commit (job 6779532); ancestor check, row #331 pattern
ACTUAL_COMMIT="$(git rev-parse --short=8 HEAD)"
if ! git merge-base --is-ancestor "$EXPECTED_COMMIT" HEAD; then
    echo "STOP: HEAD ($ACTUAL_COMMIT) does not contain required commit $EXPECTED_COMMIT -- re-sync before running" >&2
    exit 1
fi
# Physics-freeze guard (PA-HIER-34 sec 3.2/3.3): the reused h=0.73 cells are only comparable
# if the estimator and the driver are byte-identical to what 6779532 ran.
if ! git diff --quiet "$EXPECTED_COMMIT" HEAD -- darksiren_emri/ "$DRIVER"; then
    echo "STOP: darksiren_emri/ or hier_s0_driver.py differ from $EXPECTED_COMMIT -- the h=0.73 reuse (PA-HIER-34 sec 3.2) is void; do not run" >&2
    git diff --stat "$EXPECTED_COMMIT" HEAD -- darksiren_emri/ "$DRIVER" >&2
    exit 1
fi

TID="${SLURM_ARRAY_TASK_ID:?array job required (--array=0-9)}"
H_LIST=(0.665 0.780)
NODES=(truth b_plus_re b_minus_re s_plus s_minus)
H_IDX=$(( TID / 5 ))
NODE_IDX=$(( TID % 5 ))
if [[ "$H_IDX" -ge "${#H_LIST[@]}" ]]; then
    echo "STOP: array task id $TID out of range (2 h-values x 5 nodes = 10 tasks)" >&2
    exit 1
fi
H_VALUE="${H_LIST[$H_IDX]}"
NODE="${NODES[$NODE_IDX]}"
H_LABEL="h_$(echo "$H_VALUE" | sed 's/\./p/')"   # 0.665 -> h_0p665, 0.780 -> h_0p780

OUT_ROOT="$WORKSPACE/graph1_s0c_hgrid_20260904/$H_LABEL"
mkdir -p "$OUT_ROOT/logs"

LOG_BASE="$OUT_ROOT/logs/s0c_hgrid_task_${TID}_${H_LABEL}_${NODE}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-s0c-hgrid job=${SLURM_JOB_ID:-NA} array_task=$TID h=$H_VALUE node=$NODE host=$(hostname) commit=$ACTUAL_COMMIT ==="
git rev-parse HEAD > "$OUT_ROOT/GIT_COMMIT_AT_RUN_task${TID}.txt"

source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$OUT_ROOT" "graph1 r-s0c-hgrid (PA-HIER-34, row #349 item 3): registered S0-B node $NODE at h=$H_VALUE, iiib/CoR-P, theta_sites=2.2, smear=off, b-half-width=0.033, seed=900101; config identical to job 6779532 plus --h-nodes $H_VALUE"

python "$DRIVER" \
    --arm S0-A \
    --seeds 900101 \
    --nodes "$NODE" \
    --jobs 1 \
    --out-root "$OUT_ROOT" \
    --theta-sites 2.2 \
    --smear off \
    --config iiib \
    --b-half-width 0.033 \
    --h-nodes "$H_VALUE"

echo "=== done: graph1-s0c-hgrid h=$H_VALUE node=$NODE ==="
