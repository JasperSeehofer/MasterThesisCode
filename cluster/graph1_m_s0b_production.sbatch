#!/usr/bin/env bash
# cluster/graph1_m_s0b_production.sbatch — Research Graph 1, Branch D, node
# m-s0b-production: the registered S0-B decisive production read (LAUNCH 2,
# row #325 grant).
#
# **Authorization chain, quoted.** Ledger row #290 decisions-table row 6
# ("m-s0b-production behind g-byte-id and g-score-null green"); row #319
# (g-byte-id precondition discharged); rows #323-#324 (the +-0.033 b_re nodes
# and score_b_re built from the registered text, 2041 tests, no regressions);
# row #325 (chair-decided-under-grant item 10 option (A): cap 105 CPU-h,
# flagged for author veto). Resolves `exec/m-s0b-production/LAUNCH_RECORD.md`'s
# row #320 STOP (both blockers -- the b-node expressibility gap and the
# cost-anchor discrepancy -- are now resolved per DRIVER_BNODE_BUILD_RECORD.md
# and row #325's cap ruling).
#
# ## Registered run shape (PREREGISTRATION_HIER_HTHETA_20260826.md,
# PA-HIER-31(d), quoted verbatim):
#
#   truth        (0, 1)              = C0
#   b_plus_re    (+0.033, 1)
#   b_minus_re   (-0.033, 1)
#   s_plus       (0, sqrt(2))
#   s_minus      (0, 1/sqrt(2))
#
# 5 theta-nodes, h = 0.730 only, venue iiib (production, CoR-P), single
# production catalogue -- NO seed multiplicity (PA-HIER-31(d): "not a mirror
# realization swept over seeds"). Every no-BH read runs CoR-P-faithful:
# theta_sites=2.2, smear_global_selection=False (PA-HIER-31(b), re-ratified
# authoritative by row #255).
#
# ## CLI flags, sourced from PA-HIER-31(g) verbatim
# (`headreadout_20260827/iiib/run_metadata_21.json:cli_args` +
# `b-pahier33-scorer/RECORD.md`'s "What was built" section 2):
# --config iiib --theta-sites 2.2 --smear off. The T1.3 mirror-venue
# instruments (theta_phi_divisor, theta_zwindow, z_window_k, sky_cone_k) are
# NOT named anywhere in PA-HIER-31(g)'s explicit CLI list for the production
# venue -- row #287's "divisor on, zwin on zk4, sky 1.5" certification is a
# MIRROR-venue (b0i) T1.3 config, a different registration than S0-B's own.
# Left at their driver defaults (off/off/1.0/1.5) here, matching PA-HIER-31(g)
# exactly rather than importing an unregistered instrument choice into the
# decisive production read.
#
# --seeds 900101 (DEFAULT_BC_SEEDS[0], the same single-seed convention the
# driver's own S0-C branch already uses): the iiib venue LOADS the real
# production catalogue and does not draw a seeded realization
# (`build_iiib_venue`'s seed parameter is threaded through unused except as
# evaluate()'s base_seed) -- passing --seeds explicitly avoids the driver's
# default S0-A/S0-R behaviour of sweeping all 4 DEFAULT_BC_SEEDS, which would
# needlessly 4x the cost for a single-catalogue venue.
#
# --b-half-width 0.033 registers the two additional _re nodes
# (DRIVER_BNODE_BUILD_RECORD.md); --nodes selects one node per SLURM array
# task (GATE TABLE-FRESH, PA-HIER-31(j): "one BayesianStatistics construction
# per node -- the four separate sbatch tasks structurally guarantee this").
#
# ## Cost check (row #325's own anchor, PA-HIER-31 sec7.2/(i))
# 5 nodes x 14.93-22.9 CPU-h/theta-node = 74.7-101.4 CPU-h <= 105 CPU-h cap
# (row #325 option (A)). PASS -- not re-derived here, taken as the
# chair-ratified anchor under the row #325 grant; author veto remains open
# per that row's own flag.
#
# GATE SEQ (hier_s0_driver.py section 3.7 docstring): satisfied per the same
# chain the m-s0b-byteid precheck already established (BIAS_HISTORY_LEDGER.md
# row #214, ratified rows #215/#216/#220).
#
# Required env (set by the submitter):
#   PROJECT_ROOT -- optional; falls back to $HOME/darksiren-emri

#SBATCH --job-name=graph1-m-s0b-production
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --array=0-4

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

EXPECTED_COMMIT="9336364c"
ACTUAL_COMMIT="$(git rev-parse --short=8 HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "STOP: HEAD mismatch (expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT) -- re-sync before running" >&2
    exit 1
fi

NODES=(truth b_plus_re b_minus_re s_plus s_minus)
NODE="${NODES[$SLURM_ARRAY_TASK_ID]}"

OUT_ROOT="$PROJECT_ROOT/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-s0b-production/s0b_run_20260902"
mkdir -p "$OUT_ROOT/logs"

LOG_BASE="$OUT_ROOT/logs/m_s0b_production_task_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-m-s0b-production job=${SLURM_JOB_ID:-NA} array_task=${SLURM_ARRAY_TASK_ID} node=$NODE host=$(hostname) commit=$ACTUAL_COMMIT ==="

source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$OUT_ROOT" "graph1 m-s0b-production LAUNCH 2 (row #325 grant): registered S0-B node $NODE, iiib/CoR-P, theta_sites=2.2, smear=off, b-half-width=0.033, seed=900101, h=0.73"

python "$PROJECT_ROOT/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py" \
    --arm S0-A \
    --seeds 900101 \
    --nodes "$NODE" \
    --jobs 1 \
    --out-root "$OUT_ROOT" \
    --theta-sites 2.2 \
    --smear off \
    --config iiib \
    --b-half-width 0.033

echo "=== done: m-s0b-production node=$NODE ==="
