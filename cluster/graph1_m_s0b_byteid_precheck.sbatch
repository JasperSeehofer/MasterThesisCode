#!/usr/bin/env bash
# cluster/graph1_m_s0b_byteid_precheck.sbatch — Research Graph 1, Branch D, node
# m-s0b-byteid: g-byte-id MINIMAL precondition check for m-s0b-production.
#
# **Launched under ledger row #290 decisions row 6 ("m-s0b-production behind
# g-byte-id and g-score-null green") + row #295's explicit deferral quote: "the
# N>=1e5 byte-identity check against the banked production reference ... carried
# as the residual precondition for m-s0b-production's launch."**
#
# Scope note: this is NOT the full-scale iiib production byte-id run costed at
# 14.93-22.9 CPU-h/theta-node in b-pahier33-scorer's own RECORD.md (that check
# stays deferred to m-s0b-production's own preflight, per that record's explicit
# disposition). This is the MINIMAL sizing substitute: re-run ONE already-banked
# runner-11 b-node cell (b0i default path, the non-S0-B path the build must not
# have changed) at current HEAD and byte-compare against the banked reference
# (results/campaign51_20260728/realistic_20260729/tree2_20260830/
# hier_s0_zwin_bnodes_run/s0a_seed900103/node_b_minus_sites2.2_nosmear_divisor_
# zwin_zk4/). The comparison read itself happens in a later step (not this job).
#
# Cell chosen: seed=900103, node=b_minus -- the cheapest of the 8 banked runner-11
# b-node cells by measured wall time (1134.37s / 105 events; see
# tree2_20260830/hier_s0_zwin_bnodes_run/logs/runner11_bnodes.log). One cell's
# posteriors_with_bh_mass/h_0_73.json alone carries ~1.86e6 scalar leaf values
# (independently counted), so N>=1e5 is satisfied by a wide margin from this one
# cell without needing the full 1588-event production set.
#
# GATE SEQ (hier_s0_driver.py section 3.7 docstring): "no sbatch for any [HIER]
# stage until [P3-MKER] stage-1 is banked with a ledger row." Checked before this
# script was written: P3-MKER's stage-1 (R1/R2 measure/refute) IS banked with a
# ledger row -- BIAS_HISTORY_LEDGER.md row #214 ("R-MKER-1..4 ... R2 NO verdict +
# exhibit retirement"), ratified rows #215/#216/#220, the SAME row #214 that reads
# "[HIER] items 1 (venue b0i RATIFIED, S0-A unblocked)". rd-runner11's own S0-A/
# b0i run (2026-08-31, this job's comparand) already exercised this exact path
# after that unblock. GATE SEQ's precondition is therefore satisfied; this script
# submits a SINGLE non-array task running the pre-existing, byte-unmodified-by-
# the-build S0-A/b0i code path (the driver's own multiprocessing --jobs 1 branch,
# no internal sbatch/array logic added here or in the driver).
#
# Config, verbatim from the banked cell's s0a_score_output.json (rd-runner11
# RECORD.md): arm=S0-A, config=b0i, theta_sites=2.2, smear=off,
# theta_phi_divisor=on, sky_cone_k=1.5, theta_zwindow=on, z_window_k=4.0,
# catalogue_leg_1d_mass_aware=off, h_values=[0.73] (default, unset --h-nodes/
# --score-h).
#
# Two [PHYSICS] commits landed since the build (a26959b4 h-decoupling,
# 2b657255 Option A'); both argued inert for this cell in LAUNCH_RECORD.md
# (h-admissibility guard only trips above 0.86/1.00, cell uses h=0.73 only;
# Option A' is scoped to correspondence_1d.py's 2D-only call site, this cell's
# b0i/"catalogue_selected" 1D path passes apply_survival=True, its unchanged
# default).
#
# Required env (set by the submitter):
#   PROJECT_ROOT -- optional; falls back to $HOME/darksiren-emri

#SBATCH --job-name=graph1-m-s0b-byteid
#SBATCH --partition=cpu_il
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"
cd "$PROJECT_ROOT"

EXPECTED_COMMIT="c83e391d8994da46033abdfe02529b7572b892a1"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "STOP: HEAD mismatch (expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT) -- re-sync before running" >&2
    exit 1
fi

OUT_ROOT="$PROJECT_ROOT/results/campaign51_20260728/realistic_20260729/graph1_20260901/exec/m-s0b-byteid/byteid_cell_run"
mkdir -p "$OUT_ROOT/logs"

LOG_BASE="$OUT_ROOT/logs/m_s0b_byteid_task_${SLURM_JOB_ID:-nojob}"
exec > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

echo "=== graph1-m-s0b-byteid job=${SLURM_JOB_ID:-NA} host=$(hostname) commit=$ACTUAL_COMMIT ==="

source "$PROJECT_ROOT/cluster/write_provenance.sh"
write_provenance "$OUT_ROOT" "graph1 m-s0b-byteid: minimal g-byte-id precondition sizing check -- re-run of banked runner-11 cell seed=900103 node=b_minus (b0i, S0-A) at current HEAD for byte-comparison against the banked reference; row #290 decisions row 6 + row #295 deferral"

python "$PROJECT_ROOT/results/campaign51_20260728/realistic_20260729/fanout1_20260829/hier_s0_driver.py" \
    --arm S0-A \
    --seeds 900103 \
    --nodes b_minus \
    --jobs 1 \
    --out-root "$OUT_ROOT" \
    --theta-sites 2.2 \
    --smear off \
    --config b0i \
    --theta-phi-divisor on \
    --sky-cone-k 1.5 \
    --theta-zwindow on \
    --z-window-k 4.0 \
    --catalogue-leg-1d-mass-aware off

echo "=== done: m-s0b-byteid precheck cell (seed=900103, node=b_minus) ==="
