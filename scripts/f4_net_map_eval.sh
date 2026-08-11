#!/usr/bin/env bash
# scripts/f4_net_map_eval.sh -- F4-v2 net-MAP verification driver (LOCAL run).
#
# Evaluates the OLD seed200+300 production CRB through BOTH p_det estimators
# (nadaraya_watson and local_linear) over a decisive h-grid, into two separate
# run directories, then prints the net MAP for each via f4_net_map_compare.py.
#
# Prerequisites (see .planning/HANDOFF-F4V2-NET-MAP-EVAL.md):
#   * repo at the F4-v2 commit (git pull); `uv sync --extra cpu --extra dev`
#   * GLADE reduced catalog at
#       darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv
#   * SRC_CRB_DIR below contains simulations/{cramer_rao_bounds.csv,
#       prepared_cramer_rao_bounds.csv}
#   * INJ_DIR below contains injection_h_*.csv
#
# Usage:
#   scripts/f4_net_map_eval.sh <SRC_CRB_DIR> <INJ_DIR> [WORK_ROOT]
# Example:
#   scripts/f4_net_map_eval.sh ~/data/run_production_h0p73_20260506 \
#       ~/data/injections ~/data/f4v2_verify

set -euo pipefail

SRC_CRB_DIR="${1:?need SRC_CRB_DIR (has simulations/{cramer_rao_bounds,prepared_cramer_rao_bounds}.csv)}"
INJ_DIR="${2:?need INJ_DIR (has injection_h_*.csv)}"
WORK_ROOT="${3:-./f4v2_verify}"

H_GRID=(0.72 0.73 0.74 0.75 0.76 0.78)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for f in cramer_rao_bounds.csv prepared_cramer_rao_bounds.csv; do
    [[ -f "$SRC_CRB_DIR/simulations/$f" ]] || { echo "MISSING: $SRC_CRB_DIR/simulations/$f" >&2; exit 1; }
done
compgen -G "$INJ_DIR/injection_h_*.csv" > /dev/null || { echo "MISSING: $INJ_DIR/injection_h_*.csv" >&2; exit 1; }
[[ -f "$PROJECT_ROOT/darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv" ]] \
    || { echo "MISSING: reduced_galaxy_catalogue.csv (GLADE catalog)" >&2; exit 1; }

for EST in nadaraya_watson local_linear; do
    RUN_DIR="$WORK_ROOT/$EST"
    mkdir -p "$RUN_DIR/simulations"
    cp -f "$SRC_CRB_DIR/simulations/cramer_rao_bounds.csv" "$RUN_DIR/simulations/"
    cp -f "$SRC_CRB_DIR/simulations/prepared_cramer_rao_bounds.csv" "$RUN_DIR/simulations/"
    ln -sfn "$(cd "$INJ_DIR" && pwd)" "$RUN_DIR/simulations/injections"
    echo "=== estimator=$EST  RUN_DIR=$RUN_DIR ==="
    for H in "${H_GRID[@]}"; do
        echo "--- $EST h=$H ---"
        ( cd "$PROJECT_ROOT" \
          && ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations" \
          && uv run python -m darksiren_emri "$RUN_DIR" --evaluate --h_value "$H" \
               --pdet_estimator "$EST" --pdet_dl_bins 60 --pdet_mass_bins 40 \
               --log_level INFO )
    done
done

echo ""
echo "=== NET MAP COMPARISON ==="
( cd "$PROJECT_ROOT" && uv run python scripts/f4_net_map_compare.py \
    "$WORK_ROOT/nadaraya_watson" "$WORK_ROOT/local_linear" )
