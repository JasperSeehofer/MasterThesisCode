#!/usr/bin/env bash
# cluster/quick_validation.sh -- Quick validation of the /(1+z) fix.
#
# Runs the evaluation pipeline at 4 h-values to confirm the "with BH mass"
# posterior peak shifts from h=0.600 toward h=0.678 after removing the
# spurious /(1+z) Jacobian factor.
#
# Usage (on login node or compute node):
#   cd $PROJECT_ROOT
#   bash cluster/quick_validation.sh <RUN_DIR>
#
# Where <RUN_DIR> is a workspace directory containing:
#   simulations/prepared_cramer_rao_bounds.csv
#   simulations/injections/injection_h_*.csv
#
# Example:
#   bash cluster/quick_validation.sh /path/to/workspace/emri/run_20260328_seed100_v3
#
# Output: simulations/posteriors/h_*.json and simulations/posteriors_with_bh_mass/h_*.json
# for h = {0.652, 0.678, 0.704, 0.730}.
#
# After running, extract results with:
#   python3 cluster/extract_validation_results.py

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Usage: bash cluster/quick_validation.sh <RUN_DIR>"
    echo ""
    echo "  RUN_DIR  Path to workspace directory containing simulation data"
    echo "           Must contain: simulations/prepared_cramer_rao_bounds.csv"
    echo "                         simulations/injections/injection_h_*.csv"
    exit 1
fi

RUN_DIR="$1"

# Validate required data
if [[ ! -f "$RUN_DIR/simulations/prepared_cramer_rao_bounds.csv" ]]; then
    echo "ERROR: $RUN_DIR/simulations/prepared_cramer_rao_bounds.csv not found" >&2
    exit 1
fi

if ! ls "$RUN_DIR/simulations/injections"/injection_h_*.csv &>/dev/null; then
    echo "ERROR: No injection CSV files found in $RUN_DIR/simulations/injections/" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$PROJECT_ROOT/.venv/bin/activate"

# Symlink simulations/ so relative paths resolve
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

# ---------------------------------------------------------------------------
# Run evaluation at 4 h-values
# ---------------------------------------------------------------------------

H_VALUES=(0.652 0.678 0.704 0.730)

echo ""
echo "=== Quick Validation: /(1+z) Fix ==="
echo "  Run Dir:    $RUN_DIR"
echo "  H values:   ${H_VALUES[*]}"
echo "  Git commit: $(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo ""

for H_VALUE in "${H_VALUES[@]}"; do
    echo "--- Running h=$H_VALUE ---"
    python -m master_thesis_code "$RUN_DIR" \
        --evaluate \
        --h_value "$H_VALUE" \
        --log_level INFO
    echo "--- Done h=$H_VALUE ---"
    echo ""
done

echo "=== All evaluations complete ==="
echo ""
echo "Results saved to:"
echo "  simulations/posteriors/           (without BH mass)"
echo "  simulations/posteriors_with_bh_mass/  (with BH mass)"
echo ""
echo "Run 'python3 cluster/extract_validation_results.py' to extract and compare."
