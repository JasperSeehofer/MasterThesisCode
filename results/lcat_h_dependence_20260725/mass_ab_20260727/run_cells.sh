#!/usr/bin/env bash
# Cell-A'/A'' + B/B'' 2D mass-kernel A/B (derivation §3.8 branches a/c, §4 item 4).
# Runs the four cells SEQUENTIALLY from each cell's cwd/ (probe symlink recipe).
# Venue: seed-1000 deep venue (v1_probe_smeared inputs), fused 7-point h-grid.
# Stack: main @ e9bec6d (ratified mass kernel + --host_mass_kernel flag).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
HGRID="0.60,0.65,0.70,0.73,0.76,0.80,0.86"

run_cell() {
  local cell="$1"; shift
  echo "=== [$(date +%H:%M:%S)] starting $cell: $* ==="
  cd "$ROOT/$cell/cwd" || return 1
  uv run python -m master_thesis_code .. --evaluate \
    --h_values "$HGRID" --seed 1000 "$@" \
    >"$ROOT/$cell/probe.log" 2>&1
  echo "=== [$(date +%H:%M:%S)] finished $cell (exit $?) ==="
}

# A' : absolute_marginal baseline (volume_deconv z-kernel via auto; gaussian M)
run_cell cellAprime --normalization_mode absolute_marginal
# A'': absolute_marginal + ratified truncated-lognormal mass kernel (the
#      real-data-relevant discriminator, branch c)
run_cell cellApp --normalization_mode absolute_marginal --host_mass_kernel trunc_lognormal
# B  : generator_marginal legs + broadened z-kernel (attribution instrumentation
#      baseline, re-run at the current stack: #50 widths + regenerated catalogue)
run_cell cellB --normalization_mode generator_marginal --host_z_kernel volume_deconv
# B'': B + the ratified mass kernel (P2/P3 scoring cell)
run_cell cellBpp --normalization_mode generator_marginal --host_z_kernel volume_deconv --host_mass_kernel trunc_lognormal

echo "=== [$(date +%H:%M:%S)] ALL CELLS DONE ==="
