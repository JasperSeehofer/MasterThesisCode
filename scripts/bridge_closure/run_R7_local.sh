#!/usr/bin/env bash
# R7 anchor: run the REAL --evaluate locally at several h to confirm the railing
# reproduces locally (local == cluster) and to capture the per-event diagnostic CSV.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || true
for h in 0.70 0.73 0.80 0.86; do
  echo "=== R7 eval h=$h ==="
  PYTHONUNBUFFERED=1 python -m darksiren_emri . --evaluate --h_value "$h" --num_workers 14 \
    2>&1 | tail -3 || echo "h=$h FAILED"
done
echo "=== R7 done; posteriors: ==="
ls -1 /tmp/seed600_local/simulations/posteriors/ | tail
