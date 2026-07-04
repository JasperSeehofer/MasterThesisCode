#!/usr/bin/env bash
# Sequential PV-test evaluation queue (issue #16, handoff 7b) — 2 arms x 9 h-values.
BASE=/home/jasper/Repositories/MasterThesisCode/results/pv_correction_test_20260703
for arm in live nopv; do
  for h in 0.725 0.730 0.735 0.740 0.745 0.750 0.755 0.760 0.765 0.770 0.775 0.780 0.785 0.790 0.795 0.800 0.805; do
    label=$(python3 -c "import numpy as np; print(str(np.round($h,4)).replace('.','_'))")
    [ -f $BASE/run_$arm/simulations/posteriors/h_${label}.json ] && { echo "skip $arm h=$h (exists)"; continue; }
    echo "=== $arm h=$h start $(date +%H:%M:%S) ==="
    cd $BASE/run_$arm/cwd && uv run python -m master_thesis_code $BASE/run_$arm --evaluate --h_value $h --seed 600999 --allow_low_pdet_coverage --log_level WARNING > $BASE/run_$arm/eval_h${label}.log 2>&1 || echo "FAILED $arm h=$h"
  done
done
echo "ALL RUNS DONE $(date +%H:%M:%S)"
