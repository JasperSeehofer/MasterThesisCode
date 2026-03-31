# Phase 15: Quick Validation Results

## Pre-Fix Baseline (run_v12_validation, 22 detections)

Data source: `evaluation/run_v12_validation/simulations/posteriors/` and `posteriors_with_bh_mass/`
Git commit: `b2019bb` (pre-fix, contains spurious `/(1+z)`)

### Without BH Mass Channel

| h     | sum(posteriors) | n_detections |
|-------|-----------------|--------------|
| 0.600 | 4996.2900       | 22           |
| 0.626 | 5019.8282       | 22           |
| 0.652 | 5038.7891       | 22           |
| 0.678 | 5052.7217       | 22           |
| 0.704 | 5061.7825       | 22           |
| 0.730 | 5065.1721       | 22           |
| 0.756 | 5063.5014       | 22           |
| 0.782 | 5056.1839       | 22           |
| 0.808 | 5043.0052       | 22           |
| 0.834 | 5023.7381       | 22           |
| 0.860 | 4998.9928       | 22           |

**Peak: h ~ 0.730** (sum-based), monotonically increasing 0.600-0.730, then decreasing.

### With BH Mass Channel (PRE-FIX, contains spurious /(1+z))

| h     | sum(posteriors) | n_detections |
|-------|-----------------|--------------|
| 0.600 | 0.023029        | 22           |
| 0.626 | 0.022836        | 22           |
| 0.652 | 0.022565        | 22           |
| 0.678 | 0.022246        | 22           |
| 0.704 | 0.021870        | 22           |
| 0.730 | 0.021453        | 22           |
| 0.756 | 0.020960        | 22           |
| 0.782 | 0.020455        | 22           |
| 0.808 | 0.019886        | 22           |
| 0.834 | 0.019252        | 22           |
| 0.860 | 0.018599        | 22           |

**Peak: h <= 0.600** (monotonically decreasing across entire range). Consistent with STATE.md claim that pre-fix "with BH mass" peak is at h=0.600 (or below).

## Post-Fix Results

**Status: PENDING -- requires cluster execution**

The evaluation pipeline cannot run locally because `SimulationDetectionProbability` requires injection campaign data (`simulations/injections/injection_h_*.csv`) which only exists on the cluster.

### How to Run

```bash
# On the cluster:
cd $PROJECT_ROOT
git pull  # ensure /(1+z) fix from Plan 15-01 is present
bash cluster/quick_validation.sh <RUN_DIR>
python3 cluster/extract_validation_results.py
```

Where `<RUN_DIR>` is a workspace directory containing `simulations/prepared_cramer_rao_bounds.csv` and `simulations/injections/`.

### Post-Fix With BH Mass Channel

| h     | sum(posteriors) | n_detections |
|-------|-----------------|--------------|
| 0.652 | _pending_       | _pending_    |
| 0.678 | _pending_       | _pending_    |
| 0.704 | _pending_       | _pending_    |
| 0.730 | _pending_       | _pending_    |

### Post-Fix Without BH Mass Channel (expected UNCHANGED)

| h     | sum(posteriors) | n_detections |
|-------|-----------------|--------------|
| 0.652 | _pending_       | _pending_    |
| 0.678 | _pending_       | _pending_    |
| 0.704 | _pending_       | _pending_    |
| 0.730 | _pending_       | _pending_    |

## Acceptance Criteria

1. **Direction test (test-direction):** Post-fix "with BH mass" posterior at h=0.678 should be LARGER than at h=0.652, indicating the peak has shifted from below 0.600 toward 0.678.
2. **Unchanged test (test-without-unchanged):** "Without BH mass" values should match pre-fix baseline within MC noise (~1%).
3. **No overshoot:** Neither channel should peak at exactly h=0.73 (P_det=1 bias expected).
4. **Minimum shift:** The "with BH mass" peak should have shifted by at least 0.02 in h from the pre-fix baseline.
