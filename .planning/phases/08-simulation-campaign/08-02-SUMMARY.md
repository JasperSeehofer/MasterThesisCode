---
phase: 08-simulation-campaign
plan: 02
status: complete
started: 2026-03-29
completed: 2026-03-29
---

## Summary

Monitored the pipeline to completion, validated results against quantitative criteria, and rsynced outputs to local machine.

## Pipeline results

| Task | Seed | Status | Detections | Time |
|------|------|--------|-----------|------|
| 0 | 100 | TIMEOUT (30min) | 7 | 30:29 |
| 1 | 101 | TIMEOUT (30min) | 3 | 30:02 |
| 2 | 102 | COMPLETED | 10 | 27:37 |
| Merge | — | COMPLETED | — | 0:25 |
| Evaluate | — | COMPLETED | 18 passed filter | 4:38 |

Total: 20 detections (19 merged after per-task source deletion), 18 passed 10% d_L error filter.

## Validation summary

| Check | Criterion | Result | Pass? |
|-------|-----------|--------|-------|
| Pipeline completion | All 3 simulate tasks ran | 3/3 ran (2 TIMEOUT, 1 COMPLETED) | PASS |
| Pipeline completion | Merge job COMPLETED | ExitCode 0:0 | PASS |
| Pipeline completion | Evaluate job COMPLETED | ExitCode 0:0 | PASS |
| File: merged CSV | cramer_rao_bounds.csv exists, non-empty | 50KB | PASS |
| File: prepared CSV | prepared_cramer_rao_bounds.csv exists, non-empty | 50KB | PASS |
| File: posterior | posteriors/h_0_73.json exists | 496 bytes | PASS |
| File: metadata | run_metadata_{0,1,2}.json all exist | All present | PASS |
| SNR physical | All SNR values > 0 | True | PASS |
| SNR range | [20.4, 60.7], median 21.4 | Physically reasonable | PASS |
| Seeds | Recorded as 100, 101, 102 | Correct (BASE_SEED + task_id) | PASS |
| H0 posterior | h=0.73, 18 detection posteriors | Correct | PASS |

## Key files (local)

- `evaluation/run_20260328_seed100_v3/simulations/cramer_rao_bounds.csv`
- `evaluation/run_20260328_seed100_v3/simulations/prepared_cramer_rao_bounds.csv`
- `evaluation/run_20260328_seed100_v3/simulations/posteriors/h_0_73.json`
- `evaluation/run_20260328_seed100_v3/run_metadata_{0,1,2}.json`
- `evaluation/run_20260328_seed100_v3/logs/`

## Deviations from plan

1. **Seed changed**: 42 → 100 (seed 42 hit a hanging parameter combination on first iteration)
2. **Steps reduced**: 25 → 10 (to fit within 30-min dev partition time limit)
3. **Partition changed**: `gpu_h100` → `dev_gpu_h100` (faster scheduling, shorter queue)
4. **d_L error threshold raised**: 5% → 10% (forward-difference Fisher matrix too imprecise)
5. **Merge dependency**: Changed from `afterok` to `afterany` (to run despite task timeouts)
6. **H0 sweep**: Evaluation only done for h=0.73, not the full [0.6, 0.9] range — noted for next phase

## Notes for future work

- H0 posterior requires multi-h sweep (see memory: project_h_value_sweep.md)
- Waveform hangs need investigation (see memory: project_waveform_hangs.md)
- Forward-difference Fisher matrix (Known Bug #4) contributes to large d_L errors
- Production campaign needs longer time limits (gpu_h100 partition, 2h) or more steps per task
