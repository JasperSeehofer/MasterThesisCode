# Phase 11: Validation Campaign - Research

**Researched:** 2026-03-29
**Domain:** HPC validation, physics regression testing, CSV data analysis
**Confidence:** HIGH

## Summary

This phase runs a small validation campaign on bwUniCluster to confirm that the two physics corrections (galactic confusion noise in PSD from Phase 9, and 5-point stencil Fisher derivatives from Phase 10) produce valid results. The campaign uses identical parameters to the v1.1 smoke test (3 tasks, 10 steps, seed 100) so results are directly comparable.

The v1.1 baseline data is already available locally at `evaluation/run_20260328_seed100_v3/`. It contains 19 detections from ~2598 total events (0.73% detection rate), with median SNR 21.4, mean generation time 0.148s, and median fractional d_L error of 8.1%. Two of three tasks hit the SLURM wall-time limit but flushed partial results via the SIGTERM handler. The 5-point stencil (4x more waveform evaluations per parameter) will significantly increase per-event CRB computation time, making wall time a key concern.

**Primary recommendation:** The phase is purely operational -- submit jobs, wait, compare CSVs, write a report. No new code needs to be written for the simulation itself. The only code changes are conditional: increasing the CRB timeout (D-08) or tuning epsilon values (D-09) if the data warrants it. A Python comparison script should be written locally to produce the markdown report.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Validation run configuration: `--tasks 3 --steps 10 --seed 100`. Matches the latest v1.1 smoke test parameters for comparability.
- **D-02:** Seed 100 (different from original v1.1 plan seed 42) -- uses the same seed as the most recent v1.1 run, so parameter draws are identical and differences are attributable to physics corrections.
- **D-03:** Compare all four metric categories against v1.1 baseline: (1) Detection rate, (2) SNR distribution, (3) CRB magnitudes and Fisher matrix condition numbers, (4) Wall time per event.
- **D-04:** Pass/fail criteria are directional checks, not quantitative thresholds: confusion noise should produce lower SNRs; 5-point stencil should produce different CRBs; detection rate may drop but must remain >0; no crashes, NaN values, or negative CRB diagonals.
- **D-05:** Comparison reported as a markdown report file saved to the run directory. Not just terminal output.
- **D-06:** Data-driven d_L threshold: examine distribution of delta_d_L / d_L, report percentiles and recommended threshold.
- **D-07:** Document only -- do NOT update FRACTIONAL_LUMINOSITY_ERROR or filtering code in this phase. Recommendation goes into comparison report for Phase 12.
- **D-08:** If >10% of events hit the 90s CRB timeout, increase the timeout value in `main.py` within this phase.
- **D-09:** If significant fraction of events fail due to ParameterOutOfBoundsError, tune epsilon values in `parameter_space.py` within this phase.
- **D-10:** Report timeout hit rate, OOB error rate, and any changes made in the comparison report.

### Claude's Discretion
- How to structure the comparison report (table format, sections, level of detail)
- Exact rsync flags and which files to copy back from cluster
- Whether to generate any comparison plots alongside the report
- Polling interval for sacct job monitoring

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SIM-01 | Validation campaign (3-5 tasks) confirms detection rates and timing with corrected physics | Cluster pipeline infrastructure fully established from v1.1; same `submit_pipeline.sh` invocation. v1.1 baseline data available locally for comparison. |
| SIM-03 | d_L fractional error threshold recalibrated based on 5-point stencil accuracy | v1.1 baseline shows median fractional d_L error of 8.1% (range 5.3%-46.7%). Comparison script computes percentiles from new run to recommend threshold. |
</phase_requirements>

## Standard Stack

No new libraries needed. This phase uses existing infrastructure.

### Core Tools
| Tool | Purpose | Why Standard |
|------|---------|--------------|
| `cluster/submit_pipeline.sh` | Submit 3-task validation campaign | Existing orchestrator from v1.1 |
| `sacct` | Monitor job status via SSH | Standard SLURM tool |
| `rsync` | Copy results from cluster to local | Standard file transfer |
| `pandas` | CSV analysis for comparison report | Already in project deps |
| `numpy` | Statistical computations (percentiles, medians) | Already in project deps |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `cluster/resubmit_failed.sh` | Resubmit failed tasks | Only if tasks fail |
| `matplotlib` | Optional comparison plots | If planner decides to generate plots |

## Architecture Patterns

### Recommended Workflow Structure
```
Phase 11 Execution Flow:
1. Merge Phase 10 code → claudes_sidequests → push to cluster
2. ssh bwunicluster 'cd ~/MasterThesisCode && git pull && uv sync --extra gpu'
3. bash cluster/submit_pipeline.sh --tasks 3 --steps 10 --seed 100
4. Monitor with sacct until completion
5. rsync results to local evaluation/ directory
6. Run comparison script locally to produce markdown report
7. Inspect report → apply D-08/D-09 fixes if needed
```

### Critical Prerequisite: Phase 10 Code Must Be on Cluster

Phase 10 implementation (5-point stencil + 90s timeout) lives on branch `worktree-agent-a7dd0449`, NOT on `claudes_sidequests`. The code must be merged before pushing to the cluster. Key changes:
- `parameter_estimation.py`: 5-point stencil as default, condition number logging, CRB safety checks
- `main.py`: 90s timeout (was 30s), `LinAlgError`/`ParameterEstimationError` exception handling
- 6 new tests in `parameter_estimation_test.py`

Phase 9 (confusion noise) IS on `claudes_sidequests` already.

### v1.1 Baseline Data (Available Locally)

Location: `evaluation/run_20260328_seed100_v3/`

| Metric | v1.1 Value |
|--------|-----------|
| Total events | 2598 |
| Detections | 19 |
| Detection rate | 0.73% |
| Median SNR | 21.4 |
| Mean SNR | 25.5 |
| Max SNR | 60.7 |
| Median generation time | 0.141s |
| Median fractional d_L error | 8.1% |
| Mean fractional d_L error | 9.9% |
| Max fractional d_L error | 46.7% |
| Tasks that TIMEOUT'd | 2 of 3 (but flushed via SIGTERM) |
| Wall time per task | ~27-30 min |

### CSV Column Structure

The CRB CSV has 124 columns: 14 parameter values + 14x14/2 = 105 unique covariance entries + T, dt, SNR, generation_time, host_galaxy_index.

### Comparison Report Structure (Claude's Discretion)

Recommended sections for the markdown report:
1. **Run Metadata** -- git commits, seeds, timestamps for both runs
2. **Detection Rate Comparison** -- total events, detections, rate
3. **SNR Distribution** -- summary statistics table, per-event comparison
4. **CRB Analysis** -- Fisher condition numbers (from SLURM logs), d_L fractional error distribution
5. **d_L Threshold Recommendation** -- percentile table, recommended value
6. **Wall Time Analysis** -- per-task wall time, per-event timing, timeout hit rate
7. **Error Analysis** -- OOB error rate, timeout rate, any crashes
8. **Pass/Fail Summary** -- directional checks per D-04

### Comparison Script Pattern

```python
# scripts/compare_validation_runs.py (or inline in execution)
import pandas as pd
import numpy as np

v11 = pd.read_csv("evaluation/run_20260328_seed100_v3/simulations/cramer_rao_bounds.csv")
v12 = pd.read_csv("evaluation/<new_run_dir>/simulations/cramer_rao_bounds.csv")

# Detection rate
# SNR comparison (per-event since same seed)
# CRB d_L fractional error percentiles
# Fisher condition numbers from SLURM logs
```

### Anti-Patterns to Avoid
- **Running comparison on the cluster:** All analysis should happen locally after rsync. Cluster GPU time is precious.
- **Modifying physics code in this phase:** D-07 explicitly forbids changing FRACTIONAL_LUMINOSITY_ERROR. Only timeout (D-08) and epsilon (D-09) changes are permitted.
- **Skipping the merge step:** Phase 10 code MUST be on the cluster before submission. Forgetting this would run v1.1 code again.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Job submission | Custom sbatch scripts | `cluster/submit_pipeline.sh` | Already handles array jobs, dependency chaining, run directory creation |
| Job monitoring | Custom polling loop | `sacct -j <job_ids>` via SSH | Standard SLURM, already used in v1.1 |
| Failed task recovery | Manual resubmission | `cluster/resubmit_failed.sh` | Handles partial output cleanup |
| CSV merging | Custom merge script | `emri-merge` (installed entry point) | Already handles per-task CSV concatenation |

## Common Pitfalls

### Pitfall 1: Phase 10 Code Not Merged
**What goes wrong:** Validation runs with v1.1 code (forward-diff, 30s timeout) instead of v1.2 (5-point stencil, 90s timeout), making the comparison meaningless.
**Why it happens:** Phase 10 implementation is on `worktree-agent-a7dd0449`, not `claudes_sidequests`.
**How to avoid:** First task must merge the worktree branch, verify with `git log`, then push to cluster.
**Warning signs:** CRB computation taking ~0.15s (v1.1) instead of ~0.6s (expected with 4x waveforms).

### Pitfall 2: 5-Point Stencil Dramatically Increases Wall Time
**What goes wrong:** The 5-point stencil computes 56 waveforms per Fisher matrix (4 per parameter x 14 parameters) instead of 15 (1 per parameter + 1 baseline). This is a ~3.7x increase in CRB computation time.
**Why it happens:** More waveform evaluations per detected event.
**How to avoid:** The 90s CRB timeout from Phase 10 should accommodate this, but monitor closely. The v1.1 CRB times were sub-second (generation_time column shows waveform generation time, not total CRB time), so 3.7x should still fit under 90s. However, some events may now trigger the 90s timeout.
**Warning signs:** High timeout rate in SLURM logs; significantly fewer detections than expected.

### Pitfall 3: SLURM Wall Time Exceeded
**What goes wrong:** The 2-hour SLURM wall limit is hit before all 10 steps complete.
**Why it happens:** v1.1 already took ~28 min for 10 steps. With the stencil overhead, this could reach 45-90 min. Should still fit in 2 hours, but it is possible.
**How to avoid:** The SIGTERM handler flushes partial results, so even timeout produces usable data. But if all tasks timeout, the comparison will be based on partial data.
**Warning signs:** All tasks showing `TIMEOUT` state in sacct.

### Pitfall 4: Confusion Noise Kills All Detections
**What goes wrong:** Adding galactic confusion noise to PSD increases the noise floor at low frequencies, reducing SNR for all events. With a 0.73% detection rate at v1.1, even a modest SNR reduction could drop detection rate to zero.
**Why it happens:** Confusion noise is strongest at 0.1-3 mHz where EMRI signals have significant power.
**How to avoid:** D-04 acknowledges detection rate may drop but must remain >0. If zero detections, the comparison report documents this and the validation is a partial failure.
**Warning signs:** No CRB CSV files produced; only undetected_events CSV.

### Pitfall 5: Stale Cluster Environment
**What goes wrong:** Cluster venv has old package versions or the git checkout is stale.
**Why it happens:** Last cluster sync was for v1.1; confusion noise and stencil changes need fresh pull.
**How to avoid:** Always `git pull && uv sync --extra gpu` on cluster before submitting.

## Code Examples

### Merging Phase 10 Worktree Branch
```bash
# On local machine
git checkout claudes_sidequests
git merge worktree-agent-a7dd0449 --no-ff -m "merge(10): integrate five-point stencil derivatives"
git push origin claudes_sidequests
```

### Submitting Validation Campaign
```bash
# Via SSH to cluster
ssh bwunicluster 'cd ~/MasterThesisCode && git pull && uv sync --extra gpu'
ssh bwunicluster 'cd ~/MasterThesisCode && bash cluster/submit_pipeline.sh --tasks 3 --steps 10 --seed 100'
```

### Monitoring Jobs
```bash
ssh bwunicluster 'sacct -j <SIM_JOB>,<MERGE_JOB>,<EVAL_JOB> --format=JobID,JobName,State,Elapsed,MaxRSS'
```

### Rsync Results Back
```bash
rsync -avz bwunicluster:'$WORKSPACE/run_YYYYMMDD_seed100/'{simulations/,logs/,run_metadata_*.json} \
    evaluation/run_YYYYMMDD_seed100_v12/
```

### Computing Fractional d_L Error
```python
import pandas as pd
import numpy as np

df = pd.read_csv("path/to/cramer_rao_bounds.csv")
dl = df["luminosity_distance"]
crb_dl = df["delta_luminosity_distance_delta_luminosity_distance"]
frac_err = np.sqrt(np.abs(crb_dl)) / dl

print("Percentiles of fractional d_L error:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(frac_err, p):.4f}")
```

### Extracting Condition Numbers from SLURM Logs
```bash
# Phase 10 added INFO-level condition number logging
grep "condition number" evaluation/<run_dir>/logs/simulate_*.out | \
    sed 's/.*kappa = //' | sort -g
```

## State of the Art

| Old Approach (v1.1) | Current Approach (v1.2) | When Changed | Impact |
|---------------------|------------------------|--------------|--------|
| Forward-diff derivative (O(epsilon)) | 5-point stencil (O(epsilon^4)) | Phase 10 | More accurate CRBs, ~3.7x more waveform evaluations |
| No confusion noise in PSD | Galactic confusion noise included | Phase 9 | Higher noise floor at low-f, lower SNRs |
| 30s CRB timeout | 90s CRB timeout | Phase 10 | Accommodates stencil overhead |
| No Fisher condition number logging | Condition number logged before inversion | Phase 10 | Enables monitoring ill-conditioned matrices |
| No CRB safety checks | Negative diagonal and singular matrix rejection | Phase 10 | Prevents invalid CRB propagation |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `uv run pytest -m "not gpu and not slow"` |
| Full suite command | `uv run pytest` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SIM-01 | Validation campaign completes with corrected physics | manual-only | N/A -- requires GPU cluster | N/A |
| SIM-03 | d_L threshold recalibrated from validation data | manual-only | N/A -- analysis of cluster output | N/A |

### Sampling Rate
- **Per task commit:** `uv run pytest -m "not gpu and not slow"` (only if code is changed, e.g., D-08/D-09 fixes)
- **Per wave merge:** N/A
- **Phase gate:** Comparison report produced and pass/fail criteria met

### Wave 0 Gaps
None -- this phase validates existing code by running it on the cluster and analyzing outputs. No new test infrastructure needed. If D-08 or D-09 trigger code changes, those changes are to existing tested code (timeout values, epsilon values) that do not need new test files.

## Open Questions

1. **How much will the 5-point stencil increase per-event CRB time?**
   - What we know: 56 waveforms instead of 15 (~3.7x). v1.1 generation_time averaged 0.148s per waveform.
   - What is unclear: Total CRB time is not logged in v1.1 (only generation_time for the initial waveform). The stencil involves generating 56 LISA responses and computing 105 inner products.
   - Recommendation: Condition number and timing data from the v1.2 run logs will answer this. No pre-investigation needed.

2. **Will confusion noise reduce SNR enough to eliminate all detections?**
   - What we know: v1.1 had only 19 detections out of 2598 events (0.73%). Median SNR was 21.4, just above the threshold of 20. Confusion noise increases PSD at low frequencies.
   - What is unclear: The magnitude of SNR reduction depends on the frequency content of each specific EMRI waveform.
   - Recommendation: This is exactly what the validation campaign answers. If zero detections, report it and let the user decide how to proceed.

## Sources

### Primary (HIGH confidence)
- `evaluation/run_20260328_seed100_v3/` -- v1.1 baseline data, locally available
- `cluster/submit_pipeline.sh`, `cluster/simulate.sbatch` -- cluster infrastructure code
- `.planning/phases/10-five-point-stencil-derivatives/10-VERIFICATION.md` -- Phase 10 verified implementation
- `master_thesis_code/LISA_configuration.py` -- confusion noise implementation (Phase 9)
- `master_thesis_code/constants.py` -- FRACTIONAL_LUMINOSITY_ERROR = 0.1

### Secondary (MEDIUM confidence)
- v1.1 SLURM logs showing task wall times (27-30 min for 10 steps)
- v1.1 detection statistics computed from local CSV

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new tools, reuses v1.1 infrastructure
- Architecture: HIGH -- same submit/monitor/rsync workflow as v1.1
- Pitfalls: HIGH -- based on direct analysis of v1.1 baseline data and Phase 10 implementation

**Research date:** 2026-03-29
**Valid until:** 2026-04-15 (cluster workspace expiration may change)
