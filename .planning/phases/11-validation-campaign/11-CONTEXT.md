# Phase 11: Validation Campaign - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Run a validation campaign (3 tasks, 10 steps, seed 100) on bwUniCluster with the corrected physics (galactic confusion noise in PSD + 5-point stencil Fisher derivatives), compare results against the v1.1 smoke test baseline (same parameters: 3 tasks, 10 steps, seed 100), recalibrate the d_L fractional error threshold based on observed data, and tune timeouts/epsilons if needed.

</domain>

<decisions>
## Implementation Decisions

### Campaign Parameters
- **D-01:** Validation run configuration: `--tasks 3 --steps 10 --seed 100`. Matches the latest v1.1 smoke test parameters for comparability.
- **D-02:** Seed 100 (different from original v1.1 plan seed 42) — uses the same seed as the most recent v1.1 run, so parameter draws are identical and differences are attributable to physics corrections.

### Comparison Methodology
- **D-03:** Compare all four metric categories against v1.1 baseline:
  1. Detection rate (fraction passing SNR threshold)
  2. SNR distribution (per-event values)
  3. CRB magnitudes and Fisher matrix condition numbers
  4. Wall time per event
- **D-04:** Pass/fail criteria are directional checks, not quantitative thresholds:
  - Confusion noise should produce lower SNRs
  - 5-point stencil should produce different (not necessarily lower) CRBs
  - Detection rate may drop but must remain >0
  - No crashes, NaN values, or negative CRB diagonals (the latter already caught by Phase 10 safety checks)
- **D-05:** Comparison reported as a markdown report file saved to the run directory. Not just terminal output.

### d_L Threshold Recalibration
- **D-06:** Data-driven: examine the distribution of delta_d_L / d_L from the validation run. Report the distribution, percentiles, and a recommended threshold value.
- **D-07:** Document only — do NOT update `FRACTIONAL_LUMINOSITY_ERROR` in `constants.py` or the filtering code in this phase. The recommendation goes into the comparison report for Phase 12 to act on.

### Timeout & Epsilon Tuning
- **D-08:** If >10% of events hit the 90s CRB timeout, increase the timeout value in `main.py` within this phase. Don't defer to Phase 12.
- **D-09:** If a significant fraction of events fail due to `ParameterOutOfBoundsError`, tune per-parameter epsilon values in `parameter_space.py` within this phase. Don't defer to Phase 12.
- **D-10:** Report the timeout hit rate, OOB error rate, and any changes made in the comparison report.

### Claude's Discretion
- How to structure the comparison report (table format, sections, level of detail)
- Exact rsync flags and which files to copy back from cluster
- Whether to generate any comparison plots alongside the report
- Polling interval for sacct job monitoring

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cluster Infrastructure
- `cluster/submit_pipeline.sh` — Pipeline orchestrator; used to submit the validation run
- `cluster/simulate.sbatch` — GPU array job script; timeout and resource allocation
- `cluster/merge.sbatch` — Merge per-task CSVs
- `cluster/resubmit_failed.sh` — Resubmit failed tasks if needed
- `cluster/README.md` — Full cluster quickstart and monitoring guide

### Physics Changes Being Validated
- `master_thesis_code/LISA_configuration.py` — Contains confusion noise implementation (Phase 9)
- `master_thesis_code/parameter_estimation/parameter_estimation.py` — Contains 5-point stencil (Phase 10), condition number logging, CRB safety checks
- `master_thesis_code/main.py:197-198,255` — CRB timeout handler (currently 90s from Phase 10)
- `master_thesis_code/datamodels/parameter_space.py:36` — `derivative_epsilon: float = 1e-6` default

### Threshold & Filtering
- `master_thesis_code/constants.py:52` — `FRACTIONAL_LUMINOSITY_ERROR = 0.1` (current 10% threshold)
- `scripts/prepare_detections.py` — Applies truncated-normal sampling to detection parameters
- `scripts/remove_detections_out_of_bounds.py` — Filters detections by M and d_L bounds

### v1.1 Baseline
- `.planning/milestones/v1.1-phases/08-simulation-campaign/08-CONTEXT.md` — v1.1 smoke test decisions and results

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cluster/submit_pipeline.sh --tasks 3 --steps 10 --seed 100` — same invocation as v1.1 baseline
- SSH integration from Phase 7 — `ssh bwunicluster '<cmd>'` for direct cluster command execution
- `sacct` monitoring pattern from Phase 8 — check job status via SSH
- rsync pattern from Phase 8 — copy results back to local machine

### Established Patterns
- v1.1 Phase 8 workflow: submit → monitor → validate → rsync. Same flow applies here.
- `run_metadata.json` records git commit, seed, SLURM metadata per task
- Condition number logging (Phase 10) will appear in SLURM stdout logs

### Integration Points
- The validation run uses the same pipeline code as production — no special validation mode needed
- Results land in `$RUN_DIR/` on cluster workspace, then rsync to local `evaluation/` directory
- Comparison report compares CSVs from v1.1 and v1.2 runs

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 11-validation-campaign*
*Context gathered: 2026-03-29*
