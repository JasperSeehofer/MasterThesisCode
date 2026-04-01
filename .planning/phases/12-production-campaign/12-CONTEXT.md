# Phase 12: Production Campaign - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Run a production-scale EMRI simulation campaign on bwUniCluster with corrected physics (5-point stencil derivatives + galactic confusion noise), producing a statistically sufficient catalog of unfiltered Cramer-Rao bounds for downstream H0 inference. The campaign uses `submit_pipeline.sh` (full simulate→merge→evaluate chain), NOT the injection-only shortcut from Phase 11.1.

This phase covers: campaign submission, monitoring, failure handling, data transfer, and yield verification. It does NOT cover d_L threshold tuning or H0 posterior evaluation — those belong to Phase 13.

</domain>

<decisions>
## Implementation Decisions

### Campaign Scale
- **D-01:** 100 tasks x 50 steps = 5,000 total EMRI events. Target yield: ~1,000+ detections (SNR >= 20).
- **D-02:** Seed 200 — fresh seed to avoid overlap with Phase 11 (seed 100) and Phase 11.1 injections (seed 12345).
- **D-03:** Use `submit_pipeline.sh --tasks 100 --steps 50 --seed 200` for submission.

### d_L Error Threshold
- **D-04:** Do NOT filter by d_L error during simulation. Save ALL Cramer-Rao bounds regardless of relative error magnitude.
- **D-05:** Rationale: threshold filtering moves entirely to evaluation time (Phase 13). This allows tuning the threshold for different evaluation runs without re-running the campaign. The threshold has no strong physical motivation beyond "best detections define the posterior shape."
- **D-06:** If `FRACTIONAL_LUMINOSITY_ERROR` filtering is currently applied during simulation, remove or bypass it for this campaign.

### Failure Handling & Monitoring
- **D-07:** Yield-driven, not task-driven. Don't aim for 100% task completion — monitor detection count and stop when ~1,000+ detections are accumulated.
- **D-08:** Use `resubmit_failed.sh` as needed, but don't obsess over completing every task. If yield target is met, accept partial task completion.

### Data Transfer & Persistence
- **D-09:** rsync results to a local directory outside the repo for now. Do not commit large data files to git.
- **D-10:** Future consideration: Git LFS or similar for integrating production data into the repo. Deferred — not in scope for this phase.

### Claude's Discretion
- Exact rsync flags and target directory structure
- How to monitor detection yield (sacct + grep, or a small counting script)
- Whether to skip the evaluate step in submit_pipeline.sh (since Phase 13 handles evaluation)
- Merge script invocation details
- Whether to split the 100 tasks into smaller batches to avoid overwhelming the scheduler

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cluster Infrastructure
- `cluster/submit_pipeline.sh` — Pipeline orchestrator; submits simulate→merge→evaluate chain
- `cluster/simulate.sbatch` — GPU array job script; resource allocation and timeout settings
- `cluster/merge.sbatch` — Merges per-task CSVs into single output
- `cluster/evaluate.sbatch` — Evaluation job (may be skipped — Phase 13 handles evaluation)
- `cluster/resubmit_failed.sh` — Resubmit failed array tasks
- `cluster/README.md` — Full cluster quickstart, monitoring, and troubleshooting guide

### Prior Phase Artifacts
- `.planning/phases/11-validation-campaign/11-CONTEXT.md` — Validation campaign decisions; d_L threshold recommendation
- `.planning/phases/11.1-simulation-based-detection-probability/11.1-CONTEXT.md` — Injection campaign decisions; P_det grid design

### Code
- `master_thesis_code/constants.py` — `FRACTIONAL_LUMINOSITY_ERROR` constant (may need bypass)
- `master_thesis_code/main.py` — `data_simulation()` loop; CRB timeout setting

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `submit_pipeline.sh` — fully functional pipeline orchestrator, accepts `--tasks`, `--steps`, `--seed`
- `resubmit_failed.sh` — handles failed task resubmission
- `cluster/extract_validation_results.py` — may be adaptable for production result extraction

### Established Patterns
- Phase 11 campaign: `submit_pipeline.sh --tasks 3 --steps 10 --seed 100` — same pattern at larger scale
- Phase 11.1 injection campaign: `submit_injection.sh --tasks_per_h 20 --steps 500 --seed 12345` — separate infrastructure, not used here

### Integration Points
- Merged CRB CSV is the input for Phase 13 (H0 posterior sweep)
- `run_metadata.json` per task records git commit, seed, SLURM job IDs

</code_context>

<specifics>
## Specific Ideas

- The d_L threshold "only the best detections define the shape of the posterior" — this philosophy should carry forward to Phase 13 threshold tuning
- Git LFS noted as a future approach for integrating production data into the repo

</specifics>

<deferred>
## Deferred Ideas

- Git LFS integration for production data — evaluate after campaign completes
- Larger campaign (10,000+ events) if 5,000 proves insufficient — can re-run with higher task/step counts

</deferred>

---

*Phase: 12-production-campaign*
*Context gathered: 2026-04-01*
