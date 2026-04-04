# Phase 23: Deploy & Validate - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Deploy the phases 21+22 changes (log-space accumulation, physics-floor combination strategy, overflow fix removal) to bwUniCluster and validate that the new physics-floor strategy produces a physically reasonable H0 posterior against existing baselines. Validation uses the existing local campaign data — no new simulation or evaluate runs needed.

</domain>

<decisions>
## Implementation Decisions

### Cluster Status Check
- **D-01:** Check cluster job status first (`squeue` via SSH) before deploying — status determines whether deploy is safe or evaluate has already started.
- **D-02:** If evaluate is already running when we check: let it finish, then deploy the fix and run a fresh evaluation. No scancel needed — accept the already-running job used old code.

### Deployment Method
- **D-03:** Deploy via `git push` locally + `git pull` on cluster via SSH (`ssh bwunicluster 'cd ~/MasterThesisCode && git pull'`). Uses the existing Phase 7 SSH integration.

### Validation Data & Location
- **D-04:** Validate **locally only** using existing campaign data in `results/h_sweep_20260401/` — no cluster validation run needed.
- **D-05:** No need to re-run `--evaluate` on the cluster. The combination script (`--combine`) operates on the per-event JSONs already available locally.
- **D-06:** Three-way strategy comparison: `physics-floor` vs `naive` vs `exclude` (satisfies DEPL-02 and Phase 22 D-06 acceptance criterion of ±0.05 from exclude MAP).

### Validation Output
- **D-07:** Write `results/v1.4-validation.md` — a markdown comparison table (strategy × with/without BH mass × MAP) plus the ±0.05 acceptance criterion check and a pass/fail verdict.
- **D-08:** Commit `results/v1.4-validation.md` to git as the permanent thesis record of the numerical fix.

### Claude's Discretion
- Exact git commands for the deployment step (push to main, pull on cluster)
- Whether to verify the cluster pull succeeded (check git hash or import test)
- The exact format of the comparison table rows in VALIDATION.md

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Combination Pipeline (Phase 21/22 output)
- `master_thesis_code/bayesian_inference/posterior_combination.py` — `--combine --strategy` entry point; three strategies: `naive`, `exclude`, `physics-floor`
- `master_thesis_code/__main__.py` — CLI wiring for `--combine` subcommand

### Campaign Data
- `results/h_sweep_20260401/posteriors/` — Per-h-value JSONs, "without BH mass" variant
- `results/h_sweep_20260401/posteriors_with_bh_mass/` — Per-h-value JSONs, "with BH mass" variant (111 zero-events, 21%)

### Requirements
- `.planning/REQUIREMENTS.md` — DEPL-01 (deploy before evaluate), DEPL-02 (validate against baselines)

### Prior Phase Context
- `.planning/phases/22-likelihood-floor-overflow-fix/22-CONTEXT.md` — D-06/D-07 define acceptance criterion (MAP within ±0.05 of exclude for both variants)
- `.planning/phases/21-analysis-post-processing/21-CONTEXT.md` — D-05/D-06 define strategy enum and CLI flags

### Cluster Access
- `cluster/README.md` — deployment and job management quickstart
- SSH alias: `ssh bwunicluster` (ControlMaster session reuse established in Phase 7)

### Known Baselines
- Naive: MAP=0.72 (with BH mass), MAP=0.86 (without BH mass)
- Exclude (Option 1): MAP=0.68 (with BH mass), MAP=0.66 (without BH mass)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cluster/submit_pipeline.sh` — existing orchestrator; `ssh bwunicluster '<cmd>'` pattern from Phase 7
- `master_thesis_code/bayesian_inference/combine_posteriors.py` — CLI entry with `--combine --strategy` (Phase 21 output)

### Established Patterns
- Cluster deploy: `ssh bwunicluster 'cd ~/MasterThesisCode && git pull'`
- Results written to `results/` directory in working dir; thesis records committed to git
- Validation comparison table format used in Phase 8 (smoke test campaign results) — similar structure applies here

### Integration Points
- `results/h_sweep_20260401/` — input data for all three combine runs
- `results/v1.4-validation.md` — output artifact; committed alongside the fix

</code_context>

<specifics>
## Specific Ideas

- The validation plan mirrors Phase 22 D-06/D-07: run all three strategies, extract MAP per variant, check ±0.05 criterion, emit PASS/FAIL per variant
- Deployment order matters: check cluster status → deploy → validate locally → commit results
- If evaluate is already done when we check: skip the "before evaluate" urgency, but validation and commit still needed for the thesis record

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 23-deploy-validate*
*Context gathered: 2026-04-02*
