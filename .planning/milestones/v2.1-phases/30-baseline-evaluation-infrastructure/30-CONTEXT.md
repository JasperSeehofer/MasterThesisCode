# Phase 30: Baseline & Evaluation Infrastructure - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Capture a reproducible baseline posterior snapshot and build the before/after comparison framework that all subsequent phases (31-34) use to measure their effect on the H0 bias. This is infrastructure/tooling work — no physics changes.

</domain>

<decisions>
## Implementation Decisions

### Baseline Capture
- **D-01:** Add a new `--save_baseline` CLI flag to the evaluation pipeline. When used with `--evaluate` and a full h-sweep, it extracts posterior metrics and writes a `baseline.json`.
- **D-02:** `--save_baseline` requires a full h-grid sweep (e.g., 27 values). Single h-value runs are not supported for baseline creation — they don't produce enough data for CI/bias computation.

### Comparison Trigger
- **D-03:** Add a new `--compare_baseline <path>` CLI flag. When combined with `--evaluate`, it runs the evaluation AND generates the comparison report against the referenced baseline JSON.
- **D-04:** `--compare_baseline` also works standalone (without `--evaluate`) to compare two existing result sets without re-running the pipeline. Both modes must be supported.

### Output Format & Location
- **D-05:** Baseline JSON contains core metrics + per-event summary:
  - Core: MAP h, 68% CI [lower, upper], CI width, bias % ((MAP-0.73)/0.73), number of events, h-grid values + log-posteriors
  - Per-event: d_L, SNR, sigma(d_L)/d_L, condition number, quality filter pass/fail
- **D-06:** Baseline JSON and comparison reports are stored in `.planning/debug/` alongside existing bias investigation artifacts. Baseline is committed to git for cross-phase reference.

### Script Reuse
- **D-07:** Refactor comparison logic from `scripts/compare_posterior_bias.py` into the main package (e.g., `bayesian_inference/evaluation_report.py`). The new CLI flags call this module. The old script becomes a thin wrapper or is removed.

### Claude's Discretion
- Exact module placement within the package (evaluation_report.py or similar)
- Comparison report markdown formatting and ASCII chart style
- Whether the old compare_posterior_bias.py is kept as a wrapper or removed entirely
- Implementation of the h-grid sweep detection (how to know a full sweep was run)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evaluation Pipeline
- `master_thesis_code/main.py` §614-627 — `evaluate()` function, entry point for the evaluation pipeline
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` §170-340 — `BayesianStatistics.evaluate()`, posterior computation, per-h JSON output
- `master_thesis_code/arguments.py` — CLI argument definitions, `--evaluate` flag

### Existing Comparison Tools
- `scripts/compare_posterior_bias.py` — Existing two-run comparison with markdown report + ASCII chart (refactor source)
- `scripts/compare_validation_runs.py` — Another comparison script for reference

### Bias Investigation Context
- `.planning/debug/h0-inference-worsening.md` — Debug artifacts documenting the bias investigation
- `.gpd/debug/h0-posterior-bias-worsening.md` — GPD debug artifacts for bias investigation
- `.planning/REQUIREMENTS.md` — DIAG-03, EVAL-01, EVAL-02 requirement definitions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/compare_posterior_bias.py`: `load_posteriors()`, `find_peak()`, `generate_report()` — core comparison logic to refactor
- `bayesian_inference/posterior_combination.py` — log-space posterior combination utilities
- `analysis/validation.py` — additional comparison/validation patterns

### Established Patterns
- CLI flags defined in `arguments.py` as properties on the `Arguments` dataclass
- Evaluation results written as per-h JSON files to `posteriors/h_*.json`
- `run_metadata.json` pattern for recording git commit, timestamp, seed, CLI args

### Integration Points
- `main.py:evaluate()` — where baseline saving and comparison triggering need to hook in
- `arguments.py` — where `--save_baseline` and `--compare_baseline` flags are defined
- `.planning/debug/` — output location for baseline.json and comparison reports

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

*Phase: 30-baseline-evaluation-infrastructure*
*Context gathered: 2026-04-08*
