# Phase 31: Catalog-Only Diagnostic - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Confirm that the completion term L_comp is the dominant source of the h=0.66 bias by running the evaluation pipeline with f_i=1.0 (catalog-only, no completion term) and comparing the resulting posterior against the Phase 30 baseline. This phase adds one CLI flag, always-on diagnostic CSV output, and a diagnostic summary — no physics formula changes.

</domain>

<decisions>
## Implementation Decisions

### Catalog-Only Bypass
- **D-01:** Add `--catalog_only` boolean CLI flag to `arguments.py`. When set with `--evaluate`, the worker skips the completion term integral entirely and forces `f_i=1.0` in the combination formula at `bayesian_statistics.py:775`.
- **D-02:** The completion term integral (lines 703-754) is NOT computed in catalog-only mode — both for correctness (f_i=1.0 means L_comp has zero weight) and to save compute time on cluster.

### Per-Event Diagnostic Logging
- **D-03:** Diagnostic CSV is always written during `--evaluate`, not gated by a separate flag. Overhead is negligible (~16k rows, ~100KB, no extra computation — values are already computed).
- **D-04:** CSV format: one row per (event, h_value) pair. Columns: `event_idx, h, f_i, L_cat_no_bh, L_cat_with_bh, L_comp, combined_no_bh, combined_with_bh`. Written to `{working_dir}/diagnostics/event_likelihoods.csv`.
- **D-05:** In `--catalog_only` mode, L_comp column is 0.0 and f_i is 1.0 for all rows (since the integral is skipped).

### Result Comparison
- **D-06:** Primary comparison uses Phase 30 infrastructure: `--evaluate --catalog_only --compare_baseline .planning/debug/baseline.json` produces the standard before/after report (MAP h, 68% CI, bias %).
- **D-07:** Additionally, generate a diagnostic summary from the per-event CSV that reports: mean f_i across events (in normal mode), L_comp contribution statistics, and fraction of events where L_comp pulls the combined likelihood toward lower h. This summary helps explain WHY the bias changes, not just that it changed.

### Claude's Discretion
- Exact placement of diagnostic CSV writing logic (in worker vs collected after all workers finish)
- Diagnostic summary format (section in comparison report vs separate file)
- How to aggregate "L_comp pulls toward lower h" metric (e.g., compare L_comp(h=0.66) vs L_comp(h=0.73) per event)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evaluation Pipeline
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` ~668-778 — Completion term computation + Gray et al. combination formula (the code being bypassed)
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` ~347 — Completeness function setup (`get_completeness_at_redshift`)
- `master_thesis_code/arguments.py` — CLI flag definitions, existing `--save_baseline`, `--compare_baseline` patterns
- `master_thesis_code/main.py` ~72-76 — Where baseline/comparison hooks into the entry point

### Phase 30 Infrastructure (reuse)
- `master_thesis_code/bayesian_inference/evaluation_report.py` — Comparison report generation, baseline extraction
- `.planning/debug/baseline.json` — Committed baseline snapshot (MAP h=0.73 from test data; production baseline from run_v12_validation)

### Bias Investigation Context
- `.planning/debug/h0-inference-worsening.md` — Root cause analysis documenting L_comp as dominant bias source
- `.gpd/debug/h0-posterior-bias-worsening.md` — GPD debug artifacts
- `.planning/REQUIREMENTS.md` — DIAG-01, DIAG-02 requirement definitions

### Phase 30 Context (prior decisions)
- `.planning/phases/30-baseline-evaluation-infrastructure/30-CONTEXT.md` — Baseline/comparison design decisions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `evaluation_report.py`: `extract_baseline_metrics()`, `generate_comparison_report()` — Phase 30 comparison infrastructure, reuse directly for headline numbers
- `arguments.py` pattern: boolean flags as `store_true`, properties on `Arguments` dataclass
- Existing `_LOGGER.debug()` at line 765 already logs f_i, L_cat, L_comp per event — diagnostic CSV formalizes this

### Established Patterns
- CLI flags defined in `_parse_arguments()`, exposed as `@property` on `Arguments`
- Worker processes receive `BayesianStatistics` instance with `self.h` set per h-value
- `p_Di()` method (line ~600) is the per-event likelihood entry point; completion term is computed inline
- Multiprocessing via `forkserver` start method with preloaded modules

### Integration Points
- `arguments.py` — add `--catalog_only` flag
- `bayesian_statistics.py:p_Di()` — conditional skip of completion integral when catalog_only=True
- `bayesian_statistics.py:evaluate()` or worker setup — thread `catalog_only` flag to workers
- `main.py` — wire `catalog_only` from args to evaluation call
- Diagnostic CSV collection — workers compute per-event values, need to aggregate across processes

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

*Phase: 31-catalog-only-diagnostic*
*Context gathered: 2026-04-08*
