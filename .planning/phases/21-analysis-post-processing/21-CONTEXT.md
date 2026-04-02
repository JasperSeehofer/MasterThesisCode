# Phase 21: Analysis & Post-Processing - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Diagnose the zero-likelihood problem in the evaluate pipeline, compare all four posterior combination methods quantitatively, and build a robust post-processing combination script that works in log-space. This phase produces the diagnostic analysis and tooling; the physics-motivated likelihood floor itself is Phase 22.

</domain>

<decisions>
## Implementation Decisions

### Script Interface
- **D-01:** The combination script is a new `--combine` CLI subcommand on `__main__.py`, following the same pattern as `--evaluate` and `--snr_analysis`
- **D-02:** The combination logic lives in a new module under `bayesian_inference/` (importable) and is wired to the CLI via `__main__.py`

### Diagnostic Output
- **D-03:** The zero-likelihood diagnostic report is a generated markdown file (`diagnostic_report.md`) written to the working directory
- **D-04:** The report covers: which events produce zeros at which h-bins, root causes (no hosts, catalog gaps, redshift mismatch), and summary statistics

### Zero-Handling Default
- **D-05:** Default strategy is Option 3 (physics floor) when available, with automatic fallback to Option 1 (exclude zeros) when the physics floor is not yet implemented
- **D-06:** All four strategies are selectable via CLI flag: `naive`, `exclude` (Option 1), `per-event-floor` (Option 2), `physics-floor` (Option 3)
- **D-07:** Option 3 depends on Phase 22 (`single_host_likelihood` floor) — until then, selecting it explicitly warns and falls back to Option 1

### Output Artifacts
- **D-08:** The combination script outputs JSON only (`combined_posterior.json` with joint H0 posterior array + metadata)
- **D-09:** Plotting is handled separately by existing/future plotting infrastructure — the combination script is a pure data processing step

### Claude's Discretion
- Internal module structure and function decomposition
- JSON schema for combined_posterior.json (must include h-values, posterior array, method used, event count, metadata)
- Comparison table format for ANAL-02 (markdown file in working directory)
- Log-shift-exp implementation details (standard numerical technique)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Evaluate Pipeline
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — Pipeline B: `evaluate()` writes per-event JSONs, `single_host_likelihood()` is the core integration, `check_overflow()` is the underflow detection target
- `master_thesis_code/__main__.py` — CLI entry point where `--combine` will be added alongside `--evaluate`
- `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py` — Pipeline A: has `np.prod(likelihoods, axis=0)` combination (the naive approach to replace)

### Existing Campaign Data
- `results/h_sweep_20260401/posteriors/` — Per-h-value JSON files with detection index → likelihood list structure (15 h-values from 0.6 to 0.86)
- `results/h_sweep_20260401/posteriors_with_bh_mass/` — Same structure, with BH mass variant (if exists)

### Requirements
- `.planning/REQUIREMENTS.md` — ANAL-01, ANAL-02, POST-01, NFIX-01 definitions

### Prior Analysis
- `.planning/STATE.md` — Accumulated decisions section has known MAP values and option assessments from prior conversation

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `bayesian_statistics.py:BayesianStatistics.evaluate()` — writes per-event JSON format that the combination script must read
- `plotting/bayesian_plots.py:plot_combined_posterior()` — existing plot factory that can consume the JSON output
- `physical_relations.py:dist()`, `dist_to_redshift()` — needed if diagnostic analysis traces zero events back to redshift/distance mismatch

### Established Patterns
- CLI subcommands via `argparse` in `__main__.py` with mutually exclusive groups
- Per-h-value JSON structure: `{"0": [val], "1": [val], ..., "h": 0.73}` — detection index as string key, likelihood as single-element list, empty list `[]` for missing events, `0.0` for zero-likelihood events
- Working directory convention: all output written to the `<working_dir>` passed as first CLI argument

### Integration Points
- `__main__.py` — new `--combine` flag with working_dir positional argument
- `bayesian_inference/` — new module for combination logic
- JSON schema must be backward-compatible with existing per-event JSON files as input

</code_context>

<specifics>
## Specific Ideas

- The fallback chain (Option 3 → Option 1) should be automatic and logged, not silent
- The comparison table (ANAL-02) should cover both "with BH mass" and "without BH mass" variants
- Known baseline MAP values for sanity checking: naive (0.72/0.86), Option 1 (0.68/0.66)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 21-analysis-post-processing*
*Context gathered: 2026-04-02*
