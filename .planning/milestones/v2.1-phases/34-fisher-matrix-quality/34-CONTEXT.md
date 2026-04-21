# Phase 34: Fisher Matrix Quality - Context

**Gathered:** 2026-04-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace `pinv()` / `allow_singular=True` with explicit degeneracy detection and exclusion so near-singular covariance matrices are caught, logged, and excluded rather than silently producing unreliable likelihood evaluations. Includes diagnostic tooling (CSV export, debug plot) to investigate *why* singularity occurs — physically, Fisher matrices should always be positive definite.

</domain>

<decisions>
## Implementation Decisions

### Detection Strategy
- **D-01:** Check condition number of both 3D and 4D covariance matrices at evaluation time, in `bayesian_statistics.py` where `pinv()` is currently called (~lines 427, 433)
- **D-02:** Use `np.linalg.cond()` as the detection metric — standard, easy to threshold, already used upstream in `parameter_estimation.py:391`
- **D-03:** Check 3D and 4D independently. An event is flagged if either matrix exceeds the threshold

### Handling Policy
- **D-04:** Exclude flagged events from the posterior likelihood product entirely. Do not regularize or downweight — degenerate events should not contribute unreliable likelihoods
- **D-05:** The primary goal is to understand *why* singularity occurs (it shouldn't physically). Exclusion is the safe default while investigating the root cause
- **D-06:** Generate a two-panel diagnostic plot for all flagged events every evaluation run:
  - Panel 1: Eigenvalue spectrum (bar chart per flagged event, shows which direction is degenerate)
  - Panel 2: Parameter scatter of flagged events in (d_L, SNR, M) space (shows correlation with physical parameters)
- **D-07:** Debug plot is always generated (not opt-in). Cheap to produce since it only covers flagged events. Ensures regressions are never missed

### Threshold Calibration
- **D-08:** Determine threshold empirically from the current data — run evaluation, collect all condition numbers, identify the gap between well-conditioned and degenerate events
- **D-09:** Make threshold configurable via `--fisher_cond_threshold` CLI flag with the empirically determined default. Follows Phase 33 pattern (`--pdet_dl_bins`)
- **D-10:** Start with same threshold for both 3D and 4D matrices. If empirical data shows they need different thresholds, deviate to separate values (but start unified)

### Diagnostic Output
- **D-11:** Per-run log summary: total events, flagged count, excluded count, top-5 worst condition numbers. INFO level
- **D-12:** Write `fisher_quality.csv` alongside posteriors with columns: detection_index, cond_3d, cond_4d, excluded (bool). Used as input for the debug plot and post-hoc analysis
- **D-13:** Add a "Fisher Quality" section to the Phase 30 comparison report showing: events excluded before vs after, condition number distribution shift, impact on MAP h

### Claude's Discretion
- Exact eigenvalue visualization style (grouped bars, stacked, or per-event subplots)
- Whether to use `apply_style()` theming on the debug plot or keep it plain diagnostic style
- Module placement for Fisher quality utilities (inline in bayesian_statistics.py vs separate module)
- How to compute empirical threshold (gap detection, percentile, or manual inspection)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Covariance construction and pinv() usage
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` lines 370-435 — covariance matrix construction from CRB, `pinv()` calls at 427/433, `slogdet()` at 428/434
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` line 1347-1348 — testing path still uses `multivariate_normal(..., allow_singular=True)`

### Fisher matrix condition number (upstream)
- `master_thesis_code/parameter_estimation/parameter_estimation.py` lines 388-395 — `np.linalg.cond()` already computed and logged at Fisher inversion time

### CLI and threading pattern
- `master_thesis_code/arguments.py` — CLI argument definitions, `Arguments` class. Follow Phase 33 pattern for `--fisher_cond_threshold`
- `master_thesis_code/main.py` — `evaluate()` function, how args flow to `BayesianStatistics`

### Phase 30 comparison infrastructure
- `master_thesis_code/bayesian_inference/evaluation_report.py` — `extract_baseline()`, `generate_comparison_report()` — add Fisher Quality section here
- `.planning/debug/baseline.json` — current baseline (to compare against)

### Plotting
- `master_thesis_code/plotting/_style.py` — `apply_style()` for consistent theming
- `master_thesis_code/plotting/_helpers.py` — `save_figure()`, `get_figure()` utilities

### Existing research on the issue
- `.gpd/research-map/CONCERNS.md` lines 96-102 — documents the `allow_singular=True` concern
- `.gpd/research-map/CONVENTIONS.md` lines 158-163 — Fisher matrix approximation and singularity risk

### Requirements
- `.planning/REQUIREMENTS.md` — FISH-01 (degenerate handling), FISH-02 (condition number flagging)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `np.linalg.cond()` already used in `parameter_estimation.py:391` — same API for covariance check
- `np.linalg.slogdet()` already called at lines 428/434 — sign check is free (already computed)
- `_mvn_pdf()` custom implementation at line 188 — takes pre-computed `cov_inv` and `log_norm`, easy to skip for excluded events
- `apply_style()` and `save_figure()` from plotting module — for debug plot
- `evaluation_report.py` comparison framework — extend with Fisher quality section

### Established Patterns
- Pre-computed arrays (`_cov_inv_3d`, `_log_norm_3d`, etc.) allocated per-detection slot — exclusion means leaving slots empty or using a mask
- Global arrays passed to multiprocessing workers via `_initialize_worker()` — mask/exclusion list must be pickle-safe
- CLI flags defined as properties on `Arguments` dataclass, threaded through `main.py` → `BayesianStatistics`
- CSV output pattern: pandas DataFrame → `.to_csv()` in output directory

### Integration Points
- `bayesian_statistics.py` `__init__()` loop (~line 360-445) — where condition check and exclusion happen
- `_initialize_worker()` (~line 1486) — excluded events must not reach workers
- `evaluation_report.py` — where Fisher quality section is added to comparison
- Output directory (same as `posteriors/`) — where `fisher_quality.csv` and debug plot are written

</code_context>

<specifics>
## Specific Ideas

- Physically, Fisher matrices should always be positive definite (they are sums of outer products of signal derivatives weighted by 1/PSD). Singularity suggests a bug upstream — possibly zero derivatives, parameter degeneracies in the waveform model, or numerical noise in the five-point stencil
- The two-panel debug plot should help identify the root cause by showing whether degenerate events cluster in specific regions of parameter space (low SNR? extreme mass ratios? particular sky positions?)
- The empirical threshold determination is a prerequisite sub-task: must run evaluation once to collect condition numbers before the threshold can be set

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 34-fisher-matrix-quality*
*Context gathered: 2026-04-09*
