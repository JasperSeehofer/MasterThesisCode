# Phase 33: P_det Grid Resolution - Context

**Gathered:** 2026-04-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Increase P_det grid resolution (both d_L and mass bins), make bin counts configurable via CLI flags, and validate that integration bounds fall within the grid for the vast majority of events. This is a software/numerics phase — no physics formulas change.

</domain>

<decisions>
## Implementation Decisions

### Configurability
- **D-01:** Add `--pdet_dl_bins` CLI flag (default 60, was 30) and `--pdet_mass_bins` CLI flag (default 40, was 20)
- **D-02:** Thread both flags through `arguments.py` → `bayesian_statistics.py` → `SimulationDetectionProbability` constructor
- **D-03:** `SimulationDetectionProbability.__init__()` accepts `dl_bins` and `mass_bins` parameters, replacing module-level `_DL_BINS` and `_M_BINS` constants
- **D-04:** Module-level `_DL_BINS` and `_M_BINS` constants become default values only

### Grid resolution
- **D-05:** Default d_L bins: 60 (up from 30)
- **D-06:** Default mass bins: 40 (up from 20)
- **D-07:** Grid spacing unchanged: linear for d_L, geometric for mass

### Coverage validation
- **D-08:** After grid construction, compute fraction of events whose 4-sigma d_L bounds fall within the grid
- **D-09:** Log WARNING if coverage < 95%, but do not abort the run
- **D-10:** Report coverage fraction in evaluation output (INFO level)

### Claude's Discretion
- Exact implementation of 4-sigma bound check (how to compute per-event sigma from CRB)
- Whether to add coverage stats to the evaluation report JSON
- Test structure and assertion thresholds

</decisions>

<specifics>
## Specific Ideas

- Before/after comparison uses the Phase 30 baseline infrastructure (`--save_baseline` / `--compare_baseline`)
- The new baseline (MAP=0.735, 417 events, 38-point h-grid) was captured with 30 d_L bins — serves as the "before"
- Must work with `cluster/evaluate.sbatch` — the 38-point hybrid h-grid submits 38 SLURM tasks, each calling `--evaluate --h_value X`

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### P_det grid implementation
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py` — `_DL_BINS`, `_M_BINS` (line 41-46), `_build_grid_2d()` (line 302-459), `_build_grid_1d()` (line 461-498)
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — `SimulationDetectionProbability` instantiation (line 306-309), P_det lookup in evaluation

### CLI and threading
- `master_thesis_code/arguments.py` — CLI argument definitions, `Arguments` class
- `master_thesis_code/main.py` — `evaluate()` function, how args flow to `BayesianStatistics`

### Tests
- `master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py` — 10 existing test classes, none hardcode bin count

### Baseline comparison
- `master_thesis_code/bayesian_inference/evaluation_report.py` — `extract_baseline()`, `generate_comparison_report()`
- `.planning/debug/baseline.json` — current baseline (30 d_L bins, MAP=0.735, 417 events)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `RegularGridInterpolator` already used with `method="linear"`, `bounds_error=False`, `fill_value=None` — same API regardless of bin count
- `_get_or_build_grid()` LRU cache handles per-h grid building — cache size `_MAX_CACHE_SIZE = 20` may need increase with finer grids

### Established Patterns
- Module-level constants with underscore prefix (`_DL_BINS`, `_M_BINS`) — will become constructor defaults
- `SimulationDetectionProbability` is instantiated in `bayesian_statistics.py:306-309` — threading point for new params
- Pickle safety for multiprocessing already tested — constructor params must be pickle-safe (ints are fine)

### Integration Points
- `arguments.py` → `main.py:evaluate()` → `BayesianStatistics.__init__()` → `SimulationDetectionProbability.__init__()`
- `cluster/evaluate.sbatch` may need `--pdet_dl_bins 60 --pdet_mass_bins 40` flags added to the python command

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 33-pdet-grid-resolution*
*Context gathered: 2026-04-09*
