# Phase 10: Five-Point Stencil Derivatives - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Upgrade the Fisher information matrix from O(epsilon) forward-difference derivatives to O(epsilon^4) five-point stencil derivatives. The `five_point_stencil_derivative()` method already exists at `parameter_estimation.py:187` but is never called. This phase wires it into `compute_fisher_information_matrix()`, adjusts the timeout for the ~4x increase in waveform evaluations (56 vs 15), and adds condition number monitoring to detect ill-conditioned matrices.

</domain>

<decisions>
## Implementation Decisions

### Timeout Strategy
- **D-01:** CRB computation timeout is increased from 30s to **90s** (fixed value in `main.py`). This is 3x the current value, accommodating the ~4x waveform increase with margin for not-all-worst-case evaluations.
- **D-02:** The timeout value will be revisited in Phase 11 validation campaign based on observed wall times.

### Condition Number Handling
- **D-03:** The condition number kappa(Gamma) is logged at INFO level for every event, before matrix inversion.
- **D-04:** Events are skipped (with WARNING log) when: (a) `np.linalg.inv` raises `LinAlgError` (singular matrix), or (b) any CRB diagonal entry is negative (physically impossible — indicates numerical instability).
- **D-05:** No arbitrary condition number threshold is imposed. Phase 11 validation will reveal the actual distribution and inform whether threshold-based filtering is needed.

### Derivative Method Toggle
- **D-06:** Add `use_five_point_stencil: bool = True` field to `ParameterEstimation` (or its constructor). Default is True (production). Tests and regression comparisons can pass False to use the old forward-difference method.
- **D-07:** Both `finite_difference_derivative()` and `five_point_stencil_derivative()` are retained. `compute_fisher_information_matrix()` dispatches based on the toggle.

### Stencil API Refactor
- **D-08:** Refactor `five_point_stencil_derivative()` to loop over all parameters internally and return `dict[str, Any]`, matching the existing `finite_difference_derivative()` API. This makes the toggle a clean swap in `compute_fisher_information_matrix()`.
- **D-09:** Replace `print()` calls in the existing 5-point method with `_LOGGER.info()` to match the codebase logging convention.

### Epsilon Values
- **D-10:** Keep current `derivative_epsilon = 1e-6` default for all parameters. The 5-point stencil footprint (+-2*epsilon = +-2e-6) is negligible relative to parameter ranges.
- **D-11:** During Phase 11 validation, log how many events hit `ParameterOutOfBoundsError`. If the rate is significant, tune individual per-parameter epsilons then.

### GPU Memory Monitoring
- **D-12:** Log peak GPU memory usage via `MemoryManagement` before and after the derivative loop. Provides diagnostic data for Phase 11 without changing the computation flow.

### Claude's Discretion
- Internal structure of the refactored loop (how intermediate waveforms are stored/deleted per parameter iteration)
- Whether to add explicit `del` for intermediate waveforms after each parameter's derivative computation
- Cleanup of the existing `parameter_space` mutation pattern in `five_point_stencil_derivative()` (the method currently accepts and overwrites `self.parameter_space`)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Physics Reference
- Vallisneri (2008) arXiv:gr-qc/0703086 -- reference for Fisher matrix derivative accuracy requirements. Cite at the call site where `five_point_stencil_derivative()` replaces `finite_difference_derivative()`.

### Codebase Files
- `master_thesis_code/parameter_estimation/parameter_estimation.py` -- target file; contains both derivative methods (lines 140-243), `compute_fisher_information_matrix()` (line 335), and `compute_Cramer_Rao_bounds()` (line 362)
- `master_thesis_code/main.py:197-198` -- timeout handler (`signal.SIGALRM`, currently 30s)
- `master_thesis_code/datamodels/parameter_space.py:36` -- `derivative_epsilon: float = 1e-6` default
- `master_thesis_code/exceptions.py` -- `ParameterOutOfBoundsError`, `TimeoutError`
- `master_thesis_code/decorators.py` -- `timer_decorator` used on `compute_Cramer_Rao_bounds`
- `master_thesis_code/gpu/memory_management.py` -- `MemoryManagement` class for GPU monitoring

### Prior Phase Context
- `.planning/phases/09-galactic-confusion-noise/09-CONTEXT.md` -- Phase 9 toggle pattern (D-04: `include_confusion_noise: bool = True`) is the model for the derivative method toggle

### Requirements
- `.planning/REQUIREMENTS.md` -- PHYS-01 (5-point stencil), PHYS-03 (CRB timeout increase)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `five_point_stencil_derivative()` at line 187: already implements the O(epsilon^4) stencil formula `(-f(+2h) + 8f(+h) - 8f(-h) + f(-2h)) / 12h`. Needs refactoring (single-param -> loop) but math is correct.
- `finite_difference_derivative()` at line 140: provides the target API pattern (loop over all params, return dict)
- `_crop_to_same_length()`: already used by both methods for handling variable-length waveforms
- `MemoryManagement` class: has `display_GPU_information()` for logging GPU state

### Established Patterns
- Toggle pattern from Phase 9: `include_confusion_noise: bool = True` on `LisaTdiConfiguration` dataclass
- `ParameterOutOfBoundsError` is caught in `main.py` data_simulation loop -- event is skipped, loop continues
- Fisher matrix symmetry optimization (upper triangle only, line 347) -- already in place, applies to both derivative methods

### Integration Points
- `compute_fisher_information_matrix()` is the only consumer of derivative methods -- single call site to change
- `main.py:198` timeout value is the only place to update (30s -> 90s)
- `compute_Cramer_Rao_bounds()` at line 362 performs `np.matrix(...).I` -- this is where LinAlgError would be caught and condition number logged

</code_context>

<specifics>
## Specific Ideas

No specific requirements -- the implementation follows directly from the existing code structure and the Vallisneri (2008) reference for derivative accuracy.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 10-five-point-stencil-derivatives*
*Context gathered: 2026-03-29*
