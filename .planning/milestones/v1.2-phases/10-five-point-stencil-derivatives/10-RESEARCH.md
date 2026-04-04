# Phase 10: Five-Point Stencil Derivatives - Research

**Researched:** 2026-03-29
**Domain:** Numerical derivatives for Fisher information matrix (gravitational wave parameter estimation)
**Confidence:** HIGH

## Summary

Phase 10 replaces the O(epsilon) forward-difference derivative in `compute_fisher_information_matrix()` with the existing O(epsilon^4) five-point stencil method. The `five_point_stencil_derivative()` method already exists at `parameter_estimation.py:187` with correct math but needs refactoring to match the `finite_difference_derivative()` API (loop over all parameters, return `dict[str, Any]`). Additionally, the CRB computation timeout must increase from 30s to 90s, and condition number monitoring must be added.

This is a well-scoped "wiring" phase: the stencil formula is already implemented and verified. The main work is (1) refactoring the 5-point method's API, (2) adding the toggle, (3) updating timeouts, and (4) adding condition number logging.

**Primary recommendation:** Refactor `five_point_stencil_derivative()` to loop internally and return `dict[str, Any]`, add `use_five_point_stencil: bool = True` to `ParameterEstimation.__init__`, dispatch in `compute_fisher_information_matrix()`, update both `signal.alarm(30)` calls in `main.py` to `signal.alarm(90)`, and log `np.linalg.cond()` before matrix inversion.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** CRB computation timeout increased from 30s to 90s (fixed value in main.py)
- **D-02:** Timeout revisited in Phase 11 validation campaign
- **D-03:** Condition number kappa(Gamma) logged at INFO level for every event, before matrix inversion
- **D-04:** Events skipped (WARNING log) when LinAlgError (singular) or negative CRB diagonal (numerical instability)
- **D-05:** No arbitrary condition number threshold. Phase 11 validation informs filtering
- **D-06:** Add `use_five_point_stencil: bool = True` field to ParameterEstimation (or constructor). Default True. Tests can pass False
- **D-07:** Both derivative methods retained. Dispatch based on toggle
- **D-08:** Refactor five_point_stencil_derivative() to loop over all parameters internally and return dict[str, Any]
- **D-09:** Replace print() calls in 5-point method with _LOGGER.info()
- **D-10:** Keep current derivative_epsilon = 1e-6 default for all parameters
- **D-11:** Phase 11 monitors ParameterOutOfBoundsError rate; tune epsilons then if needed
- **D-12:** Log peak GPU memory via MemoryManagement before and after derivative loop

### Claude's Discretion
- Internal structure of the refactored loop (intermediate waveform storage/deletion per parameter)
- Whether to add explicit `del` for intermediate waveforms after each parameter
- Cleanup of existing parameter_space mutation pattern in five_point_stencil_derivative()

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PHYS-01 | Fisher matrix uses 5-point stencil derivative instead of forward-difference | Existing method at PE:187 has correct formula. Refactor to dict API + wire into compute_fisher_information_matrix() |
| PHYS-03 | CRB computation timeout increased to accommodate 5-point stencil (56 waveforms vs 15) | Two signal.alarm(30) calls in main.py at lines 255 and 332 both need updating to 90 |
</phase_requirements>

## Architecture Patterns

### Current Code Structure (Before)

```
parameter_estimation.py
├── finite_difference_derivative()     # lines 140-185: loops all params, returns dict[str, Any]
│   └── Waveform count: 1 (base) + 14 (perturbed) = 15 total
├── five_point_stencil_derivative()    # lines 187-243: single param, returns array
│   └── Waveform count per param: 4 stencil points (no base needed)
├── compute_fisher_information_matrix()  # line 335: calls finite_difference_derivative()
├── compute_Cramer_Rao_bounds()        # line 362: inverts Fisher matrix via np.matrix().I
└── scalar_product_of_functions()      # line 266: inner product (hot path)
```

### Target Code Structure (After)

```
parameter_estimation.py
├── __init__()                          # gains use_five_point_stencil: bool = True
├── finite_difference_derivative()      # UNCHANGED: loops all params, returns dict[str, Any]
│   └── 15 waveform evaluations
├── five_point_stencil_derivative()     # REFACTORED: loops all params, returns dict[str, Any]
│   └── 56 waveform evaluations (4 per param * 14 params)
├── compute_fisher_information_matrix() # MODIFIED: dispatches on use_five_point_stencil toggle
├── compute_Cramer_Rao_bounds()         # MODIFIED: condition number logging + negative diagonal check
└── scalar_product_of_functions()       # UNCHANGED
```

### Pattern: Derivative Method Toggle (following Phase 9 confusion noise pattern)

The Phase 9 `include_confusion_noise: bool = True` on `LisaTdiConfiguration` is the established toggle pattern. Apply identically:

```python
class ParameterEstimation:
    def __init__(
        self,
        waveform_generation_type: WaveGeneratorType,
        parameter_space: ParameterSpace,
        *,
        use_gpu: bool = True,
        use_five_point_stencil: bool = True,  # D-06: toggle, default True
    ):
        self._use_five_point_stencil = use_five_point_stencil
        # ... existing init code ...
```

### Pattern: Refactored five_point_stencil_derivative()

Current API: `five_point_stencil_derivative(self, parameter, parameter_space) -> array`
Target API: `five_point_stencil_derivative(self) -> dict[str, Any]`

Key changes from current implementation:
1. Remove `parameter` and `parameter_space` arguments -- iterate `vars(self.parameter_space)` like `finite_difference_derivative()` does
2. Accumulate results in `derivatives: dict[str, Any] = {}` like forward-diff
3. Replace `print()` with `_LOGGER.info()` (D-09)
4. Keep per-parameter memory cleanup (`del lisa_responses` after each param)

```python
def five_point_stencil_derivative(self) -> dict[str, Any]:
    """O(epsilon^4) five-point stencil partial derivatives for all parameters.

    Computes (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h) for each of the
    14 EMRI parameters. Requires 4 waveform evaluations per parameter (56 total).

    Returns:
        Dictionary mapping parameter symbols to derivative waveform arrays.

    References:
        Vallisneri (2008), arXiv:gr-qc/0703086 -- derivative accuracy for Fisher matrices
    """
    derivatives: dict[str, Any] = {}

    for parameter in vars(self.parameter_space).values():
        _LOGGER.info(f"Computing 5-point stencil derivative w.r.t. {parameter.symbol}.")

        derivative_epsilon = parameter.derivative_epsilon

        # Bounds check: +/-2*epsilon must be within parameter range
        if ((parameter.value - 2 * derivative_epsilon) < parameter.lower_limit) or (
            (parameter.value + 2 * derivative_epsilon) > parameter.upper_limit
        ):
            raise ParameterOutOfBoundsError(
                f"Parameter {parameter.symbol} out of bounds for 5-point stencil "
                f"(value={parameter.value}, eps={derivative_epsilon})."
            )

        saved_value = parameter.value
        five_point_stencil_steps = [-2.0, -1.0, 1.0, 2.0]
        lisa_responses = []
        for step in five_point_stencil_steps:
            parameter.value = saved_value + step * derivative_epsilon
            lisa_responses.append(
                self.generate_lisa_response(
                    update_parameter_dict={parameter.symbol: parameter.value}
                )
            )
        parameter.value = saved_value  # restore original value

        lisa_responses = self._crop_to_same_length(lisa_responses)

        # (-f(+2h) + 8f(+h) - 8f(-h) + f(-2h)) / 12h
        # Eq. from Vallisneri (2008), arXiv:gr-qc/0703086
        derivative = (
            -lisa_responses[3] + 8 * lisa_responses[2]
            - 8 * lisa_responses[1] + lisa_responses[0]
        ) / (12 * derivative_epsilon)

        derivatives[parameter.symbol] = derivative
        del lisa_responses

    _LOGGER.info("Finished computing 5-point stencil partial derivatives.")
    return derivatives
```

### Pattern: Condition Number Logging and Error Handling

In `compute_Cramer_Rao_bounds()`:

```python
@timer_decorator
def compute_Cramer_Rao_bounds(self) -> dict:
    fisher_information_matrix = self.compute_fisher_information_matrix()

    # D-03: Log condition number before inversion
    fisher_np = cp.asnumpy(fisher_information_matrix)
    condition_number = np.linalg.cond(fisher_np)
    _LOGGER.info(f"Fisher matrix condition number: kappa = {condition_number:.2e}")

    try:
        cramer_rao_bounds = np.linalg.inv(fisher_np)
    except np.linalg.LinAlgError:
        # D-04a: singular matrix
        _LOGGER.warning("Fisher matrix is singular (LinAlgError). Skipping event.")
        raise

    # D-04b: check for negative diagonal entries (unphysical)
    diagonal = np.diag(cramer_rao_bounds)
    if np.any(diagonal < 0):
        neg_indices = np.where(diagonal < 0)[0]
        param_symbols = list(self.parameter_space._parameters_to_dict().keys())
        neg_params = [param_symbols[i] for i in neg_indices]
        _LOGGER.warning(
            f"Negative CRB diagonal entries for {neg_params}. "
            "Indicates numerical instability. Skipping event."
        )
        raise ParameterEstimationError(
            f"Negative CRB diagonal entries: {neg_params}"
        )
    # ... rest of extraction ...
```

**Important:** Replace `np.matrix(...).I` (deprecated) with `np.linalg.inv()`. The `np.matrix` class has been deprecated since NumPy 1.20 and should not be used in new code.

### Anti-Patterns to Avoid

- **Mutating self.parameter_space in derivative method:** The current `five_point_stencil_derivative()` accepts and overwrites `self.parameter_space` (line 203-204). The refactored version must NOT accept an external parameter_space -- it should only use `self.parameter_space` and restore parameter values after each derivative.
- **Not restoring parameter values:** After computing each parameter's derivative, the parameter's `.value` must be restored to its original. The current forward-diff method doesn't do this (it only needs one perturbed value), but the 5-point method mutates the value 4 times per parameter.
- **Using np.matrix:** Currently used at line 365. Replace with `np.linalg.inv()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Matrix inversion | Custom inverse | `np.linalg.inv()` | Handles edge cases, raises `LinAlgError` cleanly |
| Condition number | Manual eigenvalue ratio | `np.linalg.cond()` | Standard, handles ill-conditioned matrices correctly |
| Derivative formula | Custom stencil weights | Existing implementation | Already correct: `(-f(+2h) + 8f(+h) - 8f(-h) + f(-2h)) / 12h` |

## Common Pitfalls

### Pitfall 1: Parameter Value Mutation Without Restore
**What goes wrong:** The 5-point stencil sets `parameter.value` to 4 different stencil points. If not restored, the next parameter's derivative starts from a wrong base point.
**Why it happens:** The `Parameter` dataclass is mutable -- setting `.value` modifies the shared object.
**How to avoid:** Save `saved_value = parameter.value` before the stencil loop, restore `parameter.value = saved_value` after.
**Warning signs:** CRB values that change depending on parameter ordering.

### Pitfall 2: Bounds Check Width Doubled for 5-Point Stencil
**What goes wrong:** Forward-diff only needs `value + epsilon` in bounds. Five-point stencil needs `value +/- 2*epsilon` in bounds -- both sides, twice as wide.
**Why it happens:** The stencil evaluates at `[-2h, -h, +h, +2h]` around the current value.
**How to avoid:** Check both `value - 2*epsilon >= lower_limit` AND `value + 2*epsilon <= upper_limit`. The existing method already does this correctly (line 210-215).
**Warning signs:** `ParameterOutOfBoundsError` rate increases significantly. D-11 defers tuning to Phase 11.

### Pitfall 3: Timeout Too Short for CRB Computation
**What goes wrong:** 56 waveform evaluations (vs 15) + 105 inner products means CRB computation takes ~3-4x longer. The 30s timeout kills almost every event.
**Why it happens:** Both `signal.alarm(30)` calls in `main.py` (lines 255 and 332) need updating.
**How to avoid:** Update BOTH alarm calls to 90s (D-01). The first alarm covers SNR computation (unchanged), but the same handler is used -- keeping it at 30s for SNR is acceptable, only the CRB alarm at line 332 strictly needs the increase. However, D-01 specifies a single timeout value for simplicity.
**Warning signs:** All events time out with "Cramer-Rao bound computation timed out" warnings.

### Pitfall 4: np.matrix Deprecation
**What goes wrong:** `np.matrix` is deprecated and scheduled for removal. The `.I` property for inversion is fragile -- doesn't raise `LinAlgError` cleanly for singular matrices.
**Why it happens:** Legacy code pattern from older NumPy.
**How to avoid:** Replace `np.matrix(fisher_np).I` with `np.linalg.inv(fisher_np)`. This also makes `LinAlgError` catching (D-04a) reliable.
**Warning signs:** DeprecationWarning from NumPy, or silent wrong results from singular matrices.

### Pitfall 5: Physics Change Protocol Trigger
**What goes wrong:** This phase modifies `parameter_estimation.py` which is a physics-trigger file.
**Why it happens:** The derivative method change alters computed CRB values (more accurate derivatives -> different Fisher matrix entries -> different CRBs).
**How to avoid:** Invoke `/physics-change` skill before implementation. Present old formula (forward-diff), new formula (5-point stencil), reference (Vallisneri 2008), dimensional analysis, and limiting case.
**Warning signs:** Commit rejected by pre-commit hook or reviewer.

## Code Examples

### Dispatch in compute_fisher_information_matrix()

```python
def compute_fisher_information_matrix(self) -> Any:
    parameter_symbol_list = list(self.parameter_space._parameters_to_dict().keys())

    # Eq. from Vallisneri (2008), arXiv:gr-qc/0703086
    if self._use_five_point_stencil:
        lisa_response_derivatives = self.five_point_stencil_derivative()
    else:
        lisa_response_derivatives = self.finite_difference_derivative()

    # ... rest unchanged (Fisher matrix assembly from inner products) ...
```

### Timeout Update in main.py

Two locations need updating:

```python
# Line ~255: SNR + waveform timeout
signal.alarm(90)  # was 30; D-01

# Line ~332: CRB computation timeout
signal.alarm(90)  # was 30; D-01
```

Also update the alarm handler message and the except block messages:
```python
def _alarm_handler(signum: int, frame: object) -> None:
    raise TimeoutError("Computation exceeded 90s timeout")  # was 30s
```

### GPU Memory Logging (D-12)

```python
def five_point_stencil_derivative(self) -> dict[str, Any]:
    # D-12: Log GPU memory before derivative loop
    if _CUPY_AVAILABLE and cp is not None:
        pool = cp.get_default_memory_pool()
        _LOGGER.info(f"GPU memory before derivatives: {pool.total_bytes() / 1e9:.2f} GB")

    derivatives: dict[str, Any] = {}
    for parameter in vars(self.parameter_space).values():
        # ... derivative computation ...
        pass

    # D-12: Log GPU memory after derivative loop
    if _CUPY_AVAILABLE and cp is not None:
        _LOGGER.info(f"GPU memory after derivatives: {pool.total_bytes() / 1e9:.2f} GB")

    return derivatives
```

## Waveform Count Analysis

| Method | Base waveform | Per-parameter | Total (14 params) | Inner products |
|--------|--------------|---------------|-------------------|----------------|
| Forward-diff | 1 | 1 | 15 | 105 |
| Five-point stencil | 0 | 4 | 56 | 105 |
| Ratio | - | 4x | 3.7x | 1x |

The inner product count (105 for upper triangle of 14x14 symmetric matrix) is unchanged. Only the derivative computation takes longer. Total wall time increase is less than 4x because inner products dominate for large waveforms.

## Project Constraints (from CLAUDE.md)

- **Physics Change Protocol:** MANDATORY before modifying `parameter_estimation.py` with formula changes. Must present old formula, new formula, reference, dimensional analysis, limiting case. Wait for user approval.
- **Git convention:** Commit must use `[PHYSICS]` prefix for the derivative method change.
- **Typing:** All functions must have complete type annotations. Use `dict[str, Any]` not `Dict[str, Any]`.
- **No np.matrix:** Replace deprecated `np.matrix` usage at the same time.
- **Logging:** Use `_LOGGER` (module-level `logging.getLogger()`), never `print()`.
- **GPU guards:** CuPy imports must be guarded with `try/except ImportError`.
- **Pre-commit:** ruff lint + format + mypy run on every commit.
- **Testing:** No test for `ParameterEstimation` can run on CPU (requires cupy/GPU). Tests must use `@pytest.mark.gpu` marker or `pytest.importorskip("cupy")`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest -m "not gpu and not slow" -x` |
| Full suite command | `uv run pytest` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PHYS-01 | five_point_stencil_derivative() called by compute_fisher_information_matrix() when toggle=True | unit (GPU) | `uv run pytest -m gpu -k test_fisher_uses_five_point_stencil -x` | No -- Wave 0 |
| PHYS-01 | Toggle dispatches to correct method | unit (CPU mock) | `uv run pytest -k test_derivative_toggle_dispatch -x` | No -- Wave 0 |
| PHYS-03 | Timeout is 90s (not 30s) | unit/integration | `uv run pytest -k test_alarm_timeout_value -x` | No -- Wave 0 |
| D-03 | Condition number logged at INFO | unit (CPU mock) | `uv run pytest -k test_condition_number_logged -x` | No -- Wave 0 |
| D-04 | Negative diagonal CRBs raise ParameterEstimationError | unit (CPU) | `uv run pytest -k test_negative_crb_diagonal_raises -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest -m "not gpu and not slow" -x`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow"`
- **Phase gate:** Full CPU suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/parameter_estimation/test_parameter_estimation.py` -- covers toggle dispatch (CPU-testable with mocks), condition number logging, negative diagonal check
- [ ] GPU tests require `@pytest.mark.gpu` and cannot run in CI -- manual verification on cluster

## Open Questions

1. **SNR alarm vs CRB alarm**
   - What we know: Both `signal.alarm(30)` calls in main.py (lines 255 and 332) will be set to 90s per D-01.
   - What's unclear: The SNR computation (15-30s typically) doesn't need 90s. Keeping both at 90s is simpler but less protective against hung waveform computations.
   - Recommendation: Set both to 90s as specified in D-01. Phase 11 can differentiate if needed.

2. **np.linalg.LinAlgError propagation**
   - What we know: D-04a says events are skipped on LinAlgError. The `main.py` loop already catches generic exceptions.
   - What's unclear: Whether to catch `LinAlgError` inside `compute_Cramer_Rao_bounds()` and re-raise as `ParameterEstimationError`, or let it propagate and catch in `main.py`.
   - Recommendation: Let `LinAlgError` propagate. The `main.py` loop's catch-all `(ZeroDivisionError, RuntimeError, ValueError)` at line 343 doesn't catch it, so add `np.linalg.LinAlgError` to that tuple.

## Sources

### Primary (HIGH confidence)
- `parameter_estimation.py` source code -- both derivative methods examined line by line
- `main.py` source code -- timeout handling, alarm calls, exception handling
- CONTEXT.md decisions D-01 through D-12 -- locked implementation choices
- Phase 9 confusion noise toggle pattern in `LISA_configuration.py` -- established toggle pattern

### Secondary (MEDIUM confidence)
- Vallisneri (2008) arXiv:gr-qc/0703086 -- cited in CLAUDE.md Known Bug 4 and CONTEXT.md. Standard reference for Fisher matrix derivative accuracy in gravitational wave parameter estimation.
- NumPy documentation on `np.matrix` deprecation -- deprecated since NumPy 1.20, `np.linalg.inv()` is the replacement.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, pure refactoring of existing code
- Architecture: HIGH -- pattern established by Phase 9 toggle, code fully inspected
- Pitfalls: HIGH -- all pitfalls derived from direct code inspection
- Validation: MEDIUM -- GPU tests cannot be verified on dev machine; CPU-mock tests for toggle/logging are straightforward

**Research date:** 2026-03-29
**Valid until:** 2026-04-28 (stable -- no external dependencies or fast-moving libraries)
