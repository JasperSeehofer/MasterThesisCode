---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - master_thesis_code/bayesian_inference/bayesian_statistics.py
autonomous: true
requirements: [diagnostic-bias-audit]
must_haves:
  truths:
    - "numerator_integrant_without_bh_mass no longer divides by d_L"
    - "numerator_integrant_with_bh_mass divides by (1+z) only, not (d_L * (1+z))"
    - "All four P_det calls are bypassed with 1.0 when debug flag is True"
    - "Module emits a warning at import time when debug flag is active"
  artifacts:
    - path: "master_thesis_code/bayesian_inference/bayesian_statistics.py"
      provides: "Diagnostic likelihood with d_L fix and P_det bypass"
      contains: "_DEBUG_DISABLE_DETECTION_PROBABILITY"
  key_links: []
---

<objective>
Apply two diagnostic changes to Pipeline B's `single_host_likelihood` to isolate the source of
H0 posterior bias: (1) remove the spurious `/d_L` factor from both numerator integrands,
(2) add a debug flag that sets P_det = 1 everywhere.

Purpose: Enable a controlled diagnostic run where the only signal comes from the Gaussian
likelihood and galaxy prior, without detection-probability weighting or the suspect d_L divisor.
Output: Modified `bayesian_statistics.py` ready for a diagnostic cluster run.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@master_thesis_code/bayesian_inference/bayesian_statistics.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add debug flag, remove /d_L, bypass P_det</name>
  <files>master_thesis_code/bayesian_inference/bayesian_statistics.py</files>
  <action>
Three targeted edits to `bayesian_statistics.py`:

**1a. Add debug constant and warning (after line 57, near other module constants):**
```python
_DEBUG_DISABLE_DETECTION_PROBABILITY = True
# TODO(bias-audit): Restore detection probability after diagnostic confirms bias source

if _DEBUG_DISABLE_DETECTION_PROBABILITY:
    _LOGGER.warning(
        "DIAGNOSTIC MODE: Detection probability disabled "
        "(_DEBUG_DISABLE_DETECTION_PROBABILITY=True)"
    )
```

**1b. Remove `/d_L` from numerator integrands:**

In `numerator_integrant_without_bh_mass` (around line 555-563): remove the trailing `/ d_L`
from the return expression. The last two factors should be:
```python
            * galaxy_redshift_normal_distribution.pdf(z)
        )
```
(no `/ d_L` at the end).

In `numerator_integrant_with_bh_mass` (around line 604-614): change `/ (d_L * (1 + z))` to
`/ (1 + z)`. This keeps the mass Jacobian but removes the spurious d_L divisor. Also remove
the `# TODO: check if this is correct` comment since we are now addressing it.

**1c. Bypass P_det calls when flag is True:**

In all four integrand functions, wrap the `detection_probability.*_interpolated(...)` call
so that when `_DEBUG_DISABLE_DETECTION_PROBABILITY` is True, the value `1.0` is used instead.

The four locations:
- `numerator_integrant_without_bh_mass`: `detection_probability.detection_probability_without_bh_mass_interpolated(d_L, phi, theta)` -> `1.0`
- `denominator_integrant_without_bh_mass`: same call -> `1.0`
- `numerator_integrant_with_bh_mass`: `detection_probability.detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta)` -> `1.0`
- `denominator_integrant_with_bh_mass_vectorized`: same call -> `1.0`

Use a simple inline ternary or local variable pattern:
```python
p_det = (
    1.0
    if _DEBUG_DISABLE_DETECTION_PROBABILITY
    else detection_probability.detection_probability_without_bh_mass_interpolated(
        d_L, phi, theta
    )
)
```
Then use `p_det` in the return expression where the call previously was.

Do NOT change any other logic, imports, type annotations, or function signatures.
  </action>
  <verify>
    <automated>cd /home/jasper/Repositories/MasterThesisCode && uv run python -c "import master_thesis_code.bayesian_inference.bayesian_statistics" 2>&1 | grep "DIAGNOSTIC MODE" && uv run ruff check master_thesis_code/bayesian_inference/bayesian_statistics.py && uv run mypy master_thesis_code/bayesian_inference/bayesian_statistics.py</automated>
  </verify>
  <done>
    - `_DEBUG_DISABLE_DETECTION_PROBABILITY = True` exists near module top
    - Warning logged at import time
    - `numerator_integrant_without_bh_mass` return has no `/ d_L`
    - `numerator_integrant_with_bh_mass` return has `/ (1 + z)` not `/ (d_L * (1 + z))`
    - All four P_det calls replaced with `1.0` when flag is True
    - ruff and mypy pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Run existing tests to confirm no regressions</name>
  <files></files>
  <action>
Run the full CPU test suite to ensure no test regressions from the changes.
The likelihood integration tests (if any) may produce different numerical values since P_det
is now 1.0 and the d_L factor is removed — if tests fail for that reason, that is expected
and should be noted but NOT fixed (since this is a diagnostic change).

```bash
uv run pytest -m "not gpu and not slow" -x -q
```
  </action>
  <verify>
    <automated>cd /home/jasper/Repositories/MasterThesisCode && uv run pytest -m "not gpu and not slow" -x -q 2>&1 | tail -5</automated>
  </verify>
  <done>
    - All existing CPU tests pass (or any failures are documented as expected due to the diagnostic changes)
  </done>
</task>

</tasks>

<verification>
1. `grep -n "_DEBUG_DISABLE_DETECTION_PROBABILITY" master_thesis_code/bayesian_inference/bayesian_statistics.py` shows the flag definition and all four usage sites
2. `grep -n "/ d_L" master_thesis_code/bayesian_inference/bayesian_statistics.py` returns zero matches in the numerator integrands (only in other unrelated lines)
3. Import of the module triggers the DIAGNOSTIC MODE warning
</verification>

<success_criteria>
- The `/d_L` factor is removed from both numerator integrands
- P_det is bypassed (set to 1.0) in all four integrand functions
- ruff, mypy, and pytest all pass
- The module is ready for a diagnostic cluster run
</success_criteria>

<output>
After completion, create `.planning/quick/260330-oaf-diagnostic-bias-fix-remove-d-l-factor-an/260330-oaf-SUMMARY.md`
</output>
