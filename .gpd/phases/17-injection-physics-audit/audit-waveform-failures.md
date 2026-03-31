# AUDT-03: Waveform Failure Characterization Report

**Phase:** 17-injection-physics-audit, Plan 02
**Date:** 2026-03-31
**Scope:** Code-level audit of waveform failure modes in `injection_campaign()` and source tracing to `parameter_estimation.py` / `few` library

## Summary Table

| Exception Type | Caught at (main.py) | Source | Physical Trigger | Expected Susceptible (z, M, e0) Region |
|---|---|---|---|---|
| `Warning` ("Mass ratio") | L510-516 | `few` library internal check | Mass ratio mu/M outside `few`'s valid range | Low M (near 10^4) with high mu, or high M (near 10^7) |
| `Warning` (other) | L517-519 | `few` / `fastlisaresponse` / NumPy | Various numerical warnings promoted to errors | Varies; edge-case parameters |
| `ParameterOutOfBoundsError` | L520-525 | `parameter_estimation.py:167,219` | Parameter + derivative_epsilon exceeds `ParameterSpace` bounds | Not applicable to injection (no Fisher matrix), but raised if `few` calls trigger internal bound checks |
| `RuntimeError` | L526-531 | `few` ODE solver / `fastlisaresponse` | Trajectory integration diverges (solver failure) | High e0 (near 0.7), extreme mass ratios, edge-case orbital parameters (p0 near separatrix) |
| `ValueError` ("EllipticK") | L532-538 | `few` internal: elliptic integral evaluation | Argument to EllipticK outside valid domain (|m| >= 1) | High e0 (eccentricity near boundaries where elliptic integrals diverge) |
| `ValueError` ("Brent root solver") | L539-545 | `few` internal: root-finding for orbital evolution | Brent method fails to converge (no bracket or oscillation) | Extreme parameter combinations where orbital evolution equations have no solution in expected interval |
| `ValueError` (other) | L544-545 | — | Not caught; re-raised | — |
| `ZeroDivisionError` | L546-551 | `few` trajectory integration or PSD computation | Division by zero in intermediate computation | Edge-case parameters producing zero denominators (e.g., zero frequency bin in PSD, degenerate orbital config) |
| `TimeoutError` | L552-554 | `main.py:428` (SIGALRM handler) | Waveform generation exceeds 30s timeout (`_TIMEOUT_S = 30`) | Long-duration waveforms (low-frequency, high M), complex orbital dynamics (high e0 with many harmonics) |

## Detailed Exception Analysis

### 1. Warning (promoted to error via `warnings.filterwarnings("error")`)

**Caught at:** `main.py:510-519`
**Mechanism:** Line 505 sets `warnings.filterwarnings("error")`, promoting all Python warnings to exceptions. Two sub-cases:

**1a. "Mass ratio" warning (L512-516):**
- **Source:** `few` library performs internal consistency checks on the mass ratio mu/M. When the ratio falls outside the training/interpolation range of the `few` surrogate model, it emits a warning.
- **Physical trigger:** The `few` Pn5AAK model is trained on a specific mass-ratio range. The `ParameterSpace` allows M in [10^4, 10^7] solar masses and mu in [1, 100] solar masses, giving mass ratios q = mu/M in [10^-7, 10^-2]. The `few` model may warn for extreme ends of this range.
- **Logging:** Yes — `_ROOT_LOGGER.warning(...)` at L513-515.
- **Counter:** No counter incremented.
- **Recording:** `continue` at L516 — event is NOT appended to `results`.

**1b. Other warnings (L517-519):**
- **Source:** Any other warning from NumPy, SciPy, `few`, or `fastlisaresponse` (e.g., overflow, invalid value encountered in floating point operation).
- **Physical trigger:** Various numerical edge cases.
- **Logging:** Yes — `_ROOT_LOGGER.warning(f"{str(e)}. Continue...")` at L518.
- **Counter:** No counter incremented.
- **Recording:** `continue` at L519 — event is NOT appended to `results`.

### 2. ParameterOutOfBoundsError

**Caught at:** `main.py:520-525`
**Source locations:**
- `parameter_estimation.py:167` — raised in `finite_difference_derivative()` when `parameter.value + derivative_epsilon > parameter.upper_limit`. However, `injection_campaign()` only calls `compute_signal_to_noise_ratio()`, NOT `finite_difference_derivative()` or `five_point_stencil_derivative()`. So this source is **not reachable** from the injection path.
- `parameter_estimation.py:219` — raised in `five_point_stencil_derivative()`. Also **not reachable** from the injection path.
- The `except` clause in `injection_campaign()` is a **defensive catch** inherited from `data_simulation()` (which does compute Fisher matrices). In the injection campaign, this exception can only be raised if `few` or `fastlisaresponse` internally raises it (unlikely — they typically raise `RuntimeError` or `ValueError`), OR if `ParameterOutOfBoundsError` is raised during `generate_lisa_response()` → `ResponseWrapper.__call__()`.

**Physical trigger:** Parameter value outside bounds during waveform generation (unlikely in injection path since no derivative stencil is computed, but kept as defensive guard).
**Logging:** Yes — L522-524.
**Counter:** No counter incremented.
**Recording:** `continue` at L525.

### 3. RuntimeError

**Caught at:** `main.py:526-531`
**Source:** The `few` waveform generators (`Pn5AAKWaveform`, `FastSchwarzschildEccentricFlux`) use ODE integrators for EMRI trajectory evolution. When the ODE solver diverges or encounters a numerical singularity, it raises `RuntimeError`.
**Physical trigger:**
- High eccentricity (e0 near 0.7): The trajectory passes close to the separatrix where the ODE becomes stiff.
- Extreme semi-latus rectum (p0 near 10.0): Close to the last stable orbit, where plunge dynamics cause rapid evolution.
- Certain spin (a) + inclination (x0) + eccentricity combinations that produce unstable trajectories.
- `fastlisaresponse.ResponseWrapper` can also raise `RuntimeError` for TDI channel computation failures.
**Logging:** Yes — L528-530.
**Counter:** No counter incremented.
**Recording:** `continue` at L531.

### 4. ValueError — EllipticK sub-case

**Caught at:** `main.py:532-538`
**Source:** `few` internal computation of elliptic integrals (EllipticK, EllipticE) for Kerr geodesic frequencies. Raised when the elliptic integral argument m approaches or exceeds 1, where EllipticK diverges.
**Physical trigger:** High eccentricity combined with specific orbital parameters that push the elliptic integral argument to the boundary of its domain. This corresponds to orbits near the separatrix in Kerr spacetime.
**Logging:** Yes — L535-537.
**Counter:** No counter incremented.
**Recording:** `continue` at L538.

### 5. ValueError — Brent root solver sub-case

**Caught at:** `main.py:539-545`
**Source:** `few` internal root-finding (Brent method) used for computing orbital parameters. The Brent solver requires a sign change in the function across the bracket; when no sign change exists (no root in the interval), it fails to converge.
**Physical trigger:** Extreme parameter combinations where the standard orbital evolution framework breaks down — typically at the boundary of the parameter space where the `few` model's assumptions are violated.
**Logging:** Yes — L540-542.
**Counter:** No counter incremented.
**Recording:** `continue` at L543.

### 6. ValueError — other (not caught)

**At:** `main.py:544-545`
**Behavior:** Any `ValueError` that does NOT match the "EllipticK" or "Brent root solver" patterns is **re-raised** (`raise` at L545). This means unexpected `ValueError`s will crash the injection campaign task.
**Risk:** If `few` introduces a new failure mode that raises `ValueError` with a different message, it will not be caught and will terminate the SLURM task.

### 7. ZeroDivisionError

**Caught at:** `main.py:546-551`
**Source:** Can arise from:
- `few` trajectory integration computing intermediate quantities that go to zero.
- Edge cases in frequency/PSD computation where a frequency bin is exactly zero (unlikely with proper frequency cropping, but defensive).
**Physical trigger:** Degenerate orbital configurations or numerical edge cases where an intermediate denominator evaluates to zero.
**Logging:** Yes — L548-550.
**Counter:** No counter incremented.
**Recording:** `continue` at L551.

### 8. TimeoutError

**Caught at:** `main.py:552-554`
**Source:** `main.py:428` — the `_alarm_handler` function raises `TimeoutError("Computation exceeded 90s timeout")` when `SIGALRM` fires. The alarm is set at L506 with `_TIMEOUT_S = 30` seconds (L462).
**Note:** The error message says "90s" (inherited from the handler definition at L428 in `data_simulation`) but the actual timeout is 30s (set by `signal.alarm(_TIMEOUT_S)` at L506 where `_TIMEOUT_S = 30` at L462). The message is misleading but the behavior is correct — the alarm fires after 30s.
**Physical trigger:** Waveform generation takes too long. This happens for:
- High central mass M: longer waveforms with more orbital cycles.
- Complex orbital configurations: high eccentricity with many harmonics requiring more computation.
- High-resolution waveforms that stress the FFT and response function computation.
**Logging:** Yes — L553 (but message says ">90s" which is incorrect; actual timeout is 30s).
**Counter:** No counter incremented.
**Recording:** `continue` at L554.

## Comparison with data_simulation() Exception Handling

The `data_simulation()` function (main.py:188-383) has the same exception types but with two key differences:

1. **Additional exception: `AssertionError`** (L295-298): Caught in `data_simulation()` but NOT in `injection_campaign()`. This is a sic (typo for `AssertionError` -> `AssertionError` exists in the code; Python accepts it as a valid identifier but it catches `AssertionError` which is the built-in).
   - **NOTE:** Python's built-in is `AssertionError` — checking: actually Python's built-in is `AssertionError`. Looking at the code: `except AssertionError` at L295. Python does NOT have a built-in `AssertionError` — the built-in is `AssertionError`. Wait — let me be precise: Python's built-in is `AssertionError`. The code says `AssertionError` at L295. Actually, Python has `AssertionError` — this IS the correct built-in name. The code catches `AssertionError` from `few`'s internal assertions.
   - Present in `data_simulation()` but **absent from `injection_campaign()`**. If `few` raises an `AssertionError`, the injection campaign will crash rather than skip.

2. **Additional exceptions for CRB computation** (L340-362): `data_simulation()` has a second try/except block for `compute_Cramer_Rao_bounds()` catching `ParameterOutOfBoundsError`, `np.linalg.LinAlgError`, `ParameterEstimationError`, `TimeoutError`, `ZeroDivisionError`, `RuntimeError`, `ValueError`. These are **not relevant** to `injection_campaign()` which does not compute CRB.

3. **Timeout value:** `data_simulation()` uses 90s timeout (L265: `signal.alarm(90)`); `injection_campaign()` uses 30s (L506: `signal.alarm(_TIMEOUT_S)` with `_TIMEOUT_S = 30` at L462).

## CSV Design Limitation

### The Problem: Failed Events Are Not Recorded

The injection campaign CSV only records events where the SNR computation **succeeds**. All failure modes result in `continue` statements that skip the `results.append()` call.

**Code path for successful events:**
- `results` list initialized empty: **L451** (`results: list[dict[str, float]] = []`)
- Successful SNR computation at L507 → event appended at **L557-567** (`results.append({...})`)
- Counter incremented at **L568** (`counter += 1`)
- CSV written from `results` at **L572** (periodic flush) and **L576** (final write)

**Code path for failed events (ALL exception handlers):**
- `Warning` (mass ratio): `continue` at **L516** — skips L557 append
- `Warning` (other): `continue` at **L519** — skips L557 append
- `ParameterOutOfBoundsError`: `continue` at **L525** — skips L557 append
- `RuntimeError`: `continue` at **L531** — skips L557 append
- `ValueError` (EllipticK): `continue` at **L538** — skips L557 append
- `ValueError` (Brent): `continue` at **L543** — skips L557 append
- `ZeroDivisionError`: `continue` at **L551** — skips L557 append
- `TimeoutError`: `continue` at **L554** — skips L557 append

**Additionally:** Events with z > z_cut (0.5) are skipped at **L477-479** (`continue` without any logging to CSV).

### CSV Column List

Defined at **L386**: `_INJECTION_COLUMNS = ["z", "M", "phiS", "qS", "SNR", "h_inj", "luminosity_distance"]`

**No failure status column exists.** The CSV contains only successfully-computed events. There is no way to determine from the CSV alone:
- How many events were attempted
- How many failed
- What type of failure occurred
- What parameters the failed events had

### What IS Available

The failure information is partially available in **SLURM log files** (stdout/stderr):
- Every exception handler logs a warning message via `_ROOT_LOGGER.warning(...)`.
- The progress log at L487-489 prints `{counter} / {iteration}` giving the ratio of successful to total attempted computations (excluding high-z skips).
- However, the warning messages do **not** include the parameter values (z, M, e0, etc.) of the failed event. Only `ParameterOutOfBoundsError` includes the error string which sometimes contains parameter information.

### Key Design Consequence

The `simulation_steps` parameter controls the number of **successful** events, not total attempts. The while loop at L464 runs until `counter == simulation_steps`. Failed events increment `iteration` but not `counter`. This means:
- Requesting 500 successful events may require >> 500 attempts
- The failure rate cannot be computed from the CSV alone (need CSV row count vs SLURM array task count is not enough — each task runs until it gets `simulation_steps` successes)
- The failure rate can only be estimated from `{counter} / {iteration}` log messages

## Expected Failure Susceptibility by Parameter Region (Code-Level Analysis)

Based on the code and known `few` library characteristics:

### High Eccentricity (e0 > 0.5, especially near 0.7)

- **RuntimeError:** ODE trajectory solver diverges near separatrix in Kerr spacetime.
- **ValueError (EllipticK):** Elliptic integral argument approaches 1 as eccentricity increases.
- **ValueError (Brent):** Root-finding for orbital evolution fails when the orbit is near plunge.
- **TimeoutError:** High-eccentricity waveforms have more harmonics and require more computation.
- **Eccentricity range:** `ParameterSpace` allows e0 in [0.05, 0.7]. The upper end (e0 > 0.5) is expected to concentrate failures.

### Extreme Central Mass (M near boundaries)

- **Warning (mass ratio):** At M ~ 10^4 with mu ~ 100, the mass ratio q = 10^-2 is at the edge of `few`'s valid range. At M ~ 10^7 with mu ~ 1, q = 10^-7 is extremely small.
- **RuntimeError:** Very high M means very low GW frequency — waveforms may not fit in the observation window, or frequency may fall below `MINIMAL_FREQUENCY`.
- **TimeoutError:** High M means more orbital cycles and longer waveform computation.
- **Mass range:** `ParameterSpace` allows M in [10^4, 10^7] solar masses (log-uniform).

### High Redshift (z > 0.3)

- **Note:** The z_cut = 0.5 already eliminates z > 0.5 events. For z in [0.3, 0.5]:
- High z means large luminosity distance, which means low SNR (signal amplitude ~ 1/d_L). These events rarely produce SNR > 20 detections.
- **TimeoutError:** Unlikely correlation with z specifically; d_L only affects the overall amplitude, not the waveform complexity.
- The primary effect of high z is **wasted computation** (computing SNR for events that will never be detected), not waveform failure.

### Orbital Parameters Near Boundaries

- **Semi-latus rectum p0 near 10.0:** Close to the last stable orbit for Kerr black holes. The `few` model may struggle here.
- **Spin a near 1.0:** Extreme Kerr limit where numerical methods become less accurate.
- **Inclination x0 = cos(I) near +/-1.0:** Polar orbits may trigger edge cases in the `few` model.

### Parameter Combinations

The most failure-prone region is the **triple combination**: high e0 + low p0 + high a. This corresponds to eccentric orbits close to the last stable orbit around a rapidly spinning black hole — exactly the regime where EMRI physics is most complex and numerical methods are most stressed.

## Future Tracking Proposal

### 1. Add Failure CSV (Recommended)

Create a parallel CSV recording failed events with columns:
```
z, M, e0, a, p0, mu, phiS, qS, h_inj, luminosity_distance, exception_type, exception_message, iteration_number
```

Implementation: After each `except` block, before `continue`, append to a failure list:
```python
failures.append({
    "z": sample.redshift,
    "M": sample.M,
    "e0": parameter_estimation.parameter_space.e0.value,
    "a": parameter_estimation.parameter_space.a.value,
    "p0": parameter_estimation.parameter_space.p0.value,
    "mu": parameter_estimation.parameter_space.mu.value,
    "phiS": parameter_estimation.parameter_space.phiS.value,
    "qS": parameter_estimation.parameter_space.qS.value,
    "h_inj": h_value,
    "luminosity_distance": luminosity_distance,
    "exception_type": type(e).__name__,
    "exception_message": str(e)[:200],
    "iteration": iteration,
})
```

Write to `injection_failures_h_{h_label}_task_{index}.csv` at the same flush intervals.

### 2. Add Failure Counters to Campaign Summary

At the end of `injection_campaign()`, log a structured summary:
```python
_ROOT_LOGGER.info(
    f"Campaign summary: {counter} successes / {iteration} attempts "
    f"({skipped_high_z} high-z skipped, {failure_counts} by type)"
)
```

Where `failure_counts` is a dict tracking per-exception-type counts.

### 3. Add Parameter Values to Warning Messages

Current warning messages do not include the parameter values. Change e.g.:
```python
except RuntimeError as e:
    signal.alarm(0)
    _ROOT_LOGGER.warning(
        f"RuntimeError during waveform generation: {str(e)}. "
        f"Parameters: z={sample.redshift:.4f}, M={sample.M:.0f}, "
        f"e0={parameter_estimation.parameter_space.e0.value:.3f}. Continue..."
    )
    continue
```

### 4. Add AssertionError Catch (Missing from injection_campaign)

The `data_simulation()` function catches `AssertionError` (L295-298) but `injection_campaign()` does not. If `few` raises an assertion error, the injection task will crash. Add:
```python
except AssertionError as e:
    signal.alarm(0)
    _ROOT_LOGGER.warning(
        f"Caught AssertionError: {str(e)}. Continue with new parameters..."
    )
    continue
```

## Data Analysis Results

_To be populated by Task 2 after running `analyze_failures.py`._
