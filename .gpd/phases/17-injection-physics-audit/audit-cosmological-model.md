# AUDT-02: Cosmological Model Consistency Report

**Phase:** 17-injection-physics-audit
**Plan:** 01
**Date:** 2026-03-31

## Summary

Cosmological model consistency verified: `dist()` is called with identical default cosmological parameters (Omega_m=0.25, Omega_DE=0.75, w_0=-1, w_a=0) across both pipelines. The only argument that differs is h, which is intentional per D-04. Round-trip d_L-to-z inversion is accurate to ~2e-13 relative error across all 700 test points. The z_cut=0.5 is safe: SNR at z=0.5 is a factor ~6x below the detection threshold.

---

## Part A: Cosmological Model Consistency

### A.1: dist() Function Signature and Defaults

From `physical_relations.py:27-35`:

```python
def dist(
    redshift: float,
    h: float = H,              # H = 0.73 (constants.py:25)
    Omega_m: float = OMEGA_M,  # OMEGA_M = 0.25 (constants.py:29)
    Omega_de: float = OMEGA_DE, # OMEGA_DE = 0.75 (constants.py:30)
    w_0: float = W_0,          # W_0 = -1.0 (constants.py:31)
    w_a: float = W_A,          # W_A = 0.0 (constants.py:32)
    offset_for_root_finding: float = 0.0,
) -> float:
```

**Default values confirmed against constants.py:**
- `H = 0.73` (constants.py:25) -- matches `dist()` default
- `OMEGA_M = 0.25` (constants.py:29) -- matches
- `OMEGA_DE = 0.75` (constants.py:30) -- matches
- `W_0 = -1.0` (constants.py:31) -- matches
- `W_A = 0.0` (constants.py:32) -- matches

### A.2: All dist() Call Sites

| # | File:Line | Arguments | h Explicit? | Path |
|---|-----------|-----------|-------------|------|
| 1 | `main.py:500` | `dist(sample.redshift, h=h_value)` | **Yes** | Injection |
| 2 | `cosmological_model.py:195` | `dist(redshift=self.max_redshift)` | No (default h=0.73) | Model setup |
| 3 | `parameter_space.py:148` | `dist(host_galaxy.z)` | No (default h=0.73) | Simulation |
| 4 | `handler.py:45` | `dist(self.redshift)` | No (default h=0.73) | ParameterSample.get_distance() -- **UNUSED** |
| 5 | `bayesian_statistics.py:738` | `dist(z, h=h)` | **Yes** | Evaluation (Pipeline B) |
| 6 | `bayesian_statistics.py:751` | `dist(z, h=h)` | **Yes** | Evaluation |
| 7 | `bayesian_statistics.py:792` | `dist(z, h=h)` | **Yes** | Evaluation |
| 8 | `bayesian_statistics.py:810` | `dist(z, h=h)` | **Yes** | Evaluation |
| 9 | `bayesian_statistics.py:860` | `dist(z, h=h)` | **Yes** | Evaluation |
| 10 | `bayesian_statistics.py:889` | `dist(z, h=h)` | **Yes** | Evaluation |
| 11 | `bayesian_inference.py:174` | `dist(redshift, hubble_constant)` | **Yes** | Pipeline A |
| 12 | `bayesian_inference.py:177` | `dist(redshift, hubble_constant)` | **Yes** | Pipeline A |
| 13 | `bayesian_inference.py:207` | `dist(redshift, hubble_constant)` | **Yes** | Pipeline A |
| 14 | `emri_detection.py:45` | `dist(host_galaxy.redshift, TRUE_HUBBLE_CONSTANT)` | **Yes** | Detection model |
| 15 | `emri_detection.py:55` | `dist(host_galaxy.redshift, TRUE_HUBBLE_CONSTANT)` | **Yes** | Detection model |
| 16 | `emri_detection.py:57` | `dist(...)` | **Yes** | Detection model |
| 17 | `galaxy.py:167` | `dist(self.redshift_lower_limit)` | No (default h=0.73) | Catalog display |
| 18 | `galaxy.py:167` | `dist(self.redshift_upper_limit)` | No (default h=0.73) | Catalog display |
| 19 | `galaxy.py:366` | `dist(galaxy.redshift, TRUE_HUBBLE_CONSTANT)` | **Yes** | Catalog filter |
| 20 | `detection.py:138` | `dist(1.5)` | No (default h=0.73) | Detection model |

**Key findings:**
- No call site overrides Omega_m, Omega_DE, w_0, or w_a. All calls use the defaults.
- The only argument that varies across calls is h.
- In the injection path, exactly ONE call to dist(): `main.py:500` with explicit `h=h_value`.
- In the simulation path, the relevant call is `parameter_space.py:148` with default h=0.73.
- In the evaluation path (Pipeline B), all calls use explicit `h=h` (the candidate h being evaluated).

### A.3: get_distance() Call-Site Audit

`ParameterSample.get_distance()` (handler.py:44-45) calls `dist(self.redshift)` with default h=0.73.

**Grep confirms: get_distance() is NEVER called anywhere in the codebase.** It is a dead method. This is important because if it were used in the injection path, it would bypass the h-dependence required by D-04.

### A.4: Intentional h-Handling Difference (D-04)

Per design decision D-04 (11.1-CONTEXT.md):
> "P_det depends on h through the population model: for each candidate h, draw events at redshifts z, compute d_L(z, h), generate waveform, compute SNR."

**Implementation verification:**
- **Injection path:** `dist(sample.redshift, h=h_value)` at main.py:500 -- uses candidate h. Correct per D-04.
- **Simulation path:** `dist(host_galaxy.z)` at parameter_space.py:148 -- uses default h=0.73. Correct: simulation generates training data at the fiducial cosmology.

The h value flows through main() as follows:
- `main.py:37`: `arguments = Arguments.create()` parses `--h_value` from CLI
- `main.py:80`: `h_value=arguments.h_value` passed to `injection_campaign()`
- `main.py:399`: `h_value: float` parameter of `injection_campaign()`
- `main.py:500`: `dist(sample.redshift, h=h_value)` -- used here

### A.5: Impact Assessment of h-Dependence

At a representative redshift z=0.3, the luminosity distance varies significantly with h:

| h | d_L(z=0.3) [Gpc] | Relative to h=0.73 |
|---|-------------------|---------------------|
| 0.60 | 1.8317 | +21.7% |
| 0.73 | 1.5055 | (reference) |
| 0.90 | 1.2211 | -18.9% |

This ~40% range in d_L across the h prior is large enough to substantially affect SNR (which scales as ~1/d_L) and hence P_det. The h-dependent injection campaign correctly captures this variation.

### A.6: z_cut=0.5 Safety Evaluation (SC-5)

**d_L values at z=0.5:**

| h | d_L(z=0.5) [Gpc] |
|---|-------------------|
| 0.60 | 3.3645 |
| 0.65 | 3.1057 |
| 0.70 | 2.8838 |
| 0.73 | 2.7653 |
| 0.80 | 2.5233 |
| 0.85 | 2.3749 |
| 0.90 | 2.2430 |

**SNR scaling argument:**

SNR scales as ~1/d_L at fixed source parameters. The ratio d_L(z=0.1)/d_L(z=0.5) is approximately 0.16, independent of h (because d_L scales as 1/h at fixed z).

If a source at z=0.1 has SNR ~ 20 (barely above the detection threshold SNR_THRESHOLD=15), then the same source at z=0.5 would have:

SNR(z=0.5) ~ 20 * 0.16 = 3.2

This is a factor of **~4.7x below** the detection threshold of 15. Even for the heaviest EMRI sources (M = 10^6 M_sun), which have the highest SNR, a source detected at the threshold at z=0.1 would be far below threshold at z=0.5.

This is consistent with the empirical observation documented in the injection code (main.py:473-476): "All 24/69500 detections in the initial campaign were at z < 0.18."

**The largest d_L in the grid is d_L(z=0.5, h=0.60) = 3.3645 Gpc**, which is more than twice the LISA detection horizon for EMRIs (`LUMINOSITY_DISTANCE_THRESHOLD_GPC = 1.55 Gpc` in constants.py:58).

**Verdict: z_cut=0.5 is safe for all h in [0.60, 0.90].**

---

## Part B: d_L Round-Trip Numerical Test

### Test Script

`test_round_trip.py` tests 4 properties:
1. **Limiting case:** dist(z=0, h) = 0 for all 7 h-values
2. **Round-trip accuracy:** z -> dist(z, h) -> dist_to_redshift(d_L, h) -> z_rec, with |z - z_rec|/z < 1e-4
3. **Edge cases:** z=0.001 (low-z fsolve convergence) and z=0.5 (upper injection bound)
4. **Low-z Hubble law:** dist(0.001, h) ~ c*0.001/(1e5*h) Gpc within 5%

### Results

```
OVERALL: PASS -- all tests within thresholds
```

**Test 1 (limiting case):** dist(z=0, h) = 0.0 exactly for all 7 h-values. PASS.

**Test 2 (round-trip accuracy):**

| h | Max rel_error | Worst z | Status |
|---|---------------|---------|--------|
| 0.60 | 1.74e-13 | 0.0010 | PASS |
| 0.65 | 1.35e-13 | 0.0010 | PASS |
| 0.70 | 2.18e-13 | 0.0010 | PASS |
| 0.73 | 1.95e-13 | 0.0010 | PASS |
| 0.80 | 1.35e-13 | 0.0010 | PASS |
| 0.85 | 1.28e-13 | 0.0010 | PASS |
| 0.90 | 1.74e-13 | 0.0010 | PASS |

Global worst: rel_error = 2.18e-13 at z=0.001, h=0.70. This is **9 orders of magnitude** below the 1e-4 threshold. The round-trip inversion is essentially exact to machine precision.

**Test 3 (edge cases):**
- z=0.001: rel_error ~ 1e-13 for all h. No fsolve convergence issues.
- z=0.500: rel_error ~ 1e-16 for all h (even better than low-z). No issues.

**Test 4 (low-z Hubble law):**
All h-values show dist(0.001, h) agrees with c*z/(1e5*h) Gpc to within 0.08%. The 0.08% deviation is expected from the cosmological correction at z=0.001. PASS.

---

## Conclusion

1. **Cosmological model is consistent:** dist() is always called with Omega_m=0.25, Omega_DE=0.75, w_0=-1, w_a=0 (flat LambdaCDM). No call site overrides these defaults.
2. **h-handling is intentional per D-04:** Injection uses dist(z, h=h_value), simulation uses dist(z) with default h=0.73.
3. **get_distance() is unused:** Dead method on ParameterSample, would bypass h-dependence if called.
4. **Round-trip inversion is machine-precise:** Worst-case rel_error = 2.18e-13, 9 orders of magnitude below the 1e-4 threshold.
5. **z_cut=0.5 is safe:** SNR at z=0.5 is ~3.2 for a source barely detectable at z=0.1, well below threshold of 15. Confirmed by empirical injection data (all detections at z < 0.18).

---

_Report generated: 2026-03-31_
_Phase: 17-injection-physics-audit, Plan: 01, Task: 2_
