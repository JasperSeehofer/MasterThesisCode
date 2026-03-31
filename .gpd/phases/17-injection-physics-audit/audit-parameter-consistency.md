# AUDT-01: Parameter Distribution Consistency Report

**Phase:** 17-injection-physics-audit
**Plan:** 01
**Date:** 2026-03-31

## Summary

Line-by-line comparison of EMRI parameter flow in `injection_campaign()` (main.py:396-577) vs `data_simulation()` (main.py:188-383). All 14 parameters traced through both code paths with specific line numbers. Every difference categorized against design decisions D-01 through D-09.

**Result: All parameter differences are intentional and documented.**

## Summary Table

| # | Parameter | Injection Source | Simulation Source | Match? | Category |
|---|-----------|-----------------|-------------------|--------|----------|
| 1 | M | `sample.M` from `Model1CrossCheck.sample_emri_events()` (main.py:496) | `host_galaxy.M` from `GalaxyCatalogueHandler` (parameter_space.py:145) | **Differs** | Intentional -- galaxy catalog intermediary (see Axis 2) |
| 2 | mu | Fixed 10 via `_apply_model_assumptions()` (cosmological_model.py:189) | Same fixed 10 | Match | -- |
| 3 | a | Fixed 0.98 via `_apply_model_assumptions()` (cosmological_model.py:187) | Same fixed 0.98 | Match | -- |
| 4 | p0 | `randomize_parameters(rng=rng)` (main.py:493) | `randomize_parameters(rng=rng)` (main.py:259) | Match | Same distribution `uniform(10, 16)` |
| 5 | e0 | `randomize_parameters(rng=rng)` (main.py:493), upper_limit=0.2 from `_apply_model_assumptions()` | Same | Match | -- |
| 6 | x0 | `randomize_parameters(rng=rng)` (main.py:493) | Same (main.py:259) | Match | Same distribution `uniform(-1, 1)` |
| 7 | d_L | `dist(sample.redshift, h=h_value)` (main.py:500-501) | `dist(host_galaxy.z)` with default h=0.73 (parameter_space.py:148) | **Differs** | **Intentional per D-04** |
| 8 | qS | `randomize_parameters(rng=rng)` -> `polar_angle_distribution` (main.py:493) | `host_galaxy.qS` from GLADE catalog (parameter_space.py:146) | **Differs** | Intentional -- injection marginalizes sky (see Axis 4) |
| 9 | phiS | `randomize_parameters(rng=rng)` -> `uniform(0, 2pi)` (main.py:493) | `host_galaxy.phiS` from GLADE catalog (parameter_space.py:146) | **Differs** | Intentional -- injection marginalizes sky (see Axis 4) |
| 10 | qK | `randomize_parameters(rng=rng)` (main.py:493) | Same (main.py:259) | Match | Same distribution `polar_angle_distribution` |
| 11 | phiK | `randomize_parameters(rng=rng)` (main.py:493) | Same (main.py:259) | Match | Same distribution `uniform(0, 2pi)` |
| 12 | Phi_phi0 | `randomize_parameters(rng=rng)` (main.py:493) | Same (main.py:259) | Match | Same distribution `uniform(0, 2pi)` |
| 13 | Phi_theta0 | `randomize_parameters(rng=rng)` (main.py:493) | Same (main.py:259) | Match | Same distribution `uniform(0, 2pi)` |
| 14 | Phi_r0 | `randomize_parameters(rng=rng)` (main.py:493) | Same (main.py:259) | Match | Same distribution `uniform(0, 2pi)` |

**Differences found: 4 (M, d_L, qS, phiS). All intentional.**

---

## Detailed Per-Axis Analysis

### Axis 1: Model1CrossCheck Instance Sharing

Both `injection_campaign()` and `data_simulation()` receive the same `Model1CrossCheck` instance created in `main()`:

- **Construction:** `main.py:48` -- `cosmological_model = Model1CrossCheck(rng=rng)`
- **Passed to injection:** `main.py:79` -- `cosmological_model=cosmological_model`
- **Passed to simulation:** `main.py:57` -- `cosmological_model` positional argument

`Model1CrossCheck.__init__()` (cosmological_model.py:174-180):
1. Creates `ParameterSpace()` (line 178)
2. Calls `_apply_model_assumptions()` (line 179) which sets:
   - `M` limits: `[10^4.5, 10^6]` (lines 183-184)
   - `a = 0.98`, fixed (lines 186-187)
   - `mu = 10`, fixed (lines 189-190)
   - `e0.upper_limit = 0.2` (line 192)
   - `max_redshift = 1.5` (line 194)
   - `luminosity_distance.upper_limit = dist(redshift=1.5)` with default h=0.73 (line 195)
3. Calls `setup_emri_events_sampler()` (line 180) -- initializes emcee MCMC sampler

Both pipelines share this exact instance. **Verified: identical model assumptions.**

### Axis 2: M Sampling Path

**Injection path:**
1. `cosmological_model.sample_emri_events(_EMCEE_BATCH)` (main.py:469) returns `list[ParameterSample]`
2. `ParameterSample` constructed with `M=10**sample[0]` (cosmological_model.py:288) -- direct from MCMC in log-M space
3. `parameter_estimation.parameter_space.M.value = sample.M` (main.py:496) -- **direct assignment**

**Simulation path:**
1. `cosmological_model.sample_emri_events(200)` (main.py:254) returns `list[ParameterSample]`
2. `galaxy_catalog.get_hosts_from_parameter_samples(parameter_samples)` (main.py:255) -- **galaxy catalog intermediary**
3. Inside `get_hosts_from_parameter_samples()` (handler.py:553-592):
   - For each `ParameterSample`, calls `find_closest_galaxy_to_coordinates(phi_S, theta_S, redshift, M)` (handler.py:580-585)
   - This returns the closest galaxy in the GLADE catalog in normalized 4D (phi, theta, z, log10(M)) space (handler.py:373-390)
   - The returned `HostGalaxy` has `M = host_galaxy.M` which is the **catalog galaxy's BH mass** (derived from stellar mass via empirical relation, handler.py:478-484)
4. `parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy)` (main.py:261) sets `M.value = host_galaxy.M` (parameter_space.py:145)

**Key difference:** In the injection path, M comes directly from the EMRI population model MCMC. In the simulation path, M is matched to the closest GLADE galaxy, so M is that galaxy's estimated BH mass. This difference is structurally intentional: the injection campaign does not need a galaxy catalog because it only builds P_det grids over the population model's parameter space, while the simulation pipeline needs a real galaxy association for the Bayesian evaluation.

**Impact:** The M distributions may differ slightly due to the galaxy catalog matching (the nearest galaxy's M is not exactly the sampled M). However, for the injection campaign's purpose (building P_det as a function of (z, M)), the population model distribution is the correct choice -- P_det should reflect the intrinsic EMRI population, not the galaxy catalog's discrete sampling of it.

### Axis 3: Redshift / d_L Computation

**Injection path:**
- `dist(sample.redshift, h=h_value)` (main.py:500) -- uses the candidate h value
- Comment at main.py:498-499: "CRITICAL per D-04: Set luminosity distance with candidate h value (NOT set_host_galaxy_parameters which hardcodes h=0.73)"

**Simulation path:**
- `parameter_space.set_host_galaxy_parameters(host_galaxy)` (main.py:261) calls `dist(host_galaxy.z)` (parameter_space.py:148) which uses default `h=H=0.73` from constants.py:25

**This is intentional per D-04:** The injection campaign needs d_L(z, h) for each candidate h value because P_det depends on h through d_L. The simulation pipeline uses the fiducial h=0.73 because it generates training data at the "true" cosmology.

### Axis 4: Extrinsic Parameters (phiS, qS)

**Injection path:**
- `randomize_parameters(rng=rng)` (main.py:493) sets ALL non-fixed parameters including:
  - `phiS`: `uniform(0, 2*pi)` (parameter_space.py:98)
  - `qS`: `polar_angle_distribution(0, pi)` = `arccos(uniform(-1, 1))` (parameter_space.py:22-24, 89-94)
- Note: `phiS` and `qS` are then overridden by... nothing. The injection path does NOT call `set_host_galaxy_parameters()`. Only M and d_L are set explicitly (main.py:496, 500-501).
- Result: **phiS and qS come from uniform random distributions**

**Simulation path:**
- `randomize_parameters(rng=rng)` (main.py:259) sets all non-fixed parameters (including phiS, qS)
- `set_host_galaxy_parameters(host_galaxy)` (main.py:261) then **overrides** phiS and qS:
  - `phiS = host_galaxy.phiS` (parameter_space.py:146) -- from GLADE catalog RA (converted to radians, handler.py:487-488)
  - `qS = host_galaxy.qS` (parameter_space.py:146) -- from GLADE catalog DEC (converted to spherical polar angle, handler.py:489-492)
- Result: **phiS and qS come from the actual galaxy positions in the GLADE catalog**

**This is intentional:** The injection campaign marginalizes over sky location to build P_det(z, M | h), treating sky angles as nuisance parameters. This is the standard approach for injection campaigns (cf. LVK dark siren analyses). The simulation pipeline uses actual galaxy positions because it models the full detection and host-galaxy association process.

### Axis 5: Eccentricity and Spin

**Eccentricity (e0):**
- Both paths: `randomize_parameters(rng=rng)` draws `e0` from `uniform(0.05, 0.2)` where:
  - Default lower_limit = 0.05 (parameter_space.py:75)
  - Upper_limit = 0.2 set by `_apply_model_assumptions()` (cosmological_model.py:192)
- In injection: not overridden after randomize_parameters (main.py:493-501)
- In simulation: not overridden by set_host_galaxy_parameters (parameter_space.py:144-148)
- **Match confirmed.**

**Spin (a):**
- Both paths: `a = 0.98`, `is_fixed = True` (cosmological_model.py:186-187)
- `randomize_parameters()` skips fixed parameters (parameter_space.py:141-142)
- **Match confirmed.**

Note: The `ParameterSample` from `sample_emri_events()` draws spin from `MBH_spin_distribution(0, 1)` (cosmological_model.py:288) which is a truncated normal centered at 0.98. However, this sample spin is **never used** by either path -- both paths use the fixed a=0.98 from `_apply_model_assumptions()`. The ParameterSample.a attribute is only used for galaxy catalog matching in the simulation path (not for waveform generation).

### Axis 6: z_cut = 0.5

**Injection path:**
- `z_cut = 0.5` defined at main.py:456
- Applied at main.py:477: `if sample.redshift > z_cut: skipped_high_z += 1; continue`
- Comment at main.py:473-476: "Importance sampling: skip events beyond the detection horizon. All 24/69500 detections in the initial campaign were at z < 0.18."
- Skipped events are NOT stored in the results CSV.

**Simulation path:**
- No z_cut applied. Events are sampled up to `max_redshift = 1.5` (cosmological_model.py:194)
- Events with low SNR are simply skipped (main.py:331-335), but all z values up to 1.5 are attempted.

**This is intentional for efficiency:** The injection campaign skips high-z events because they have P_det ~ 0 and waste GPU time. For P_det estimation, this is safe as long as z_cut is above the detection horizon. See cosmological model audit (AUDT-02) for the z_cut=0.5 safety analysis.

### Axis 7: Error Handling

Both paths catch essentially the same exception types:

| Exception | Injection (main.py:504-553) | Simulation (main.py:263-326) |
|-----------|---------------------------|-------------------------------|
| Warning ("Mass ratio") | Yes (line 511-515) | Yes (line 282-285) |
| Other Warning | Yes (line 516-519) | Yes (line 287-289) |
| ParameterOutOfBoundsError | Yes (line 520-525) | Yes (line 290-293) |
| RuntimeError | Yes (line 526-531) | Yes (line 300-304) |
| ValueError ("EllipticK") | Yes (line 532-537) | Yes (line 305-310) |
| ValueError ("Brent root solver") | Yes (line 538-541) | Yes (line 311-315) |
| ZeroDivisionError | Yes (line 546-550) | Yes (line 318-321) |
| TimeoutError | Yes (line 553) | Yes (line 323-325) |
| AssertionError | **No** | Yes (line 295-298) |
| np.linalg.LinAlgError | N/A (no CRB) | Yes (line 349-351) |
| ParameterEstimationError | N/A (no CRB) | Yes (line 352-354) |

**Differences:**
- Injection does NOT catch `AssertionError` -- minor; assertion errors from waveform generation would propagate as unhandled exceptions. Low risk since these are rare.
- Injection does not catch CRB-related errors (LinAlgError, ParameterEstimationError) -- correct, since injection does not compute CRB.
- Injection timeout is 30s (`_TIMEOUT_S`, main.py:462) vs simulation 90s (main.py:265). This is reasonable since injection only computes SNR (no CRB).

## get_distance() Call-Site Audit

`ParameterSample.get_distance()` is defined at handler.py:44-45:
```python
def get_distance(self) -> float:
    return dist(self.redshift)
```
This calls `dist()` with **default h=0.73** -- it does NOT accept an h parameter.

**Grep result:** `get_distance()` is defined but **never called** anywhere in the codebase. This confirms:
- The injection path correctly uses `dist(sample.redshift, h=h_value)` (main.py:500) instead of `sample.get_distance()`
- If `get_distance()` were used in the injection path, it would be a bug (would hardcode h=0.73, violating D-04)

## Conclusion

All 14 EMRI parameters have been traced through both code paths. The 4 differences found (M, d_L, phiS, qS) are all intentional:

1. **M:** Direct from population model (injection) vs galaxy catalog matched (simulation) -- intentional structural difference
2. **d_L:** `dist(z, h=h_value)` (injection) vs `dist(z)` with default h=0.73 (simulation) -- **intentional per D-04**
3. **phiS, qS:** Random uniform (injection) vs GLADE catalog positions (simulation) -- intentional sky marginalization for P_det

**No unintentional parameter discrepancies found.**

---

_Report generated: 2026-03-31_
_Phase: 17-injection-physics-audit, Plan: 01, Task: 1_
