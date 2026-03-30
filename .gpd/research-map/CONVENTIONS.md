# Derivation Quality Analysis

**Analysis Date:** 2026-03-30

## Unit System

**Primary unit system: SI-adjacent astrophysical units (NOT natural units).**

All computations use explicit dimensional constants. There is no hbar = c = 1 convention anywhere in the codebase.

| Quantity | Unit | Source |
|----------|------|--------|
| Speed of light `C` | m/s (299792458.0) | `master_thesis_code/constants.py` (line 19) |
| Gravitational constant `G` | Gpc^3 / (s^2 M_sun) | `master_thesis_code/constants.py` (line 20) |
| Hubble parameter `H` | dimensionless h = H_0 / (100 km/s/Mpc) | `master_thesis_code/constants.py` (line 25) |
| Hubble constant `H0` | m / (s * Mpc) = 73e3 | `master_thesis_code/constants.py` (line 22) |
| Luminosity distance | Gpc | `master_thesis_code/physical_relations.py` (line 36, return value) |
| Mass (MBH, CO) | solar masses M_sun | `master_thesis_code/datamodels/parameter_space.py` (lines 48, 58) |
| Angular quantities | radians | `master_thesis_code/datamodels/parameter_space.py` (lines 87-129) |
| Time | seconds | `master_thesis_code/constants.py` (line 70, YEAR_IN_SEC) |
| Frequency | Hz | `master_thesis_code/LISA_configuration.py` (PSD functions) |
| PSD | Hz^{-1} (one-sided strain PSD) | `master_thesis_code/LISA_configuration.py` (line 97) |
| LISA arm length | meters (2.5e9) | `master_thesis_code/constants.py` (line 69) |
| Speed of light (approx.) `SPEED_OF_LIGHT_KM_S` | km/s (300000.0) | `master_thesis_code/constants.py` (line 21) |

**Unit conversion constants:**
- `GPC_TO_MPC = 1e3` (1 Gpc = 1000 Mpc) -- `constants.py` (line 38)
- `KM_TO_M = 1e3` (1 km = 1000 m) -- `constants.py` (line 39)
- `RADIAN_TO_DEGREE = 360 / (2*pi)` -- `constants.py` (line 37)
- `M_IN_GPC = 3.2407788498994e-26` (m / Gpc) -- `constants.py` (line 18)

**Hubble constant construction pattern:**
Throughout the codebase, H_0 in physical units is reconstructed from the dimensionless h:
```python
H_0 = h * 100.0 * KM_TO_M / GPC_TO_MPC ** (-1)  # m/(s*Gpc)
```
This appears in `physical_relations.py` (lines 69, 108, 145, 185). Note: `GPC_TO_MPC ** (-1)` is `1e-3`, so `H_0 = h * 100 * 1e3 * 1e-3 = h * 100` in m/(s*Gpc). Use this pattern for all new distance computations.

**Two distinct c values exist:**
1. `C = 299792458.0` (exact SI, m/s) -- used in PSD, inner products, distance functions
2. `SPEED_OF_LIGHT_KM_S = 300000.0` (approx, km/s) -- used in `datamodels/galaxy.py` comoving volume element (line 126) and Pipeline A

These MUST NOT be mixed within a single formula. The approximate value introduces ~0.07% error.

## Notation Table

### EMRI Parameter Space (14 parameters)

| Symbol (code) | Physics meaning | Unit | Distribution | Bounds | File |
|----------------|----------------|------|--------------|--------|------|
| `M` | MBH mass | M_sun | log-uniform | [1e4, 1e7] | `parameter_space.py` (line 47) |
| `mu` | CO mass | M_sun | uniform | [1, 100] | `parameter_space.py` (line 57) |
| `a` | MBH dimensionless spin | dimensionless | uniform | [0, 1] | `parameter_space.py` (line 62) |
| `p0` | Semi-latus rectum | meters | uniform | [10, 16] | `parameter_space.py` (line 67) |
| `e0` | Eccentricity | dimensionless | uniform | [0.05, 0.7] | `parameter_space.py` (line 72) |
| `x0` | cos(inclination) = x_I0 | dimensionless | uniform | [-1, 1] | `parameter_space.py` (line 77) |
| `luminosity_distance` | d_L | Gpc | (set by host galaxy) | [0, 7] | `parameter_space.py` (line 82) |
| `qS` | Sky polar angle (ecliptic) | rad | arccos(uniform(-1,1)) | [0, pi] | `parameter_space.py` (line 87) |
| `phiS` | Sky azimuthal angle (ecliptic) | rad | uniform | [0, 2*pi] | `parameter_space.py` (line 94) |
| `qK` | BH spin polar angle (ecliptic) | rad | arccos(uniform(-1,1)) | [0, pi] | `parameter_space.py` (line 100) |
| `phiK` | BH spin azimuthal angle (ecliptic) | rad | uniform | [0, 2*pi] | `parameter_space.py` (line 107) |
| `Phi_phi0` | Initial azimuthal phase | rad | uniform | [0, 2*pi] | `parameter_space.py` (line 113) |
| `Phi_theta0` | Initial polar phase | rad | uniform | [0, 2*pi] | `parameter_space.py` (line 118) |
| `Phi_r0` | Initial radial phase | rad | uniform | [0, 2*pi] | `parameter_space.py` (line 123) |

**Derivative step sizes:** Each parameter has a `derivative_epsilon = 1e-6` (default). This is used for both the forward-difference and 5-point stencil derivatives.

### Cramér-Rao Bounds CSV Column Naming

The Fisher matrix inverse (covariance) entries are stored with the naming convention:
```
delta_{param_i}_delta_{param_j}
```
For example: `delta_luminosity_distance_delta_luminosity_distance` (variance), `delta_phiS_delta_qS` (covariance).

Only the lower-triangular entries are stored (row >= column). See `parameter_estimation.py` (lines 413-419).

### Detection Dataclass Field Mapping

The `Detection` class (`datamodels/detection.py`) remaps CSV columns to physics symbols:

| CSV column | Detection field | Unit |
|------------|----------------|------|
| `luminosity_distance` | `d_L` | Gpc |
| sqrt(`delta_luminosity_distance_delta_luminosity_distance`) | `d_L_uncertainty` | Gpc |
| `phiS` | `phi` | rad |
| `qS` | `theta` | rad |
| `M` | `M` (redshifted) | M_sun |
| `SNR` | `snr` | dimensionless |

Note the rename: `qS` -> `theta`, `phiS` -> `phi`. This remapping only occurs in the Detection class. All other code uses `qS`/`phiS` directly.

## Cosmological Parameters (Fiducial)

| Parameter | Symbol | Value | File | Note |
|-----------|--------|-------|------|------|
| Matter density | Omega_m | 0.25 | `constants.py` (line 29) | WMAP-era; Planck 2018 is 0.3153 |
| Dark energy density | Omega_DE | 0.75 | `constants.py` (line 30) | Flat universe: Omega_m + Omega_DE = 1 |
| Dark energy EoS | w_0 | -1.0 | `constants.py` (line 31) | LCDM fiducial |
| DE EoS evolution | w_a | 0.0 | `constants.py` (line 32) | LCDM fiducial |
| Hubble (simulation) | H | 0.73 | `constants.py` (line 25) | Dimensionless h for simulation pipeline |
| Hubble (inference true) | TRUE_HUBBLE_CONSTANT | 0.70 | `constants.py` (line 26) | Dimensionless h for Bayesian inference |
| SNR threshold | SNR_THRESHOLD | 15 | `constants.py` (line 48) | Detection criterion |

**Two distinct Hubble values coexist:** `H = 0.73` (simulation default) and `TRUE_HUBBLE_CONSTANT = 0.70` (inference truth). They serve different roles: H parameterizes the fiducial cosmology for distance calculations; TRUE_HUBBLE_CONSTANT is the "true" value the Bayesian inference pipeline attempts to recover.

## Approximations Made

**Equal-arm-length LISA approximation:**
- What is neglected: Orbital motion-induced arm length variations (breathing modes)
- Justification quality: Adequate (standard in LISA literature for signal detection studies)
- Parameter controlling approximation: arm-length fractional variation ~1e-4
- File: `master_thesis_code/LISA_configuration.py` (line 54, class docstring)

**LCDM hardcoded in `dist()` despite wCDM signature:**
- What is neglected: w_0, w_a parameters are accepted but silently ignored in the analytic hypergeometric formula `lambda_cdm_analytic_distance()`
- Justification quality: Missing (no warning or documentation that wCDM is not implemented)
- Impact: Any call to `dist()` with non-LCDM w_0, w_a returns the LCDM result
- File: `master_thesis_code/physical_relations.py` (lines 72, 240-268)

**Lagrangian interpolation order 35 in LISA response:**
- What is neglected: Higher-order interpolation errors in the TDI response
- Justification quality: Adequate (standard choice in fastlisaresponse)
- File: `master_thesis_code/waveform_generator.py` (line 23)

**Speed of light approximation (300,000 km/s):**
- What is neglected: 0.07% of c
- Justification quality: Weak (no comment on why the approximation is used)
- Impact: Only affects `galaxy.py` comoving volume element and Pipeline A
- File: `master_thesis_code/constants.py` (line 21)

**Galaxy redshift uncertainty scaling:**
- Formula: `0.013 * (1+z)^3`, capped at 0.015
- Justification quality: Missing (no reference provided; standard spectroscopic errors scale as (1+z) not (1+z)^3)
- File: `master_thesis_code/datamodels/galaxy.py` (line 67)
- Constants: `master_thesis_code/constants.py` (line 51)

**Hardcoded 10% fractional d_L error (Pipeline A only):**
- What is neglected: Per-source Fisher-matrix based d_L uncertainty
- Justification quality: Adequate for cross-check pipeline; inappropriate for production
- File: `master_thesis_code/bayesian_inference/bayesian_inference.py` (line 31), `constants.py` (line 52)

## Assumptions Catalog

**Explicit Assumptions:**
- Flat universe (Omega_m + Omega_DE = 1): Used in all distance calculations
  - File: `master_thesis_code/constants.py` (lines 29-30)
  - Verified in test: `master_thesis_code_test/test_constants.py::test_flat_universe`
- Ecliptic coordinate system for sky angles: qS, phiS, qK, phiK are ecliptic coordinates
  - File: `master_thesis_code/datamodels/parameter_space.py` (lines 87-114)
- LISA observes for T = 5 years (default) or 1 year (SNR pre-check)
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 71-72)

**Implicit Assumptions:**
- No spin precession effects beyond what `few` Pn5AAKWaveform provides
  - File: `master_thesis_code/waveform_generator.py` (line 96)
  - Risk: Spin precession may affect Fisher matrix accuracy for high-spin EMRIs
- Fisher matrix = Gaussian posterior approximation
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (line 350)
  - Risk: Non-Gaussian tails in posterior are lost; may underestimate parameter uncertainties
- `allow_singular=True` in multivariate normal construction (Pipeline B)
  - File: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (lines 222, 227)
  - Risk: Singular covariance matrices indicate degenerate parameters; likelihoods may be unreliable

## Sign and Factor Conventions

**Fourier transform convention:**
The project uses NumPy/CuPy `rfft` which implements:
```
X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
```
This is the physics convention with -i in the exponent. Consistent throughout.
- File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 318-321)

**Inner product convention:**
```
<h1|h2> = 4 Re sum_{channels} integral_{f_min}^{f_max} h1_tilde(f) * conj(h2_tilde(f)) / S_n(f) df
```
The factor of 4 (not 2) corresponds to the one-sided PSD convention. Consistent with standard GW literature (Cutler & Flanagan 1994).
- File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 286-299, implementation at line 332)

**SNR convention:**
```
SNR = sqrt(<h|h>)
```
- File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (line 433)

**Sky localization error:**
```
Delta_Omega = 2*pi * |sin(theta)| * sqrt(sigma_phi^2 * sigma_theta^2 - C_{theta,phi}^2)
```
- File: `master_thesis_code/datamodels/detection.py` (lines 20-27)

## Notation Consistency

**Consistent Usage:**
- `d_L` / `luminosity_distance`: Always luminosity distance in Gpc. No confusion with comoving distance.
- `M`: Always the central MBH mass in solar masses. Redshifted mass is `M_z = M * (1+z)`.
- `z` / `redshift`: Always cosmological redshift (dimensionless).

**Conflicts / Ambiguity:**
- `M` in `Detection` dataclass is the REDSHIFTED mass M_z (CSV stores M_z from the simulation), while `M` in `ParameterSpace` and `Galaxy` is the TRUE (rest-frame) mass. The `Detection` docstring (line 82) says "redshifted" but the field name is just `M`.
  - Files: `datamodels/detection.py` (line 81), `datamodels/parameter_space.py` (line 47), `datamodels/galaxy.py` (line 42)
- `dist` is used as both a function name (`physical_relations.dist()`) and occasionally as a local variable name in test helper functions (e.g., `cosmological_model_test.py::_make_detection_series` parameter name `dist`). This shadows the import in test files.
  - Files: `master_thesis_code_test/cosmological_model_test.py` (line 13), `master_thesis_code_test/datamodels/test_detection.py` (line 11)
- `theta`/`qS` and `phi`/`phiS` rename in `Detection` class may cause confusion when comparing with other code that uses `qS`/`phiS` directly.

## Dimensional Analysis

**Luminosity distance formula:**
```
d_L(z) = (C / H_0) * (1 + z) * integral_0^z dz'/E(z')
```
- `[C]` = m/s
- `[H_0]` = m/(s*Gpc) (after unit conversion)
- `[C/H_0]` = Gpc -- dimensionless integral * (1+z) = dimensionless
- Result: Gpc -- verified
- File: `master_thesis_code/physical_relations.py` (lines 69-75)

**PSD S_OMS:**
```
S_OMS(f) = 15^2 * 1e-24 * (1 + (2e-3/f)^4) * (2*pi*f/c)^2
```
- `[15^2 * 1e-24]` = m^2/Hz (position noise)
- `[(2*pi*f/c)^2]` = Hz^2 / (m/s)^2 = s^2/m^2
- Result: m^2/Hz * s^2/m^2 = s^2/(Hz*m^2) -- this is strain^2/Hz after accounting for the arm response
- Dimensional consistency: Plausible (standard LISA parameterization)
- File: `master_thesis_code/LISA_configuration.py` (line 162)

**Comoving volume element:**
```
dV_c/dz = 4*pi * (c/H_0)^3 * I(z)^2 / E(z)
```
where c is in km/s, H_0 = h0 * 100 in km/s/Mpc (from SPEED_OF_LIGHT_KM_S / h0).
- `[(c/H_0)]` = Mpc
- `[(c/H_0)^3]` = Mpc^3
- `[I(z)^2 / E(z)]` = dimensionless
- Result: Mpc^3 -- verified (standard comoving volume element)
- File: `master_thesis_code/datamodels/galaxy.py` (line 126)

---

*Derivation quality analysis: 2026-03-30*
