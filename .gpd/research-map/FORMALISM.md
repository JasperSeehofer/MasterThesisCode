# Theoretical Frameworks

**Analysis Date:** 2026-03-30

## Physical System

**Subject:** Gravitational wave cosmology -- Hubble constant inference from LISA Extreme Mass Ratio Inspiral (EMRI) detections using the "dark siren" statistical method.

**Scales:**

- Energy: Not directly relevant (classical GR + statistical inference); EMRI systems involve compact objects with M ~ 10^4--10^7 M_sun (MBH) and mu ~ 1--100 M_sun (CO)
- Length: LISA arm length L = 2.5e9 m; luminosity distances d_L ~ 0--1.55 Gpc; gravitational wavelengths in the mHz band
- Time: LISA observation time T = 0.5--5 years; time step dt = 10 s (YEAR_IN_SEC / LISA_STEPS = 3.15576e7 / 10000)
- Dimensionless parameters: h = H_0 / (100 km/s/Mpc) in [0.6, 0.86]; Omega_m = 0.25; eccentricity e0 in [0.05, 0.7]; MBH spin a in [0, 0.998]

**Degrees of Freedom:**

- 14-parameter EMRI waveform: {M, mu, a, p0, e0, x0, d_L, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0}
  - Defined in: `master_thesis_code/datamodels/parameter_space.py` (lines 42--129)
- Cosmological parameters: {h, Omega_m, Omega_DE, w_0, w_a}
  - Defined in: `master_thesis_code/constants.py` (lines 25--32)

## Theoretical Framework

**Primary Framework:**

- General relativistic waveform generation for EMRIs in Kerr spacetime
- Formulation: Post-Newtonian augmented Analytic Kludge (Pn5AAK) waveforms via the `few` package, processed through LISA TDI response via `fastlisaresponse`
- File: `master_thesis_code/waveform_generator.py` (lines 74--109)

**Secondary/Supporting Frameworks:**

- Flat LCDM / wCDM cosmology for distance-redshift relations
  - File: `master_thesis_code/physical_relations.py` (lines 27--268)
- Bayesian statistical inference for H_0 posterior from dark sirens
  - File: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (Pipeline B, production)
  - File: `master_thesis_code/bayesian_inference/bayesian_inference.py` (Pipeline A, dev cross-check)
- EMRI event rate model "Model M1" from Babak et al. (2017)
  - File: `master_thesis_code/cosmological_model.py` (lines 166--293)

## Fundamental Equations

### Equation Catalog

| ID | Equation | Type | Location | Dimensions | Status | Depends On | Used By |
|----|----------|------|----------|------------|--------|------------|---------|
| EQ-001 | d_L(z) = (c/H_0)(1+z) integral_0^z dz'/E(z') | Defining | `physical_relations.py` (line 43, docstring) | [length] = Gpc | Verified (d_L(0)=0 test exists) | EQ-002 | EQ-003, EQ-004, EQ-005, EQ-006, EQ-010 |
| EQ-002 | E(z) = sqrt(Omega_m(1+z)^3 + Omega_DE(1+z)^{3(1+w_0+w_a)} exp(-3 w_a z/(1+z))) | Defining | `physical_relations.py` (lines 198--237) | dimensionless -- verified | Postulated (wCDM) | -- | EQ-001, EQ-009 |
| EQ-003 | dd_L/dz = (c/H_0)[(1+z)/E(z) + integral_0^z dz'/E(z')] | Derived | `physical_relations.py` (lines 156--195) | [length] = Gpc | Derived from EQ-001 | EQ-001, EQ-002 | EQ-011 |
| EQ-004 | Comoving distance integral via _2F_1 hypergeometric | Derived | `physical_relations.py` (lines 240--268) | dimensionless -- verified | Derived (LCDM only) | EQ-002 | EQ-001 |
| EQ-005 | SNR = sqrt(<h|h>) | Derived | `parameter_estimation.py` (lines 426--435) | dimensionless -- verified | Derived | EQ-006 | EQ-007 |
| EQ-006 | <h1|h2> = 4 Re sum_alpha integral (h1~_alpha h2~*_alpha / S_n^alpha) df | Defining | `parameter_estimation.py` (lines 281--333) | dimensionless -- verified (waveform units cancel with PSD units) | Postulated (standard GW inner product) | EQ-008 | EQ-005, EQ-007 |
| EQ-007 | Gamma_ij = <dh/dtheta_i | dh/dtheta_j> (Fisher information matrix) | Defining | `parameter_estimation.py` (lines 350--379) | [theta_i]^{-1} [theta_j]^{-1} | Postulated | EQ-006 | EQ-010 |
| EQ-008 | S_n^{A,E}(f) = instrumental PSD + S_c(f) confusion noise | Defining | `LISA_configuration.py` (lines 111--137) | [Hz^{-1}] | Postulated (arXiv:2303.15929 + arXiv:1703.09858) | -- | EQ-006 |
| EQ-009 | dV_c/dz = 4 pi (c/H_0)^3 I(z)^2 / E(z) | Derived | `datamodels/galaxy.py` (lines 114--127) | [length^3] = (km/s/Mpc)^{-3} | Verified (Hogg 1999 Eq. 27) | EQ-002 | EQ-012 |
| EQ-010 | p(H_0|{d_i}) propto prod_i [integral L_GW(d_i|z,H_0) p_det(z,H_0) p(z|cat) dz / integral p_det(z,H_0) p(z|cat) dz] | Defining | `bayesian_inference.py` (lines 214--304) | dimensionless -- verified (posterior is probability) | Postulated (dark siren method) | EQ-001, EQ-007, EQ-011, EQ-012 | -- |
| EQ-011 | L_GW(d_L_hat|z,h) = N(d_L_hat; d_L(z,h), f_sigma * d_L(z,h)) | Defining (Pipeline A) | `bayesian_inference.py` (lines 183--212) | [length^{-1}] = Gpc^{-1} | Postulated | EQ-001 | EQ-010 |
| EQ-012 | p_det(z) = (1/2)[1 + erf((D_thr - d_L)/(sqrt(2) sigma_dL))] | Defining (Pipeline A) | `bayesian_inference.py` (lines 150--181) | dimensionless -- verified | Postulated (erf model) | EQ-001 | EQ-010 |
| EQ-013 | M_z = M(1+z) (redshifted mass) | Defining | `physical_relations.py` (lines 330--332) | [mass] = M_sun | Postulated | -- | EQ-010 |
| EQ-014 | 5-point stencil: df/dx = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h | Numerical | `parameter_estimation.py` (lines 189--258) | [waveform units / parameter units] | Verified (O(epsilon^4) accuracy) | -- | EQ-007 |
| EQ-015 | S_c(f) = A f^{-7/3} exp(-(f/f1)^alpha) (1/2)(1 + tanh(-(f-fk)/f2)) | Defining | `LISA_configuration.py` (lines 86--109) | [Hz^{-1}] | Postulated (Cornish & Robson 2017) | -- | EQ-008 |
| EQ-016 | S_OMS(f) = 15^2 * 1e-24 * (1 + (2e-3/f)^4) * (2 pi f / c)^2 | Defining | `LISA_configuration.py` (lines 159--162) | [Hz^{-1}] (strain PSD component) | Postulated | -- | EQ-008 |
| EQ-017 | S_TM(f) = 9e-30 * (1 + (0.4e-3/f)^2) * (1 + (f/8e-3)^4) * (1/(2 pi f c))^2 | Defining | `LISA_configuration.py` (lines 164--172) | [Hz^{-1}] (strain PSD component) | Postulated | -- | EQ-008 |
| EQ-018 | dN/dz(M,z) = polynomial_fit(z) * R_emri(M) | Defining | `cosmological_model.py` (lines 198--239) | [yr^{-1}] (EMRI merger rate) | Postulated (from Babak et al. 2017 M1 model) | -- | EQ-010 |
| EQ-019 | DeltaOmega = 2 pi |sin(theta)| sqrt(sigma_phi^2 sigma_theta^2 - C_{theta,phi}^2) | Derived | `datamodels/detection.py` (lines 12--38) | [sr] (steradians) | Derived (error ellipse area) | EQ-007 | EQ-010 |

**Equation of Motion / Field Equations:**

- The EMRI orbital dynamics are computed by the `few` package (not implemented in this codebase). The `Pn5AAKWaveform` class uses a 5th-order post-Newtonian trajectory with analytic kludge waveform summation.
  - File: `master_thesis_code/waveform_generator.py` (lines 92--103)
  - The waveform generator accepts the 14 EMRI parameters and returns time-domain TDI channel data.

**Constraints:**

- Flat universe: Omega_m + Omega_DE = 1 (assumed throughout; see `physical_relations.py` default values and `galaxy.py` line 126 where Omega_DE = 1 - Omega_m implicitly)
- Parameter bounds enforced in `ParameterSpace`: e.g., M in [1e4, 1e7] M_sun, a in [0, 1], e0 in [0.05, 0.7], p0 in [10, 16]
  - File: `master_thesis_code/datamodels/parameter_space.py` (lines 47--129)
- Model M1 assumptions: M in [10^4.5, 10^6] M_sun, a fixed at 0.98, mu fixed at 10 M_sun, e0 < 0.2
  - File: `master_thesis_code/cosmological_model.py` (lines 182--196)

## Symmetries and Conservation Laws

**Exact Symmetries:**

- Cosmological isotropy and homogeneity: FRW metric assumed (flat LCDM)
  - File: `master_thesis_code/physical_relations.py` (entire module)

**Approximate Symmetries:**

- Not applicable at the level of this analysis (no field theory symmetry analysis in the codebase).

**Gauge Symmetries:**

- TDI formulation removes laser frequency noise via specific linear combinations of detector outputs (A, E, T channels)
  - File: `master_thesis_code/LISA_configuration.py` (line 76, `channel` argument selects A/E/T)
  - The gauge freedom of GR is handled internally by the `few` waveform generator.

**Ward Identities / Selection Rules:**

- Not applicable.

**Anomalies:**

- Not applicable (classical GR + statistical inference).

**Topological Properties:**

- Not applicable.

**Dualities and Correspondences:**

- Not applicable.

## Parameters and Couplings

**Fundamental Parameters:**

- h (dimensionless Hubble parameter): H_0 / (100 km/s/Mpc), fiducial 0.73, search range [0.6, 0.86]
  - Defined in: `constants.py` (lines 23--26)
- Omega_m (matter density): 0.25 (fiducial)
  - Defined in: `constants.py` (line 29)
- Omega_DE (dark energy density): 0.75 (= 1 - Omega_m, flat universe)
  - Defined in: `constants.py` (line 30)
- w_0 (dark energy EoS): -1.0 (LCDM)
  - Defined in: `constants.py` (line 31)
- w_a (dark energy EoS evolution): 0.0 (LCDM)
  - Defined in: `constants.py` (line 32)
- c (speed of light): 299792458.0 m/s (from astropy)
  - Defined in: `constants.py` (line 19)
- G (Newton's constant): in Gpc^3 / (s^2 M_sun) (from astropy)
  - Defined in: `constants.py` (line 20)
- SNR_THRESHOLD: 15 (detection threshold)
  - Defined in: `constants.py` (line 48)
- LISA_ARM_LENGTH: 2.5e9 m
  - Defined in: `constants.py` (line 69)

**Derived Quantities:**

- H_0 = h * 100 * KM_TO_M / GPC_TO_MPC^{-1} (Hubble constant in m/s/Gpc)
  - Computed in: `physical_relations.py` (line 69)
  - NOTE: The expression `GPC_TO_MPC ** (-1)` evaluates to 1e-3. So H_0 = h * 100 * 1e3 / 1e-3 = h * 1e8. This gives H_0 in units of m/s per Gpc (since 100 km/s/Mpc = 1e5 m/s/Mpc = 1e8 m/s/Gpc).
- Luminosity distance d_L(z, h): Gpc, via hypergeometric function
  - Computed in: `physical_relations.py` (lines 27--77)
- Comoving volume element dV_c/dz: (km/s/Mpc)^{-3} effective units
  - Computed in: `datamodels/galaxy.py` (lines 114--131)
- Fisher information matrix Gamma_ij: 14x14 symmetric matrix
  - Computed in: `parameter_estimation/parameter_estimation.py` (lines 350--379)
- Cramer-Rao lower bounds sigma_i = sqrt((Gamma^{-1})_{ii})
  - Computed in: `parameter_estimation/parameter_estimation.py` (lines 382--424)

**Dimensionless Ratios:**

- SNR = sqrt(<h|h>): detection criterion SNR >= 15
- Fractional d_L error: sigma(d_L) / d_L, threshold 10% for Pipeline B filtering
  - Defined in: `bayesian_inference/bayesian_statistics.py` (line 58)
- Fisher matrix condition number: kappa = cond(Gamma), logged before inversion
  - Computed in: `parameter_estimation/parameter_estimation.py` (line 393)

## Phase Structure / Regimes

**Regimes Studied:**

- Dark siren regime: EMRIs detected by LISA with SNR >= 15, d_L < 1.55 Gpc, z < 1.5
  - File: `constants.py` (line 48, line 58), `cosmological_model.py` (line 194--196)
- LCDM fiducial: w_0 = -1, w_a = 0 (the wCDM parameters are accepted but silently ignored in `dist()`)
  - File: `physical_relations.py` (line 72 uses `lambda_cdm_analytic_distance` regardless of w_0, w_a)

**Phase Transitions / Crossovers:**

- Not applicable.

**Known Limiting Cases:**

- d_L(z=0) = 0: verified by docstring example in `physical_relations.py` (line 68)
- E(z=0) = 1 for Omega_m + Omega_DE = 1 (flat universe): follows from EQ-002
- p_det(z -> 0) -> 1 (nearby sources are always detected): follows from EQ-012
- p_det(z -> large) -> 0 (distant sources undetectable): follows from EQ-012

## Units and Conventions

**Unit System:**

- SI-adjacent mixed units (NOT natural units):
  - Distances: Gpc (luminosity distance, comoving distance)
  - Masses: solar masses M_sun
  - Time: seconds (waveform sampling)
  - Frequencies: Hz
  - Angles: radians (ecliptic coordinates)
  - PSD: Hz^{-1} (one-sided strain power spectral density)
- Metric signature: (+,-,-,-) (standard GR, handled by `few`)
- File: `constants.py` (lines 17--39), `datamodels/parameter_space.py` (units in Parameter fields)

**Key Conventions:**

- Hubble parameter h is dimensionless (H_0 / 100 km/s/Mpc); TRUE_HUBBLE_CONSTANT = 0.7 for Bayesian inference, H = 0.73 for simulation fiducial
- MBH spin a is dimensionless (0 to 0.998)
- Sky coordinates: ecliptic frame (qS = polar, phiS = azimuthal); `is_ecliptic_latitude=False` in `waveform_generator.py` (line 60)
- TDI channels: "AE" (first-generation TDI, equal arm length approximation)
  - File: `constants.py` (line 34), `waveform_generator.py` (line 29)
- Fisher matrix derivatives: 5-point central difference stencil with per-parameter epsilon
  - File: `parameter_estimation/parameter_estimation.py` (lines 189--258)
- Pipeline A uses hardcoded 10% fractional d_L error; Pipeline B uses full Fisher matrix covariance
- Pipeline B normalizes covariance entries by dividing off d_L and M to form fractional covariances
  - File: `bayesian_inference/bayesian_statistics.py` (lines 175--217)
- Redshift uncertainty model: sigma_z = 0.013 * (1+z)^3, capped at 0.015
  - File: `datamodels/galaxy.py` (line 67), `constants.py` (line 51)

---

_Framework analysis: 2026-03-30_
