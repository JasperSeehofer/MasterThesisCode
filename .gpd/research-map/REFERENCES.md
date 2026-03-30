# Reference and Anchor Map

**Analysis Date:** 2026-03-30

## Active Anchor Registry

| Anchor ID | Anchor | Type | Source / Locator | Why It Matters | Contract Subject IDs | Required Action | Carry Forward To |
|-----------|--------|------|------------------|----------------|----------------------|-----------------|------------------|
| REF-001 | Hogg (1999) distance measures | method | arXiv:astro-ph/9905116, Eq. (16), (27) | Defines d_L(z) formula (EQ-001) and comoving volume element dV_c/dz (EQ-009) used throughout | -- | use/cite | execution/verification |
| REF-002 | Babak et al. (2023) LISA sensitivity | benchmark | arXiv:2303.15929 | LISA PSD for A/E/T channels; cited in `LISA_configuration.py` (line 76) and `constants.py` (line 74). NOTE: confusion noise formula is NOT from this paper (see REF-003). | -- | cite/compare | execution |
| REF-003 | Cornish & Robson (2017) confusion noise | method | arXiv:1703.09858, Eq. (3) | Galactic foreground confusion noise S_c(f) parameterization; actually implements confusion noise in `LISA_configuration.py` (lines 83--109) | -- | use/cite | execution |
| REF-004 | Robson, Cornish & Liu (2019) | method | arXiv:1803.01944, Eq. (14) | Alternative/updated galactic confusion noise fit; cited alongside REF-003 in `LISA_configuration.py` (line 85) | -- | cite | verification |
| REF-005 | Babak et al. (2017) Model M1 | benchmark | Phys. Rev. D 95, 103012 (2017) | EMRI event rate model; `Model1CrossCheck` in `cosmological_model.py` (line 167). EMRI rate = 294/yr. Polynomial merger distribution coefficients (lines 70--126) extracted from this paper. | -- | use/compare | execution/verification |
| REF-006 | Vallisneri (2008) Fisher matrix accuracy | method | arXiv:gr-qc/0703086 | Justifies O(epsilon^4) 5-point stencil for Fisher matrix derivatives; cited in `parameter_estimation.py` (lines 203, 242) | -- | use/cite | execution |
| REF-007 | Laghi et al. (2021) dark siren method | method | arXiv:2102.01708 | Dark siren H_0 inference framework; cited in `LamCDMScenario` (line 303) and likely the primary methodological reference for Pipeline B | -- | use/cite | execution/writing |
| REF-008 | Reynolds (2013) MBH spin distribution | benchmark | ApJ 762 68 (2013) / doi:10.1088/0004-637X/762/2/68 | MBH spin distribution: truncated normal, mean 0.98, sigma 0.05; cited in `cosmological_model.py` (line 162) | -- | use/cite | execution |
| REF-009 | GLADE galaxy catalog | prior artifact | GLADE+ catalog (external data file, loaded by `galaxy_catalogue/handler.py`) | Real galaxy catalog used in Pipeline B for host galaxy identification via BallTree spatial lookups | -- | use | execution |
| REF-010 | Planck 2018 cosmological parameters | benchmark | Planck Collaboration (2020), arXiv:1807.06209 | Best-fit Omega_m = 0.3153, h = 0.6736; project uses WMAP-era values (Omega_m = 0.25, h = 0.73). Known issue in `constants.py` (lines 25, 29). | -- | compare | verification |
| REF-011 | fastemriwaveforms (few) package | method | `few` Python package, v2.0.0rc1 | EMRI waveform generation (Pn5AAKWaveform, FastSchwarzschildEccentricFlux); core dependency for SNR and Fisher matrix computation | -- | use | execution |
| REF-012 | fastlisaresponse package | method | `fastlisaresponse` Python package, v1.1.9 | LISA TDI response wrapper; generates detector-frame waveforms from source-frame signals | -- | use | execution |
| REF-013 | M1 model detection fraction grid | prior artifact | `master_thesis_code/M1_model_extracted_data/detection_fraction.py` | Pre-computed detection fraction grid on (M, z) for Model M1; used by `Model1CrossCheck.detection_fraction` | -- | use | execution |
| REF-014 | M1 model merger distribution coefficients | prior artifact | `master_thesis_code/cosmological_model.py` (lines 70--126) | 9th-order polynomial coefficients for dN/dz in 5 mass bins; extracted from Babak et al. (2017) figures/data | -- | use | execution |

## Benchmarks and Comparison Targets

- **d_L(z=0) = 0 Gpc**: Basic sanity check for the luminosity distance function.
  - Source: EQ-001 with z=0
  - Compared in: `physical_relations.py` docstring (line 68), test suite
  - Status: matched

- **E(z=0) = 1 for flat LCDM**: Hubble function normalization.
  - Source: EQ-002 definition
  - Compared in: implicit (no explicit test found)
  - Status: pending

- **EMRI rate = 294/yr (Model M1)**: Total EMRI detection rate.
  - Source: REF-005 (Babak et al. 2017)
  - Compared in: `cosmological_model.py` (line 172)
  - Status: postulated (hardcoded), not independently verified against the paper

- **SNR threshold = 15**: Detection criterion.
  - Source: REF-005, REF-007
  - Compared in: `constants.py` (line 48)
  - Status: matched (recently lowered from 20)

- **Planck 2018 cosmology vs project fiducial**: Omega_m = 0.3153 vs 0.25; h = 0.6736 vs 0.73.
  - Source: REF-010
  - Compared in: not compared (known issue)
  - Status: contested -- project uses WMAP-era values

- **MBH spin a = 0.98 (truncnorm, sigma = 0.05)**: Model M1 assumption.
  - Source: REF-008 (Reynolds 2013)
  - Compared in: `cosmological_model.py` (lines 61--67)
  - Status: matched

## Prior Artifacts and Baselines

- `master_thesis_code/M1_model_extracted_data/detection_fraction.py`: Pre-computed 69x100 detection fraction grid over (log10(M), z). Used by `Model1CrossCheck` to weight EMRI event sampling. Must remain accessible for any EMRI rate computation.

- `master_thesis_code/M1_model_extracted_data/emri_distribution.py`: EMRI distribution data extracted from Babak et al. (2017).

- `master_thesis_code/M1_model_extracted_data/detection_horizon.py`: Detection horizon data for Model M1.

- `master_thesis_code/M1_model_extracted_data/detection_distribution_simplified.py`: Simplified detection distribution.

- `simulations/cramer_rao_bounds.csv` (runtime artifact): Merged Cramer-Rao bounds from all simulation tasks. Input to Pipeline B evaluation.

- `simulations/prepared_cramer_rao_bounds.csv` (runtime artifact): Filtered/prepared detection data for Pipeline B.

- `simulations/undetected_events.csv` (runtime artifact): Undetected events, used to build KDE detection probability in Pipeline B.

## Open Reference Questions

- **Missing citation for redshift uncertainty scaling**: `galaxy.py` (line 67) uses sigma_z = 0.013 * (1+z)^3, which has no reference. Standard spectroscopic redshift errors scale as (1+z), not (1+z)^3. This non-standard form caps out at z ~ 0.14, making it only relevant for the low-redshift catalog. Source and justification needed.

- **Confusion noise citation mismatch**: `constants.py` (line 74) cites arXiv:2303.15929 for the galactic confusion noise PSD coefficients, but a comment on line 75--77 notes this paper does NOT contain the formula. The actual source is Cornish & Robson (2017) arXiv:1703.09858 Eq. (3) / Robson et al. (2019) arXiv:1803.01944 Eq. (14). Citation should be corrected.

- **wCDM implementation incomplete**: `physical_relations.py` accepts w_0 and w_a parameters but `dist()` (line 72) always calls `lambda_cdm_analytic_distance()` which hardcodes LCDM (w_0=-1, w_a=0). The `DarkEnergyScenario` class in `cosmological_model.py` (lines 329--355) defines w_0 and w_a parameter spaces but these cannot actually be used for distance computations.

- **Pipeline B likelihood normalization**: `bayesian_statistics.py` (line 563) divides numerator integrand by d_L. The physical justification (Jacobian from d_L to z space?) needs explicit documentation or citation.

- **Laghi et al. (2021) vs project methodology**: The `LamCDMScenario` cites arXiv:2102.01708 but the exact correspondence between the paper's methodology and Pipeline B's implementation is not documented. A detailed comparison would clarify which approximations are project-specific.

## Background Reading

- **LISA Science Requirements Document**: Defines the LISA mission parameters (arm length, noise levels, observation time). Background context for the PSD and TDI implementation.

- **Amaro-Seoane et al. (2017)**: EMRI physics review. Background for the 14-parameter EMRI model and the role of EMRIs as standard sirens.

- **Schutz (1986)**: Original dark siren proposal. Foundational context for using GW sources without EM counterparts to measure H_0.

- **Chen, Fishbach & Holz (2018)**: Modern dark siren methodology review. Provides context for the statistical framework implemented in Pipeline A and Pipeline B.

---

_Reference map: 2026-03-30_
