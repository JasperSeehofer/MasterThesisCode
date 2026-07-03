"""Physical constants, cosmological parameters, and simulation configuration.

All numeric values are module-level constants.  Physical constants are derived
from astropy for traceability; cosmological parameters are the fiducial ΛCDM
values used in the simulation.
"""

import numpy as np
from astropy import constants as ac
from astropy import units as u

# infinity
INFINITY: float = 1e12
REAL_PART: str = "real"
IMAGINARY_PART: str = "imaginary"

# physical constants (values derived from astropy for traceability)
M_IN_GPC: float = 3.2407788498994e-26  # m / Gpc (conversion factor)
C: float = float(ac.c.to(u.m / u.s).value)  # 299792458.0 m/s
G: float = float(ac.G.to(u.Gpc**3 / (u.s**2 * u.solMass)).value)  # Gpc^3 / (s^2 M_sun)
SPEED_OF_LIGHT_KM_S: float = C / 1000  # km/s, derived from C (m/s)
H0: float = 73e3  # m / (s * Mpc), Hubble constant in SI-adjacent units
H_MIN: float = 60.0  # lower limit for dimensionless h
H_MAX: float = 86.0  # upper limit for dimensionless h
H: float = 0.73  # dimensionless h = H₀ / (100 km/s/Mpc), fiducial simulation value
TRUE_HUBBLE_CONSTANT: float = 0.7  # dimensionless h, fiducial value for Bayesian inference

# cosmological parameters fiducial values
# [PHYSICS] G11: matched to the M1 EMRI population model's cosmology — the
# Barausse (2012) semi-analytic MBH model underlying Babak et al. (2017) M1
# assumes flat LambdaCDM with Omega_DM = 0.227, Omega_b = 0.0456
# (=> Omega_m = 0.2726), H0 = 70.4 km/s/Mpc (arXiv:1201.5888, end of Intro).
# The extracted M1 horizon/rate/dN-dz data (M1_model_extracted_data/) live in
# that cosmology; sampling them with a different Omega_m would be inconsistent.
# The "true universe is Planck (0.3153)" case is quoted as a systematic in
# .planning/gate/G7_systematics_budget.md, NOT absorbed into the fiducial.
OMEGA_M: float = 0.2726
OMEGA_DE: float = 0.7274
W_0: float = -1.0
W_A: float = 0.0

ESA_TDI_CHANNELS: str = "AE"

# unit conversions
RADIAN_TO_DEGREE: float = 360 / (2 * np.pi)  # rad → deg
GPC_TO_MPC: float = 1e3  # 1 Gpc = 1000 Mpc
KM_TO_M: float = 1e3  # 1 km = 1000 m

# simulation configuration
SIMULATION_CONFIGURATION_FILE: str = "simulation_configuration.json"
SIMULATION_PATH: str = "simulation_path"
DEFAULT_SIMULATION_PATH: str = "simulations/simulation"
# Parameter configuration
MINIMAL_FREQUENCY: float = 1e-5
MAXIMAL_FREQUENCY: float = 1
SNR_THRESHOLD: float = 20
PRE_SCREEN_SNR_FACTOR: float = 0.3  # pre-screen heuristic (main.py simulation loop)
# G10 gate: Fisher matrices with kappa above this are numerical noise after
# inversion (float64 ~16 digits; 1e14 leaves <2). Event is skipped, not stored.
FISHER_CONDITION_NUMBER_MAX: float = 1e14

# galaxy catalog and EMRI detection
GALAXY_REDSHIFT_ERROR_COEFFICIENT: float = 0.013  # Galaxy.redshift_uncertainty ∝ 0.013*(1+z)^3
FRACTIONAL_LUMINOSITY_ERROR: float = 0.1  # fractional error on measured luminosity distance
FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR: float = 0.1  # fractional BH mass catalog uncertainty
FRACTIONAL_MEASURED_MASS_ERROR: float = 1e-8  # fractional error on measured redshifted mass
SKY_LOCALIZATION_ERROR: float = 2 / 180 * np.pi  # rad, EMRI sky localization error (2 degrees)
# [PHYSICS] Residual host peculiar-velocity dispersion, marginalized into the
# host-z kernel at inference time (issue #16 decision 2026-07-03):
# sigma_z_pv = (1 + z_g) * SIGMA_V_PEC_KM_S / c, added in quadrature to the
# catalogue sigma_z in bayesian_statistics.single_host_likelihood.
# (1+z) factor: Davis et al. (2011), arXiv:1012.2912, Eqs. (1)/(A1); quadrature
# convention: Mastrogiovanni et al. (2023), arXiv:2305.10488, Sec. IV.
# 200 km/s follows Fishbach et al. (2019), arXiv:1807.05667, Sec. 2.2 and
# Chen et al. (2018), arXiv:1712.06531; the LISA-EMRI precedent (Laghi et al.
# 2021, arXiv:2102.01708, Sec. 4) uses 500 km/s — kept as a systematics-budget
# row, not the default. Distinct from (residual on top of) the GLADE+
# PV-CORRECTION error already folded into the catalogue z_error at parse time
# (handler.parse_to_reduced_catalog, 0.0015 floor for rows without it).
SIGMA_V_PEC_KM_S: float = 200.0
GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT: float = 0.00001  # minimum redshift for galaxy catalog
# Documented catalogue depth bound. NOTE: currently UNWIRED — the reduced
# catalogue CSV is written full-depth (no z cut in parse_to_reduced_catalog)
# and the effective load-time depth is Model1CrossCheck.max_redshift via
# _get_pruned_galaxy_catalog. Kept as documentation of the depth the pipeline
# is validated for; raised alongside HOST_DRAW_Z_MAX (issue #20).
GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT: float = 1.55  # maximum redshift for galaxy catalog
# [PHYSICS] Campaign population depth for the in-catalog host draw (issue #20,
# user decision 2026-07-03: "first go for z=1.5 and then see the results and
# HPC performance"). The pre-dt² justification ("horizon z ≈ 0.18, truncation
# EXACT") is retired: after the dt² fix the EMRI horizon reaches z ~ 1.5+, so
# this is a deliberate population-model choice matching
# Model1CrossCheck.max_redshift = 1.5 (cosmological_model.py), NOT a claim
# that p_det = 0 beyond. The injection-campaign z_cut derives from this
# constant (main.py) so the P_det grid always spans the host-draw volume.
HOST_DRAW_Z_MAX: float = 1.5
LUMINOSITY_DISTANCE_THRESHOLD_GPC: float = 1.55  # Gpc, LISA detection horizon for EMRIs
# Multiplicative safety margin on the population-derived d_L pre-screen bound
# (physical_relations.luminosity_distance_prescreen_gpc). Placeholder pending
# re-measurement on post-dt^2 injection data — issue #19. Replaces the retired
# LUMINOSITY_DISTANCE_PRESCREEN_GPC = 2.0, which was calibrated on pre-dt^2
# (SNR/10-scale) injections and lay inside the z <= 0.5 host-draw volume.
PRESCREEN_DL_MARGIN: float = 1.05

# saving Cramer-Rao bounds for marginalization.
CRAMER_RAO_BOUNDS_PATH: str = "simulations/cramer_rao_bounds_simulation_$index.csv"
CRAMER_RAO_BOUNDS_OUTPUT_PATH: str = "simulations/cramer_rao_bounds.csv"
SNR_ANALYSIS_PATH: str = "simulations/snr_analysis.csv"
PREPARED_CRAMER_RAO_BOUNDS_PATH: str = "simulations/prepared_cramer_rao_bounds.csv"

# ── Coordinate-frame provenance tag (the SINGLE source of truth) ──────────────
# Every sky angle (qS/phiS) and sky covariance in this pipeline is in the
# barycentric ecliptic frame BarycentricTrueEcliptic(J2000) AFTER the one-and-only
# rotation at GLADE ingestion (handler._rotate_equatorial_to_ecliptic, COORD-03,
# commit b460297). A fresh simulation run is ECLIPTIC-NATIVE and must NEVER be
# rotated again; the rotating scripts/migrate_crb_to_ecliptic.py is for LEGACY
# pre-COORD-03 equatorial CRBs ONLY. The simulation stamps this tag into the
# `_coord_frame`/`_cov_frame` CRB columns at write time so the data is
# self-describing and the evaluation guard passes natively (see .planning/FRAME-AUDIT.md).
ECLIPTIC_FRAME_TAG: str = "ecliptic_BarycentricTrue_J2000"

# Injection campaign paths (for simulation-based detection probability)
INJECTION_DATA_DIR: str = "simulations/injections"
INJECTION_CSV_PATH: str = "simulations/injections/injection_h_{h_label}_task_{index}.csv"

# ── LISA hardware constants ──────────────────────────────────────────────────
LISA_ARM_LENGTH: float = 2.5e9  # m, interferometer arm length
YEAR_IN_SEC: int = int(365.5 * 24 * 60 * 60)  # s, seconds per year
LISA_STEPS: int = 10_000  # number of time steps per observation year
LISA_DT: float = YEAR_IN_SEC / LISA_STEPS  # s, time step size

# LISA galactic confusion noise PSD coefficients (arXiv:2303.15929, Eq. 17)
# NOTE: arXiv:2303.15929 does not contain this formula; actual source is
# Cornish & Robson (2017) arXiv:1703.09858 Eq. (3) / Robson et al. (2019)
# arXiv:1803.01944 Eq. (14). Keeping original citation for literature traceability.
LISA_PSD_A: float = 1.14e-44  # overall amplitude
LISA_PSD_ALPHA: float = 1.8  # low-frequency spectral slope
LISA_PSD_F2: float = 0.31e-3  # Hz, knee frequency
LISA_PSD_A1: float = -0.25  # time-dependent exponent coefficient
LISA_PSD_B1: float = -2.7  # time-dependent exponent coefficient
LISA_PSD_AK: float = -0.27  # time-dependent exponent coefficient
LISA_PSD_BK: float = -2.47  # time-dependent exponent coefficient
