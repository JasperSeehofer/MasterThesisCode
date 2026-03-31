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
SPEED_OF_LIGHT_KM_S: float = 300000.0  # km/s (approximation used in bayesian_inference_mwe.py)
H0: float = 73e3  # m / (s * Mpc), Hubble constant in SI-adjacent units
H_MIN: float = 60.0  # lower limit for dimensionless h
H_MAX: float = 86.0  # upper limit for dimensionless h
H: float = 0.73  # dimensionless h = H₀ / (100 km/s/Mpc), fiducial simulation value
TRUE_HUBBLE_CONSTANT: float = 0.7  # dimensionless h, fiducial value for Bayesian inference

# cosmological parameters fiducial values
OMEGA_M: float = 0.25
OMEGA_DE: float = 0.75
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
SNR_THRESHOLD: float = 15

# galaxy catalog and EMRI detection
GALAXY_REDSHIFT_ERROR_COEFFICIENT: float = 0.013  # Galaxy.redshift_uncertainty ∝ 0.013*(1+z)^3
FRACTIONAL_LUMINOSITY_ERROR: float = 0.1  # fractional error on measured luminosity distance
FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR: float = 0.1  # fractional BH mass catalog uncertainty
FRACTIONAL_MEASURED_MASS_ERROR: float = 1e-8  # fractional error on measured redshifted mass
SKY_LOCALIZATION_ERROR: float = 2 / 180 * np.pi  # rad, EMRI sky localization error (2 degrees)
GALAXY_CATALOG_REDSHIFT_LOWER_LIMIT: float = 0.00001  # minimum redshift for galaxy catalog
GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT: float = 0.55  # maximum redshift for galaxy catalog
LUMINOSITY_DISTANCE_THRESHOLD_GPC: float = 1.55  # Gpc, LISA detection horizon for EMRIs

# saving Cramer-Rao bounds for marginalization.
CRAMER_RAO_BOUNDS_PATH: str = "simulations/cramer_rao_bounds_simulation_$index.csv"
CRAMER_RAO_BOUNDS_OUTPUT_PATH: str = "simulations/cramer_rao_bounds.csv"
SNR_ANALYSIS_PATH: str = "simulations/snr_analysis.csv"
UNDETECTED_EVENTS_PATH: str = "simulations/undetected_events_simulation_$index.csv"
UNDETECTED_EVENTS_OUTPUT_PATH: str = "simulations/undetected_events.csv"
PREPARED_CRAMER_RAO_BOUNDS_PATH: str = "simulations/prepared_cramer_rao_bounds.csv"

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
