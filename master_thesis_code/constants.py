import numpy as np

# infinity
INFINITY: float = 1e12
REAL_PART: str = "real"
IMAGINARY_PART: str = "imaginary"

# physical constants
M_IN_GPC = 3.2407788498994e-26
C = 299792458.0  # m / s
# C = 1.0
G = 3.24077929**2 * 4.3009172706e-58  # Gpc^3 / (s^2 * solar masses)
# G = 1.0
H0 = float(70e3)  # m / (s * Mpc) TODO: check if this is correct

ESA_TDI_CHANNELS = "AE"

# unit trafos
RADIAN_TO_DEGREE = 360 / 2 / np.pi
GPC_TO_MPC = float(1e3)

# simulation configuration
SIMULATION_CONFIGURATION_FILE = "simulation_configuration.json"
SIMULATION_PATH = "simulation_path"
DEFAULT_SIMULATION_PATH = "simulations/simulation"
IS_PLOTTING_ACTIVATED = False

# Parameter configuration
MINIMAL_FREQUENCY = 1e-5
MAXIMAL_FREQUENCY = 1
SNR_THRESHOLD = 20

# saving Cramer-Rao bounds for marginalization.
CRAMER_RAO_BOUNDS_PATH = "simulations/cramer_rao_bounds_unbiased.csv"
SNR_ANALYSIS_PATH = "simulations/snr_analysis.csv"
UNDETECTED_EVENTS_PATH = "simulations/undetected_events_unbiased.csv"
