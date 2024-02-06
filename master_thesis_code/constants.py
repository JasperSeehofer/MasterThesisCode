# infinity
INFINITY: float = 1e12
REAL_PART: str = "real"
IMAGINARY_PART: str = "imaginary"

# physical constants
C = 299792458 * 3.2408e-26  # Gpc / s
# C = 1.0
G = 3.24077929**2 * 4.3009172706e-58  # Gpc^3 / (s^2 * solar masses)
# G = 1.0

# simulation configuration
SIMULATION_CONFIGURATION_FILE = "simulation_configuration.json"
SIMULATION_PATH = "simulation_path"
DEFAULT_SIMULATION_PATH = "simulations/simulation"
IS_PLOTTING_ACTIVATED = False

# Parameter configuration
MINIMAL_FREQUENCY = 1e-5
MAXIMAL_FREQUENCY = 1
SNR_THRESHOLD = 10

# saving Cramer-Rao bounds for marginalization.
CRAMER_RAO_BOUNDS_PATH = "simulations/cramer_rao_bounds.csv"
