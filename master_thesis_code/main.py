import logging
import time

from parameter_estimation.parameter_estimation import ParameterEstimation
from LISA_configuration import LISAConfiguration
from constants import SNR_THRESHOLD

# logging setup
logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.INFO)

def main() -> None:
    """
    Run main to start the program.
    """
    logging.info("---------- STARTING MASTER THESIS CODE ----------")
    parameter_estimation = ParameterEstimation(wave_generation_type="FastSchwarzschildEccentricFlux")
    simulate = True
    check_dependency = True

    if check_dependency:
        for parameter_symbol in ["qS", "phiS"]:
            parameter_estimation.check_parameter_dependency(parameter_symbol=parameter_symbol, steps=3)

    counter = 0
    if simulate:
        for i in range(15):

            logging.info(f"{counter} steps done.")
            parameter_estimation.parameter_space.randomize_parameters()
            start = time.time()
            snr = parameter_estimation.compute_signal_to_noise_ratio()
            end = time.time()
            logging.info(f"SNR computation took {end-start}s.")
            if snr < SNR_THRESHOLD:
                logging.info(f"SNR threshold check failed: {snr} < {SNR_THRESHOLD}.")
                continue
            else:
                logging.info(f"SNR threshold check successful: {snr} >= {SNR_THRESHOLD}")
            cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds(parameter_list=["M", "qS", "phiS"])
            parameter_estimation.save_cramer_rao_bound(cramer_rao_bound_dictionary=cramer_rao_bounds, snr=snr)
            counter += 1
            end = time.time()
            logging.info(f"Simulation iteration took {end-start}s.")

    parameter_estimation.lisa_configuration._visualize_lisa_configuration()
    parameter_estimation._visualize_cramer_rao_bounds()
if __name__ == "__main__":
    main()