import logging
import matplotlib
import numpy as np

from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation
from master_thesis_code.LISA_configuration import LISAConfiguration
from master_thesis_code.constants import SNR_THRESHOLD

# logging setup
logging.basicConfig(filename='logfile.log', encoding='utf-8', level=logging.DEBUG)

# disable matplotlib logger - it is very talkative in debug mode
matplotlib.pyplot.set_loglevel (level = 'warning')
# get the the logger with the name 'PIL'
pil_logger = logging.getLogger('PIL')  
# override the logger logging level to INFO
pil_logger.setLevel(logging.INFO)


def main() -> None:
    """
    Run main to start the program.
    """
    logging.info("---------- STARTING MASTER THESIS CODE ----------")
    parameter_estimation = ParameterEstimation(wave_generation_type="FastSchwarzschildEccentricFlux")
    simulate = False
    check_dependency = False

    if check_dependency:
        for parameter_symbol in ["qS", "phiS"]:
            parameter_estimation.check_parameter_dependency(parameter_symbol=parameter_symbol, steps=3)

    counter = 0
    if simulate:
        for i in range(50):
            logging.debug(f"simulation step {i}.")
            logging.info(f"{counter} evaluations successful.")
            parameter_estimation.parameter_space.randomize_parameters()
            snr = parameter_estimation.compute_signal_to_noise_ratio()
            if snr < SNR_THRESHOLD:
                logging.info(f"SNR threshold check failed: {np.round(snr, 3)} < {SNR_THRESHOLD}.")
                continue
            else:
                logging.info(f"SNR threshold check successful: {np.round(snr, 3)} >= {SNR_THRESHOLD}")
            cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds(parameter_list=["M", "qS", "phiS"])
            parameter_estimation.save_cramer_rao_bound(cramer_rao_bound_dictionary=cramer_rao_bounds, snr=snr)
            counter += 1

    parameter_estimation.lisa_configuration._visualize_lisa_configuration()
    parameter_estimation._visualize_cramer_rao_bounds()

if __name__ == "__main__":
    main()