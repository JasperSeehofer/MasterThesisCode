import logging
import matplotlib.pyplot as plt
import numpy as np
import resource
import datetime
import os

from master_thesis_code.arguments import Arguments
from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation
from master_thesis_code.constants import SNR_THRESHOLD

# logging setup
_ROOT_LOGGER = logging.getLogger()


def main() -> None:
    """
    Run main to start the program.
    """
    arguments = Arguments.create()
    _configure_logger(arguments.working_directory, arguments.log_level)
    arguments.validate()
    _ROOT_LOGGER.info("---------- STARTING MASTER THESIS CODE ----------")

    parameter_estimation = ParameterEstimation(wave_generation_type="FastSchwarzschildEccentricFlux", use_gpu=arguments.use_gpu)
    check_dependency = False

    if check_dependency:
        for parameter_symbol in ["qS", "phiS"]:
            parameter_estimation.check_parameter_dependency(parameter_symbol=parameter_symbol, steps=5)

    counter = 0
    for i in range(arguments.simulation_steps):
        _ROOT_LOGGER.debug(f"simulation step {i}.")
        _ROOT_LOGGER.info(f"{counter} evaluations successful.")
        parameter_estimation.parameter_space.randomize_parameters()
        snr = parameter_estimation.compute_signal_to_noise_ratio()
        if snr < SNR_THRESHOLD:
            _ROOT_LOGGER.info(f"SNR threshold check failed: {np.round(snr, 3)} < {SNR_THRESHOLD}.")
            continue
        else:
            _ROOT_LOGGER.info(f"SNR threshold check successful: {np.round(snr, 3)} >= {SNR_THRESHOLD}")
        cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds(parameter_list=["M", "qS", "phiS"])
        parameter_estimation.save_cramer_rao_bound(cramer_rao_bound_dictionary=cramer_rao_bounds, snr=snr)
        counter += 1

    parameter_estimation.lisa_configuration._visualize_lisa_configuration()
    parameter_estimation._visualize_cramer_rao_bounds()

    _ROOT_LOGGER.debug(f"Peak memory usage: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *1e-6} in GB.")

def _configure_logger(working_directory: str, log_level: int) -> None:
    _ROOT_LOGGER.setLevel(log_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    _ROOT_LOGGER.addHandler(stream_handler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(working_directory, f"master_thesis_code_{timestamp}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s")
    file_handler.setFormatter(formatter)
    _ROOT_LOGGER.addHandler(file_handler)

    _ROOT_LOGGER.info(f"Log file location: {log_file_path}")

if __name__ == "__main__":
    main()
