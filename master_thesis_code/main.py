import logging
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import resource
import datetime
import os
from time import time
import GPUtil
from tabulate import tabulate

from master_thesis_code.arguments import Arguments
from master_thesis_code.memory_management import MemoryManagement
from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation, WaveGeneratorType
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
    
    memory_management = MemoryManagement()
    memory_management.display_GPU_information()
    memory_management.display_fft_cache()

    parameter_estimation = ParameterEstimation(wave_generation_type=WaveGeneratorType.pn5, use_gpu=True)

    counter = 0
    iteration = 0
    while counter < arguments.simulation_steps:
        memory_management.gpu_usage_stamp()
        memory_management.memory_pool.free_all_blocks()
        memory_management.gpu_usage_stamp()

        _ROOT_LOGGER.info(f"{counter} / {iteration} evaluations successful. ({counter/(time()-memory_management._start_time)*60}/min)")
        iteration += 1
        parameter_estimation.parameter_space.randomize_parameters()
        snr = parameter_estimation.compute_signal_to_noise_ratio()
        if snr < SNR_THRESHOLD:
            _ROOT_LOGGER.info(f"SNR threshold check failed: {np.round(snr, 3)} < {SNR_THRESHOLD}.")
            continue
        _ROOT_LOGGER.info(f"SNR threshold check successful: {np.round(snr, 3)} >= {SNR_THRESHOLD}")
        cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds()
        parameter_estimation.save_cramer_rao_bound(cramer_rao_bound_dictionary=cramer_rao_bounds, snr=snr)
        counter += 1

        memory_management.display_GPU_information()
        memory_management.display_fft_cache()
        
    parameter_estimation.lisa_configuration._visualize_lisa_configuration()
    parameter_estimation._visualize_cramer_rao_bounds()
    
    memory_management.plot_GPU_usage()
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

    # set matplotlib logging to info, because it is very talkative
    plt.set_loglevel("warning")

    _ROOT_LOGGER.info(f"Log file location: {log_file_path}")


if __name__ == "__main__":
    main()
