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
    
    _ROOT_LOGGER.debug(f"CUDA version: 11.7")
    _display_GPU_information()
    mempool = cp.get_default_memory_pool()

    parameter_estimation = ParameterEstimation(wave_generation_type=WaveGeneratorType.pn5, use_gpu=True)
    check_dependency = False

    if check_dependency:
        for parameter_symbol in ["qS", "phiS"]:
            parameter_estimation.check_parameter_dependency(parameter_symbol=parameter_symbol, steps=5)

    counter = 0
    start_time = time()
    GPU_usage = []
    while counter < arguments.simulation_steps:
        GPU_usage.append(mempool.total_bytes()/10**9)
        mempool.free_all_blocks()
        GPU_usage.append(mempool.total_bytes()/10**9)
        _ROOT_LOGGER.debug(f"currently used GPU memory: {int(mempool.used_bytes())/10**9}/{int(mempool.total_bytes())/10**9} Gbytes (used/total)")
        _ROOT_LOGGER.info(f"{counter} evaluations successful. ({counter/(time()-start_time)*60}/min)")
        parameter_estimation.parameter_space.randomize_parameters()
        snr = parameter_estimation.compute_signal_to_noise_ratio()
        if snr < SNR_THRESHOLD:
            _ROOT_LOGGER.info(f"SNR threshold check failed: {np.round(snr, 3)} < {SNR_THRESHOLD}.")
            continue
        else:
            _ROOT_LOGGER.info(f"SNR threshold check successful: {np.round(snr, 3)} >= {SNR_THRESHOLD}")
        cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds()
        _display_GPU_information()
        parameter_estimation.save_cramer_rao_bound(cramer_rao_bound_dictionary=cramer_rao_bounds, snr=snr)
        counter += 1

    plt.plot(GPU_usage)
    plt.savefig("GPU_usage.png")
        
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

    # set matplotlib logging to info, because it is very talkative
    plt.set_loglevel("warning")

    _ROOT_LOGGER.info(f"Log file location: {log_file_path}")

def _display_GPU_information() -> None:
    _ROOT_LOGGER.info("="*40, "GPU Details", "="*40)
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        # get % percentage of GPU usage of that GPU
        gpu_load = f"{gpu.load*100}%"
        # get free memory in MB format
        gpu_free_memory = f"{gpu.memoryFree}MB"
        # get used memory
        gpu_used_memory = f"{gpu.memoryUsed}MB"
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        # get GPU temperature in Celsius
        gpu_temperature = f"{gpu.temperature} °C"
        gpu_uuid = gpu.uuid
        list_gpus.append((
            gpu_id, gpu_name, gpu_load, gpu_free_memory, gpu_used_memory,
            gpu_total_memory, gpu_temperature, gpu_uuid
        ))

    _ROOT_LOGGER.info(tabulate(list_gpus, headers=("id", "name", "load", "free memory", "used memory", "total memory",
                                   "temperature", "uuid")))


if __name__ == "__main__":
    main()
