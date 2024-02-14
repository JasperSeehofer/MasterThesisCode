import logging
import matplotlib.pyplot as plt
import numpy as np
import resource
import datetime
import os
import warnings
from time import time

from master_thesis_code.parameter_estimation.evaluation import DataEvaluation
from master_thesis_code.arguments import Arguments
from master_thesis_code.exceptions import ParameterOutOfBoundsError
from master_thesis_code.cosmological_model import Model1CrossCheck
from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler

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
    start_time = time()

    cosmological_model = Model1CrossCheck()
    galaxy_catalog = GalaxyCatalogueHandler()

    if arguments.simulation_steps > 0:
        data_simulation(arguments.simulation_steps, cosmological_model, galaxy_catalog)

    if arguments.evaluate:
        evaluate(cosmological_model, galaxy_catalog)

    end_time = time()
    _ROOT_LOGGER.debug(f"Finished in {end_time - start_time}s.")


def _configure_logger(working_directory: str, log_level: int) -> None:
    _ROOT_LOGGER.setLevel(log_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    _ROOT_LOGGER.addHandler(stream_handler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(
        working_directory, f"master_thesis_code_{timestamp}.log"
    )
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    )
    file_handler.setFormatter(formatter)
    _ROOT_LOGGER.addHandler(file_handler)

    # set matplotlib logging to info, because it is very talkative
    plt.set_loglevel("warning")

    _ROOT_LOGGER.info(f"Log file location: {log_file_path}")


def data_simulation(
    simulation_steps: int,
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
) -> None:
    # conditional imports because they require GPU
    from master_thesis_code.memory_management import MemoryManagement
    from master_thesis_code.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
        WaveGeneratorType,
    )

    memory_management = MemoryManagement()
    memory_management.display_GPU_information()
    memory_management.display_fft_cache()

    parameter_estimation = ParameterEstimation(
        waveform_generation_type=WaveGeneratorType.PN5_AAK,
        parameter_space=cosmological_model.parameter_space,
    )

    parameter_estimation.lisa_configuration._visualize_lisa_configuration()

    counter = 0
    iteration = 0
    while counter < simulation_steps:
        memory_management.gpu_usage_stamp()
        memory_management.memory_pool.free_all_blocks()
        memory_management.gpu_usage_stamp()

        _ROOT_LOGGER.info(
            f"{counter} / {iteration} evaluations successful. ({counter/(time()-memory_management._start_time)*60}/min)"
        )
        iteration += 1

        parameter_estimation.parameter_space.randomize_parameters()
        host_galaxy = galaxy_catalog.get_random_host_in_mass_range(
            parameter_estimation.parameter_space.M.lower_limit,
            parameter_estimation.parameter_space.M.upper_limit,
        )
        parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy)

        _ROOT_LOGGER.debug(
            f"Parameters used: {parameter_estimation.parameter_space._parameters_to_dict()}, host galaxy: {host_galaxy}"
        )

        try:
            warnings.filterwarnings("error")
            snr = parameter_estimation.compute_signal_to_noise_ratio()
            warnings.resetwarnings()
        except Warning as e:
            if "Mass ratio" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught warning that mass ratio is out of bounds. Continue with new parameters..."
                )
                continue
            else:
                _ROOT_LOGGER.warning(f"{str(e)}. Continue with new parameters...")
                continue
        except ParameterOutOfBoundsError as e:
            _ROOT_LOGGER.warning(
                f"Caught ParameterOutOfBoundsError during parameter estimation: {str(e)}. Continue with new parameters..."
            )
            continue
        except AssertionError as e:
            _ROOT_LOGGER.warning(
                f"caught AssertionError: {str(e)}. Continue with new parameters..."
            )
            continue
        except RuntimeError as e:
            _ROOT_LOGGER.warning(
                f"Caught RuntimeError during waveform generation : {str(e)} .\n Continue with new parameters..."
            )
            continue
        except ValueError as e:
            if "EllipticK" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught EllipticK error from waveform generator. Continue with new parameters..."
                )
                continue
            elif "Brent root solver does not converge" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught brent root solver error because it did not converge. Continue with new parameters..."
                )
                continue
            else:
                raise ValueError(e)

        if snr < cosmological_model.snr_threshold:
            _ROOT_LOGGER.info(
                f"SNR threshold check failed: {np.round(snr, 3)} < {cosmological_model.snr_threshold}."
            )
            continue

        _ROOT_LOGGER.info(
            f"SNR threshold check successful: {np.round(snr, 3)} >= {cosmological_model.snr_threshold}"
        )
        cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds()
        parameter_estimation.save_cramer_rao_bound(
            cramer_rao_bound_dictionary=cramer_rao_bounds, snr=snr
        )
        counter += 1

        memory_management.display_GPU_information()
        memory_management.display_fft_cache()

    parameter_estimation.lisa_configuration._visualize_lisa_configuration()
    parameter_estimation._visualize_cramer_rao_bounds()

    memory_management.plot_GPU_usage()


def evaluate(
    cosmological_model: Model1CrossCheck, galaxy_catalog: GalaxyCatalogueHandler
) -> None:
    data_simulation = DataEvaluation()
    data_simulation.visualize()


if __name__ == "__main__":
    main()
