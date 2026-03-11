import datetime
import json
import logging
import os
import subprocess
import warnings
from collections.abc import Iterator
from time import time
from typing import TYPE_CHECKING

import numpy as np

from master_thesis_code.arguments import Arguments
from master_thesis_code.cosmological_model import BayesianStatistics, Model1CrossCheck
from master_thesis_code.exceptions import ParameterOutOfBoundsError

if TYPE_CHECKING:
    from master_thesis_code.callbacks import SimulationCallback
from master_thesis_code.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
)

# logging setup
_ROOT_LOGGER = logging.getLogger()


def main() -> None:
    """
    Run main to start the program.
    """
    from master_thesis_code.plotting import apply_style

    apply_style()

    arguments = Arguments.create()
    _configure_logger(arguments.working_directory, arguments.log_level, arguments.h_value)
    arguments.validate()
    _ROOT_LOGGER.info("---------- STARTING MASTER THESIS CODE ----------")
    start_time = time()

    seed = arguments.seed
    np.random.seed(seed)
    _ROOT_LOGGER.info(f"Random seed: {seed}")
    _write_run_metadata(arguments.working_directory, seed, arguments)

    cosmological_model = Model1CrossCheck()
    galaxy_catalog = GalaxyCatalogueHandler(
        M_min=cosmological_model.parameter_space.M.lower_limit,
        M_max=cosmological_model.parameter_space.M.upper_limit,
        z_max=cosmological_model.max_redshift,
    )

    if arguments.simulation_steps > 0:
        data_simulation(
            arguments.simulation_steps,
            cosmological_model,
            galaxy_catalog,
            arguments.simulation_index,
        )

    if arguments.evaluate:
        evaluate(cosmological_model, galaxy_catalog, arguments.h_value)

    if arguments.snr_analysis:
        snr_analysis()

    if arguments.generate_figures is not None:
        generate_figures(arguments.generate_figures)

    end_time = time()
    _ROOT_LOGGER.debug(f"Finished in {end_time - start_time}s.")


def _get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_run_metadata(working_directory: str, seed: int, arguments: Arguments) -> None:
    metadata = {
        "git_commit": _get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": seed,
        "cli_args": {
            "simulation_steps": arguments.simulation_steps,
            "simulation_index": arguments.simulation_index,
            "evaluate": arguments.evaluate,
            "h_value": arguments.h_value,
            "snr_analysis": arguments.snr_analysis,
        },
    }
    metadata_path = os.path.join(working_directory, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    _ROOT_LOGGER.info(f"Run metadata written to: {metadata_path}")


def _configure_logger(working_directory: str, log_level: int, h_value: float) -> None:
    _ROOT_LOGGER.setLevel(log_level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    _ROOT_LOGGER.addHandler(stream_handler)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(
        working_directory,
        f"master_thesis_code_{timestamp}_h_{str(np.round(h_value, 3)).replace('.', '_')}.log",
    )
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
    )
    file_handler.setFormatter(formatter)
    _ROOT_LOGGER.addHandler(file_handler)

    # set matplotlib logging to warning, because it is very talkative
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    _ROOT_LOGGER.info(f"Log file location: {log_file_path}")


def snr_analysis() -> None:
    from master_thesis_code.datamodels.parameter_space import ParameterSpace
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
        parameter_space=ParameterSpace(),
    )

    parameter_estimation.SNR_analysis()


def data_simulation(
    simulation_steps: int,
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
    simulation_index: int,
    callbacks: list["SimulationCallback"] | None = None,
) -> None:
    # conditional imports because they require GPU
    from master_thesis_code.memory_management import MemoryManagement
    from master_thesis_code.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
        WaveGeneratorType,
    )

    _callbacks: list[SimulationCallback] = callbacks or []

    memory_management = MemoryManagement()
    memory_management.display_GPU_information()
    memory_management.display_fft_cache()

    parameter_estimation = ParameterEstimation(
        waveform_generation_type=WaveGeneratorType.PN5_AAK,
        parameter_space=cosmological_model.parameter_space,
    )

    for cb in _callbacks:
        cb.on_simulation_start(simulation_steps)

    counter = 0
    iteration = 0
    host_galaxies: Iterator[HostGalaxy] = iter([])

    while counter < simulation_steps:
        memory_management.gpu_usage_stamp()
        memory_management.memory_pool.free_all_blocks()
        memory_management.gpu_usage_stamp()

        _ROOT_LOGGER.info(
            f"{counter} / {iteration} evaluations successful. ({counter / (time() - memory_management._start_time) * 60}/min)"
        )
        iteration += 1

        try:
            host_galaxy = next(host_galaxies)
        except StopIteration:
            parameter_samples = cosmological_model.sample_emri_events(200)
            host_galaxies = iter(galaxy_catalog.get_hosts_from_parameter_samples(parameter_samples))
            host_galaxy = next(host_galaxies)
        assert isinstance(host_galaxy, HostGalaxy)

        parameter_estimation.parameter_space.randomize_parameters()

        parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy)

        try:
            warnings.filterwarnings("error")
            quick_snr = parameter_estimation.compute_signal_to_noise_ratio(
                use_snr_check_generator=True
            )

            if quick_snr < cosmological_model.snr_threshold * 0.2:
                _ROOT_LOGGER.info(
                    f"Quick SNR threshold check failed: {np.round(quick_snr, 3)} < {cosmological_model.snr_threshold * 0.2}."
                )
                parameter_estimation.save_not_detected(quick_snr * 5, simulation_index)
                for cb in _callbacks:
                    cb.on_snr_computed(counter, quick_snr * 5, False)
                continue
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

        passed = snr >= cosmological_model.snr_threshold
        for cb in _callbacks:
            cb.on_snr_computed(counter, snr, passed)

        if not passed:
            _ROOT_LOGGER.info(
                f"SNR threshold check failed: {np.round(snr, 3)} < {cosmological_model.snr_threshold}."
            )
            continue

        _ROOT_LOGGER.info(
            f"SNR threshold check successful: {np.round(snr, 3)} >= {cosmological_model.snr_threshold}"
        )
        try:
            cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds()
        except ParameterOutOfBoundsError:
            _ROOT_LOGGER.warning(
                "Caught ParameterOutOfBoundsError in dervative. Continue with new parameters..."
            )
            continue
        parameter_estimation.save_cramer_rao_bound(
            cramer_rao_bound_dictionary=cramer_rao_bounds,
            snr=snr,
            host_galaxy_index=host_galaxy.catalog_index,
            simulation_index=simulation_index,
        )
        counter += 1

        for cb in _callbacks:
            cb.on_detection(counter, snr, cramer_rao_bounds, host_galaxy.catalog_index)

        memory_management.display_GPU_information()
        memory_management.display_fft_cache()

        for cb in _callbacks:
            cb.on_step_end(counter, iteration)

    parameter_estimation.flush_pending_results()

    for cb in _callbacks:
        cb.on_simulation_end(counter, iteration)


def evaluate(
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
    h_value: float,
) -> None:
    # data_simulation = DataEvaluation()
    # data_simulation.evaluate_snr_analysis()

    hubble_constant_evaluation = BayesianStatistics()
    hubble_constant_evaluation.evaluate(galaxy_catalog, cosmological_model, h_value)


def generate_figures(output_dir: str) -> None:
    """Load saved simulation data and produce all thesis figures.

    Called by ``--generate_figures <dir>``.  Factory functions from the
    ``plotting`` subpackage are used; each returns ``(fig, ax)`` and the
    figures are saved to *output_dir*.
    """
    _ROOT_LOGGER.info(f"Generating figures to {output_dir}")
    _ROOT_LOGGER.info(
        "Figure generation is a stub — implement per-figure calls "
        "using plotting.* factory functions as data becomes available."
    )


if __name__ == "__main__":
    main()
