import datetime
import json
import logging
import os
import signal
import subprocess
import warnings
from collections.abc import Iterator
from time import time
from typing import TYPE_CHECKING

import numpy as np

from master_thesis_code.arguments import Arguments
from master_thesis_code.cosmological_model import Model1CrossCheck
from master_thesis_code.exceptions import ParameterEstimationError, ParameterOutOfBoundsError

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
    rng = np.random.default_rng(seed)
    _ROOT_LOGGER.info(f"Random seed: {seed}")
    _write_run_metadata(arguments.working_directory, seed, arguments)

    cosmological_model = Model1CrossCheck(rng=rng)
    galaxy_catalog = GalaxyCatalogueHandler(
        M_min=cosmological_model.parameter_space.M.lower_limit,
        M_max=cosmological_model.parameter_space.M.upper_limit,
        z_max=cosmological_model.max_redshift,
    )

    if arguments.simulation_steps > 0 and not arguments.injection_campaign:
        data_simulation(
            arguments.simulation_steps,
            cosmological_model,
            galaxy_catalog,
            arguments.simulation_index,
            rng=rng,
            use_gpu=arguments.use_gpu,
        )

    if arguments.evaluate:
        evaluate(
            cosmological_model,
            galaxy_catalog,
            arguments.h_value,
            num_workers=arguments.num_workers,
        )

    if arguments.snr_analysis:
        snr_analysis(use_gpu=arguments.use_gpu)

    if arguments.injection_campaign:
        injection_campaign(
            simulation_steps=arguments.simulation_steps,
            cosmological_model=cosmological_model,
            h_value=arguments.h_value,
            simulation_index=arguments.simulation_index,
            rng=rng,
            use_gpu=arguments.use_gpu,
        )

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
            "use_gpu": arguments.use_gpu,
            "num_workers": arguments.num_workers,
        },
    }
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
        "HOSTNAME",
    ]
    slurm_info = {var: os.environ[var] for var in slurm_vars if var in os.environ}
    if slurm_info:
        metadata["slurm"] = slurm_info

    index = arguments.simulation_index
    if index > 0 or "SLURM_ARRAY_TASK_ID" in os.environ:
        filename = f"run_metadata_{index}.json"
    else:
        filename = "run_metadata.json"
    metadata_path = os.path.join(working_directory, filename)
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


def snr_analysis(*, use_gpu: bool = False) -> None:
    from master_thesis_code.datamodels.parameter_space import ParameterSpace
    from master_thesis_code.memory_management import MemoryManagement
    from master_thesis_code.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
        WaveGeneratorType,
    )

    memory_management = MemoryManagement(use_gpu=use_gpu)
    memory_management.display_GPU_information()
    memory_management.display_fft_cache()

    parameter_estimation = ParameterEstimation(
        waveform_generation_type=WaveGeneratorType.PN5_AAK,
        parameter_space=ParameterSpace(),
        use_gpu=use_gpu,
    )

    parameter_estimation.SNR_analysis()


def data_simulation(
    simulation_steps: int,
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
    simulation_index: int,
    callbacks: list["SimulationCallback"] | None = None,
    rng: np.random.Generator | None = None,
    *,
    use_gpu: bool = False,
) -> None:
    # conditional imports because they require GPU
    from master_thesis_code.memory_management import MemoryManagement
    from master_thesis_code.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
        WaveGeneratorType,
    )

    _callbacks: list[SimulationCallback] = callbacks or []

    def _alarm_handler(signum: int, frame: object) -> None:
        raise TimeoutError("Computation exceeded 90s timeout")

    signal.signal(signal.SIGALRM, _alarm_handler)

    # Flush buffered results on SLURM timeout (SIGTERM) before the process is killed.
    _pe_ref: list[ParameterEstimation] = []

    def _sigterm_handler(signum: int, frame: object) -> None:
        if _pe_ref:
            _ROOT_LOGGER.warning("SIGTERM received — flushing buffered Cramér-Rao bounds...")
            _pe_ref[0].flush_pending_results()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    memory_management = MemoryManagement(use_gpu=use_gpu)
    memory_management.display_GPU_information()
    memory_management.display_fft_cache()

    parameter_estimation = ParameterEstimation(
        waveform_generation_type=WaveGeneratorType.PN5_AAK,
        parameter_space=cosmological_model.parameter_space,
        use_gpu=use_gpu,
    )
    _pe_ref.append(parameter_estimation)

    for cb in _callbacks:
        cb.on_simulation_start(simulation_steps)

    counter = 0
    iteration = 0
    host_galaxies: Iterator[HostGalaxy] = iter([])

    while counter < simulation_steps:
        memory_management.gpu_usage_stamp()
        memory_management.free_gpu_memory()
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

        parameter_estimation.parameter_space.randomize_parameters(rng=rng)

        parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy)

        try:
            warnings.filterwarnings("error")
            signal.alarm(90)
            quick_snr = parameter_estimation.compute_signal_to_noise_ratio(
                use_snr_check_generator=True
            )

            if quick_snr < cosmological_model.snr_threshold * 0.2:
                signal.alarm(0)
                _ROOT_LOGGER.info(
                    f"Quick SNR threshold check failed: {np.round(quick_snr, 3)} < {cosmological_model.snr_threshold * 0.2}."
                )
                for cb in _callbacks:
                    cb.on_snr_computed(counter, quick_snr * 5, False)
                continue
            snr = parameter_estimation.compute_signal_to_noise_ratio()
            signal.alarm(0)
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
        except ZeroDivisionError:
            _ROOT_LOGGER.warning(
                "Caught ZeroDivisionError during trajectory integration. Continue with new parameters..."
            )
            continue
        except TimeoutError:
            _ROOT_LOGGER.warning("Waveform/SNR computation timed out (>90s). Skipping event...")
            continue

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
            signal.alarm(90)
            cramer_rao_bounds = parameter_estimation.compute_Cramer_Rao_bounds()
            signal.alarm(0)
        except ParameterOutOfBoundsError:
            _ROOT_LOGGER.warning(
                "Caught ParameterOutOfBoundsError in dervative. Continue with new parameters..."
            )
            continue
        except np.linalg.LinAlgError:
            _ROOT_LOGGER.warning("Fisher matrix is singular (LinAlgError). Skipping event...")
            continue
        except ParameterEstimationError as e:
            _ROOT_LOGGER.warning(f"CRB computation failed: {e}. Skipping event...")
            continue
        except TimeoutError:
            _ROOT_LOGGER.warning("Cramér-Rao bound computation timed out (>90s). Skipping event...")
            continue
        except (ZeroDivisionError, RuntimeError, ValueError) as e:
            _ROOT_LOGGER.warning(
                f"Caught {type(e).__name__} during CRB computation: {e}. Skipping event..."
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


def injection_campaign(
    simulation_steps: int,
    cosmological_model: Model1CrossCheck,
    h_value: float,
    simulation_index: int,
    rng: np.random.Generator | None = None,
    *,
    use_gpu: bool = False,
) -> None:
    """Run SNR-only injection campaign for detection probability estimation.

    Draws EMRI events from the population model, computes SNR (no Fisher matrix),
    and stores ALL events (detected and undetected) to a per-task CSV file.

    Args:
        simulation_steps: Number of successful SNR computations to accumulate.
        cosmological_model: Model1CrossCheck instance for event sampling.
        h_value: Hubble constant value used for luminosity distance computation.
        simulation_index: Task index for unique CSV file naming (SLURM array compatibility).
        rng: Random number generator for reproducibility.
        use_gpu: Whether to use GPU acceleration.
    """
    import pandas as pd

    from master_thesis_code.constants import INJECTION_CSV_PATH
    from master_thesis_code.galaxy_catalogue.handler import ParameterSample
    from master_thesis_code.memory_management import MemoryManagement
    from master_thesis_code.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
        WaveGeneratorType,
    )
    from master_thesis_code.physical_relations import dist

    def _alarm_handler(signum: int, frame: object) -> None:
        raise TimeoutError("Computation exceeded 90s timeout")

    signal.signal(signal.SIGALRM, _alarm_handler)

    memory_management = MemoryManagement(use_gpu=use_gpu)
    memory_management.display_GPU_information()

    parameter_estimation = ParameterEstimation(
        waveform_generation_type=WaveGeneratorType.PN5_AAK,
        parameter_space=cosmological_model.parameter_space,
        use_gpu=use_gpu,
    )

    # Resolve CSV path: replace {h_label} and {index} placeholders
    h_label = str(round(h_value, 3)).replace(".", "p")
    csv_path = INJECTION_CSV_PATH.format(h_label=h_label, index=simulation_index)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    _ROOT_LOGGER.info(
        f"Starting injection campaign: h={h_value}, steps={simulation_steps}, "
        f"index={simulation_index}, output={csv_path}"
    )

    results: list[dict[str, float]] = []
    counter = 0
    iteration = 0
    parameter_samples_iter: Iterator[ParameterSample] = iter([])

    z_cut = 0.5  # generous margin above max observed detection z ≈ 0.18
    skipped_high_z = 0

    while counter < simulation_steps:
        # Sample events from population model (batch of 200)
        try:
            sample = next(parameter_samples_iter)
        except StopIteration:
            samples_list = cosmological_model.sample_emri_events(200)
            parameter_samples_iter = iter(samples_list)
            sample = next(parameter_samples_iter)

        # Importance sampling: skip events beyond the detection horizon.
        # All 24/69500 detections in the initial campaign were at z < 0.18.
        # Events at z > z_cut have P_det ≈ 0 and waste GPU time on waveforms
        # that will never produce detectable SNR.  Truncating here does not
        # bias P_det because the numerator (detected) and denominator (total)
        # are both zero in the truncated region.
        if sample.redshift > z_cut:
            skipped_high_z += 1
            continue

        memory_management.gpu_usage_stamp()
        memory_management.free_gpu_memory()
        iteration += 1

        _ROOT_LOGGER.info(
            f"Injection campaign: {counter} / {iteration} successful SNR computations "
            f"({skipped_high_z} high-z skipped)."
        )

        # Randomize extrinsic parameters (sky angles, orbital phases, etc.)
        parameter_estimation.parameter_space.randomize_parameters(rng=rng)

        # Set M from population model sample
        parameter_estimation.parameter_space.M.value = sample.M

        # CRITICAL per D-04: Set luminosity distance with candidate h value
        # (NOT set_host_galaxy_parameters which hardcodes h=0.73)
        luminosity_distance = dist(sample.redshift, h=h_value)
        parameter_estimation.parameter_space.luminosity_distance.value = luminosity_distance

        # Compute SNR only (no Fisher matrix, no CRB)
        try:
            warnings.filterwarnings("error")
            signal.alarm(90)
            snr = parameter_estimation.compute_signal_to_noise_ratio()
            signal.alarm(0)
            warnings.resetwarnings()
        except Warning as e:
            signal.alarm(0)
            if "Mass ratio" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught warning that mass ratio is out of bounds. Continue with new parameters..."
                )
                continue
            else:
                _ROOT_LOGGER.warning(f"{str(e)}. Continue with new parameters...")
                continue
        except ParameterOutOfBoundsError as e:
            signal.alarm(0)
            _ROOT_LOGGER.warning(
                f"Caught ParameterOutOfBoundsError: {str(e)}. Continue with new parameters..."
            )
            continue
        except RuntimeError as e:
            signal.alarm(0)
            _ROOT_LOGGER.warning(
                f"Caught RuntimeError during waveform generation: {str(e)}. Continue..."
            )
            continue
        except ValueError as e:
            signal.alarm(0)
            if "EllipticK" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught EllipticK error from waveform generator. Continue..."
                )
                continue
            elif "Brent root solver does not converge" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught Brent root solver error. Continue with new parameters..."
                )
                continue
            else:
                raise
        except ZeroDivisionError:
            signal.alarm(0)
            _ROOT_LOGGER.warning(
                "Caught ZeroDivisionError during trajectory integration. Continue..."
            )
            continue
        except TimeoutError:
            _ROOT_LOGGER.warning("Waveform/SNR computation timed out (>90s). Skipping event...")
            continue

        # Store ALL events regardless of SNR (per D-03: do NOT threshold)
        results.append(
            {
                "z": sample.redshift,
                "M": sample.M,
                "phiS": parameter_estimation.parameter_space.phiS.value,
                "qS": parameter_estimation.parameter_space.qS.value,
                "SNR": float(snr),
                "h_inj": h_value,
                "luminosity_distance": luminosity_distance,
            }
        )
        counter += 1

    # Write results to CSV
    df = pd.DataFrame(
        results, columns=["z", "M", "phiS", "qS", "SNR", "h_inj", "luminosity_distance"]
    )
    df.to_csv(csv_path, index=False)
    _ROOT_LOGGER.info(f"Injection campaign complete: {len(results)} events stored to {csv_path}")


def evaluate(
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
    h_value: float,
    *,
    num_workers: int | None = None,
) -> None:
    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

    hubble_constant_evaluation = BayesianStatistics()
    hubble_constant_evaluation.evaluate(
        galaxy_catalog, cosmological_model, h_value, num_workers=num_workers
    )


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
