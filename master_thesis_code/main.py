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

    # Fast-path: --combine and --generate_figures don't need heavy initialization
    _needs_model = (
        arguments.simulation_steps > 0
        or arguments.evaluate
        or arguments.snr_analysis
        or arguments.injection_campaign
    )

    if arguments.combine:
        from master_thesis_code.bayesian_inference.posterior_combination import combine_posteriors

        for variant_dir in ["posteriors", "posteriors_with_bh_mass"]:
            posteriors_dir = os.path.join(arguments.working_directory, variant_dir)
            if os.path.isdir(posteriors_dir):
                _ROOT_LOGGER.info(f"Combining posteriors from {posteriors_dir}")
                combine_posteriors(
                    posteriors_dir=posteriors_dir,
                    strategy=arguments.strategy,
                    output_dir=os.path.join(arguments.working_directory, variant_dir),
                )
            else:
                _ROOT_LOGGER.warning(f"Posteriors directory not found: {posteriors_dir}")

    if arguments.generate_figures is not None:
        generate_figures(arguments.generate_figures)

    if arguments.generate_interactive is not None:
        generate_interactive_figures(arguments.generate_interactive)

    if arguments.save_baseline:
        _save_baseline(arguments.working_directory)

    if arguments.compare_baseline is not None:
        _compare_baseline(
            arguments.working_directory,
            arguments.compare_baseline,
            label="catalog_only" if arguments.catalog_only else "current",
        )

    if not _needs_model:
        end_time = time()
        _ROOT_LOGGER.debug(f"Finished in {end_time - start_time}s.")
        return

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
            catalog_only=arguments.catalog_only,
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

    end_time = time()
    _ROOT_LOGGER.debug(f"Finished in {end_time - start_time}s.")


def _save_baseline(working_directory: str) -> None:
    """Extract baseline metrics from existing posteriors and save as baseline.json."""
    from pathlib import Path

    from master_thesis_code.bayesian_inference.evaluation_report import extract_baseline

    posteriors_dir = Path(working_directory) / "posteriors"
    crb_csv = Path(working_directory) / "prepared_cramer_rao_bounds.csv"

    baseline = extract_baseline(
        posteriors_dir=posteriors_dir,
        crb_csv_path=crb_csv if crb_csv.exists() else None,
    )

    import json

    project_root = Path(__file__).resolve().parents[1]
    debug_dir = project_root / ".planning" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    output_path = debug_dir / "baseline.json"
    output_path.write_text(json.dumps(baseline.to_json(), indent=2))
    _ROOT_LOGGER.info(
        "Baseline saved to %s (MAP h=%.4f, bias=%.1f%%)",
        output_path,
        baseline.map_h,
        baseline.bias_percent,
    )


def _compare_baseline(working_directory: str, baseline_path: str, label: str = "current") -> None:
    """Generate comparison report between baseline and current posteriors."""
    import json
    from pathlib import Path

    from master_thesis_code.bayesian_inference.evaluation_report import (
        BaselineSnapshot,
        extract_baseline,
        generate_comparison_report,
    )

    baseline_data = json.loads(Path(baseline_path).read_text())
    baseline = BaselineSnapshot.from_json(baseline_data)

    posteriors_dir = Path(working_directory) / "posteriors"
    crb_csv = Path(working_directory) / "prepared_cramer_rao_bounds.csv"

    current = extract_baseline(
        posteriors_dir=posteriors_dir,
        crb_csv_path=crb_csv if crb_csv.exists() else None,
    )

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / ".planning" / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = generate_comparison_report(baseline, current, output_dir, label=label)
    _ROOT_LOGGER.info("Comparison report written to %s", report_path)

    # Diagnostic summary if event_likelihoods.csv exists (D-07)
    diag_csv = Path(working_directory) / "diagnostics" / "event_likelihoods.csv"
    if diag_csv.exists():
        from master_thesis_code.bayesian_inference.evaluation_report import (
            generate_diagnostic_summary,
        )

        diag_summary = generate_diagnostic_summary(diag_csv, output_dir, label=label)
        frac_low = float(str(diag_summary["frac_L_comp_pulls_low_h"]))
        frac_comb = float(str(diag_summary["mean_L_comp_fraction_of_combined"]))
        _ROOT_LOGGER.info(
            "Diagnostic: mean_f_i=%.4f, L_comp_pulls_low=%.1f%%, L_comp_frac=%.1f%%",
            diag_summary["mean_f_i"],
            frac_low * 100,
            frac_comb * 100,
        )

    print(f"\n{'=' * 60}")
    print(f"  Baseline: MAP h={baseline.map_h:.4f}, bias={baseline.bias_percent:+.1f}%")
    print(f"  Current:  MAP h={current.map_h:.4f}, bias={current.bias_percent:+.1f}%")
    print(
        f"  Delta:    MAP h={current.map_h - baseline.map_h:+.4f}, "
        f"bias={current.bias_percent - baseline.bias_percent:+.1f}%"
    )
    print(f"{'=' * 60}\n")


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
            "combine": arguments.combine,
            "strategy": arguments.strategy,
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

            # SNR scales as √T for stationary sources; EMRIs chirp so the
            # 1-yr / 5-yr ratio can be even lower.  Factor 0.3 is a conservative
            # compromise between the √T bound (0.447) and chirp margin.
            if quick_snr < cosmological_model.snr_threshold * 0.3:
                signal.alarm(0)
                _ROOT_LOGGER.info(
                    f"Quick SNR threshold check failed: {np.round(quick_snr, 3)} < {cosmological_model.snr_threshold * 0.3}."
                )
                for cb in _callbacks:
                    cb.on_snr_computed(counter, quick_snr * np.sqrt(5), False)
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


_INJECTION_COLUMNS = ["z", "M", "phiS", "qS", "SNR", "h_inj", "luminosity_distance"]


def _flush_injection_results(results: list[dict[str, float]], csv_path: str) -> None:
    """Write injection results to CSV (overwrites previous flush)."""
    import pandas as pd

    pd.DataFrame(results, columns=_INJECTION_COLUMNS).to_csv(csv_path, index=False)


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
    _EMCEE_BATCH = 1000  # large batch to amortize MCMC overhead (93.5% z-rejected)
    _LOG_INTERVAL = 100  # log every N successful events
    _GPU_FREE_INTERVAL = 50  # free GPU memory every N waveform computations
    _FLUSH_INTERVAL = 2000  # flush to disk every N events
    _TIMEOUT_S = 30  # SNR-only is fast; 30s is generous

    while counter < simulation_steps:
        # Sample events from population model
        try:
            sample = next(parameter_samples_iter)
        except StopIteration:
            samples_list = cosmological_model.sample_emri_events(_EMCEE_BATCH)
            parameter_samples_iter = iter(samples_list)
            sample = next(parameter_samples_iter)

        # Importance sampling: skip events beyond the detection horizon.
        # All 24/69500 detections in the initial campaign were at z < 0.18.
        # Events at z > z_cut have P_det ≈ 0 and waste GPU time on waveforms
        # that will never produce detectable SNR.
        if sample.redshift > z_cut:
            skipped_high_z += 1
            continue

        if iteration % _GPU_FREE_INTERVAL == 0:
            memory_management.gpu_usage_stamp()
            memory_management.free_gpu_memory()
        iteration += 1

        if counter % _LOG_INTERVAL == 0:
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
            signal.alarm(_TIMEOUT_S)
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
                _ROOT_LOGGER.warning("Caught EllipticK error from waveform generator. Continue...")
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

        # Flush to disk periodically so SLURM timeouts don't lose all work
        if counter % _FLUSH_INTERVAL == 0:
            _flush_injection_results(results, csv_path)
            _ROOT_LOGGER.info(f"Flushed {len(results)} events to {csv_path}")

    # Final write
    _flush_injection_results(results, csv_path)
    _ROOT_LOGGER.info(f"Injection campaign complete: {len(results)} events stored to {csv_path}")


def evaluate(
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
    h_value: float,
    *,
    num_workers: int | None = None,
    catalog_only: bool = False,
) -> None:
    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

    hubble_constant_evaluation = BayesianStatistics()
    hubble_constant_evaluation.evaluate(
        galaxy_catalog,
        cosmological_model,
        h_value,
        num_workers=num_workers,
        catalog_only=catalog_only,
    )


_TWO_MB = 2 * 1024 * 1024


def _check_file_size(path: str, name: str) -> None:
    """Log a warning if *path* exceeds 2 MB.

    Parameters
    ----------
    path : str
        File system path to check.
    name : str
        Human-readable name for the log message.
    """
    try:
        size = os.path.getsize(path)
        if size > _TWO_MB:
            _ROOT_LOGGER.warning(
                "%s exceeds 2 MB (%d bytes) -- consider rasterizing dense elements",
                name,
                size,
            )
    except OSError:
        pass


def generate_figures(output_dir: str) -> None:
    """Load saved simulation data and produce all thesis figures.

    Called by ``--generate_figures <dir>``.  Iterates a manifest of
    ``(name, generator)`` tuples.  Each generator returns ``(fig, ax)``
    or ``None`` (when required data is missing).  Figures are saved as
    PDF to ``<output_dir>/figures/``.
    """
    import glob
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd

    from master_thesis_code.plotting._data import PARAMETER_NAMES, reconstruct_covariance
    from master_thesis_code.plotting._helpers import save_figure
    from master_thesis_code.plotting._style import apply_style

    apply_style()
    figures_dir = os.path.join(output_dir, "figures")
    _ROOT_LOGGER.info("Generating figures to %s", figures_dir)

    # ------------------------------------------------------------------
    # Data loading helpers (return None when data is missing)
    # ------------------------------------------------------------------

    def _load_crb_data() -> pd.DataFrame | None:
        """Load and concatenate all CRB CSV files."""
        csv_files = sorted(glob.glob(os.path.join(output_dir, "cramer_rao_bounds*.csv")))
        if not csv_files:
            # Fallback: check simulations/ directory relative to project root
            project_root = Path(__file__).resolve().parents[1]
            fallback = project_root / "simulations" / "cramer_rao_bounds.csv"
            if fallback.is_file():
                csv_files = [str(fallback)]
        if not csv_files:
            return None
        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
        # Derive redshift from luminosity_distance if column is missing
        if "redshift" not in df.columns and "luminosity_distance" in df.columns:
            from master_thesis_code.physical_relations import dist_to_redshift

            df["redshift"] = df["luminosity_distance"].apply(
                lambda d: dist_to_redshift(float(d), h=0.73)
            )
        return df

    def _load_posteriors(
        subdir: str,
    ) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """Load posterior JSONs from *subdir*, return (h_values, event_posteriors)."""
        from master_thesis_code.bayesian_inference.posterior_combination import (
            load_posterior_jsons,
        )

        posteriors_dir = Path(output_dir) / subdir
        if not posteriors_dir.is_dir():
            return None
        try:
            h_values_list, event_likelihoods = load_posterior_jsons(posteriors_dir)
            h_values = np.array(h_values_list, dtype=np.float64)
            event_posteriors: list[np.ndarray] = []
            for event_idx in sorted(event_likelihoods.keys()):
                lh = event_likelihoods[event_idx]
                event_posteriors.append(
                    np.array([lh.get(h, 0.0) for h in h_values_list], dtype=np.float64)
                )
            return h_values, event_posteriors
        except (FileNotFoundError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Save helper with size check
    # ------------------------------------------------------------------

    def _save(fig: object, name: str) -> None:
        """Save figure as PDF and check size."""
        path = os.path.join(figures_dir, name)
        save_figure(fig, path, formats=("pdf",))  # type: ignore[arg-type]
        _check_file_size(f"{path}.pdf", name)

    # ------------------------------------------------------------------
    # Pre-load shared data
    # ------------------------------------------------------------------

    crb_df = _load_crb_data()
    post_data = _load_posteriors("posteriors")
    post_data_with = _load_posteriors("posteriors_with_bh_mass")

    # ------------------------------------------------------------------
    # Manifest: list of (output_name, generator_callable)
    # Per D-06: Python list of tuples, not YAML config.
    # Per D-11: Full set of thesis-relevant figures (15 entries).
    # ------------------------------------------------------------------

    manifest: list[tuple[str, Callable[[], tuple[object, object] | None]]] = []

    # 1. H0 posterior (combined) -- needs posterior data
    def _gen_h0_posterior_combined() -> tuple[object, object] | None:
        if post_data is None:
            return None
        from master_thesis_code.plotting._colors import VARIANT_NO_MASS, VARIANT_WITH_MASS
        from master_thesis_code.plotting.bayesian_plots import plot_combined_posterior

        h_vals, event_posts = post_data
        log_posts = [np.log(np.maximum(p, 1e-300)) for p in event_posts]
        log_combined = np.sum(log_posts, axis=0)
        log_combined -= log_combined.max()
        combined = np.exp(log_combined)
        fig, ax = plot_combined_posterior(
            h_vals,
            combined,
            0.73,
            label=r"Without $M_z$",
            color=VARIANT_NO_MASS,
        )
        if post_data_with is not None:
            h_w, ep_w = post_data_with
            log_w = [np.log(np.maximum(p, 1e-300)) for p in ep_w]
            log_comb_w = np.sum(log_w, axis=0)
            log_comb_w -= log_comb_w.max()
            comb_w = np.exp(log_comb_w)
            plot_combined_posterior(
                h_w,
                comb_w,
                0.73,
                label=r"With $M_z$",
                color=VARIANT_WITH_MASS,
                show_references=False,
                ax=ax,
            )
        return fig, ax

    manifest.append(("fig01_h0_posterior_combined", _gen_h0_posterior_combined))

    # 2. Individual event posteriors
    def _gen_event_posteriors() -> tuple[object, object] | None:
        if post_data is None:
            return None
        from master_thesis_code.plotting._colors import VARIANT_WITH_MASS
        from master_thesis_code.plotting.bayesian_plots import (
            plot_event_posteriors,
        )

        h_vals, event_posts = post_data
        log_posts = [np.log(np.maximum(p, 1e-300)) for p in event_posts]
        log_combined = np.sum(log_posts, axis=0)
        log_combined -= log_combined.max()
        combined = np.exp(log_combined)
        fig, ax = plot_event_posteriors(h_vals, event_posts, 0.73, combined_posterior=combined)
        # Overlay with-M_z combined posterior
        if post_data_with is not None:
            h_w, ep_w = post_data_with
            log_w = [np.log(np.maximum(p, 1e-300)) for p in ep_w]
            log_comb_w = np.sum(log_w, axis=0)
            log_comb_w -= log_comb_w.max()
            comb_w = np.exp(log_comb_w)
            norm_w = np.trapezoid(comb_w, h_w)
            if norm_w > 0:
                comb_w /= norm_w
            ax.plot(
                h_w, comb_w, color=VARIANT_WITH_MASS, linewidth=2, label=r"Combined (with $M_z$)"
            )
            ax.legend(fontsize="small")
        return fig, ax

    manifest.append(("fig02_event_posteriors", _gen_event_posteriors))

    # 3. SNR distribution -- needs CRB data with SNR column
    def _gen_snr_distribution() -> tuple[object, object] | None:
        if crb_df is None or "SNR" not in crb_df.columns:
            return None
        from master_thesis_code.plotting.bayesian_plots import plot_snr_distribution

        return plot_snr_distribution(crb_df["SNR"].to_numpy(dtype=np.float64))

    manifest.append(("fig03_snr_distribution", _gen_snr_distribution))

    # 4. Detection yield -- needs redshift column in CRB
    def _gen_detection_yield() -> tuple[object, object] | None:
        if crb_df is None or "redshift" not in crb_df.columns:
            return None
        from master_thesis_code.plotting.simulation_plots import plot_detection_yield

        detected_z = crb_df["redshift"].to_numpy(dtype=np.float64)
        return plot_detection_yield(detected_z, detected_z)

    manifest.append(("fig04_detection_yield", _gen_detection_yield))

    # 5. Sky localization (Mollweide)
    def _gen_sky_localization() -> tuple[object, object] | None:
        if crb_df is None or not {"qS", "phiS", "SNR"}.issubset(crb_df.columns):
            return None
        from master_thesis_code.plotting.sky_plots import plot_sky_localization_mollweide

        theta_s = crb_df["qS"].to_numpy(dtype=np.float64)
        phi_s = crb_df["phiS"].to_numpy(dtype=np.float64)
        snr = crb_df["SNR"].to_numpy(dtype=np.float64)
        return plot_sky_localization_mollweide(theta_s, phi_s, snr)

    manifest.append(("fig05_sky_localization", _gen_sky_localization))

    # 6. Fisher ellipses (3 parameter pairs)
    def _gen_fisher_ellipses() -> tuple[object, object] | None:
        if crb_df is None or len(crb_df) < 1:
            return None
        from master_thesis_code.plotting.fisher_plots import plot_fisher_ellipses

        row = crb_df.iloc[0]
        cov = reconstruct_covariance(row)
        param_vals = np.array([float(row.get(p, 0.0)) for p in PARAMETER_NAMES], dtype=np.float64)
        return plot_fisher_ellipses(cov, param_vals)

    manifest.append(("fig06_fisher_ellipses", _gen_fisher_ellipses))

    # 7. Corner plot
    def _gen_corner_plot() -> tuple[object, object] | None:
        if crb_df is None or len(crb_df) < 1:
            return None
        from master_thesis_code.plotting.fisher_plots import plot_fisher_corner

        row = crb_df.iloc[0]
        cov = reconstruct_covariance(row)
        param_vals = np.array([float(row.get(p, 0.0)) for p in PARAMETER_NAMES], dtype=np.float64)
        return plot_fisher_corner(cov, param_vals)

    manifest.append(("fig07_corner_plot", _gen_corner_plot))

    # 8. H0 convergence
    def _gen_h0_convergence() -> tuple[object, object] | None:
        if post_data is None:
            return None
        from master_thesis_code.plotting.convergence_plots import plot_h0_convergence

        h_vals, event_posts = post_data
        h_alt, ep_alt = post_data_with if post_data_with is not None else (None, None)
        return plot_h0_convergence(
            h_vals,
            event_posts,
            true_h=0.73,
            h_values_alt=h_alt,
            event_posteriors_alt=ep_alt,
        )

    manifest.append(("fig08_h0_convergence", _gen_h0_convergence))

    # 9. Detection efficiency
    def _gen_detection_efficiency() -> tuple[object, object] | None:
        if crb_df is None or "redshift" not in crb_df.columns or "SNR" not in crb_df.columns:
            return None
        from master_thesis_code.plotting.convergence_plots import plot_detection_efficiency

        z = crb_df["redshift"].to_numpy(dtype=np.float64)
        snr = crb_df["SNR"].to_numpy(dtype=np.float64)
        detected = snr >= 20.0
        return plot_detection_efficiency(z, detected)

    manifest.append(("fig09_detection_efficiency", _gen_detection_efficiency))

    # 10. LISA PSD with noise decomposition
    def _gen_lisa_psd() -> tuple[object, object] | None:
        from master_thesis_code.plotting.simulation_plots import plot_lisa_psd

        freqs = np.geomspace(1e-5, 1.0, 1000)
        return plot_lisa_psd(freqs, decompose=True)

    manifest.append(("fig10_lisa_psd", _gen_lisa_psd))

    # 11. Luminosity distance d_L(z) with multi-H0
    def _gen_distance_redshift() -> tuple[object, object] | None:
        from master_thesis_code.physical_relations import dist_vectorized
        from master_thesis_code.plotting.physical_relations_plots import (
            plot_distance_redshift,
        )

        z = np.linspace(0.01, 3.0, 200)
        d = dist_vectorized(z, 0.73)
        return plot_distance_redshift(
            z,
            d,  # type: ignore[arg-type]  # np.floating[Any] <: np.float64 at runtime
            h0_values=[0.67, 0.70, 0.73, 0.76],
            distance_fn=dist_vectorized,  # type: ignore[arg-type]  # same floating variance
        )

    manifest.append(("fig11_distance_redshift", _gen_distance_redshift))

    # 12. Parameter uncertainty violins
    def _gen_uncertainty_violins() -> tuple[object, object] | None:
        if crb_df is None or len(crb_df) < 10:
            return None
        from master_thesis_code.plotting.fisher_plots import plot_parameter_uncertainties

        param_cols = [p for p in PARAMETER_NAMES if p in crb_df.columns]
        if not param_cols:
            return None
        return plot_parameter_uncertainties(crb_df, crb_df[param_cols])

    manifest.append(("fig12_uncertainty_violins", _gen_uncertainty_violins))

    # 13. Characteristic strain
    def _gen_characteristic_strain() -> tuple[object, object] | None:
        from master_thesis_code.plotting.fisher_plots import plot_characteristic_strain

        return plot_characteristic_strain()

    manifest.append(("fig13_characteristic_strain", _gen_characteristic_strain))

    # 14. CRB coverage (3D parameter-space scatter per D-11)
    def _gen_crb_coverage() -> tuple[object, object] | None:
        if crb_df is None or not {"M", "qS", "phiS"}.issubset(crb_df.columns):
            return None
        from master_thesis_code.plotting.simulation_plots import plot_cramer_rao_coverage

        M = crb_df["M"].to_numpy(dtype=np.float64)
        qS = crb_df["qS"].to_numpy(dtype=np.float64)
        phiS = crb_df["phiS"].to_numpy(dtype=np.float64)
        return plot_cramer_rao_coverage(
            M,
            qS,
            phiS,
            M_limits=(float(M.min()), float(M.max())),
            qS_limits=(float(qS.min()), float(qS.max())),
            phiS_limits=(float(phiS.min()), float(phiS.max())),
        )

    manifest.append(("fig14_crb_coverage", _gen_crb_coverage))

    # 15. Campaign dashboard (composite)
    def _gen_dashboard() -> tuple[object, object] | None:
        if crb_df is None or post_data is None:
            return None
        if not {"qS", "phiS", "SNR", "redshift"}.issubset(crb_df.columns):
            return None
        from master_thesis_code.plotting.dashboard_plots import plot_campaign_dashboard

        h_vals, event_posts = post_data
        log_posts = [np.log(np.maximum(p, 1e-300)) for p in event_posts]
        log_combined = np.sum(log_posts, axis=0)
        log_combined -= log_combined.max()
        combined = np.exp(log_combined)
        return plot_campaign_dashboard(
            h_values=h_vals,
            posterior=combined,
            true_h=0.73,
            snr_values=crb_df["SNR"].to_numpy(dtype=np.float64),
            injected_redshifts=crb_df["redshift"].to_numpy(dtype=np.float64),
            detected_redshifts=crb_df["redshift"].to_numpy(dtype=np.float64),
            theta_s=crb_df["qS"].to_numpy(dtype=np.float64),
            phi_s=crb_df["phiS"].to_numpy(dtype=np.float64),
            sky_snr=crb_df["SNR"].to_numpy(dtype=np.float64),
        )

    manifest.append(("fig15_campaign_dashboard", _gen_dashboard))

    # 16. Paper figure: H0 posterior comparison (D-01, D-09)
    def _gen_paper_h0_posterior() -> tuple[object, object] | None:
        from master_thesis_code.plotting.paper_figures import plot_h0_posterior_comparison

        try:
            return plot_h0_posterior_comparison(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError):
            return None

    manifest.append(("paper_h0_posterior", _gen_paper_h0_posterior))

    # 17. Paper figure: single-event likelihoods
    def _gen_paper_single_event() -> tuple[object, object] | None:
        from master_thesis_code.plotting.paper_figures import plot_single_event_likelihoods

        try:
            return plot_single_event_likelihoods(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError, ValueError):
            return None

    manifest.append(("paper_single_event", _gen_paper_single_event))

    # 18. Paper figure: posterior convergence
    def _gen_paper_convergence() -> tuple[object, object] | None:
        from master_thesis_code.plotting.paper_figures import plot_posterior_convergence

        try:
            return plot_posterior_convergence(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError):
            return None

    manifest.append(("paper_convergence", _gen_paper_convergence))

    # 19. Paper figure: SNR distribution
    def _gen_paper_snr_distribution() -> tuple[object, object] | None:
        from master_thesis_code.plotting.paper_figures import plot_snr_distribution

        try:
            return plot_snr_distribution(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError, ValueError):
            return None

    manifest.append(("paper_snr_distribution", _gen_paper_snr_distribution))

    # 20. Paper figure: KDE-smoothed H0 posterior (D-05)
    def _gen_paper_h0_posterior_kde() -> tuple[object, object] | None:
        from master_thesis_code.plotting.paper_figures import plot_h0_posterior_kde

        try:
            return plot_h0_posterior_kde(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError):
            return None

    manifest.append(("paper_h0_posterior_kde", _gen_paper_h0_posterior_kde))

    # ------------------------------------------------------------------
    # Execute manifest
    # ------------------------------------------------------------------
    generated = 0
    skipped = 0
    failed = 0
    for name, generator in manifest:
        try:
            result = generator()
            if result is None:
                _ROOT_LOGGER.warning("Skipping %s: required data not found", name)
                skipped += 1
                continue
            fig = result[0]  # (fig, ax) or (fig, dict)
            _save(fig, name)
            generated += 1
        except Exception:
            _ROOT_LOGGER.warning("Failed to generate %s", name, exc_info=True)
            failed += 1

    _ROOT_LOGGER.info(
        "Figure generation complete: %d generated, %d skipped, %d failed",
        generated,
        skipped,
        failed,
    )


def generate_interactive_figures(data_dir: str) -> None:
    """Load saved simulation data and produce interactive Plotly HTML figures.

    Called by ``--generate_interactive <dir>``.  Writes HTML files to
    ``<data_dir>/interactive/`` and logs the paths of written files.

    Parameters
    ----------
    data_dir:
        Working directory containing CRB CSVs and posterior JSON subdirectories.
        HTML output is written to ``<data_dir>/interactive/``.
    """
    from master_thesis_code.plotting.interactive import generate_all_interactive

    output_dir = os.path.join(data_dir, "interactive")
    _ROOT_LOGGER.info("Generating interactive figures to %s", output_dir)
    written = generate_all_interactive(output_dir=output_dir, data_dir=data_dir)
    if written:
        _ROOT_LOGGER.info(
            "Interactive figure generation complete: %d file(s) written", len(written)
        )
        for path in written:
            _ROOT_LOGGER.info("  Written: %s", path)
    else:
        _ROOT_LOGGER.info(
            "Interactive figure generation complete: no files written (data not found)"
        )


if __name__ == "__main__":
    main()
