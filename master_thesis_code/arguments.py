import argparse
import logging
import os
import random
import sys

from master_thesis_code.constants import H
from master_thesis_code.exceptions import ArgumentsError

_LOGGER = logging.getLogger()
_VALID_LOG_LEVELS = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]


class Arguments:
    """this class includes the parsed input arguments."""

    def __init__(self, parsed_arguments: argparse.Namespace):
        self._parsed_arguments = parsed_arguments
        self._working_directory_replaced: bool = False
        self._log_level_replaced: bool = False
        self._working_directory: str = parsed_arguments.working_directory
        if not os.path.isdir(self._working_directory):
            self._working_directory_replaced = True
            self._working_directory = os.getcwd()

        if parsed_arguments.log_level.upper() in _VALID_LOG_LEVELS:
            self._log_level: int = getattr(logging, parsed_arguments.log_level.upper())
        else:
            self._log_level_replaced = True
            self._log_level = logging.INFO

    @property
    def working_directory(self) -> str:
        """Path to the working directory, where temporary files are stored, default is the current working directory."""
        return self._working_directory

    @property
    def log_level(self) -> int:
        """Log level of the stream and file logger, default is log level 'INFO'."""
        return self._log_level

    @property
    def simulation_steps(self) -> int:
        """Number of waveforms generated in the simulation."""
        return int(self._parsed_arguments.simulation_steps)

    @property
    def simulation_index(self) -> int:
        """Index for unique file name where cramer rao bounds are saved."""
        return int(self._parsed_arguments.simulation_index)

    @property
    def evaluate(self) -> bool:
        """Indicates whether the gathered Rao-Cramer-bounds are evaluated or not."""
        return bool(self._parsed_arguments.evaluate)

    @property
    def h_value(self) -> float:
        """Hubble constant value."""
        return float(self._parsed_arguments.h_value)

    @property
    def snr_analysis(self) -> bool:
        """Indicates whether the snr analysis should be run."""
        return bool(self._parsed_arguments.snr_analysis)

    @property
    def injection_campaign(self) -> bool:
        """Indicates whether to run SNR-only injection campaign for detection probability estimation."""
        return bool(self._parsed_arguments.injection_campaign)

    @property
    def generate_figures(self) -> str | None:
        """Output directory for figure generation. None means do not generate figures."""
        val: str | None = self._parsed_arguments.generate_figures
        return val

    @property
    def generate_interactive(self) -> str | None:
        """Output directory for interactive Plotly figure generation. None means skip."""
        val: str | None = self._parsed_arguments.generate_interactive
        return val

    @property
    def use_gpu(self) -> bool:
        """Whether to use GPU acceleration."""
        return bool(self._parsed_arguments.use_gpu)

    @property
    def num_workers(self) -> int:
        """Number of multiprocessing workers for Bayesian inference."""
        raw: int | None = self._parsed_arguments.num_workers
        if raw is not None:
            return max(1, raw)
        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:
            available = os.cpu_count() or 1
        return max(1, available - 2)

    @property
    def seed(self) -> int:
        """Random seed for reproducibility. A random seed is chosen if not provided."""
        raw = self._parsed_arguments.seed
        if raw is None:
            return random.randint(0, 2**31 - 1)
        return int(raw)

    @property
    def combine(self) -> bool:
        """Indicates whether to combine per-event posteriors into joint H0 posterior."""
        return bool(self._parsed_arguments.combine)

    @property
    def strategy(self) -> str:
        """Zero-handling strategy for posterior combination."""
        return str(self._parsed_arguments.strategy)

    @staticmethod
    def create(sys_args: list[str] = sys.argv[1:]) -> "Arguments":
        parsed_arguments = _parse_arguments(sys_args)
        return Arguments(parsed_arguments=parsed_arguments)

    def validate(self) -> None:
        """Validate the parsed arguments."""
        if self._working_directory_replaced is True:
            _LOGGER.warning(
                f"The path to the provided working directory does not exist. It is replaced by "
                f"{self._working_directory}."
            )
        if self._log_level_replaced is True:
            _LOGGER.warning(
                f"The provided log level is not valid. Valid values are: {', '.join(_VALID_LOG_LEVELS)}."
                f"The log level is set to {logging.getLevelName(self._log_level)}"
            )

        try:
            self._simulation_steps = int(self._parsed_arguments.simulation_steps)
        except ValueError as original_error:
            raise ArgumentsError(
                f"{self._parsed_arguments.simulation_steps} could not be converted to integer."
                "Please provide an integer value as follows '--simulation_steps <int>'."
            ) from original_error


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "working_directory",
        help="Path to the working directory, where temporary files are stored.",
    )
    parser.add_argument(
        "--simulation_steps",
        help="Number of waveforms that are generated for data evaluation. (default is 0)",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--simulation_index",
        help="Index for unique file name where cramer rao bounds are saved. (default is 0)",
        default=0,
        type=int,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--h_value", help="Hubble constant value.", type=float, default=H)
    parser.add_argument("--snr_analysis", action="store_true")
    parser.add_argument(
        "--injection_campaign",
        action="store_true",
        default=False,
        help="Run SNR-only injection campaign for detection probability estimation.",
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility. If omitted, a random seed is chosen and logged.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--generate_figures",
        help="Output directory for generating all thesis figures from saved data.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration (requires CUDA and cupy). Default: CPU only.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of multiprocessing workers for Bayesian inference. "
        "Default: available CPUs - 2 (minimum 1).",
    )
    parser.add_argument(
        "--log_level",
        nargs="?",
        default="INFO",
        help="Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'). Default is 'INFO'.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        default=False,
        help="Combine per-event posteriors into joint H0 posterior.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="physics-floor",
        choices=["naive", "exclude", "per-event-floor", "physics-floor"],
        help="Zero-handling strategy for posterior combination. Default: physics-floor (falls back to exclude until Phase 22).",
    )
    parser.add_argument(
        "--generate_interactive",
        help="Output directory for generating interactive Plotly HTML figures.",
        type=str,
        default=None,
    )
    parsed_arguments: argparse.Namespace = parser.parse_args(arguments)
    return parsed_arguments
