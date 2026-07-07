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
        self._resolved_seed: int | None = None
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
        """Random seed for reproducibility. A random seed is chosen ONCE if not
        provided, then cached so repeated access (e.g. the combine-metadata path)
        returns the same value rather than a fresh draw (review REP-03)."""
        if self._resolved_seed is None:
            raw = self._parsed_arguments.seed
            self._resolved_seed = random.randint(0, 2**31 - 1) if raw is None else int(raw)
        return self._resolved_seed

    def to_dict(self) -> dict[str, object]:
        """Full parsed-argument namespace as a JSON-serialisable dict.

        Serialising the whole namespace means ``run_metadata`` captures EVERY flag —
        including the inference-critical ones (``normalization_mode``, ``pdet_*``,
        ``catalog_only``, ``fisher_cond_threshold``, ``allow_low_pdet_coverage``) that
        a hand-maintained key list silently omitted (review REP-02). ``seed`` here is
        the RAW CLI value (``None`` when unset), distinct from the resolved
        ``random_seed`` recorded alongside it.
        """
        return dict(vars(self._parsed_arguments))

    @property
    def save_baseline(self) -> bool:
        """Indicates whether to save baseline posterior metrics to baseline.json."""
        return bool(self._parsed_arguments.save_baseline)

    @property
    def compare_baseline(self) -> str | None:
        """Path to a baseline.json file for comparison, or None if not set."""
        val: str | None = self._parsed_arguments.compare_baseline
        return val

    @property
    def pdet_dl_bins(self) -> int:
        """Number of luminosity distance bins for P_det grid."""
        return int(self._parsed_arguments.pdet_dl_bins)

    @property
    def pdet_mass_bins(self) -> int:
        """Number of mass bins for P_det grid."""
        return int(self._parsed_arguments.pdet_mass_bins)

    @property
    def pdet_estimator(self) -> str:
        """P_det kernel-regression estimator ('local_linear' or 'nadaraya_watson')."""
        return str(self._parsed_arguments.pdet_estimator)

    @property
    def fisher_cond_threshold(self) -> float:
        """Condition number threshold for flagging near-singular covariance matrices."""
        return float(self._parsed_arguments.fisher_cond_threshold)

    @property
    def normalization_mode(self) -> str:
        """In-catalogue L_cat normalization ('global'/'local_ratio'/'volume_deconv')."""
        return str(self._parsed_arguments.normalization_mode)

    @property
    def catalog_only(self) -> bool:
        """Skip completion integral: set f_i=1, L_comp=0 (catalog-only diagnostic)."""
        return bool(self._parsed_arguments.catalog_only)

    @property
    def allow_low_pdet_coverage(self) -> bool:
        """Escape hatch for the hard P_det grid-coverage / shallow-pool gate."""
        return bool(self._parsed_arguments.allow_low_pdet_coverage)

    @property
    def prescreen_audit(self) -> bool:
        """Bypass the quick-SNR early skip while logging (quick, full) SNR pairs."""
        return bool(self._parsed_arguments.prescreen_audit)

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
    parser.add_argument(
        "--save_baseline",
        action="store_true",
        default=False,
        help="Extract baseline posterior metrics from existing h-sweep posteriors and save as baseline.json. "
        "Requires a full h-grid sweep (3+ h-values) in the posteriors directory.",
    )
    parser.add_argument(
        "--compare_baseline",
        type=str,
        default=None,
        help="Path to a baseline.json file. Generates a comparison report between the baseline "
        "and the current posteriors directory. Works with or without --evaluate.",
    )
    parser.add_argument(
        "--catalog_only",
        action="store_true",
        default=False,
        help="Skip completion integral in evaluation: set f_i=1, L_comp=0 (catalog-only diagnostic).",
    )
    parser.add_argument(
        "--allow_low_pdet_coverage",
        action="store_true",
        default=False,
        help=(
            "Proceed despite <95%% P_det grid coverage or a shallow injection pool "
            "(stale-pool gate). Only for deliberate re-evaluations of archived "
            "shallow baselines."
        ),
    )
    parser.add_argument(
        "--prescreen_audit",
        action="store_true",
        default=False,
        help=(
            "Audit mode for the quick-SNR pre-screen: compute the full SNR even "
            "when the quick gate would skip, and log PRESCREEN_AUDIT lines with "
            "(quick_snr, full_snr, params). Smoke-test use only (issue #19 / "
            "PRE_SCREEN_SNR_FACTOR re-validation)."
        ),
    )
    parser.add_argument(
        "--pdet_dl_bins",
        type=int,
        default=60,
        help="Number of luminosity distance bins for P_det grid (default: 60).",
    )
    parser.add_argument(
        "--pdet_mass_bins",
        type=int,
        default=40,
        help="Number of mass bins for P_det grid (default: 40).",
    )
    parser.add_argument(
        "--pdet_estimator",
        type=str,
        choices=["local_linear", "nadaraya_watson"],
        default="local_linear",
        help=(
            "P_det kernel-regression estimator. 'local_linear' (default, F4-v2) "
            "corrects the d_L->0 boundary bias; 'nadaraya_watson' is the pre-F4-v2 "
            "local-constant form, kept for regression/comparison (default: local_linear)."
        ),
    )
    parser.add_argument(
        "--fisher_cond_threshold",
        type=float,
        default=1e16,
        help="Condition number threshold for excluding near-singular covariance matrices (default: 1e16).",
    )
    parser.add_argument(
        "--normalization_mode",
        type=str,
        choices=["global", "local_ratio", "volume_deconv"],
        default="volume_deconv",
        help=(
            "In-catalogue L_cat normalization (commission de-rail study). "
            "'volume_deconv' (default): Gray A.9/A.10 local ratio-of-sums with the host-z "
            "prior deconvolved through the comoving-volume element dV_c/(1+z) -- de-railed "
            "AND statistically calibrated (D2 P-P). 'local_ratio': the same local ratio with "
            "a bare-Gaussian host-z prior (de-railed but ~2-3%% low-biased). 'global': the "
            "pre-fix global-denominator single ratio (rails to a grid edge on photo-z data). "
            "See .planning/INDEPENDENT-VERIFICATION-REPORT-20260701.md sec 7."
        ),
    )
    parsed_arguments: argparse.Namespace = parser.parse_args(arguments)
    return parsed_arguments
