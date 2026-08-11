import argparse
import logging
import os
import random
import sys

from darksiren_emri.constants import H
from darksiren_emri.exceptions import ArgumentsError

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
    def h_values(self) -> str | None:
        """Comma-separated h-grid for a fused evaluation pass (supersedes h_value)."""
        value = self._parsed_arguments.h_values
        return str(value) if value is not None else None

    @property
    def snr_analysis(self) -> bool:
        """Indicates whether the snr analysis should be run."""
        return bool(self._parsed_arguments.snr_analysis)

    @property
    def injection_campaign(self) -> bool:
        """Indicates whether to run SNR-only injection campaign for detection probability estimation."""
        return bool(self._parsed_arguments.injection_campaign)

    @property
    def injection_mixture(self) -> bool:
        """Stratified 3-component injection sampling measure (issue #51, default off)."""
        return bool(self._parsed_arguments.injection_mixture)

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
    def pdet_z_resolved(self) -> bool:
        """FIX-2: z-resolved detection survival S(d_L | z) (default off = pooled)."""
        return bool(self._parsed_arguments.pdet_z_resolved)

    @property
    def pdet_wbh_z_resolved(self) -> bool:
        """FIX-3 §7.1: joint z x M_z-resolved with-BH survival (default off)."""
        return bool(self._parsed_arguments.pdet_wbh_z_resolved)

    @property
    def fisher_cond_threshold(self) -> float:
        """Condition number threshold for flagging near-singular covariance matrices."""
        return float(self._parsed_arguments.fisher_cond_threshold)

    @property
    def normalization_mode(self) -> str:
        """In-catalogue L_cat normalization ('global'/'local_ratio'/'volume_deconv')."""
        return str(self._parsed_arguments.normalization_mode)

    @property
    def smear_global_selection(self) -> bool:
        """Opt-in sigma_z-smeared Sigma_glob (num/denom symmetry, issue #30 R4)."""
        return bool(self._parsed_arguments.smear_global_selection)

    @property
    def host_z_kernel(self) -> str:
        """Numerator host-z kernel decomposition flag (issue #40a): 'auto'/'point'/'volume_deconv'."""
        return str(self._parsed_arguments.host_z_kernel)

    @property
    def host_mass_kernel(self) -> str:
        """2D host-mass kernel decomposition flag (#40 remainder): 'auto'/'gaussian'/'trunc_lognormal'."""
        return str(self._parsed_arguments.host_mass_kernel)

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
    def snapshot_ics(self) -> bool:
        """Restore the retired snapshot p0 ~ U[10, 16] draw (archaeology only)."""
        return bool(self._parsed_arguments.snapshot_ics)

    @property
    def realize_observed_catalogue(self) -> bool:
        """Generate a seeded observed-catalogue realization (campaign #53, §6.1)."""
        return bool(self._parsed_arguments.realize_observed_catalogue)

    @property
    def realization_seed(self) -> int | None:
        """Seed of the observed-catalogue realization (required with --realize_observed_catalogue)."""
        val: int | None = self._parsed_arguments.realization_seed
        return val

    @property
    def realization_sigma_scale(self) -> float:
        """Global width multiplier of the realization; 0 = byte-identical copy (R6 gate)."""
        return float(self._parsed_arguments.realization_sigma_scale)

    @property
    def realization_parent(self) -> str | None:
        """Parent (TRUE) catalogue CSV for the realization; None = the reduced catalogue."""
        val: str | None = self._parsed_arguments.realization_parent
        return val

    @property
    def observed_catalogue(self) -> str | None:
        """Observed-catalogue CSV (+ sidecar) to evaluate against; None = baseline catalogue."""
        val: str | None = self._parsed_arguments.observed_catalogue
        return val

    @property
    def combine(self) -> bool:
        """Indicates whether to combine per-event posteriors into joint H0 posterior."""
        return bool(self._parsed_arguments.combine)

    @property
    def strategy(self) -> str:
        """Zero-handling strategy for posterior combination."""
        return str(self._parsed_arguments.strategy)

    @property
    def max_redshift(self) -> float | None:
        """Analysis-depth truncation override for Model1CrossCheck.max_redshift.

        None (default) leaves the constructor's built-in depth (1.5) untouched --
        byte-identical to pre-flag behavior. When given, overrides the population
        depth cap used by the numerator candidate-host window AND the selection
        integrals D(h)/beta_Gbar(h)/Sigma_global(h)/B_num (issue #30 depth-truncation
        study; see results/campaign_phase2_runs/MAX_REDSHIFT_SEMANTICS.md).
        """
        val: float | None = self._parsed_arguments.max_redshift
        return val

    @property
    def freeze_g_frac_ref_h(self) -> float | None:
        """Reference h for the frozen-g_frac counterfactual (INSTRUMENTATION).

        None (default) is the production path and is BYTE-IDENTICAL to
        pre-flag behaviour. When set to ``h_ref``, each event's 2D completion
        numerator becomes ``B_num(h) * g_ref`` with
        ``g_ref = B_num_wbh(h_ref)/B_num(h_ref)`` — the h-slope of the
        completion-leg mass factor is removed while every other h-dependence
        still moves. Diagnostic only (gate (vii) follow-up); NOT a physics
        change and never a production posterior.
        """
        val: float | None = self._parsed_arguments.freeze_g_frac_ref_h
        return val

    @property
    def selection_in_completion_numerator(self) -> str:
        """N-2 selection-in-numerator counterfactual cell (INSTRUMENTATION).

        ``"off"`` (default) is the production path and is BYTE-IDENTICAL to
        pre-flag behaviour. ``"1d"`` multiplies the **1D** completion-leg
        z-integrand by the phi-marginal survival ``S_bar_phi(z;h)`` — the same
        table (and the same ``np.interp`` accessor) the ``D~^phi``
        normalisation already uses — leaving the 2D channel, both catalogue
        legs and the whole selection stack untouched. Diagnostic only
        (N-2 counterfactual); NOT a physics change and never a production
        posterior.
        """
        val: str = self._parsed_arguments.selection_in_completion_numerator
        return val

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

        # Observed-catalogue realization stage (campaign #53, docs/derivations/
        # realistic_host_observation_model.md §6.1): the realization is a pure
        # function of (parent CSV, seed, sigma_scale) — an explicit seed is
        # therefore mandatory (no silent random realization seeds).
        if self.realize_observed_catalogue and self.realization_seed is None:
            raise ArgumentsError(
                "--realize_observed_catalogue requires an explicit "
                "--realization_seed <int> (the realization must be a pure, "
                "reproducible function of parent + seed + sigma_scale)."
            )
        if self.realization_sigma_scale < 0:
            raise ArgumentsError(
                f"--realization_sigma_scale must be >= 0, got {self.realization_sigma_scale}."
            )
        # Convention (A) [RATIFY-R1]: the observed catalogue is visible to the
        # INFERENCE only. The injection/simulation side always runs on the TRUE
        # (baseline) catalogue — refuse any pairing that would leak the
        # realization into the generative path.
        if self.observed_catalogue is not None and (
            self.simulation_steps > 0 or self.injection_campaign or self.snr_analysis
        ):
            raise ArgumentsError(
                "--observed_catalogue is an evaluation-side override only "
                "(convention (A): the injection/simulation side uses the TRUE "
                "catalogue; docs/derivations/realistic_host_observation_model.md "
                "§1.2/§5). Drop --simulation_steps/--injection_campaign/"
                "--snr_analysis or drop --observed_catalogue."
            )


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
    parser.add_argument(
        "--h_values",
        type=str,
        default=None,
        help=(
            "Comma-separated Hubble constant values for a fused evaluation pass "
            "(e.g. '0.70,0.705,0.71'). Supersedes --h_value: all h-invariant setup "
            "(catalogue, BallTree, injection pool, P_det grid, Fisher staging, worker "
            "pool) is paid once, and per-h posterior JSONs are written as each h "
            "completes. Default: single-h evaluation via --h_value."
        ),
    )
    parser.add_argument("--snr_analysis", action="store_true")
    parser.add_argument(
        "--injection_campaign",
        action="store_true",
        default=False,
        help="Run SNR-only injection campaign for detection probability estimation.",
    )
    parser.add_argument(
        "--injection_mixture",
        action="store_true",
        default=False,
        help=(
            "Stratified 3-component injection sampling measure for the campaign "
            "redesign (issue #51; ratified sizing recommendation in results/"
            "lcat_h_dependence_20260725/campaign_sizing_20260728/SIZING_ANALYSIS.md "
            "§6): 0.50 stratum 'a' (status-quo Babak M1 emcee draw), 0.25 stratum "
            "'b' (catalogue-coverage, (z, M) ~ R_eff/(1+z)-weighted catalogue "
            "rows), 0.25 stratum 'c' (flat in (u = ln(1+z), m = log10 M_z) on the "
            "reachable region). Every injection CSV row records its 'stratum'. "
            "Default OFF: pure stratum-a draw, byte-identical to the pre-#51 "
            "campaign. Only meaningful together with --injection_campaign."
        ),
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
        "--snapshot_ics",
        action="store_true",
        default=False,
        help=(
            "Archaeology only: restore the retired snapshot initial-condition "
            "draw p0 ~ U[10, 16] (the pre-2026-07-28 convention) instead of the "
            "plunge-window draw t_plunge ~ U[0, T] with p0 from the PN5 "
            "time-to-plunge root-find. See "
            "docs/derivations/plunge_window_initial_conditions.md."
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
        "--pdet_z_resolved",
        # [PHYSICS] production default since 2026-07-26 (author-ratified adoption,
        # results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md):
        # multi-seed verification passed bias + width criteria on 4 deep venues.
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "[PHYSICS] FIX-2: z-resolved detection survival (production default). "
            "Every 3D (without-BH-mass) selection query uses the z-CONDITIONAL "
            "survival S(d_L | z) = P(d_hor >= d_L | z) (Gaussian kernel in "
            "u = ln(1+z), Scott d=1 bandwidth, Abramson-adaptive; exact "
            "suffix-survival in d_L) instead of the pooled S(d_L). The 2D "
            "M_z-conditioned grid keeps its current form. Use "
            "--no-pdet_z_resolved for the pooled legacy behavior. Ships/gates "
            "jointly with the generator_marginal normalization (stacked "
            "prediction, packet §6). "
            "results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md."
        ),
    )
    parser.add_argument(
        "--pdet_wbh_z_resolved",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "[PHYSICS] FIX-3 SS7.1 (default OFF = byte-identical to the current "
            "stack): joint z x M_z-resolved with-BH detection survival. Every "
            "with-BH (2D) selection query uses the joint conditional "
            "S(d_L | z, M_z) (product Gaussian kernel in u = ln(1+z) and "
            "log10 M_z, Scott d=2 bandwidths, Abramson-adaptive on u; exact "
            "suffix-survival in d_L; ESS-weighted shrinkage toward "
            "S(d_L | M_z)) instead of the pooled-in-z S(d_L | M_z). Requires "
            "--pdet_z_resolved (RATIFY-Z7 guard). "
            "docs/derivations/fix3_zmz_catalog_selection.md."
        ),
    )
    parser.add_argument(
        "--fisher_cond_threshold",
        type=float,
        default=1e16,
        help="Condition number threshold for excluding near-singular covariance matrices (default: 1e16).",
    )
    parser.add_argument(
        "--host_z_kernel",
        type=str,
        choices=["auto", "point", "volume_deconv"],
        default="auto",
        help=(
            "Issue #40(a) decomposition flag (redteam F2/F3): selects the "
            "in-catalogue NUMERATOR host-z kernel independently of the "
            "normalization leg. 'auto' (default) preserves the historical "
            "bundling — the delta-kernel (point/point) numerator iff "
            "--normalization_mode=generator_marginal, else the quadrature "
            "kernel. 'point'/'volume_deconv' force the numerator kernel for "
            "per-leg attribution A/Bs; the n_hat_w/D_gen normalization "
            "machinery stays governed by --normalization_mode. The real-data "
            "PV/photo-z kernel is a pending derivation "
            "(docs/derivations/hostz_pv_photoz_kernel.md, issue #40b)."
        ),
    )
    parser.add_argument(
        "--host_mass_kernel",
        type=str,
        choices=["auto", "gaussian", "trunc_lognormal"],
        default="auto",
        help=(
            "#40-remainder decomposition flag (RATIFIED 2026-07-27): selects "
            "the 2D (with-BH-mass) host-mass kernel independently of the "
            "normalization leg. 'auto' (default) preserves the historical "
            "bundling — the truncated lognormal x R_eff kernel iff "
            "--normalization_mode=mass_trunc, else the analytic Gaussian "
            "product (+ G2d shift in the calibrated kernels). "
            "'gaussian'/'trunc_lognormal' force the mass kernel for the "
            "kernel A/Bs; the normalization machinery stays governed by "
            "--normalization_mode. Combining 'trunc_lognormal' with a "
            "point-resolving host-z numerator raises (prior-consistency "
            "guard). docs/derivations/mass_marginal_2d_kernel.md."
        ),
    )
    parser.add_argument(
        "--normalization_mode",
        type=str,
        choices=[
            "global",
            "local_ratio",
            "volume_deconv",
            "absolute_marginal",
            "generator_marginal",
        ],
        # [PHYSICS] production default since 2026-07-26 (author-ratified adoption,
        # results/lcat_h_dependence_20260725/MULTISEED_READOUT_20260726.md;
        # derivation: DERIVATION_GENERATOR_CONSISTENT_NORM.md).
        default="generator_marginal",
        help=(
            "In-catalogue L_cat normalization. 'generator_marginal' (default, "
            "production since 2026-07-26): see below. "
            "'volume_deconv' (pre-2026-07-26 default): Gray A.9/A.10 local ratio-of-sums with the host-z "
            "prior deconvolved through the comoving-volume element dV_c/(1+z) -- de-railed "
            "AND statistically calibrated (D2 P-P). 'local_ratio': the same local ratio with "
            "a bare-Gaussian host-z prior (de-railed but ~2-3%% low-biased). 'global': the "
            "pre-fix global-denominator single ratio (rails to a grid edge on photo-z data). "
            "'absolute_marginal': the absolute-mass per-event host marginal "
            "p_i = (A_i + B_num)/D with A_i = (Sum_ball w_g N_g)/n_bar_w, "
            "n_bar_w = Sigma_glob/beta_G (issue #30 estimator redesign, Variant 1; "
            "volume_deconv host-z kernel; empty balls reduce continuously to B_num/D). "
            "'generator_marginal': the generator-consistent normalization (E1 FIX-3): "
            "p_i = (Sum_ball w_g N_g / n_hat_w + B_num)/D_gen with the DRAW-SIDE "
            "calibration n_hat_w = W_cat/V_f(h) (no P_det inside) and "
            "D_gen = Sigma_glob_wbh/n_hat_w + beta_Gbar (4D-exact catalogue selection); "
            "point/point sigma_z pairing (N_g point-evaluated at the catalogue z_g; "
            "incompatible with --smear_global_selection); empty balls reduce "
            "continuously to B_num/D_gen. "
            "See .planning/INDEPENDENT-VERIFICATION-REPORT-20260701.md sec 7, "
            "results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md and "
            "results/lcat_h_dependence_20260725/DERIVATION_GENERATOR_CONSISTENT_NORM.md."
        ),
    )
    parser.add_argument(
        "--smear_global_selection",
        action="store_true",
        help=(
            "Opt-in [PHYSICS] refinement of the global in-catalogue selection sum "
            "Sigma_glob: replace the point evaluation P_det(d_L(z_g;h)) per galaxy "
            "with the expectation over the SAME volume-deconvolved host-z kernel "
            "the in-catalogue numerator uses (num/denom sigma_z symmetry, issue #30 "
            "estimator redesign risk R4). Off by default (point evaluation, "
            "byte-identical legacy behavior). Relevant to normalization modes that "
            "consume Sigma_glob ('global', 'absolute_marginal'). Incompatible with "
            "'generator_marginal' (that mode is defined with the point/point "
            "sigma_z pairing and rejects this flag)."
        ),
    )
    parser.add_argument(
        "--realize_observed_catalogue",
        action="store_true",
        default=False,
        help=(
            "[PHYSICS] Campaign #53 (RATIFIED 2026-07-29, docs/derivations/"
            "realistic_host_observation_model.md §6.1): write a seeded "
            "OBSERVED-catalogue realization of the reduced catalogue into the "
            "working directory (observed_catalogue_seed{S}.csv + hashed "
            ".meta.json sidecar). One total Gaussian per row from the stored "
            "width columns ([RATIFY-R2] counted-once): z_obs = z + N(0, "
            "z_error), clipped at 1e-5; ln M_obs = ln M + N(0, M_error/M). "
            "Requires --realization_seed. The parent catalogue, every CRB and "
            "the injection pool are untouched (convention (A))."
        ),
    )
    parser.add_argument(
        "--realization_seed",
        type=int,
        default=None,
        help=(
            "Seed for the observed-catalogue realization (required with "
            "--realize_observed_catalogue). The realization is a pure function "
            "of (parent CSV, seed, sigma_scale) — bit-reproducible."
        ),
    )
    parser.add_argument(
        "--realization_sigma_scale",
        type=float,
        default=1.0,
        help=(
            "Global width multiplier for the realization (default 1.0). "
            "0 writes a BYTE-IDENTICAL copy of the parent catalogue — the "
            "mandatory sigma->0 regression gate ([RATIFY-R6] / P5)."
        ),
    )
    parser.add_argument(
        "--realization_parent",
        type=str,
        default=None,
        help=(
            "Parent (TRUE) catalogue CSV for --realize_observed_catalogue. "
            "Default: the production reduced catalogue "
            "(darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv)."
        ),
    )
    parser.add_argument(
        "--observed_catalogue",
        type=str,
        default=None,
        help=(
            "[PHYSICS] Evaluate against an OBSERVED-catalogue realization: path "
            "to observed_catalogue_seed{S}.csv (the .meta.json sidecar is read "
            "from the same location and its hash verified). Scattered sidecar "
            "(sigma_scale > 0) activates the one-directional prior-consistency "
            "guards: --host_z_kernel point and --normalization_mode "
            "generator_marginal are refused (use absolute_marginal + "
            "volume_deconv, [RATIFY-R3]). Evaluation-side only; the simulation/"
            "injection stages always use the TRUE catalogue (convention (A)). "
            "Default: the baseline reduced catalogue, byte-identical behaviour."
        ),
    )
    parser.add_argument(
        "--max_redshift",
        type=float,
        default=None,
        help=(
            "Analysis-depth truncation override for Model1CrossCheck.max_redshift "
            "(default: None, leaves the built-in depth of 1.5 unchanged -- a no-op "
            "for the current population, since z_max(h) <= ~1.33). Caps the "
            "candidate-host window, D(h)/beta_Gbar(h)/Sigma_global(h), and the "
            "B_num completion numerator at min(z_max(h), max_redshift) (issue #30 "
            "depth-truncation study)."
        ),
    )
    parser.add_argument(
        "--freeze_g_frac_ref_h",
        type=float,
        default=None,
        help=(
            "INSTRUMENTATION (default None = OFF, byte-identical production "
            "path): frozen-g_frac counterfactual for the gate (vii) follow-up. "
            "When set to h_ref, every event's 2D (with-BH-mass) completion "
            "numerator becomes B_num(h) * g_ref with g_ref = "
            "B_num_wbh(h_ref)/B_num(h_ref) computed by the SAME quadrature at "
            "the reference h. Removes the h-slope of the completion-leg mass "
            "factor while the catalogue legs, w~_G, B_num itself and the whole "
            "1D channel keep their h-dependence. The diagnostics CSV g_frac "
            "column emits the value actually used (h-constant per event when "
            "frozen). Diagnostic counterfactual only -- NOT a physics change "
            "and never a production posterior. See "
            "results/run_20260804_postfix/gate_vii/"
            "PREREGISTRATION_FROZEN_GFRAC.md."
        ),
    )
    parser.add_argument(
        "--selection_in_completion_numerator",
        type=str,
        choices=["off", "1d"],
        default="off",
        help=(
            "INSTRUMENTATION (default 'off' = byte-identical production path): "
            "the N-2 selection-in-numerator counterfactual. Under '1d' the 1D "
            "completion-leg z-integrand is multiplied by the phi-marginal "
            "survival S_bar_phi(z;h) INSIDE the quadrature, i.e. B_num = "
            "INTEGRAL (1-f_k) p_gw dVc/(1+z) S_bar_phi(z;h) dz — the SAME "
            "table and the SAME np.interp accessor the D~^phi normalisation "
            "uses (precompute_phi_marginal_survival). The 2D (with-BH-mass) "
            "leg, both catalogue legs, w~_G, r_Malm, beta^phi and D~^phi are "
            "untouched; the D~^phi denominator deliberately stays the "
            "PRODUCTION object so the run is a pure numerator contrast. The "
            "'both' cell of the derivation draft is DELETED: measurement M2 "
            "found the 2D arm inert at |Sum| <= 7.2 nats over the grid, 200x "
            "under the 20 nats/h tolerance. Diagnostic counterfactual only -- "
            "NOT a physics change and never a production posterior. Requires "
            "--normalization_mode absolute_marginal. See "
            "results/run_20260804_postfix/gate_vii/PREREGISTRATION_N2_SEL1D.md."
        ),
    )
    parsed_arguments: argparse.Namespace = parser.parse_args(arguments)
    return parsed_arguments
