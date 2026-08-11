import datetime
import json
import logging
import os
import shutil
import signal
import subprocess
import warnings
from collections.abc import Iterator
from time import time
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from darksiren_emri.arguments import Arguments
from darksiren_emri.constants import M_SOURCE_FRAME_MAX, M_SOURCE_FRAME_MIN
from darksiren_emri.cosmological_model import Model1CrossCheck
from darksiren_emri.exceptions import ParameterEstimationError, ParameterOutOfBoundsError

if TYPE_CHECKING:
    from darksiren_emri.callbacks import SimulationCallback
from darksiren_emri.galaxy_catalogue.handler import (
    GalaxyCatalogueHandler,
    HostGalaxy,
)

# logging setup
_ROOT_LOGGER = logging.getLogger()


def main() -> None:
    """
    Run main to start the program.
    """
    from darksiren_emri.plotting import apply_style

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
        from darksiren_emri.bayesian_inference.posterior_combination import combine_posteriors

        # Provenance for the fast-path combine stage (readiness sweep TC-10):
        # the early return below skips _write_run_metadata, so the combine
        # stage previously recorded no git_commit/args at all. Distinct
        # filename avoids colliding with simulation task metadata.
        _write_run_metadata(
            arguments.working_directory,
            arguments.seed,
            arguments,
            filename="run_metadata_combine.json",
        )

        for variant_dir in ["posteriors", "posteriors_with_bh_mass"]:
            posteriors_dir = os.path.join(arguments.working_directory, variant_dir)
            if os.path.isdir(posteriors_dir):
                _ROOT_LOGGER.info(f"Combining posteriors from {posteriors_dir}")
                combine_posteriors(
                    posteriors_dir=posteriors_dir,
                    strategy=arguments.strategy,
                    output_dir=os.path.join(arguments.working_directory, variant_dir),
                    allow_shallow_pool=arguments.allow_low_pdet_coverage,
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

    if arguments.realize_observed_catalogue:
        # [PHYSICS] Campaign #53 observed-catalogue realization stage
        # (docs/derivations/realistic_host_observation_model.md §6.1). Pure
        # CPU, no model needed; provenance goes to a dedicated metadata file
        # (the sidecar itself carries the full realization record).
        from darksiren_emri.galaxy_catalogue.handler import REDUCED_CATALOGUE_FILE_PATH
        from darksiren_emri.galaxy_catalogue.observed_realization import (
            observed_catalogue_filename,
            realize_observed_catalogue,
        )

        _write_run_metadata(
            arguments.working_directory,
            arguments.seed,
            arguments,
            filename="run_metadata_realization.json",
        )
        _realization_seed = arguments.realization_seed
        assert _realization_seed is not None  # enforced by Arguments.validate()
        _observed_path = os.path.join(
            arguments.working_directory, observed_catalogue_filename(_realization_seed)
        )
        _parent_path = (
            arguments.realization_parent
            if arguments.realization_parent is not None
            else REDUCED_CATALOGUE_FILE_PATH
        )
        _sidecar = realize_observed_catalogue(
            parent_csv_path=_parent_path,
            output_csv_path=_observed_path,
            realization_seed=_realization_seed,
            sigma_scale=arguments.realization_sigma_scale,
        )
        _ROOT_LOGGER.info(
            "Observed catalogue written to %s (sha256 %s); evaluate with "
            "--observed_catalogue %s --normalization_mode absolute_marginal "
            "--host_z_kernel volume_deconv",
            _observed_path,
            _sidecar["observed_csv_sha256"],
            _observed_path,
        )

    if not _needs_model:
        end_time = time()
        _ROOT_LOGGER.debug(f"Finished in {end_time - start_time}s.")
        return

    seed = arguments.seed
    rng = np.random.default_rng(seed)
    _ROOT_LOGGER.info(f"Random seed: {seed}")
    _write_run_metadata(arguments.working_directory, seed, arguments)

    cosmological_model = Model1CrossCheck(rng=rng, max_redshift_override=arguments.max_redshift)
    # Catalogue pruning on the SOURCE-frame population band (host BH masses are
    # rest-frame): constants.M_SOURCE_FRAME_*, the single mass boundary (issue
    # #51). parameter_space.M.limits are the detector-frame M_z domain and must
    # NOT be used here.
    galaxy_catalog = GalaxyCatalogueHandler(
        M_min=M_SOURCE_FRAME_MIN,
        M_max=M_SOURCE_FRAME_MAX,
        z_max=cosmological_model.max_redshift,
        # [PHYSICS] Campaign #53: evaluation-side OBSERVED-catalogue override
        # (None = baseline reduced catalogue, byte-identical behaviour).
        # Arguments.validate() refuses pairing this with any generative stage
        # (convention (A); realistic_host_observation_model.md §1.2/§5).
        observed_catalogue_path=arguments.observed_catalogue,
    )

    if arguments.simulation_steps > 0 and not arguments.injection_campaign:
        data_simulation(
            arguments.simulation_steps,
            cosmological_model,
            galaxy_catalog,
            arguments.simulation_index,
            arguments.h_value,
            rng=rng,
            use_gpu=arguments.use_gpu,
            prescreen_audit=arguments.prescreen_audit,
            snapshot_ics=arguments.snapshot_ics,
        )

    if arguments.evaluate:
        # --h_values (fused h-grid pass) supersedes --h_value when given.
        _h_values: list[float] | None = None
        if arguments.h_values is not None:
            _h_values = [float(tok) for tok in arguments.h_values.split(",") if tok.strip()]
        evaluate(
            cosmological_model,
            galaxy_catalog,
            arguments.h_value,
            num_workers=arguments.num_workers,
            catalog_only=arguments.catalog_only,
            pdet_dl_bins=arguments.pdet_dl_bins,
            pdet_mass_bins=arguments.pdet_mass_bins,
            pdet_estimator=arguments.pdet_estimator,
            fisher_cond_threshold=arguments.fisher_cond_threshold,
            normalization_mode=arguments.normalization_mode,
            # G4: --seed now reaches the inference layer (deterministic MC denominator).
            base_seed=seed,
            allow_low_pdet_coverage=arguments.allow_low_pdet_coverage,
            h_values=_h_values,
            smear_global_selection=arguments.smear_global_selection,
            pdet_z_resolved=arguments.pdet_z_resolved,
            pdet_wbh_z_resolved=arguments.pdet_wbh_z_resolved,
            host_z_kernel=arguments.host_z_kernel,
            host_mass_kernel=arguments.host_mass_kernel,
            freeze_g_frac_ref_h=arguments.freeze_g_frac_ref_h,
            selection_in_completion_numerator=arguments.selection_in_completion_numerator,
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
            # Issue #51 stratified 3-component sampling measure (SIZING_ANALYSIS.md
            # §6): opt-in via --injection_mixture; stratum 'b' draws from the
            # pruned catalogue, so the handler is threaded through.
            galaxy_catalog=galaxy_catalog,
            injection_mixture=arguments.injection_mixture,
            # Realized stratum counts are appended to the run metadata written
            # above (provenance: flag itself is already in cli_args).
            run_metadata_path=os.path.join(
                arguments.working_directory, _run_metadata_filename(arguments)
            ),
            snapshot_ics=arguments.snapshot_ics,
        )

    end_time = time()
    _ROOT_LOGGER.debug(f"Finished in {end_time - start_time}s.")


def _save_baseline(working_directory: str) -> None:
    """Extract baseline metrics from existing posteriors and save as baseline.json."""
    from pathlib import Path

    from darksiren_emri.bayesian_inference.evaluation_report import extract_baseline

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

    from darksiren_emri.bayesian_inference.evaluation_report import (
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
        from darksiren_emri.bayesian_inference.evaluation_report import (
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
    # Anchor at the package's real location, NOT the process CWD: cluster jobs
    # run from a private $RUN_DIR/cwd (not a git repo), which would silently
    # yield "unknown". realpath resolves the cwd's darksiren_emri symlink
    # back to the checkout.
    repo_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=repo_root
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _run_metadata_filename(arguments: Arguments) -> str:
    """Default run-metadata filename (single source shared with _write_run_metadata)."""
    index = arguments.simulation_index
    if index > 0 or "SLURM_ARRAY_TASK_ID" in os.environ:
        return f"run_metadata_{index}.json"
    return "run_metadata.json"


def _write_run_metadata(
    working_directory: str, seed: int, arguments: Arguments, filename: str | None = None
) -> None:
    metadata = {
        "git_commit": _get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": seed,
        # Full parsed namespace so every flag is captured — including the
        # inference-critical knobs (normalization_mode, pdet_*, catalog_only, ...)
        # that a hand-maintained subset silently dropped (review REP-02).
        "cli_args": arguments.to_dict(),
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

    if filename is None:
        filename = _run_metadata_filename(arguments)
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
        f"darksiren_emri_{timestamp}_h_{str(np.round(h_value, 4)).replace('.', '_')}.log",
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
    from darksiren_emri.datamodels.parameter_space import ParameterSpace
    from darksiren_emri.memory_management import MemoryManagement
    from darksiren_emri.parameter_estimation.parameter_estimation import (
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
    h_value: float,
    callbacks: list["SimulationCallback"] | None = None,
    rng: np.random.Generator | None = None,
    *,
    use_gpu: bool = False,
    prescreen_audit: bool = False,
    snapshot_ics: bool = False,
) -> None:
    # conditional imports because they require GPU
    from darksiren_emri.memory_management import MemoryManagement
    from darksiren_emri.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
        WaveGeneratorType,
    )

    _callbacks: list[SimulationCallback] = callbacks or []

    # Normalize the rng once so the rate-weighted host draw
    # (draw_rate_weighted_hosts) and parameter randomization share a single,
    # reproducible generator under --seed.
    if rng is None:
        rng = np.random.default_rng()

    def _alarm_handler(signum: int, frame: object) -> None:
        raise TimeoutError("Computation exceeded the alarm timeout")

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

    from darksiren_emri.constants import (
        HOST_DRAW_Z_MAX,
        PRE_SCREEN_SNR_FACTOR,
        PRESCREEN_DL_MARGIN,
    )
    from darksiren_emri.dark_siren_injection import (
        compute_global_catalog_fraction,
        draw_mixture_hosts,
    )
    from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build
    from darksiren_emri.physical_relations import luminosity_distance_prescreen_gpc

    # CHANGE 4b/5: split injected hosts into an in-catalog fraction F and an
    # out-of-catalog (dark) fraction 1-F so the injected population matches the
    # inference mixture f*L_cat + (1-f)*L_comp (Gray et al. 2020 Eq. 9; Chen et
    # al. 2024 arXiv:2212.08694 self-consistency). F is the completeness f_bar(z)
    # marginalised over the source-frame redshift population prior; it is
    # precomputed ONCE per run at the injection cosmology h_value. The completeness
    # object is the per-pixel PixelCompleteness loaded from the SAME frozen cached
    # m_th map the inference uses (C1 byte-identity; bayesian_statistics.evaluate).
    completeness = from_cache_or_build()
    global_catalog_fraction = compute_global_catalog_fraction(
        completeness, h=h_value, z_max=HOST_DRAW_Z_MAX
    )
    _ROOT_LOGGER.info(
        "CHANGE 4b dark-event injection: global in-catalog fraction F = %.4f "
        "(h_inj=%.4f, z_max=%.3f); injecting (1-F) = %.4f dark hosts.",
        global_catalog_fraction,
        h_value,
        HOST_DRAW_Z_MAX,
        1.0 - global_catalog_fraction,
    )

    # Population-derived d_L pre-screen (issue #19): the M1 rate model samples
    # z <= max_redshift, so no valid event lies beyond d_L(max_redshift; h_value).
    # At physical SNR semantics (G8 dt² fix) the EMRI detection horizon exceeds
    # this reach — the pre-screen is inert for in-population events and only
    # guards pathological draws. Margin pending post-dt² injection re-measurement.
    # Babak et al. (2017), arXiv:1703.09722; Hogg (1999), arXiv:astro-ph/9905116 Eq. (16).
    d_L_prescreen_gpc = luminosity_distance_prescreen_gpc(
        cosmological_model.max_redshift, h=h_value
    )
    _ROOT_LOGGER.info(
        "d_L pre-screen bound: %.3f Gpc (z_max=%.2f, h=%.4f, margin=%.2f).",
        d_L_prescreen_gpc,
        cosmological_model.max_redshift,
        h_value,
        PRESCREEN_DL_MARGIN,
    )

    counter = 0
    iteration = 0
    # Per-(stage, exception-class) skip tally so any parameter-correlated drop rate
    # is quantifiable rather than silently swallowed (review SIM-03). CRB-stage drops
    # in particular happen AFTER the SNR>=threshold cut, so a correlated CRB failure
    # would be a selection-function inconsistency against the gate-free injection pool.
    skip_counts: dict[str, int] = {}
    host_galaxies: Iterator[HostGalaxy] = iter([])

    while counter < simulation_steps:
        memory_management.gpu_usage_stamp()
        memory_management.free_gpu_memory_if_pressured()
        memory_management.gpu_usage_stamp()

        _ROOT_LOGGER.info(
            f"{counter} / {iteration} evaluations successful. ({counter / (time() - memory_management._start_time) * 60}/min)"
        )
        iteration += 1

        try:
            host_galaxy = next(host_galaxies)
        except StopIteration:
            # CHANGE 4b refill: each of the 200 hosts is independently in-catalog
            # with probability F (rate-weighted draw from the z < z_max catalog:
            # P(g) ∝ w(g) = R_eff_per_mbh(M_g) / (1 + z_g), the self-consistent
            # generative model for the in-catalog inference term —
            # bayesian_statistics.p_Di reweights the catalog likelihood by the
            # SAME w(g)) or out-of-catalog/dark with probability 1-F (drawn from
            # the missing-galaxy population, NOT in the catalog). Together the
            # injected population matches the inference mixture
            # f*L_cat + (1-f)*L_comp. A dark host carries catalog_index = -1.
            # Babak et al. (2017), arXiv:1703.09722 (per-MBH rate, via emri_rate);
            # Gray et al. (2020), arXiv:1908.06050 (galaxy weighting + completeness);
            # Chen et al. (2024), arXiv:2212.08694 (in/out-of-catalog mixture).
            host_galaxies = iter(
                draw_mixture_hosts(
                    200,
                    rng,
                    galaxy_catalog,
                    completeness,
                    global_catalog_fraction,
                    h=h_value,
                    z_max=HOST_DRAW_Z_MAX,
                )
            )
            host_galaxy = next(host_galaxies)
        assert isinstance(host_galaxy, HostGalaxy)

        parameter_estimation.parameter_space.randomize_parameters(rng=rng)

        parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy, h=h_value)

        # Distance pre-screen: skip pathological events beyond the population's
        # maximum reach (d_L_prescreen_gpc, derived above from the rate model's
        # z_max at the runtime h) before generating any waveform. Inert for
        # in-population events by construction — a hit here indicates a bad
        # draw, hence WARNING, not DEBUG (issue #19).
        d_L = parameter_estimation.parameter_space.luminosity_distance.value
        if d_L > d_L_prescreen_gpc:
            _ROOT_LOGGER.warning(
                "Skipping event: d_L = %.2f Gpc > %.3f Gpc population-derived "
                "pre-screen bound (pathological draw?).",
                d_L,
                d_L_prescreen_gpc,
            )
            continue

        try:
            signal.alarm(90)
            # [PHYSICS] Plunge-window initial conditions (author-ratified
            # 2026-07-28, docs/derivations/plunge_window_initial_conditions.md):
            # t_plunge ~ U[0, T_mission] and p0 = root of t_insp(p0) = t_plunge,
            # AFTER M_z/d_L are set (the map depends on the detector-frame
            # mass). Replaces the snapshot p0 ~ U[10, 16] draw (HIGHM_AUDIT.md
            # item 1: a few-input-domain artifact that contradicted the Babak
            # 2017 plunge-rate semantics). --snapshot_ics restores the old draw
            # (archaeology only). Inside the try: the trajectory root-find
            # raises the same exception classes (ValueError "Brent...",
            # ZeroDivisionError) as waveform generation, covered by the same
            # per-event skip handlers and the 90 s alarm.
            if not snapshot_ics:
                from darksiren_emri.plunge_window import (  # noqa: PLC0415
                    draw_plunge_window_initial_conditions,
                )

                draw_plunge_window_initial_conditions(
                    parameter_estimation.parameter_space,
                    rng,
                    parameter_estimation.T,
                )
            # warnings-as-errors is scoped to the waveform/SNR computation via
            # catch_warnings() so it is restored on EVERY exit path (success,
            # exception, or the quick-gate continue) and cannot leak into the
            # inter-iteration host-refill code (review SIM-02). The alarm is
            # cancelled in the finally block below (review SIM-01).
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # [PHYSICS] Quick-SNR pre-screen DISABLED for the depth-1.5 campaign
                # (PRE_SCREEN_SNR_FACTOR = 0.0): the 2026-07-03 smoke audit (job
                # 5740080, 543 (quick, full) pairs at depth 1.5) measured 3 false
                # negatives with full SNR >= 20 at quick SNR as low as 0.25 —
                # sources plunging in years 2-5 accumulate SNR the 1-yr check
                # generator cannot see, so NO positive factor is safe. A lossy gate
                # here is a selection-function inconsistency against the gate-free
                # injection pool. The quick waveform is skipped entirely when the
                # factor is 0 (audit mode still computes it for pair logging).
                _quick_gate_enabled = PRE_SCREEN_SNR_FACTOR > 0.0
                quick_snr = float("nan")
                if _quick_gate_enabled or prescreen_audit:
                    quick_snr = parameter_estimation.compute_signal_to_noise_ratio(
                        use_snr_check_generator=True
                    )
                _quick_gate_failed = _quick_gate_enabled and (
                    quick_snr < cosmological_model.snr_threshold * PRE_SCREEN_SNR_FACTOR
                )
                if _quick_gate_failed and not prescreen_audit:
                    _ROOT_LOGGER.info(
                        f"Quick SNR threshold check failed: {np.round(quick_snr, 3)} < {cosmological_model.snr_threshold * PRE_SCREEN_SNR_FACTOR}."
                    )
                    for cb in _callbacks:
                        # sqrt(T) extrapolation of the 1-yr quick SNR to the
                        # full observation span (callback bookkeeping only).
                        cb.on_snr_computed(
                            counter, quick_snr * np.sqrt(parameter_estimation.T), False
                        )
                    continue
                snr = parameter_estimation.compute_signal_to_noise_ratio()
            if prescreen_audit:
                # Greppable audit record for the smoke run (issue #19 +
                # PRE_SCREEN_SNR_FACTOR re-validation): quick/full SNR pairs
                # with the parameters that drive waveform cost. A false
                # negative is full_snr >= threshold while the quick gate
                # would have skipped.
                _ROOT_LOGGER.info(
                    "PRESCREEN_AUDIT quick_snr=%.4f full_snr=%.4f gate_would_skip=%s "
                    "d_L=%.4f M=%.6e e0=%.4f p0=%.4f",
                    quick_snr,
                    float(snr),
                    _quick_gate_failed,
                    parameter_estimation.parameter_space.luminosity_distance.value,
                    parameter_estimation.parameter_space.M.value,
                    parameter_estimation.parameter_space.e0.value,
                    parameter_estimation.parameter_space.p0.value,
                )
        except Warning as e:
            skip_counts["snr:Warning"] = skip_counts.get("snr:Warning", 0) + 1
            if "Mass ratio" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught warning that mass ratio is out of bounds. Continue with new parameters..."
                )
                continue
            else:
                _ROOT_LOGGER.warning(f"{str(e)}. Continue with new parameters...")
                continue
        except ParameterOutOfBoundsError as e:
            skip_counts["snr:ParameterOutOfBoundsError"] = (
                skip_counts.get("snr:ParameterOutOfBoundsError", 0) + 1
            )
            _ROOT_LOGGER.warning(
                f"Caught ParameterOutOfBoundsError during parameter estimation: {str(e)}. Continue with new parameters..."
            )
            continue
        except AssertionError as e:
            skip_counts["snr:AssertionError"] = skip_counts.get("snr:AssertionError", 0) + 1
            _ROOT_LOGGER.warning(
                f"caught AssertionError: {str(e)}. Continue with new parameters..."
            )
            continue
        except RuntimeError as e:
            skip_counts["snr:RuntimeError"] = skip_counts.get("snr:RuntimeError", 0) + 1
            _ROOT_LOGGER.warning(
                f"Caught RuntimeError during waveform generation : {str(e)} .\n Continue with new parameters..."
            )
            continue
        except ValueError as e:
            if "EllipticK" in str(e):
                skip_counts["snr:ValueError:EllipticK"] = (
                    skip_counts.get("snr:ValueError:EllipticK", 0) + 1
                )
                _ROOT_LOGGER.warning(
                    "Caught EllipticK error from waveform generator. Continue with new parameters..."
                )
                continue
            elif "Brent root solver does not converge" in str(e):
                skip_counts["snr:ValueError:Brent"] = skip_counts.get("snr:ValueError:Brent", 0) + 1
                _ROOT_LOGGER.warning(
                    "Caught brent root solver error because it did not converge. Continue with new parameters..."
                )
                continue
            elif "must have different signs" in str(e):
                # few's separatrix kernel brentq at EVOLVED (a, e(t), x(t))
                # inside get_p_at_t/trajectory integration — a rare numerical
                # corner of the draw (measured ~0.5-1% of events, campaign #51
                # pilot #2 job 6073027); skippable like the Brent branch above.
                skip_counts["snr:ValueError:SeparatrixSigns"] = (
                    skip_counts.get("snr:ValueError:SeparatrixSigns", 0) + 1
                )
                _ROOT_LOGGER.warning(
                    "Caught separatrix-kernel sign error during trajectory root-find. "
                    "Continue with new parameters... params=%s",
                    parameter_estimation.parameter_space._parameters_to_dict(),
                )
                continue
            else:
                # SIM-07: re-raise the original (bare raise preserves type + traceback).
                raise
        except ZeroDivisionError:
            skip_counts["snr:ZeroDivisionError"] = skip_counts.get("snr:ZeroDivisionError", 0) + 1
            _ROOT_LOGGER.warning(
                "Caught ZeroDivisionError during trajectory integration. Continue with new parameters..."
            )
            continue
        except TimeoutError:
            skip_counts["snr:TimeoutError"] = skip_counts.get("snr:TimeoutError", 0) + 1
            # G9 gate: log the full parameter set so timeout selection can be
            # binned by (M, mu, e0, p0, ...) — see .planning/gate/G9_timeout_scan.md
            _ROOT_LOGGER.warning(
                "Waveform/SNR computation timed out (>90s). Skipping event... params=%s",
                parameter_estimation.parameter_space._parameters_to_dict(),
            )
            continue
        finally:
            # Always cancel the pending alarm — including on the quick-gate continue
            # and every exception path — so no stale SIGALRM fires in inter-iteration
            # code outside any try (review SIM-01).
            signal.alarm(0)

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
        except ParameterOutOfBoundsError:
            skip_counts["crb:ParameterOutOfBoundsError"] = (
                skip_counts.get("crb:ParameterOutOfBoundsError", 0) + 1
            )
            _ROOT_LOGGER.warning(
                "Caught ParameterOutOfBoundsError in dervative. Continue with new parameters..."
            )
            continue
        except np.linalg.LinAlgError:
            skip_counts["crb:LinAlgError"] = skip_counts.get("crb:LinAlgError", 0) + 1
            _ROOT_LOGGER.warning("Fisher matrix is singular (LinAlgError). Skipping event...")
            continue
        except ParameterEstimationError as e:
            skip_counts["crb:ParameterEstimationError"] = (
                skip_counts.get("crb:ParameterEstimationError", 0) + 1
            )
            _ROOT_LOGGER.warning(f"CRB computation failed: {e}. Skipping event...")
            continue
        except TimeoutError:
            skip_counts["crb:TimeoutError"] = skip_counts.get("crb:TimeoutError", 0) + 1
            _ROOT_LOGGER.warning(
                "Cramér-Rao bound computation timed out (>90s). Skipping event... params=%s",
                parameter_estimation.parameter_space._parameters_to_dict(),
            )
            continue
        except (ZeroDivisionError, RuntimeError, ValueError) as e:
            # CRB-stage drops happen AFTER the SNR>=threshold cut — log params so a
            # parameter-correlated failure rate is measurable (review SIM-03).
            skip_counts[f"crb:{type(e).__name__}"] = (
                skip_counts.get(f"crb:{type(e).__name__}", 0) + 1
            )
            _ROOT_LOGGER.warning(
                "Caught %s during CRB computation: %s. Skipping event... params=%s",
                type(e).__name__,
                e,
                parameter_estimation.parameter_space._parameters_to_dict(),
            )
            continue
        finally:
            signal.alarm(0)  # review SIM-01: cancel on every path
        parameter_estimation.save_cramer_rao_bound(
            cramer_rao_bound_dictionary=cramer_rao_bounds,
            snr=snr,
            host_galaxy_index=host_galaxy.catalog_index,
            # CHANGE 4b: record whether this injected host was in-catalog or dark
            # (catalog_index = -1) so the realised in-catalog fraction ≈ F is
            # recoverable from the saved Cramér-Rao bounds.
            in_catalog=host_galaxy.catalog_index != -1,
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

    # Per-class skip tally (review SIM-03): CRB-stage drops occur after the SNR cut,
    # so a nonzero, parameter-correlated crb:* count is a selection-function signal to
    # bin against (d_L, M_z, SNR) before trusting the sample completeness.
    if skip_counts:
        _ROOT_LOGGER.info(
            "Skip tally over %d attempts (%d successful): %s",
            iteration,
            counter,
            ", ".join(f"{k}={v}" for k, v in sorted(skip_counts.items())),
        )

    for cb in _callbacks:
        cb.on_simulation_end(counter, iteration)


# NOTE (W-PRE-12): this list is the injection CSV schema — pd.DataFrame(...,
# columns=...) SILENTLY DROPS any row key missing here. Keep in sync with the
# row dict built in injection_campaign.
_INJECTION_COLUMNS = [
    "z",
    "M",
    "phiS",
    "qS",
    "SNR",
    "h_inj",
    "luminosity_distance",
    "z_cut",
    "code_rev",
    # Issue #51 stratified sampling measure (SIZING_ANALYSIS.md §4 option 1):
    # 'a' = Babak M1 emcee population draw, 'b' = catalogue-coverage draw,
    # 'c' = flat-(u, m) draw. Pure-a campaigns write 'a' for every row; the
    # estimator treats an absent column as all-'a' (legacy pools).
    "stratum",
    # Plunge-window provenance (2026-07-28): drawn plunge time (yr; NaN under
    # --snapshot_ics) and derived p0. NB _flush_injection_results writes with
    # an explicit columns= list, so a key missing HERE is silently dropped —
    # pilot #3 (job 6073215, 6k rows) shipped without these two columns for
    # exactly that reason; optional for the estimator, so those rows stay valid.
    "t_plunge_yr",
    "p0",
]

# Ratified stratified-mixture proportions (alpha_a, alpha_b, alpha_c) —
# mix3_50_25_25, SIZING_ANALYSIS.md §6 recommendation block. Module constant so
# tests can exercise degenerate mixtures (e.g. (1, 0, 0)) via the
# ``stratum_probs`` parameter without touching the production default.
_STRATUM_PROBS: tuple[float, float, float] = (0.50, 0.25, 0.25)
_STRATA: tuple[str, str, str] = ("a", "b", "c")


def _flush_injection_results(results: list[dict[str, float | str]], csv_path: str) -> None:
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
    galaxy_catalog: GalaxyCatalogueHandler | None = None,
    injection_mixture: bool = False,
    stratum_probs: tuple[float, float, float] = _STRATUM_PROBS,
    run_metadata_path: str | None = None,
    snapshot_ics: bool = False,
) -> None:
    """Run SNR-only injection campaign for detection probability estimation.

    Draws EMRI events from the population model, computes SNR (no Fisher matrix),
    and stores ALL events (detected and undetected) to a per-task CSV file.

    With ``injection_mixture=True`` (issue #51, SIZING_ANALYSIS.md §6) the (z, M)
    draw is the stratified 3-component mixture mix3_50_25_25:

    * stratum 'a' (0.50): the status-quo Babak M1 emcee population draw —
      the ONLY stratum valid for pool-marginal estimator legs;
    * stratum 'b' (0.25): catalogue-coverage draw — a pruned-catalogue row with
      probability ∝ R_eff(M_g)/(1+z_g) (the Σ_glob_wbh rate weighting);
    * stratum 'c' (0.25): flat in (u = ln(1+z), m = log10 M_z) on the reachable
      region (M_source ∈ [M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX] enforced by
      rejection; the unreachable wedge m > 7 + log10(1+z) yields no rows).

    All strata share the identical downstream path (extrinsic randomization,
    d_L = dist(z, h), SNR, CSV row); the per-row ``stratum`` column lets the
    estimator apply the measure-match rule (marginal legs ← 'a' only, joint
    (u, m)-conditional grid ← all rows).  Stratum labels and the 'b'/'c'
    coordinate draws come from generators SPAWNED off ``rng`` so the parent
    stream — and therefore the stratum-'a' path — is bit-identical to a
    non-mixture run for the degenerate proportions (1, 0, 0).

    Args:
        simulation_steps: Number of successful SNR computations to accumulate.
        cosmological_model: Model1CrossCheck instance for event sampling.
        h_value: Hubble constant value used for luminosity distance computation.
        simulation_index: Task index for unique CSV file naming (SLURM array compatibility).
        rng: Random number generator for reproducibility.
        use_gpu: Whether to use GPU acceleration.
        galaxy_catalog: Pruned GLADE+ handler; REQUIRED for the stratum-'b' draw
            when ``injection_mixture`` is on, unused otherwise.
        injection_mixture: Opt-in stratified 3-component sampling measure
            (default False = pure stratum-a, exactly the pre-#51 behaviour).
        stratum_probs: Mixture proportions (alpha_a, alpha_b, alpha_c); the
            production value is the ratified ``_STRATUM_PROBS`` — override is a
            test hook only.
        run_metadata_path: Existing run_metadata JSON to augment with the
            realized per-stratum counts at campaign completion (None = skip).
    """
    from darksiren_emri.constants import HOST_DRAW_Z_MAX, INJECTION_CSV_PATH
    from darksiren_emri.galaxy_catalogue.handler import ParameterSample
    from darksiren_emri.memory_management import MemoryManagement
    from darksiren_emri.parameter_estimation.parameter_estimation import (
        ParameterEstimation,
        WaveGeneratorType,
    )
    from darksiren_emri.physical_relations import dist, redshifted_mass

    def _alarm_handler(signum: int, frame: object) -> None:
        raise TimeoutError("Computation exceeded the alarm timeout")

    signal.signal(signal.SIGALRM, _alarm_handler)

    memory_management = MemoryManagement(use_gpu=use_gpu)
    memory_management.display_GPU_information()

    parameter_estimation = ParameterEstimation(
        waveform_generation_type=WaveGeneratorType.PN5_AAK,
        parameter_space=cosmological_model.parameter_space,
        use_gpu=use_gpu,
    )

    # Resolve CSV path: replace {h_label} and {index} placeholders. 4-decimal
    # precision matches the posterior-filename convention; existing 3-decimal
    # injection CSVs (e.g. h_0p73) still resolve since round(0.73, 4) == 0.73.
    h_label = str(round(h_value, 4)).replace(".", "p")
    csv_path = INJECTION_CSV_PATH.format(h_label=h_label, index=simulation_index)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    _ROOT_LOGGER.info(
        f"Starting injection campaign: h={h_value}, steps={simulation_steps}, "
        f"index={simulation_index}, output={csv_path}"
    )

    results: list[dict[str, float | str]] = []
    counter = 0
    iteration = 0
    parameter_samples_iter: Iterator[ParameterSample] = iter([])

    # [PHYSICS] Injection population depth = host-draw depth (issue #20): the
    # P_det grid built from these injections must span the full host-draw
    # volume, otherwise the selection function is blind above z_cut (the
    # pre-#20 hardcoded 0.5 capped the grid while hosts now reach z = 1.5).
    z_cut = HOST_DRAW_Z_MAX
    skipped_high_z = 0
    separatrix_sign_skips = 0
    timeout_count = 0
    stratum_counts: dict[str, int] = {"a": 0, "b": 0, "c": 0}

    # ── Issue #51 stratified 3-component mixture setup (opt-in) ──
    # SIZING_ANALYSIS.md §6: alpha = (0.50 a, 0.25 cat, 0.25 flat_um).
    # Stratum labels and the 'b'/'c' coordinate draws consume SPAWNED child
    # generators, never the parent ``rng``: SeedSequence spawning does not
    # advance the parent bit stream, so the stratum-'a' path (emcee batches +
    # extrinsic randomization on ``rng``) stays bit-identical to a
    # non-mixture run under degenerate proportions (1, 0, 0).
    strat_rng: np.random.Generator | None = None
    cat_rng: np.random.Generator | None = None
    flat_rng: np.random.Generator | None = None
    cat_z_arr: npt.NDArray[np.float64] | None = None
    cat_M_arr: npt.NDArray[np.float64] | None = None
    cat_cdf: npt.NDArray[np.float64] | None = None
    probs_arr: npt.NDArray[np.float64] | None = None
    if injection_mixture:
        if rng is None:
            msg = "injection_mixture=True requires an explicit rng (reproducibility)."
            raise ValueError(msg)
        probs_arr = np.asarray(stratum_probs, dtype=np.float64)
        if probs_arr.shape != (3,) or np.any(probs_arr < 0.0) or probs_arr.sum() <= 0.0:
            msg = f"stratum_probs must be 3 non-negative weights, got {stratum_probs}"
            raise ValueError(msg)
        probs_arr = probs_arr / probs_arr.sum()
        strat_rng, cat_rng, flat_rng = rng.spawn(3)
        if probs_arr[1] > 0.0:
            if galaxy_catalog is None:
                msg = "injection_mixture=True requires galaxy_catalog for the stratum-'b' draw."
                raise ValueError(msg)
            from darksiren_emri.emri_rate import R_eff_per_mbh
            from darksiren_emri.galaxy_catalogue.handler import InternalCatalogColumns

            cat_df = galaxy_catalog.reduced_galaxy_catalog
            cat_M_all = cat_df[InternalCatalogColumns.BH_MASS].to_numpy(dtype=np.float64)
            cat_z_all = cat_df[InternalCatalogColumns.REDSHIFT].to_numpy(dtype=np.float64)
            # Strict in-band membership: the handler's pruning keeps rows whose
            # ERROR BARS overlap the band; the injection draw needs the central
            # values inside the SOURCE-frame band and z in (0, z_cut].
            in_band = (
                (cat_M_all >= M_SOURCE_FRAME_MIN)
                & (cat_M_all <= M_SOURCE_FRAME_MAX)
                & (cat_z_all > 0.0)
                & (cat_z_all <= z_cut)
            )
            cat_M_arr = cat_M_all[in_band]
            cat_z_arr = cat_z_all[in_band]
            if cat_M_arr.size == 0:
                msg = "stratum-'b' draw: no catalogue rows inside the source band."
                raise ValueError(msg)
            # Row weight ∝ R_eff_per_mbh(M_g)/(1+z_g): the catalogue's
            # rate-weighted (z, M) profile — exactly the Σ_glob_wbh weighting
            # the acceptance criterion scores (SIZING_ANALYSIS.md §2 item 1,
            # 'cat' measure of §3). Precomputed ONCE; draws are O(log n)
            # inverse-CDF lookups.
            w_cat = np.asarray(R_eff_per_mbh(cat_M_arr), dtype=np.float64) / (1.0 + cat_z_arr)
            if not np.all(np.isfinite(w_cat)) or w_cat.sum() <= 0.0:
                msg = "stratum-'b' draw: non-finite or zero catalogue rate weights."
                raise ValueError(msg)
            cat_cdf = np.cumsum(w_cat)
            cat_cdf /= cat_cdf[-1]
    # Stratum-'c' box: u ~ U[0, ln(1+z_cut)], m ~ U[log10 M_min,
    # log10(M_max·(1+z_cut))] with SOURCE-band rejection (flat on the
    # reachable region; SIZING_ANALYSIS.md §3 'flat_um').
    _u_max_c = float(np.log1p(z_cut))
    _m_lo_c = float(np.log10(M_SOURCE_FRAME_MIN))
    _m_hi_c = float(np.log10(M_SOURCE_FRAME_MAX * (1.0 + z_cut)))
    # Provenance stamped into every injection row (stale-pool gate,
    # readiness sweep A2, 2026-07-03): h_inj alone cannot discriminate
    # pre-/post-dt² or shallow/deep pools (0.73 in every era).
    code_rev = _get_git_commit()
    _EMCEE_BATCH = 1000  # large batch to amortize MCMC overhead (z-rejection ~0 now
    # that z_cut = HOST_DRAW_Z_MAX matches the sampler depth; the pre-#20 93.5%
    # figure was measured at z_cut = 0.5)
    _LOG_INTERVAL = 100  # log every N successful events
    _GPU_FREE_INTERVAL = 50  # free GPU memory every N waveform computations
    _FLUSH_INTERVAL = 2000  # flush to disk every N events
    # Aligned with the main simulation loop's 90 s alarm (readiness sweep A1,
    # 2026-07-03): the injection SNR uses the FULL T-yr generator and depth 1.5
    # lifts M_z into corners never timing-profiled at the old 30 s budget;
    # timed-out events are DROPPED from the pool, so a timeout-rate correlation
    # with (d_L, M_z) would bias the p_det grid. Smoke test bins the counter.
    _TIMEOUT_S = 90

    # Normalize the rng for the per-event plunge-window t_plunge draw. Seeded
    # runs always pass an explicit rng (main() does); the fallback only affects
    # unseeded ad-hoc runs, where randomize_parameters previously created a
    # fresh default_rng per event anyway (non-reproducible either way).
    if rng is None:
        rng = np.random.default_rng()

    def _sigterm_handler(signum: int, frame: object) -> None:
        # SLURM sends SIGTERM at the wall-time cap. Flush the events accumulated
        # since the last periodic flush so up to _FLUSH_INTERVAL full-5yr-generator
        # SNR evaluations (the dominant GPU cost) are not lost (review SIM-04). Mirrors
        # data_simulation's SIGTERM flush; the CRB-side loss bound was already 5 rows.
        _ROOT_LOGGER.warning(
            "SIGTERM received (wall-time cap?) — flushing %d injection rows to %s",
            len(results),
            csv_path,
        )
        _flush_injection_results(results, csv_path)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    while counter < simulation_steps:
        # Stratum assignment (issue #51): fixed proportions from the ratified
        # mixture; drawn from the SPAWNED strat_rng so the parent stream (and
        # hence the stratum-'a' path) is untouched. Non-mixture runs make NO
        # extra rng calls — the default path is byte-identical to pre-#51.
        if injection_mixture:
            assert strat_rng is not None and probs_arr is not None
            stratum = _STRATA[int(strat_rng.choice(3, p=probs_arr))]
        else:
            stratum = "a"

        if stratum == "a":
            # Sample events from population model (status-quo Babak M1 emcee
            # draw on the widened box — SIZING_ANALYSIS.md §3 measure 'a').
            try:
                sample = next(parameter_samples_iter)
            except StopIteration:
                samples_list = cosmological_model.sample_emri_events(_EMCEE_BATCH)
                parameter_samples_iter = iter(samples_list)
                sample = next(parameter_samples_iter)

            # Population-depth consistency cut: z_cut = HOST_DRAW_Z_MAX so the
            # P_det grid spans exactly the host-draw volume. NOT a p_det = 0
            # claim — post-dt² the horizon reaches z ~ 1.5+ (issue #20 retired
            # the pre-dt² "24/69500 detections at z < 0.18" justification).
            if sample.redshift > z_cut:
                skipped_high_z += 1
                continue

            event_z = sample.redshift
            # sample.M is SOURCE-frame; lifted below.
            # Eq. (4.7) in Maggiore (2008) GW Vol. 1 §4.1.4: M_z = M_source·(1+z)
            redshifted_M = redshifted_mass(sample.M, event_z)  # M_z = M·(1+z)
        elif stratum == "b":
            # Catalogue-coverage draw ('cat' measure): a pruned-catalogue row
            # with probability ∝ R_eff(M_g)/(1+z_g), inverse-CDF lookup.
            # Catalogue masses/redshifts are SOURCE-frame rest quantities.
            assert cat_rng is not None and cat_cdf is not None
            assert cat_z_arr is not None and cat_M_arr is not None
            g_idx = int(np.searchsorted(cat_cdf, cat_rng.random(), side="right"))
            g_idx = min(g_idx, cat_cdf.size - 1)
            event_z = float(cat_z_arr[g_idx])
            # Same source→detector lift as stratum 'a' (Maggiore 2008 §4.1.4).
            redshifted_M = redshifted_mass(float(cat_M_arr[g_idx]), event_z)
        else:
            # Flat-(u, m) draw ('flat_um' measure): u = ln(1+z), m = log10 M_z
            # uniform on the box, SOURCE-band rejection keeps the density flat
            # on the reachable region — the unreachable wedge
            # m > log10(M_SOURCE_FRAME_MAX) + log10(1+z) yields no rows by
            # construction (rejection), SIZING_ANALYSIS.md §2 item 2.
            # NB: this stratum draws DIRECTLY in detector-frame m = log10 M_z;
            # M_z is used as-is (NO second (1+z) lift).
            assert flat_rng is not None
            while True:
                u_draw = float(flat_rng.uniform(0.0, _u_max_c))
                m_draw = float(flat_rng.uniform(_m_lo_c, _m_hi_c))
                z_candidate = float(np.expm1(u_draw))
                M_z_candidate = float(10.0**m_draw)
                M_source_candidate = M_z_candidate / (1.0 + z_candidate)
                if M_SOURCE_FRAME_MIN <= M_source_candidate <= M_SOURCE_FRAME_MAX:
                    break
            event_z = z_candidate
            redshifted_M = M_z_candidate  # already detector-frame

        if iteration % _GPU_FREE_INTERVAL == 0:
            memory_management.gpu_usage_stamp()
            memory_management.free_gpu_memory_if_pressured()
        iteration += 1

        if counter % _LOG_INTERVAL == 0:
            _ROOT_LOGGER.info(
                f"Injection campaign: {counter} / {iteration} successful SNR computations "
                f"({skipped_high_z} high-z skipped)."
            )

        # Randomize extrinsic parameters (sky angles, orbital phases, etc.) —
        # identical full extrinsic randomization in EVERY stratum
        # (SIZING_ANALYSIS.md §5 caveat (ii)).
        parameter_estimation.parameter_space.randomize_parameters(rng=rng)

        # Set the DETECTOR-FRAME (redshifted) mass M_z: FEW expects M_z, and the
        # injection CSV "M" column holds M_z consistently with the event CRBs
        # (parameter_space.set_host_galaxy_parameters) and the p_det grid axis
        # (simulation_detection_probability.py, which does not re-lift).
        # Maggiore (2008) GW Vol. 1 §4.1.4; Babak et al. (2017) arXiv:1703.09722.
        #
        # No detector-frame truncation (issue #51, supersedes readiness sweep
        # A3): parameter_space.M.upper_limit is now the (1+z_max)-lifted image
        # of the source-frame draw band (Model1CrossCheck), so
        # M_z = M_source*(1+z) <= M_SOURCE_FRAME_MAX*(1+max_redshift) holds by
        # construction in every stratum — pool and CRB event set share the full
        # support with no extra clamp (the old A3 pool/event consistency
        # argument is preserved structurally instead of by a cut).
        parameter_estimation.parameter_space.M.value = redshifted_M

        # Set luminosity distance with candidate h value (injection pipeline does not use
        # set_host_galaxy_parameters — it sets d_L directly since no host galaxy is needed).
        luminosity_distance = dist(event_z, h=h_value)
        parameter_estimation.parameter_space.luminosity_distance.value = luminosity_distance

        # Compute SNR only (no Fisher matrix, no CRB)
        try:
            signal.alarm(_TIMEOUT_S)
            # [PHYSICS] Plunge-window initial conditions — identical convention
            # and call as the CRB loop (data_simulation): t_plunge ~ U[0, T],
            # p0 from the PN5 time-to-plunge root-find, AFTER M_z is set.
            # docs/derivations/plunge_window_initial_conditions.md;
            # --snapshot_ics restores the retired p0 ~ U[10, 16] draw.
            if not snapshot_ics:
                from darksiren_emri.plunge_window import (  # noqa: PLC0415
                    draw_plunge_window_initial_conditions,
                )

                draw_plunge_window_initial_conditions(
                    parameter_estimation.parameter_space,
                    rng,
                    parameter_estimation.T,
                )
            # warnings-as-errors scoped so it is restored on every exit path and
            # cannot leak into the next iteration's population sampling (review SIM-02).
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                snr = parameter_estimation.compute_signal_to_noise_ratio()
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
                f"Caught ParameterOutOfBoundsError: {str(e)}. Continue with new parameters..."
            )
            continue
        except RuntimeError as e:
            _ROOT_LOGGER.warning(
                f"Caught RuntimeError during waveform generation: {str(e)}. Continue..."
            )
            continue
        except ValueError as e:
            if "EllipticK" in str(e):
                _ROOT_LOGGER.warning("Caught EllipticK error from waveform generator. Continue...")
                continue
            elif "Brent root solver does not converge" in str(e):
                _ROOT_LOGGER.warning(
                    "Caught Brent root solver error. Continue with new parameters..."
                )
                continue
            elif "must have different signs" in str(e):
                # few's separatrix kernel brentq at EVOLVED (a, e(t), x(t))
                # inside get_p_at_t/trajectory integration (campaign #51
                # pilot #2, job 6073027: killed 22/60 tasks before this
                # branch existed) — skippable numerical corner of the draw.
                separatrix_sign_skips += 1
                _ROOT_LOGGER.warning(
                    "Caught separatrix-kernel sign error during trajectory root-find "
                    "(%d total). Continue with new parameters...",
                    separatrix_sign_skips,
                )
                continue
            else:
                raise
        except ZeroDivisionError:
            _ROOT_LOGGER.warning(
                "Caught ZeroDivisionError during trajectory integration. Continue..."
            )
            continue
        except TimeoutError:
            # G9 gate: params logged for timeout binning (smoke test checks
            # for (d_L, M_z) correlation before full-campaign sizing).
            timeout_count += 1
            _ROOT_LOGGER.warning(
                "Injection waveform/SNR computation timed out (>%ss, %d total). "
                "Skipping event... params=%s",
                _TIMEOUT_S,
                timeout_count,
                parameter_estimation.parameter_space._parameters_to_dict(),
            )
            continue
        finally:
            signal.alarm(0)  # cancel on every path (review SIM-01/SIM-02)

        # Store ALL events regardless of SNR (per D-03: do NOT threshold)
        results.append(
            {
                "z": event_z,
                # Store the DETECTOR-FRAME mass M_z (not the source-frame mass) so the
                # injection CSV "M" column matches the value FEW saw and the p_det grid
                # axis expects (simulation_detection_probability.py no longer re-lifts).
                "M": redshifted_M,
                "phiS": parameter_estimation.parameter_space.phiS.value,
                "qS": parameter_estimation.parameter_space.qS.value,
                "SNR": float(snr),
                "h_inj": h_value,
                "luminosity_distance": luminosity_distance,
                # Provenance (stale-pool gate): z_cut discriminates pool depth
                # eras, code_rev ties rows to the generating commit.
                "z_cut": z_cut,
                "code_rev": code_rev,
                # Issue #51: sampling-measure stratum of this row (estimator
                # measure-match rule, SIZING_ANALYSIS.md §4).
                "stratum": stratum,
                # Plunge-window provenance (2026-07-28 convention change): the
                # drawn plunge time (yr; NaN under --snapshot_ics) and the
                # derived p0, so pool rows are auditable against the
                # t_insp(p0) = t_plunge convention and pool eras are
                # discriminable. docs/derivations/plunge_window_initial_conditions.md.
                "t_plunge_yr": parameter_estimation.parameter_space.t_plunge_yr,
                "p0": parameter_estimation.parameter_space.p0.value,
            }
        )
        counter += 1
        stratum_counts[stratum] += 1

        # Flush to disk periodically so SLURM timeouts don't lose all work
        if counter % _FLUSH_INTERVAL == 0:
            _flush_injection_results(results, csv_path)
            _ROOT_LOGGER.info(f"Flushed {len(results)} events to {csv_path}")

    # Final write
    _flush_injection_results(results, csv_path)
    _ROOT_LOGGER.info(
        f"Injection campaign complete: {len(results)} events stored to {csv_path} "
        f"(skipped: {skipped_high_z} high-z, "
        f"{separatrix_sign_skips} separatrix-sign, "
        f"{timeout_count} timeouts @ {_TIMEOUT_S}s); "
        f"realized stratum counts: a={stratum_counts['a']}, "
        f"b={stratum_counts['b']}, c={stratum_counts['c']} "
        f"(mixture={'on' if injection_mixture else 'off'})"
    )

    # Record the realized stratum counts in the run metadata (the
    # --injection_mixture flag itself is already captured via cli_args).
    if run_metadata_path is not None and os.path.isfile(run_metadata_path):
        try:
            with open(run_metadata_path) as f:
                metadata = json.load(f)
            metadata["injection_mixture"] = injection_mixture
            metadata["injection_stratum_counts"] = dict(stratum_counts)
            with open(run_metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            _ROOT_LOGGER.info(f"Stratum counts recorded in {run_metadata_path}")
        except (OSError, json.JSONDecodeError) as e:
            _ROOT_LOGGER.warning(f"Could not update run metadata with stratum counts: {e}")


def evaluate(
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
    h_value: float,
    *,
    num_workers: int | None = None,
    catalog_only: bool = False,
    pdet_dl_bins: int = 60,
    pdet_mass_bins: int = 40,
    pdet_estimator: str = "local_linear",
    fisher_cond_threshold: float = 1e16,
    # [PHYSICS] production default since 2026-07-26 (MULTISEED_READOUT_20260726.md)
    normalization_mode: str = "generator_marginal",
    base_seed: int | None = None,
    allow_low_pdet_coverage: bool = False,
    h_values: list[float] | None = None,
    smear_global_selection: bool = False,
    pdet_z_resolved: bool = True,
    # FIX-3 §7.1 (default OFF, byte-identical):
    # docs/derivations/fix3_zmz_catalog_selection.md.
    pdet_wbh_z_resolved: bool = False,
    host_z_kernel: str = "auto",
    host_mass_kernel: str = "auto",
    # INSTRUMENTATION (default OFF, byte-identical): frozen-g_frac counterfactual.
    freeze_g_frac_ref_h: float | None = None,
    # INSTRUMENTATION (default OFF, byte-identical): N-2 selection-in-numerator
    # counterfactual ("off" | "1d").
    selection_in_completion_numerator: str = "off",
) -> None:
    from darksiren_emri.bayesian_inference.bayesian_statistics import BayesianStatistics

    hubble_constant_evaluation = BayesianStatistics()
    hubble_constant_evaluation.evaluate(
        galaxy_catalog,
        cosmological_model,
        h_value,
        num_workers=num_workers,
        catalog_only=catalog_only,
        pdet_dl_bins=pdet_dl_bins,
        pdet_mass_bins=pdet_mass_bins,
        pdet_estimator=pdet_estimator,
        fisher_cond_threshold=fisher_cond_threshold,
        normalization_mode=normalization_mode,
        base_seed=base_seed if base_seed is not None else 0,
        allow_low_pdet_coverage=allow_low_pdet_coverage,
        h_values=h_values,
        smear_global_selection=smear_global_selection,
        pdet_z_resolved=pdet_z_resolved,
        pdet_wbh_z_resolved=pdet_wbh_z_resolved,
        host_z_kernel=host_z_kernel,
        host_mass_kernel=host_mass_kernel,
        freeze_g_frac_ref_h=freeze_g_frac_ref_h,
        selection_in_completion_numerator=selection_in_completion_numerator,
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

    # Injected-truth H0 for the figure truth-lines: use the fiducial constant rather
    # than a hardcoded 0.73 literal so every truth marker agrees if H changes
    # (review PLT-03/04). The nested generators below close over this.
    from darksiren_emri.constants import H as TRUTH_H
    from darksiren_emri.plotting._data import PARAMETER_NAMES, reconstruct_covariance
    from darksiren_emri.plotting._helpers import save_figure
    from darksiren_emri.plotting._style import apply_style

    # VIZ-01: auto-detect a local LaTeX install and route to the matching style.
    if shutil.which("latex"):
        apply_style(use_latex=True)
        _ROOT_LOGGER.info("LaTeX detected; rendering figures with text.usetex=True")
    else:
        apply_style()
        _ROOT_LOGGER.info("LaTeX not detected; using mathtext fallback")
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
            from darksiren_emri.physical_relations import dist_to_redshift

            df["redshift"] = df["luminosity_distance"].apply(
                lambda d: dist_to_redshift(float(d), h=TRUTH_H)
            )
        return df

    def _load_posteriors(
        subdir: str,
    ) -> tuple[np.ndarray, list[np.ndarray]] | None:
        """Load posterior JSONs from *subdir*, return (h_values, event_posteriors)."""
        from darksiren_emri.bayesian_inference.posterior_combination import (
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

    def _load_injection_data() -> tuple[np.ndarray, np.ndarray] | None:
        """Pool the injection CSVs ``(z, SNR)`` from ``<output_dir>/injections``
        for the selection-function figures (fig04/fig09).  Returns ``(z, SNR)``
        over all injected events (detected and sub-threshold), or ``None`` when
        no injection pool is linked into the run.
        """
        inj_dir = os.path.join(output_dir, "injections")
        csvs = sorted(glob.glob(os.path.join(inj_dir, "injection_*.csv")))
        if not csvs:
            return None
        frames = [pd.read_csv(f, usecols=["z", "SNR"]) for f in csvs]
        df = pd.concat(frames, ignore_index=True)
        return (
            df["z"].to_numpy(dtype=np.float64),
            df["SNR"].to_numpy(dtype=np.float64),
        )

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
    # Uses the canonical raw Σ log L_i loader (Phase A); see
    # darksiren_emri/plotting/_helpers.py::load_canonical_combined_posterior
    # so that fig01 agrees with paper_h0_posterior, fig08 left panel, and
    # paper_m_z_improvement top-right on the MAP.
    def _gen_h0_posterior_combined() -> tuple[object, object] | None:
        if post_data is None:
            return None
        from darksiren_emri.plotting._colors import VARIANT_STYLE
        from darksiren_emri.plotting._helpers import load_canonical_combined_posterior
        from darksiren_emri.plotting.bayesian_plots import plot_combined_posterior

        try:
            h_vals, combined, _meta = load_canonical_combined_posterior(
                Path(output_dir), "posteriors"
            )
        except FileNotFoundError:
            return None
        fig, ax = plot_combined_posterior(
            h_vals,
            combined,
            TRUTH_H,
            label=r"Without $M_z$",
            color=VARIANT_STYLE["no_mass"][0],
            linestyle=VARIANT_STYLE["no_mass"][1],
        )
        if post_data_with is not None:
            try:
                h_w, comb_w, _meta_w = load_canonical_combined_posterior(
                    Path(output_dir), "posteriors_with_bh_mass"
                )
            except FileNotFoundError:
                h_w, comb_w = None, None
            if h_w is not None and comb_w is not None:
                plot_combined_posterior(
                    h_w,
                    comb_w,
                    TRUTH_H,
                    label=r"With $M_z$",
                    color=VARIANT_STYLE["with_mass"][0],
                    linestyle=VARIANT_STYLE["with_mass"][1],
                    show_references=False,
                    ax=ax,
                )
        return fig, ax

    manifest.append(("fig01_h0_posterior_combined", _gen_h0_posterior_combined))

    # 2. Individual event posteriors -- per-event curves come from per-event
    # arrays, but the combined overlay uses the canonical raw Σ log L_i loader
    # for consistency with fig01.
    def _gen_event_posteriors() -> tuple[object, object] | None:
        if post_data is None:
            return None
        from darksiren_emri.plotting._colors import VARIANT_WITH_MASS
        from darksiren_emri.plotting._helpers import load_canonical_combined_posterior
        from darksiren_emri.plotting.bayesian_plots import (
            plot_event_posteriors,
        )

        h_vals, event_posts = post_data
        try:
            _h_canon, combined, _meta = load_canonical_combined_posterior(
                Path(output_dir), "posteriors"
            )
        except FileNotFoundError:
            return None
        fig, ax = plot_event_posteriors(h_vals, event_posts, TRUTH_H, combined_posterior=combined)
        # Overlay with-M_z canonical combined posterior
        if post_data_with is not None:
            try:
                h_w, comb_w, _meta_w = load_canonical_combined_posterior(
                    Path(output_dir), "posteriors_with_bh_mass"
                )
            except FileNotFoundError:
                h_w, comb_w = None, None
            if h_w is not None and comb_w is not None:
                norm_w = float(np.trapezoid(comb_w, h_w))
                if norm_w > 0:
                    comb_w = comb_w / norm_w
                ax.plot(
                    h_w,
                    comb_w,
                    color=VARIANT_WITH_MASS,
                    linewidth=2,
                    label=r"Combined (with $M_z$)",
                )
                ax.legend(fontsize="small")
        return fig, ax

    manifest.append(("fig02_event_posteriors", _gen_event_posteriors))

    # 3. SNR distribution -- needs CRB data with SNR column
    def _gen_snr_distribution() -> tuple[object, object] | None:
        if crb_df is None or "SNR" not in crb_df.columns:
            return None
        from darksiren_emri.plotting.bayesian_plots import plot_snr_distribution

        return plot_snr_distribution(crb_df["SNR"].to_numpy(dtype=np.float64))

    manifest.append(("fig03_snr_distribution", _gen_snr_distribution))

    # 4. Detection yield -- needs redshift column in CRB
    def _gen_detection_yield() -> tuple[object, object] | None:
        from darksiren_emri.constants import SNR_THRESHOLD
        from darksiren_emri.plotting.evaluation_plots import plot_detection_yield

        inj = _load_injection_data()
        if inj is None:
            return None
        injected_z, snr = inj
        detected_z = injected_z[snr >= float(SNR_THRESHOLD)]
        return plot_detection_yield(injected_z, detected_z)

    manifest.append(("fig04_detection_yield", _gen_detection_yield))

    # 5. Sky localization (Mollweide)
    def _gen_sky_localization() -> tuple[object, object] | None:
        if crb_df is None or not {"qS", "phiS", "SNR"}.issubset(crb_df.columns):
            return None
        from darksiren_emri.plotting.sky_plots import plot_sky_localization_mollweide

        theta_s = crb_df["qS"].to_numpy(dtype=np.float64)
        phi_s = crb_df["phiS"].to_numpy(dtype=np.float64)
        snr = crb_df["SNR"].to_numpy(dtype=np.float64)
        return plot_sky_localization_mollweide(theta_s, phi_s, snr)

    manifest.append(("fig05_sky_localization", _gen_sky_localization))

    # 6. Fisher ellipses (3 parameter pairs)
    def _gen_fisher_ellipses() -> tuple[object, object] | None:
        if crb_df is None or len(crb_df) < 1:
            return None
        from darksiren_emri.plotting.fisher_plots import plot_fisher_ellipses

        row = crb_df.iloc[0]
        cov = reconstruct_covariance(row)
        param_vals = np.array([float(row.get(p, 0.0)) for p in PARAMETER_NAMES], dtype=np.float64)
        return plot_fisher_ellipses(cov, param_vals)

    manifest.append(("fig06_fisher_ellipses", _gen_fisher_ellipses))

    # 7. Corner plot
    def _gen_corner_plot() -> tuple[object, object] | None:
        if crb_df is None or len(crb_df) < 1:
            return None
        from darksiren_emri.plotting.fisher_plots import plot_fisher_corner

        row = crb_df.iloc[0]
        cov = reconstruct_covariance(row)
        param_vals = np.array([float(row.get(p, 0.0)) for p in PARAMETER_NAMES], dtype=np.float64)
        return plot_fisher_corner(cov, param_vals)

    manifest.append(("fig07_corner_plot", _gen_corner_plot))

    # 8. H0 convergence -- left panel uses canonical raw Σ log L_i (Phase A)
    # so that the MAP visible on fig08 matches fig01 / paper_h0_posterior.
    def _gen_h0_convergence() -> tuple[object, object] | None:
        if post_data is None:
            return None
        from darksiren_emri.constants import H as TRUE_H
        from darksiren_emri.plotting._helpers import load_canonical_combined_posterior
        from darksiren_emri.plotting.convergence_analysis import (
            compute_m_z_improvement_bank,
        )
        from darksiren_emri.plotting.convergence_plots import plot_h0_convergence

        h_vals, event_posts = post_data
        h_alt, ep_alt = post_data_with if post_data_with is not None else (None, None)
        # VIZ-02: try to load the cached improvement bank for the right-panel band.
        # Cached on disk by compute_m_z_improvement_bank — one JSON read per call.
        try:
            bootstrap_bank = compute_m_z_improvement_bank(Path(output_dir), h_true=float(TRUE_H))
        except (FileNotFoundError, ValueError, KeyError):
            bootstrap_bank = None
        canonical_no_mass: tuple[np.ndarray, np.ndarray] | None = None
        canonical_with_mass: tuple[np.ndarray, np.ndarray] | None = None
        try:
            h_c, p_c, _m = load_canonical_combined_posterior(Path(output_dir), "posteriors")
            canonical_no_mass = (h_c, p_c)
        except FileNotFoundError:
            pass
        if post_data_with is not None:
            try:
                h_c2, p_c2, _m2 = load_canonical_combined_posterior(
                    Path(output_dir), "posteriors_with_bh_mass"
                )
                canonical_with_mass = (h_c2, p_c2)
            except FileNotFoundError:
                pass
        return plot_h0_convergence(
            h_vals,
            event_posts,
            true_h=float(TRUE_H),
            h_values_alt=h_alt,
            event_posteriors_alt=ep_alt,
            bootstrap_bank=bootstrap_bank,
            canonical_no_mass=canonical_no_mass,
            canonical_with_mass=canonical_with_mass,
        )

    manifest.append(("fig08_h0_convergence", _gen_h0_convergence))

    # 9. Detection efficiency
    def _gen_detection_efficiency() -> tuple[object, object] | None:
        from darksiren_emri.constants import SNR_THRESHOLD
        from darksiren_emri.plotting.evaluation_plots import plot_detection_efficiency

        inj = _load_injection_data()
        if inj is None:
            return None
        injected_z, snr = inj
        detected = snr >= float(SNR_THRESHOLD)
        return plot_detection_efficiency(injected_z, detected)

    manifest.append(("fig09_detection_efficiency", _gen_detection_efficiency))

    # 10. LISA PSD with noise decomposition
    def _gen_lisa_psd() -> tuple[object, object] | None:
        from darksiren_emri.plotting.model_plots import plot_lisa_psd

        freqs = np.geomspace(1e-5, 1.0, 1000)
        return plot_lisa_psd(freqs, decompose=True)

    manifest.append(("fig10_lisa_psd", _gen_lisa_psd))

    # 11. Luminosity distance d_L(z) with multi-H0
    def _gen_distance_redshift() -> tuple[object, object] | None:
        from darksiren_emri.physical_relations import dist_vectorized
        from darksiren_emri.plotting.physical_relations_plots import (
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
        from darksiren_emri.plotting.fisher_plots import plot_parameter_uncertainties

        param_cols = [p for p in PARAMETER_NAMES if p in crb_df.columns]
        if not param_cols:
            return None
        return plot_parameter_uncertainties(crb_df, crb_df[param_cols])

    manifest.append(("fig12_uncertainty_violins", _gen_uncertainty_violins))

    # 13. Characteristic strain
    def _gen_characteristic_strain() -> tuple[object, object] | None:
        from darksiren_emri.plotting.model_plots import plot_characteristic_strain

        return plot_characteristic_strain()

    manifest.append(("fig13_characteristic_strain", _gen_characteristic_strain))

    # 14. CRB coverage (3D parameter-space scatter per D-11)
    def _gen_crb_coverage() -> tuple[object, object] | None:
        if crb_df is None or not {"M", "qS", "phiS"}.issubset(crb_df.columns):
            return None
        from darksiren_emri.plotting.fisher_plots import plot_crb_coverage

        M = crb_df["M"].to_numpy(dtype=np.float64)
        qS = crb_df["qS"].to_numpy(dtype=np.float64)
        phiS = crb_df["phiS"].to_numpy(dtype=np.float64)
        return plot_crb_coverage(M, qS, phiS)

    manifest.append(("fig14_crb_coverage", _gen_crb_coverage))

    # 15. H0-in-context forest plot (Di Valentino-style). Replaces the retired
    # campaign dashboard, which only re-rendered fig01/03/04/05 at thumbnail
    # scale with no new information and colliding labels. Ships with a default
    # literature dataset (Planck 2018 / SH0ES / GWTC-3) plus this work's MAP/HDI.
    def _gen_h0_forest() -> tuple[object, object] | None:
        from darksiren_emri.plotting.validation_plots import plot_h0_forest

        return plot_h0_forest()

    manifest.append(("fig15_h0_forest", _gen_h0_forest))

    # 16. Catalog completeness + per-event coverage (Phase C)
    def _gen_catalog_completeness() -> tuple[object, object] | None:
        host_csv = Path(output_dir) / "diagnostics" / "host_counts.csv"
        if not host_csv.is_file():
            # Try to build it from inference logs on the fly.
            try:
                from darksiren_emri.analysis.parse_host_counts import build_host_count_csv

                build_host_count_csv(Path(output_dir))
            except FileNotFoundError:
                return None
        import pandas as pd

        from darksiren_emri.plotting._helpers import get_figure
        from darksiren_emri.plotting.catalog_plots import (
            gehrels_2016_reference_completeness,
            plot_event_catalog_coverage,
            plot_glade_completeness,
        )

        host_counts = pd.read_csv(host_csv)
        # Optional join to CRB CSV to obtain per-event d_L. The CSV index
        # corresponds to event_idx by construction.
        d_l_array: np.ndarray | None = None
        if crb_df is not None and "luminosity_distance" in crb_df.columns:
            dl_values = crb_df["luminosity_distance"].to_numpy(dtype=np.float64)
            if len(dl_values) >= len(host_counts):
                d_l_array = dl_values[: len(host_counts)]

        # Empirical completeness proxy: per-d_L bin, fraction of events with
        # at least one catalog host. Same axis as the schematic reference.
        fig, axes = get_figure(nrows=1, ncols=2, preset="double")
        ax_left, ax_right = axes[0], axes[1]
        # Left panel: schematic GLADE+ completeness curve.
        if d_l_array is not None:
            d_l_grid = np.linspace(
                max(d_l_array.min(), 1e-3), d_l_array.max(), 80, dtype=np.float64
            )
            ref = gehrels_2016_reference_completeness(d_l_grid)
            # Empirical coverage curve over the same grid.
            edges = np.linspace(d_l_grid.min(), d_l_grid.max(), 13)
            centers = 0.5 * (edges[:-1] + edges[1:])
            emp = np.zeros_like(centers)
            for i in range(len(centers)):
                mask = (d_l_array >= edges[i]) & (d_l_array < edges[i + 1])
                if mask.any():
                    emp[i] = float((host_counts["n_without_mass"].to_numpy()[mask] > 0).mean())
            plot_glade_completeness(
                centers,
                emp,
                reference_curve=np.interp(centers, d_l_grid, ref),
                label="Empirical coverage (this campaign)",
                reference_label="Schematic GLADE+ reference",
                ax=ax_left,
            )
            ax_left.set_title("Catalog completeness", fontsize="medium")
        else:
            ax_left.text(0.5, 0.5, "No CRB d_L data", ha="center", va="center")
        # Right panel: per-event host counts + reduction.
        plot_event_catalog_coverage(host_counts, d_l_per_event=d_l_array, ax=ax_right)
        ax_right.set_title("Host candidates per event", fontsize="medium")
        fig.tight_layout()
        return fig, axes

    manifest.append(("fig16_catalog_completeness", _gen_catalog_completeness))

    # 17. Detailed single-event multi-panel (Phase D)
    # Works against any data directory whose 2D posteriors include the
    # `galaxy_likelihoods` key. The cluster Phase 48 posteriors were stripped
    # to save disk; in that case the figure transparently falls back to the
    # nearest available unstripped data directory (`simulations/`).
    def _gen_single_event_detail() -> tuple[object, object] | None:
        from darksiren_emri.plotting.single_event_detail import (
            plot_single_event_detail,
            select_representative_event_id,
        )

        candidate_dirs = [Path(output_dir)]
        # Look one and two levels up for an unstripped sibling dataset.
        for parent_levels in (2, 3):
            try:
                candidate = Path(output_dir).resolve().parents[parent_levels - 1] / "simulations"
                if candidate.is_dir():
                    candidate_dirs.append(candidate)
            except IndexError:
                pass
        chosen_dir: Path | None = None
        for cand in candidate_dirs:
            with_mass_dir = cand / "posteriors_with_bh_mass"
            if not with_mass_dir.is_dir():
                continue
            # Detect whether the JSONs carry galaxy_likelihoods (unstripped).
            sample_files = sorted(with_mass_dir.glob("h_*.json"))
            if not sample_files:
                continue
            with open(sample_files[0]) as fh:
                first = json.load(fh)
            if "galaxy_likelihoods" in first and first["galaxy_likelihoods"]:
                chosen_dir = cand
                break
        if chosen_dir is None:
            _ROOT_LOGGER.info(
                "fig17 skipped: no posteriors_with_bh_mass directory with galaxy_likelihoods found "
                "(cluster Phase 48 data was stripped; sync a sibling dataset with the key intact)."
            )
            return None
        try:
            event_id = select_representative_event_id(chosen_dir, percentile=0.5)
            return plot_single_event_detail(chosen_dir, event_id)
        except (FileNotFoundError, KeyError, ValueError) as e:
            _ROOT_LOGGER.warning("fig17 generation failed: %s", e)
            return None

    manifest.append(("fig17_single_event_detail", _gen_single_event_detail))

    # 18. Closure-test posterior overlay (Phase F1)
    def _gen_closure_test_overlay() -> tuple[object, object] | None:
        from darksiren_emri.plotting.paper_figures import plot_closure_test_overlay

        project_root = Path(__file__).resolve().parents[1]
        sim_root = project_root / "simulations"
        # Discover closure-test directories: `closure_h{0p60,0p65,0p70,0p73,0p75,...}/posteriors`
        # plus the production h=0.73 run as the "self-consistency" point.
        h_runs: dict[float, Path] = {}
        for closure_dir in sorted(sim_root.glob("closure_h*")):
            posts = closure_dir / "posteriors"
            if not posts.is_dir():
                continue
            # Parse h_true from directory name: closure_h0p65 → 0.65
            name = closure_dir.name
            try:
                tag = name.split("_h", 1)[1]
                tag = tag.split("_", 1)[0]
                h_str = tag.replace("p", ".")
                h_true = float(h_str)
            except (IndexError, ValueError):
                continue
            h_runs[h_true] = closure_dir
        # Add the production h=0.73 run if not already covered.
        prod_dir = Path(output_dir)
        if (prod_dir / "posteriors").is_dir() and 0.73 not in h_runs:
            h_runs[0.73] = prod_dir
        if len(h_runs) < 2:
            _ROOT_LOGGER.info("fig18 skipped: need ≥2 closure runs (have %d)", len(h_runs))
            return None
        try:
            return plot_closure_test_overlay(h_runs)
        except FileNotFoundError as e:
            _ROOT_LOGGER.warning("fig18 generation failed: %s", e)
            return None

    manifest.append(("fig18_closure_test", _gen_closure_test_overlay))

    # 19. Info monotonicity (Phase F2) — per-event HDI68 scatter (1D vs 2D)
    def _gen_info_monotonicity() -> tuple[object, object] | None:
        from darksiren_emri.plotting.evaluation_plots import plot_info_monotonicity

        try:
            return plot_info_monotonicity(Path(output_dir))
        except (FileNotFoundError, ValueError) as e:
            _ROOT_LOGGER.info("fig19 skipped: %s", e)
            return None

    manifest.append(("fig19_info_monotonicity", _gen_info_monotonicity))

    # 20. P_det surface from injection campaign (Phase F3)
    def _gen_pdet_surface() -> tuple[object, object] | None:
        from darksiren_emri.plotting.evaluation_plots import plot_pdet_surface

        # Prefer the project root's `simulations/injections/` campaign data;
        # fall back to a sibling `injections/` under output_dir if present.
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            project_root / "simulations" / "injections" / "injection_h_0p73_task_*.csv",
            Path(output_dir) / "injections" / "injection_h_0p73_task_*.csv",
        ]
        for pat in candidates:
            try:
                return plot_pdet_surface(str(pat), snr_threshold=20.0)
            except (FileNotFoundError, ValueError):
                continue
        _ROOT_LOGGER.info("fig20 skipped: no injection campaign CSVs available")
        return None

    manifest.append(("fig20_pdet_surface", _gen_pdet_surface))

    # 21-23. Per-pixel HEALPix catalog completeness (Change 5, GMV-2022). These use
    # ONLY the committed frozen m_th map (no run data), so they always render and
    # show the pixelation/ZoA the inference's completeness actually uses.
    def _gen_completeness_mth_skymap() -> tuple[object, object] | None:
        try:
            from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build
            from darksiren_emri.plotting.completeness_plots import plot_completeness_sky_map

            return plot_completeness_sky_map(from_cache_or_build(), quantity="m_th")
        except (FileNotFoundError, ValueError):
            _ROOT_LOGGER.info("fig21 skipped: no m_th map / catalog available")
            return None

    manifest.append(("fig21_completeness_mth_skymap", _gen_completeness_mth_skymap))

    def _gen_completeness_fk_skymap() -> tuple[object, object] | None:
        try:
            from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build
            from darksiren_emri.plotting.completeness_plots import plot_completeness_sky_map

            return plot_completeness_sky_map(from_cache_or_build(), quantity="f_k", z=0.05)
        except (FileNotFoundError, ValueError):
            _ROOT_LOGGER.info("fig22 skipped: no m_th map / catalog available")
            return None

    manifest.append(("fig22_completeness_fk_skymap_z0p05", _gen_completeness_fk_skymap))

    def _gen_sky_averaged_completeness() -> tuple[object, object] | None:
        try:
            from darksiren_emri.constants import HOST_DRAW_Z_MAX
            from darksiren_emri.galaxy_catalogue.pixel_completeness import from_cache_or_build
            from darksiren_emri.plotting.completeness_plots import (
                plot_sky_averaged_completeness,
            )

            # Track the campaign depth so the figure shows the full host volume.
            return plot_sky_averaged_completeness(from_cache_or_build(), z_max=HOST_DRAW_Z_MAX)
        except (FileNotFoundError, ValueError):
            _ROOT_LOGGER.info("fig23 skipped: no m_th map / catalog available")
            return None

    manifest.append(("fig23_sky_averaged_completeness", _gen_sky_averaged_completeness))

    # 16. Paper figure: H0 posterior comparison (D-01, D-09)
    def _gen_paper_h0_posterior() -> tuple[object, object] | None:
        from darksiren_emri.plotting.paper_figures import plot_h0_posterior_comparison

        try:
            return plot_h0_posterior_comparison(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError):
            return None

    manifest.append(("paper_h0_posterior", _gen_paper_h0_posterior))

    # 17. Paper figure: single-event likelihoods
    def _gen_paper_single_event() -> tuple[object, object] | None:
        from darksiren_emri.plotting.paper_figures import plot_single_event_likelihoods

        try:
            return plot_single_event_likelihoods(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError, ValueError):
            return None

    manifest.append(("paper_single_event", _gen_paper_single_event))

    # 18. Paper figure: posterior convergence
    def _gen_paper_convergence() -> tuple[object, object] | None:
        from darksiren_emri.plotting.paper_figures import plot_posterior_convergence

        try:
            return plot_posterior_convergence(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError):
            return None

    manifest.append(("paper_convergence", _gen_paper_convergence))

    # 19. Paper figure: SNR distribution
    def _gen_paper_snr_distribution() -> tuple[object, object] | None:
        from darksiren_emri.plotting.paper_figures import plot_snr_distribution

        try:
            return plot_snr_distribution(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError, ValueError):
            return None

    manifest.append(("paper_snr_distribution", _gen_paper_snr_distribution))

    # 20. Paper figure: KDE-smoothed H0 posterior (D-05)
    def _gen_paper_h0_posterior_kde() -> tuple[object, object] | None:
        from darksiren_emri.plotting.paper_figures import plot_h0_posterior_kde

        try:
            return plot_h0_posterior_kde(data_dir=Path(output_dir))
        except (FileNotFoundError, KeyError):
            return None

    manifest.append(("paper_h0_posterior_kde", _gen_paper_h0_posterior_kde))

    # 21. Paper figure: M_z improvement panels v2 (Phase E)
    # Top-right panel switched from "representative single-bootstrap draw" to
    # the canonical raw Σ log L_i joint posterior so it matches fig01/fig08/
    # paper_h0_posterior. Plus a new host-count reduction violin panel.
    def _gen_paper_m_z_improvement() -> tuple[object, object] | None:
        from darksiren_emri.constants import H as TRUE_H
        from darksiren_emri.plotting._helpers import load_canonical_combined_posterior
        from darksiren_emri.plotting.convergence_analysis import (
            compute_m_z_improvement_bank,
            plot_m_z_improvement_panels,
        )

        try:
            bank = compute_m_z_improvement_bank(Path(output_dir), h_true=float(TRUE_H))
        except (FileNotFoundError, ValueError, KeyError):
            return None
        if bank is None:
            return None
        canon_no: tuple[np.ndarray, np.ndarray] | None = None
        canon_w: tuple[np.ndarray, np.ndarray] | None = None
        try:
            h_n, p_n, _m = load_canonical_combined_posterior(Path(output_dir), "posteriors")
            canon_no = (h_n, p_n)
        except FileNotFoundError:
            pass
        try:
            h_w, p_w, _m = load_canonical_combined_posterior(
                Path(output_dir), "posteriors_with_bh_mass"
            )
            canon_w = (h_w, p_w)
        except FileNotFoundError:
            pass

        # Optional host-count CSV from Phase B (parse_host_counts).
        host_counts = None
        host_csv = Path(output_dir) / "diagnostics" / "host_counts.csv"
        if host_csv.is_file():
            import pandas as pd

            host_counts = pd.read_csv(host_csv)

        return plot_m_z_improvement_panels(
            bank,
            canonical_combined_no_mass=canon_no,
            canonical_combined_with_mass=canon_w,
            host_counts=host_counts,
        )

    manifest.append(("paper_m_z_improvement", _gen_paper_m_z_improvement))

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
    from darksiren_emri.plotting.interactive import generate_all_interactive

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
