"""Posterior combination module for merging per-event likelihoods.

Combines per-h-value posterior JSON files from the evaluation campaign into
a joint posterior over the Hubble constant, with multiple zero-handling
strategies and diagnostic reporting.

Four zero-handling strategies:
- **naive**: Replace 0.0 with ``np.finfo(float).tiny`` before log.
- **exclude**: Remove events that have any zero likelihood.
- **per-event-floor**: Replace 0.0 with ``min(nonzero) / 100`` per event.
- **physics-floor**: Per-event minimum nonzero likelihood as floor value; all-zero events excluded.
"""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class CombinationStrategy(StrEnum):
    """Zero-handling strategy for posterior combination."""

    NAIVE = "naive"
    EXCLUDE = "exclude"
    PER_EVENT_FLOOR = "per-event-floor"
    PHYSICS_FLOOR = "physics-floor"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_posterior_jsons(
    posteriors_dir: Path,
) -> tuple[list[float], dict[int, dict[float, float]]]:
    """Load per-h-value posterior JSON files from a directory.

    Parameters
    ----------
    posteriors_dir : Path
        Directory containing ``h_*.json`` files.

    Returns
    -------
    h_values : list[float]
        Sorted list of h values found across all files.
    event_likelihoods : dict[int, dict[float, float]]
        Nested dict mapping ``detection_index -> {h_value: likelihood}``.
        Events with empty lists (missing data) are excluded.
    """
    json_files = sorted(posteriors_dir.glob("h_*.json"))
    if not json_files:
        msg = f"No h_*.json files found in {posteriors_dir}"
        raise FileNotFoundError(msg)

    h_values_set: set[float] = set()
    # detection_index -> {h_value: likelihood}
    event_likelihoods: dict[int, dict[float, float]] = {}

    for path in json_files:
        with open(path) as f:
            data: dict[str, list[float] | float] = json.load(f)

        h_raw = data["h"]
        h_val = float(h_raw)  # type: ignore[arg-type]
        h_values_set.add(h_val)

        for key, value in data.items():
            if key == "h":
                continue
            try:
                detection_idx = int(key)
            except ValueError:
                continue  # skip non-integer keys (e.g. "galaxy_likelihoods")
            # Empty list means missing event — skip
            if not isinstance(value, list) or len(value) == 0:
                continue
            likelihood = float(value[0])
            if detection_idx not in event_likelihoods:
                event_likelihoods[detection_idx] = {}
            event_likelihoods[detection_idx][h_val] = likelihood

    h_values = sorted(h_values_set)
    logger.info(
        "Loaded %d JSON files: %d h-values, %d events with data",
        len(json_files),
        len(h_values),
        len(event_likelihoods),
    )
    return h_values, event_likelihoods


# ---------------------------------------------------------------------------
# Array construction
# ---------------------------------------------------------------------------


def build_likelihood_array(
    h_values: list[float],
    event_likelihoods: dict[int, dict[float, float]],
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Build a 2-D likelihood array from the nested dict structure.

    Parameters
    ----------
    h_values : list[float]
        Sorted h-values (columns).
    event_likelihoods : dict[int, dict[float, float]]
        Nested dict from :func:`load_posterior_jsons`.

    Returns
    -------
    array : ndarray of shape ``(n_events, n_h_values)``
        Likelihood values.  ``NaN`` where an event is missing a particular
        h-value; ``0.0`` where the likelihood was explicitly zero.
    detection_indices : list[int]
        Sorted detection indices (row labels).
    """
    detection_indices = sorted(event_likelihoods.keys())
    n_events = len(detection_indices)
    n_h = len(h_values)

    array = np.full((n_events, n_h), np.nan, dtype=np.float64)

    for row, det_idx in enumerate(detection_indices):
        h_map = event_likelihoods[det_idx]
        for col, h in enumerate(h_values):
            if h in h_map:
                array[row, col] = h_map[h]

    logger.info("Built likelihood array: %d events x %d h-bins", n_events, n_h)
    return array, detection_indices


# ---------------------------------------------------------------------------
# Zero-handling strategies
# ---------------------------------------------------------------------------


def apply_strategy(
    likelihoods: npt.NDArray[np.float64],
    strategy: CombinationStrategy,
) -> tuple[npt.NDArray[np.float64], int]:
    """Apply a zero-handling strategy to the likelihood array.

    Parameters
    ----------
    likelihoods : ndarray of shape ``(n_events, n_h_values)``
        Raw likelihood array (may contain 0.0 entries).
    strategy : CombinationStrategy
        Which strategy to apply.

    Returns
    -------
    processed : ndarray
        Likelihood array with zeros handled.
    excluded_count : int
        Number of events removed (0 for strategies that keep all events).
    """
    result = likelihoods.copy()

    if strategy == CombinationStrategy.NAIVE:
        result[result == 0.0] = np.finfo(float).tiny
        return result, 0

    if strategy == CombinationStrategy.EXCLUDE:
        return _exclude_zero_events(result)

    if strategy == CombinationStrategy.PER_EVENT_FLOOR:
        return _per_event_floor(result)

    if strategy == CombinationStrategy.PHYSICS_FLOOR:
        return _physics_floor(result)

    msg = f"Unknown strategy: {strategy}"
    raise ValueError(msg)


def _exclude_zero_events(
    likelihoods: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """Remove rows that contain any exact zero."""
    has_zero = np.any(likelihoods == 0.0, axis=1)
    excluded = int(np.sum(has_zero))
    kept = likelihoods[~has_zero]
    logger.info("Exclude strategy: removed %d events with zeros", excluded)
    return kept, excluded


def _per_event_floor(
    likelihoods: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """Replace zeros with min(nonzero values in that row) / 100."""
    result = likelihoods.copy()
    for i in range(result.shape[0]):
        row = result[i]
        zero_mask = row == 0.0
        if not np.any(zero_mask):
            continue
        nonzero_vals = row[~zero_mask & ~np.isnan(row)]
        if len(nonzero_vals) == 0:
            # All-zero event: use tiny
            result[i, zero_mask] = np.finfo(float).tiny
        else:
            floor = float(np.min(nonzero_vals)) / 100.0
            result[i, zero_mask] = floor
    return result, 0


def _physics_floor(
    likelihoods: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """Replace zeros with the per-event minimum nonzero likelihood.

    For each event (row), zeros are replaced with the smallest nonzero
    likelihood value in that row.  Events that are entirely zero have no
    nonzero value to use as a floor and are excluded instead.

    Parameters
    ----------
    likelihoods : ndarray of shape ``(n_events, n_h_values)``
        Likelihood array (may contain 0.0 entries).

    Returns
    -------
    processed : ndarray
        Likelihood array with zeros replaced by per-event min-nonzero floor.
    excluded_count : int
        Number of all-zero events that were excluded.
    """
    result = likelihoods.copy()
    exclude_mask = np.zeros(result.shape[0], dtype=bool)

    for i in range(result.shape[0]):
        row = result[i]
        zero_mask = row == 0.0
        if not np.any(zero_mask):
            continue

        nonzero_vals = row[~zero_mask & ~np.isnan(row)]
        n_zeros = int(np.sum(zero_mask))
        n_bins = len(row)

        if len(nonzero_vals) == 0:
            # All-zero event: no nonzero value to derive floor from
            logger.warning(
                "Physics floor: event row %d has all-zero likelihoods, "
                "excluding (no nonzero value for floor)",
                i,
            )
            exclude_mask[i] = True
        else:
            floor_value = float(np.min(nonzero_vals))
            result[i, zero_mask] = floor_value
            logger.info(
                "Physics floor: event row %d: floored %d of %d bins with value %.6e",
                i,
                n_zeros,
                n_bins,
                floor_value,
            )

    excluded = int(np.sum(exclude_mask))
    kept = result[~exclude_mask]
    if excluded > 0:
        logger.info("Physics floor: excluded %d all-zero events", excluded)
    return kept, excluded


# ---------------------------------------------------------------------------
# Log-space combination
# ---------------------------------------------------------------------------


def combine_log_space(
    likelihoods: npt.NDArray[np.float64],
    log_D_h: npt.NDArray[np.float64] | None = None,  # noqa: ARG001 (kept for backward compat / diagnostics)
    n_events_used: int = 0,  # noqa: ARG001
) -> npt.NDArray[np.float64]:
    """Combine per-event likelihoods into a joint posterior using log-space.

    The per-event likelihoods :math:`L_i(h)` are already conditional on
    detection: ``L_comp = num/D(h)`` (Gray et al. 2020 Eq. 31, where ``D(h)``
    plays the role of the prior normalization for ``p_galaxy ∝ p_det · dV_c``)
    and ``L_cat = (1/N_g) Σ_g (N_g_local/D_g_local)`` (per-galaxy
    detection-conditional expectation).  The joint posterior is therefore
    just ``Π_i L_i(h)`` with no additional ``β(h)^N`` selection correction
    (Loredo 2004; Mandel et al. 2019, arXiv:1809.02063 §3): conditioning on
    the observed N_detected makes the β^N factor in the unconditional
    likelihood cancel against the Poisson rate prior.

    The ``log_D_h`` and ``n_events_used`` arguments are retained for
    backward compatibility with older callers, but are now **ignored** —
    applying ``−N · log D(h)`` here would double-count the selection
    correction already inside each per-event ``L_comp``.

    Parameters
    ----------
    likelihoods : ndarray of shape ``(n_events, n_h_values)``
        Likelihood array with zeros already handled (all values > 0).
    log_D_h : ndarray of shape ``(n_h_values,)`` or None
        Ignored.  Kept for backward compatibility; pass ``None`` for new
        callers.  Phase 43-H1 commit ``2853c32`` introduced an
        ``−N · log D(h)`` subtraction here that we have since shown
        (Tier 3 audit, 2026-05-04) over-applies the selection correction.
    n_events_used : int
        Ignored.  Kept for backward compatibility.

    Returns
    -------
    posterior : ndarray of shape ``(n_h_values,)``
        Normalized joint posterior.
    """
    log_likes = np.log(likelihoods)
    joint_log = np.sum(log_likes, axis=0)
    max_log = np.max(joint_log)
    posterior = np.exp(joint_log - max_log)
    posterior = posterior / np.sum(posterior)
    return np.asarray(posterior, dtype=np.float64)


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------


def generate_diagnostic_report(
    h_values: list[float],
    likelihoods: npt.NDArray[np.float64],
    detection_indices: list[int],
) -> str:
    """Generate a markdown diagnostic report about zero-likelihood events.

    Parameters
    ----------
    h_values : list[float]
        Sorted h-values.
    likelihoods : ndarray of shape ``(n_events, n_h_values)``
        Raw likelihood array (before strategy application).
    detection_indices : list[int]
        Detection indices corresponding to rows.

    Returns
    -------
    report : str
        Markdown-formatted diagnostic report.
    """
    n_events = likelihoods.shape[0]

    # Identify zero events
    zero_det_indices: list[int] = []
    zero_h_bins: list[list[float]] = []
    zero_patterns: list[str] = []
    for i in range(n_events):
        row = likelihoods[i]
        bins = [h_values[j] for j in range(len(h_values)) if row[j] == 0.0]
        if bins:
            all_zero = len(bins) == len(h_values)
            low_h_only = all(h <= 0.70 for h in bins) and not all_zero
            if all_zero:
                pattern = "all-zeros"
            elif low_h_only:
                pattern = "low-h-only"
            else:
                pattern = "partial-zeros"
            zero_det_indices.append(detection_indices[i])
            zero_h_bins.append(bins)
            zero_patterns.append(pattern)

    # Count NaN rows (empty events in the original data)
    nan_rows = int(np.sum(np.all(np.isnan(likelihoods), axis=1)))
    all_zero_count = sum(1 for p in zero_patterns if p == "all-zeros")

    # Zero distribution by h-bin
    zeros_per_h: dict[float, int] = {}
    for j, h in enumerate(h_values):
        zeros_per_h[h] = int(np.sum(likelihoods[:, j] == 0.0))

    lines: list[str] = []
    lines.append("# Zero-Likelihood Diagnostic Report\n")

    lines.append("## Summary\n")
    lines.append(f"- **Total events:** {n_events}")
    lines.append(f"- **Events with zeros:** {len(zero_det_indices)}")
    lines.append(f"- **Empty events (all NaN):** {nan_rows}")
    lines.append(f"- **All-zeros events:** {all_zero_count}")
    lines.append("")

    lines.append("## Zero-Event Detail\n")
    if zero_det_indices:
        lines.append("| Detection Index | Zero h-bins | Pattern |")
        lines.append("|---|---|---|")
        for det_idx, bins, pat in zip(zero_det_indices, zero_h_bins, zero_patterns):
            bins_str = ", ".join(f"{h:.2f}" for h in bins)
            lines.append(f"| {det_idx} | {bins_str} | {pat} |")
    else:
        lines.append("No events with zero likelihoods.")
    lines.append("")

    lines.append("## Zero Distribution by h-bin\n")
    lines.append("| h-value | Number of zeros |")
    lines.append("|---|---|")
    for h, count in sorted(zeros_per_h.items()):
        lines.append(f"| {h:.2f} | {count} |")
    lines.append("")

    lines.append("## Root Cause Analysis\n")
    lines.append(
        "Zero likelihoods arise when a detection event has no compatible "
        "host galaxy at the given Hubble constant value. "
        "**All-zeros** events have no viable host at any h-value, indicating "
        "the event lies outside the galaxy catalog coverage entirely. "
        "**Low-h-only** zeros occur at smaller h-values where the implied "
        "luminosity distance pushes the source beyond the catalog's redshift "
        "completeness boundary. "
        "**Partial-zeros** arise at coverage boundaries where the galaxy "
        "catalog transitions between complete and incomplete."
    )
    lines.append("")

    lines.append("## Impact on Posterior\n")
    lines.append(
        "Under naive multiplication, a single zero at any h-bin drives the "
        "entire joint posterior to zero at that bin. With "
        f"{len(zero_det_indices)} zero-events, the naive posterior is dominated "
        "by the *absence* of catalog coverage rather than the actual "
        "likelihood information. Log-space combination with zero-handling "
        "strategies mitigates this."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def generate_comparison_table(
    h_values: npt.NDArray[np.float64],
    likelihoods: npt.NDArray[np.float64],
    detection_indices: list[int],
    variant: str,
    log_D_h: npt.NDArray[np.float64] | None = None,
) -> str:
    """Generate a markdown comparison table for all strategies.

    Parameters
    ----------
    h_values : ndarray
        Sorted h-values.
    likelihoods : ndarray of shape ``(n_events, n_h_values)``
        Raw likelihood array.
    detection_indices : list[int]
        Detection indices.
    variant : str
        Name of the posterior variant (e.g. ``"posteriors"``).
    log_D_h : ndarray of shape ``(n_h_values,)`` or None
        Ignored (kept for backward compatibility).  See ``combine_log_space``
        docstring — D(h) enters via L_comp = num/D inside each per-event
        likelihood (Gray Eq. 31), not as an outer correction.

    Returns
    -------
    table : str
        Markdown-formatted comparison table.
    """
    n_total = likelihoods.shape[0]
    h_arr = np.asarray(h_values, dtype=np.float64)

    lines: list[str] = []
    lines.append("# Posterior Combination Method Comparison\n")
    lines.append(f"## Variant: {variant}\n")
    lines.append("| Strategy | Events Used | Events Excluded | MAP h | MAP Posterior Value |")
    lines.append("|---|---|---|---|---|")

    for strat in CombinationStrategy:
        processed, excluded = apply_strategy(likelihoods, strat)
        n_used = n_total - excluded
        if processed.shape[0] == 0:
            lines.append(f"| {strat.value} | 0 | {excluded} | N/A | N/A |")
            continue
        posterior = combine_log_space(processed, log_D_h=log_D_h, n_events_used=n_used)
        map_idx = int(np.argmax(posterior))
        map_h = float(h_arr[map_idx])
        map_val = float(posterior[map_idx])
        lines.append(f"| {strat.value} | {n_used} | {excluded} | {map_h:.4f} | {map_val:.6f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def combine_posteriors(
    posteriors_dir: str,
    strategy: str,
    output_dir: str,
    d_h_table: dict[float, float] | None = None,
) -> dict[str, object]:
    """Combine per-event posteriors into a joint posterior.

    This is the main entry point called from CLI.

    Parameters
    ----------
    posteriors_dir : str
        Path to directory containing ``h_*.json`` files.
    strategy : str
        Zero-handling strategy name (one of the ``CombinationStrategy`` values).
    output_dir : str
        Path to write output files.
    d_h_table : dict[float, float] or None
        Pre-computed ``{h: D(h)}`` mapping (Gray et al. 2020, Eq. A.19).
        When ``None``, ``D(h)`` is computed automatically using
        :func:`~master_thesis_code.bayesian_inference.bayesian_statistics.precompute_completion_denominator`.

    Returns
    -------
    result : dict
        Combined posterior result with keys ``h_values``, ``posterior``,
        ``strategy``, ``n_events_total``, ``n_events_used``,
        ``n_events_excluded``, ``n_events_empty``, ``map_h``,
        ``map_posterior``, ``variant``, ``D_h_per_h``.
    """
    posteriors_path = Path(posteriors_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse strategy
    strat = CombinationStrategy(strategy)
    effective_strategy = strat

    # Load and build array
    h_values, event_likelihoods = load_posterior_jsons(posteriors_path)
    likelihoods, detection_indices = build_likelihood_array(h_values, event_likelihoods)

    n_total = likelihoods.shape[0]

    # Count truly empty events (those that didn't make it into event_likelihoods
    # are not in the array; we count by checking the original JSONs)
    json_files = sorted(posteriors_path.glob("h_*.json"))
    all_keys: set[str] = set()
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        all_keys.update(k for k in data if k != "h")
    n_empty = len(all_keys) - n_total

    # Compute or validate D(h) selection-function denominator
    # Gray et al. (2020), arXiv:1908.06050, Eq. A.19.
    if d_h_table is None:
        # Lazy import: avoids pulling heavy scipy/pandas/astropy into this
        # module's top-level import at the cost of a slightly slower first call.
        from master_thesis_code.bayesian_inference.bayesian_statistics import (  # noqa: PLC0415
            precompute_completion_denominator,
        )
        from master_thesis_code.bayesian_inference.simulation_detection_probability import (  # noqa: PLC0415
            SimulationDetectionProbability,
        )
        from master_thesis_code.constants import (  # noqa: PLC0415
            INJECTION_DATA_DIR,
            OMEGA_DE,
            OMEGA_M,
            SNR_THRESHOLD,
        )

        detection_probability = SimulationDetectionProbability(
            injection_data_dir=INJECTION_DATA_DIR,
            snr_threshold=SNR_THRESHOLD,
        )
        for h in h_values:
            detection_probability._get_or_build_grid(h)

        d_h_table = precompute_completion_denominator(
            h_values=h_values,
            detection_probability_obj=detection_probability,
            Omega_m=OMEGA_M,
            Omega_DE=OMEGA_DE,
        )

    D_h_array = np.array([d_h_table[h] for h in h_values], dtype=np.float64)
    log_D_h = np.log(D_h_array)
    logger.info(
        "D(h) range: %.4e – %.4e (ratio %.2fx)",
        float(np.min(D_h_array)),
        float(np.max(D_h_array)),
        float(np.max(D_h_array) / max(float(np.min(D_h_array)), 1e-300)),
    )

    # Apply strategy
    processed, n_excluded = apply_strategy(likelihoods, effective_strategy)
    n_used = n_total - n_excluded

    # Combine with Gray Eq. A.19 selection-function correction
    posterior = combine_log_space(processed, log_D_h=log_D_h, n_events_used=n_used)

    # MAP estimate
    h_arr = np.array(h_values, dtype=np.float64)
    map_idx = int(np.argmax(posterior))
    map_h = float(h_arr[map_idx])
    map_posterior = float(posterior[map_idx])

    variant = posteriors_path.name

    logger.info(
        "Combined %d events (excluded %d, empty %d) with strategy '%s': MAP h=%.4f",
        n_used,
        n_excluded,
        n_empty,
        effective_strategy.value,
        map_h,
    )

    # Generate reports
    diag_report = generate_diagnostic_report(h_values, likelihoods, detection_indices)
    (output_path / "diagnostic_report.md").write_text(diag_report)

    comp_table = generate_comparison_table(
        h_arr, likelihoods, detection_indices, variant, log_D_h=log_D_h
    )
    (output_path / "comparison_table.md").write_text(comp_table)

    # Build result
    result: dict[str, object] = {
        "h_values": [float(h) for h in h_values],
        "posterior": [float(p) for p in posterior],
        "strategy": effective_strategy.value,
        "n_events_total": n_total,
        "n_events_used": n_used,
        "n_events_excluded": n_excluded,
        "n_events_empty": n_empty,
        "map_h": map_h,
        "map_posterior": map_posterior,
        "variant": variant,
        "D_h_per_h": [float(d) for d in D_h_array],
    }

    # Write combined posterior JSON
    with open(output_path / "combined_posterior.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Canonical combination (raw Σ log L) — paper-grade reference implementation
# ---------------------------------------------------------------------------
#
# The functions below are the canonical "raw Σ log L" combination used by
# the bias-investigation suite (test_24_multi_truth_bias_sweep.py,
# test_28_production_finegrid_analyze.py) and quoted throughout
# docs/H0_BIAS_RESOLUTION.md. They are the source-of-truth combination
# for all plotting code; figures must consume these helpers rather than
# `combine_posteriors` (which applies a `physics-floor` zero-handling
# strategy that biases the MAP relative to raw Σ log L).


_H_FILENAME_RE = re.compile(r"h_(\d+)_(\d+)\.json")


def _h_from_filename(path: Path) -> float:
    """Parse the h-value out of an ``h_<int>_<int>.json`` filename.

    Returns ``nan`` if the filename does not match the expected pattern.
    """
    m = _H_FILENAME_RE.match(path.name)
    if m is None:
        return float("nan")
    return float(f"{m.group(1)}.{m.group(2)}")


def load_per_h_likelihoods(
    directory: Path,
) -> tuple[list[float], npt.NDArray[np.float64]]:
    """Load per-event log-likelihoods across all h-value JSONs in *directory*.

    Reads every ``h_*.json`` file, extracts the scalar per-event likelihood
    for each integer event key, drops events that have missing data at any
    h-value (``full_mask`` filter), and returns ``(h_values, log_L)`` where
    ``log_L`` has shape ``(n_events_full, n_h_values)``.

    This is the reference implementation used by the bias-investigation
    suite and is the canonical loader for all paper-grade H0 posterior
    figures.

    Parameters
    ----------
    directory:
        Path to a directory containing ``h_*.json`` files (typically
        ``posteriors/`` or ``posteriors_with_bh_mass/``).

    Returns
    -------
    h_values:
        Sorted list of h-values (one per JSON file).
    log_L:
        Float64 array of shape ``(n_events, n_h_values)``; entries are
        ``log(max(L_event_at_h, 1e-300))``. Events with NaN at any h are
        excluded.
    """
    if not directory.exists():
        return [], np.empty((0, 0), dtype=np.float64)
    files = sorted(directory.glob("h_*.json"), key=_h_from_filename)
    h_values: list[float] = [_h_from_filename(f) for f in files]
    event_indices: set[int] = set()
    per_h_data: list[dict[int, float]] = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        per_h: dict[int, float] = {}
        for k, v in d.items():
            try:
                ev = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                if len(v) == 0:
                    continue
                val = v[0]
            else:
                val = v
            event_indices.add(ev)
            per_h[ev] = float(val)
        per_h_data.append(per_h)
    events_sorted = sorted(event_indices)
    log_L = np.full((len(events_sorted), len(h_values)), np.nan, dtype=np.float64)
    for j, per_h in enumerate(per_h_data):
        for i, ev in enumerate(events_sorted):
            if ev in per_h:
                log_L[i, j] = float(np.log(max(per_h[ev], 1e-300)))
    full_mask = ~np.isnan(log_L).any(axis=1)
    log_L_filtered: npt.NDArray[np.float64] = log_L[full_mask]
    return h_values, log_L_filtered


def parabolic_refine_map(
    h_grid: npt.NDArray[np.float64],
    log_posterior: npt.NDArray[np.float64],
) -> float:
    """Refine the discrete MAP via 3-point parabolic interpolation.

    Given a log-posterior on a discrete h-grid, fits a parabola through
    the discrete argmax and its two neighbours and returns the analytic
    vertex of the parabola. Falls back to the discrete argmax when the
    peak is on the grid boundary or the parabola degenerates.

    Parameters
    ----------
    h_grid:
        Monotonically increasing h-values.
    log_posterior:
        Log-posterior values on ``h_grid``.

    Returns
    -------
    float
        Sub-grid continuous MAP estimate.
    """
    i = int(np.argmax(log_posterior))
    if i <= 0 or i >= len(h_grid) - 1:
        return float(h_grid[i])
    h0, h1, h2 = float(h_grid[i - 1]), float(h_grid[i]), float(h_grid[i + 1])
    y0, y1, y2 = (
        float(log_posterior[i - 1]),
        float(log_posterior[i]),
        float(log_posterior[i + 1]),
    )
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return h1
    return float(h1 - 0.5 * (h2 - h0) * (y2 - y0) / (2 * denom))


def compute_canonical_combined_posterior(
    posteriors_dir: Path,
) -> dict[str, object]:
    """Compute the canonical joint H0 posterior from per-h JSONs.

    Aggregates per-event log-likelihoods via raw ``Σ log L_i`` (no
    physics-floor, no outer D(h) correction — the Tier 3 fix removed
    the latter from ``combine_log_space``). This is the combination
    quoted in Phase 48 verdict JSON and throughout
    ``docs/H0_BIAS_RESOLUTION.md``.

    Parameters
    ----------
    posteriors_dir:
        Directory of ``h_*.json`` files (1D or 2D variant).

    Returns
    -------
    dict with keys
        - ``h_values``: list[float] — sorted h-grid
        - ``posterior``: list[float] — peak-normalised posterior
        - ``log_posterior``: list[float] — un-normalised Σ log L_i
        - ``n_events_used``: int — events surviving the full-mask filter
        - ``discrete_map``: float — argmax on the discrete grid
        - ``continuous_map``: float — parabolic-refined sub-grid MAP
        - ``strategy``: literal string ``"raw-sum-log"``
    """
    h_values, log_L = load_per_h_likelihoods(posteriors_dir)
    if not h_values or log_L.size == 0:
        return {
            "h_values": [],
            "posterior": [],
            "log_posterior": [],
            "n_events_used": 0,
            "discrete_map": float("nan"),
            "continuous_map": float("nan"),
            "strategy": "raw-sum-log",
        }
    h_arr = np.asarray(h_values, dtype=np.float64)
    log_joint = log_L.sum(axis=0)
    # Peak-normalise the posterior in linear space for plotting convenience.
    log_joint_shifted = log_joint - log_joint.max()
    posterior = np.exp(log_joint_shifted)
    discrete_argmax = int(np.argmax(log_joint))
    discrete_map = float(h_arr[discrete_argmax])
    continuous_map = parabolic_refine_map(h_arr, log_joint)
    return {
        "h_values": [float(h) for h in h_values],
        "posterior": [float(p) for p in posterior],
        "log_posterior": [float(p) for p in log_joint],
        "n_events_used": int(log_L.shape[0]),
        "discrete_map": discrete_map,
        "continuous_map": continuous_map,
        "strategy": "raw-sum-log",
    }
