"""Paper figures for the EMRI dark siren H0 inference results.

Publication-quality figures:

1. **H0 posterior comparison** -- combined posteriors for the two analysis
   variants (without / with BH mass channel) on a single axes.
2. **Single-event likelihoods** -- 4 representative events showing how the
   BH mass channel narrows the per-event likelihood.
3. **Posterior convergence** -- CI width vs number of events with
   N^{-1/2} reference line (both analysis variants).
4. **SNR distribution** -- histogram and scatter of detected-event SNR
   (requires CRB CSV data from cluster).

All functions follow the project plotting convention: data in,
``(Figure, Axes)`` out.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.constants import SNR_THRESHOLD
from master_thesis_code.plotting._colors import (
    CMAP,
    CYCLE,
    EDGE,
    MEAN,
    REFERENCE,
    TRUTH,
    VARIANT_NO_MASS,
    VARIANT_WITH_MASS,
)
from master_thesis_code.plotting._helpers import (
    compute_credible_interval,
    get_figure,
)
from master_thesis_code.plotting.convergence_analysis import (
    _load_per_event_no_mass,
    _load_per_event_with_mass_scalars,
)

__all__ = [
    "_load_per_event_no_mass",
    "_load_per_event_with_mass_scalars",
    "plot_closure_test_overlay",
    "plot_h0_posterior_comparison",
    "plot_h0_posterior_kde",
    "plot_posterior_convergence",
    "plot_single_event_likelihoods",
    "plot_snr_distribution",
]

# ---------------------------------------------------------------------------
# KDE smoothing helper
# ---------------------------------------------------------------------------


def _kde_smooth_posterior(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    n_fine: int = 500,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Smooth posterior with Gaussian KDE (Scott's rule).

    Parameters
    ----------
    h_values:
        Sorted 1-D array of h grid points.
    posterior:
        Posterior density on the same grid (need not be normalized).
    n_fine:
        Number of points in the fine evaluation grid.

    Returns
    -------
    h_fine, kde_fine : smoothed posterior on a fine grid.
        Returns (h_values.copy(), posterior.copy()) if norm <= 0.
    """
    from scipy.stats import gaussian_kde

    norm = posterior.sum()
    if norm <= 0:
        return h_values.copy(), posterior.copy()
    weights = posterior / norm
    kde = gaussian_kde(h_values, weights=weights, bw_method="scott")
    h_fine = np.linspace(float(h_values[0]), float(h_values[-1]), n_fine)
    kde_fine = kde(h_fine)
    return h_fine, kde_fine


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_combined_posterior(variant: str, data_dir: Path) -> dict[str, Any]:
    """Load the canonical combined H0 posterior for *variant*.

    Phase A unification (2026-05-15): paper figures now consume the canonical
    raw ``Σ log L_i`` combination from
    :func:`master_thesis_code.plotting._helpers.load_canonical_combined_posterior`
    rather than the legacy physics-floor ``combined_posterior.json`` files at
    the working-directory root. This ensures every H0-posterior figure shows
    the same MAP (the canonical value quoted in
    ``docs/H0_BIAS_RESOLUTION.md`` and the Phase 48 verdict JSON).

    A back-compat fallback reads the legacy
    ``combined_posterior[_with_bh_mass].json`` schema when the per-h
    subdirectory is not present (used by unit-test fixtures and by older
    cached results from physics-floor combines).

    Parameters
    ----------
    variant:
        ``"posteriors"`` (1D channel) or ``"posteriors_with_bh_mass"`` (2D).
    data_dir:
        Root directory containing ``posteriors/`` and
        ``posteriors_with_bh_mass/`` subdirectories (canonical path), or
        the legacy ``combined_posterior[_with_bh_mass].json`` files.

    Returns
    -------
    dict with keys ``h_values``, ``posterior``, ``map_h`` (continuous, parabolic),
    ``discrete_map``, ``n_events_used``, ``strategy``. Field names are kept
    backwards-compatible with the legacy ``combined_posterior.json`` schema.
    """
    if variant not in ("posteriors", "posteriors_with_bh_mass"):
        msg = f"Unknown variant: {variant}"
        raise ValueError(msg)

    # Canonical path: per-h JSONs in subdirectory.
    posteriors_subdir = data_dir / variant
    if posteriors_subdir.is_dir():
        # Local import to avoid a circular dependency at module import time.
        from master_thesis_code.plotting._helpers import load_canonical_combined_posterior

        h_grid, posterior, meta = load_canonical_combined_posterior(data_dir, variant)
        return {
            "h_values": [float(h) for h in h_grid],
            "posterior": [float(p) for p in posterior],
            "map_h": meta["continuous_map"],
            "discrete_map": meta["discrete_map"],
            "n_events_used": meta["n_events_used"],
            "strategy": meta["strategy"],
            "variant": variant,
        }

    # Back-compat: legacy single-file JSON at data_dir root.
    legacy_name = (
        "combined_posterior.json"
        if variant == "posteriors"
        else "combined_posterior_with_bh_mass.json"
    )
    legacy_path = data_dir / legacy_name
    if legacy_path.is_file():
        import json as _json

        with open(legacy_path) as f:
            return _json.load(f)  # type: ignore[no-any-return]

    raise FileNotFoundError(
        f"Neither canonical {posteriors_subdir} nor legacy {legacy_path} exists."
    )


# Per-event loaders are imported from convergence_analysis (above) so
# both this module and the M_z improvement bank can use them without a
# circular import.


# ---------------------------------------------------------------------------
# Figure 1: H0 posterior comparison
# ---------------------------------------------------------------------------


def plot_h0_posterior_comparison(
    data_dir: Path,
) -> tuple[Figure, Axes]:
    """Plot combined H0 posteriors for both analysis variants.

    Parameters
    ----------
    data_dir:
        Root directory containing the combined posterior JSONs.

    Returns
    -------
    (fig, ax) following the project factory convention.
    """
    # Local import to avoid a module-level circular dependency
    # (bayesian_plots <-> paper_figures via the canonical factory's KDE path).
    from master_thesis_code.plotting.bayesian_plots import plot_combined_posterior

    p_no = _load_combined_posterior("posteriors", data_dir)
    p_with = _load_combined_posterior("posteriors_with_bh_mass", data_dir)

    h_no = np.array(p_no["h_values"])
    h_with = np.array(p_with["h_values"])
    post_no = np.array(p_no["posterior"])
    post_with = np.array(p_with["posterior"])

    # Headline (without M_z): navy SOLID, area-normalized PDF, nested 68/95%
    # HDI shading, inline MAP, dotted "Injected" truth line. NO Planck/SH0ES
    # bands (the paper figure deliberately omits the reference bands).
    fig, ax = plot_combined_posterior(
        h_no,
        post_no,
        0.73,
        label=r"Without $M_z$",
        normalize="density",
        color=VARIANT_NO_MASS,
        linestyle="-",
        show_credible=True,
        show_references=False,
        annotate_map=True,
        show_truth=True,
        truth_linestyle=":",
        truth_label="Injected",
        ylabel=r"$p(h \mid \mathrm{data})$",
        xlim=(0.59, 0.87),
        legend=False,
    )

    # Secondary (with M_z): gold DASHED, HDI band, no truth/MAP/references.
    plot_combined_posterior(
        h_with,
        post_with,
        0.73,
        label=r"With $M_z$",
        normalize="density",
        color=VARIANT_WITH_MASS,
        linestyle="--",
        show_credible=True,
        show_references=False,
        annotate_map=False,
        show_truth=False,
        legend=False,
        ax=ax,
    )

    ax.legend(loc="upper right")

    # constrained_layout is on (mplstyle); do NOT also call tight_layout (§1.8).
    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2: Single-event likelihoods
# ---------------------------------------------------------------------------


def _select_representative_events(
    h_vals: npt.NDArray[np.float64],
    events: dict[str, npt.NDArray[np.float64]],
) -> list[str]:
    """Select 4 representative events by likelihood shape.

    Criteria: peaked, moderately peaked, broad, and multi-modal or varied.

    Parameters
    ----------
    h_vals:
        Array of h grid points.
    events:
        Dict mapping event IDs to likelihood arrays.

    Returns
    -------
    List of 4 event ID strings.
    """
    stats: list[tuple[str, float, int, float]] = []
    for eid, lik in events.items():
        mx = np.max(lik)
        if mx == 0:
            continue
        p = lik / np.sum(lik)
        h_mean = float(np.sum(p * h_vals))
        h_std = float(np.sqrt(np.sum(p * (h_vals - h_mean) ** 2)))
        n_above = int(np.sum(lik / mx > 0.1))
        stats.append((eid, h_std, n_above, mx))

    # Sort by width (h_std)
    stats.sort(key=lambda x: x[1])
    n = len(stats)

    # Pick: narrowest, 25th percentile, 50th percentile, broadest
    selected = [
        stats[max(1, n // 20)][0],  # very peaked (5th percentile)
        stats[n // 4][0],  # moderately peaked
        stats[n // 2][0],  # median width
        stats[int(0.95 * n)][0],  # very broad (95th percentile)
    ]
    return selected


def plot_single_event_likelihoods(
    data_dir: Path,
) -> tuple[Figure, Any]:
    """Plot single-event likelihoods for 4 representative events.

    Creates a 4-row x 2-column grid: left column = without BH mass,
    right column = with BH mass.

    Parameters
    ----------
    data_dir:
        Root directory containing per-event posterior subdirectories.

    Returns
    -------
    (fig, axes) where axes is a 4x2 ndarray of Axes.
    """
    h_no, events_no = _load_per_event_no_mass(data_dir / "posteriors")
    h_with, events_with = _load_per_event_with_mass_scalars(data_dir / "posteriors_with_bh_mass")

    # Events present in both variants
    common_ids = sorted(
        set(events_no.keys()) & set(events_with.keys()),
        key=int,
    )

    # Filter to events with nonzero data in both variants
    valid: dict[str, npt.NDArray[np.float64]] = {}
    for eid in common_ids:
        if np.max(events_no[eid]) > 0 and np.max(events_with[eid]) > 0:
            valid[eid] = events_no[eid]

    selected = _select_representative_events(h_no, valid)

    fig, axes = get_figure(
        nrows=4,
        ncols=2,
        figsize=(3.375, 6.0),
        sharex=True,
    )

    labels = ["Peaked", "Moderate", "Median", "Broad"]

    for row, (eid, label) in enumerate(zip(selected, labels)):
        ax_no: Axes = axes[row, 0]
        ax_with: Axes = axes[row, 1]

        # No-mass likelihood
        lik_no = events_no[eid]
        lik_no_norm = lik_no / np.max(lik_no) if np.max(lik_no) > 0 else lik_no
        ax_no.plot(h_no, lik_no_norm, "-", color=VARIANT_NO_MASS, linewidth=1.0)

        # With-mass likelihood
        lik_with = events_with[eid]
        lik_with_norm = lik_with / np.max(lik_with) if np.max(lik_with) > 0 else lik_with
        ax_with.plot(h_with, lik_with_norm, "-", color=VARIANT_WITH_MASS, linewidth=1.0)

        # Row label
        ax_no.set_ylabel(f"{label}\n(event {eid})")

        # Truth lines
        ax_no.axvline(0.73, color=TRUTH, linestyle=":", linewidth=0.8, alpha=0.7)
        ax_with.axvline(0.73, color=TRUTH, linestyle=":", linewidth=0.8, alpha=0.7)

        ax_no.set_ylim(-0.05, 1.15)
        ax_with.set_ylim(-0.05, 1.15)

        # Remove y-tick labels for right column
        ax_with.set_yticklabels([])

    # Column titles
    axes[0, 0].set_title(r"Without $M_z$")
    axes[0, 1].set_title(r"With $M_z$")

    # Bottom row x-labels
    axes[-1, 0].set_xlabel(r"$h$")
    axes[-1, 1].set_xlabel(r"$h$")

    fig.align_ylabels(axes[:, 0])

    return fig, axes


# ---------------------------------------------------------------------------
# Figure 3: Posterior convergence (both analysis variants)
# ---------------------------------------------------------------------------

# Subset sizes for convergence study.  Chosen to span one-and-a-half
# decades on a log scale with good coverage at both ends.
_CONVERGENCE_SUBSET_SIZES: list[int] = [10, 20, 50, 100, 150, 200, 300, 400, 500]
_CONVERGENCE_N_SUBSETS: int = 50


def _compute_convergence_stats(
    log_event_matrix: npt.NDArray[np.float64],
    n_events_total: int,
    subset_sizes: list[int],
    n_subsets: int,
    rng: np.random.Generator,
    h_values: npt.NDArray[np.float64],
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Compute convergence statistics for one analysis variant.

    For each subset size, draws ``n_subsets`` random subsets of events,
    combines their log-posteriors, and computes the 68% CI width.

    Parameters
    ----------
    log_event_matrix:
        Array of shape (n_events_total, n_h) containing pre-computed
        log-likelihoods clipped to avoid log(0).
    n_events_total:
        Number of valid events (rows in log_event_matrix).
    subset_sizes:
        Candidate event counts to probe.
    n_subsets:
        Number of random draws per subset size.
    rng:
        NumPy random Generator instance (caller owns the seed).
    h_values:
        Sorted 1-D array of h grid points (length n_h).

    Returns
    -------
    used_sizes, medians, lo_pctiles, hi_pctiles
        Parallel lists of the subset sizes that were actually used
        (those <= n_events_total) and their per-size 50th/16th/84th
        percentile CI widths.
    """
    used_sizes: list[int] = []
    medians: list[float] = []
    lo_pctiles: list[float] = []
    hi_pctiles: list[float] = []

    for n_sub in subset_sizes:
        if n_sub > n_events_total:
            continue
        used_sizes.append(n_sub)
        widths: list[float] = []
        for _ in range(n_subsets):
            idx = rng.choice(n_events_total, size=n_sub, replace=False)
            log_combined = np.sum(log_event_matrix[idx, :], axis=0)
            log_p = log_combined - np.max(log_combined)
            p = np.exp(log_p)
            lo, hi = compute_credible_interval(h_values, p)
            w = hi - lo
            if not np.isnan(w):
                widths.append(w)

        if widths:
            medians.append(float(np.median(widths)))
            lo_pctiles.append(float(np.percentile(widths, 16)))
            hi_pctiles.append(float(np.percentile(widths, 84)))
        else:
            medians.append(np.nan)
            lo_pctiles.append(np.nan)
            hi_pctiles.append(np.nan)

    return used_sizes, medians, lo_pctiles, hi_pctiles


def plot_posterior_convergence(
    data_dir: Path,
    *,
    subset_sizes: list[int] | None = None,
    n_subsets: int = _CONVERGENCE_N_SUBSETS,
    seed: int = 20260407,
) -> tuple[Figure, Axes]:
    """Plot 68% CI width vs number of events for both analysis variants.

    Demonstrates the expected N^{-1/2} narrowing of the posterior as more
    independent EMRI events are combined.  Both the without-BH-mass and
    with-BH-mass channels are shown as separate errorbar curves.

    Parameters
    ----------
    data_dir:
        Root directory containing ``posteriors/`` and
        ``posteriors_with_bh_mass/`` subdirectories with per-event JSON
        files.
    subset_sizes:
        List of event counts to probe.  Defaults to
        ``[10, 20, 50, 100, 150, 200, 300, 400, 500]``.
    n_subsets:
        Number of random subsets drawn at each size (default 50).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (fig, ax) following the project factory convention.
    """
    if subset_sizes is None:
        subset_sizes = list(_CONVERGENCE_SUBSET_SIZES)

    # --- Load without-BH-mass data ---
    h_values, events = _load_per_event_no_mass(data_dir / "posteriors")
    n_h = len(h_values)
    event_ids = sorted(events.keys(), key=int)
    valid_ids = [eid for eid in event_ids if np.max(events[eid]) > 0]
    n_events_total = len(valid_ids)
    event_matrix = np.empty((n_events_total, n_h))
    for i, eid in enumerate(valid_ids):
        event_matrix[i, :] = events[eid]
    log_event_matrix = np.log(np.clip(event_matrix, 1e-300, None))

    # --- Load with-BH-mass data ---
    h_values_wm, events_wm = _load_per_event_with_mass_scalars(data_dir / "posteriors_with_bh_mass")
    n_h_wm = len(h_values_wm)
    event_ids_wm = sorted(events_wm.keys(), key=int)
    valid_ids_wm = [eid for eid in event_ids_wm if np.max(events_wm[eid]) > 0]
    n_events_total_wm = len(valid_ids_wm)
    event_matrix_wm = np.empty((n_events_total_wm, n_h_wm))
    for i, eid in enumerate(valid_ids_wm):
        event_matrix_wm[i, :] = events_wm[eid]
    log_event_matrix_wm = np.log(np.clip(event_matrix_wm, 1e-300, None))

    # Seed once; each call to _compute_convergence_stats advances the rng
    rng = np.random.default_rng(seed)

    used_sizes, medians, lo_pctiles, hi_pctiles = _compute_convergence_stats(
        log_event_matrix, n_events_total, subset_sizes, n_subsets, rng, h_values
    )
    used_sizes_wm, medians_wm, lo_pctiles_wm, hi_pctiles_wm = _compute_convergence_stats(
        log_event_matrix_wm, n_events_total_wm, subset_sizes, n_subsets, rng, h_values_wm
    )

    x = np.array(used_sizes, dtype=float)
    y_med = np.array(medians)
    y_lo = np.array(lo_pctiles)
    y_hi = np.array(hi_pctiles)

    x_wm = np.array(used_sizes_wm, dtype=float)
    y_med_wm = np.array(medians_wm)
    y_lo_wm = np.array(lo_pctiles_wm)
    y_hi_wm = np.array(hi_pctiles_wm)

    # -- Plot --
    fig, ax = get_figure(preset="single")

    ax.errorbar(
        x,
        y_med,
        yerr=[y_med - y_lo, y_hi - y_med],
        fmt="o",
        color=VARIANT_NO_MASS,
        markersize=4,
        capsize=3,
        linewidth=1.0,
        label=r"Without $M_z$",
        zorder=3,
    )

    ax.errorbar(
        x_wm,
        y_med_wm,
        yerr=[y_med_wm - y_lo_wm, y_hi_wm - y_med_wm],
        fmt="s",
        color=VARIANT_WITH_MASS,
        markersize=4,
        capsize=3,
        linewidth=1.0,
        label=r"With $M_z$",
        zorder=3,
    )

    # N^{-1/2} reference line anchored to the largest N median of no-mass variant
    if len(used_sizes) > 0 and not np.isnan(y_med[-1]):
        n_ref = x[-1]
        y_ref = y_med[-1]
        x_line = np.logspace(np.log10(x[0] * 0.8), np.log10(x[-1] * 1.2), 100)
        y_line = y_ref * np.sqrt(n_ref / x_line)
        ax.plot(
            x_line,
            y_line,
            "--",
            color=REFERENCE,
            linewidth=1.0,
            label=r"$\propto N_\mathrm{det}^{-1/2}$",
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of events $N_\mathrm{det}$")
    ax.set_ylabel(r"$1\sigma$ width of $h$ posterior")
    ax.minorticks_on()
    ax.legend(loc="upper right")

    return fig, ax


# ---------------------------------------------------------------------------
# Figure 4: SNR distribution
# ---------------------------------------------------------------------------


def plot_snr_distribution(
    data_dir: Path,
    snr_threshold: float = SNR_THRESHOLD,
) -> tuple[Figure, Any]:
    """Plot SNR distribution of detected EMRI events.

    If CRB CSV data is available in *data_dir*, creates a two-panel figure:
    left = SNR histogram, right = SNR vs luminosity distance scatter.
    Otherwise creates a placeholder documenting what data is needed.

    Parameters
    ----------
    data_dir:
        Root directory to search for CRB CSV files.
    snr_threshold:
        Detection threshold (vertical line on histogram).

    Returns
    -------
    (fig, axes) following the project factory convention.
    """
    import glob

    # Search for CRB CSV files
    csv_patterns = [
        str(data_dir / "**" / "crb*.csv"),
        str(data_dir / "**" / "CRB*.csv"),
        str(data_dir / "**" / "cramer_rao*.csv"),
        str(data_dir / "*.csv"),
    ]
    csv_files: list[str] = []
    for pat in csv_patterns:
        csv_files.extend(glob.glob(pat, recursive=True))

    if csv_files:
        # ------ Data available path ------
        import pandas as pd

        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)

        # Identify columns (case-insensitive)
        col_map = {c.lower(): c for c in df.columns}
        snr_col = col_map.get("snr", col_map.get("signal_to_noise_ratio"))
        dl_col = col_map.get("d_l", col_map.get("luminosity_distance"))
        z_col = col_map.get("z", col_map.get("redshift"))

        if snr_col is None:
            msg = f"No SNR column found in CRB CSV. Columns: {list(df.columns)}"
            raise ValueError(msg)

        snr = df[snr_col].to_numpy()
        detected = snr >= snr_threshold

        fig, axes = get_figure(nrows=1, ncols=2, preset="double")
        ax_hist: Axes = axes[0]
        ax_scat: Axes = axes[1]

        # Left: SNR histogram
        snr_det = snr[detected]
        ax_hist.hist(snr_det, bins=30, color=CYCLE[0], edgecolor=EDGE, alpha=0.8)
        ax_hist.axvline(snr_threshold, color=MEAN, linestyle="--", linewidth=1.2, label="Threshold")
        ax_hist.set_xlabel("SNR")
        ax_hist.set_ylabel("Number of events")
        ax_hist.legend()

        # Right: SNR vs d_L scatter
        if dl_col is not None:
            d_l = df[dl_col].to_numpy()
            if z_col is not None:
                z = df[z_col].to_numpy()
                sc = ax_scat.scatter(
                    d_l[detected],
                    snr_det,
                    c=z[detected],
                    # Route the sequential redshift coloring through the house
                    # cmap (D-CMAP-05) -- drops the orphan "viridis".
                    cmap=CMAP,
                    s=8,
                    alpha=0.6,
                )
                fig.colorbar(sc, ax=ax_scat, label=r"Redshift $z$")
            else:
                ax_scat.scatter(d_l[detected], snr_det, color=CYCLE[0], s=8, alpha=0.6)
            ax_scat.axhline(snr_threshold, color=MEAN, linestyle="--", linewidth=1.0, alpha=0.7)
            ax_scat.set_xlabel(r"$d_L$ [Gpc]")
            ax_scat.set_ylabel("SNR")
        else:
            ax_scat.text(
                0.5,
                0.5,
                r"$d_L$ column not found",
                transform=ax_scat.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        return fig, axes

    # ------ Placeholder path (no CRB data locally) ------
    fig, ax = get_figure(preset="single")
    ax.text(
        0.5,
        0.55,
        "SNR distribution data not available locally",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.38,
        "Copy CRB CSV files from the cluster to\n"
        "cluster_results/eval_corrected_full/\n"
        "and re-run this figure.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


# ---------------------------------------------------------------------------
# Figure 5: KDE-smoothed H0 posterior comparison (D-05, D-06)
# ---------------------------------------------------------------------------


def plot_h0_posterior_kde(
    data_dir: Path,
) -> tuple[Figure, Axes]:
    """Plot KDE-smoothed H0 posterior comparison.

    Shows the discrete posterior points with a KDE-smoothed overlay
    for both analysis variants. Auto-detects h-grid resolution (D-06).

    Parameters
    ----------
    data_dir:
        Root directory containing combined posterior JSONs.

    Returns
    -------
    (fig, ax) following the project factory convention.
    """
    import logging

    # Local import to avoid a module-level circular dependency
    # (bayesian_plots <-> paper_figures via the canonical factory's KDE path).
    from master_thesis_code.plotting.bayesian_plots import plot_combined_posterior

    _log = logging.getLogger(__name__)

    p_no = _load_combined_posterior("posteriors", data_dir)
    p_with = _load_combined_posterior("posteriors_with_bh_mass", data_dir)

    h_no = np.array(p_no["h_values"])
    h_with = np.array(p_with["h_values"])
    post_no = np.array(p_no["posterior"])
    post_with = np.array(p_with["posterior"])

    # Auto-detect grid spacing (D-06) — no hardcoded grid size
    grid_spacing_no = float(np.diff(h_no).mean())
    grid_spacing_with = float(np.diff(h_with).mean())

    # KDE-MAP-drift diagnostic (kept verbatim): the canonical factory does the
    # smoothing for the render, but we still log here when the smoothed MAP
    # drifts more than one grid spacing from the discrete MAP.
    h_fine_no, kde_no = _kde_smooth_posterior(h_no, post_no)
    h_fine_with, kde_with = _kde_smooth_posterior(h_with, post_with)

    discrete_map_no = h_no[int(np.argmax(post_no))]
    kde_map_no = h_fine_no[int(np.argmax(kde_no))]
    if abs(kde_map_no - discrete_map_no) >= grid_spacing_no:
        _log.warning(
            "KDE MAP (%.4f) drifted more than one grid spacing (%.4f) from discrete MAP (%.4f) "
            "for 'without BH mass' variant",
            kde_map_no,
            grid_spacing_no,
            discrete_map_no,
        )

    discrete_map_with = h_with[int(np.argmax(post_with))]
    kde_map_with = h_fine_with[int(np.argmax(kde_with))]
    if abs(kde_map_with - discrete_map_with) >= grid_spacing_with:
        _log.warning(
            "KDE MAP (%.4f) drifted more than one grid spacing (%.4f) from discrete MAP (%.4f) "
            "for 'with BH mass' variant",
            kde_map_with,
            grid_spacing_with,
            discrete_map_with,
        )

    # Headline (without M_z): KDE-smoothed, area-normalized PDF, navy SOLID,
    # nested 68/95% HDI, inline MAP, dotted "Injected" truth. NO reference bands.
    fig, ax = plot_combined_posterior(
        h_no,
        post_no,
        0.73,
        label=r"Without $M_z$",
        normalize="density",
        kde=True,
        color=VARIANT_NO_MASS,
        linestyle="-",
        linewidth=1.4,
        show_credible=True,
        show_references=False,
        annotate_map=True,
        show_truth=True,
        truth_linestyle=":",
        truth_label="Injected",
        ylabel=r"$p(h \mid \mathrm{data})$",
        xlim=(0.59, 0.87),
        legend=False,
    )

    # Secondary (with M_z): KDE-smoothed gold DASHED, HDI band, no truth/MAP/refs.
    plot_combined_posterior(
        h_with,
        post_with,
        0.73,
        label=r"With $M_z$",
        normalize="density",
        kde=True,
        color=VARIANT_WITH_MASS,
        linestyle="--",
        linewidth=1.4,
        show_credible=True,
        show_references=False,
        annotate_map=False,
        show_truth=False,
        legend=False,
        ax=ax,
    )

    ax.legend(loc="upper right")

    # constrained_layout is on (mplstyle); do NOT also call tight_layout (§1.8).
    return fig, ax


# ---------------------------------------------------------------------------
# Phase F1: Closure-test posterior overlay
# ---------------------------------------------------------------------------


def plot_closure_test_overlay(
    h_runs: dict[float, Path],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Overlay combined H0 posteriors from injections at multiple truths.

    Each run is loaded via the canonical raw Σ log L_i loader (Phase A),
    peak-normalised, and plotted with a colour-coded curve plus a vertical
    truth line at its injected h-value. Demonstrates pipeline closure:
    every posterior should peak near its own truth.

    Parameters
    ----------
    h_runs:
        Dict mapping ``h_true`` (the injection truth) to the run directory
        that contains the ``posteriors/`` subdirectory (1D channel; closure
        runs may not have a 2D channel).
    ax:
        Existing axes; new figure when None.

    Returns
    -------
    (fig, ax) following the project factory convention.
    """
    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        from master_thesis_code.plotting._helpers import _fig_from_ax

        fig = _fig_from_ax(ax)

    from master_thesis_code.plotting._helpers import load_canonical_combined_posterior

    sorted_truths = sorted(h_runs.keys())
    colors = [CYCLE[i % len(CYCLE)] for i in range(len(sorted_truths))]

    plotted = 0
    for color, h_true in zip(colors, sorted_truths, strict=True):
        run_dir = h_runs[h_true]
        try:
            h_grid, posterior, meta = load_canonical_combined_posterior(run_dir, "posteriors")
        except FileNotFoundError:
            continue
        norm = posterior / posterior.max() if posterior.max() > 0 else posterior
        label = (
            rf"$h_\mathrm{{true}}={h_true:.2f}$ "
            f"(MAP {meta['continuous_map']:.3f})"
        )
        ax.plot(h_grid, norm, color=color, linewidth=1.4, label=label)
        ax.axvline(h_true, color=color, linewidth=0.7, linestyle=":")
        plotted += 1

    if plotted == 0:
        raise FileNotFoundError("No closure-test runs could be loaded.")

    ax.set_xlabel(r"$h$")
    ax.set_ylabel("Posterior (peak-normalised)")
    ax.set_xlim(0.55, 0.85)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="best", fontsize="small")
    ax.set_title("Closure test: pipeline recovers each injection truth", fontsize="medium")
    return fig, ax
