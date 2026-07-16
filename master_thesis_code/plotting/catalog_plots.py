"""Factory functions for galaxy catalog plots.

Extracted from ``GalaxyCatalogueHandler.visualize_galaxy_catalog()`` in ``handler.py``.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import (
    EDGE,
    REFERENCE,
    VARIANT_NO_MASS,
    VARIANT_WITH_MASS,
)
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
from master_thesis_code.plotting._labels import LABELS


def plot_bh_mass_distribution(
    masses: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of estimated black hole masses."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    log_bins = np.geomspace(masses.min(), masses.max(), bins).tolist()
    ax.hist(masses, bins=log_bins, edgecolor=EDGE, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["M"])
    ax.set_ylabel("Count")
    return fig, ax


def plot_redshift_distribution(
    redshifts: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of galaxy redshifts."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(redshifts, bins=bins, edgecolor=EDGE, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Count")
    return fig, ax


def plot_glade_completeness(
    distance_range: npt.NDArray[np.float64],
    completeness: npt.NDArray[np.float64],
    *,
    reference_curve: npt.NDArray[np.float64] | None = None,
    label: str = "Empirical",
    reference_label: str = "Reference (literature)",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """GLADE catalog completeness as a function of luminosity distance.

    Parameters
    ----------
    distance_range:
        Bin centres in luminosity distance, in Gpc (matches the axis label).
    completeness:
        Empirical completeness fraction in ``[0, 1]`` on ``distance_range``.
    reference_curve:
        Optional literature completeness curve (e.g. Gehrels et al. 2016
        K-band completeness) to overlay for visual reference.
    label:
        Legend label for the empirical curve.
    reference_label:
        Legend label for the optional reference curve.
    ax:
        Existing axes to plot on. When ``None``, a new figure is created.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(distance_range, completeness, color=VARIANT_NO_MASS, linewidth=1.6, label=label)
    if reference_curve is not None:
        ax.plot(
            distance_range,
            reference_curve,
            color=REFERENCE,
            linestyle="--",
            linewidth=1.2,
            label=reference_label,
        )
    ax.set_xlabel(r"$d_L\,[\mathrm{Gpc}]$")
    ax.set_ylabel("Completeness")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize="small", loc="best")
    return fig, ax


def plot_event_catalog_coverage(
    host_counts: pd.DataFrame,
    *,
    d_l_per_event: npt.NDArray[np.float64] | None = None,
    n_bins: int = 12,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Per-event GLADE host-candidate counts and BH-mass reduction.

    Built from the host-count CSV produced by
    :func:`master_thesis_code.analysis.parse_host_counts.build_host_count_csv`
    and (optionally) the per-event luminosity distances from the CRB CSV.

    Parameters
    ----------
    host_counts:
        DataFrame with columns ``event_idx``, ``n_without_mass``,
        ``n_with_mass``, ``reduction_frac``.
    d_l_per_event:
        Optional float array of length ``len(host_counts)`` giving the
        observed luminosity distance per event. When provided, the
        histogram is binned by d_L; otherwise it is binned over the
        ``event_idx`` axis.
    n_bins:
        Number of bins along the d_L (or event_idx) axis.
    ax:
        Existing axes; new figure when ``None``.
    """
    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    n_events = len(host_counts)
    if d_l_per_event is not None and len(d_l_per_event) == n_events:
        x_values = np.asarray(d_l_per_event, dtype=np.float64)
        x_label = r"$d_L\,[\mathrm{Gpc}]$"
    else:
        x_values = host_counts["event_idx"].to_numpy(dtype=np.float64)
        x_label = "Event index"

    # Bin and compute median ± p16/p84 per bin (more robust than mean ± sd
    # when the host-count distribution is heavy-tailed).
    bin_edges = np.linspace(x_values.min(), x_values.max(), n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    no_med = np.zeros(n_bins)
    no_lo = np.zeros(n_bins)
    no_hi = np.zeros(n_bins)
    wm_med = np.zeros(n_bins)
    wm_lo = np.zeros(n_bins)
    wm_hi = np.zeros(n_bins)
    coverage = np.zeros(n_bins)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (x_values >= lo) & (x_values < hi if i < n_bins - 1 else x_values <= hi)
        if not mask.any():
            continue
        no_in = host_counts["n_without_mass"].to_numpy()[mask]
        wm_in = host_counts["n_with_mass"].to_numpy()[mask]
        no_med[i] = float(np.median(no_in))
        no_lo[i] = float(np.quantile(no_in, 0.16))
        no_hi[i] = float(np.quantile(no_in, 0.84))
        wm_med[i] = float(np.median(wm_in))
        wm_lo[i] = float(np.quantile(wm_in, 0.16))
        wm_hi[i] = float(np.quantile(wm_in, 0.84))
        coverage[i] = float((no_in > 0).mean())

    ax.fill_between(centers, no_lo, no_hi, color=VARIANT_NO_MASS, alpha=0.20)
    ax.plot(
        centers,
        no_med,
        color=VARIANT_NO_MASS,
        marker="o",
        linewidth=1.4,
        label="Hosts without $M_z$ cut (median ± p16/p84)",
    )
    ax.fill_between(centers, wm_lo, wm_hi, color=VARIANT_WITH_MASS, alpha=0.20)
    ax.plot(
        centers,
        wm_med,
        color=VARIANT_WITH_MASS,
        marker="s",
        linewidth=1.4,
        linestyle="--",
        label="Hosts with $M_z$ cut (median ± p16/p84)",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Host-galaxy candidates per event")
    ax.set_yscale("log")
    ax.legend(fontsize="small", loc="upper right")

    # Twin axis with coverage fraction.
    ax_cov = ax.twinx()
    ax_cov.plot(
        centers,
        coverage,
        color=EDGE,
        marker="d",
        linewidth=1.0,
        linestyle=":",
        label="Catalog coverage (frac. with $\\geq 1$ host)",
    )
    ax_cov.set_ylim(-0.05, 1.05)
    ax_cov.set_ylabel("Catalog coverage fraction")
    ax_cov.legend(fontsize="small", loc="lower right")

    summary = (
        f"N events = {n_events}\n"
        f"median reduction = {host_counts['reduction_frac'].median():.0%}\n"
        f"mean hosts (no cut) = {host_counts['n_without_mass'].mean():.0f}\n"
        f"mean hosts (+$M_z$) = {host_counts['n_with_mass'].mean():.0f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": EDGE},
    )
    return fig, ax


def gehrels_2016_reference_completeness(
    d_l_grid_gpc: npt.NDArray[np.float64],
    *,
    d_l_50pct_gpc: float = 0.10,
    sharpness: float = 6.0,
) -> npt.NDArray[np.float64]:
    """Schematic GLADE+ completeness reference curve.

    A smooth sigmoid roll-off in luminosity distance approximating the
    Gehrels et al. (2016) / GLADE+ literature completeness behaviour
    (K-band complete to ~100 Mpc, falling off into the 1-Gpc regime).
    This is a *schematic* curve for visual reference only — replace with
    an empirical estimate when one becomes available.

    Parameters
    ----------
    d_l_grid_gpc:
        Luminosity distance grid in Gpc.
    d_l_50pct_gpc:
        Distance at which completeness reaches 50 %.
    sharpness:
        Steepness of the sigmoid roll-off.

    Returns
    -------
    npt.NDArray[np.float64]
        Completeness fraction in ``[0, 1]`` on ``d_l_grid_gpc``.
    """
    x = sharpness * (np.log10(d_l_grid_gpc) - np.log10(d_l_50pct_gpc))
    result: npt.NDArray[np.float64] = (1.0 / (1.0 + np.exp(x))).astype(np.float64)
    return result


def plot_comoving_volume_sampling(
    samples: npt.NDArray[np.float64],
    *,
    bins: int = 20,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of comoving volume MCMC samples."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(samples, bins=bins, density=True, edgecolor=EDGE, alpha=0.7)
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Density")
    return fig, ax
