"""Factory functions for Bayesian H0 inference plots.

Extracted from ``BayesianStatistics.visualize()`` and
``BayesianStatistics.visualize_galaxy_weights()`` in ``cosmological_model.py``.

All functions take data in and return ``(fig, ax)`` out.  None call
``plt.show()`` or ``plt.savefig()`` -- the caller decides where to save.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import (
    CMAP,
    CYCLE,
    EDGE,
    MEAN,
    METHOD,
    PLANCK_BAND,
    PRIOR,
    REFERENCE,
    SHOES_BAND,
    TRUTH,
)
from master_thesis_code.plotting._helpers import (
    _fig_from_ax,
    compute_hdi_interval,
    get_figure,
)
from master_thesis_code.plotting._labels import LABELS


def _normalize_posterior(
    posterior: npt.NDArray[np.float64],
    h_values: npt.NDArray[np.float64],
    mode: str,
) -> npt.NDArray[np.float64]:
    """Normalize a posterior array by peak or density.

    Parameters
    ----------
    posterior:
        Raw posterior values.
    h_values:
        Corresponding h grid.
    mode:
        ``"peak"`` divides by the maximum so the peak equals 1.
        ``"density"`` divides by the integral so the area equals 1.

    Returns
    -------
    Normalized posterior array.
    """
    if mode == "peak":
        peak = np.max(posterior)
        return posterior / peak if peak > 0 else posterior
    if mode == "density":
        area = np.trapezoid(posterior, h_values)
        return posterior / area if area > 0 else posterior
    msg = f"normalize must be 'peak' or 'density', got {mode!r}"
    raise ValueError(msg)


def plot_combined_posterior(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    true_h: float,
    *,
    label: str | None = None,
    normalize: str = "peak",
    show_credible: bool = True,
    show_references: bool = True,
    show_prior: bool = True,
    color: str | None = None,
    linestyle: str = "-",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a combined Hubble-constant posterior (Observatory style).

    The two pipeline variants share one hue and are separated by *linestyle*
    (solid = Without M_z, dashed = With M_z).  The contextual elements -- the
    flat prior, the nested 50/68/95% HDI shading, the Planck/SH0ES reference
    bands, the injected-truth line, the km/s/Mpc top axis, and the MAP title --
    are drawn once, on the *primary* call (``ax is None``); an overlay call
    (``ax`` given) only adds its curve.

    Parameters
    ----------
    h_values:
        Grid of dimensionless Hubble parameter values.
    posterior:
        Posterior probability at each *h_values* point.
    true_h:
        Injected value of h, drawn as the truth line.
    label:
        Curve label for the legend.
    normalize:
        ``"peak"`` (default) normalizes so the maximum equals 1; ``"density"``
        normalizes so the integral equals 1.
    show_credible:
        Shade the nested 50/68/95% HDI under the (primary) curve.
    show_references:
        Draw the Planck and SH0ES reference bands (primary call only).
    show_prior:
        Draw the flat H0 prior overlay (primary call only).
    color:
        Curve color.  Defaults to the dark-siren blue ``METHOD["dark"]``.
    linestyle:
        Curve linestyle (``"-"`` solid, ``"--"`` dashed).
    ax:
        Optional pre-existing Axes to overlay on.  When given, only the curve
        is drawn (contextual elements are skipped).
    """
    is_primary = ax is None
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if color is None:
        color = METHOD["dark"]

    normalized = _normalize_posterior(posterior, h_values, normalize)

    # --- Reference bands (Planck pink, SH0ES cyan), behind everything ---
    if is_primary and show_references:
        ax.axvspan(0.669, 0.679, color=PLANCK_BAND, alpha=0.18, lw=0, zorder=0, label="Planck")
        ax.axvspan(0.720, 0.740, color=SHOES_BAND, alpha=0.18, lw=0, zorder=0, label="SH0ES")

    # --- Flat H0 prior (peak-normalized -> constant 1.0 over the support) ---
    if is_primary and show_prior:
        ax.plot(
            [float(h_values.min()), float(h_values.max())],
            [1.0, 1.0],
            color=PRIOR,
            linestyle=(0, (4, 3)),
            linewidth=0.8,
            zorder=1,
            label="flat prior",
        )

    # --- Nested 50/68/95% HDI shading under the primary curve ---
    if is_primary and show_credible:
        for level, alpha in ((0.954, 0.12), (0.683, 0.22), (0.500, 0.34)):
            lo, hi = compute_hdi_interval(h_values, normalized, level)
            mask = (h_values >= lo) & (h_values <= hi)
            ax.fill_between(
                h_values, 0.0, normalized, where=mask.tolist(), color=color, alpha=alpha, lw=0
            )

    # --- Main posterior curve ---
    ax.plot(
        h_values,
        normalized,
        color=color,
        linestyle=linestyle,
        linewidth=1.6 if is_primary else 1.3,
        label=label,
        zorder=4,
    )

    if is_primary:
        # Injected truth.
        ax.axvline(
            true_h,
            color=TRUTH,
            linestyle="dashed",
            linewidth=1.0,
            zorder=3,
            label=rf"truth $h={true_h:g}$",
        )
        # Secondary top axis in km/s/Mpc (H0 = 100 h).
        sec = ax.secondary_xaxis(
            "top",
            functions=(
                lambda h: np.asarray(h, dtype=np.float64) * 100.0,
                lambda hh: np.asarray(hh, dtype=np.float64) / 100.0,
            ),
        )
        sec.set_xlabel(r"$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
        # Title: MAP with 68% HDI (kept inside math mode so '%' renders under
        # both mathtext and usetex).
        map_h = float(h_values[int(np.argmax(normalized))])
        lo68, hi68 = compute_hdi_interval(h_values, normalized, 0.683)
        ax.set_title(
            rf"$h = {map_h:.3f}^{{+{hi68 - map_h:.3f}}}_{{-{map_h - lo68:.3f}}}\,(68\,\%)$"
        )
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(LABELS["h"])
        ax.set_ylabel(r"$p(h\,|\,\mathrm{data})$")

    ax.legend(loc="upper left", fontsize=6)
    return fig, ax


def plot_event_posteriors(
    h_values: npt.NDArray[np.float64],
    posteriors: list[npt.NDArray[np.float64]] | dict[int, list[float]],
    true_h: float,
    *,
    color_by: str | None = None,
    color_values: npt.NDArray[np.float64] | None = None,
    combined_posterior: npt.NDArray[np.float64] | None = None,
    normalize: str = "peak",
    title: str = "Individual event posteriors",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot posteriors from individual EMRI detections.

    Parameters
    ----------
    h_values:
        Grid of dimensionless Hubble parameter values.
    posteriors:
        Either a list of arrays or a dict mapping event index to list of
        floats (backward compatible).
    true_h:
        True (injected) value of h.
    color_by:
        If set, color each posterior by a metadata value. One of
        ``"snr"``, ``"redshift"``, ``"dl_error"``.  Requires
        *color_values* to be provided.
    color_values:
        Array of metadata values (same length as *posteriors*) used for
        the colormap when *color_by* is set.
    combined_posterior:
        If provided, overlaid as a thick line on top of individual
        posteriors.
    normalize:
        ``"peak"`` or ``"density"``.
    title:
        Kept for backward compatibility; only set on the axes if
        explicitly provided by the caller (non-default).
    ax:
        Optional pre-existing Axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    # Convert dict -> list for backward compat
    post_list: list[npt.NDArray[np.float64]]
    if isinstance(posteriors, dict):
        post_list = [np.asarray(v, dtype=np.float64) for v in posteriors.values()]
    else:
        post_list = [np.asarray(p, dtype=np.float64) for p in posteriors]

    # Color mapping setup
    colorbar_label_map: dict[str, str] = {
        "snr": LABELS["SNR"],
        "redshift": LABELS["z"],
        "dl_error": r"$\sigma(d_L)/d_L$",
    }
    cmap_obj = None
    norm_obj = None
    if color_by is not None:
        if color_values is None:
            msg = "color_values must be provided when color_by is set"
            raise ValueError(msg)
        norm_obj = Normalize(
            vmin=float(np.min(color_values)),
            vmax=float(np.max(color_values)),
        )
        cmap_obj = plt.get_cmap(CMAP)

    # Plot individual posteriors
    for i, post in enumerate(post_list):
        normed = _normalize_posterior(post, h_values, normalize)
        if color_by is not None and cmap_obj is not None and norm_obj is not None:
            color = cmap_obj(norm_obj(float(color_values[i])))  # type: ignore[index]
            ax.plot(h_values, normed, alpha=0.5, linewidth=0.5, color=color)
        else:
            ax.plot(h_values, normed, alpha=0.3, linewidth=0.5, color=CYCLE[0])

    # Colorbar
    if color_by is not None and cmap_obj is not None and norm_obj is not None:
        sm = ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])
        cb_label = colorbar_label_map.get(color_by, color_by)
        fig.colorbar(sm, ax=ax, label=cb_label)

    # Combined posterior overlay
    if combined_posterior is not None:
        normed_combined = _normalize_posterior(combined_posterior, h_values, normalize)
        ax.plot(
            h_values,
            normed_combined,
            color=EDGE,
            linewidth=2.0,
            label="Combined",
        )

    ax.axvline(true_h, color=TRUTH, linestyle="dashed")
    ax.set_xlabel(LABELS["h"])
    ax.set_ylabel(r"$p(h|\mathrm{data})$")

    # Only set title if caller explicitly passed a non-default value
    if title != "Individual event posteriors":
        ax.set_title(title)

    return fig, ax


def plot_subset_posteriors(
    h_values: npt.NDArray[np.float64],
    subset_posteriors: list[npt.NDArray[np.float64]],
    true_h: float,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot combined posteriors for random subsets of detections."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    for posterior in subset_posteriors:
        normalized = posterior / np.max(posterior) if np.max(posterior) > 0 else posterior
        ax.plot(h_values, normalized, alpha=0.5, linewidth=0.8)
    ax.axvline(true_h, color=TRUTH, linestyle="dashed")
    ax.set_xlabel(LABELS["h"])
    ax.set_ylabel(r"$p(h|\mathrm{data})$")
    return fig, ax


def plot_detection_redshift_distribution(
    redshifts: npt.NDArray[np.float64],
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of detection redshifts."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(redshifts, bins=bins, edgecolor=EDGE, alpha=0.7)
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Count")
    return fig, ax


def plot_number_of_possible_hosts(
    host_counts: npt.NDArray[np.float64],
    *,
    bins: int = 30,
    label: str = "Possible hosts",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of number of possible host galaxies per detection."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(host_counts, bins=bins, edgecolor=EDGE, alpha=0.7, label=label)
    ax.set_xlabel("Number of possible hosts")
    ax.set_ylabel("Count")
    ax.legend()
    return fig, ax


def plot_snr_distribution(
    snr_values: npt.NDArray[np.float64],
    *,
    snr_threshold: float = 20.0,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of SNR values with CDF overlay and threshold annotation.

    Parameters
    ----------
    snr_values:
        Array of signal-to-noise ratios.
    snr_threshold:
        Threshold value drawn as a vertical dashed line.
    bins:
        Number of histogram bins.
    ax:
        Optional pre-existing Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and the primary (left) Axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    # Left y-axis: histogram
    ax.hist(snr_values, bins=bins, edgecolor=EDGE, alpha=0.7, color=CYCLE[0])

    # Right y-axis: CDF step function
    ax2 = ax.twinx()
    sorted_snr = np.sort(snr_values)
    cdf = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr)
    ax2.step(sorted_snr, cdf, color=MEAN, where="post", linewidth=1.5)
    ax2.set_ylabel("Cumulative fraction")
    ax2.set_ylim(0, 1)

    # Threshold annotation
    ax.axvline(snr_threshold, color=REFERENCE, linestyle="--", linewidth=1)
    frac_above = float(np.mean(snr_values >= snr_threshold))
    ax.annotate(
        f"{frac_above:.0%} above threshold",
        xy=(snr_threshold, 0),
        xytext=(snr_threshold * 1.1, ax.get_ylim()[1] * 0.8),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": REFERENCE},
    )

    ax.set_xlabel(LABELS["SNR"])
    ax.set_ylabel("Count")
    return fig, ax
