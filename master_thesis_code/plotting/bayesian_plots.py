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

from master_thesis_code.plotting._colors import CMAP, CYCLE, EDGE, TRUTH
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
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
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single combined Hubble constant posterior.

    Parameters
    ----------
    h_values:
        Grid of dimensionless Hubble parameter values.
    posterior:
        Posterior probability at each *h_values* point.
    true_h:
        True (injected) value of h for the reference line.
    label:
        Optional curve label for the legend.
    normalize:
        ``"peak"`` (default) normalizes so the maximum equals 1.
        ``"density"`` normalizes so the integral equals 1.
    show_credible:
        If ``True`` (default), shade 68% and 95% credible intervals.
    show_references:
        If ``True`` (default), show Planck and SH0ES reference bands.
    ax:
        Optional pre-existing Axes to draw on.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    normalized = _normalize_posterior(posterior, h_values, normalize)

    # Main posterior curve
    ax.plot(h_values, normalized, label=label)

    # --- Credible intervals (D-01) ---
    if show_credible:
        cumsum = np.cumsum(normalized)
        cumsum = cumsum / cumsum[-1]  # CDF
        quantiles = [0.025, 0.16, 0.84, 0.975]
        indices = [int(np.searchsorted(cumsum, q)) for q in quantiles]
        # Clamp to valid range
        indices = [min(i, len(h_values) - 1) for i in indices]
        h_q = [h_values[i] for i in indices]

        # 95% region
        mask_95 = (h_values >= h_q[0]) & (h_values <= h_q[3])
        ax.fill_between(
            h_values,
            0,
            normalized,
            where=mask_95,
            alpha=0.15,
            color=CYCLE[0],
        )
        # 68% region
        mask_68 = (h_values >= h_q[1]) & (h_values <= h_q[2])
        ax.fill_between(
            h_values,
            0,
            normalized,
            where=mask_68,
            alpha=0.3,
            color=CYCLE[0],
        )
        # Thin boundary lines at interval edges
        for h_edge in h_q:
            ax.axvline(h_edge, color=CYCLE[0], linewidth=0.5, alpha=0.5)

    # --- Reference bands (D-02) ---
    if show_references:
        # Planck: h = 0.674 +/- 0.005
        planck_h, planck_sigma = 0.674, 0.005
        ax.axvspan(
            planck_h - planck_sigma,
            planck_h + planck_sigma,
            alpha=0.15,
            color=CYCLE[4],
            zorder=0,
        )
        ax.axvline(planck_h, color=CYCLE[4], linewidth=0.8, linestyle="--")
        ax.text(
            planck_h,
            0.95,
            "Planck",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            color=CYCLE[4],
        )

        # SH0ES: h = 0.73 +/- 0.01
        shoes_h, shoes_sigma = 0.73, 0.01
        ax.axvspan(
            shoes_h - shoes_sigma,
            shoes_h + shoes_sigma,
            alpha=0.15,
            color=CYCLE[1],
            zorder=0,
        )
        ax.axvline(shoes_h, color=CYCLE[1], linewidth=0.8, linestyle="--")
        ax.text(
            shoes_h,
            0.95,
            "SH0ES",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            color=CYCLE[1],
        )

    # Truth line
    ax.axvline(true_h, color=TRUTH, linestyle="dashed", label=f"True $h = {true_h}$")

    ax.set_xlabel(LABELS["h"])
    ax.set_ylabel(r"$p(h|\mathrm{data})$")
    ax.legend()
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
