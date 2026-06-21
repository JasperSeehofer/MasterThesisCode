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
    PLANCK,
    REFERENCE,
    SH0ES,
    TRUTH,
    VARIANT_NO_MASS,
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
    annotate_map: bool = True,
    show_truth: bool = True,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float | None = None,
    truth_linestyle: str = "dashed",
    truth_label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ylabel: str | None = None,
    kde: bool = False,
    legend: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single combined Hubble constant posterior.

    Headline single-posterior treatment (viz-redesign §1.3, §3.2): an
    area-normalizable PDF with nested 68%/95% highest-density-interval (HDI)
    shading and an inline ``MAP +x/-y`` annotation on the curve (the
    "Dispatch" number-on-the-curve discipline). The HDI bounds come from the
    shared :func:`compute_hdi_interval` helper (LIGO/Virgo minimal-credible
    convention), replacing the older cumsum-index CI machinery.

    This is the **single canonical combined-H0-posterior factory** (v2.3
    HORIZON Phase 01). Every combined-posterior render path — fig01 (manifest),
    ``paper_h0_posterior``/``paper_h0_posterior_kde``, the M_z improvement
    top-middle panel, and the fig08 left panel — delegates here so every
    recolor/annotation edit lands in one place and the quadruplicate-drift
    hazard (different MAPs from copy-pasted plotting code) cannot return.

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
        ``"density"`` normalizes so the integral equals 1. The headline
        area-normalized PDF is selected by passing ``"density"``; the default
        stays ``"peak"`` so multi-variant overlay callers are unchanged.
    show_credible:
        If ``True`` (default), shade nested 68% and 95% HDI regions.
    show_references:
        If ``True`` (default), show Planck and SH0ES reference bands.
    annotate_map:
        If ``True`` (default), add an inline ``MAP = .. +.. /-..`` text
        annotation near the peak. Suppress (``False``) for multi-variant
        overlays where several MAP labels would collide.
    show_truth:
        If ``True`` (default), draw the dashed injected-``h`` truth line and
        its legend entry. Suppress (``False``) on secondary overlay curves so
        a multi-variant comparison shows a single truth line.
    color:
        Curve and HDI shading color.  Defaults to ``VARIANT_NO_MASS``.
    linestyle:
        Linestyle of the posterior curve (default ``"-"``). Secondary overlay
        variants pass ``"--"`` for the redundant color+linestyle encoding.
    linewidth:
        Posterior curve linewidth. ``None`` (default) inherits the stylesheet
        default; callers pass e.g. ``1.4`` for the KDE / panel curves.
    truth_linestyle:
        Linestyle for the injected-``h`` truth line (default ``"dashed"``).
        Paper figures pass ``":"``; the fig08 left panel passes ``"dashed"``.
    truth_label:
        Legend label for the truth line. ``None`` (default) yields
        ``f"True $h = {true_h}$"``; callers pass ``"Injected"`` / ``"Truth"``.
    xlim:
        Optional ``(low, high)`` x-axis limits. ``None`` leaves them auto.
    ylim:
        Optional ``(low, high)`` y-axis limits. ``None`` leaves them auto.
    ylabel:
        Y-axis label override. ``None`` (default) uses ``p(h|data)``; the
        fig08 left panel passes ``"Posterior density"``.
    kde:
        If ``True``, Gaussian-KDE-smooth ``(h_values, posterior)`` (Scott's
        rule) BEFORE normalizing/plotting, and compute the HDI/MAP from the
        smoothed curve so the rendered MAP matches what the panel shows.
        Default ``False`` leaves the discrete grid untouched.
    legend:
        If ``True`` (default), call ``ax.legend()``. Overlay callers pass
        ``False`` so the surrounding figure owns a single legend.
    ax:
        Optional pre-existing Axes to draw on.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if color is None:
        color = VARIANT_NO_MASS

    # KDE smoothing (path 3): smooth onto a fine grid BEFORE normalize/HDI/MAP
    # so the rendered curve, its shaded HDI, and the inline MAP all agree. The
    # import is local to avoid a circular dependency — paper_figures imports
    # from convergence_analysis, and bayesian_plots must not import
    # paper_figures at module top.
    if kde:
        from master_thesis_code.plotting.paper_figures import _kde_smooth_posterior

        h_values, posterior = _kde_smooth_posterior(h_values, posterior)

    normalized = _normalize_posterior(posterior, h_values, normalize)

    # Main posterior curve (plain line; no per-point markers)
    plot_kwargs: dict[str, object] = {"label": label, "color": color, "linestyle": linestyle}
    if linewidth is not None:
        plot_kwargs["linewidth"] = linewidth
    ax.plot(h_values, normalized, **plot_kwargs)  # type: ignore[arg-type]

    # --- Nested HDI bands via the shared HDI helper (one CI definition) ---
    # 68% HDI is needed for both the bands and the inline MAP annotation.
    lo68, hi68 = compute_hdi_interval(h_values, posterior, level=0.683)
    if show_credible:
        lo95, hi95 = compute_hdi_interval(h_values, posterior, level=0.954)
        # 95% region (lighter), then nested 68% region (darker) on top.
        if not (np.isnan(lo95) or np.isnan(hi95)):
            mask_95 = ((h_values >= lo95) & (h_values <= hi95)).tolist()
            ax.fill_between(h_values, 0, normalized, where=mask_95, alpha=0.15, color=color)
        if not (np.isnan(lo68) or np.isnan(hi68)):
            mask_68 = ((h_values >= lo68) & (h_values <= hi68)).tolist()
            ax.fill_between(h_values, 0, normalized, where=mask_68, alpha=0.30, color=color)

    # --- Reference bands (D-02) — reserved PLANCK/SH0ES band colors ---
    if show_references:
        # Planck: h = 0.674 +/- 0.005
        planck_h, planck_sigma = 0.674, 0.005
        ax.axvspan(
            planck_h - planck_sigma,
            planck_h + planck_sigma,
            alpha=0.15,
            color=PLANCK,
            zorder=0,
        )
        ax.axvline(planck_h, color=PLANCK, linewidth=0.8, linestyle="--")
        ax.text(
            planck_h,
            0.95,
            "Planck",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            color=PLANCK,
        )

        # SH0ES: h = 0.73 +/- 0.01
        shoes_h, shoes_sigma = 0.73, 0.01
        ax.axvspan(
            shoes_h - shoes_sigma,
            shoes_h + shoes_sigma,
            alpha=0.15,
            color=SH0ES,
            zorder=0,
        )
        ax.axvline(shoes_h, color=SH0ES, linewidth=0.8, linestyle="--")
        ax.text(
            shoes_h,
            0.95,
            "SH0ES",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6,
            color=SH0ES,
        )

    # --- Inline MAP +/- 68% HDI annotation (Dispatch number-on-the-curve) ---
    if annotate_map and not (np.isnan(lo68) or np.isnan(hi68)):
        map_idx = int(np.argmax(normalized))
        map_h = float(h_values[map_idx])
        map_y = float(normalized[map_idx])
        ax.annotate(
            rf"MAP $= {map_h:.3f}^{{+{hi68 - map_h:.3f}}}_{{-{map_h - lo68:.3f}}}$",
            xy=(map_h, map_y),
            xytext=(0.0, 6.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )

    # Truth line (suppressed on secondary overlay curves to avoid duplicate
    # legend entries / overlapping reference lines in multi-variant comparisons)
    if show_truth:
        truth_lbl = truth_label if truth_label is not None else f"True $h = {true_h}$"
        ax.axvline(true_h, color=TRUTH, linestyle=truth_linestyle, label=truth_lbl)

    ax.set_xlabel(LABELS["h"])
    ax.set_ylabel(ylabel if ylabel is not None else r"$p(h|\mathrm{data})$")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
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
