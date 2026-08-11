"""Factory functions for cosmological model plots.

Extracted from ``Model1CrossCheck`` and ``DetectionProbability`` in
``cosmological_model.py``.
"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from darksiren_emri.plotting._colors import CMAP, CYCLE, EDGE, METHOD, REFERENCE
from darksiren_emri.plotting._helpers import _fig_from_ax, get_figure, make_colorbar
from darksiren_emri.plotting._labels import LABELS


def _plot_detection_heatmap(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    prob: npt.NDArray[np.float64],
    xlabel: str,
    ylabel: str,
    *,
    contour_levels: list[float] | None = None,
    injected_coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    detected_mask: npt.NDArray[np.bool_] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Shared implementation for detection probability heatmaps.

    Parameters
    ----------
    x, y:
        Meshgrid arrays for the two axes.
    prob:
        Detection probability values on the grid, expected in [0, 1].
    xlabel, ylabel:
        LaTeX axis labels.
    contour_levels:
        Probability thresholds for contour lines (default ``[0.5, 0.9]``).
    injected_coords:
        Tuple ``(x_array, y_array)`` of injected event coordinates for
        scatter overlay.
    detected_mask:
        Boolean mask selecting detected events.  When *injected_coords*
        is given, detected events are shown as filled circles and missed
        events as open circles.  Ignored when *injected_coords* is None.
    ax:
        Optional pre-existing axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if contour_levels is None:
        contour_levels = [0.5, 0.9]

    cs = ax.contourf(
        x,
        y,
        prob,
        levels=np.linspace(0, 1, 51),
        cmap=CMAP,
        vmin=0,
        vmax=1,
    )

    # Contour lines at specified probability thresholds
    contours = ax.contour(
        x,
        y,
        prob,
        levels=contour_levels,
        colors=EDGE,
        linewidths=1.0,
    )
    ax.clabel(contours, inline=True, fontsize=8)

    # Scatter overlay for injected population
    if injected_coords is not None:
        inj_x, inj_y = injected_coords
        if detected_mask is not None:
            ax.scatter(
                inj_x[detected_mask],
                inj_y[detected_mask],
                marker="o",
                facecolors=CYCLE[0],
                edgecolors=EDGE,
                s=10,
                alpha=0.6,
                zorder=3,
            )
            ax.scatter(
                inj_x[~detected_mask],
                inj_y[~detected_mask],
                marker="o",
                facecolors="none",
                edgecolors=CYCLE[3],
                s=10,
                alpha=0.6,
                zorder=3,
            )
        else:
            ax.scatter(
                inj_x,
                inj_y,
                marker="o",
                facecolors=CYCLE[0],
                edgecolors=EDGE,
                s=10,
                alpha=0.6,
                zorder=3,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    make_colorbar(cs, fig, ax, label=r"$P_\mathrm{det}$")
    return fig, ax


def plot_emri_distribution(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    distribution: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of the EMRI event distribution in (z, M) space."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    cs = ax.contourf(redshifts, masses, distribution, cmap=CMAP, levels=30)
    fig.colorbar(cs, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(LABELS["M"])
    return fig, ax


def plot_emri_rate(
    masses: npt.NDArray[np.float64],
    rates: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Log-log plot of EMRI rate vs MBH mass."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.plot(masses, rates)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(LABELS["M"])
    ax.set_ylabel("EMRI rate R [1/Gyr]")
    return fig, ax


def plot_emri_sampling(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    redshift_bins: npt.NDArray[np.float64],
    mass_bins: npt.NDArray[np.float64],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """2D histogram of sampled EMRI events."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    h = ax.hist2d(redshifts, masses, bins=[redshift_bins, mass_bins], cmap=CMAP)
    fig.colorbar(h[3], ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(LABELS["M"])
    return fig, ax


def plot_detection_probability_grid(
    d_L_range: npt.NDArray[np.float64],
    M_range: npt.NDArray[np.float64],
    detection_prob: npt.NDArray[np.float64],
    *,
    contour_levels: list[float] | None = None,
    injected_coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    detected_mask: npt.NDArray[np.bool_] | None = None,
    title: str = "",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of detection probability in (d_L, M) space.

    Parameters
    ----------
    d_L_range, M_range:
        Meshgrid arrays for luminosity distance and mass.
    detection_prob:
        Detection probability on the grid.
    contour_levels:
        Probability thresholds for contour lines (default ``[0.5, 0.9]``).
    injected_coords:
        ``(d_L_array, M_array)`` for scatter overlay of injected events.
    detected_mask:
        Boolean mask selecting detected events (filled circles).
        Missed events are shown as open circles.
    title:
        Optional figure title.  Omitted when empty (thesis convention).
    ax:
        Optional pre-existing axes.
    """
    fig, ax = _plot_detection_heatmap(
        d_L_range,
        M_range,
        detection_prob,
        LABELS["d_L"],
        LABELS["M"],
        contour_levels=contour_levels,
        injected_coords=injected_coords,
        detected_mask=detected_mask,
        ax=ax,
    )
    if title:
        ax.set_title(title)
    return fig, ax


def plot_detection_probability_zM(
    z_range: npt.NDArray[np.float64],
    M_range: npt.NDArray[np.float64],
    detection_prob: npt.NDArray[np.float64],
    *,
    contour_levels: list[float] | None = None,
    injected_coords: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
    detected_mask: npt.NDArray[np.bool_] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Contour plot of detection probability in (z, M) space.

    Parameters
    ----------
    z_range, M_range:
        Meshgrid arrays for redshift and mass.
    detection_prob:
        Detection probability on the grid.
    contour_levels:
        Probability thresholds for contour lines (default ``[0.5, 0.9]``).
    injected_coords:
        ``(z_array, M_array)`` for scatter overlay of injected events.
    detected_mask:
        Boolean mask selecting detected events (filled circles).
        Missed events are shown as open circles.
    ax:
        Optional pre-existing axes.
    """
    return _plot_detection_heatmap(
        z_range,
        M_range,
        detection_prob,
        LABELS["z"],
        LABELS["M"],
        contour_levels=contour_levels,
        injected_coords=injected_coords,
        detected_mask=detected_mask,
        ax=ax,
    )


# ---------------------------------------------------------------------------
# LISA noise / sensitivity (Observatory style)
# ---------------------------------------------------------------------------
#
# The total A-channel PSD splits as  S_n = S_inst + S_gal.  Both fig10 (PSD)
# and fig13 (characteristic strain) decompose the same physics; each component
# is given its OWN linestyle so the figures survive greyscale printing and CVD:
#   total       -> solid, near-black (EDGE)
#   instrument  -> dashed, sky-blue (REFERENCE)
#   confusion   -> dash-dot, dark-siren blue (METHOD["dark"])
# A single shared style table keeps the two figures visually consistent.
# Matplotlib accepts either a named style ("-") or an (offset, on-off dash)
# tuple as a linestyle; the on-off sequence encodes each noise component.
_LineStyle = str | tuple[int, tuple[int, ...]]
_NOISE_STYLE: dict[str, tuple[str, _LineStyle, str]] = {
    # key -> (color, linestyle, math label)
    "total": (EDGE, "-", r"$S_n(f)$ total"),
    "instrument": (REFERENCE, (0, (5, 2)), r"$S_\mathrm{inst}(f)$"),
    "confusion": (METHOD["dark"], (0, (3, 1, 1, 1)), r"$S_\mathrm{gal}(f)$"),
}


def _lisa_psd_components(
    frequencies: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(total, instrument, confusion)`` A-channel PSD on *frequencies*.

    The confusion (galactic foreground) term is recovered as the difference of
    the total and instrument-only PSDs and clipped at zero to suppress
    numerical noise where the foreground is negligible.
    """
    # Deferred import: LISA_configuration pulls cupy at module top level and is
    # not importable on a CPU-only box without the guarded try/except.
    from darksiren_emri.LISA_configuration import LisaTdiConfiguration

    lisa_total = LisaTdiConfiguration(include_confusion_noise=True)
    lisa_inst = LisaTdiConfiguration(include_confusion_noise=False)

    psd_total = np.asarray(
        lisa_total.power_spectral_density_a_channel(frequencies), dtype=np.float64
    )
    psd_inst = np.asarray(lisa_inst.power_spectral_density_a_channel(frequencies), dtype=np.float64)
    psd_confusion = np.maximum(psd_total - psd_inst, 0.0)
    return psd_total, psd_inst, psd_confusion


def plot_lisa_psd(
    frequencies: npt.NDArray[np.float64],
    psd_values: dict[str, npt.NDArray[np.float64]] | None = None,
    *,
    decompose: bool = False,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot the LISA A-channel power spectral density :math:`S_n(f)`.

    With ``decompose=True`` the total PSD is broken into its instrument and
    galactic-confusion contributions, each carrying its own linestyle (not just
    colour) so the decomposition reads in greyscale.  REVTeX single-column.

    Parameters
    ----------
    frequencies:
        Frequency array in Hz (log-spaced).
    psd_values:
        Backward-compatible mode: mapping ``channel -> PSD array`` plotted as
        plain solid curves.  Ignored when *decompose* is ``True``.
    decompose:
        If ``True``, compute and overlay the total / instrument / confusion
        curves via :class:`LisaTdiConfiguration`.
    ax:
        Optional pre-existing Axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    if decompose:
        psd_total, psd_inst, psd_confusion = _lisa_psd_components(frequencies)
        for key, psd, lw in (
            ("total", psd_total, 1.8),
            ("instrument", psd_inst, 1.3),
            ("confusion", psd_confusion, 1.3),
        ):
            color, linestyle, math_label = _NOISE_STYLE[key]
            ax.plot(
                frequencies,
                psd,
                color=color,
                linestyle=linestyle,
                linewidth=lw,
                label=math_label,
            )
    elif psd_values is not None:
        for label, psd in psd_values.items():
            ax.plot(
                frequencies,
                psd,
                color=EDGE,
                linestyle="-",
                linewidth=1.0,
                label=f"$S_{{{label}}}(f)$",
            )

    ax.set_xlabel(LABELS["f"])
    ax.set_ylabel(LABELS["PSD"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7, loc="upper center")
    return fig, ax


def plot_characteristic_strain(
    *,
    f_min: float = 1e-5,
    f_max: float = 1.0,
    n_points: int = 1000,
    emri_amplitude: float = 1e-20,
    emri_f_ref: float = 1e-2,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    r"""Plot the LISA characteristic-strain sensitivity :math:`h_c(f)`.

    This is the single canonical sensitivity figure (it supersedes the older
    PSD/strain duplication).  Characteristic strain
    :math:`h_c = \sqrt{f\,S_n(f)}` is shown for the total, instrument-only and
    galactic-confusion noise -- each with its own linestyle, matching
    :func:`plot_lisa_psd` -- together with a representative inspiral EMRI track
    (:math:`h \propto f^{-7/6}`) to set the scale.  REVTeX single-column.

    Parameters
    ----------
    f_min, f_max:
        Frequency bounds in Hz.
    n_points:
        Number of log-spaced frequency samples.
    emri_amplitude:
        Strain amplitude of the example EMRI track at *emri_f_ref*.
    emri_f_ref:
        Reference frequency (Hz) at which the example track equals
        *emri_amplitude*.
    ax:
        Optional pre-existing Axes.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    freqs = np.geomspace(f_min, f_max, n_points)
    psd_total, psd_inst, psd_confusion = _lisa_psd_components(freqs)

    # Characteristic strain h_c = sqrt(f * S_n(f)).
    h_total = np.sqrt(freqs * psd_total)
    h_inst = np.sqrt(freqs * psd_inst)
    h_confusion = np.sqrt(freqs * psd_confusion)

    for key, h_c, lw in (
        ("total", h_total, 1.8),
        ("instrument", h_inst, 1.3),
        ("confusion", h_confusion, 1.3),
    ):
        color, linestyle, _ = _NOISE_STYLE[key]
        # Re-label in strain terms (h_c) rather than PSD (S_n).
        strain_label = {
            "total": r"$h_c$ total",
            "instrument": r"$h_\mathrm{inst}$",
            "confusion": r"$h_\mathrm{gal}$",
        }[key]
        ax.loglog(freqs, h_c, color=color, linestyle=linestyle, linewidth=lw, label=strain_label)

    # Representative EMRI inspiral track: leading-order h_c ~ f^{-7/6}.
    h_emri = emri_amplitude * (freqs / emri_f_ref) ** (-7.0 / 6.0)
    ax.loglog(
        freqs,
        h_emri,
        color=METHOD["spectral"],
        linestyle="-",
        linewidth=1.3,
        label="example EMRI",
    )

    ax.set_xlabel(LABELS["f"])
    ax.set_ylabel(r"$h_c(f)$")
    ax.legend(fontsize=7, loc="upper right")
    return fig, ax
