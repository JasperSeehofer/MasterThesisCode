"""Fisher matrix visualization factory functions.

Error ellipses, characteristic strain sensitivity curves, and parameter
uncertainty distributions.  All functions follow the project convention:
data in, ``(fig, ax)`` out.  None call ``plt.show()`` or ``plt.savefig()``.
"""

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
import corner
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from master_thesis_code.plotting._colors import CYCLE, EDGE, REFERENCE, TRUTH
from master_thesis_code.plotting._data import (
    EXTRINSIC,
    INTRINSIC,
    PARAMETER_NAMES,
    label_key,
)
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
from master_thesis_code.plotting._labels import LABELS

# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _ellipse_params(cov_2x2: npt.NDArray[np.float64], n_sigma: float) -> tuple[float, float, float]:
    """Compute ellipse width, height, and angle from a 2x2 covariance matrix.

    Parameters
    ----------
    cov_2x2 : npt.NDArray[np.float64]
        Symmetric 2x2 covariance sub-matrix.
    n_sigma : float
        Number of standard deviations for the ellipse boundary.

    Returns
    -------
    tuple[float, float, float]
        ``(width, height, angle_degrees)`` suitable for
        :class:`matplotlib.patches.Ellipse`.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
    # Guard against numerical noise producing tiny negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, 0.0)
    width = 2.0 * n_sigma * np.sqrt(eigenvalues[1])
    height = 2.0 * n_sigma * np.sqrt(eigenvalues[0])
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    return float(width), float(height), float(angle)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------

_DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("M", "mu"),
    ("luminosity_distance", "qS"),
    ("qS", "phiS"),
]


def plot_fisher_ellipses(
    covariance: npt.NDArray[np.float64],
    param_values: npt.NDArray[np.float64],
    pairs: list[tuple[str, str]] | None = None,
    *,
    events: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None = None,
    sigma_levels: tuple[float, ...] = (1.0, 2.0),
    ax: Axes | None = None,
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Plot Fisher error ellipses for selected parameter pairs.

    Parameters
    ----------
    covariance : npt.NDArray[np.float64]
        14x14 covariance matrix (used for single-event mode).
    param_values : npt.NDArray[np.float64]
        14-element array of parameter values (ellipse centres).
    pairs : list[tuple[str, str]] | None
        Parameter pairs to plot.  Defaults to
        ``[("M", "mu"), ("luminosity_distance", "qS"), ("qS", "phiS")]``.
    events : list[tuple[npt.NDArray, npt.NDArray]] | None
        If provided, overlay ellipses for multiple events.  Each element
        is ``(covariance, param_values)``.
    sigma_levels : tuple[float, ...]
        Confidence levels to draw (number of standard deviations).
    ax : Axes | None
        Ignored when using subplot grid (kept for API consistency).

    Returns
    -------
    tuple[Figure, npt.NDArray[np.object_]]
        Figure and ndarray of Axes (one per parameter pair).
    """
    if pairs is None:
        pairs = _DEFAULT_PAIRS

    n_pairs = len(pairs)
    fig, axes = get_figure(nrows=1, ncols=n_pairs, preset="double", squeeze=False)
    # axes shape is (1, n_pairs); flatten to 1-D
    axes_flat: npt.NDArray[np.object_] = np.asarray(axes).flatten()

    # Build event list
    if events is not None:
        event_list = events
    else:
        event_list = [(covariance, param_values)]

    for pair_idx, (name_x, name_y) in enumerate(pairs):
        cur_ax: Axes = axes_flat[pair_idx]
        idx_x = PARAMETER_NAMES.index(name_x)
        idx_y = PARAMETER_NAMES.index(name_y)

        for ev_idx, (cov, vals) in enumerate(event_list):
            color = CYCLE[ev_idx % len(CYCLE)]
            cx = float(vals[idx_x])
            cy = float(vals[idx_y])

            # Extract 2x2 sub-matrix
            indices = [idx_x, idx_y]
            cov_2x2 = cov[np.ix_(indices, indices)]

            for level in sorted(sigma_levels, reverse=True):
                w, h, angle = _ellipse_params(cov_2x2, level)
                alpha = 0.4 / level
                ellipse = Ellipse(
                    xy=(cx, cy),
                    width=w,
                    height=h,
                    angle=angle,
                    facecolor=color,
                    edgecolor=EDGE,
                    linewidth=1.5,
                    alpha=alpha,
                )
                cur_ax.add_patch(ellipse)

            # Auto-scale to ellipse extents
            cur_ax.autoscale_view()

        cur_ax.set_xlabel(LABELS[label_key(name_x)])
        cur_ax.set_ylabel(LABELS[label_key(name_y)])

    fig.tight_layout()
    return fig, axes_flat


def plot_characteristic_strain(
    *,
    f_min: float = 1e-5,
    f_max: float = 1.0,
    n_points: int = 1000,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot LISA characteristic strain sensitivity curve.

    Shows three noise components (total, instrument-only, galactic foreground)
    and a representative EMRI signal track on a log-log scale.

    Parameters
    ----------
    f_min : float
        Lower frequency bound in Hz.
    f_max : float
        Upper frequency bound in Hz.
    n_points : int
        Number of frequency samples (log-spaced).
    ax : Axes | None
        Existing axes to draw on.  Created if ``None``.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes with the strain plot.
    """
    # Deferred import to avoid CPU import issues with LISA_configuration
    from master_thesis_code.LISA_configuration import LisaTdiConfiguration

    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    freqs = np.geomspace(f_min, f_max, n_points)

    lisa_total = LisaTdiConfiguration(include_confusion_noise=True)
    lisa_inst = LisaTdiConfiguration(include_confusion_noise=False)

    psd_total = lisa_total.power_spectral_density_a_channel(freqs)
    psd_inst = lisa_inst.power_spectral_density_a_channel(freqs)
    psd_confusion = psd_total - psd_inst

    # Characteristic strain: h_c = sqrt(f * S_n(f))
    h_total = np.sqrt(freqs * psd_total)
    h_inst = np.sqrt(freqs * psd_inst)
    # Guard against negative confusion PSD from numerical noise
    h_confusion = np.sqrt(freqs * np.maximum(psd_confusion, 0.0))

    ax.loglog(freqs, h_total, color=EDGE, linestyle="-", label="Total")
    ax.loglog(freqs, h_inst, color=REFERENCE, linestyle="--", label="Instrument")
    ax.loglog(freqs, h_confusion, color=CYCLE[1], linestyle=":", label="Galactic foreground")

    # Representative EMRI signal: power-law approximation
    A = 1e-20
    f_ref = 1e-2
    h_emri = A * (freqs / f_ref) ** (-7.0 / 6.0)
    ax.loglog(freqs, h_emri, color=CYCLE[0], linestyle="-", label="Example EMRI")

    ax.set_xlabel(LABELS["f"])
    ax.set_ylabel(r"$h_c(f)$")
    ax.legend(fontsize="small", loc="upper right")

    return fig, ax


def plot_parameter_uncertainties(
    data: pd.DataFrame | pd.Series,
    param_values: pd.DataFrame | pd.Series,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot fractional parameter uncertainties as violin or bar chart.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series
        CRB data.  DataFrame for multi-event (violin plot), Series for
        single-event (bar chart).  Must contain the ``delta_*_delta_*``
        covariance columns.
    param_values : pd.DataFrame | pd.Series
        Parameter values corresponding to *data*.  Columns (or index)
        must match :data:`PARAMETER_NAMES`.
    ax : Axes | None
        Existing axes to draw on.  Created if ``None``.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes with the uncertainty plot.
    """

    is_multi = isinstance(data, pd.DataFrame)

    if is_multi and len(data) >= 10:
        return _plot_violin(data, param_values, ax=ax)
    else:
        # Single event or too few rows for violin
        if is_multi:
            # Use first row for bar chart
            row = data.iloc[0]
            pv = param_values.iloc[0] if isinstance(param_values, pd.DataFrame) else param_values
        else:
            row = data
            pv = param_values
        return _plot_bar(row, pv, ax=ax)


def _plot_violin(
    data: pd.DataFrame,
    param_values: pd.DataFrame | pd.Series,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Violin plot of fractional uncertainties for multiple events."""
    from master_thesis_code.plotting._data import reconstruct_covariance

    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    # Compute fractional uncertainties for each row
    all_frac: list[list[float]] = []
    for idx in range(len(data)):
        row = data.iloc[idx]
        cov = reconstruct_covariance(row)
        sigma = np.sqrt(np.diag(cov))
        if isinstance(param_values, pd.DataFrame):
            pv = np.array([float(param_values.iloc[idx][p]) for p in PARAMETER_NAMES])
        else:
            pv = np.array([float(param_values[p]) for p in PARAMETER_NAMES])
        frac = sigma / np.abs(pv)
        all_frac.append(list(frac))

    frac_array = np.array(all_frac)  # shape: (n_events, 14)

    # Parameter order: INTRINSIC then EXTRINSIC
    ordered_params = INTRINSIC + EXTRINSIC
    ordered_indices = [PARAMETER_NAMES.index(p) for p in ordered_params]
    ordered_data = [frac_array[:, i] for i in ordered_indices]

    parts = ax.violinplot(ordered_data, positions=range(len(ordered_params)), showmedians=True)

    # Color violin bodies by group
    # violinplot "bodies" is a list of PolyCollection; cast to satisfy mypy
    from collections.abc import Sequence as _Seq
    from typing import cast

    raw_bodies = parts.get("bodies")
    if raw_bodies is not None:
        body_list = cast(_Seq[object], raw_bodies)
        for i, poly in enumerate(body_list):
            color = CYCLE[0] if i < len(INTRINSIC) else CYCLE[1]
            poly.set_facecolor(color)  # type: ignore[attr-defined]
            poly.set_alpha(0.7)  # type: ignore[attr-defined]

    # Separator between intrinsic and extrinsic
    sep_x = len(INTRINSIC) - 0.5
    ax.axvline(sep_x, color=REFERENCE, linestyle="--", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_xticks(range(len(ordered_params)))
    ax.set_xticklabels([LABELS[label_key(p)] for p in ordered_params], rotation=45, ha="right")
    ax.set_ylabel(r"$\sigma_i / |x_i|$")

    return fig, ax


def _plot_bar(
    row: pd.Series,
    param_values: pd.Series,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal bar chart of fractional uncertainties for a single event."""
    from master_thesis_code.plotting._data import reconstruct_covariance

    if ax is None:
        # Taller figure to prevent label overlap with 14 parameters
        fig, ax = get_figure(figsize=(7.0, 6.0))
    else:
        fig = _fig_from_ax(ax)

    cov = reconstruct_covariance(row)
    sigma = np.sqrt(np.diag(cov))
    pv = np.array([float(param_values[p]) for p in PARAMETER_NAMES])
    frac = sigma / np.abs(pv)

    # Parameter order: INTRINSIC then EXTRINSIC
    ordered_params = INTRINSIC + EXTRINSIC
    ordered_indices = [PARAMETER_NAMES.index(p) for p in ordered_params]
    ordered_frac = frac[ordered_indices]

    colors = [CYCLE[0]] * len(INTRINSIC) + [CYCLE[1]] * len(EXTRINSIC)
    y_pos = np.arange(len(ordered_params))

    ax.barh(y_pos, ordered_frac, color=colors, edgecolor=EDGE, linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS[label_key(p)] for p in ordered_params], fontsize="small")
    ax.set_xlabel(r"$\sigma_i / |x_i|$")

    return fig, ax


# ---------------------------------------------------------------------------
# Corner plot
# ---------------------------------------------------------------------------

_DEFAULT_CORNER_PARAMS: list[str] = ["M", "mu", "a", "luminosity_distance", "qS", "phiS"]


def plot_fisher_corner(
    covariance: npt.NDArray[np.float64],
    param_values: npt.NDArray[np.float64],
    params: list[str] | None = None,
    *,
    overlay_events: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] | None = None,
    n_samples: int = 5000,
    seed: int = 42,
    ax: None = None,
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Plot a corner (triangle) plot from Fisher-matrix covariance.

    Generates multivariate Gaussian samples from the covariance matrix
    and delegates to :func:`corner.corner` for the triangle plot.

    Parameters
    ----------
    covariance : npt.NDArray[np.float64]
        14x14 covariance matrix.
    param_values : npt.NDArray[np.float64]
        14-element array of parameter best-fit values.
    params : list[str] | None
        Subset of parameter names to show.  Defaults to
        ``["M", "mu", "a", "luminosity_distance", "qS", "phiS"]``.
    overlay_events : list[tuple[npt.NDArray, npt.NDArray]] | None
        Additional events to overlay, each ``(covariance, param_values)``.
        At most 4 events are shown, each in a distinct color.
    n_samples : int
        Number of samples to draw from the Gaussian.
    seed : int
        Random seed for reproducibility.
    ax : None
        Ignored.  ``corner.corner`` creates its own figure.

    Returns
    -------
    tuple[Figure, npt.NDArray[np.object_]]
        Figure and 2-D array of axes with shape ``(n, n)`` where
        ``n = len(params)``.
    """
    if params is None:
        params = _DEFAULT_CORNER_PARAMS

    # Map param names to indices in the 14-element arrays
    indices = [PARAMETER_NAMES.index(p) for p in params]
    sub_cov = covariance[np.ix_(indices, indices)]
    sub_mean = param_values[indices]

    # Build labels from the label mapping
    labels = [LABELS[label_key(p)] for p in params]

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        sub_mean, sub_cov, size=n_samples, check_valid="warn"
    )

    n = len(params)

    # corner.corner uses tight_layout internally which conflicts with
    # constrained_layout; disable it explicitly
    with matplotlib.rc_context({"figure.constrained_layout.use": False}):
        fig = corner.corner(
            samples,
            labels=labels,
            truths=list(sub_mean),
            truth_color=TRUTH,
            color=CYCLE[0],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
            hist_kwargs={"edgecolor": EDGE},
        )

        if overlay_events is not None:
            for ev_idx, (ev_cov, ev_vals) in enumerate(overlay_events[:4]):
                ev_sub_cov = ev_cov[np.ix_(indices, indices)]
                ev_sub_mean = ev_vals[indices]
                overlay_samples = rng.multivariate_normal(
                    ev_sub_mean, ev_sub_cov, size=n_samples, check_valid="warn"
                )
                corner.corner(
                    overlay_samples,
                    fig=fig,
                    color=CYCLE[(ev_idx + 1) % len(CYCLE)],
                    hist_kwargs={"edgecolor": EDGE},
                )

    axes: npt.NDArray[np.object_] = np.array(fig.axes, dtype=object).reshape(n, n)
    return fig, axes
