"""Shared plotting utilities: figure creation and saving."""

import os
from collections.abc import Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

# REVTeX two-column figure width presets (inches)
_PRESETS: dict[str, tuple[float, float]] = {
    "single": (3.375, 3.375 / 1.618),  # ~3.375 x 2.086
    "double": (7.0, 7.0 / 1.618),  # ~7.0 x 4.327
}


def compute_credible_interval(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.68,
) -> tuple[float, float]:
    """Compute the central credible interval at *level* using trapezoidal CDF.

    Shared CI utility (per D-07 from phase 35 CONTEXT.md) used by both
    ``convergence_plots.py`` and ``paper_figures.py`` to ensure a consistent
    trapezoidal CDF everywhere (PFIG-03).

    Parameters
    ----------
    h_values:
        Monotonically increasing grid of Hubble-constant values.
    posterior:
        Posterior density evaluated on *h_values* (need not be normalized).
    level:
        Probability mass enclosed by the interval (default 0.68 for 68%).

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` bounds of the central credible interval.  Returns
        ``(nan, nan)`` when *posterior* integrates to zero or less.
    """
    norm = np.trapezoid(posterior, h_values)
    if norm <= 0:
        return (float("nan"), float("nan"))

    p = posterior / norm

    # Build CDF by accumulating per-step trapezoid areas
    cdf = np.zeros(len(h_values), dtype=np.float64)
    for i in range(1, len(h_values)):
        cdf[i] = cdf[i - 1] + np.trapezoid(p[i - 1 : i + 1], h_values[i - 1 : i + 1])

    # Normalize so CDF ends at exactly 1.0
    cdf /= cdf[-1]

    lo = float(np.interp((1.0 - level) / 2.0, cdf, h_values))
    hi = float(np.interp((1.0 + level) / 2.0, cdf, h_values))
    return (lo, hi)


def _fig_from_ax(ax: Axes) -> Figure:
    """Extract Figure from an Axes, asserting it is not None."""
    fig = ax.get_figure()
    assert isinstance(fig, Figure)
    return fig


def get_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    preset: Literal["single", "double"] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure and axes using the OO API.

    Parameters
    ----------
    nrows, ncols:
        Subplot grid dimensions.
    figsize:
        Explicit (width, height) in inches.  Overrides *preset*.
    preset:
        Named size preset: ``"single"`` (~3.375in, REVTeX single column)
        or ``"double"`` (~7.0in, REVTeX double column).  Ignored when
        *figsize* is given.  When neither is given, the active style
        sheet default is used.
    **kwargs:
        Forwarded to :func:`matplotlib.pyplot.subplots`.
    """
    if figsize is None and preset is not None:
        figsize = _PRESETS[preset]
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, ax


def save_figure(
    fig: Figure,
    path: str,
    *,
    formats: Sequence[str] = ("pdf",),
    dpi: int = 300,
    close: bool = True,
) -> None:
    """Save *fig* to *path*, creating parent directories as needed.

    Parameters
    ----------
    fig:
        The figure to save.
    path:
        Output path **without** extension.  The extension is appended from
        *formats*.
    formats:
        One or more file extensions (e.g. ``("pdf", "png")``).
    dpi:
        Resolution for raster formats.
    close:
        If ``True`` (default), close the figure after saving to free memory.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{path}.{fmt}", dpi=dpi)
    if close:
        plt.close(fig)


def make_colorbar(
    mappable: ScalarMappable,
    fig: Figure,
    ax: Axes,
    label: str | None = None,
    **kwargs: Any,
) -> Colorbar:
    """Add a colorbar to *ax* for *mappable*."""
    return fig.colorbar(mappable, ax=ax, label=label or "", **kwargs)
