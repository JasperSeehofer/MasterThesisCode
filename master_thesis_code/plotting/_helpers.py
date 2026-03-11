"""Shared plotting utilities: figure creation and saving."""

import os
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.image import AxesImage


def get_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure and axes using the OO API.

    When *figsize* is ``None`` the default from the active style sheet is used.
    """
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
    mappable: AxesImage,
    fig: Figure,
    ax: Axes,
    label: str | None = None,
    **kwargs: Any,
) -> Colorbar:
    """Add a colorbar to *ax* for *mappable*."""
    return fig.colorbar(mappable, ax=ax, label=label or "", **kwargs)
