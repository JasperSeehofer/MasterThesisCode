"""Shared plotting utilities: figure creation and saving."""

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

_logger = logging.getLogger(__name__)

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


def compute_hdi_interval(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.683,
) -> tuple[float, float]:
    """Compute the minimal (highest-density) credible interval at *level*.

    For a unimodal posterior this returns the shortest interval enclosing
    *level* of the probability mass — equivalent to the LIGO/Virgo
    "minimal credible interval" reporting convention used for H0 dark
    standard sirens (e.g. ``H0 = 70.0^{+12.0}_{-8.0}`` in Abbott et al.
    2017 for GW170817).  For symmetric posteriors it agrees with
    :func:`compute_credible_interval`; for skewed posteriors the HDI is
    narrower and shifted toward the mode.

    Algorithm: sort grid points by posterior density (descending),
    accumulate trapezoidal mass, and stop at the first density level
    where the enclosed mass crosses *level*.  The HDI is the
    ``[h_min, h_max]`` envelope of all grid points above that level.

    Parameters
    ----------
    h_values:
        Monotonically increasing grid of Hubble-constant values.
    posterior:
        Posterior density evaluated on *h_values* (need not be normalized).
    level:
        Probability mass enclosed by the interval (default 0.683 for the
        1-sigma equivalent — matches LIGO HDI convention).

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` bounds of the highest-density interval.  Returns
        ``(nan, nan)`` when *posterior* integrates to zero or less.
    """
    norm = np.trapezoid(posterior, h_values)
    if norm <= 0:
        return (float("nan"), float("nan"))

    p = posterior / norm

    # Trapezoidal cell mass for each grid point: half of each adjacent edge
    dh = np.zeros_like(h_values)
    if len(h_values) > 1:
        diffs = np.diff(h_values)
        dh[0] = diffs[0] / 2.0
        dh[-1] = diffs[-1] / 2.0
        dh[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    cell_mass = p * dh

    # Sort by density descending and accumulate mass
    order = np.argsort(-p)
    cum = np.cumsum(cell_mass[order])
    # Find smallest k such that cum[k] >= level
    k_arr = np.searchsorted(cum, level, side="left")
    k = int(k_arr)
    if k >= len(order):
        k = len(order) - 1

    selected = order[: k + 1]
    lo = float(h_values[selected].min())
    hi = float(h_values[selected].max())
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


# ---------------------------------------------------------------------------
# Heatmap norms (HORIZON v2.3 Phase 2): explicit norms on every map.
# ---------------------------------------------------------------------------


def diverging_norm(
    vcenter: float = 0.73,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> TwoSlopeNorm:
    """Build a two-slope diverging norm centered on *vcenter*.

    Canonical bias-map recipe (pairs with ``_colors.DIVERGING_CMAP``)::

        from master_thesis_code.plotting._colors import DIVERGING_CMAP, NO_DATA
        cmap = plt.get_cmap(DIVERGING_CMAP).copy()
        cmap.set_bad(NO_DATA)
        ax.pcolormesh(X, Y, residual, cmap=cmap, norm=diverging_norm(0.73))
        # redundant grayscale-safe channel: a single iso-line at vcenter
        ax.contour(X, Y, residual, levels=[0.73], colors=EDGE, linewidths=0.8)

    so the H0 truth (0.73) maps to the colormap midpoint (white for ``RdBu_r``)
    and over-/under-estimates read as the two diverging hues.  No bias/residual
    heatmap exists in the package yet, so this is the reusable convention for the
    Phase-5 consumer (proposal §6b).  UPGRADE PATH: ``cmcrameri`` ``vik`` is the
    preferred perceptually uniform diverging map -- ``uv add cmcrameri`` and swap
    ``DIVERGING_CMAP`` when a real bias map is built.

    Parameters
    ----------
    vcenter:
        Value mapped to the colormap midpoint (default 0.73, the H0 truth).
    vmin, vmax:
        Optional lower / upper data bounds.  ``TwoSlopeNorm`` requires
        ``vmin < vcenter < vmax``; when the supplied bounds would violate that
        (e.g. an all-positive-bias residual whose ``vmin`` exceeds 0.73) the
        center is nudged to the midpoint of the bounds so the norm never raises
        ``ValueError``.

    Returns
    -------
    matplotlib.colors.TwoSlopeNorm
        Diverging norm with the (possibly nudged) center inside ``(vmin, vmax)``.
    """
    center = vcenter
    if vmin is not None and vmax is not None and not (vmin < center < vmax):
        # Out-of-range center would crash TwoSlopeNorm -- nudge to the midpoint.
        center = 0.5 * (vmin + vmax)
    return TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)


def make_heatmap_norm(
    data: npt.NDArray[np.float64],
    *,
    mode: Literal["robust", "log"] = "robust",
    low: float = 2.0,
    high: float = 98.0,
) -> Normalize:
    """Build an explicit heatmap norm, robust to NaN / non-positive values.

    Replaces matplotlib's silent linear autoscale-from-zero, which washes out
    dynamic range when a few extreme cells (or counts spanning orders of
    magnitude) dominate the range.

    - ``mode="robust"``: a linear :class:`~matplotlib.colors.Normalize` clipped
      to the *low*/*high* percentiles of the **finite** data (NaN ignored), so
      outliers do not flatten the bulk of the map.
    - ``mode="log"``: a :class:`~matplotlib.colors.LogNorm` computed over the
      **strictly-positive finite** values only.  Zeros, negatives and NaN are
      masked before the min/max are taken so ``LogNorm`` never receives ``<= 0``
      (which would raise / produce an invalid mappable).  When no positive value
      remains, a safe ``(1, 10)`` range is used.

    Pair the returned norm with ``cmap.set_bad(NO_DATA)`` so the masked
    (no-data) cells render as neutral gray rather than the lowest color.

    Parameters
    ----------
    data:
        Array of heatmap values (may contain NaN / zeros / negatives).
    mode:
        ``"robust"`` (percentile-clipped linear) or ``"log"``.
    low, high:
        Percentile bounds for ``mode="robust"`` (default 2 / 98).

    Returns
    -------
    matplotlib.colors.Normalize
        ``Normalize`` for ``"robust"`` or ``LogNorm`` for ``"log"``.
    """
    finite = np.asarray(data, dtype=np.float64)
    finite = finite[np.isfinite(finite)]

    if mode == "log":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            return LogNorm(vmin=1.0, vmax=10.0)
        vmin = float(positive.min())
        vmax = float(positive.max())
        if vmax <= vmin:
            vmax = vmin * 10.0
        return LogNorm(vmin=vmin, vmax=vmax)

    # mode == "robust"
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return Normalize(vmin=vmin, vmax=vmax)


# ---------------------------------------------------------------------------
# Canonical combined posterior loader (Phase A)
# ---------------------------------------------------------------------------

CANONICAL_CACHE_FILENAME = "canonical_combined.json"


def load_canonical_combined_posterior(
    data_dir: Path,
    variant: str,
    *,
    refresh: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict[str, Any]]:
    """Return ``(h_grid, combined_posterior, metadata)`` for plotting.

    Uses the raw ``Σ log L_i`` combination from
    :mod:`master_thesis_code.bayesian_inference.posterior_combination`
    (see ``compute_canonical_combined_posterior``). This is the paper-grade
    canonical reference used by the bias-investigation suite and quoted in
    ``docs/H0_BIAS_RESOLUTION.md`` — every H0-posterior figure must consume
    this loader so all figures agree on the MAP.

    Parameters
    ----------
    data_dir:
        Directory holding ``posteriors/`` and ``posteriors_with_bh_mass/``
        subdirectories of per-h JSON files.
    variant:
        ``"posteriors"`` (1D channel) or ``"posteriors_with_bh_mass"`` (2D).
    refresh:
        If True, ignore any cached ``canonical_combined.json`` and recompute.

    Returns
    -------
    h_grid:
        Float64 array of sorted h-values.
    posterior:
        Peak-normalised linear posterior on ``h_grid``.
    metadata:
        Dict with keys ``n_events_used``, ``discrete_map``,
        ``continuous_map``, ``strategy``, plus ``log_posterior``
        (un-normalised Σ log L_i).
    """
    posteriors_dir = data_dir / variant
    cache_path = posteriors_dir / CANONICAL_CACHE_FILENAME

    if not refresh and cache_path.is_file():
        with open(cache_path) as f:
            cached = json.load(f)
        h_grid = np.asarray(cached["h_values"], dtype=np.float64)
        posterior = np.asarray(cached["posterior"], dtype=np.float64)
        meta = {
            "n_events_used": int(cached["n_events_used"]),
            "discrete_map": float(cached["discrete_map"]),
            "continuous_map": float(cached["continuous_map"]),
            "strategy": str(cached["strategy"]),
            "log_posterior": np.asarray(cached["log_posterior"], dtype=np.float64),
        }
        return h_grid, posterior, meta

    if not posteriors_dir.is_dir():
        raise FileNotFoundError(f"Posteriors directory not found: {posteriors_dir}")

    # Local import to avoid a hard dependency from the plotting top-level
    # module on the bayesian-inference layer (matplotlib-only environments
    # used by interactive notebooks should still import _helpers cleanly).
    from master_thesis_code.bayesian_inference.posterior_combination import (
        compute_canonical_combined_posterior,
    )

    result = compute_canonical_combined_posterior(posteriors_dir)
    if not result["h_values"]:
        raise FileNotFoundError(
            f"No usable h_*.json files in {posteriors_dir}; cannot build canonical posterior."
        )

    # Persist for fast reload next time. The JSON is small (<10 kB per variant).
    try:
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        _logger.warning("Could not write canonical-posterior cache %s: %s", cache_path, e)

    h_grid = np.asarray(result["h_values"], dtype=np.float64)
    posterior = np.asarray(result["posterior"], dtype=np.float64)
    n_used_raw = result["n_events_used"]
    discrete_raw = result["discrete_map"]
    continuous_raw = result["continuous_map"]
    assert isinstance(n_used_raw, int)
    assert isinstance(discrete_raw, float)
    assert isinstance(continuous_raw, float)
    meta = {
        "n_events_used": n_used_raw,
        "discrete_map": discrete_raw,
        "continuous_map": continuous_raw,
        "strategy": str(result["strategy"]),
        "log_posterior": np.asarray(result["log_posterior"], dtype=np.float64),
    }
    return h_grid, posterior, meta
