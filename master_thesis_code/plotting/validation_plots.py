"""Validation / contextual factory functions for the EMRI dark-siren H0 work.

Two "Observatory + Atlas" figures:

* :func:`plot_h0_forest` -- NF-1, a Di-Valentino-style horizontal whisker
  ("forest") plot placing this work's H0 measurement among literature values,
  grouped into early/indirect vs late/direct probes, with the same Planck-pink
  and SH0ES-cyan vertical bands as ``fig01`` (``plot_combined_posterior``).
* :func:`plot_pp_coverage` -- NF-2, a P-P / coverage plot: per-parameter ECDF
  of the standardized-residual percentile rank versus the diagonal, with a grey
  1/2/3-sigma confidence band from the binomial coverage of an ideal P-P curve.

Both obey the factory contract: ``plot_*(..., *, ax=None) -> (fig, ax)``.
Neither calls ``plt.show()`` or ``plt.savefig()`` -- the caller decides where to
save.

Data note
---------
The locally-available Cramer-Rao-bounds CSV stores the *injected* (true)
parameter values plus the Fisher/covariance matrix (diagonal
``delta_X_delta_X`` entries are the per-parameter variances ``sigma_X^2``), but
it does **not** contain a *recovered* (noisy-realization) estimate for each
event.  A true probability-integral-transform P-P test therefore cannot be
built from this CSV alone -- there is no ``(true, recovered)`` pair to form a
percentile rank.  :func:`plot_pp_coverage` is consequently written to accept
``(true, recovered, sigma)`` arrays directly so it can consume a future
injection-recovery campaign; the local verification draws a synthetic Gaussian
realization to exercise the rendering path.
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy import stats

from master_thesis_code.plotting._colors import (
    DIVERGING_CMAP,
    EDGE,
    METHOD,
    PLANCK_BAND,
    PRIOR,
    SHOES_BAND,
)
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
from master_thesis_code.plotting._labels import LABELS


def _resolve_cmap(name: str) -> Colormap:
    """Resolve a palette colormap token to a registered ``Colormap`` object.

    The Atlas tokens in ``_colors`` use bare ``cmcrameri`` names (e.g.
    ``"vik"``), but ``cmcrameri`` registers them under a ``cmc.`` prefix.  Try
    the prefixed name first, then the bare name (covers the built-in fallback
    such as ``"RdBu"`` when ``cmcrameri`` is absent).
    """
    for candidate in (f"cmc.{name}", name):
        try:
            return plt.get_cmap(candidate)
        except (KeyError, ValueError):
            continue
    return plt.get_cmap(name)


def _as_float(value: object) -> float:
    """Narrow a measurement-dict ``object`` value to ``float``.

    The measurement dicts are typed ``dict[str, object]`` because they mix a
    ``str`` name with numeric fields; this helper performs the cast at the use
    site so mypy is satisfied while keeping the heterogeneous public schema.
    """
    if isinstance(value, (int, float)):
        return float(value)
    msg = f"expected a numeric value, got {type(value).__name__}: {value!r}"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# NF-1: H0 forest / whisker plot
# ---------------------------------------------------------------------------

# Literature + this-work H0 measurements, in km/s/Mpc.  ``group`` is one of
# {"early", "late"} -- early/indirect (CMB, BBN, sound horizon) vs late/direct
# (distance ladder, sirens).  ``lo``/``hi`` are the *one-sided* 68% CL widths
# (positive numbers); the whisker spans ``[H0 - lo, H0 + hi]``.
#
# References:
#   Planck 2018  : Planck Collab. (2020), A&A 641, A6, arXiv:1807.06209 (TT,TE,EE+lowE+lensing).
#   SH0ES (R22)  : Riess et al. (2022), ApJL 934, L7, arXiv:2112.04510.
#   GWTC-3 dark  : Abbott et al. (2023), ApJ 949, 76, arXiv:2111.03604 (dark-siren + GLADE+).
#   This work    : seed-500 phase-50 EMRI dark-siren combined posterior MAP + 68% HDI.
DEFAULT_H0_MEASUREMENTS: list[dict[str, object]] = [
    {"name": "Planck 2018", "H0": 67.36, "lo": 0.54, "hi": 0.54, "group": "early"},
    {"name": "SH0ES (Riess+ 2022)", "H0": 73.04, "lo": 1.04, "hi": 1.04, "group": "late"},
    {"name": "GWTC-3 dark siren", "H0": 68.0, "lo": 6.0, "hi": 8.0, "group": "late"},
    {
        "name": "This work (EMRI dark siren)",
        "H0": 73.7,
        "lo": 0.4,
        "hi": 0.5,
        "group": "late",
    },
]

# Planck / SH0ES band extents in km/s/Mpc (H0 = 100 h), matching fig01's
# axvspan(0.669, 0.679) and axvspan(0.720, 0.740).
_PLANCK_BAND_H0: tuple[float, float] = (66.9, 67.9)
_SHOES_BAND_H0: tuple[float, float] = (72.0, 74.0)

_GROUP_LABELS: dict[str, str] = {
    "early": "Early / indirect",
    "late": "Late / direct",
}


def plot_h0_forest(
    measurements: Sequence[dict[str, object]] = DEFAULT_H0_MEASUREMENTS,
    *,
    highlight: str = "This work (EMRI dark siren)",
    show_bands: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal whisker ("forest") plot of H0 measurements in context.

    One row per measurement, sorted into early/indirect and late/direct groups
    (groups stacked vertically with a faint separator and a group label).  The
    point marker is the central H0; the asymmetric error cap spans the 68% CL
    ``[H0 - lo, H0 + hi]``.  The Planck-pink and SH0ES-cyan vertical bands are
    the same as on the H0 posterior (``fig01``), so the two figures share one
    visual grammar.  The *highlight* row (this work) is emphasised with the
    dark-siren blue ``METHOD["dark"]`` and a heavier marker.

    Parameters
    ----------
    measurements:
        Sequence of dicts, each with keys ``name`` (str), ``H0`` (float),
        ``lo`` (float, lower 68% half-width, positive), ``hi`` (float, upper
        68% half-width, positive) and ``group`` (``"early"`` or ``"late"``).
        Defaults to :data:`DEFAULT_H0_MEASUREMENTS`.
    highlight:
        ``name`` of the row to emphasise (this work).  Pass an empty string to
        disable highlighting.
    show_bands:
        Draw the Planck and SH0ES reference bands behind the rows.
    ax:
        Optional pre-existing Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and the Axes the forest was drawn on.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    # Order rows: early group first (top), then late group; preserve input
    # order within each group.  Rows are laid out top-to-bottom, so we assign
    # descending y-positions.
    early = [m for m in measurements if str(m.get("group", "late")) == "early"]
    late = [m for m in measurements if str(m.get("group", "late")) != "early"]
    ordered: list[dict[str, object]] = [*early, *late]
    n = len(ordered)

    # --- Reference bands (Planck pink, SH0ES cyan), behind everything ---
    if show_bands:
        ax.axvspan(*_PLANCK_BAND_H0, color=PLANCK_BAND, alpha=0.18, lw=0, zorder=0)
        ax.axvspan(*_SHOES_BAND_H0, color=SHOES_BAND, alpha=0.18, lw=0, zorder=0)

    y_positions: list[float] = []
    y_labels: list[str] = []
    for i, m in enumerate(ordered):
        # Top row gets the largest y so reading order is top-to-bottom.
        y = float(n - 1 - i)
        y_positions.append(y)
        name = str(m["name"])
        y_labels.append(name)

        h0 = _as_float(m["H0"])
        lo = _as_float(m["lo"])
        hi = _as_float(m["hi"])

        is_highlight = name == highlight
        color = METHOD["dark"] if is_highlight else EDGE
        marker_size = 5.0 if is_highlight else 3.5
        cap_lw = 1.8 if is_highlight else 1.1

        ax.errorbar(
            h0,
            y,
            xerr=[[lo], [hi]],
            fmt="o",
            color=color,
            markersize=marker_size,
            markeredgecolor=EDGE,
            markeredgewidth=0.5,
            elinewidth=cap_lw,
            capsize=2.5,
            capthick=cap_lw,
            zorder=5 if is_highlight else 4,
        )

    # --- Group separator + labels ---
    if early and late:
        # Separator sits between the last-early row and the first-late row.
        sep_y = float(n - 1 - len(early)) + 0.5
        ax.axhline(sep_y, color=PRIOR, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)

    # Annotate each group block just above its top row, in the upper plot
    # margin so the italic group label never collides with a row marker.
    if early:
        ax.annotate(
            _GROUP_LABELS["early"],
            xy=(0.02, float(n - 1) + 0.42),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="bottom",
            fontsize=6,
            color=PRIOR,
            fontstyle="italic",
        )
    if late:
        ax.annotate(
            _GROUP_LABELS["late"],
            xy=(0.02, float(n - 1 - len(early)) + 0.42),
            xycoords=("axes fraction", "data"),
            ha="left",
            va="bottom",
            fontsize=6,
            color=PRIOR,
            fontstyle="italic",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=6)
    # Extra top headroom so the italic group label above the top row is visible.
    ax.set_ylim(-0.6, n - 1 + 0.85)
    ax.set_xlabel(LABELS["H0"])

    # Band swatch legend (matches fig01's Planck/SH0ES grammar); placed below
    # the axes so it never overlaps a row marker.
    handles = [
        Line2D([0], [0], color=PLANCK_BAND, lw=6, alpha=0.5, label="Planck 2018"),
        Line2D([0], [0], color=SHOES_BAND, lw=6, alpha=0.5, label="SH0ES"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=6,
        framealpha=0.85,
    )

    return fig, ax


# ---------------------------------------------------------------------------
# NF-2: P-P / coverage plot
# ---------------------------------------------------------------------------


def _binomial_pp_band(
    n_events: int,
    grid: npt.NDArray[np.float64],
    n_sigma: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pointwise ``n_sigma`` confidence band for an ideal P-P curve.

    For ``n_events`` independent draws that are uniform under the null, the
    empirical CDF at credibility level ``x`` has mean ``x`` and standard error
    ``sqrt(x (1 - x) / n_events)`` (binomial).  This returns the lower/upper
    envelopes ``x +/- n_sigma * se`` clipped to ``[0, 1]``.

    Parameters
    ----------
    n_events:
        Number of (true, recovered) pairs entering the P-P test.
    grid:
        Credibility-level grid in ``[0, 1]`` (the P-P x-axis).
    n_sigma:
        Band half-width in standard deviations.

    Returns
    -------
    tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ``(lower, upper)`` envelopes on *grid*.
    """
    se = np.sqrt(np.clip(grid * (1.0 - grid) / max(n_events, 1), 0.0, None))
    lower = np.clip(grid - n_sigma * se, 0.0, 1.0)
    upper = np.clip(grid + n_sigma * se, 0.0, 1.0)
    return lower.astype(np.float64), upper.astype(np.float64)


def plot_pp_coverage(
    true_values: dict[str, npt.NDArray[np.float64]],
    recovered_values: dict[str, npt.NDArray[np.float64]],
    sigmas: dict[str, npt.NDArray[np.float64]],
    *,
    param_labels: dict[str, str] | None = None,
    show_sigma_band: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Probability-integral-transform P-P / coverage plot, per parameter.

    For each parameter, the standardized residual
    ``z = (recovered - true) / sigma`` is converted to a percentile rank via
    the standard-normal CDF ``Phi(z)`` (the probability-integral transform under
    the Gaussian Cramer-Rao / Fisher approximation).  Under a correctly
    calibrated pipeline these ranks are uniform on ``[0, 1]``, so the empirical
    CDF of the ranks should track the diagonal.  Each parameter's ECDF is drawn
    as a step curve coloured along :data:`DIVERGING_CMAP` (signed-deviation
    encoding); systematic departures above/below the diagonal read as a
    consistent colour drift.  A grey 1/2/3-sigma binomial band around the
    diagonal flags significant miscalibration.

    Parameters
    ----------
    true_values:
        Mapping ``param_name -> injected values`` (1-D float array per param).
    recovered_values:
        Mapping ``param_name -> recovered point estimates`` (same keys/shapes).
    sigmas:
        Mapping ``param_name -> 1-sigma uncertainties`` (e.g. the sqrt of the
        Cramer-Rao diagonal; same keys/shapes).
    param_labels:
        Optional pretty axis labels per parameter; falls back to
        :data:`LABELS` then the raw key.
    show_sigma_band:
        Draw the grey 1/2/3-sigma binomial coverage band around the diagonal.
    ax:
        Optional pre-existing Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes]
        The figure and the Axes the P-P curves were drawn on.

    Notes
    -----
    The keys of *true_values* define the parameters and plotting order; every
    key must also be present in *recovered_values* and *sigmas* with the same
    length.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    params = list(true_values.keys())
    if not params:
        msg = "true_values is empty; nothing to plot"
        raise ValueError(msg)

    # Largest sample size across params drives the binomial band width.
    n_max = 0
    for p in params:
        if p not in recovered_values or p not in sigmas:
            msg = f"parameter {p!r} missing from recovered_values or sigmas"
            raise ValueError(msg)
        n_max = max(n_max, int(np.asarray(true_values[p]).size))

    grid = np.linspace(0.0, 1.0, 256, dtype=np.float64)

    # --- Grey 1/2/3-sigma binomial coverage band (drawn behind curves) ---
    if show_sigma_band and n_max > 0:
        for n_sigma, alpha in ((3.0, 0.10), (2.0, 0.14), (1.0, 0.20)):
            lower, upper = _binomial_pp_band(n_max, grid, n_sigma)
            ax.fill_between(grid, lower, upper, color=PRIOR, alpha=alpha, lw=0, zorder=0)

    # Diagonal (perfect calibration).
    ax.plot([0.0, 1.0], [0.0, 1.0], color=EDGE, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)

    cmap = _resolve_cmap(DIVERGING_CMAP)
    denom = max(len(params) - 1, 1)

    for i, p in enumerate(params):
        true_arr = np.asarray(true_values[p], dtype=np.float64).ravel()
        rec_arr = np.asarray(recovered_values[p], dtype=np.float64).ravel()
        sig_arr = np.asarray(sigmas[p], dtype=np.float64).ravel()

        # Guard against zero / non-finite sigmas.
        good = np.isfinite(true_arr) & np.isfinite(rec_arr) & np.isfinite(sig_arr) & (sig_arr > 0)
        if not np.any(good):
            continue

        z = (rec_arr[good] - true_arr[good]) / sig_arr[good]
        ranks = stats.norm.cdf(z).astype(np.float64)
        ranks_sorted = np.sort(ranks)
        n_good = ranks_sorted.size
        # Empirical CDF evaluated at each rank (step on the right).
        ecdf = np.arange(1, n_good + 1, dtype=np.float64) / n_good

        color = cmap(i / denom)
        label = (param_labels or {}).get(p) or LABELS.get(p, p)
        ax.step(
            ranks_sorted,
            ecdf,
            where="post",
            color=color,
            linewidth=1.2,
            label=label,
            zorder=4,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Credibility level (rank)")
    ax.set_ylabel("Empirical CDF")
    ax.legend(loc="upper left", fontsize=5, framealpha=0.85, ncol=2)

    return fig, ax
