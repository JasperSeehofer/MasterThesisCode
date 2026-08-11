"""H0 convergence diagnostics and detection efficiency curves.

Factory functions for two key thesis diagnostic plots:

- **H0 convergence** (single-panel): credible-interval width vs number of
  events, with a 1/sqrt(N) statistical guide and horizontal Planck/SH0ES
  target-precision reference bands.  The two pipeline variants share one
  hue and are separated by linestyle (``VARIANT_STYLE``).  The posterior
  panel that used to sit on the left was dropped (redundant with fig01).
- **Detection efficiency**: binned detection fraction with Wilson score
  confidence intervals.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from astropy.stats import binom_conf_interval
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from darksiren_emri.plotting._colors import (
    CYCLE,
    PLANCK_BAND,
    PRIOR,
    SHOES_BAND,
    VARIANT_STYLE,
)
from darksiren_emri.plotting._helpers import _fig_from_ax, compute_credible_interval, get_figure
from darksiren_emri.plotting._labels import LABELS

if TYPE_CHECKING:
    # Type-only import avoids a circular dep with convergence_analysis at runtime.
    from darksiren_emri.plotting.convergence_analysis import ImprovementBank

# Default subset sizes for convergence analysis
_DEFAULT_SUBSETS: list[int] = [1, 5, 10, 25, 50, 100]

# Target H0-precision bands, expressed as a credible-interval WIDTH in
# dimensionless h.  These mirror the full-height tension bands on the
# posterior figures (fig01): Planck 0.669-0.679 (width 0.010) and SH0ES
# 0.720-0.740 (width 0.020).  Drawn here as horizontal reference bands so
# the convergence curve can be read against "how many events to reach
# Planck/SH0ES-level precision".
_PLANCK_TARGET_WIDTH: float = 0.010
_SHOES_TARGET_WIDTH: float = 0.020


def _convergence_ci_widths(
    h_values: npt.NDArray[np.float64],
    posteriors_list: list[npt.NDArray[np.float64]],
    sizes: list[int],
    rng: np.random.Generator,
    level: float,
) -> list[float]:
    """Compute CI widths for random subsets of increasing size."""
    n_events = len(posteriors_list)
    ci_widths: list[float] = []
    for n in sizes:
        indices = rng.choice(n_events, size=n, replace=False)
        log_posteriors = [np.log(np.maximum(posteriors_list[i], 1e-300)) for i in indices]
        log_combined = np.sum(log_posteriors, axis=0)
        log_combined -= log_combined.max()
        combined = np.exp(log_combined)
        norm = np.trapezoid(combined, h_values)
        if norm > 0:
            combined /= norm
        lo, hi = compute_credible_interval(h_values, combined, level=level)
        ci_widths.append(hi - lo)
    return ci_widths


def plot_h0_convergence(
    h_values: npt.NDArray[np.float64],
    event_posteriors: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    *,
    true_h: float | None = None,  # noqa: ARG001 — accepted for call-site compat (was left-panel truth)
    subset_sizes: list[int] | None = None,
    seed: int = 42,
    level: float = 0.68,
    h_values_alt: npt.NDArray[np.float64] | None = None,
    event_posteriors_alt: list[npt.NDArray[np.float64]] | None = None,
    label: str = r"Without $M_z$",
    label_alt: str = r"With $M_z$",
    color: str | None = None,  # noqa: ARG001 — variants now share one hue (VARIANT_STYLE)
    color_alt: str | None = None,  # noqa: ARG001 — kept for call-site compatibility
    bootstrap_bank: "ImprovementBank | None" = None,
    canonical_no_mass: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,  # noqa: ARG001 — left posterior panel dropped (redundant with fig01)
    canonical_with_mass: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,  # noqa: ARG001 — left posterior panel dropped (redundant with fig01)
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Single-panel H0 convergence plot: credible-interval width vs N.

    The redundant posterior panel (formerly the left subplot, duplicating
    fig01) has been dropped.  This factory now produces a single
    credible-interval-width-versus-event-count curve in the
    "Observatory + Atlas" grammar:

    * The two pipeline variants share one hue (the dark-siren blue) and are
      separated only by linestyle (``VARIANT_STYLE``: ``no_mass`` solid,
      ``with_mass`` dashed) -- colorblind- and greyscale-safe.
    * A ``1/sqrt(N)`` statistical guide, anchored to the first point of the
      primary variant, shows the ideal Poisson tightening.
    * Two horizontal target-precision reference bands mark the Planck and
      SH0ES credible-interval widths, so the reader can see at a glance how
      many events are needed to reach each measurement's precision.

    Parameters
    ----------
    h_values:
        Grid of Hubble-constant values (shared x-axis for posteriors).
    event_posteriors:
        Per-event posterior arrays evaluated on *h_values*.
    true_h:
        Ignored.  Retained for call-site compatibility (was the left-panel
        truth line, which lived on the now-removed posterior panel).
    subset_sizes:
        Number of events in each subset.  Capped at ``len(event_posteriors)``.
    seed:
        RNG seed for reproducible random sub-sampling.
    level:
        Credible-interval probability mass (default 68%).
    h_values_alt:
        H-grid for the alternative (with-mass) variant.
    event_posteriors_alt:
        Per-event posteriors for the alternative variant.
    label:
        Legend label for the primary (without-mass) variant.
    label_alt:
        Legend label for the alternative (with-mass) variant.
    color:
        Ignored.  Variants now share one hue and differ by linestyle.
    color_alt:
        Ignored.  Retained for call-site compatibility.
    bootstrap_bank:
        Optional :class:`ImprovementBank` from
        :func:`compute_m_z_improvement_bank`.  When provided, the curve
        draws a 16/84 percentile band around the CI-width line, per
        variant.  Default ``None`` draws no band.
    canonical_no_mass, canonical_with_mass:
        Ignored.  Retained for call-site compatibility (fed the removed
        posterior panel).
    ax:
        Optional pre-existing Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and the single credible-interval-width Axes.
    """
    primary_color, primary_ls = VARIANT_STYLE["no_mass"]
    alt_color, alt_ls = VARIANT_STYLE["with_mass"]

    posteriors_list: list[npt.NDArray[np.float64]] = list(event_posteriors)
    n_events = len(posteriors_list)

    # Resolve subset sizes, cap at available events
    if subset_sizes is None:
        sizes = [s for s in _DEFAULT_SUBSETS if s <= n_events]
        if not sizes:
            sizes = [n_events]
    else:
        sizes = [min(s, n_events) for s in subset_sizes]

    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    rng = np.random.default_rng(seed)

    # --- Primary variant (without M_z) ---
    ci_widths = _convergence_ci_widths(h_values, posteriors_list, sizes, rng, level)
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    ci_arr = np.asarray(ci_widths, dtype=np.float64)
    ax.plot(
        sizes_arr,
        ci_arr,
        marker="o",
        linestyle=primary_ls,
        color=primary_color,
        markersize=4,
        linewidth=1.4,
        label=label,
        zorder=4,
    )

    # --- Alternative variant (with M_z), if provided ---
    if event_posteriors_alt is not None:
        h_alt = h_values_alt if h_values_alt is not None else h_values
        posteriors_alt_list: list[npt.NDArray[np.float64]] = list(event_posteriors_alt)
        n_alt = len(posteriors_alt_list)
        sizes_alt = [min(s, n_alt) for s in sizes]

        rng_alt = np.random.default_rng(seed)
        ci_widths_alt = _convergence_ci_widths(
            h_alt, posteriors_alt_list, sizes_alt, rng_alt, level
        )
        sizes_alt_arr = np.asarray(sizes_alt, dtype=np.float64)
        ci_alt_arr = np.asarray(ci_widths_alt, dtype=np.float64)
        ax.plot(
            sizes_alt_arr,
            ci_alt_arr,
            marker="s",
            linestyle=alt_ls,
            color=alt_color,
            markersize=4,
            linewidth=1.4,
            label=label_alt,
            zorder=4,
        )

    # --- Optional bootstrap 16/84 percentile band (VIZ-02) ---
    if bootstrap_bank is not None:
        b_sizes = np.asarray(bootstrap_bank.sizes, dtype=np.float64)
        # Primary variant (no mass)
        w_no_lo = np.asarray(bootstrap_bank.metrics_no_mass["hdi68_width"]["p16"], dtype=np.float64)
        w_no_hi = np.asarray(bootstrap_bank.metrics_no_mass["hdi68_width"]["p84"], dtype=np.float64)
        ax.fill_between(b_sizes, w_no_lo, w_no_hi, color=primary_color, alpha=0.18, lw=0, zorder=2)
        # Alt variant (with mass) — only if alt posteriors were provided
        if event_posteriors_alt is not None:
            w_with_lo = np.asarray(
                bootstrap_bank.metrics_with_mass["hdi68_width"]["p16"], dtype=np.float64
            )
            w_with_hi = np.asarray(
                bootstrap_bank.metrics_with_mass["hdi68_width"]["p84"], dtype=np.float64
            )
            ax.fill_between(
                b_sizes, w_with_lo, w_with_hi, color=alt_color, alpha=0.18, lw=0, zorder=2
            )

    # --- 1/sqrt(N) statistical guide, anchored to the primary first point ---
    if len(sizes) > 1 and ci_widths[0] > 0:
        ref = ci_widths[0] * np.sqrt(sizes_arr[0]) / np.sqrt(sizes_arr)
        ax.plot(
            sizes_arr,
            ref,
            linestyle=(0, (1, 2)),
            color=PRIOR,
            linewidth=1.0,
            alpha=0.9,
            zorder=1,
            label=r"$\propto 1/\sqrt{N}$",
        )

    # --- Horizontal target-precision reference bands (Planck / SH0ES) ---
    # Drawn behind the curves as thin horizontal bands at the target CI width.
    _bandwidth = 0.0008  # visual half-thickness of the horizontal swatch in h-width units
    ax.axhspan(
        _PLANCK_TARGET_WIDTH - _bandwidth,
        _PLANCK_TARGET_WIDTH + _bandwidth,
        color=PLANCK_BAND,
        alpha=0.30,
        lw=0,
        zorder=0,
        label="Planck precision",
    )
    ax.axhspan(
        _SHOES_TARGET_WIDTH - _bandwidth,
        _SHOES_TARGET_WIDTH + _bandwidth,
        color=SHOES_BAND,
        alpha=0.30,
        lw=0,
        zorder=0,
        label="SH0ES precision",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of events $N_\mathrm{det}$")
    ax.set_ylabel(rf"{int(level * 100)}\% CI width of {LABELS['h']}")
    ax.legend(loc="upper right", fontsize=6)

    return fig, ax


def plot_detection_efficiency(
    variable: npt.NDArray[np.float64],
    detected: npt.NDArray[np.bool_],
    *,
    bins: int = 20,
    confidence: float = 0.68,
    xlabel: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Detection efficiency curve with Wilson score confidence intervals.

    Parameters
    ----------
    variable:
        Independent variable (e.g. redshift) for each injection.
    detected:
        Boolean mask — ``True`` for detected injections.
    bins:
        Number of equal-width bins.
    confidence:
        Confidence level for Wilson score interval (default 68%).
    xlabel:
        X-axis label.  Falls back to ``LABELS["z"]`` if not given.
    ax:
        Optional pre-existing Axes.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and Axes with the efficiency step curve and CI band.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    edges = np.linspace(float(variable.min()), float(variable.max()), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_inj: npt.NDArray[np.float64] = np.histogram(variable, bins=edges)[0].astype(np.float64)
    n_det: npt.NDArray[np.float64] = np.histogram(variable[detected], bins=edges)[0].astype(
        np.float64
    )

    mask = n_inj > 0
    efficiency = np.where(mask, n_det / n_inj, np.nan)

    # Wilson score CI via astropy
    ci: npt.NDArray[np.float64] = binom_conf_interval(
        n_det.astype(np.int64),
        n_inj.astype(np.int64),
        confidence_level=confidence,
        interval="wilson",
    )
    # ci shape: (2, bins) — set empty bins to NaN
    ci[:, ~mask] = np.nan

    ax.step(centers, efficiency, where="mid", color=CYCLE[0], linewidth=1.5)
    ax.fill_between(
        centers,
        ci[0],
        ci[1],
        alpha=0.3,
        color=CYCLE[0],
        step="mid",
    )

    ax.set_xlabel(xlabel if xlabel is not None else LABELS["z"])
    ax.set_ylabel(r"$P_\mathrm{det}$")
    ax.set_ylim(-0.05, 1.05)

    return fig, ax
