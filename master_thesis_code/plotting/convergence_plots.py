"""H0 convergence diagnostics and detection efficiency curves.

Factory functions for two key thesis diagnostic plots:

- **H0 convergence** (two-panel): posterior curves narrowing with increasing
  event count (left) and credible-interval width vs N with 1/sqrt(N)
  reference (right).
- **Detection efficiency**: binned detection fraction with Wilson score
  confidence intervals.
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from astropy.stats import binom_conf_interval
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import CYCLE, TRUTH, VARIANT_NO_MASS, VARIANT_WITH_MASS
from master_thesis_code.plotting._helpers import _fig_from_ax, compute_credible_interval, get_figure
from master_thesis_code.plotting._labels import LABELS

if TYPE_CHECKING:
    # Type-only import avoids a circular dep with convergence_analysis at runtime.
    from master_thesis_code.plotting.convergence_analysis import ImprovementBank

# Default subset sizes for convergence analysis
_DEFAULT_SUBSETS: list[int] = [1, 5, 10, 25, 50, 100]


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
    true_h: float | None = None,
    subset_sizes: list[int] | None = None,
    seed: int = 42,
    level: float = 0.68,
    h_values_alt: npt.NDArray[np.float64] | None = None,
    event_posteriors_alt: list[npt.NDArray[np.float64]] | None = None,
    label: str = r"Without $M_z$",
    label_alt: str = r"With $M_z$",
    color: str | None = None,
    color_alt: str | None = None,
    bootstrap_bank: "ImprovementBank | None" = None,
    ax: None = None,  # noqa: ARG001 — reserved for API consistency
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Two-panel H0 convergence plot, optionally comparing two variants.

    Left panel: combined posterior curves for increasing event counts.
    Right panel: credible-interval width vs number of events with a
    1/sqrt(N) reference curve.

    Parameters
    ----------
    h_values:
        Grid of Hubble-constant values (shared x-axis for posteriors).
    event_posteriors:
        Per-event posterior arrays evaluated on *h_values*.
    true_h:
        If given, draw a vertical truth line on the left panel.
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
        Legend label for the primary variant.
    label_alt:
        Legend label for the alternative variant.
    color:
        Curve color for the primary variant.
    color_alt:
        Curve color for the alternative variant.
    bootstrap_bank:
        Optional :class:`ImprovementBank` from
        :func:`compute_m_z_improvement_bank`.  When provided, the right
        panel draws a 16/84 percentile HDI band around the CI-width
        curve, per variant (primary, alt).  Default ``None`` preserves
        the pre-VIZ-02 behavior (no band).
    ax:
        Ignored (two-panel layout always created internally).

    Returns
    -------
    tuple[Figure, NDArray[object]]
        Figure and array of two Axes ``[ax_posterior, ax_ci_width]``.
    """
    if color is None:
        color = VARIANT_NO_MASS
    if color_alt is None:
        color_alt = VARIANT_WITH_MASS

    posteriors_list: list[npt.NDArray[np.float64]] = list(event_posteriors)
    n_events = len(posteriors_list)

    # Resolve subset sizes, cap at available events
    if subset_sizes is None:
        sizes = [s for s in _DEFAULT_SUBSETS if s <= n_events]
        if not sizes:
            sizes = [n_events]
    else:
        sizes = [min(s, n_events) for s in subset_sizes]

    fig, (ax_post, ax_ci) = get_figure(nrows=1, ncols=2, preset="double")

    rng = np.random.default_rng(seed)

    # --- Primary variant ---
    ci_widths = _convergence_ci_widths(h_values, posteriors_list, sizes, rng, level)

    # Show combined posterior at largest subset for left panel
    rng_post = np.random.default_rng(seed)
    indices = rng_post.choice(n_events, size=sizes[-1], replace=False)
    log_posts = [np.log(np.maximum(posteriors_list[i], 1e-300)) for i in indices]
    log_combined = np.sum(log_posts, axis=0)
    log_combined -= log_combined.max()
    combined = np.exp(log_combined)
    norm = np.trapezoid(combined, h_values)
    if norm > 0:
        combined /= norm
    ax_post.plot(h_values, combined, color=color, label=label)

    # Right panel: CI width vs N
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    ci_arr = np.asarray(ci_widths, dtype=np.float64)
    ax_ci.plot(sizes_arr, ci_arr, "o-", color=color, label=label)

    # --- Alternative variant (if provided) ---
    if event_posteriors_alt is not None:
        h_alt = h_values_alt if h_values_alt is not None else h_values
        posteriors_alt_list: list[npt.NDArray[np.float64]] = list(event_posteriors_alt)
        n_alt = len(posteriors_alt_list)
        sizes_alt = [min(s, n_alt) for s in sizes]

        rng_alt = np.random.default_rng(seed)
        ci_widths_alt = _convergence_ci_widths(
            h_alt, posteriors_alt_list, sizes_alt, rng_alt, level
        )

        # Combined posterior for left panel
        rng_alt_post = np.random.default_rng(seed)
        indices_alt = rng_alt_post.choice(n_alt, size=sizes_alt[-1], replace=False)
        log_posts_alt = [np.log(np.maximum(posteriors_alt_list[i], 1e-300)) for i in indices_alt]
        log_combined_alt = np.sum(log_posts_alt, axis=0)
        log_combined_alt -= log_combined_alt.max()
        combined_alt = np.exp(log_combined_alt)
        norm_alt = np.trapezoid(combined_alt, h_alt)
        if norm_alt > 0:
            combined_alt /= norm_alt
        ax_post.plot(h_alt, combined_alt, color=color_alt, linestyle="--", label=label_alt)

        sizes_alt_arr = np.asarray(sizes_alt, dtype=np.float64)
        ci_alt_arr = np.asarray(ci_widths_alt, dtype=np.float64)
        ax_ci.plot(sizes_alt_arr, ci_alt_arr, "s--", color=color_alt, label=label_alt)

    # --- Optional bootstrap HDI band on the right panel (VIZ-02) ---
    if bootstrap_bank is not None:
        b_sizes = np.asarray(bootstrap_bank.sizes, dtype=np.float64)
        # Primary variant (no mass)
        w_no_lo = np.asarray(bootstrap_bank.metrics_no_mass["hdi68_width"]["p16"], dtype=np.float64)
        w_no_hi = np.asarray(bootstrap_bank.metrics_no_mass["hdi68_width"]["p84"], dtype=np.float64)
        ax_ci.fill_between(b_sizes, w_no_lo, w_no_hi, color=color, alpha=0.2, zorder=2)
        # Alt variant (with mass) — only if alt posteriors were provided
        if event_posteriors_alt is not None:
            w_with_lo = np.asarray(
                bootstrap_bank.metrics_with_mass["hdi68_width"]["p16"], dtype=np.float64
            )
            w_with_hi = np.asarray(
                bootstrap_bank.metrics_with_mass["hdi68_width"]["p84"], dtype=np.float64
            )
            ax_ci.fill_between(b_sizes, w_with_lo, w_with_hi, color=color_alt, alpha=0.2, zorder=2)

    # 1/sqrt(N) reference curve scaled to match first point of primary
    if len(sizes) > 1 and ci_widths[0] > 0:
        ref = ci_widths[0] * np.sqrt(sizes_arr[0]) / np.sqrt(sizes_arr)
        ax_ci.plot(
            sizes_arr,
            ref,
            ":",
            color=CYCLE[5],
            alpha=0.6,
            label=r"$1/\sqrt{N}$ ref",
        )

    # Left panel styling
    ax_post.set_xlabel(LABELS["h"])
    ax_post.set_ylabel("Posterior density")
    if true_h is not None:
        ax_post.axvline(true_h, color=TRUTH, linestyle="--", label="Truth")
    ax_post.legend(fontsize="small")

    # Right panel styling
    ax_ci.set_xlabel("Number of events")
    ax_ci.set_ylabel(rf"{int(level * 100)}\% CI width")
    ax_ci.legend(fontsize="small")

    fig.tight_layout()
    return fig, np.array([ax_post, ax_ci], dtype=object)


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
