"""H0 convergence diagnostics and detection efficiency curves.

Factory functions for two key thesis diagnostic plots:

- **H0 convergence** (two-panel): posterior curves narrowing with increasing
  event count (left) and credible-interval width vs N with 1/sqrt(N)
  reference (right).
- **Detection efficiency**: binned detection fraction with Wilson score
  confidence intervals.
"""

import numpy as np
import numpy.typing as npt
from astropy.stats import binom_conf_interval
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import CYCLE, TRUTH
from master_thesis_code.plotting._helpers import _fig_from_ax, get_figure
from master_thesis_code.plotting._labels import LABELS

# Default subset sizes for convergence analysis
_DEFAULT_SUBSETS: list[int] = [1, 5, 10, 25, 50, 100]


def _credible_interval_width(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.68,
) -> float:
    """Compute the symmetric credible-interval width at *level*.

    Parameters
    ----------
    h_values:
        Grid of h values.
    posterior:
        Posterior density evaluated on *h_values* (need not be normalized).
    level:
        Probability mass enclosed (default 68%).

    Returns
    -------
    float
        Width ``hi - lo`` of the central credible interval.
    """
    # Normalize
    norm = np.trapezoid(posterior, h_values)
    if norm <= 0:
        return float(h_values[-1] - h_values[0])
    p = posterior / norm
    # CDF via cumulative trapezoid
    dh = np.gradient(h_values)
    cdf = np.cumsum(p * dh)
    cdf /= cdf[-1]  # ensure exactly [0, 1]
    lo = float(np.interp((1 - level) / 2, cdf, h_values))
    hi = float(np.interp((1 + level) / 2, cdf, h_values))
    return hi - lo


def plot_h0_convergence(
    h_values: npt.NDArray[np.float64],
    event_posteriors: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    *,
    true_h: float | None = None,
    subset_sizes: list[int] | None = None,
    seed: int = 42,
    level: float = 0.68,
    ax: None = None,  # noqa: ARG001 — reserved for API consistency
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Two-panel H0 convergence plot.

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
    ax:
        Ignored (two-panel layout always created internally).

    Returns
    -------
    tuple[Figure, NDArray[object]]
        Figure and array of two Axes ``[ax_posterior, ax_ci_width]``.
    """
    posteriors_list: list[npt.NDArray[np.float64]] = (
        list(event_posteriors)
        if isinstance(event_posteriors, np.ndarray)
        else list(event_posteriors)
    )
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
    ci_widths: list[float] = []

    for idx, n in enumerate(sizes):
        indices = rng.choice(n_events, size=n, replace=False)
        # Log-sum-exp for numerical stability
        log_posteriors = [np.log(np.maximum(posteriors_list[i], 1e-300)) for i in indices]
        log_combined = np.sum(log_posteriors, axis=0)
        log_combined -= log_combined.max()
        combined = np.exp(log_combined)
        norm = np.trapezoid(combined, h_values)
        if norm > 0:
            combined /= norm

        color = CYCLE[idx % len(CYCLE)]
        ax_post.plot(h_values, combined, color=color, label=f"N={n}")

        width = _credible_interval_width(h_values, combined, level=level)
        ci_widths.append(width)

    # Left panel styling
    ax_post.set_xlabel(LABELS["h"])
    ax_post.set_ylabel("Posterior density")
    if true_h is not None:
        ax_post.axvline(true_h, color=TRUTH, linestyle="--", label="Truth")
    ax_post.legend(fontsize="small")

    # Right panel: CI width vs N
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    ci_arr = np.asarray(ci_widths, dtype=np.float64)
    ax_ci.plot(sizes_arr, ci_arr, "o-", color=CYCLE[0], label="CI width")

    # 1/sqrt(N) reference curve scaled to match first point
    if len(sizes) > 1 and ci_widths[0] > 0:
        ref = ci_widths[0] * np.sqrt(sizes_arr[0]) / np.sqrt(sizes_arr)
        ax_ci.plot(
            sizes_arr,
            ref,
            "--",
            color=CYCLE[1],
            alpha=0.6,
            label=r"$1/\sqrt{N}$ ref",
        )
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
