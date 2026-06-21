"""Population "where do the constraints come from" composite (fig22).

The single highest-value population-level static figure the pipeline lacked
(viz-redesign proposal §5.4, requirement VR-NEW-04): a one-glance answer to
"which detected events drive the H0 constraint, and how do their individual
likelihoods stack into the combined posterior?". It composes four already-tested
single-panel encodings into one 2x2 figure:

  - TOP-LEFT  (driver) : SNR x z scatter — loud, nearby events sit top-left.
  - TOP-RIGHT (sky)    : Mollweide sky map of the SAME events (delegated to
    sky_plots.plot_sky_localization_mollweide).
  - BOTTOM-LEFT (spaghetti) : de-emphasized per-event posteriors colored by a
    per-event scalar (delegated to bayesian_plots.plot_event_posteriors, which
    surfaces the latent color_by).
  - BOTTOM-RIGHT (stacked) : the canonical combined posterior (delegated to
    bayesian_plots.plot_combined_posterior) — the hero line whose MAP is anchored
    to the canonical combination.

Only the 2x2 composition (panel layout) is novel; every sub-encoding is a reuse.
NO new data source, NO physics.

All functions follow the project convention: data in, ``(fig, ax)`` out.
None call ``plt.show()`` or ``plt.savefig()``.
"""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from master_thesis_code.plotting._colors import CMAP, REFERENCE
from master_thesis_code.plotting._helpers import get_figure, make_colorbar
from master_thesis_code.plotting._labels import LABELS


def plot_population_constraint_view(
    h_values: npt.NDArray[np.float64],
    event_posteriors: list[npt.NDArray[np.float64]],
    combined_posterior: npt.NDArray[np.float64],
    true_h: float,
    theta_s: npt.NDArray[np.float64],
    phi_s: npt.NDArray[np.float64],
    snr: npt.NDArray[np.float64],
    redshift: npt.NDArray[np.float64],
    *,
    color_by: Literal["snr", "redshift"] = "snr",
    snr_threshold: float = 20.0,
) -> tuple[Figure, Any]:
    """Population constraint-provenance composite (fig22).

    Builds a 2x2 layout: a SNR x z driver scatter (top-left), a Mollweide sky map
    of the same events (top-right), de-emphasized per-event spaghetti colored by a
    per-event scalar (bottom-left), and the canonical stacked combined posterior
    (bottom-right). The sky panel is delegated to
    :func:`master_thesis_code.plotting.sky_plots.plot_sky_localization_mollweide`,
    the spaghetti to
    :func:`master_thesis_code.plotting.bayesian_plots.plot_event_posteriors`
    (surfacing ``color_by``), and the stacked posterior to
    :func:`master_thesis_code.plotting.bayesian_plots.plot_combined_posterior`.

    The figure SIZE is inherited from the REVTeX-double ``get_figure`` preset
    (no hardcoded figsize), after which the figure is cleared and a 2x2
    ``GridSpec`` is laid out so the sky cell can carry a Mollweide projection
    while the other three stay rectilinear.

    Parameters
    ----------
    h_values:
        Grid of dimensionless Hubble parameter values.
    event_posteriors:
        Per-event posterior arrays (one per detected event), aligned with
        ``theta_s`` / ``phi_s`` / ``snr`` / ``redshift``.
    combined_posterior:
        The canonical combined posterior on ``h_values`` (the hero line).
    true_h:
        True (injected) value of h for the reference lines.
    theta_s, phi_s:
        Source colatitude / longitude in radians (sky panel).
    snr:
        Achieved SNR per event (driver + color scalar + sky color).
    redshift:
        Source redshift per event (driver x-axis; alternate color scalar).
    color_by:
        Per-event scalar driving the spaghetti color — ``"snr"`` (default) or
        ``"redshift"``. The driver scatter is colored by the same scalar so the
        two panels share a visual key.
    snr_threshold:
        SNR cut drawn as the driver-panel reference line.

    Returns
    -------
    tuple[Figure, Any]
        Figure and a dict of the four panel Axes keyed
        ``"driver"``, ``"sky"``, ``"spaghetti"``, ``"stacked"``.
    """
    from master_thesis_code.plotting.bayesian_plots import (
        plot_combined_posterior,
        plot_event_posteriors,
    )
    from master_thesis_code.plotting.sky_plots import plot_sky_localization_mollweide

    # Inherit the REVTeX-double SIZE from the preset (the test asserts the size
    # comes from a preset, not a literal), then clear and lay out a 2x2 GridSpec
    # so the sky cell alone can carry a Mollweide projection.
    fig, _ = get_figure(preset="double")
    fig.clf()
    gs = GridSpec(2, 2, figure=fig)

    ax_driver: Axes = fig.add_subplot(gs[0, 0])
    ax_sky: Axes = fig.add_subplot(gs[0, 1], projection="mollweide")
    ax_spaghetti: Axes = fig.add_subplot(gs[1, 0])
    ax_stacked: Axes = fig.add_subplot(gs[1, 1])

    # Per-event color scalar shared between the driver and the spaghetti panels.
    color_values = snr if color_by == "snr" else redshift

    # --- TOP-LEFT: SNR x z driver scatter (colored by the shared scalar) ---
    sc = ax_driver.scatter(
        redshift,
        snr,
        c=color_values,
        cmap=CMAP,
        s=14,
        edgecolor=REFERENCE,
        linewidths=0.3,
        alpha=0.85,
        rasterized=True,
    )
    ax_driver.axhline(
        snr_threshold, color=REFERENCE, linestyle="--", linewidth=0.8, label="SNR threshold"
    )
    ax_driver.set_xlabel(LABELS["z"])
    ax_driver.set_ylabel(LABELS["SNR"])
    ax_driver.legend(loc="upper right", fontsize="small")
    make_colorbar(sc, fig, ax_driver, label=LABELS["SNR"] if color_by == "snr" else LABELS["z"])

    # --- TOP-RIGHT: Mollweide sky map of the same events ---
    plot_sky_localization_mollweide(theta_s, phi_s, snr, ax=ax_sky)

    # --- BOTTOM-LEFT: de-emphasized per-event spaghetti, colored by scalar ---
    # plot_event_posteriors draws per-event curves at alpha=0.5, linewidth=0.5
    # (de-emphasized) and adds the color_by colorbar — both surfaced here.
    plot_event_posteriors(
        h_values,
        event_posteriors,
        true_h,
        color_by=color_by,
        color_values=color_values,
        ax=ax_spaghetti,
    )

    # --- BOTTOM-RIGHT: canonical stacked combined posterior (the hero line) ---
    # linewidth strictly exceeds the spaghetti per-event linewidth (0.5) so the
    # stacked posterior is the visually dominant curve.
    plot_combined_posterior(
        h_values,
        combined_posterior,
        true_h,
        label="Combined",
        linewidth=2.0,
        annotate_map=True,
        legend=True,
        ax=ax_stacked,
    )

    # No fig.tight_layout: constrained_layout (project mplstyle) owns packing.
    axes = {
        "driver": ax_driver,
        "sky": ax_sky,
        "spaghetti": ax_spaghetti,
        "stacked": ax_stacked,
    }
    return fig, axes
