"""Campaign dashboard composite factory function.

Assembles a 2x2 multi-panel summary figure combining the four key
campaign diagnostics: H0 posterior, SNR distribution, detection yield,
and sky localization map (Mollweide).

All functions follow the project convention: data in, ``(fig, ax)`` out.
None call ``plt.show()`` or ``plt.savefig()``.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _PRESETS
from master_thesis_code.plotting.bayesian_plots import (
    plot_combined_posterior,
    plot_snr_distribution,
)
from master_thesis_code.plotting.simulation_plots import plot_detection_yield
from master_thesis_code.plotting.sky_plots import plot_sky_localization_mollweide


def plot_campaign_dashboard(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    true_h: float,
    snr_values: npt.NDArray[np.float64],
    injected_redshifts: npt.NDArray[np.float64],
    detected_redshifts: npt.NDArray[np.float64],
    theta_s: npt.NDArray[np.float64],
    phi_s: npt.NDArray[np.float64],
    sky_snr: npt.NDArray[np.float64],
) -> tuple[Figure, dict[str, Axes]]:
    """Create a 2x2 campaign summary dashboard.

    Parameters
    ----------
    h_values:
        Hubble constant grid values.
    posterior:
        Combined posterior array over ``h_values``.
    true_h:
        True Hubble constant value for reference line.
    snr_values:
        Array of signal-to-noise ratios from the campaign.
    injected_redshifts:
        Redshifts of all injected EMRI events.
    detected_redshifts:
        Redshifts of events passing the SNR threshold.
    theta_s:
        Source colatitude in radians, range ``[0, pi]``.
    phi_s:
        Source longitude in radians, range ``[0, 2*pi]``.
    sky_snr:
        SNR values for sky map color coding.

    Returns
    -------
    tuple[Figure, dict[str, Axes]]
        Figure and dictionary of named Axes with keys
        ``"posterior"``, ``"snr"``, ``"yield"``, ``"sky"``.
    """
    width = _PRESETS["double"][0]
    figsize = (width, width * 0.75)

    fig, axd = plt.subplot_mosaic(
        [["posterior", "snr"], ["yield", "sky"]],
        figsize=figsize,
        layout="constrained",
        per_subplot_kw={"sky": {"projection": "mollweide"}},
    )

    plot_combined_posterior(h_values, posterior, true_h, ax=axd["posterior"])
    plot_snr_distribution(snr_values, ax=axd["snr"])
    plot_detection_yield(injected_redshifts, detected_redshifts, ax=axd["yield"])
    plot_sky_localization_mollweide(theta_s, phi_s, sky_snr, ax=axd["sky"])
    return fig, axd
