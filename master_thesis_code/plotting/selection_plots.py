"""Selection-function / detection-horizon explainer composite (fig21).

The selection function — "which sources does LISA actually detect?" — is the
single most field-expected static figure the pipeline lacked (viz-redesign
proposal §5.3, requirement VR-NEW-03). This module composes it from data the
package already saves: the injection-campaign CSVs that ``plot_pdet_surface``
(fig20) already pools.

The composite is a 1x2 panel:

  - LEFT: the 1D ``p_det(d_L)`` survival marginal — the detection fraction
    ``N(SNR >= threshold) / N_total`` per luminosity-distance bin, falling from
    ~1 (nearby, always detected) to ~0 (distant, never detected). This is the
    detection horizon read as a curve.
  - RIGHT: the full 2D ``p_det(d_L, M_z)`` heatmap with the 0.5 / 0.9 horizon
    contours, DELEGATED unchanged to ``evaluation_plots.plot_pdet_surface``.

Only the composition (panel layout) is novel; both sub-encodings are reuses of
tested factories / histogram-ratio code. NO new data source, NO physics: the
survival curve is a histogram ratio of already-saved SNR values.

All functions follow the project convention: data in, ``(fig, ax)`` out.
None call ``plt.show()`` or ``plt.savefig()``.
"""

import glob
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import REFERENCE, VARIANT_NO_MASS
from master_thesis_code.plotting._helpers import get_figure
from master_thesis_code.plotting.evaluation_plots import plot_pdet_surface


def _pdet_survival_curve(
    d_l: npt.NDArray[np.float64],
    snr: npt.NDArray[np.float64],
    snr_threshold: float,
    n_bins: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """1D ``p_det(d_L)`` survival marginal of the injection campaign.

    The detection fraction ``N(SNR >= threshold) / N_total`` evaluated in each
    luminosity-distance bin — the 1D marginal of the ``plot_pdet_surface`` 2D
    surface. This is a histogram ratio of already-saved SNR values: NO new data
    source and NO physics (no formula or constant).

    Parameters
    ----------
    d_l:
        Luminosity distances of the pooled injection rows (Gpc).
    snr:
        Achieved SNR for each injection row (same length as ``d_l``).
    snr_threshold:
        SNR cut defining "detected".
    n_bins:
        Number of luminosity-distance bins.

    Returns
    -------
    centers:
        Bin-center luminosity distances (ascending, length ``n_bins``).
    fraction:
        Detection fraction per bin; bins with zero injections are ``NaN``.
    """
    d_l = np.asarray(d_l, dtype=np.float64)
    snr = np.asarray(snr, dtype=np.float64)
    detected = (snr >= snr_threshold).astype(np.float64)

    edges = np.linspace(d_l.min(), d_l.max(), n_bins + 1)
    n_det, _ = np.histogram(d_l, bins=edges, weights=detected)
    n_all, _ = np.histogram(d_l, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(n_all > 0, n_det / n_all, np.nan)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, fraction


def plot_selection_function_explainer(
    injection_csv_glob: str,
    *,
    snr_threshold: float = 20.0,
    h_inj_filter: float | None = 0.73,
    n_survival_bins: int = 18,
    axes: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, Any]:
    """Selection-function / detection-horizon explainer composite (fig21).

    Composes the 1D ``p_det(d_L)`` survival marginal (left) and the 2D
    ``p_det(d_L, M_z)`` heatmap with 0.5 / 0.9 horizon contours (right) from the
    injection-campaign CSVs. The heatmap panel is delegated unchanged to
    :func:`master_thesis_code.plotting.evaluation_plots.plot_pdet_surface`; the
    survival curve reuses the same glob + pandas pooling pattern (no new loader).

    Parameters
    ----------
    injection_csv_glob:
        Glob pattern for injection-campaign CSVs (each row = one drawn EMRI
        source with its achieved SNR and parameters), e.g.
        ``"simulations/injections/injection_h_0p73_task_*.csv"``. Columns:
        ``z, M, phiS, qS, SNR, h_inj, luminosity_distance`` (d_L in Gpc).
    snr_threshold:
        SNR cut defining "detected" (default 20.0, the production threshold).
    h_inj_filter:
        When not None, keep only rows with ``h_inj`` ~= this value (default
        0.73). Mirrors ``plot_pdet_surface``'s filter so both panels pool the
        identical population.
    n_survival_bins:
        Number of luminosity-distance bins for the 1D survival curve.
    axes:
        Optional ``(left, right)`` pre-existing Axes to draw into. When None a
        REVTeX-double 1x2 composite is created via ``get_figure`` (no hardcoded
        figsize).

    Returns
    -------
    tuple[Figure, Any]
        Figure and the array/tuple of the two panel Axes ``(left, right)``.
    """
    if axes is None:
        fig, ax_pair = get_figure(1, 2, preset="double")
        ax_left, ax_right = ax_pair[0], ax_pair[1]
    else:
        ax_left, ax_right = axes
        fig_obj = ax_left.get_figure()
        assert isinstance(fig_obj, Figure)
        fig = fig_obj

    # --- LEFT: 1D p_det(d_L) survival marginal ---
    # Reuse the SAME glob + pandas concat + h_inj filter that plot_pdet_surface
    # uses (copied here, the canonical loader is untouched), so both panels pool
    # the identical injection population.
    files = sorted(glob.glob(injection_csv_glob))
    if not files:
        raise FileNotFoundError(f"No injection CSVs match: {injection_csv_glob}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if h_inj_filter is not None and "h_inj" in df.columns:
        df = df[np.isclose(df["h_inj"], h_inj_filter, atol=1e-3)]
    if df.empty:
        raise ValueError("No injection rows left after filtering.")

    d_l = df["luminosity_distance"].to_numpy(dtype=np.float64)
    snr = df["SNR"].to_numpy(dtype=np.float64)
    centers, fraction = _pdet_survival_curve(d_l, snr, snr_threshold, n_survival_bins)

    # Plain line (no per-point markers, per VR-ANNO-02 spirit). Drop NaN
    # (empty) bins so the line does not break across a no-data gap.
    finite = np.isfinite(fraction)
    ax_left.plot(
        centers[finite],
        fraction[finite],
        color=VARIANT_NO_MASS,
        linewidth=1.5,
        label=r"$P_\mathrm{det}(d_L)$",
    )
    # 0.5 horizon guide (redundant grayscale-safe channel for the heatmap's
    # 0.5 contour) drawn in the neutral REFERENCE gray.
    ax_left.axhline(0.5, color=REFERENCE, linestyle="--", linewidth=0.8, label="0.5 horizon")
    ax_left.set_ylim(-0.02, 1.02)
    # d_L is in Gpc here (injection CSV convention; see plot_pdet_surface) —
    # carry the Gpc label literally rather than routing through LABELS["d_L"]
    # which is Mpc.
    ax_left.set_xlabel(r"$d_L\,[\mathrm{Gpc}]$")
    ax_left.set_ylabel(r"$P_\mathrm{det}$")
    ax_left.legend(loc="upper right", fontsize="small")

    # --- RIGHT: delegate to the tested fig20 heatmap (with horizon contours) ---
    plot_pdet_surface(
        injection_csv_glob,
        snr_threshold=snr_threshold,
        h_inj_filter=h_inj_filter,
        ax=ax_right,
    )

    # No fig.tight_layout: constrained_layout (project mplstyle) owns packing.
    return fig, np.array([ax_left, ax_right], dtype=object)
