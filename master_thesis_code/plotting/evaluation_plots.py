"""Factory functions for data evaluation plots.

Extracted from ``DataEvaluation.visualize()`` in ``evaluation.py``.

All functions take data in and return ``(fig, ax)`` out.
"""

from math import ceil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from master_thesis_code.plotting._colors import (
    CMAP,
    CYCLE,
    EDGE,
    MEAN,
    NO_DATA,
    REFERENCE,
    TRUTH,
    VARIANT_WITH_MASS,
)
from master_thesis_code.plotting._data import label_key
from master_thesis_code.plotting._helpers import (
    _fig_from_ax,
    compute_hdi_interval,
    get_figure,
    make_colorbar,
    make_heatmap_norm,
)
from master_thesis_code.plotting._labels import LABELS

_DEFAULT_RECOVERY_PARAMS: list[str] = ["M", "mu", "luminosity_distance", "a", "e0", "qS"]


def plot_mean_cramer_rao_bounds(
    covariance_matrix: npt.NDArray[np.float64],
    parameter_names: list[str],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of mean Cramer-Rao bound covariance matrix."""
    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    # Covariance entries span orders of magnitude -> robust percentile-clip norm
    # so a single large diagonal element does not flatten the off-diagonals; NaN
    # entries render as the no-data gray. (The diverging correlation-matrix
    # reframe is a Phase-5 concern; here the sequential covariance map only needs
    # an explicit norm + set_bad.)
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(NO_DATA)
    im = ax.imshow(
        covariance_matrix,
        cmap=cmap,
        aspect="auto",
        norm=make_heatmap_norm(np.asarray(covariance_matrix, dtype=np.float64), mode="robust"),
    )
    tick_labels = [LABELS.get(label_key(p), p) for p in parameter_names]
    ax.set_xticks(range(len(parameter_names)))
    ax.set_yticks(range(len(parameter_names)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    fig.colorbar(im, ax=ax)
    return fig, ax


def plot_uncertainty_violins(
    uncertainties: dict[str, npt.NDArray[np.float64]],
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Violin plot of relative parameter uncertainties."""
    if ax is None:
        fig, ax = get_figure(preset="double")
    else:
        fig = _fig_from_ax(ax)

    names = list(uncertainties.keys())
    data = [uncertainties[name] for name in names]
    tick_labels = [LABELS.get(label_key(n), n) for n in names]

    parts = ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Relative uncertainty")
    return fig, ax


def plot_sky_localization_3d(
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    sky_error: npt.NDArray[np.float64],
) -> tuple[Figure, Any]:
    """Sky-localization uncertainty on a 2D Mollweide projection.

    VR-ANNO-05: replaces the former 3D scatter (a perspective-distorted,
    grayscale-hostile anti-pattern) with a flat all-sky Mollweide map that
    matches :func:`master_thesis_code.plotting.sky_plots.plot_sky_localization_mollweide`.
    Source positions are colored by their sky-localization error through the
    house sequential cmap with an explicit robust norm.  The function name and
    ``(Figure, Any)`` return are kept so existing wiring is unaffected.

    Parameters
    ----------
    theta:
        Source colatitude in radians, range ``[0, pi]``.
    phi:
        Source longitude in radians, range ``[0, 2*pi]``.
    sky_error:
        Sky-localization error (steradians) for each source; drives the color.

    Returns
    -------
    tuple[Figure, Any]
        Figure and the 2D Mollweide Axes.
    """
    fig, ax = get_figure(preset="double", subplot_kw={"projection": "mollweide"})

    # Colatitude -> latitude; longitude wrapped to [-pi, pi] for the Mollweide.
    lat = np.pi / 2.0 - theta
    lon = ((phi + np.pi) % (2.0 * np.pi)) - np.pi

    # Explicit robust norm for color consistency with the other sequential maps
    # (replaces matplotlib's silent autoscale-from-zero).
    sc = ax.scatter(
        lon,
        lat,
        c=sky_error,
        cmap=CMAP,
        alpha=0.7,
        s=12,
        zorder=5,
        rasterized=True,
        norm=make_heatmap_norm(np.asarray(sky_error, dtype=np.float64), mode="robust"),
    )
    make_colorbar(sc, fig, ax, label=r"Sky-localization error $[\mathrm{sr}]$")
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_detection_contour(
    redshifts: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """2D histogram of detections in redshift-mass space."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    mass_bins = np.geomspace(masses.min(), masses.max(), bins)
    # Counts span orders of magnitude -> LogNorm over positive counts only
    # (zeros/NaN masked by make_heatmap_norm); empty bins render as no-data gray.
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(NO_DATA)
    counts, _, _ = np.histogram2d(redshifts, masses, bins=[bins, mass_bins])
    h = ax.hist2d(
        redshifts,
        masses,
        bins=[bins, mass_bins],  # type: ignore[arg-type]
        cmap=cmap,
        norm=make_heatmap_norm(counts.astype(np.float64), mode="log"),
    )
    fig.colorbar(h[3], ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel(LABELS["M"])
    return fig, ax


def plot_generation_time_histogram(
    generation_times: npt.NDArray[np.float64],
    *,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Histogram of waveform generation times."""
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    ax.hist(generation_times, bins=bins, edgecolor=EDGE, alpha=0.7)
    ax.axvline(float(np.mean(generation_times)), color=MEAN, linestyle="dashed", label="Mean")
    ax.set_xlabel(LABELS["t"])
    ax.set_ylabel("Count")
    ax.legend()
    return fig, ax


def plot_injected_vs_recovered(
    injected: dict[str, npt.NDArray[np.float64]],
    recovered: dict[str, npt.NDArray[np.float64]],
    *,
    uncertainties: dict[str, npt.NDArray[np.float64]] | None = None,
    parameters: list[str] | None = None,
    ncols: int = 3,
) -> tuple[Figure, npt.NDArray[np.object_]]:
    """Multi-panel scatter grid comparing injected vs recovered parameters.

    Each parameter gets a main scatter panel (with identity line and
    optional 1-sigma CRB error bars) and a residual sub-panel showing
    ``recovered - injected``.

    Parameters
    ----------
    injected, recovered:
        Dicts mapping parameter name to array of values.
    uncertainties:
        Optional dict mapping parameter name to 1-sigma CRB errors.
    parameters:
        Subset of parameter names to plot.  Defaults to
        ``["M", "mu", "luminosity_distance", "a", "e0", "qS"]``.
    ncols:
        Number of columns in the grid layout.

    Returns
    -------
    tuple[Figure, npt.NDArray[np.object_]]
        Figure and 2D array of all axes (shape ``(nrows * 2, ncols)``).
    """
    if parameters is None:
        parameters = list(_DEFAULT_RECOVERY_PARAMS)

    n_params = len(parameters)
    nrows = ceil(n_params / ncols)
    # Route the figure-creation primitive through get_figure (VR-ANNO-04). The
    # dynamic row-count drives the figsize, which get_figure forwards to subplots;
    # the bare axes it returns is discarded and the panel GridSpec is added below.
    fig, _ = get_figure(figsize=(7.0, 2.8 * nrows))
    gs = GridSpec(
        nrows * 2,
        ncols,
        height_ratios=[3, 1] * nrows,
        hspace=0.05,
        wspace=0.35,
    )

    all_axes: list[list[Axes]] = []
    for _ in range(nrows * 2):
        all_axes.append([])

    for idx, p in enumerate(parameters):
        row = idx // ncols
        col = idx % ncols
        gs_main = gs[row * 2, col]
        gs_resid = gs[row * 2 + 1, col]

        ax_main: Axes = fig.add_subplot(gs_main)
        ax_resid: Axes = fig.add_subplot(gs_resid, sharex=ax_main)

        inj = injected[p]
        rec = recovered[p]

        # Identity line
        lo = min(float(inj.min()), float(rec.min()))
        hi = max(float(inj.max()), float(rec.max()))
        ax_main.plot([lo, hi], [lo, hi], color=REFERENCE, linestyle="--", linewidth=1)

        # Main scatter / errorbar
        if uncertainties is not None and p in uncertainties:
            ax_main.errorbar(
                inj,
                rec,
                yerr=uncertainties[p],
                fmt=".",
                color=CYCLE[0],
                capsize=2,
                markersize=3,
                alpha=0.7,
            )
        else:
            ax_main.scatter(inj, rec, s=9, color=CYCLE[0], alpha=0.7, rasterized=True)

        # y-axis label on leftmost column only
        if col == 0:
            lbl = LABELS.get(label_key(p), p)
            ax_main.set_ylabel(f"{lbl} (recovered)")

        # Hide x-tick labels on main panel (shared with residual)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Residual sub-panel
        residual = rec - inj
        if uncertainties is not None and p in uncertainties:
            ax_resid.errorbar(
                inj,
                residual,
                yerr=uncertainties[p],
                fmt=".",
                color=CYCLE[0],
                capsize=2,
                markersize=3,
                alpha=0.7,
            )
        else:
            ax_resid.scatter(inj, residual, s=9, color=CYCLE[0], alpha=0.7, rasterized=True)

        ax_resid.axhline(0, color=REFERENCE, linestyle="--", linewidth=1)

        # x-axis label on bottom row only
        is_bottom_row = row == nrows - 1
        if is_bottom_row:
            lbl = LABELS.get(label_key(p), p)
            ax_resid.set_xlabel(f"{lbl} (injected)")

        # Residual y-axis label on leftmost column
        if col == 0:
            ax_resid.set_ylabel(r"$\Delta$")

        all_axes[row * 2].append(ax_main)
        all_axes[row * 2 + 1].append(ax_resid)

    # Pad rows that have fewer columns than ncols
    for row_axes in all_axes:
        while len(row_axes) < ncols:
            row_axes.append(None)  # type: ignore[arg-type]

    axes_array: npt.NDArray[np.object_] = np.array(all_axes, dtype=object)
    return fig, axes_array


# ---------------------------------------------------------------------------
# Phase F2: Information monotonicity (1D vs 2D per-event HDI68 widths)
# ---------------------------------------------------------------------------


def plot_info_monotonicity(
    data_dir: Path,
    *,
    h_value_for_filter: float = 0.73,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Per-event HDI68 width in 1D vs 2D channel — info monotonicity check.

    Adding the BH-mass channel must NOT broaden any individual event's
    likelihood relative to the 1D channel: more information should
    sharpen the posterior, not blur it. This scatter visualises that
    constraint event-by-event — points below the identity line confirm
    monotonicity; points above flag a structural inconsistency.

    Parameters
    ----------
    data_dir:
        Directory holding ``posteriors/`` and ``posteriors_with_bh_mass/``
        with the per-h posterior JSONs.
    h_value_for_filter:
        Optional filter — only events with non-zero likelihood at this h
        in both channels are plotted (default 0.73, the production truth).
    ax:
        Existing axes; new figure when None.
    """
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    from master_thesis_code.plotting.convergence_analysis import (
        _load_per_event_no_mass,
        _load_per_event_with_mass_scalars,
    )

    posts_dir = Path(data_dir) / "posteriors"
    with_dir = Path(data_dir) / "posteriors_with_bh_mass"
    if not (posts_dir.is_dir() and with_dir.is_dir()):
        raise FileNotFoundError(
            f"Need both posteriors/ and posteriors_with_bh_mass/ under {data_dir}"
        )

    h_no, events_no = _load_per_event_no_mass(posts_dir)
    h_w, events_w = _load_per_event_with_mass_scalars(with_dir)
    common = sorted(set(events_no.keys()) & set(events_w.keys()), key=int)
    widths_no: list[float] = []
    widths_w: list[float] = []
    for eid in common:
        L_no = events_no[eid]
        L_w = events_w[eid]
        # Drop events with all-zero or NaN likelihoods in either channel.
        if not np.isfinite(L_no).all() or not np.isfinite(L_w).all():
            continue
        if L_no.max() <= 0 or L_w.max() <= 0:
            continue
        # Peak-normalise then compute HDI68.
        p_no = L_no / L_no.max()
        p_w = L_w / L_w.max()
        lo_n, hi_n = compute_hdi_interval(h_no, p_no, level=0.68)
        lo_w, hi_w = compute_hdi_interval(h_w, p_w, level=0.68)
        if any(np.isnan(v) for v in (lo_n, hi_n, lo_w, hi_w)):
            continue
        widths_no.append(hi_n - lo_n)
        widths_w.append(hi_w - lo_w)

    if not widths_no:
        raise ValueError("No event passed the both-channels-active filter.")

    w_no_arr = np.asarray(widths_no, dtype=np.float64)
    w_w_arr = np.asarray(widths_w, dtype=np.float64)

    # Identity line
    upper = float(max(w_no_arr.max(), w_w_arr.max()))
    ax.plot([0, upper], [0, upper], color=TRUTH, linestyle=":", linewidth=0.8, label="Identity")
    ax.scatter(
        w_no_arr,
        w_w_arr,
        s=12,
        c=VARIANT_WITH_MASS,
        edgecolor=EDGE,
        linewidths=0.3,
        alpha=0.7,
    )

    n_below = int(np.sum(w_w_arr <= w_no_arr))
    n_total = len(w_no_arr)
    ax.set_xlabel(r"HDI68 width (without $M_z$)")
    ax.set_ylabel(r"HDI68 width (with $M_z$)")
    ax.set_xlim(0, 1.05 * upper)
    ax.set_ylim(0, 1.05 * upper)
    ax.legend(loc="upper left", fontsize="small")
    ax.set_title(
        rf"Info monotonicity ({n_below}/{n_total} = {n_below / n_total:.0%} tightened by $M_z$)",
        fontsize="medium",
    )
    return fig, ax


# ---------------------------------------------------------------------------
# Phase F3: P_det(d_L, M_z) surface (injection-campaign empirical)
# ---------------------------------------------------------------------------


def plot_pdet_surface(
    injection_csv_glob: str,
    *,
    snr_threshold: float = 20.0,
    n_d_l_bins: int = 18,
    n_m_bins: int = 14,
    h_inj_filter: float | None = 0.73,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Empirical P_det(d_L, M_z) heatmap from the injection-campaign CSVs.

    Pools injection CSVs (each row = one drawn EMRI source with its
    achieved SNR and parameters) and computes the detection fraction
    ``N(SNR ≥ threshold) / N_total`` in each (d_L, M_z) bin. Optionally
    restricted to a single injection cosmology.

    Parameters
    ----------
    injection_csv_glob:
        Glob pattern for injection campaign CSVs (e.g.
        ``"simulations/injections/injection_h_0p73_task_*.csv"``).
    snr_threshold:
        SNR cut defining "detected".
    n_d_l_bins, n_m_bins:
        Histogram resolution along the d_L and M axes.
    h_inj_filter:
        When not None, keep only rows with ``h_inj`` ≈ this value.
    ax:
        Existing axes; new figure when None.
    """
    import glob

    import pandas as pd

    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    files = sorted(glob.glob(injection_csv_glob))
    if not files:
        raise FileNotFoundError(f"No injection CSVs match: {injection_csv_glob}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if h_inj_filter is not None and "h_inj" in df.columns:
        df = df[np.isclose(df["h_inj"], h_inj_filter, atol=1e-3)]

    if df.empty:
        raise ValueError("No injection rows left after filtering.")

    d_l = df["luminosity_distance"].to_numpy(dtype=np.float64)
    M = df["M"].to_numpy(dtype=np.float64)
    snr = df["SNR"].to_numpy(dtype=np.float64)
    detected = (snr >= snr_threshold).astype(np.float64)

    # Bin EDGES (length n+1) -- reused directly by pcolormesh (which wants edges,
    # not centers, so no off-by-one). M_bins is geometric -> true log spacing.
    d_l_bins = np.linspace(d_l.min(), d_l.max(), n_d_l_bins + 1)
    M_bins = np.geomspace(M.min(), M.max(), n_m_bins + 1)

    H_det, _, _ = np.histogram2d(d_l, M, bins=[d_l_bins, M_bins], weights=detected)
    H_all, _, _ = np.histogram2d(d_l, M, bins=[d_l_bins, M_bins])
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(H_all > 0, H_det / H_all, np.nan)

    # pcolormesh on the real (d_L, M) edges -> a true log mass y-axis (set below),
    # replacing the old fake-index imshow + hand-formatted ticks. P_det is bounded
    # [0, 1] and read linearly (the 0.5 horizon is the key feature), so use an
    # explicit Normalize(0, 1); NaN bins (empty (d_L, M) cells) are masked and
    # render as the set_bad no-data gray (never crashes the draw).
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(NO_DATA)
    P_masked = np.ma.masked_invalid(P.T)  # shape (n_m_bins, n_d_l_bins)
    mesh = ax.pcolormesh(
        d_l_bins,
        M_bins,
        P_masked,
        cmap=cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        shading="flat",
    )
    ax.set_yscale("log")

    # P_det = 0.5 (and 0.9) detection-horizon contour on the bin CENTERS, in EDGE
    # color, as a redundant grayscale-safe channel. Guard the all-NaN / too-few
    # case (contour needs >= 2 finite rows and cols) and only request levels that
    # actually fall inside the data range -- otherwise matplotlib draws an empty
    # contour. When neither 0.5 nor 0.9 is attainable (e.g. a high-SNR-threshold
    # slice whose P_det never reaches 0.5) the horizon is genuinely absent and we
    # skip drawing rather than fabricate a line.
    finite = np.isfinite(P)
    if finite.any() and P.shape[0] >= 2 and P.shape[1] >= 2:
        d_l_centers = 0.5 * (d_l_bins[:-1] + d_l_bins[1:])
        M_centers = np.sqrt(M_bins[:-1] * M_bins[1:])  # geometric midpoints
        with np.errstate(invalid="ignore"):
            P_contour = np.where(finite, P, 0.0)
        p_max = float(np.nanmax(P))
        attainable = [lvl for lvl in (0.5, 0.9) if 0.0 < lvl < p_max]
        if attainable:
            cs = ax.contour(
                d_l_centers,
                M_centers,
                P_contour.T,
                levels=attainable,
                colors=EDGE,
                linewidths=1.0,
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # d_L is in Gpc here: the injection-CSV luminosity_distance values span
    # ~0.4..11 (c/H0 ~ 4.1 Gpc, so z~1.5 -> ~11 Gpc). This differs from
    # LABELS["d_L"] (Mpc, used by the per-source recovery plots whose data is in
    # Mpc), so fig20 carries its own correct Gpc label rather than routing
    # through LABELS["d_L"].
    ax.set_xlabel(r"$d_L\,[\mathrm{Gpc}]$")
    ax.set_ylabel(LABELS["M"])
    fig.colorbar(mesh, ax=ax, label=r"$P_\mathrm{det}(\mathrm{SNR}\geq " + f"{snr_threshold:g})$")
    ax.set_title(
        rf"Detection probability surface ($N_\mathrm{{inj}}={len(df)}$)",
        fontsize="medium",
    )
    # Skip tight_layout: matplotlib's constrained-layout (set in the
    # project mplstyle) handles colorbar packing and conflicts with a
    # post-hoc tight_layout call.
    return fig, ax
