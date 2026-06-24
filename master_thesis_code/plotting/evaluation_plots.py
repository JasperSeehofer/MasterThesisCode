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
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from master_thesis_code.plotting._colors import (
    CMAP,
    CYCLE,
    EDGE,
    MEAN,
    METHOD,
    REFERENCE,
    SEQUENTIAL_CMAP,
    TRUTH,
    VARIANT_WITH_MASS,
)
from master_thesis_code.plotting._data import label_key
from master_thesis_code.plotting._helpers import _fig_from_ax, compute_hdi_interval, get_figure
from master_thesis_code.plotting._labels import LABELS

_DEFAULT_RECOVERY_PARAMS: list[str] = ["M", "mu", "luminosity_distance", "a", "e0", "qS"]


def _resolve_cmap(name: str) -> Colormap:
    """Resolve a palette colormap token to a registered ``Colormap`` object.

    The Atlas tokens in ``_colors`` use bare ``cmcrameri`` names (e.g.
    ``"batlow"``), but ``cmcrameri`` registers them under a ``cmc.`` prefix.
    Try the prefixed name first, then the bare name (covers the built-in
    fallback such as ``"cividis"`` when ``cmcrameri`` is absent).
    """
    for candidate in (f"cmc.{name}", name):
        try:
            return plt.get_cmap(candidate)
        except (KeyError, ValueError):
            continue
    return plt.get_cmap(name)


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

    im = ax.imshow(covariance_matrix, cmap=CMAP, aspect="auto")
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
    """3D scatter plot of sky-localization uncertainty."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(theta, phi, sky_error, c=sky_error, cmap=CMAP, alpha=0.6)
    ax.set_xlabel("theta")
    ax.set_ylabel("phi")
    ax.set_zlabel("Sky localization error")
    fig.colorbar(sc, ax=ax, label="Error")
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
    h = ax.hist2d(redshifts, masses, bins=[bins, mass_bins], cmap=CMAP)  # type: ignore[arg-type]
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
    fig = plt.figure(figsize=(7.0, 2.8 * nrows))
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
    fig.tight_layout()
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

    d_l_bins = np.linspace(d_l.min(), d_l.max(), n_d_l_bins + 1)
    M_bins = np.geomspace(M.min(), M.max(), n_m_bins + 1)

    H_det, _, _ = np.histogram2d(d_l, M, bins=[d_l_bins, M_bins], weights=detected)
    H_all, _, _ = np.histogram2d(d_l, M, bins=[d_l_bins, M_bins])
    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(H_all > 0, H_det / H_all, np.nan)

    im = ax.imshow(
        P.T,
        origin="lower",
        aspect="auto",
        extent=(float(d_l_bins[0]), float(d_l_bins[-1]), 0.0, float(n_m_bins)),
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    # Use the log-mass tick labels.
    ax.set_yticks(np.arange(n_m_bins + 1)[::2])
    ax.set_yticklabels([f"{m:.1e}" for m in M_bins[::2]])
    ax.set_xlabel(r"$d_L\,[\mathrm{Gpc}]$")
    ax.set_ylabel(LABELS["M"])
    fig.colorbar(im, ax=ax, label=r"$P_\mathrm{det}(\mathrm{SNR}\geq " + f"{snr_threshold:g})$")
    ax.set_title(
        rf"Detection probability surface ($N_\mathrm{{inj}}={len(df)}$)",
        fontsize="medium",
    )
    # Skip tight_layout: matplotlib's constrained-layout (set in the
    # project mplstyle) handles colorbar packing and conflicts with a
    # post-hoc tight_layout call.
    return fig, ax


# ---------------------------------------------------------------------------
# fig04: detection yield (injected vs detected redshift + detection fraction)
# ---------------------------------------------------------------------------


def plot_detection_yield(
    injected_redshifts: npt.NDArray[np.float64] | None,
    detected_redshifts: npt.NDArray[np.float64] | None,
    *,
    bins: int = 30,
    ax: Axes | None = None,
) -> tuple[Figure, Axes] | None:
    """Injected-vs-detected redshift yield with the true detection fraction.

    The full injected sample is drawn as an open (step) histogram and the
    detected sub-sample as a filled histogram in the shared dark-siren hue;
    the per-bin detection fraction ``N_det / N_inj`` is overlaid on a twin
    axis. This is a *selection* diagnostic: it only carries meaning when the
    injected and detected samples are genuinely distinct populations.

    The injection campaign that supplies the full (sub-threshold-inclusive)
    sample is not stored alongside the production run, so this figure GATES:
    it returns ``None`` whenever the injected sample is absent, empty, or
    identical to the detected sample (which would force a meaningless
    fraction of ~1 everywhere).

    Parameters
    ----------
    injected_redshifts:
        Redshifts of *all* injected events (detected and not). ``None`` or an
        empty array triggers the data gate.
    detected_redshifts:
        Redshifts of the events that passed the SNR threshold. ``None`` or an
        empty array triggers the data gate.
    bins:
        Number of shared histogram bins.
    ax:
        Optional pre-existing Axes to draw on.

    Returns
    -------
    tuple[Figure, Axes] | None
        Figure and the primary (left) Axes, or ``None`` when the injected
        sample required for a meaningful yield is unavailable.
    """
    # --- Data gate: the injection campaign is not local -------------------
    if injected_redshifts is None or detected_redshifts is None:
        return None
    if injected_redshifts.size == 0 or detected_redshifts.size == 0:
        return None
    # A detected==injected pasted array yields fraction ~1 in every bin and
    # tells us nothing about selection; treat it as a missing injected sample.
    if injected_redshifts.size == detected_redshifts.size and np.array_equal(
        np.sort(injected_redshifts), np.sort(detected_redshifts)
    ):
        return None

    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    dark = METHOD["dark"]

    # Shared bin edges spanning both populations.
    lo = min(float(injected_redshifts.min()), float(detected_redshifts.min()))
    hi = max(float(injected_redshifts.max()), float(detected_redshifts.max()))
    bin_edges_arr = np.linspace(lo, hi, bins + 1)
    bin_edges: list[float] = bin_edges_arr.tolist()
    bin_centers = 0.5 * (bin_edges_arr[:-1] + bin_edges_arr[1:])

    # Left y-axis: injected (open outline) + detected (filled), one hue.
    ax.hist(
        injected_redshifts,
        bins=bin_edges,
        histtype="step",
        color=dark,
        linewidth=1.4,
        zorder=3,
        label="Injected",
    )
    ax.hist(
        detected_redshifts,
        bins=bin_edges,
        color=dark,
        alpha=0.30,
        linewidth=0,
        zorder=2,
        label="Detected",
    )

    # Right y-axis: true per-bin detection fraction.
    counts_inj, _ = np.histogram(injected_redshifts, bins=bin_edges_arr)
    counts_det, _ = np.histogram(detected_redshifts, bins=bin_edges_arr)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(counts_inj > 0, counts_det / counts_inj, np.nan)

    ax2 = ax.twinx()
    ax2.plot(
        bin_centers,
        fraction,
        color=MEAN,
        linewidth=1.6,
        marker="o",
        markersize=2.5,
        zorder=4,
        label="Detection fraction",
    )
    ax2.set_ylabel("Detection fraction")
    ax2.set_ylim(0.0, 1.05)

    # Combined legend (both axes), small to fit a single-column figure.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=6)

    overall = float(detected_redshifts.size) / float(injected_redshifts.size)
    # Keep '%' inside math mode and escape it so it renders under both
    # mathtext and usetex (mirrors plot_combined_posterior's title).
    ax.set_title(
        rf"Detection yield ($N_\mathrm{{det}}/N_\mathrm{{inj}} = {overall * 100:.1f}\,\%$)"
    )
    ax.set_xlabel(LABELS["z"])
    ax.set_ylabel("Count")
    return fig, ax


# ---------------------------------------------------------------------------
# fig09: detection efficiency / selection function (smooth p_det vs z)
# ---------------------------------------------------------------------------


def plot_detection_efficiency(
    injected: npt.NDArray[np.float64] | None,
    detected: npt.NDArray[np.bool_] | None,
    *,
    bins: int = 20,
    xlabel: str | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes] | None:
    """Selection function: a smooth detection probability ``p_det`` vs redshift.

    The raw per-bin efficiency ``N_det / N_inj`` is shown as faint points and
    a sequential-colormap-shaded band, then summarised by a smooth, monotone
    selection curve (a logistic-style fit) so the figure reads as a *selection
    function* rather than a noisy step histogram. Single-column aspect.

    A genuine efficiency curve needs a mix of detected and non-detected
    injections. The production run only persists the *detected* Cramer-Rao
    rows (every injection there has ``detected = True``), and the injection
    campaign that holds the sub-threshold draws is not local, so this figure
    GATES: it returns ``None`` whenever the detection labels are absent or
    contain no non-detections (which would pin the curve at 1.0 everywhere).

    Parameters
    ----------
    injected:
        Independent variable (redshift) for every injection. ``None`` or an
        empty array triggers the data gate.
    detected:
        Boolean mask, ``True`` for injections that passed the SNR threshold.
        ``None``, or a mask that is entirely ``True``/``False``, triggers the
        data gate.
    bins:
        Number of equal-width redshift bins used for the empirical points.
    xlabel:
        X-axis label. Falls back to ``LABELS["z"]`` when not given.
    ax:
        Optional pre-existing Axes.

    Returns
    -------
    tuple[Figure, Axes] | None
        Figure and Axes with the selection-function curve, or ``None`` when
        a meaningful efficiency cannot be computed from local data.
    """
    # --- Data gate: need a mix of detected and non-detected injections ----
    if injected is None or detected is None:
        return None
    if injected.size == 0 or detected.size == 0 or injected.size != detected.size:
        return None
    n_det_total = int(np.count_nonzero(detected))
    if n_det_total == 0 or n_det_total == detected.size:
        # All-True (production CRB) or all-False -> no selection information.
        return None

    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    cmap = _resolve_cmap(SEQUENTIAL_CMAP)

    edges = np.linspace(float(injected.min()), float(injected.max()), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    n_inj: npt.NDArray[np.float64] = np.histogram(injected, bins=edges)[0].astype(np.float64)
    n_det: npt.NDArray[np.float64] = np.histogram(injected[detected], bins=edges)[0].astype(
        np.float64
    )
    mask = n_inj > 0
    efficiency = np.where(mask, n_det / n_inj, np.nan)

    # Binomial standard error per bin for the faint empirical band.
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.where(
            mask, np.sqrt(np.clip(efficiency * (1.0 - efficiency), 0.0, None) / n_inj), np.nan
        )

    # Empirical points + shaded uncertainty (sequential cmap -> P_det role).
    band_color = cmap(0.35)
    point_color = cmap(0.7)
    ax.fill_between(
        centers,
        efficiency - sigma,
        efficiency + sigma,
        where=mask.tolist(),
        color=band_color,
        alpha=0.35,
        linewidth=0,
        zorder=1,
    )
    ax.scatter(
        centers[mask],
        efficiency[mask],
        s=12,
        color=point_color,
        edgecolor=EDGE,
        linewidths=0.3,
        zorder=2,
        label="Empirical",
    )

    # Smooth selection function: monotone logistic fit p_det(z) = 1/(1+e^{k(z-z0)}).
    z_fit = np.linspace(float(injected.min()), float(injected.max()), 200)
    p_smooth = _fit_selection_function(centers[mask], efficiency[mask], n_inj[mask], z_fit)
    ax.plot(
        z_fit,
        p_smooth,
        color=cmap(0.0),
        linewidth=1.8,
        zorder=3,
        label=r"Selection function $p_\mathrm{det}(z)$",
    )

    ax.set_xlabel(xlabel if xlabel is not None else LABELS["z"])
    ax.set_ylabel(r"$p_\mathrm{det}$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=6)
    return fig, ax


def _fit_selection_function(
    z: npt.NDArray[np.float64],
    p: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    z_eval: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Weighted logistic fit of a (decreasing) selection function ``p_det(z)``.

    Fits ``p(z) = 1 / (1 + exp(k * (z - z0)))`` by weighted least squares and
    evaluates it on *z_eval*. Falls back to a clipped linear interpolation if
    the fit fails to converge or there are too few usable bins.

    Parameters
    ----------
    z, p:
        Bin-centre redshifts and empirical efficiencies of the fitted bins.
    weights:
        Per-bin injection counts (used as least-squares weights).
    z_eval:
        Redshift grid on which to evaluate the smooth curve.

    Returns
    -------
    npt.NDArray[np.float64]
        Smooth ``p_det`` evaluated on *z_eval*, clipped to ``[0, 1]``.
    """
    from scipy.optimize import curve_fit

    def _logistic(zz: npt.NDArray[np.float64], k: float, z0: float) -> npt.NDArray[np.float64]:
        return 1.0 / (1.0 + np.exp(k * (zz - z0)))

    finite = np.isfinite(p) & np.isfinite(z)
    z_f = z[finite]
    p_f = np.clip(p[finite], 1e-6, 1.0 - 1e-6)
    w_f = weights[finite]
    if z_f.size >= 3:
        try:
            popt, _ = curve_fit(
                _logistic,
                z_f,
                p_f,
                p0=[10.0, float(np.median(z_f))],
                sigma=1.0 / np.sqrt(np.clip(w_f, 1.0, None)),
                maxfev=10000,
            )
            return np.clip(_logistic(z_eval, float(popt[0]), float(popt[1])), 0.0, 1.0)
        except (RuntimeError, ValueError):
            pass
    # Fallback: clipped linear interpolation over the finite bins.
    if z_f.size >= 2:
        return np.clip(np.interp(z_eval, z_f, p_f), 0.0, 1.0)
    return np.full_like(z_eval, float(np.mean(p_f)) if p_f.size else np.nan)
