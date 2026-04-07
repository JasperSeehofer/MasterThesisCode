"""Paper figures for the EMRI dark siren H0 inference results.

Publication-quality figures:

1. **H0 posterior comparison** -- combined posteriors for the two analysis
   variants (without / with BH mass channel) on a single axes.
2. **Single-event likelihoods** -- 4 representative events showing how the
   BH mass channel narrows the per-event likelihood.
3. **Posterior convergence** -- CI width vs number of events with
   N^{-1/2} reference line (without-BH-mass channel only).
4. **SNR distribution** -- histogram and scatter of detected-event SNR
   (requires CRB CSV data from cluster).

All functions follow the project plotting convention: data in,
``(Figure, Axes)`` out.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import CYCLE, EDGE, MEAN, REFERENCE, TRUTH
from master_thesis_code.plotting._helpers import get_figure, save_figure
from master_thesis_code.plotting._style import apply_style

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("cluster_results/eval_corrected_full")


def _load_combined_posterior(variant: str) -> dict[str, Any]:
    """Load a toplevel combined posterior JSON.

    Parameters
    ----------
    variant:
        ``"posteriors"`` or ``"posteriors_with_bh_mass"``.

    Returns
    -------
    dict with keys ``h_values``, ``posterior``, ``map_h``, etc.
    """
    if variant == "posteriors":
        path = _DATA_ROOT / "combined_posterior.json"
    elif variant == "posteriors_with_bh_mass":
        path = _DATA_ROOT / "combined_posterior_with_bh_mass.json"
    else:
        msg = f"Unknown variant: {variant}"
        raise ValueError(msg)

    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _load_per_event_no_mass(
    base: Path,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
    """Load all per-event likelihoods from the no-mass posteriors directory.

    Returns
    -------
    h_values : sorted array of h grid points
    events : dict mapping event-ID string to likelihood array (same order as h_values)
    """
    files = sorted([f for f in os.listdir(base) if f.startswith("h_") and f.endswith(".json")])

    def _h_from_file(fname: str) -> float:
        parts = fname.replace(".json", "").split("_")
        return float(parts[1] + "." + parts[2])

    raw: dict[float, dict[str, list[float]]] = {}
    for f in files:
        h = _h_from_file(f)
        with open(base / f) as fh:
            raw[h] = json.load(fh)

    h_sorted = sorted(raw.keys())
    h_arr = np.array(h_sorted)

    # Collect event IDs from the first file
    event_ids = sorted(
        [k for k in raw[h_sorted[0]] if k.isdigit()],
        key=int,
    )

    events: dict[str, npt.NDArray[np.float64]] = {}
    for eid in event_ids:
        vals = []
        for h in h_sorted:
            v = raw[h].get(eid, [])
            vals.append(v[0] if v else 0.0)
        events[eid] = np.array(vals)

    return h_arr, events


def _load_per_event_with_mass_scalars(
    base: Path,
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
    """Load aggregated per-event likelihoods from with-mass posteriors.

    The with-BH-mass JSON files are large (~585 MB) because they contain
    per-galaxy breakdowns.  The aggregated scalar likelihoods are stored
    at the end of each file.  This function reads only the last 300 KB
    of each file and extracts the scalars with a regex, avoiding full
    JSON parsing.

    Returns
    -------
    h_values : sorted array of h grid points
    events : dict mapping event-ID string to likelihood array
    """
    files = sorted([f for f in os.listdir(base) if f.startswith("h_") and f.endswith(".json")])

    def _h_from_file(fname: str) -> float:
        parts = fname.replace(".json", "").split("_")
        return float(parts[1] + "." + parts[2])

    pattern = re.compile(r'"(\d+)": \[(\d+\.\d+(?:e[+-]?\d+)?)\]')

    raw: dict[float, dict[str, float]] = {}
    for f in files:
        h = _h_from_file(f)
        filepath = base / f
        with open(filepath, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            read_size = min(300_000, size)
            fh.seek(-read_size, 2)
            tail = fh.read().decode("utf-8")
        matches = pattern.findall(tail)
        raw[h] = {k: float(v) for k, v in matches}

    h_sorted = sorted(raw.keys())
    h_arr = np.array(h_sorted)

    event_ids = sorted(raw[h_sorted[0]].keys(), key=int)
    events: dict[str, npt.NDArray[np.float64]] = {}
    for eid in event_ids:
        vals = [raw[h].get(eid, 0.0) for h in h_sorted]
        events[eid] = np.array(vals)

    return h_arr, events


# ---------------------------------------------------------------------------
# Figure 1: H0 posterior comparison
# ---------------------------------------------------------------------------


def plot_h0_posterior_comparison(
    data_dir: Path = _DATA_ROOT,
) -> tuple[Figure, Axes]:
    """Plot combined H0 posteriors for both analysis variants.

    Parameters
    ----------
    data_dir:
        Root directory containing the combined posterior JSONs.

    Returns
    -------
    (fig, ax) following the project factory convention.
    """
    p_no = _load_combined_posterior("posteriors")
    p_with = _load_combined_posterior("posteriors_with_bh_mass")

    h_no = np.array(p_no["h_values"])
    h_with = np.array(p_with["h_values"])
    post_no = np.array(p_no["posterior"])
    post_with = np.array(p_with["posterior"])

    # Peak-normalize
    post_no_norm = post_no / np.max(post_no)
    post_with_norm = post_with / np.max(post_with)

    fig, ax = get_figure(preset="single")

    # Plot with markers to honestly show grid resolution
    ax.plot(
        h_no,
        post_no_norm,
        "o-",
        color=CYCLE[0],
        label=r"Without $M_z$",
        markersize=4,
        zorder=3,
    )
    ax.plot(
        h_with,
        post_with_norm,
        "s--",
        color=CYCLE[3],
        label=r"With $M_z$",
        markersize=4,
        zorder=3,
    )

    # 68% CI shading via CDF interpolation
    for h_arr, post_arr, color in [
        (h_no, post_no, CYCLE[0]),
        (h_with, post_with, CYCLE[3]),
    ]:
        norm = np.trapezoid(post_arr, h_arr)
        pn = post_arr / norm
        cdf = np.zeros(len(h_arr))
        for i in range(1, len(h_arr)):
            cdf[i] = cdf[i - 1] + np.trapezoid(pn[i - 1 : i + 1], h_arr[i - 1 : i + 1])
        cdf /= cdf[-1]
        h_fine = np.linspace(h_arr[0], h_arr[-1], 1000)
        cdf_fine = np.interp(h_fine, h_arr, cdf)
        h16 = h_fine[np.searchsorted(cdf_fine, 0.16)]
        h84 = h_fine[np.searchsorted(cdf_fine, 0.84)]
        ax.axvspan(h16, h84, alpha=0.12, color=color, zorder=1)

    # Truth line
    ax.axvline(0.73, color=TRUTH, linestyle=":", linewidth=1.2, label="Injected", zorder=2)

    ax.set_xlabel(r"$h$")
    ax.set_ylabel("Posterior (peak-normalized)")
    ax.set_xlim(0.59, 0.87)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper right", fontsize=9)

    return fig, ax


# ---------------------------------------------------------------------------
# Figure 2: Single-event likelihoods
# ---------------------------------------------------------------------------


def _select_representative_events(
    h_vals: npt.NDArray[np.float64],
    events: dict[str, npt.NDArray[np.float64]],
) -> list[str]:
    """Select 4 representative events by likelihood shape.

    Criteria: peaked, moderately peaked, broad, and multi-modal or varied.

    Parameters
    ----------
    h_vals:
        Array of h grid points.
    events:
        Dict mapping event IDs to likelihood arrays.

    Returns
    -------
    List of 4 event ID strings.
    """
    stats: list[tuple[str, float, int, float]] = []
    for eid, lik in events.items():
        mx = np.max(lik)
        if mx == 0:
            continue
        p = lik / np.sum(lik)
        h_mean = float(np.sum(p * h_vals))
        h_std = float(np.sqrt(np.sum(p * (h_vals - h_mean) ** 2)))
        n_above = int(np.sum(lik / mx > 0.1))
        stats.append((eid, h_std, n_above, mx))

    # Sort by width (h_std)
    stats.sort(key=lambda x: x[1])
    n = len(stats)

    # Pick: narrowest, 25th percentile, 50th percentile, broadest
    selected = [
        stats[max(1, n // 20)][0],  # very peaked (5th percentile)
        stats[n // 4][0],  # moderately peaked
        stats[n // 2][0],  # median width
        stats[int(0.95 * n)][0],  # very broad (95th percentile)
    ]
    return selected


def plot_single_event_likelihoods(
    data_dir: Path = _DATA_ROOT,
) -> tuple[Figure, Any]:
    """Plot single-event likelihoods for 4 representative events.

    Creates a 4-row x 2-column grid: left column = without BH mass,
    right column = with BH mass.

    Parameters
    ----------
    data_dir:
        Root directory containing per-event posterior subdirectories.

    Returns
    -------
    (fig, axes) where axes is a 4x2 ndarray of Axes.
    """
    h_no, events_no = _load_per_event_no_mass(data_dir / "posteriors")
    h_with, events_with = _load_per_event_with_mass_scalars(data_dir / "posteriors_with_bh_mass")

    # Events present in both variants
    common_ids = sorted(
        set(events_no.keys()) & set(events_with.keys()),
        key=int,
    )

    # Filter to events with nonzero data in both variants
    valid: dict[str, npt.NDArray[np.float64]] = {}
    for eid in common_ids:
        if np.max(events_no[eid]) > 0 and np.max(events_with[eid]) > 0:
            valid[eid] = events_no[eid]

    selected = _select_representative_events(h_no, valid)

    fig, axes = get_figure(
        nrows=4,
        ncols=2,
        figsize=(3.375, 6.0),
        sharex=True,
    )

    labels = ["Peaked", "Moderate", "Median", "Broad"]

    for row, (eid, label) in enumerate(zip(selected, labels)):
        ax_no: Axes = axes[row, 0]
        ax_with: Axes = axes[row, 1]

        # No-mass likelihood
        lik_no = events_no[eid]
        lik_no_norm = lik_no / np.max(lik_no) if np.max(lik_no) > 0 else lik_no
        ax_no.plot(h_no, lik_no_norm, "o-", color=CYCLE[0], markersize=2.5, linewidth=1.0)

        # With-mass likelihood
        lik_with = events_with[eid]
        lik_with_norm = lik_with / np.max(lik_with) if np.max(lik_with) > 0 else lik_with
        ax_with.plot(h_with, lik_with_norm, "s-", color=CYCLE[3], markersize=2.5, linewidth=1.0)

        # Row label
        ax_no.set_ylabel(f"{label}\n(event {eid})", fontsize=7)

        # Truth lines
        ax_no.axvline(0.73, color=TRUTH, linestyle=":", linewidth=0.8, alpha=0.7)
        ax_with.axvline(0.73, color=TRUTH, linestyle=":", linewidth=0.8, alpha=0.7)

        ax_no.set_ylim(-0.05, 1.15)
        ax_with.set_ylim(-0.05, 1.15)

        # Remove y-tick labels for right column
        ax_with.set_yticklabels([])

    # Column titles
    axes[0, 0].set_title(r"Without $M_z$", fontsize=9)
    axes[0, 1].set_title(r"With $M_z$", fontsize=9)

    # Bottom row x-labels
    axes[-1, 0].set_xlabel(r"$h$", fontsize=9)
    axes[-1, 1].set_xlabel(r"$h$", fontsize=9)

    fig.align_ylabels(axes[:, 0])

    return fig, axes


# ---------------------------------------------------------------------------
# Figure 3: Posterior convergence (without-BH-mass only)
# ---------------------------------------------------------------------------

# Subset sizes for convergence study.  Chosen to span one-and-a-half
# decades on a log scale with good coverage at both ends.
_CONVERGENCE_SUBSET_SIZES: list[int] = [10, 20, 50, 100, 150, 200, 300, 400, 500]
_CONVERGENCE_N_SUBSETS: int = 50


def _ci_width_from_log_posteriors(
    h_values: npt.NDArray[np.float64],
    log_posteriors: npt.NDArray[np.float64],
) -> float:
    """Compute 68% credible interval width from log-posterior samples.

    Parameters
    ----------
    h_values:
        Sorted 1-D array of h grid points (length G).
    log_posteriors:
        1-D array of log-posterior values on the same grid (length G).

    Returns
    -------
    Width of the symmetric 68% CI (h_84 - h_16) via CDF interpolation.
    """
    # Shift for numerical stability, exponentiate, normalise
    log_p = log_posteriors - np.max(log_posteriors)
    p = np.exp(log_p)
    norm = np.trapezoid(p, h_values)
    if norm <= 0:
        return np.nan
    pn = p / norm

    # Build CDF by cumulative trapezoid integration
    cdf = np.zeros(len(h_values))
    for i in range(1, len(h_values)):
        cdf[i] = cdf[i - 1] + np.trapezoid(pn[i - 1 : i + 1], h_values[i - 1 : i + 1])
    cdf /= cdf[-1]

    # Interpolate to find 16th and 84th percentile h values
    h_fine = np.linspace(h_values[0], h_values[-1], 2000)
    cdf_fine = np.interp(h_fine, h_values, cdf)
    idx16 = np.searchsorted(cdf_fine, 0.16)
    idx84 = np.searchsorted(cdf_fine, 0.84)
    if idx16 >= len(h_fine) or idx84 >= len(h_fine):
        return np.nan
    return float(h_fine[idx84] - h_fine[idx16])


def plot_posterior_convergence(
    data_dir: Path = _DATA_ROOT,
    *,
    subset_sizes: list[int] | None = None,
    n_subsets: int = _CONVERGENCE_N_SUBSETS,
    seed: int = 20260407,
) -> tuple[Figure, Axes]:
    """Plot 68% CI width vs number of events for the without-BH-mass channel.

    Demonstrates the expected N^{-1/2} narrowing of the posterior as more
    independent EMRI events are combined.

    The with-BH-mass channel is intentionally omitted: its per-event
    posteriors collapse to a delta function on the coarse h-grid, making
    CI-width vs N scientifically meaningless.

    Parameters
    ----------
    data_dir:
        Root directory containing ``posteriors/`` subdirectory with
        per-event JSON files.
    subset_sizes:
        List of event counts to probe.  Defaults to
        ``[10, 20, 50, 100, 150, 200, 300, 400, 500]``.
    n_subsets:
        Number of random subsets drawn at each size (default 50).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    (fig, ax) following the project factory convention.
    """
    if subset_sizes is None:
        subset_sizes = list(_CONVERGENCE_SUBSET_SIZES)

    # Load all per-event posteriors (without BH mass, 23-point h-grid)
    h_values, events = _load_per_event_no_mass(data_dir / "posteriors")
    n_h = len(h_values)

    # Build event-posterior matrix: (n_events, n_h)
    event_ids = sorted(events.keys(), key=int)
    # Filter out zero-posterior events (4 missing indices)
    valid_ids = [eid for eid in event_ids if np.max(events[eid]) > 0]
    n_events_total = len(valid_ids)

    event_matrix = np.empty((n_events_total, n_h))
    for i, eid in enumerate(valid_ids):
        event_matrix[i, :] = events[eid]

    # Pre-compute log-posteriors (clip zeros to avoid log(0))
    log_event_matrix = np.log(np.clip(event_matrix, 1e-300, None))

    rng = np.random.default_rng(seed)

    # For each subset size, draw n_subsets random subsets and compute CI width
    medians: list[float] = []
    lo_pctiles: list[float] = []
    hi_pctiles: list[float] = []
    used_sizes: list[int] = []

    for n_sub in subset_sizes:
        if n_sub > n_events_total:
            continue
        used_sizes.append(n_sub)
        widths: list[float] = []
        for _ in range(n_subsets):
            idx = rng.choice(n_events_total, size=n_sub, replace=False)
            log_combined = np.sum(log_event_matrix[idx, :], axis=0)
            w = _ci_width_from_log_posteriors(h_values, log_combined)
            if not np.isnan(w):
                widths.append(w)

        if widths:
            medians.append(float(np.median(widths)))
            lo_pctiles.append(float(np.percentile(widths, 16)))
            hi_pctiles.append(float(np.percentile(widths, 84)))
        else:
            medians.append(np.nan)
            lo_pctiles.append(np.nan)
            hi_pctiles.append(np.nan)

    x = np.array(used_sizes, dtype=float)
    y_med = np.array(medians)
    y_lo = np.array(lo_pctiles)
    y_hi = np.array(hi_pctiles)

    # -- Plot --
    fig, ax = get_figure(preset="single")

    ax.errorbar(
        x,
        y_med,
        yerr=[y_med - y_lo, y_hi - y_med],
        fmt="o",
        color=CYCLE[0],
        markersize=4,
        capsize=3,
        linewidth=1.0,
        label=r"Without $M_z$",
        zorder=3,
    )

    # N^{-1/2} reference line anchored to the largest N median
    if len(used_sizes) > 0 and not np.isnan(y_med[-1]):
        n_ref = x[-1]
        y_ref = y_med[-1]
        x_line = np.logspace(np.log10(x[0] * 0.8), np.log10(x[-1] * 1.2), 100)
        y_line = y_ref * np.sqrt(n_ref / x_line)
        ax.plot(
            x_line,
            y_line,
            "--",
            color=REFERENCE,
            linewidth=1.0,
            label=r"$\propto N_\mathrm{det}^{-1/2}$",
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of events $N_\mathrm{det}$")
    ax.set_ylabel(r"$1\sigma$ width of $h$ posterior")
    ax.legend(loc="upper right", fontsize=9)

    return fig, ax


# ---------------------------------------------------------------------------
# Figure 4: SNR distribution
# ---------------------------------------------------------------------------


def plot_snr_distribution(
    data_dir: Path = _DATA_ROOT,
    snr_threshold: float = 15.0,
) -> tuple[Figure, Any]:
    """Plot SNR distribution of detected EMRI events.

    If CRB CSV data is available in *data_dir*, creates a two-panel figure:
    left = SNR histogram, right = SNR vs luminosity distance scatter.
    Otherwise creates a placeholder documenting what data is needed.

    Parameters
    ----------
    data_dir:
        Root directory to search for CRB CSV files.
    snr_threshold:
        Detection threshold (vertical line on histogram).

    Returns
    -------
    (fig, axes) following the project factory convention.
    """
    import glob

    # Search for CRB CSV files
    csv_patterns = [
        str(data_dir / "**" / "crb*.csv"),
        str(data_dir / "**" / "CRB*.csv"),
        str(data_dir / "**" / "cramer_rao*.csv"),
        str(data_dir / "*.csv"),
    ]
    csv_files: list[str] = []
    for pat in csv_patterns:
        csv_files.extend(glob.glob(pat, recursive=True))

    if csv_files:
        # ------ Data available path ------
        import pandas as pd

        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)

        # Identify columns (case-insensitive)
        col_map = {c.lower(): c for c in df.columns}
        snr_col = col_map.get("snr", col_map.get("signal_to_noise_ratio"))
        dl_col = col_map.get("d_l", col_map.get("luminosity_distance"))
        z_col = col_map.get("z", col_map.get("redshift"))

        if snr_col is None:
            msg = f"No SNR column found in CRB CSV. Columns: {list(df.columns)}"
            raise ValueError(msg)

        snr = df[snr_col].to_numpy()
        detected = snr >= snr_threshold

        fig, axes = get_figure(nrows=1, ncols=2, preset="double")
        ax_hist: Axes = axes[0]
        ax_scat: Axes = axes[1]

        # Left: SNR histogram
        snr_det = snr[detected]
        ax_hist.hist(snr_det, bins=30, color=CYCLE[0], edgecolor=EDGE, alpha=0.8)
        ax_hist.axvline(snr_threshold, color=MEAN, linestyle="--", linewidth=1.2, label="Threshold")
        ax_hist.set_xlabel("SNR")
        ax_hist.set_ylabel("Number of events")
        ax_hist.legend(fontsize=8)

        # Right: SNR vs d_L scatter
        if dl_col is not None:
            d_l = df[dl_col].to_numpy()
            if z_col is not None:
                z = df[z_col].to_numpy()
                sc = ax_scat.scatter(
                    d_l[detected],
                    snr_det,
                    c=z[detected],
                    cmap="viridis",
                    s=8,
                    alpha=0.6,
                )
                fig.colorbar(sc, ax=ax_scat, label=r"Redshift $z$")
            else:
                ax_scat.scatter(d_l[detected], snr_det, color=CYCLE[0], s=8, alpha=0.6)
            ax_scat.axhline(snr_threshold, color=MEAN, linestyle="--", linewidth=1.0, alpha=0.7)
            ax_scat.set_xlabel(r"$d_L$ [Gpc]")
            ax_scat.set_ylabel("SNR")
        else:
            ax_scat.text(
                0.5,
                0.5,
                r"$d_L$ column not found",
                transform=ax_scat.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        return fig, axes

    # ------ Placeholder path (no CRB data locally) ------
    fig, ax = get_figure(preset="single")
    ax.text(
        0.5,
        0.55,
        "SNR distribution data not available locally",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.38,
        "Copy CRB CSV files from the cluster to\n"
        "cluster_results/eval_corrected_full/\n"
        "and re-run this figure.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all paper figures and save to ``paper/figures/``."""
    apply_style()

    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: H0 posterior comparison
    fig1, _ = plot_h0_posterior_comparison()
    save_figure(fig1, str(out_dir / "h0_posterior_comparison"))
    print(f"Saved {out_dir / 'h0_posterior_comparison.pdf'}")

    # Figure 2: Single-event likelihoods
    fig2, _ = plot_single_event_likelihoods()
    save_figure(fig2, str(out_dir / "single_event_likelihoods"))
    print(f"Saved {out_dir / 'single_event_likelihoods.pdf'}")

    # Figure 3: Posterior convergence
    fig3, _ = plot_posterior_convergence()
    save_figure(fig3, str(out_dir / "posterior_convergence"))
    print(f"Saved {out_dir / 'posterior_convergence.pdf'}")

    # Figure 4: SNR distribution
    fig4, _ = plot_snr_distribution()
    save_figure(fig4, str(out_dir / "snr_distribution"))
    print(f"Saved {out_dir / 'snr_distribution.pdf'}")


if __name__ == "__main__":
    main()
