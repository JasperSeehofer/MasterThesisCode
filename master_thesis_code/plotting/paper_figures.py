"""Paper figures for the EMRI dark siren H0 inference results.

Two publication-quality figures:

1. **H0 posterior comparison** -- combined posteriors for the two analysis
   variants (without / with BH mass channel) on a single axes.
2. **Single-event likelihoods** -- 4 representative events showing how the
   BH mass channel narrows the per-event likelihood.

Both functions follow the project plotting convention: data in,
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

from master_thesis_code.plotting._colors import CYCLE, TRUTH
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
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate both paper figures and save to ``paper/figures/``."""
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


if __name__ == "__main__":
    main()
