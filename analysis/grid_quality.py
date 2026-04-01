"""Grid quality analysis for P_det injection campaign data.

Computes per-bin Wilson 95% confidence intervals for 30x20 and 15x10 grids,
compares grid resolutions, and quantifies interpolation error.

Uses IDENTICAL binning logic to SimulationDetectionProbability._build_grid_2d()
to ensure quality flags are meaningful.

Reference: Brown, Cai, DasGupta (2001) Stat. Sci. 16:101-133 (Wilson score CI).
Library: astropy.stats.binom_conf_interval (handles k=0, k=n edge cases).

Units: d_L in Gpc, M in solar masses, h dimensionless.
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd
from astropy.stats import binom_conf_interval  # type: ignore[import-untyped]
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SNR_THRESHOLD: float = 15.0
CONFIDENCE_LEVEL: float = 0.9545  # 95% (2-sigma)
UNRELIABLE_THRESHOLD: int = 10

# Default grid dimensions
FINE_DL_BINS: int = 30
FINE_M_BINS: int = 20
COARSE_DL_BINS: int = 15
COARSE_M_BINS: int = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class GridResult:
    """Result of binning injection data into a P_det grid with Wilson CIs."""

    h_val: float
    dl_bins: int
    M_bins: int
    dl_edges: npt.NDArray[np.float64]
    M_edges: npt.NDArray[np.float64]
    dl_centers: npt.NDArray[np.float64]
    M_centers: npt.NDArray[np.float64]
    n_total: npt.NDArray[np.float64]
    n_detected: npt.NDArray[np.float64]
    p_hat: npt.NDArray[np.float64]
    ci_lower: npt.NDArray[np.float64]
    ci_upper: npt.NDArray[np.float64]
    ci_half_width: npt.NDArray[np.float64]
    reliable: npt.NDArray[np.bool_]
    n_events_total: int


@dataclass
class GridSummary:
    """Summary statistics for a single grid at one h-value."""

    h_val: float
    dl_bins: int
    M_bins: int
    n_events: int
    bins_total: int
    bins_empty: int
    bins_unreliable: int
    bins_reliable: int
    median_ci_hw_reliable: float
    max_ci_hw_nonempty: float
    boundary_bins: int  # 0.05 < p_hat < 0.95
    boundary_median_ci_hw: float | None
    pdet_gt0_bins: int


@dataclass
class ComparisonResult:
    """Comparison metrics between fine and coarse grids."""

    h_val: float
    fine_summary: GridSummary
    coarse_summary: GridSummary
    median_interp_error: float
    max_interp_error: float
    frac_error_gt_005: float
    n_interp_bins: int  # bins with P_det > 0 used for interpolation error
    boundary_fine_median_ci_hw: float | None
    boundary_coarse_median_ci_hw: float | None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_injection_data(
    injection_dir: str | Path,
) -> dict[float, pd.DataFrame]:
    """Load injection CSVs grouped by h-value.

    Uses the same regex and file-globbing logic as
    SimulationDetectionProbability.__init__().

    Args:
        injection_dir: Directory containing injection CSV files.

    Returns:
        Dict mapping h-value to concatenated DataFrame.
    """
    injection_dir = Path(injection_dir)
    patterns = [
        str(injection_dir / "injection_h_*_task_*.csv"),
        str(injection_dir / "injection_h_*.csv"),
    ]
    csv_files: list[str] = []
    for pattern in patterns:
        csv_files.extend(glob.glob(pattern))
    csv_files = sorted(set(csv_files))

    if not csv_files:
        msg = f"No injection CSV files found in '{injection_dir}'."
        raise FileNotFoundError(msg)

    h_pattern = re.compile(r"injection_h_(\d+p\d+)")
    h_file_map: dict[float, list[str]] = {}
    for f in csv_files:
        match = h_pattern.search(f)
        if match:
            h_val = float(match.group(1).replace("p", "."))
            h_file_map.setdefault(h_val, []).append(f)

    result: dict[float, pd.DataFrame] = {}
    for h_val in sorted(h_file_map):
        dfs = [pd.read_csv(f) for f in h_file_map[h_val]]
        result[h_val] = pd.concat(dfs, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# Grid construction (mirrors SimulationDetectionProbability._build_grid_2d)
# ---------------------------------------------------------------------------
def compute_bin_edges(
    dl_vals: npt.NDArray[np.float64],
    M_vals: npt.NDArray[np.float64],
    dl_bins: int,
    M_bins: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    float,
    float,
]:
    """Compute bin edges matching SimulationDetectionProbability._build_grid_2d.

    Returns:
        (dl_edges, M_edges, dl_max, M_min, M_max)
    """
    dl_max = float(np.max(dl_vals)) * 1.1
    dl_edges = np.linspace(0, dl_max, dl_bins + 1)

    M_min = float(np.min(M_vals)) * 0.9
    M_max = float(np.max(M_vals)) * 1.1
    M_edges = np.geomspace(M_min, M_max, M_bins + 1)

    return dl_edges, M_edges, dl_max, M_min, M_max


def build_grid_with_ci(
    df: pd.DataFrame,
    h_val: float,
    dl_bins: int,
    M_bins: int,
    snr_threshold: float = SNR_THRESHOLD,
    confidence_level: float = CONFIDENCE_LEVEL,
    *,
    dl_edges: npt.NDArray[np.float64] | None = None,
    M_edges: npt.NDArray[np.float64] | None = None,
) -> GridResult:
    """Build a P_det grid with Wilson confidence intervals.

    If dl_edges / M_edges are provided, use them directly (for matched-range
    comparison). Otherwise compute from data.

    Args:
        df: Injection DataFrame with columns luminosity_distance, M, SNR.
        h_val: Hubble parameter value for labeling.
        dl_bins: Number of d_L bins.
        M_bins: Number of M bins.
        snr_threshold: SNR detection threshold.
        confidence_level: Confidence level for Wilson CI (0.9545 = 95%).
        dl_edges: Pre-computed d_L bin edges (optional).
        M_edges: Pre-computed M bin edges (optional).

    Returns:
        GridResult with per-bin statistics and CIs.
    """
    dl_vals = df["luminosity_distance"].values.astype(np.float64)
    M_vals = df["M"].values.astype(np.float64)
    snr_vals = df["SNR"].values.astype(np.float64)

    if dl_edges is None or M_edges is None:
        dl_edges, M_edges, _, _, _ = compute_bin_edges(dl_vals, M_vals, dl_bins, M_bins)

    # Total events histogram
    n_total, _, _ = np.histogram2d(dl_vals, M_vals, bins=[dl_edges, M_edges])

    # Detected events histogram
    detected_mask = snr_vals >= snr_threshold
    n_detected, _, _ = np.histogram2d(
        dl_vals[detected_mask],
        M_vals[detected_mask],
        bins=[dl_edges, M_edges],
    )

    # P_det = n_detected / n_total (0/0 -> 0.0)
    p_hat = np.divide(
        n_detected,
        n_total,
        out=np.zeros_like(n_detected, dtype=np.float64),
        where=n_total > 0,
    )

    # Wilson 95% CIs via astropy
    # binom_conf_interval expects integer k, n
    n_total_int = n_total.astype(int)
    n_detected_int = n_detected.astype(int)

    # Flatten for astropy, which expects 1D arrays
    flat_k = n_detected_int.ravel()
    flat_n = n_total_int.ravel()

    # Handle empty bins: astropy raises for n=0, so mask them
    nonempty = flat_n > 0
    ci_lo_flat = np.zeros_like(flat_k, dtype=np.float64)
    ci_hi_flat = np.zeros_like(flat_k, dtype=np.float64)

    if nonempty.any():
        ci = binom_conf_interval(
            flat_k[nonempty],
            flat_n[nonempty],
            confidence_level=confidence_level,
            interval="wilson",
        )
        ci_lo_flat[nonempty] = ci[0]
        ci_hi_flat[nonempty] = ci[1]

    ci_lower = ci_lo_flat.reshape(n_total.shape)
    ci_upper = ci_hi_flat.reshape(n_total.shape)
    ci_half_width = (ci_upper - ci_lower) / 2.0

    # Reliable mask
    reliable = n_total_int >= UNRELIABLE_THRESHOLD

    # Bin centers (matching SimulationDetectionProbability)
    dl_centers = 0.5 * (dl_edges[:-1] + dl_edges[1:])
    M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])  # geometric mean

    return GridResult(
        h_val=h_val,
        dl_bins=dl_bins,
        M_bins=M_bins,
        dl_edges=dl_edges,
        M_edges=M_edges,
        dl_centers=dl_centers,
        M_centers=M_centers,
        n_total=n_total,
        n_detected=n_detected,
        p_hat=p_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_half_width=ci_half_width,
        reliable=reliable,
        n_events_total=len(df),
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize_grid(gr: GridResult) -> GridSummary:
    """Compute summary statistics for a grid result."""
    nonempty = gr.n_total > 0
    bins_total = gr.n_total.size
    bins_empty = int(np.sum(~nonempty))
    bins_unreliable = int(np.sum(~gr.reliable & nonempty))  # n>0 but n<10
    bins_unreliable_total = int(np.sum(gr.n_total.ravel() < UNRELIABLE_THRESHOLD))
    bins_reliable = int(np.sum(gr.reliable))

    # Median CI half-width for reliable bins
    reliable_hw = gr.ci_half_width[gr.reliable]
    median_ci_hw_reliable = float(np.median(reliable_hw)) if reliable_hw.size > 0 else float("nan")

    # Max CI half-width across non-empty bins
    nonempty_hw = gr.ci_half_width[nonempty]
    max_ci_hw_nonempty = float(np.max(nonempty_hw)) if nonempty_hw.size > 0 else float("nan")

    # Boundary region: 0.05 < p_hat < 0.95
    boundary_mask = (gr.p_hat > 0.05) & (gr.p_hat < 0.95) & nonempty
    boundary_bins = int(np.sum(boundary_mask))
    boundary_median_ci_hw = (
        float(np.median(gr.ci_half_width[boundary_mask])) if boundary_bins >= 1 else None
    )

    # P_det > 0 bins
    pdet_gt0 = (gr.p_hat > 0) & nonempty
    pdet_gt0_bins = int(np.sum(pdet_gt0))

    return GridSummary(
        h_val=gr.h_val,
        dl_bins=gr.dl_bins,
        M_bins=gr.M_bins,
        n_events=gr.n_events_total,
        bins_total=bins_total,
        bins_empty=bins_empty,
        bins_unreliable=bins_unreliable_total,  # includes empty
        bins_reliable=bins_reliable,
        median_ci_hw_reliable=median_ci_hw_reliable,
        max_ci_hw_nonempty=max_ci_hw_nonempty,
        boundary_bins=boundary_bins,
        boundary_median_ci_hw=boundary_median_ci_hw,
        pdet_gt0_bins=pdet_gt0_bins,
    )


# ---------------------------------------------------------------------------
# Grid comparison
# ---------------------------------------------------------------------------
def compare_grids(
    fine: GridResult,
    coarse: GridResult,
) -> ComparisonResult:
    """Compare fine and coarse grids: CI statistics and interpolation error.

    Builds a RegularGridInterpolator from the coarse grid and evaluates
    at fine grid bin centers. Reports interpolation error for bins where
    P_det > 0 in the fine grid.
    """
    fine_summary = summarize_grid(fine)
    coarse_summary = summarize_grid(coarse)

    # Build interpolator from coarse grid
    interp = RegularGridInterpolator(
        (coarse.dl_centers, coarse.M_centers),
        coarse.p_hat,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Evaluate at fine grid bin centers
    dl_mesh, M_mesh = np.meshgrid(fine.dl_centers, fine.M_centers, indexing="ij")
    points = np.column_stack([dl_mesh.ravel(), M_mesh.ravel()])
    p_interp = interp(points).reshape(fine.p_hat.shape)

    # Compute error where fine P_det > 0
    pdet_gt0_mask = fine.p_hat > 0
    n_interp_bins = int(np.sum(pdet_gt0_mask))

    if n_interp_bins > 0:
        errors = np.abs(p_interp[pdet_gt0_mask] - fine.p_hat[pdet_gt0_mask])
        median_interp_error = float(np.median(errors))
        max_interp_error = float(np.max(errors))
        frac_error_gt_005 = float(np.sum(errors > 0.05) / n_interp_bins)
    else:
        median_interp_error = float("nan")
        max_interp_error = float("nan")
        frac_error_gt_005 = float("nan")

    # Boundary region CI comparison
    fine_nonempty = fine.n_total > 0
    fine_boundary = (fine.p_hat > 0.05) & (fine.p_hat < 0.95) & fine_nonempty
    coarse_nonempty = coarse.n_total > 0
    coarse_boundary = (coarse.p_hat > 0.05) & (coarse.p_hat < 0.95) & coarse_nonempty

    boundary_fine_median_ci_hw = (
        float(np.median(fine.ci_half_width[fine_boundary])) if np.sum(fine_boundary) >= 1 else None
    )
    boundary_coarse_median_ci_hw = (
        float(np.median(coarse.ci_half_width[coarse_boundary]))
        if np.sum(coarse_boundary) >= 1
        else None
    )

    return ComparisonResult(
        h_val=fine.h_val,
        fine_summary=fine_summary,
        coarse_summary=coarse_summary,
        median_interp_error=median_interp_error,
        max_interp_error=max_interp_error,
        frac_error_gt_005=frac_error_gt_005,
        n_interp_bins=n_interp_bins,
        boundary_fine_median_ci_hw=boundary_fine_median_ci_hw,
        boundary_coarse_median_ci_hw=boundary_coarse_median_ci_hw,
    )


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------
def run_consistency_checks(gr: GridResult) -> list[str]:
    """Run consistency checks on a grid result. Returns list of failures."""
    failures: list[str] = []
    nonempty = gr.n_total > 0

    # Check 1: CI_lower <= p_hat <= CI_upper for non-empty bins
    if nonempty.any():
        below = gr.ci_lower[nonempty] > gr.p_hat[nonempty] + 1e-12
        above = gr.p_hat[nonempty] > gr.ci_upper[nonempty] + 1e-12
        if np.any(below):
            n_bad = int(np.sum(below))
            failures.append(f"CI_lower > p_hat in {n_bad} non-empty bins")
        if np.any(above):
            n_bad = int(np.sum(above))
            failures.append(f"p_hat > CI_upper in {n_bad} non-empty bins")

    # Check 2: 0 <= CI_lower, CI_upper <= 1
    if np.any(gr.ci_lower < -1e-12):
        failures.append(f"CI_lower < 0 in {int(np.sum(gr.ci_lower < -1e-12))} bins")
    if np.any(gr.ci_upper > 1 + 1e-12):
        failures.append(f"CI_upper > 1 in {int(np.sum(gr.ci_upper > 1 + 1e-12))} bins")

    # Check 3: n_det <= n_total
    if np.any(gr.n_detected > gr.n_total + 0.5):
        failures.append("n_detected > n_total in some bins")

    # Check 4: sum of n_total ~ total events
    # Events outside bin range are not counted, so allow deficit
    total_binned = int(np.sum(gr.n_total))
    if total_binned > gr.n_events_total:
        failures.append(f"Binned events ({total_binned}) exceed total ({gr.n_events_total})")

    return failures


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_ci_heatmap(
    gr: GridResult,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot Wilson CI half-width heatmap with unreliable bins marked.

    Args:
        gr: Grid result for a single h-value.
        save_path: Path to save figure. If None, show interactively.

    Returns:
        (fig, ax) tuple.
    """
    try:
        from master_thesis_code.plotting._style import apply_style

        apply_style()
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))

    # Use pcolormesh with dl_edges on x, M_edges on y
    # ci_half_width shape is (dl_bins, M_bins); pcolormesh expects (M, dL) for
    # imshow-like orientation, but with edges it's fine transposed
    hw = gr.ci_half_width.T  # shape (M_bins, dl_bins) for y=M, x=dL

    # Mask empty bins for plotting
    nonempty_T = (gr.n_total > 0).T
    hw_masked = np.ma.masked_where(~nonempty_T, hw)

    im = ax.pcolormesh(
        gr.dl_edges,
        gr.M_edges,
        hw_masked,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.5,
        shading="flat",
    )
    cbar = fig.colorbar(im, ax=ax, label="Wilson 95% CI half-width")

    # Mark unreliable bins (n < 10) with hatching
    unreliable_T = (~gr.reliable & (gr.n_total > 0)).T
    for i in range(gr.M_bins):
        for j in range(gr.dl_bins):
            if unreliable_T[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (gr.dl_edges[j], gr.M_edges[i]),
                        gr.dl_edges[j + 1] - gr.dl_edges[j],
                        gr.M_edges[i + 1] - gr.M_edges[i],
                        fill=False,
                        hatch="///",
                        edgecolor="gray",
                        linewidth=0.5,
                    )
                )

    # Mark boundary bins (0.05 < P_det < 0.95) with border
    boundary_T = ((gr.p_hat > 0.05) & (gr.p_hat < 0.95) & (gr.n_total > 0)).T
    for i in range(gr.M_bins):
        for j in range(gr.dl_bins):
            if boundary_T[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (gr.dl_edges[j], gr.M_edges[i]),
                        gr.dl_edges[j + 1] - gr.dl_edges[j],
                        gr.M_edges[i + 1] - gr.M_edges[i],
                        fill=False,
                        edgecolor="blue",
                        linewidth=1.5,
                    )
                )

    ax.set_yscale("log")
    ax.set_xlabel(r"$d_L$ [Gpc]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title(f"Wilson CI half-width, 30x20 grid (h = {gr.h_val:.2f})")

    # Legend for markers
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="none", edgecolor="gray", hatch="///", label=r"Unreliable ($n < 10$)"),
        Patch(
            facecolor="none",
            edgecolor="blue",
            linewidth=1.5,
            label=r"Boundary ($0.05 < \hat{P} < 0.95$)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax


def plot_grid_comparison(
    fine: GridResult,
    coarse: GridResult,
    comp: ComparisonResult,
    save_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """Multi-panel comparison figure: P_det, CI half-widths, interpolation error.

    Args:
        fine: Fine (30x20) grid result.
        coarse: Coarse (15x10) grid result.
        comp: Comparison result.
        save_path: Path to save figure.

    Returns:
        (fig, axes) tuple.
    """
    try:
        from master_thesis_code.plotting._style import apply_style

        apply_style()
    except ImportError:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    h_val = fine.h_val

    # Panel (a): 30x20 P_det
    ax = axes[0, 0]
    pdet_fine_T = np.ma.masked_where((fine.n_total == 0).T, fine.p_hat.T)
    im_a = ax.pcolormesh(
        fine.dl_edges,
        fine.M_edges,
        pdet_fine_T,
        cmap="viridis",
        vmin=0,
        vmax=1,
        shading="flat",
    )
    fig.colorbar(im_a, ax=ax, label=r"$\hat{P}_\mathrm{det}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_L$ [Gpc]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title(f"(a) 30x20 $P_{{\\mathrm{{det}}}}$ (h={h_val:.2f})")

    # Panel (b): 15x10 P_det
    ax = axes[0, 1]
    pdet_coarse_T = np.ma.masked_where((coarse.n_total == 0).T, coarse.p_hat.T)
    im_b = ax.pcolormesh(
        coarse.dl_edges,
        coarse.M_edges,
        pdet_coarse_T,
        cmap="viridis",
        vmin=0,
        vmax=1,
        shading="flat",
    )
    fig.colorbar(im_b, ax=ax, label=r"$\hat{P}_\mathrm{det}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_L$ [Gpc]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title(f"(b) 15x10 $P_{{\\mathrm{{det}}}}$ (h={h_val:.2f})")

    # Panel (c): CI half-width comparison
    ax = axes[1, 0]
    hw_fine_T = np.ma.masked_where((fine.n_total == 0).T, fine.ci_half_width.T)
    hw_coarse_T = np.ma.masked_where((coarse.n_total == 0).T, coarse.ci_half_width.T)
    im_c = ax.pcolormesh(
        fine.dl_edges,
        fine.M_edges,
        hw_fine_T,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.5,
        shading="flat",
    )
    fig.colorbar(im_c, ax=ax, label="CI half-width (30x20)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_L$ [Gpc]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title(f"(c) CI half-width, 30x20 (h={h_val:.2f})")

    # Panel (d): Interpolation error
    ax = axes[1, 1]
    interp_obj = RegularGridInterpolator(
        (coarse.dl_centers, coarse.M_centers),
        coarse.p_hat,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    dl_mesh, M_mesh = np.meshgrid(fine.dl_centers, fine.M_centers, indexing="ij")
    points = np.column_stack([dl_mesh.ravel(), M_mesh.ravel()])
    p_interp = interp_obj(points).reshape(fine.p_hat.shape)
    interp_error = np.abs(p_interp - fine.p_hat)
    # Mask where fine P_det == 0 (not informative)
    interp_error_T = np.ma.masked_where((fine.p_hat == 0).T, interp_error.T)
    im_d = ax.pcolormesh(
        fine.dl_edges,
        fine.M_edges,
        interp_error_T,
        cmap="Reds",
        vmin=0,
        vmax=0.2,
        shading="flat",
    )
    fig.colorbar(im_d, ax=ax, label=r"$|P_\mathrm{interp} - P_\mathrm{fine}|$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_L$ [Gpc]")
    ax.set_ylabel(r"$M$ [$M_\odot$]")
    ax.set_title(f"(d) Interpolation error, 15x10 vs 30x20 (h={h_val:.2f})")

    fig.suptitle(
        f"Grid resolution comparison: 30x20 vs 15x10 (h = {h_val:.2f})",
        fontsize=14,
        y=1.01,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, axes


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(
    injection_dir: str | Path,
    figures_dir: str | Path | None = None,
) -> tuple[dict[float, GridResult], dict[float, GridResult], dict[float, ComparisonResult]]:
    """Run full grid quality analysis for all h-values.

    Args:
        injection_dir: Path to injection CSV directory.
        figures_dir: Path to save figures. If None, figures are not saved.

    Returns:
        (fine_grids, coarse_grids, comparisons) dicts keyed by h-value.
    """
    print("Loading injection data...")
    data = load_injection_data(injection_dir)
    print(f"Loaded data for {len(data)} h-values: {sorted(data.keys())}")

    fine_grids: dict[float, GridResult] = {}
    coarse_grids: dict[float, GridResult] = {}
    comparisons: dict[float, ComparisonResult] = {}

    for h_val, df in sorted(data.items()):
        print(f"\n{'=' * 60}")
        print(f"h = {h_val:.2f} ({len(df)} events)")
        print(f"{'=' * 60}")

        # Compute bin edges from data (fine grid defines the ranges)
        dl_vals = df["luminosity_distance"].values.astype(np.float64)
        M_vals = df["M"].values.astype(np.float64)
        dl_edges_fine, M_edges_fine, dl_max, M_min, M_max = compute_bin_edges(
            dl_vals, M_vals, FINE_DL_BINS, FINE_M_BINS
        )

        # Build fine grid (30x20)
        fine = build_grid_with_ci(
            df,
            h_val,
            FINE_DL_BINS,
            FINE_M_BINS,
            dl_edges=dl_edges_fine,
            M_edges=M_edges_fine,
        )
        fine_grids[h_val] = fine

        # Build coarse grid (15x10) with SAME ranges
        dl_edges_coarse = np.linspace(0, dl_max, COARSE_DL_BINS + 1)
        M_edges_coarse = np.geomspace(M_min, M_max, COARSE_M_BINS + 1)
        coarse = build_grid_with_ci(
            df,
            h_val,
            COARSE_DL_BINS,
            COARSE_M_BINS,
            dl_edges=dl_edges_coarse,
            M_edges=M_edges_coarse,
        )
        coarse_grids[h_val] = coarse

        # Consistency checks
        for label, gr in [("30x20", fine), ("15x10", coarse)]:
            failures = run_consistency_checks(gr)
            if failures:
                print(f"  CONSISTENCY FAILURES ({label}):")
                for f in failures:
                    print(f"    - {f}")
            else:
                print(f"  {label} consistency checks: PASSED")

        # Summary
        fine_summary = summarize_grid(fine)
        coarse_summary = summarize_grid(coarse)

        print("\n  30x20 grid:")
        print(f"    Empty bins:      {fine_summary.bins_empty}")
        print(f"    Unreliable (n<10): {fine_summary.bins_unreliable}")
        print(f"    Reliable (n>=10):  {fine_summary.bins_reliable}")
        print(f"    Median CI hw (reliable): {fine_summary.median_ci_hw_reliable:.4f}")
        print(f"    Max CI hw (non-empty):   {fine_summary.max_ci_hw_nonempty:.4f}")
        print(f"    Boundary bins (0.05<P<0.95): {fine_summary.boundary_bins}")
        print(f"    P_det > 0 bins: {fine_summary.pdet_gt0_bins}")
        if fine_summary.boundary_bins > 0:
            print(f"    Boundary median CI hw: {fine_summary.boundary_median_ci_hw:.4f}")

        print("\n  15x10 grid:")
        print(f"    Empty bins:      {coarse_summary.bins_empty}")
        print(f"    Unreliable (n<10): {coarse_summary.bins_unreliable}")
        print(f"    Reliable (n>=10):  {coarse_summary.bins_reliable}")
        print(f"    Median CI hw (reliable): {coarse_summary.median_ci_hw_reliable:.4f}")
        print(f"    Max CI hw (non-empty):   {coarse_summary.max_ci_hw_nonempty:.4f}")
        print(f"    Boundary bins (0.05<P<0.95): {coarse_summary.boundary_bins}")
        print(f"    P_det > 0 bins: {coarse_summary.pdet_gt0_bins}")
        if coarse_summary.boundary_bins > 0:
            print(f"    Boundary median CI hw: {coarse_summary.boundary_median_ci_hw:.4f}")

        # Grid comparison
        comp = compare_grids(fine, coarse)
        comparisons[h_val] = comp

        print("\n  Interpolation error (15x10 -> 30x20):")
        print(f"    Bins evaluated: {comp.n_interp_bins}")
        print(f"    Median |error|: {comp.median_interp_error:.4f}")
        print(f"    Max |error|:    {comp.max_interp_error:.4f}")
        print(f"    Frac |error|>0.05: {comp.frac_error_gt_005:.3f}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE: 30x20 vs 15x10")
    print(f"{'=' * 80}")
    header = (
        f"{'h':>6} | {'Grid':>5} | {'Empty':>5} | {'Unrel':>5} | {'Relia':>5} | "
        f"{'Med CI':>7} | {'Max CI':>7} | {'Bndry':>5} | {'P>0':>5}"
    )
    print(header)
    print("-" * len(header))
    for h_val in sorted(fine_grids.keys()):
        for label, gr in [("30x20", fine_grids[h_val]), ("15x10", coarse_grids[h_val])]:
            s = summarize_grid(gr)
            print(
                f"{h_val:6.2f} | {label:>5} | {s.bins_empty:5d} | {s.bins_unreliable:5d} | "
                f"{s.bins_reliable:5d} | {s.median_ci_hw_reliable:7.4f} | "
                f"{s.max_ci_hw_nonempty:7.4f} | {s.boundary_bins:5d} | {s.pdet_gt0_bins:5d}"
            )

    print(f"\n{'=' * 80}")
    print("INTERPOLATION ERROR SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'h':>6} | {'N bins':>6} | {'Med err':>8} | {'Max err':>8} | {'Frac>0.05':>9}")
    print("-" * 50)
    for h_val in sorted(comparisons.keys()):
        c = comparisons[h_val]
        print(
            f"{h_val:6.2f} | {c.n_interp_bins:6d} | {c.median_interp_error:8.4f} | "
            f"{c.max_interp_error:8.4f} | {c.frac_error_gt_005:9.3f}"
        )

    return fine_grids, coarse_grids, comparisons


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    injection_dir = project_root / "simulations" / "injections"
    figures_dir = project_root / "figures"

    matplotlib.use("Agg")

    fine_grids, coarse_grids, comparisons = run_analysis(injection_dir, figures_dir)

    # Generate figures for h=0.73
    h_repr = 0.73
    if h_repr in fine_grids:
        print(f"\nGenerating figures for h = {h_repr}...")
        plot_ci_heatmap(
            fine_grids[h_repr],
            save_path=figures_dir / "grid_wilson_ci_heatmap.pdf",
        )
        plot_grid_comparison(
            fine_grids[h_repr],
            coarse_grids[h_repr],
            comparisons[h_repr],
            save_path=figures_dir / "grid_30x20_vs_15x10_comparison.pdf",
        )

    print("\nDone.")
