"""Tests for the selection-function / detection-horizon explainer composite.

The composite ``plot_selection_function_explainer`` orchestrates the
already-tested fig20 ``plot_pdet_surface`` heatmap (right panel) plus a
1D ``p_det(d_L)`` survival marginal (left panel) derived from the same
injection CSVs. Only the composition is novel; every sub-encoding is a
reuse of a tested factory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _PRESETS
from master_thesis_code.plotting.selection_plots import (
    _pdet_survival_curve,
    plot_selection_function_explainer,
)


def _write_injection_csv(
    path: str,
    *,
    n: int = 600,
    seed: int = 0,
    horizon: bool = True,
) -> None:
    """Write a synthetic injection-campaign CSV with the production columns.

    When ``horizon`` is True the low-d_L rows are overwhelmingly above the
    SNR threshold while the high-d_L rows are below it, so the pooled p_det
    crosses 0.5 (the detection horizon exists). When False, the SNR values
    are uniformly low and at large distances so 0.5 is never reached.
    """
    rng = np.random.default_rng(seed)
    if horizon:
        luminosity_distance = rng.uniform(0.4, 11.0, n)  # Gpc
        # SNR falls with distance: close sources detected, far ones not.
        base = 80.0 * (0.4 / np.clip(luminosity_distance, 0.4, None))
        snr = base + rng.normal(0.0, 5.0, n)
        snr = np.clip(snr, 1.0, None)
    else:
        # Far, faint population: SNR never clears 20 in a majority of any bin.
        luminosity_distance = rng.uniform(8.0, 11.0, n)
        snr = rng.uniform(1.0, 8.0, n)
    M = 10 ** rng.uniform(4.5, 7.0, n)
    df = pd.DataFrame(
        {
            "z": rng.uniform(0.1, 1.5, n),
            "M": M,
            "phiS": rng.uniform(0, 2 * np.pi, n),
            "qS": rng.uniform(0, np.pi, n),
            "SNR": snr,
            "h_inj": np.full(n, 0.73),
            "luminosity_distance": luminosity_distance,
        }
    )
    df.to_csv(path, index=False)


def test_selection_explainer_returns_figure_and_two_axes(tmp_path: Path) -> None:
    """Smoke: composite returns (Figure, sequence of Axes) without raising."""
    _write_injection_csv(str(tmp_path / "injection_h_0p73_task_0.csv"), n=600, seed=1)
    fig, axes = plot_selection_function_explainer(
        str(tmp_path / "injection_h_0p73_task_*.csv"), snr_threshold=20.0
    )
    assert isinstance(fig, Figure)
    flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
    assert len([a for a in flat if isinstance(a, Axes)]) >= 2


def test_survival_curve_is_monotone_non_increasing(tmp_path: Path) -> None:
    """The p_det(d_L) survival marginal falls from ~1 at low d_L to ~0 at high."""
    _write_injection_csv(str(tmp_path / "injection_h_0p73_task_0.csv"), n=4000, seed=2)
    df = pd.read_csv(str(tmp_path / "injection_h_0p73_task_0.csv"))
    centers, frac = _pdet_survival_curve(
        df["luminosity_distance"].to_numpy(dtype=np.float64),
        df["SNR"].to_numpy(dtype=np.float64),
        snr_threshold=20.0,
        n_bins=12,
    )
    finite = np.isfinite(frac)
    vals = frac[finite]
    assert vals.size >= 2
    # First (lowest-d_L) bin detects more than the last (highest-d_L) bin.
    assert vals[0] >= vals[-1]
    # Globally non-increasing within numerical noise (allow small upticks).
    diffs = np.diff(vals)
    assert float(np.max(diffs)) <= 0.25


def test_survival_curve_centers_sorted_ascending(tmp_path: Path) -> None:
    """Bin centers are returned in ascending d_L order (a proper survival x-axis)."""
    _write_injection_csv(str(tmp_path / "injection_h_0p73_task_0.csv"), n=800, seed=7)
    df = pd.read_csv(str(tmp_path / "injection_h_0p73_task_0.csv"))
    centers, _ = _pdet_survival_curve(
        df["luminosity_distance"].to_numpy(dtype=np.float64),
        df["SNR"].to_numpy(dtype=np.float64),
        snr_threshold=20.0,
        n_bins=10,
    )
    assert np.all(np.diff(centers) > 0)


def test_selection_explainer_has_horizon_contour_when_pdet_crosses_half(
    tmp_path: Path,
) -> None:
    """Right (heatmap) panel carries >= 1 horizon contour collection when p_det>=0.5.

    Reuses the fig20 assertion pattern: a contour collection is any collection
    on the heatmap axes that is NOT the QuadMesh from the pcolormesh.
    """
    _write_injection_csv(str(tmp_path / "injection_h_0p73_task_0.csv"), n=900, seed=3)
    _write_injection_csv(str(tmp_path / "injection_h_0p73_task_1.csv"), n=900, seed=4)
    _, axes = plot_selection_function_explainer(
        str(tmp_path / "injection_h_0p73_task_*.csv"), snr_threshold=20.0
    )
    flat = [a for a in np.atleast_1d(np.asarray(axes, dtype=object)).ravel() if isinstance(a, Axes)]
    # The heatmap panel is the one carrying a QuadMesh.
    heatmap_axes = [a for a in flat if any(isinstance(c, QuadMesh) for c in a.collections)]
    assert heatmap_axes, "expected a heatmap panel with a QuadMesh"
    ax_heat = heatmap_axes[0]
    contour_colls = [c for c in ax_heat.collections if not isinstance(c, QuadMesh)]
    assert len(contour_colls) >= 1, "expected a P_det horizon contour collection"


def test_selection_explainer_uses_double_preset_size(tmp_path: Path) -> None:
    """Figure size equals the get_figure 'double' preset (no hardcoded figsize)."""
    _write_injection_csv(str(tmp_path / "injection_h_0p73_task_0.csv"), n=500, seed=5)
    fig, _ = plot_selection_function_explainer(
        str(tmp_path / "injection_h_0p73_task_*.csv"), snr_threshold=20.0
    )
    w, h = fig.get_size_inches()
    expected_w, expected_h = _PRESETS["double"]
    assert np.isclose(w, expected_w) and np.isclose(h, expected_h), (
        f"figure size ({w}, {h}) != double preset {_PRESETS['double']}"
    )


def test_selection_explainer_no_horizon_does_not_raise(tmp_path: Path) -> None:
    """A faint/distant CSV that never reaches p_det=0.5 still renders without error."""
    _write_injection_csv(str(tmp_path / "injection_h_0p73_task_0.csv"), n=80, seed=6, horizon=False)
    fig, _ = plot_selection_function_explainer(
        str(tmp_path / "injection_h_0p73_task_*.csv"), snr_threshold=20.0
    )
    assert isinstance(fig, Figure)
