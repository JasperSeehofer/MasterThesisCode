"""Tests for the bilby-style PP-plot / coverage figure (fig24, VR-NEW-02).

The PP-plot draws nested grey 1/2/3-sigma binomial bands, per-parameter
cumulative empirical-CDF lines, and KS p-values. Calibrated synthetic ranks must
fall INSIDE the bands (the required calibration assertion); mis-calibrated ranks
must exit them (opposite-direction sanity). The ranks are data-gated
(``DEFAULT_PP_PARAMS`` + ``load_pp_ranks``), falling back to synthetic when no
real injection-recovery campaign is present.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _PRESETS
from master_thesis_code.plotting.pp_plot import (
    DEFAULT_PP_PARAMS,
    binomial_confidence_bands,
    load_pp_ranks,
    make_synthetic_ranks,
    plot_pp_coverage,
)


def _cdf_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical CDF of *values* evaluated on *grid* in [0, 1]."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    return np.searchsorted(v, grid, side="right").astype(np.float64) / v.size


def test_pp_returns_figure_and_axes() -> None:
    """Smoke: PP-plot returns (Figure, Axes) on synthetic calibrated ranks."""
    ranks = make_synthetic_ranks(300, ["M", "mu", "d_L"], calibrated=True, seed=1)
    fig, ax = plot_pp_coverage(ranks)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_calibrated_ranks_inside_bands() -> None:
    """Calibrated ranks' empirical CDF lies inside the 3-sigma band almost everywhere.

    Required calibration assertion (success criterion 2): for each parameter, the
    fraction of grid points where the empirical CDF is within [lower, upper] of
    the 3-sigma binomial band is >= 0.99 (allowing rare boundary noise).
    """
    n = 2000
    ranks = make_synthetic_ranks(n, list(DEFAULT_PP_PARAMS), calibrated=True, seed=7)
    grid = np.linspace(0.0, 1.0, 101)
    bands = binomial_confidence_bands(n, n_grid=101)
    lower, upper = bands[0.997]  # 3-sigma envelope

    for name, values in ranks.items():
        cdf = _cdf_on_grid(values, grid)
        inside = (cdf >= lower - 1e-9) & (cdf <= upper + 1e-9)
        frac_inside = float(np.mean(inside))
        assert frac_inside >= 0.99, f"{name}: only {frac_inside:.3f} of grid inside 3-sigma band"


def test_ks_pvalue_computed_and_finite() -> None:
    """Per-parameter KS p-values are finite in [0,1]; calibrated combined p is non-tiny."""
    from scipy.stats import kstest

    ranks = make_synthetic_ranks(1000, list(DEFAULT_PP_PARAMS), calibrated=True, seed=3)
    for values in ranks.values():
        p = float(kstest(np.sort(values), "uniform").pvalue)
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0

    pooled = np.concatenate([np.asarray(v) for v in ranks.values()])
    p_all = float(kstest(pooled, "uniform").pvalue)
    assert p_all > 0.01, f"calibrated combined KS p unexpectedly tiny: {p_all}"


def test_miscalibrated_ranks_fall_outside_bands() -> None:
    """Mis-calibrated ranks: at least one parameter's CDF exits the 3-sigma band."""
    n = 2000
    ranks = make_synthetic_ranks(n, list(DEFAULT_PP_PARAMS), calibrated=False, seed=11)
    grid = np.linspace(0.0, 1.0, 101)
    bands = binomial_confidence_bands(n, n_grid=101)
    lower, upper = bands[0.997]

    any_outside = False
    for values in ranks.values():
        cdf = _cdf_on_grid(values, grid)
        if np.any((cdf < lower - 1e-9) | (cdf > upper + 1e-9)):
            any_outside = True
            break
    assert any_outside, "mis-calibrated ranks should exit the 3-sigma band somewhere"


def test_pp_uses_single_preset_size() -> None:
    """Figure size equals the get_figure 'single' preset (no hardcoded figsize)."""
    ranks = make_synthetic_ranks(200, ["M", "mu"], calibrated=True, seed=2)
    fig, _ = plot_pp_coverage(ranks)
    w, h = fig.get_size_inches()
    expected_w, expected_h = _PRESETS["single"]
    assert np.isclose(w, expected_w) and np.isclose(h, expected_h), (
        f"figure size ({w}, {h}) != single preset {_PRESETS['single']}"
    )


def test_data_gate_loader_falls_back_to_synthetic() -> None:
    """``load_pp_ranks(None)`` returns synthetic ranks over DEFAULT_PP_PARAMS."""
    ranks = load_pp_ranks(None)
    assert set(ranks.keys()) == set(DEFAULT_PP_PARAMS)
    for values in ranks.values():
        arr = np.asarray(values, dtype=np.float64)
        assert arr.size > 0
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0


def test_data_gate_loader_missing_file_falls_back(tmp_path: object) -> None:
    """A data_dir without ranks.json yields synthetic ranks (no raise)."""
    from pathlib import Path

    ranks = load_pp_ranks(Path(str(tmp_path)))
    assert set(ranks.keys()) == set(DEFAULT_PP_PARAMS)


def test_data_gate_loader_reads_real_ranks(tmp_path: object) -> None:
    """When ranks.json is present and valid, load_pp_ranks reads it (gate auto-closes)."""
    import json
    from pathlib import Path

    d = Path(str(tmp_path))
    (d / "injection_recovery").mkdir(parents=True)
    payload = {"M": [0.1, 0.5, 0.9], "d_L": [0.2, 0.4, 0.6]}
    with open(d / "injection_recovery" / "ranks.json", "w") as f:
        json.dump(payload, f)
    ranks = load_pp_ranks(d)
    assert set(ranks.keys()) == {"M", "d_L"}
    assert np.allclose(np.sort(ranks["M"]), [0.1, 0.5, 0.9])


def test_data_gate_loader_malformed_ranks_falls_back(tmp_path: object) -> None:
    """Malformed ranks.json falls back to synthetic rather than plotting garbage (T-04-02)."""
    from pathlib import Path

    d = Path(str(tmp_path))
    (d / "injection_recovery").mkdir(parents=True)
    with open(d / "injection_recovery" / "ranks.json", "w") as f:
        f.write("{ not valid json")
    ranks = load_pp_ranks(d)
    assert set(ranks.keys()) == set(DEFAULT_PP_PARAMS)


def test_binomial_bands_are_nested() -> None:
    """3-sigma band contains 2-sigma contains 1-sigma at every grid point."""
    bands = binomial_confidence_bands(500, n_grid=51)
    lo1, hi1 = bands[0.68]
    lo2, hi2 = bands[0.95]
    lo3, hi3 = bands[0.997]
    assert np.all(lo3 <= lo2 + 1e-9) and np.all(hi3 >= hi2 - 1e-9)
    assert np.all(lo2 <= lo1 + 1e-9) and np.all(hi2 >= hi1 - 1e-9)
