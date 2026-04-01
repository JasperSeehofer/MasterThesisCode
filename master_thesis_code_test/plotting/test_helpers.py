"""Tests for _helpers.py: figure presets, figsize override, _fig_from_ax."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting import apply_style, get_figure
from master_thesis_code.plotting._helpers import _fig_from_ax


def test_get_figure_preset_single_width() -> None:
    apply_style()
    fig, ax = get_figure(preset="single")
    w, _ = fig.get_size_inches()
    assert abs(w - 3.375) < 0.01, f"single preset width {w}, expected ~3.375"
    plt.close(fig)


def test_get_figure_preset_double_width() -> None:
    apply_style()
    fig, ax = get_figure(preset="double")
    w, _ = fig.get_size_inches()
    assert abs(w - 7.0) < 0.01, f"double preset width {w}, expected ~7.0"
    plt.close(fig)


def test_get_figure_no_args_uses_mplstyle_default() -> None:
    apply_style()
    fig, ax = get_figure()
    w, h = fig.get_size_inches()
    assert abs(w - 6.4) < 0.01, f"default width {w}, expected 6.4"
    assert abs(h - 4.0) < 0.01, f"default height {h}, expected 4.0"
    plt.close(fig)


def test_get_figure_figsize_overrides_preset() -> None:
    apply_style()
    fig, ax = get_figure(figsize=(10, 5), preset="single")
    w, h = fig.get_size_inches()
    assert abs(w - 10) < 0.01, f"figsize width {w}, expected 10"
    assert abs(h - 5) < 0.01, f"figsize height {h}, expected 5"
    plt.close(fig)


def test_get_figure_explicit_figsize() -> None:
    apply_style()
    fig, ax = get_figure(figsize=(10, 5))
    w, h = fig.get_size_inches()
    assert abs(w - 10) < 0.01
    assert abs(h - 5) < 0.01
    plt.close(fig)


def test_fig_from_ax_round_trip() -> None:
    apply_style()
    fig, ax = get_figure()
    assert isinstance(ax, Axes)
    recovered = _fig_from_ax(ax)
    assert isinstance(recovered, Figure)
    assert recovered is fig
    plt.close(fig)


def test_fig_from_ax_importable_from_helpers() -> None:
    """Verify _fig_from_ax is importable from _helpers (not just simulation_plots)."""
    from master_thesis_code.plotting._helpers import _fig_from_ax as fn

    assert callable(fn)
