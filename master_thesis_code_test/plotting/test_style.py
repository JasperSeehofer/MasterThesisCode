"""Tests for the plotting subpackage foundation (Phase 1)."""

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting import apply_style, get_figure, save_figure


def test_apply_style_sets_agg_backend() -> None:
    apply_style()
    assert matplotlib.get_backend().lower() == "agg"


def test_get_figure_returns_figure_and_axes() -> None:
    apply_style()
    fig, ax = get_figure()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_get_figure_with_subplots() -> None:
    apply_style()
    fig, axes = get_figure(nrows=2, ncols=2)
    assert isinstance(fig, Figure)
    # 2x2 subplots returns a 2D array of Axes
    assert len(axes) == 2
    plt.close(fig)


def test_get_figure_custom_figsize() -> None:
    apply_style()
    fig, ax = get_figure(figsize=(10, 5))
    width, height = fig.get_size_inches()
    assert abs(width - 10) < 0.01
    assert abs(height - 5) < 0.01
    plt.close(fig)


def test_save_figure_writes_pdf(tmp_path: object) -> None:
    apply_style()
    fig, ax = get_figure()
    ax.plot([0, 1], [0, 1])
    output = os.path.join(str(tmp_path), "test_output")
    save_figure(fig, output, formats=("pdf",))
    assert os.path.isfile(f"{output}.pdf")


def test_save_figure_writes_multiple_formats(tmp_path: object) -> None:
    apply_style()
    fig, ax = get_figure()
    ax.plot([0, 1], [0, 1])
    output = os.path.join(str(tmp_path), "test_output")
    save_figure(fig, output, formats=("pdf", "png"))
    assert os.path.isfile(f"{output}.pdf")
    assert os.path.isfile(f"{output}.png")


def test_save_figure_creates_parent_dirs(tmp_path: object) -> None:
    apply_style()
    fig, ax = get_figure()
    output = os.path.join(str(tmp_path), "nested", "dir", "fig")
    save_figure(fig, output, formats=("png",))
    assert os.path.isfile(f"{output}.png")


def test_save_figure_closes_by_default(tmp_path: object) -> None:
    apply_style()
    fig, ax = get_figure()
    output = os.path.join(str(tmp_path), "test_close")
    save_figure(fig, output, formats=("png",))
    # After close, the figure should not be in the pyplot figure list
    assert fig not in [plt.figure(i) for i in plt.get_fignums()]


def test_save_figure_keep_open(tmp_path: object) -> None:
    apply_style()
    fig, ax = get_figure()
    output = os.path.join(str(tmp_path), "test_keep")
    save_figure(fig, output, formats=("png",), close=False)
    assert fig.number in plt.get_fignums()
    plt.close(fig)


def test_style_sheet_sets_chunksize() -> None:
    apply_style()
    assert matplotlib.rcParams["agg.path.chunksize"] == 10000
