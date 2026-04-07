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


def test_apply_style_latex_mode() -> None:
    """apply_style(use_latex=True) enables usetex and serif fonts."""
    apply_style(use_latex=True)
    assert matplotlib.rcParams["text.usetex"] is True
    assert "serif" in matplotlib.rcParams["font.family"]
    # Reset to default and verify usetex is back to False
    apply_style()
    assert matplotlib.rcParams["text.usetex"] is False


def test_apply_style_default_unchanged() -> None:
    """apply_style() with no args matches the mplstyle defaults exactly."""
    apply_style()
    assert matplotlib.rcParams["text.usetex"] is False
    assert matplotlib.rcParams["font.size"] == 8.0
    assert matplotlib.rcParams["axes.titlesize"] == 9.0
    assert matplotlib.rcParams["axes.labelsize"] == 8.0
    assert matplotlib.rcParams["xtick.labelsize"] == 7.0
    assert matplotlib.rcParams["ytick.labelsize"] == 7.0
    assert matplotlib.rcParams["legend.fontsize"] == 7.0


def test_rcparams_snapshot() -> None:
    """Regression test: pin all 18 emri_thesis.mplstyle rcParams.

    If any value changes, this test fails — forcing an intentional update
    to both the .mplstyle file and this test.
    """
    apply_style()

    expected: dict[str, object] = {
        "figure.figsize": [6.4, 4.0],
        "figure.dpi": 150.0,
        "savefig.dpi": 300.0,
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "lines.linewidth": 1.5,
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "agg.path.chunksize": 10000,
        "image.cmap": "viridis",
        "figure.constrained_layout.use": True,
    }

    for key, expected_value in expected.items():
        actual = matplotlib.rcParams[key]
        if isinstance(expected_value, list):
            assert list(actual) == expected_value, (
                f"rcParam '{key}' drifted: expected {expected_value}, got {list(actual)}"
            )
        else:
            assert actual == expected_value, (
                f"rcParam '{key}' drifted: expected {expected_value}, got {actual}"
            )


def test_no_type3_fonts_in_pdf(tmp_path: object) -> None:
    """Generated PDFs must not contain Type 3 bitmap fonts."""
    import shutil
    import subprocess

    if shutil.which("pdffonts") is None:
        import pytest

        pytest.skip("pdffonts binary not available")

    apply_style()
    fig, ax = get_figure()
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.set_title("title")
    output = os.path.join(str(tmp_path), "font_check.pdf")
    fig.savefig(output)
    plt.close(fig)

    result = subprocess.run(["pdffonts", output], capture_output=True, text=True, check=False)
    assert "Type 3" not in result.stdout, f"Type 3 fonts found in PDF:\n{result.stdout}"
