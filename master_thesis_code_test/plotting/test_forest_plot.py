"""Tests for the H0 forest / tension figure (fig23, VR-NEW-01).

The forest renders one point + asymmetric 68% CI row per published H0
measurement, grouped early (top) vs late (bottom), with full-height Planck and
SH0ES reference bands and a bold, visually-distinct this-work row. The this-work
number is data-gated (``THIS_WORK_H0`` + ``load_this_work_h0``); the loader
falls back to the placeholder when no posterior is present.
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from master_thesis_code.plotting._colors import VARIANT_NO_MASS
from master_thesis_code.plotting._labels import LABELS
from master_thesis_code.plotting.forest_plot import (
    LITERATURE_H0,
    THIS_WORK_H0,
    Measurement,
    load_this_work_h0,
    plot_h0_forest,
)


def test_forest_returns_figure_and_axes() -> None:
    """Smoke: forest returns (Figure, Axes) without raising on defaults."""
    fig, ax = plot_h0_forest()
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_this_work_row_is_distinct() -> None:
    """The this-work errorbar marker is larger AND navy vs a literature row."""
    fig, ax = plot_h0_forest()
    # errorbar produces ErrorbarContainer objects; each carrline holds the data
    # marker as its first Line2D ([0]).
    containers = list(ax.containers)
    assert len(containers) >= 2, "expected one errorbar container per row"

    sizes: list[float] = []
    this_work_size: float | None = None
    lit_size: float | None = None
    for c in containers:
        line = c[0]  # data-point Line2D
        ms = float(line.get_markersize())
        sizes.append(ms)
        rgba = line.get_markerfacecolor()
        # Match the this-work navy color robustly via the hex token's RGB.
        from matplotlib.colors import to_rgba

        if np.allclose(to_rgba(rgba), to_rgba(VARIANT_NO_MASS), atol=1e-3):
            this_work_size = ms
        else:
            lit_size = ms

    assert this_work_size is not None, "no navy this-work marker found"
    assert lit_size is not None, "no literature marker found"
    assert this_work_size > lit_size, (
        f"this-work marker ({this_work_size}) not larger than literature ({lit_size})"
    )


def test_reference_bands_present() -> None:
    """>= 2 full-height axvspan patches (Planck + SH0ES) sit behind the rows."""
    fig, ax = plot_h0_forest()
    # axvspan adds Rectangle patches spanning the full y data-range (in axes
    # y-coords) at zorder=0. There should be >= 2 (Planck + SH0ES).
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) >= 2, f"expected >= 2 reference-band rectangles, got {len(rects)}"


def test_early_and_late_both_grouped() -> None:
    """Both eras appear and the early block sits above the late block.

    Compares the mean y-position of early literature rows to that of late rows;
    early (top) must have a larger y than late (bottom).
    """
    fig, ax = plot_h0_forest()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    ticks = list(ax.get_yticks())
    label_to_y = dict(zip(labels, ticks))

    early_labels = [m.label for m in LITERATURE_H0 if m.era == "early"]
    late_labels = [m.label for m in LITERATURE_H0 if m.era == "late"]
    assert early_labels, "no early-era literature rows"
    assert late_labels, "no late-era literature rows"

    early_y = np.mean([label_to_y[lbl] for lbl in early_labels])
    late_y = np.mean([label_to_y[lbl] for lbl in late_labels])
    assert early_y > late_y, f"early block (mean y={early_y}) not above late ({late_y})"


def test_xaxis_is_h0_not_dimensionless() -> None:
    """X-axis label is the H0 (km/s/Mpc) label, and this-work x ~ 100*h."""
    fig, ax = plot_h0_forest()
    assert ax.get_xlabel() == LABELS["H0"]

    # The this-work data point's x-coordinate must be ~73 (km/s/Mpc), not ~0.73.
    from matplotlib.colors import to_rgba

    this_work_x: float | None = None
    for c in ax.containers:
        line = c[0]
        if np.allclose(to_rgba(line.get_markerfacecolor()), to_rgba(VARIANT_NO_MASS), atol=1e-3):
            this_work_x = float(line.get_xdata()[0])
    assert this_work_x is not None
    assert this_work_x == 100.0 * THIS_WORK_H0.h
    assert this_work_x > 10.0, "this-work x looks dimensionless (h), not H0 km/s/Mpc"


def test_data_gate_loader_falls_back_to_placeholder() -> None:
    """``load_this_work_h0(None)`` returns the placeholder exactly."""
    assert load_this_work_h0(None) == THIS_WORK_H0


def test_data_gate_loader_missing_dir_falls_back(tmp_path: object) -> None:
    """A data_dir without posteriors still yields the placeholder (no raise)."""
    from pathlib import Path

    result = load_this_work_h0(Path(str(tmp_path)))
    assert result == THIS_WORK_H0


def test_literature_table_has_expected_canonical_values() -> None:
    """The hardcoded literature table matches the planned canonical values.

    Guards against a transcription error in the curated context (T-04-03).
    """
    by_label = {m.label: m for m in LITERATURE_H0}
    # (label, H0, +err, -err) in km/s/Mpc.
    expected: list[tuple[str, float, float, float]] = [
        ("Planck 2018", 67.36, 0.54, 0.54),
        ("DESI 2024 BAO+BBN", 68.52, 0.62, 0.62),
        ("SH0ES 2022", 73.04, 1.04, 1.04),
        ("TRGB (CCHP) 2021", 69.8, 1.9, 1.9),
        ("TDCOSMO+SLACS 2020", 67.4, 4.1, 3.2),
        ("GW170817 siren", 70.0, 12.0, 8.0),
        ("LVK GWTC-3 dark", 68.0, 8.0, 6.0),
    ]
    for label, h0, sp, sm in expected:
        m = by_label[label]
        assert np.isclose(100.0 * m.h, h0, atol=1e-6), label
        assert np.isclose(100.0 * m.sigma_plus, sp, atol=1e-6), label
        assert np.isclose(100.0 * m.sigma_minus, sm, atol=1e-6), label
        assert m.citation, f"{label} missing citation"


def test_measurement_is_namedtuple_with_fields() -> None:
    """Measurement carries the contract fields and is immutable."""
    m = Measurement(label="x", h=0.7, sigma_plus=0.01, sigma_minus=0.01, era="late", citation="c")
    assert m.label == "x"
    assert m.h == 0.7
    assert m.era == "late"
