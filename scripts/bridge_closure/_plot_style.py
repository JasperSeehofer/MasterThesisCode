"""Shared paper-grade matplotlib style for the bridge-closure figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
    }
)

TRUTH_COLOR = "#444444"
RAIL_COLOR = "#c0392b"
OK_COLOR = "#2c7fb8"
