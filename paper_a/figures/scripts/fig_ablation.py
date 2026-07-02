"""Paper A figure ``fig:ablation`` -- G3 ablation-cube two-factor decomposition.

Matrix of combined H0 posteriors on the real 494-event seed600 GLADE+ subsample,
one panel per {numerator host-z kernel} x {L_cat denominator} cell of the G3
soundness-gate ablation cube (all cells carry the isotropic 1/(4pi) completion
sky-marginal, the third cube factor; the pre-4pi baseline railed at h = 0.86,
see results/commission_20260701/redteam/posteriors_per_mode/README.md).

Cell -> data-key mapping (from scripts/ablation_cube_seed600.py):

    row (kernel)        col (denominator)      key             MAP
    bare Gaussian       global sum             prod_global     0.60 (railed)
    bare Gaussian       local ratio-of-sums    local_ratio     0.73
    bare Gaussian       catalogue-only (ctrl)  catonly         0.73
    volume-deconvolved  global sum             volume_global   0.76
    volume-deconvolved  local ratio-of-sums    volume_deconv   0.73 (production)
    volume-deconvolved  catalogue-only         --              not run

Arrows trace the two-factor de-rail path: volume kernel 0.60 -> 0.76,
local denominator 0.76 -> 0.73.

Data source: .planning/gate/G3_ablation_cube.json (no numbers are hardcoded;
MAPs and posterior masses are read from the artifact at run time).

Usage (from repo root):
    .venv/bin/python paper_a/figures/scripts/fig_ablation.py

Output: paper_a/figures/fig_ablation.pdf
"""

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.plotting._style import apply_style  # noqa: E402

apply_style()

from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

from master_thesis_code.plotting._colors import EDGE, METHOD, TRUTH  # noqa: E402
from master_thesis_code.plotting._helpers import get_figure, save_figure  # noqa: E402

DATA_PATH = REPO_ROOT / ".planning" / "gate" / "G3_ablation_cube.json"
OUTPUT_STEM = REPO_ROOT / "paper_a" / "figures" / "fig_ablation"

H_TRUTH = 0.73  # injected simulation truth (notation contract)

BAR_COLOR = METHOD["dark"]  # Okabe-Ito blue -- dark-siren grammar
RAIL_COLOR = "#D55E00"  # Okabe-Ito vermillion -- failure flag
GREY = "#7a7a7a"

# (row, col) -> (json key, annotation lines, annotation corner)
CELLS: dict[tuple[int, int], tuple[str, list[str], str]] = {
    (0, 0): ("prod_global", ["railed"], "right"),
    (0, 1): ("local_ratio", [], "left"),
    (0, 2): ("catonly", [], "left"),
    (1, 0): ("volume_global", [], "left"),
    (1, 1): ("volume_deconv", ["production"], "left"),
}

COL_TITLES = [
    "global\ndenominator",
    "local ratio-of-sums\ndenominator",
    "catalogue-only\n(control)",
]
ROW_LABELS = [
    "bare Gaussian\nhost-$z$ kernel",
    "volume-deconvolved\nhost-$z$ kernel",
]

XLIM = (0.575, 0.885)
YLIM = (0.0, 1.14)
XTICKS = [0.60, 0.73, 0.86]


def _draw_cell(ax: Axes, entry: dict[str, Any], notes: list[str], corner: str) -> None:
    """Bar-strip of normalized posterior mass vs h for one ablation cell."""
    h = entry["h_values"]
    p = entry["posterior"]
    railed = bool(entry["railed"])
    color = RAIL_COLOR if railed else BAR_COLOR
    hatch = "///" if railed else None  # grayscale-safe failure flag
    ax.bar(
        h,
        p,
        width=0.018,
        color=color,
        edgecolor="white" if railed else color,
        linewidth=0.3 if railed else 0.0,
        hatch=hatch,
    )
    ax.axvline(H_TRUTH, color=TRUTH, linestyle="--", linewidth=0.7, zorder=0)

    lines = [f"MAP {entry['MAP']:.2f}"] + notes
    if corner == "right":
        x, ha = 0.96, "right"
    else:
        x, ha = 0.05, "left"
    ax.text(
        x,
        0.99,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=6.0,
        color=EDGE,
        linespacing=1.25,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6),
    )


def main() -> None:
    cube = json.loads(DATA_PATH.read_text())

    fig, axes = get_figure(
        2,
        3,
        figsize=(3.32, 2.45),
        sharex=True,
        sharey=True,
    )
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(hspace=0.10, wspace=0.14, h_pad=0.02, w_pad=0.02)  # type: ignore[call-arg]

    for (row, col), (key, notes, corner) in CELLS.items():
        _draw_cell(axes[row, col], cube[key], notes, corner)

    # Empty cube cell (volume kernel x catalogue-only was not run); reuse the
    # dead space to state the third, constant cube factor (sky-marginal).
    ax_empty = axes[1, 2]
    ax_empty.axis("off")
    ax_empty.text(
        0.47,
        0.80,
        "not run",
        transform=ax_empty.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        style="italic",
        color=GREY,
    )
    ax_empty.text(
        0.47,
        0.36,
        "all panels:\nisotropic $1/(4\\pi)$\ncompletion\nsky-marginal",
        transform=ax_empty.transAxes,
        ha="center",
        va="center",
        fontsize=5.6,
        color=GREY,
        linespacing=1.35,
    )

    for row in range(2):
        for col in range(3):
            ax = axes[row, col]
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)
            ax.set_xticks(XTICKS)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=6, length=2.5)
            if col > 0:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(COL_TITLES[col], fontsize=6.5, pad=3, linespacing=1.2)
                # bottom-row neighbour of col 2 is switched off -> show the
                # x tick labels on the top-row panel instead
                ax.tick_params(labelbottom=(col == 2))

    # Row (kernel-factor) labels on the right margin
    for row, label in enumerate(ROW_LABELS):
        axes[row, 2].text(
            1.14,
            0.5,
            label,
            transform=axes[row, 2].transAxes,
            rotation=-90,
            ha="center",
            va="center",
            fontsize=6.5,
            linespacing=1.2,
        )

    # Two-factor de-rail path: volume kernel (down), then local denominator (right)
    arrow_style = dict(arrowstyle="-|>", color=EDGE, linewidth=1.0, mutation_scale=8)
    fig.add_artist(
        ConnectionPatch(
            xyA=(0.5, -0.04),
            coordsA=axes[0, 0].transAxes,
            xyB=(0.5, 1.04),
            coordsB=axes[1, 0].transAxes,
            **arrow_style,
        )
    )
    fig.add_artist(
        ConnectionPatch(
            xyA=(1.03, 0.5),
            coordsA=axes[1, 0].transAxes,
            xyB=(-0.03, 0.5),
            coordsB=axes[1, 1].transAxes,
            **arrow_style,
        )
    )

    fig.supxlabel(
        r"$h = H_0 / (100\ \mathrm{km\,s^{-1}\,Mpc^{-1}})$",
        fontsize=7,
    )
    fig.supylabel("normalized posterior mass", fontsize=7)

    save_figure(fig, str(OUTPUT_STEM), formats=("pdf",))
    print(f"wrote {OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
