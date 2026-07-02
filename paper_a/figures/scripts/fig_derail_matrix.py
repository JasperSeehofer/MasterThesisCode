"""Flagship de-rail figure for Paper A (fig:derail).

Combined H0 posterior p(h) on the 7-point evaluation grid for each
completion-term normalization mode, real GLADE+ data (seed600 CRB
subsample, 494 events, 491 with usable posteriors), injected truth
h = 0.73:

===============  =============================================  =====  ==========
mode             estimator                                      MAP    verdict
===============  =============================================  =====  ==========
prod             production pre-4pi (peak-density completion,   0.86   railed up
                 global denominator, bare-Gaussian numerator)
prod_global      + isotropic 1/(4pi) completion sky-marginal    0.60   railed down
                 only (commit cb16142)                                 (sign flip)
local_ratio      + Gray A.9/A.10 local ratio-of-sums            0.73   peaked
                 (fix #2, commit 6d4c4e1)
volume_deconv    + dV_c/(1+z) host-z prior deconvolution        0.73   peaked
                 (fix #1, commit 6d4c4e1)
catonly          diagnostic: completion term dropped,           0.73   peaked
                 local self-normalized ratio
===============  =============================================  =====  ==========

Data sources (schema: combined_posterior.json with keys ``h_values``,
``posterior`` — unit-sum-normalized over the grid — and ``map_h``):

- results/commission_20260701/redteam/posteriors_per_mode/<mode>/combined_posterior.json
- cross-checked against results/commission_20260701/redteam/derail_matrix_results.json
  and crux_results.json / crux_results_fixed.json (same posteriors, MAP/mean/edge_mass)
- provenance: results/commission_20260701/redteam/posteriors_per_mode/README.md

Output: paper_a/figures/fig_derail_matrix.pdf  (label fig:derail)

Run from the repo root:

    .venv/bin/python paper_a/figures/scripts/fig_derail_matrix.py
"""

import json
import sys
from pathlib import Path
from typing import Any

# Repo root = three levels up from this file (paper_a/figures/scripts/).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.plotting._colors import TRUTH  # noqa: E402
from master_thesis_code.plotting._helpers import get_figure, save_figure  # noqa: E402
from master_thesis_code.plotting._style import apply_style  # noqa: E402

DATA_DIR = REPO_ROOT / "results" / "commission_20260701" / "redteam" / "posteriors_per_mode"
OUTPUT_STEM = REPO_ROOT / "paper_a" / "figures" / "fig_derail_matrix"

TRUTH_H = 0.73  # injected simulation truth (README.md in DATA_DIR)

# Draw order: pathological modes first (dashed), then the two principled
# fixes (solid / dash-dot), then the catalogue-only diagnostic (dotted).
# Colors are Okabe-Ito (colorblind-safe); linestyle + marker carry the
# distinction in grayscale.  MAP values (annotated in the panel) come from
# each mode's combined_posterior.json ("map_h").
MODES: list[dict[str, Any]] = [
    {
        "dir": "prod",
        "label": "production (pre-$4\\pi$)",
        "color": "#D55E00",  # vermillion
        "linestyle": (0, (4, 1.5)),
        "marker": "o",
    },
    {
        "dir": "prod_global",
        "label": "$+\\,1/(4\\pi)$ sky-marginal only",
        "color": "#E69F00",  # orange
        "linestyle": (0, (2, 1)),
        "marker": "s",
    },
    {
        "dir": "local_ratio",
        "label": "local ratio-of-sums",
        "color": "#0072B2",  # blue
        "linestyle": "-",
        "marker": "D",
    },
    {
        "dir": "volume_deconv",
        "label": "volume deconvolution",
        "color": "#56B4E9",  # sky blue
        "linestyle": "-.",
        "marker": "^",
    },
    {
        "dir": "catonly",
        "label": "catalogue-only (diagnostic)",
        "color": "#CC79A7",  # reddish purple
        "linestyle": ":",
        "marker": "v",
    },
]


def load_mode_posterior(mode_dir: str) -> tuple[list[float], list[float], float]:
    """Return ``(h_values, posterior, map_h)`` for one normalization mode.

    The posterior is stored unit-sum-normalized over the 7-point h grid.
    """
    path = DATA_DIR / mode_dir / "combined_posterior.json"
    with open(path) as f:
        data = json.load(f)
    return data["h_values"], data["posterior"], data["map_h"]


def main() -> None:
    apply_style()

    # Slightly taller than the "single" preset so the two-column legend
    # above the axes does not squeeze the panel.
    fig, ax = get_figure(figsize=(3.375, 2.7))

    for mode in MODES:
        h_values, posterior, map_h = load_mode_posterior(mode["dir"])
        ax.plot(
            h_values,
            posterior,
            color=mode["color"],
            linestyle=mode["linestyle"],
            marker=mode["marker"],
            markersize=3.2,
            linewidth=1.2,
            label=mode["label"],
            clip_on=False,
        )

    # Injected truth h = 0.73 (simulation truth, not a measurement of nature).
    ax.axvline(
        TRUTH_H,
        color=TRUTH,
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        zorder=0,
        label=f"truth $h = {TRUTH_H}$",
    )

    # MAP annotations (values from combined_posterior.json "map_h" per mode):
    # prod rails at 0.86, prod_global flips to 0.60, the fixes and the
    # catalogue-only diagnostic peak at the injected truth 0.73.
    # src: results/commission_20260701/redteam/posteriors_per_mode/prod/combined_posterior.json
    ax.annotate("MAP 0.86", xy=(0.852, 0.94), ha="right", fontsize=6, color="#D55E00")
    # src: results/commission_20260701/redteam/posteriors_per_mode/prod_global/combined_posterior.json
    ax.annotate("MAP 0.60", xy=(0.608, 0.94), ha="left", fontsize=6, color="#E69F00")
    # src: {local_ratio,volume_deconv,catonly}/combined_posterior.json (all MAP 0.73)
    ax.annotate("MAP 0.73", xy=(0.7385, 0.90), ha="left", fontsize=6, color="#1a1a1a")

    ax.set_xlabel("$h = H_0\\,/\\,(100\\ \\mathrm{km\\,s^{-1}\\,Mpc^{-1}})$")
    ax.set_ylabel("$p(h)$ (unit sum on $h$ grid)")
    ax.set_xlim(0.59, 0.87)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0.60, 0.65, 0.70, 0.73, 0.76, 0.80, 0.86])
    ax.tick_params(axis="x", labelsize=6.5)

    fig.legend(
        loc="outside upper center",
        ncols=2,
        fontsize=6.5,
        handlelength=2.4,
        columnspacing=1.2,
        handletextpad=0.5,
    )

    save_figure(fig, str(OUTPUT_STEM), formats=("pdf",))
    print(f"wrote {OUTPUT_STEM}.pdf")


if __name__ == "__main__":
    main()
