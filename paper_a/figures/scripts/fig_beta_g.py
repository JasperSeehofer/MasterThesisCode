"""Paper A figure: beta_G discrete-sum consistency check (fig:betag).

Visualizes the G1 gate result (.planning/gate/G1_beta_g_check.json): the
h-dependence of the 'global'-mode in-catalogue normalization ratio
Sigma_global / beta_G, normalized to h_ref.  Panel (a) shows the raw shape
against the expected n_gal ~ h^3 scaling; panel (b) shows the residual after
dividing out h^3 — a real ~ -17% end-to-end tilt remains, i.e. the global
mode's in-catalogue channel is distorted.  Local modes carry no Sigma_global
factor and are immune by construction (unity line).

Run from the repository root:

    .venv/bin/python paper_a/figures/scripts/fig_beta_g.py

Output: paper_a/figures/fig_beta_g.pdf (vector).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from master_thesis_code.plotting import apply_style, get_figure, save_figure
from master_thesis_code.plotting._colors import METHOD, PRIOR

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / ".planning" / "gate" / "G1_beta_g_check.json"
OUTPUT_STEM = REPO_ROOT / "paper_a" / "figures" / "fig_beta_g"

# Single-column MNRAS-ish width; two stacked panels.
FIGSIZE = (3.375, 3.9)


def load_g1_table(
    path: Path,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    float,
    dict[str, Any],
]:
    """Load the G1 beta_G check table.

    Returns
    -------
    h:
        Sorted grid of dimensionless Hubble parameters h.
    shape_raw:
        Raw normalization shape S(h) = R(h)/R(h_ref) with
        R = Sigma_global / beta_G.
    shape_h3:
        h^3-corrected shape S(h) * (h_ref/h)^3.
    h_ref:
        Reference h at which the shapes are pinned to unity.
    meta:
        Full JSON payload (for the summary tilt numbers).
    """
    with open(path) as f:
        payload: dict[str, Any] = json.load(f)

    table: dict[str, dict[str, float]] = payload["table"]
    h = np.array(sorted(float(k) for k in table), dtype=np.float64)
    shape_raw = np.array([table[f"{hv:.4f}"]["shape_vs_href"] for hv in h], dtype=np.float64)
    shape_h3 = np.array([table[f"{hv:.4f}"]["shape_h3_corrected"] for hv in h], dtype=np.float64)
    h_ref = float(payload["h_ref"])
    return h, shape_raw, shape_h3, h_ref, payload


def make_figure() -> None:
    """Build and save fig_beta_g.pdf."""
    apply_style()

    h, shape_raw, shape_h3, h_ref, meta = load_g1_table(DATA_PATH)
    tilt_h3 = float(meta["end_to_end_tilt_h3_corrected"])  # -0.1719 (-17.2%)

    fig, (ax_raw, ax_res) = get_figure(nrows=2, ncols=1, figsize=FIGSIZE, sharex=True)

    dark = METHOD["dark"]  # Okabe-Ito blue: galaxy-catalogue (dark siren) grammar

    # --- Panel (a): raw shape vs expected h^3 galaxy-count scaling ----------
    expected_h3 = (h / h_ref) ** 3
    ax_raw.plot(
        h,
        expected_h3,
        color="0.15",
        linestyle="--",
        linewidth=1.2,
        label=r"expected $n_{\mathrm{gal}} \propto h^{3}$",
        zorder=2,
    )
    ax_raw.plot(
        h,
        shape_raw,
        color=dark,
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label=r"global mode: $\Sigma_{\mathrm{global}}/\beta_G$",
        zorder=3,
    )
    ax_raw.set_ylabel(r"normalization shape $S(h)$")
    ax_raw.legend(loc="upper left", handlelength=1.8)

    # --- Panel (b): residual after dividing out h^3 -------------------------
    ax_res.axhline(
        1.0,
        color=PRIOR,
        linestyle="--",
        linewidth=1.2,
        label="prior-consistent (local modes)",
        zorder=2,
    )
    ax_res.plot(
        h,
        shape_h3,
        color=dark,
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="global mode, $h^{3}$ removed",
        zorder=3,
    )
    # End-to-end residual tilt annotation (value straight from the artifact).
    ax_res.text(
        0.735,
        1.05,
        rf"${100.0 * tilt_h3:+.1f}\%$ end-to-end",
        color="0.25",
        fontsize=7,
        ha="left",
        va="bottom",
    )
    ax_res.set_xlabel(r"$h = H_0 / (100\ \mathrm{km\,s^{-1}\,Mpc^{-1}})$")
    ax_res.set_ylabel(r"residual $S(h)\,(h_{\mathrm{ref}}/h)^{3}$")
    ax_res.legend(loc="lower left", handlelength=1.8)

    # --- Shared cosmetics ----------------------------------------------------
    for ax, letter, (lx, ly, va) in (
        (ax_raw, "(a)", (0.97, 0.06, "bottom")),
        (ax_res, "(b)", (0.97, 0.94, "top")),
    ):
        ax.axvline(h_ref, color="0.85", linewidth=0.8, zorder=0)
        ax.text(
            lx,
            ly,
            letter,
            transform=ax.transAxes,
            ha="right",
            va=va,
            fontsize=8,
        )
    ax_raw.text(
        h_ref,
        ax_raw.get_ylim()[1],
        r"$h_{\mathrm{ref}}$",
        color="0.55",
        fontsize=7,
        ha="center",
        va="top",
    )
    ax_res.set_xlim(h[0] - 0.01, h[-1] + 0.01)

    save_figure(fig, str(OUTPUT_STEM), formats=("pdf",))


if __name__ == "__main__":
    make_figure()
