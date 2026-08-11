"""Summary figure for the H0-railing bias-resolution investigation.

Produces ``docs/figures/bias_resolution_summary.png``, a two-panel summary of the
in-catalogue photo-z normalisation study:

  * Panel A -- candidate gate-vs-de-rail map. Each normalisation candidate is shown
    at the sigma_z -> 0 GATE (sigma_z = 0.002, open marker) and at the GLADE
    photometric DE-RAIL regime (sigma_z = 0.035, filled marker). The injected truth
    h = 0.73 sits strictly between the lower rail (0.60) and the upper rail (0.87):
    STANDARD rails DOWN, every numerator-only clean rails UP, the local consistent
    denominator fails the gate, and the global photo-z-smeared D_sm de-biases but
    produces no peak.

  * Panel B -- D_sm multi-seed scatter. Per-seed posterior MAP peaks (n_events =
    2000) land at 0.64, 0.64, 0.69, 0.87 -- never at the truth -- with std ~ 0.10
    that does not shrink with the event count. The ensemble mean E[h] ~ 0.735 is a
    grid-midpoint artefact, not a recovered peak.

Data sources:
  * ``scripts/bridge_closure/outputs/rungI_results.json`` -- STANDARD and
    local consistent-denominator gate/de-rail values (measured).
  * Canonical numbers from ``.planning/derivation-photoz-incatalog/
    INCREMENT3-DSM-VERDICT.md`` and ``NORMALISATION-FIX.md`` for the
    numerator-only cleans (Angle A/C, Angle B) and the global D_sm candidate
    (these were run in the prototype ``_rungI_verify_B.py`` and recorded in the
    derivation docs rather than emitted as standalone JSON).

Run with::

    uv run python scripts/bridge_closure/plot_bias_resolution_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OUTPUTS = _REPO_ROOT / "scripts" / "bridge_closure" / "outputs"
_STYLE = _REPO_ROOT / "darksiren_emri" / "plotting" / "emri_thesis.mplstyle"
_FIG_PATH = _REPO_ROOT / "docs" / "figures" / "bias_resolution_summary.png"

# --------------------------------------------------------------------------- #
# Canonical constants (injected truth and grid edges = the two rails)
# --------------------------------------------------------------------------- #
TRUTH = 0.73
RAIL_LOW = 0.60
RAIL_HIGH = 0.87
GATE_SIGMA = 0.002  # spec-z gate
DERAIL_SIGMA = 0.035  # GLADE flag-1 photometric

# Okabe-Ito palette (matches emri_thesis.mplstyle)
C_ORANGE = "#E69F00"
C_BLUE = "#56B4E9"
C_GREEN = "#009E73"
C_DARKBLUE = "#0072B2"
C_VERMILLION = "#D55E00"
C_PURPLE = "#CC79A7"


def _load_rungI() -> dict[str, float]:
    """Read measured STANDARD and consistent-denom gate/de-rail MAPs from JSON."""
    with (_OUTPUTS / "rungI_results.json").open() as fh:
        data = json.load(fh)

    out: dict[str, float] = {}
    for row in data["standard"]:
        if abs(row["sigma_z"] - GATE_SIGMA) < 1e-6:
            out["standard_gate"] = row["h_refined"]
        else:
            out["standard_derail"] = row["h_refined"]
    # consistent_denom gate run used sigma_z = 0.001; both entries rail to 0.87
    out["consistent_gate"] = data["consistent_denom"][0]["h_refined"]
    out["consistent_derail"] = data["consistent_denom"][1]["h_refined"]
    return out


def _candidate_table(rungI: dict[str, float]) -> list[dict[str, object]]:
    """Assemble the candidate gate/de-rail table (measured + canonical numbers)."""
    # Angle A/C, Angle B, D_sm: canonical numbers from NORMALISATION-FIX.md and
    # INCREMENT3-DSM-VERDICT.md (prototype _rungI_verify_B.py, commit 5ef8c6e).
    return [
        {
            "label": "STANDARD\n(bare Gaussian)",
            "gate": rungI["standard_gate"],  # 0.7438 measured
            "derail": rungI["standard_derail"],  # 0.6000 measured
            "outcome": "rail DOWN",
            "color": C_VERMILLION,
            "derail_span": None,
        },
        {
            "label": "Angle A/C\n(reg. posterior)",
            "gate": 0.7478,  # canonical
            "derail": 0.8700,  # canonical
            "outcome": "rail UP",
            "color": C_ORANGE,
            "derail_span": None,
        },
        {
            "label": "Angle B\n(vol. de-count)",
            "gate": 0.7439,  # canonical
            "derail": 0.8700,  # canonical
            "outcome": "rail UP",
            "color": C_ORANGE,
            "derail_span": None,
        },
        {
            "label": "Local same-kernel\n(consistent denom.)",
            "gate": rungI["consistent_gate"],  # 0.8700 measured -> gate FAIL
            "derail": rungI["consistent_derail"],  # 0.8700 measured
            "outcome": "gate FAIL",
            "color": C_PURPLE,
            "derail_span": None,
        },
        {
            "label": "Global $D_{sm}$\n(photo-z-smeared)",
            "gate": 0.740,  # canonical: gate PASS ~0.74
            "derail": 0.735,  # canonical ensemble mean (grid-midpoint artefact)
            "outcome": "de-bias, NO peak",
            "color": C_GREEN,
            # multi-seed scatter span (per-seed MAPs span the full grid)
            "derail_span": (RAIL_LOW, RAIL_HIGH),
        },
    ]


def _panel_candidates(ax: plt.Axes, candidates: list[dict[str, object]]) -> None:
    """Panel A: gate (open) vs de-rail (filled) dumbbell per candidate."""
    n = len(candidates)
    y_positions = list(range(n, 0, -1))  # top-to-bottom in table order

    # Shaded grid / rail band and truth line
    ax.axvspan(RAIL_LOW, RAIL_HIGH, color="0.92", zorder=0)
    ax.axvline(TRUTH, color="k", ls="--", lw=1.2, zorder=1)
    ax.axvline(RAIL_LOW, color="0.55", ls=":", lw=1.0, zorder=1)
    ax.axvline(RAIL_HIGH, color="0.55", ls=":", lw=1.0, zorder=1)

    for y, cand in zip(y_positions, candidates):
        color = cand["color"]
        gate = float(cand["gate"])
        derail = float(cand["derail"])
        span = cand["derail_span"]

        # connecting line gate -> de-rail
        ax.plot([gate, derail], [y, y], color=color, lw=1.4, zorder=2, alpha=0.7)

        # de-rail scatter span (only D_sm): horizontal error bar over the full rail
        if span is not None:
            lo, hi = span
            ax.plot(
                [lo, hi],
                [y - 0.16, y - 0.16],
                color=color,
                lw=2.5,
                solid_capstyle="butt",
                alpha=0.45,
                zorder=2,
            )
            ax.text(
                0.5 * (lo + hi),
                y - 0.40,
                "multi-seed scatter",
                ha="center",
                va="top",
                fontsize=5.5,
                color=color,
            )

        # gate marker (open) and de-rail marker (filled)
        ax.scatter(
            [gate], [y], s=42, facecolors="white", edgecolors=color, linewidths=1.4, zorder=4
        )
        ax.scatter([derail], [y], s=46, facecolors=color, edgecolors="k", linewidths=0.5, zorder=5)
        ax.text(
            derail,
            y + 0.22,
            str(cand["outcome"]),
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=color,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(c["label"]) for c in candidates], fontsize=6.5)
    ax.set_ylim(0.3, n + 0.9)
    ax.set_xlim(0.585, 0.885)
    ax.set_xlabel(r"recovered $H_0/100$ (MAP)")
    ax.set_title("(a) Candidate normalisations: gate vs de-rail", fontsize=8)

    ax.text(RAIL_LOW, n + 0.7, "lower\nrail", ha="center", va="top", fontsize=6, color="0.4")
    ax.text(RAIL_HIGH, n + 0.7, "upper\nrail", ha="center", va="top", fontsize=6, color="0.4")
    ax.text(
        TRUTH, 0.45, "truth 0.73", ha="center", va="bottom", fontsize=6.5, rotation=90, color="k"
    )

    # legend proxies
    ax.scatter(
        [],
        [],
        s=42,
        facecolors="white",
        edgecolors="0.3",
        linewidths=1.4,
        label=r"gate $\sigma_z=0.002$",
    )
    ax.scatter(
        [],
        [],
        s=46,
        facecolors="0.3",
        edgecolors="k",
        linewidths=0.5,
        label=r"de-rail $\sigma_z=0.035$",
    )
    ax.legend(loc="lower right", fontsize=6, handletextpad=0.3)


def _panel_dsm_scatter(ax: plt.Axes) -> None:
    """Panel B: D_sm multi-seed per-seed posterior peaks (never on truth)."""
    # Canonical per-seed posterior MAP peaks, n_events = 2000 (INCREMENT3 verdict).
    per_seed_peaks = [0.64, 0.64, 0.69, 0.87]
    fav_single = 0.693  # the single favourable seed that "looked like a win"

    # Shaded rail band + truth
    ax.axhspan(RAIL_LOW, RAIL_HIGH, color="0.92", zorder=0)
    ax.axhline(TRUTH, color="k", ls="--", lw=1.2, zorder=1, label="truth 0.73")
    ax.axhline(RAIL_LOW, color="0.55", ls=":", lw=1.0, zorder=1)
    ax.axhline(RAIL_HIGH, color="0.55", ls=":", lw=1.0, zorder=1)

    # per-seed peaks (jittered x)
    xs = [1, 2, 3, 4]
    ax.scatter(
        xs,
        per_seed_peaks,
        s=70,
        facecolors=C_GREEN,
        edgecolors="k",
        linewidths=0.6,
        zorder=5,
        label="per-seed MAP ($n_{ev}=2000$)",
    )
    for x, peak in zip(xs, per_seed_peaks):
        ax.annotate(
            f"{peak:.2f}",
            (x, peak),
            textcoords="offset points",
            xytext=(7, 0),
            fontsize=6,
            va="center",
        )

    # the favourable single-seed draw
    ax.scatter(
        [0.2],
        [fav_single],
        s=55,
        marker="D",
        facecolors=C_BLUE,
        edgecolors="k",
        linewidths=0.5,
        zorder=5,
    )
    ax.annotate(
        "0.693\n(favourable\nsingle seed)",
        (0.2, fav_single),
        textcoords="offset points",
        xytext=(2, -28),
        fontsize=5.5,
        ha="left",
        color=C_DARKBLUE,
    )

    # ensemble mean (artefact) line
    ax.axhline(0.735, color=C_VERMILLION, ls="-.", lw=1.0, zorder=2)
    ax.text(
        4.6,
        0.735,
        r"$E[h]\approx0.735$" + "\n(grid-midpoint\nartefact)",
        ha="right",
        va="center",
        fontsize=5.8,
        color=C_VERMILLION,
    )

    ax.set_xlim(-0.6, 5.0)
    ax.set_ylim(0.585, 0.885)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {i}" for i in range(1, 5)], fontsize=6.5)
    ax.set_ylabel(r"posterior peak $H_0/100$")
    ax.set_title(r"(b) Global $D_{sm}$: multi-seed, no peak at truth", fontsize=8)
    ax.text(
        2.2,
        0.605,
        r"std $\approx0.10$, does NOT shrink with $n_{ev}$;"
        "\nposteriors flat / multimodal",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="0.3",
    )
    ax.legend(loc="upper left", fontsize=6, handletextpad=0.3)


def main() -> None:
    if _STYLE.exists():
        plt.style.use(str(_STYLE))

    rungI = _load_rungI()
    candidates = _candidate_table(rungI)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )
    _panel_candidates(ax_a, candidates)
    _panel_dsm_scatter(ax_b)

    fig.suptitle(
        "In-catalogue photo-z normalisation: the truth lies between two rails, "
        "and no candidate recovers a peak",
        fontsize=8.5,
    )

    _FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIG_PATH, dpi=300)
    plt.close(fig)
    print(f"Wrote {_FIG_PATH}")
    print("STANDARD gate/de-rail:", rungI["standard_gate"], rungI["standard_derail"])


if __name__ == "__main__":
    main()
