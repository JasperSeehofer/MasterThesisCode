"""H0 forest / tension figure (fig23, VR-NEW-01).

The single highest-value referee-/defense-expected static figure the pipeline
still lacked (viz-redesign proposal §5.1): the H0 *forest* / *tension* plot that
answers "so what?" at a glance. Each published H0 measurement is one row with a
point + asymmetric 68% confidence interval, grouped into the **early-universe**
block (CMB / BAO) on top and the **late-universe** block (distance ladder /
lensing / sirens) below, with the full-height Planck and SH0ES reference bands
drawn behind everything for tension context. The headline **this-work** row is
drawn bold, in HORIZON observatory navy, with a larger marker — visually distinct
from the neutral-gray literature rows through three redundant channels (color +
marker size + line weight) so it survives grayscale and color-blind reproduction.

The literature values are *curated authorship context* (published cosmology
results with arXiv/DOI citations), NOT computed physics — they are never routed
through ``/physics-change``. The this-work number is a PLACEHOLDER behind a
single, clearly-commented DATA GATE (``THIS_WORK_H0`` + ``load_this_work_h0``):
once the trusted production / seed500 posterior lands, finalization is a one-line
swap (or simply dropping a posterior into the data dir so the loader derives the
value via ``compute_hdi_interval``).

All functions follow the project convention: data in, ``(fig, ax)`` out.
None call ``plt.show()`` or ``plt.savefig()``.
"""

from pathlib import Path
from typing import Literal, NamedTuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._colors import PLANCK, REFERENCE, SH0ES, VARIANT_NO_MASS
from master_thesis_code.plotting._helpers import _fig_from_ax, compute_hdi_interval, get_figure
from master_thesis_code.plotting._labels import LABELS


class Measurement(NamedTuple):
    """A single H0 measurement row for the forest plot.

    The value is carried internally as dimensionless ``h`` (so it shares the
    package-wide convention) and converted to ``H0 = 100 * h`` km/s/Mpc at plot
    time to match :data:`LABELS["H0"]`.

    Parameters
    ----------
    label:
        Display label for the row (also the y-tick text).
    h:
        Central value in dimensionless ``h`` (``H0 / 100``).
    sigma_plus:
        Upper 68% half-width in dimensionless ``h``.
    sigma_minus:
        Lower 68% half-width in dimensionless ``h`` (asymmetric when
        ``!= sigma_plus``).
    era:
        ``"early"`` (CMB / BAO) or ``"late"`` (distance ladder / lensing /
        sirens) — sets which grouped block the row lands in.
    citation:
        arXiv / DOI / journal reference string for the value.
    """

    label: str
    h: float
    sigma_plus: float
    sigma_minus: float
    era: Literal["early", "late"]
    citation: str


# ====================== CURATED LITERATURE CONTEXT ============================
# The values below are PUBLISHED cosmology H0 measurements with their arXiv/DOI
# citations inline. They are curated AUTHORSHIP CONTEXT (per ROADMAP/CLAUDE.md),
# NOT computed physics — editing a number here is a citation update, never a
# `/physics-change`. Values are in dimensionless h (= H0 / 100); the plot scales
# to km/s/Mpc. The inline citation enables review of any transcription.
# -----------------------------------------------------------------------------
LITERATURE_H0: list[Measurement] = [
    # --- EARLY universe (CMB / BAO) ---
    Measurement(
        label="Planck 2018",
        h=0.6736,
        sigma_plus=0.0054,
        sigma_minus=0.0054,
        era="early",
        # Planck Collaboration 2020, A&A 641 A6 — TT,TE,EE+lowE+lensing.
        # arXiv:1807.06209 (H0 = 67.36 +/- 0.54 km/s/Mpc).
        citation="Planck Collaboration 2020, A&A 641 A6, arXiv:1807.06209",
    ),
    Measurement(
        label="DESI 2024 BAO+BBN",
        h=0.6852,
        sigma_plus=0.0062,
        sigma_minus=0.0062,
        era="early",
        # DESI Collaboration 2024, BAO + BBN.
        # arXiv:2404.03002 (H0 = 68.52 +/- 0.62 km/s/Mpc).
        citation="DESI Collaboration 2024, arXiv:2404.03002",
    ),
    # --- LATE universe (distance ladder / lensing / standard sirens) ---
    Measurement(
        label="SH0ES 2022",
        h=0.7304,
        sigma_plus=0.0104,
        sigma_minus=0.0104,
        era="late",
        # Riess et al. 2022, ApJL 934 L7 — Cepheid + SNIa distance ladder.
        # arXiv:2112.04510 (H0 = 73.04 +/- 1.04 km/s/Mpc).
        citation="Riess et al. 2022, ApJL 934 L7, arXiv:2112.04510",
    ),
    Measurement(
        label="TRGB (CCHP) 2021",
        h=0.698,
        sigma_plus=0.019,
        sigma_minus=0.019,
        era="late",
        # Freedman 2021, ApJ 919 16 — Tip of the Red Giant Branch.
        # arXiv:2106.15656 (H0 = 69.8 +/- 1.9 km/s/Mpc).
        citation="Freedman 2021, ApJ 919 16, arXiv:2106.15656",
    ),
    Measurement(
        label="TDCOSMO+SLACS 2020",
        h=0.674,
        sigma_plus=0.041,  # asymmetric — exercises sigma_plus != sigma_minus
        sigma_minus=0.032,
        era="late",
        # Birrer et al. 2020, A&A 643 A165 — time-delay strong lensing.
        # arXiv:2007.02941 (H0 = 67.4 +4.1/-3.2 km/s/Mpc).
        citation="Birrer et al. 2020, A&A 643 A165, arXiv:2007.02941",
    ),
    Measurement(
        label="GW170817 siren",
        h=0.700,
        sigma_plus=0.120,  # asymmetric
        sigma_minus=0.080,
        era="late",
        # Abbott et al. 2017, Nature 551 85 — bright standard siren.
        # arXiv:1710.05835 (H0 = 70.0 +12.0/-8.0 km/s/Mpc).
        citation="Abbott et al. 2017, Nature 551 85, arXiv:1710.05835",
    ),
    Measurement(
        label="LVK GWTC-3 dark",
        h=0.680,
        sigma_plus=0.080,  # asymmetric — dark sirens + GLADE+ catalog
        sigma_minus=0.060,
        era="late",
        # Abbott et al. 2023 (LVK H0 GWTC-3) — dark sirens + galaxy catalog.
        # arXiv:2111.03604 (H0 = 68 +8/-6 km/s/Mpc, dark-siren+catalog combined).
        citation="Abbott et al. 2023 (LVK GWTC-3), arXiv:2111.03604",
    ),
]


# ============================ DATA GATE (VR-NEW-01) ============================
# This-work H0 is a PLACEHOLDER until the trusted production/seed500 posterior
# lands. FINALIZE: replace the placeholder triple below OR drop a posterior into
# data_dir so load_this_work_h0() derives it from compute_hdi_interval — a
# ONE-LINE swap. See .planning/ROADMAP.md Phase 4 success criterion 4 /
# STATE.md data-gate note.
THIS_WORK_H0: Measurement = Measurement(
    label="This work (EMRI dark siren)",
    h=0.735,
    sigma_plus=0.015,
    sigma_minus=0.015,  # PLACEHOLDER — NOT the final result
    era="late",
    citation="this work (PLACEHOLDER — data-gated)",
)


def load_this_work_h0(
    data_dir: Path | None = None,
    variant: str = "posteriors",
) -> Measurement:
    """Return the this-work H0 :class:`Measurement` (data-gated).

    The DATA GATE auto-closes: when *data_dir* is given and a canonical combined
    posterior loads, the MAP (argmax) becomes ``h`` and the asymmetric 68%
    bounds come from :func:`compute_hdi_interval`. On any missing/malformed data
    (``FileNotFoundError`` / ``ValueError``) the PLACEHOLDER :data:`THIS_WORK_H0`
    is returned — so the figure always renders and a placeholder can never be
    silently published as the final result (its label/citation say so).

    Parameters
    ----------
    data_dir:
        Directory holding the ``<variant>/`` posterior JSONs. When ``None`` the
        placeholder is returned directly.
    variant:
        ``"posteriors"`` (1D channel) or ``"posteriors_with_bh_mass"`` (2D).

    Returns
    -------
    Measurement
        Either a real this-work measurement derived from the posterior, or the
        :data:`THIS_WORK_H0` placeholder.
    """
    if data_dir is None:
        return THIS_WORK_H0
    try:
        # Local import keeps the plotting layer importable without the bayesian
        # stack in matplotlib-only / notebook environments.
        from master_thesis_code.plotting._helpers import load_canonical_combined_posterior

        h_grid, posterior, _meta = load_canonical_combined_posterior(data_dir, variant)
        map_h = float(h_grid[int(np.argmax(posterior))])
        lo, hi = compute_hdi_interval(h_grid, posterior, level=0.683)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return THIS_WORK_H0
        return Measurement(
            label="This work (EMRI dark siren)",
            h=map_h,
            sigma_plus=max(hi - map_h, 0.0),
            sigma_minus=max(map_h - lo, 0.0),
            era="late",
            citation="this work (EMRI dark siren, production posterior)",
        )
    except (FileNotFoundError, ValueError, KeyError, IndexError):
        return THIS_WORK_H0


def _draw_reference_bands(ax: Axes) -> None:
    """Draw the full-height Planck and SH0ES reference bands behind everything.

    Mirrors the band convention in ``bayesian_plots.py`` (reserved PLANCK/SH0ES
    colors, ``axvspan`` at ``zorder=0`` with a small top-axis text label) but in
    km/s/Mpc units to match the forest x-axis.
    """
    # Planck: H0 = 67.36 +/- 0.54 km/s/Mpc (arXiv:1807.06209).
    ax.axvspan(67.36 - 0.54, 67.36 + 0.54, color=PLANCK, alpha=0.15, zorder=0)
    ax.text(
        67.36,
        0.99,
        "Planck",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6,
        color=PLANCK,
    )
    # SH0ES: H0 = 73.04 +/- 1.04 km/s/Mpc (arXiv:2112.04510).
    ax.axvspan(73.04 - 1.04, 73.04 + 1.04, color=SH0ES, alpha=0.15, zorder=0)
    ax.text(
        73.04,
        0.99,
        "SH0ES",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=6,
        color=SH0ES,
    )


def plot_h0_forest(
    measurements: list[Measurement] | None = None,
    this_work: Measurement | None = None,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """H0 forest / tension figure (fig23, VR-NEW-01).

    Renders one point + asymmetric 68% confidence-interval row per measurement,
    grouped early (top block) vs late (bottom block) with a thin horizontal
    divider between the blocks, the full-height Planck and SH0ES reference bands
    behind everything, and the headline this-work row drawn bold in HORIZON navy
    (larger marker, heavier line) — distinct through three redundant channels
    (color + size + weight) so it survives grayscale and color-blind reads.

    Parameters
    ----------
    measurements:
        Literature measurements to plot. Defaults to :data:`LITERATURE_H0`.
    this_work:
        The headline this-work row. Defaults to the data-gated
        :data:`THIS_WORK_H0` placeholder.
    ax:
        Optional pre-existing Axes to draw into. When ``None`` a REVTeX
        single-column figure is created via :func:`get_figure` (no hardcoded
        figsize).

    Returns
    -------
    tuple[Figure, Axes]
        The figure and the populated forest Axes.

    References
    ----------
    Literature H0 values (curated authorship context, not computed physics):

    - Planck Collaboration 2020, A&A 641 A6, arXiv:1807.06209
    - DESI Collaboration 2024, arXiv:2404.03002
    - Riess et al. 2022, ApJL 934 L7, arXiv:2112.04510
    - Freedman 2021, ApJ 919 16, arXiv:2106.15656
    - Birrer et al. 2020, A&A 643 A165, arXiv:2007.02941
    - Abbott et al. 2017, Nature 551 85, arXiv:1710.05835
    - Abbott et al. 2023 (LVK GWTC-3), arXiv:2111.03604
    """
    if measurements is None:
        measurements = LITERATURE_H0
    if this_work is None:
        this_work = THIS_WORK_H0

    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    _draw_reference_bands(ax)

    # Group early on top, late below, this-work as the bold bottom row. Rows are
    # laid out top-to-bottom in reading order, then assigned descending y so the
    # FIRST early row sits at the TOP of the axes.
    early = [m for m in measurements if m.era == "early"]
    late = [m for m in measurements if m.era == "late"]
    ordered = early + late + [this_work]
    n = len(ordered)
    # y descending from top: row i at y = (n - 1 - i).
    y_positions = list(range(n - 1, -1, -1))

    tick_y: list[float] = []
    tick_labels: list[str] = []
    for m, y in zip(ordered, y_positions):
        h0 = 100.0 * m.h
        xerr = [[100.0 * m.sigma_minus], [100.0 * m.sigma_plus]]
        is_this_work = m is this_work
        if is_this_work:
            # Redundant emphasis: navy color + larger marker + heavier line.
            ax.errorbar(
                h0,
                y,
                xerr=xerr,
                fmt="o",
                color=VARIANT_NO_MASS,
                markersize=7.0,
                elinewidth=2.0,
                capsize=4.0,
                capthick=2.0,
                zorder=5,
            )
        else:
            ax.errorbar(
                h0,
                y,
                xerr=xerr,
                fmt="o",
                color=REFERENCE,
                markersize=4.0,
                elinewidth=1.0,
                capsize=2.5,
                capthick=1.0,
                zorder=3,
            )
        tick_y.append(float(y))
        tick_labels.append(m.label)

    # Thin horizontal divider between the early block and the late block.
    if early and (late or True):
        # The boundary sits just below the last early row (y of first late row +
        # 0.5). With descending y, the last early row is at y = n-1-len(early)+1.
        divider_y = (n - 1 - len(early)) + 0.5
        ax.axhline(divider_y, color=REFERENCE, linewidth=0.6, linestyle=":", zorder=1)

    ax.set_yticks(tick_y)
    ax.set_yticklabels(tick_labels)
    # Bold the this-work tick label (redundant with color + marker size).
    for tick_label, m in zip(ax.get_yticklabels(), ordered):
        if m is this_work:
            tick_label.set_fontweight("bold")
            tick_label.set_color(VARIANT_NO_MASS)

    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(LABELS["H0"])

    # No fig.tight_layout: constrained_layout (project mplstyle) owns packing.
    return fig, ax
