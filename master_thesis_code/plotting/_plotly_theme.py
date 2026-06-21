"""Shared HORIZON ``go.layout.Template`` for the interactive Plotly web layer.

This module ports the settled static HORIZON design (Phases 1-5) to the Plotly
interactive figures so the GitHub-Pages ``interactive/`` set uses the SAME hex
tokens and sequential colormap as the static PDFs. There is exactly ONE source
of truth for color: :mod:`master_thesis_code.plotting._colors`. No hex literal
is ever defined here -- every color is imported from ``_colors``.

Why a single template
---------------------
Before this module, ``interactive.py`` hardcoded hex (``CYCLE[0]`` derived
rgba fills, the literal ``"Viridis"`` colorscale) that drifted from the static
HORIZON palette. Building one :func:`horizon_plotly_template` and applying it to
all 8 factories guarantees the interactive set matches the static figures and
makes a palette change a one-line edit in ``_colors``.

Web typography
--------------
:func:`master_thesis_code.plotting._style.apply_style` is matplotlib-only, so it
cannot size Plotly figures. Instead this template mirrors the ``"web"`` theme
intent (``_THEME_SCALES["web"] == 1.8`` relative to the paper base, with heavier
``lines.linewidth == 2.5``) directly in the Plotly ``layout.font`` / line
defaults: a readable on-screen sans-serif family at a web-scaled base size.

References
----------
- Color tokens: ``master_thesis_code.plotting._colors`` (single source of truth).
- cividis colorscale: Nuñez, Anderton & Renslow (2018),
  doi:10.1371/journal.pone.0199239 (perceptually uniform + deuteranopia-safe).
"""

import matplotlib
import plotly.graph_objects as go

from master_thesis_code.plotting._colors import (
    CMAP,
    CYCLE,
    REFERENCE,
    VARIANT_NO_MASS,
    VARIANT_WITH_MASS,
)

# ---------------------------------------------------------------------------
# Web typography (mirrors apply_style theme="web" intent without calling it)
# ---------------------------------------------------------------------------

# apply_style("web") scales the paper base font by 1.8x; the paper base body is
# ~8pt, so the on-screen base lands near 14-15px. A readable sans-serif stack
# keeps the interactive figures legible on screen and in slide decks.
WEB_FONT_FAMILY: str = "Helvetica Neue, Arial, sans-serif"
WEB_FONT_SIZE: int = 15  # paper base (~8pt) * web scale (1.8) rounded to a screen size
# Mirror lines.linewidth == 2.5 from _THEME_LINE_WEIGHTS["web"].
WEB_LINE_WIDTH: float = 2.5

# Clean light backgrounds + a light-gray grid, consistent with the static look.
_PAPER_BG: str = "#ffffff"
_PLOT_BG: str = "#ffffff"
_GRID_COLOR: str = "#e6e6e6"
_AXIS_COLOR: str = "#1a1a1a"


def _cividis_plotly_colorscale(n_stops: int = 10) -> list[list[float | str]]:
    """Sample matplotlib's ``CMAP`` (cividis) into a Plotly colorscale list.

    Parameters
    ----------
    n_stops:
        Number of evenly spaced stops to sample (default 10).

    Returns
    -------
    list[list[float | str]]
        ``[[t, "rgb(r,g,b)"], ...]`` with ``t`` in ``[0, 1]`` -- the Plotly
        colorscale form. Built from :data:`master_thesis_code.plotting._colors.CMAP`
        ("cividis") so the interactive heatmaps/scattergeo inherit the SAME
        sequential ramp as the static figures, never Plotly's default Viridis.
    """
    cmap = matplotlib.colormaps[CMAP]
    stops: list[list[float | str]] = []
    for i in range(n_stops):
        t = i / (n_stops - 1)
        r, g, b, _ = cmap(t)
        rgb = f"rgb({round(r * 255)},{round(g * 255)},{round(b * 255)})"
        stops.append([t, rgb])
    return stops


def horizon_plotly_template() -> go.layout.Template:
    """Build the shared HORIZON Plotly template.

    All 8 interactive factories apply this template so the web layer carries the
    SAME hex tokens (imported from :mod:`master_thesis_code.plotting._colors`) and
    the SAME cividis sequential colorscale as the static HORIZON figures.

    The ordered ``colorway`` puts the headline comparison colors first
    (``VARIANT_NO_MASS`` navy, ``VARIANT_WITH_MASS`` gold) so the brand contrast
    gets the leading slots, followed by the Okabe-Ito :data:`CYCLE` for any
    incidental traces.

    Returns
    -------
    go.layout.Template
        A template ready for ``fig.update_layout(template=...)``.

    Notes
    -----
    No hex literal is defined in this function -- ``_colors`` is the single
    source of truth (VR-INT-01).
    """
    tmpl = go.layout.Template()

    # Ordered HORIZON data colors: navy + gold lead, then the Okabe-Ito cycle.
    tmpl.layout.colorway = [VARIANT_NO_MASS, VARIANT_WITH_MASS, *CYCLE]

    # Sequential colorscale from cividis (imported via _colors.CMAP), not Viridis.
    tmpl.layout.colorscale.sequential = _cividis_plotly_colorscale()

    # Web typography mirrors apply_style("web"); apply_style itself is mpl-only.
    tmpl.layout.font = {"family": WEB_FONT_FAMILY, "size": WEB_FONT_SIZE, "color": _AXIS_COLOR}

    # Heavier default line width to mirror lines.linewidth == 2.5 (web theme).
    tmpl.data.scatter = [go.Scatter(line={"width": WEB_LINE_WIDTH})]

    # Clean light backgrounds + light-gray gridlines, consistent with static look.
    tmpl.layout.paper_bgcolor = _PAPER_BG
    tmpl.layout.plot_bgcolor = _PLOT_BG
    tmpl.layout.xaxis = {
        "gridcolor": _GRID_COLOR,
        "zerolinecolor": _GRID_COLOR,
        "linecolor": _AXIS_COLOR,
    }
    tmpl.layout.yaxis = {
        "gridcolor": _GRID_COLOR,
        "zerolinecolor": _GRID_COLOR,
        "linecolor": _AXIS_COLOR,
    }
    # Reference-line / annotation default color from the HORIZON scaffold gray.
    tmpl.layout.shapedefaults = {"line": {"color": REFERENCE}}

    return tmpl


# Build once at import; factories apply this shared instance.
HORIZON_TEMPLATE: go.layout.Template = horizon_plotly_template()

# Reusable cividis colorscale list for traces (Scattergeo markers, Heatmap) whose
# ``colorscale`` does NOT inherit the template's sequential ramp automatically.
# Built from _colors.CMAP -- the SAME ramp the template carries (no Viridis).
CIVIDIS_COLORSCALE: list[list[float | str]] = _cividis_plotly_colorscale()
