"""Plotting style initialisation.

Call :func:`apply_style` once at program entry to set the Agg backend and
load the project style sheet.
"""

import os
from typing import Literal

import matplotlib
import matplotlib.style

# Font-size and line-weight rcParams that the per-theme scale factor multiplies.
# Kept as a module constant so the "paper == base sheet" invariant is obvious:
# paper applies NO scaling, so these are never touched for the default theme.
_FONT_RCPARAMS: tuple[str, ...] = (
    "font.size",
    "axes.titlesize",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
)

# Per-theme typographic deltas relative to the base ``emri_thesis.mplstyle``.
# "paper" is intentionally absent: it means "apply no override at all" so the
# default output stays byte-for-byte identical to the base sheet.
_THEME_SCALES: dict[str, float] = {
    "talk": 1.8,  # slide-deck sizing
    "web": 1.8,  # interactive matches the talk scale (CSS/Plotly export deferred)
}
# Heavier line weights for non-paper themes (legible at projector / screen sizes).
_THEME_LINE_WEIGHTS: dict[str, dict[str, float]] = {
    "talk": {"lines.linewidth": 2.5, "axes.linewidth": 1.2},
    "web": {"lines.linewidth": 2.5, "axes.linewidth": 1.2},
}


def apply_style(
    *, theme: Literal["paper", "talk", "web"] = "paper", use_latex: bool = False
) -> None:
    """Set the Agg backend and load the ``emri_thesis`` style sheet.

    Parameters
    ----------
    theme:
        Output target for the figure typography. One base style sheet
        (``emri_thesis.mplstyle``) is always loaded, then a small set of
        programmatic rcParams overrides is layered on per theme:

        - ``"paper"`` (default): NO extra overrides. Output is byte-for-byte
          identical to loading the base sheet alone -- the invariant that the
          ``test_apply_style_default_unchanged`` / ``test_rcparams_snapshot``
          regression tests pin.
        - ``"talk"``: scale all font sizes by 1.8 and use heavier line weights
          (``lines.linewidth`` 2.5, ``axes.linewidth`` 1.2) for slide decks.
        - ``"web"``: same matplotlib sizing as ``"talk"`` (the redesign proposal
          says web "matches interactive"). NOTE: the CSS-custom-property / Plotly
          export is interactive-layer work and is OUT OF SCOPE here; the ``web``
          theme currently affects only matplotlib sizing.
    use_latex:
        If ``True``, enable full LaTeX rendering (requires a TeX
        installation). Sets ``text.usetex = True``, switches to
        serif / Computer Modern fonts, and pins font sizes to match a
        10pt paper body. The LaTeX block is applied *after* the theme
        override, so it intentionally overrides theme font scaling when
        both are set (the LaTeX sizes are fixed for typographic fidelity).
        Default ``False`` keeps mathtext rendering that works on headless CI.

    Notes
    -----
    Design decision -- ONE base sheet plus programmatic per-theme overrides,
    rather than three separate ``.mplstyle`` files. The themes are thin deltas
    (a font scale factor + two line weights), so a small in-code dict is less
    duplication than three near-identical sheets, keeps a single source of truth
    for the base, and avoids file-path plumbing. This mirrors the existing
    ``use_latex`` pattern, which already does an in-code ``rcParams.update``.

    Safe to call multiple times; each call reloads the base sheet first, so
    switching themes (or back to ``"paper"``) is a clean reset.
    """
    matplotlib.use("Agg")

    style_path = os.path.join(os.path.dirname(__file__), "emri_thesis.mplstyle")
    matplotlib.style.use(style_path)

    # --- Per-theme typographic override (paper applies nothing) ---
    scale = _THEME_SCALES.get(theme)
    if scale is not None:
        overrides: dict[str, float] = {
            key: float(matplotlib.rcParams[key]) * scale for key in _FONT_RCPARAMS
        }
        overrides.update(_THEME_LINE_WEIGHTS.get(theme, {}))
        matplotlib.rcParams.update(overrides)

    # --- LaTeX override (layered last, wins on font sizes by design) ---
    if use_latex:
        matplotlib.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
            }
        )
