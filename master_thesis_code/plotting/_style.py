"""Plotting style initialisation.

Call :func:`apply_style` once at program entry to set the Agg backend and
load the project style sheet.
"""

import os

import matplotlib
import matplotlib.style


def apply_style(*, use_latex: bool = False) -> None:
    """Set the Agg backend and load the ``emri_thesis`` style sheet.

    Parameters
    ----------
    use_latex:
        If ``True``, enable full LaTeX rendering (requires a TeX
        installation).  Sets ``text.usetex = True``, switches to
        serif / Computer Modern fonts, and adjusts font sizes to
        match a 10pt paper body.  Default ``False`` keeps mathtext
        rendering that works on headless CI.

    Safe to call multiple times; subsequent calls are no-ops for the backend
    (matplotlib ignores ``use()`` after the first pyplot import, but we call
    it before any pyplot import in main).
    """
    matplotlib.use("Agg")

    style_path = os.path.join(os.path.dirname(__file__), "emri_thesis.mplstyle")
    matplotlib.style.use(style_path)

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
