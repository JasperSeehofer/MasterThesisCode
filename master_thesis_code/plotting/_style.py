"""Plotting style initialisation.

Call :func:`apply_style` once at program entry to set the Agg backend and
load the project style sheet.
"""

import os

import matplotlib
import matplotlib.style


def apply_style() -> None:
    """Set the Agg backend and load the ``emri_thesis`` style sheet.

    Safe to call multiple times; subsequent calls are no-ops for the backend
    (matplotlib ignores ``use()`` after the first pyplot import, but we call
    it before any pyplot import in main).
    """
    matplotlib.use("Agg")

    style_path = os.path.join(os.path.dirname(__file__), "emri_thesis.mplstyle")
    matplotlib.style.use(style_path)
