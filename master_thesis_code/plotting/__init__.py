"""Plotting subpackage for the EMRI thesis code.

Public API::

    from master_thesis_code.plotting import apply_style, get_figure, save_figure
"""

from master_thesis_code.plotting._helpers import (
    _fig_from_ax,
    get_figure,
    make_colorbar,
    save_figure,
)
from master_thesis_code.plotting._style import apply_style

__all__ = ["_fig_from_ax", "apply_style", "get_figure", "make_colorbar", "save_figure"]
