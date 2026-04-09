"""Centralized color palette for EMRI thesis plots -- Okabe-Ito edition.

Palette source: Wong (2011) Nature Methods, doi:10.1038/nmeth.1618
Colorblind-safe: verified for deuteranopia, protanopia, tritanopia.

Exported names (consumed by 10 plotting modules):
    TRUTH, MEAN, EDGE, REFERENCE, ACCENT  -- semantic role colors
    CYCLE                                  -- ordered 7-color cycle
    CMAP                                   -- default colormap name (str)
    SEQUENTIAL_BLUES                       -- truncated Blues cmap object
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Okabe-Ito cycle (7 colors; black excluded -- reserved for text/edges) ---
# Wong (2011) Table 1, columns 2-8
CYCLE: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# --- Semantic role colors (drawn from Okabe-Ito for consistency) ---
TRUTH: str = "#009E73"  # bluish green -- truth / reference lines
MEAN: str = "#D55E00"  # vermillion -- mean / summary lines
EDGE: str = "#1a1a1a"  # near-black -- histogram edges, outlines
REFERENCE: str = "#56B4E9"  # sky blue -- secondary reference lines
ACCENT: str = "#E69F00"  # orange -- accent for annotations/highlights

# --- Variant comparison colors (without / with BH mass channel) ---
VARIANT_NO_MASS: str = "#0072B2"  # Okabe-Ito blue (petrol) -- without M_z
VARIANT_WITH_MASS: str = "#56B4E9"  # Okabe-Ito sky blue -- with M_z

# --- Sequential Blues (truncated 0.1-0.85 to avoid near-white/near-black) ---
_blues_base = plt.colormaps["Blues"]
SEQUENTIAL_BLUES: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "Blues_trunc", _blues_base(np.linspace(0.1, 0.85, 256))
)

# --- Default colormap name (kept as viridis for backward compat) ---
# Use SEQUENTIAL_BLUES object directly for 2D/heatmap plots in future phases.
CMAP: str = "viridis"
