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

# ---------------------------------------------------------------------------
# Observatory + Atlas redesign palette (v2).  See docs/VIZ_REDESIGN_PROPOSAL.md.
# Additive: the names above are kept for backward compatibility while factories
# migrate to the tokens below.
# ---------------------------------------------------------------------------

# Method -> color map (de-facto LVK / gwcosmo grammar); reuse everywhere.
METHOD: dict[str, str] = {
    "bright": "#F0E442",  # gold   -- bright siren (EM counterpart)
    "spectral": "#E69F00",  # orange -- spectral siren (mass spectrum)
    "dark": "#0072B2",  # blue   -- dark / galaxy-catalog siren
    "combined": "#1a1a1a",  # black  -- combined / fiducial headline
}

# The two pipeline variants share ONE hue, distinguished by linestyle
# (colorblind- and greyscale-safe): (color, linestyle) per variant.
VARIANT_STYLE: dict[str, tuple[str, str]] = {
    "no_mass": (METHOD["dark"], "-"),  # Without M_z  (solid)
    "with_mass": (METHOD["dark"], "--"),  # With M_z     (dashed)
}

# Tension anchors -- full-height bands, identical on the posterior and the
# (future) H0-in-context forest plot.
PLANCK_BAND: str = "#CC79A7"  # reddish-purple / pink, low alpha
SHOES_BAND: str = "#56B4E9"  # cyan
PRIOR: str = "#9e9e9e"  # neutral grey, dashed -- flat H0 prior

# Scientific colormaps (Atlas).  cmcrameri is an OPTIONAL dependency; fall back
# to perceptually-uniform, CVD-safe matplotlib built-ins when it is absent so
# the no-dep path still renders.
try:
    import cmcrameri.cm  # noqa: F401  -- registers 'batlow', 'vik', 'romaO', ...

    _HAS_CRAMERI = True
except ImportError:
    _HAS_CRAMERI = False

SEQUENTIAL_CMAP: str = "batlow" if _HAS_CRAMERI else "cividis"  # SNR, density, P_det
DIVERGING_CMAP: str = "vik" if _HAS_CRAMERI else "RdBu"  # residuals, pulls, MAP bias
CYCLIC_CMAP: str = "romaO" if _HAS_CRAMERI else "twilight"  # phase / sky angle
GREY_BAD: str = "#bdbdbd"  # reserved neutral for out-of-range / reference
