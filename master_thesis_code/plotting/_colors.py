"""Centralized color palette for EMRI thesis plots -- HORIZON v2 edition.

HORIZON semantic palette (viz-redesign proposal §3.2, decision 2026-06-21).
The headline contrast of the thesis -- Without-M_z vs With-M_z -- is encoded as
observatory **navy** (#1B2A4A) vs signal **gold** (#E8A317): two hues with a
large lightness separation so the comparison survives both grayscale printing
and deuteranopia. Earlier this contrast relied on two near-identical blues
(#0072B2 / #56B4E9) plus linestyle alone and collapsed to indistinguishable
grays in print -- that "two-blues" bug is what this palette kills.

Encoding discipline (HARD rule for every comparison figure):
    Comparisons MUST be redundantly encoded -- color + linestyle + direct label.
    Never rely on color alone. Navy/gold differ strongly in lightness, but the
    redundant channels guarantee readability for the colorblind and in B/W.

Reserved band colors:
    PLANCK (#3E7CB1) and SH0ES (#9A6FB0) are reserved for cosmology reference
    bands (early- vs late-universe H0 tension context). They must NEVER be used
    for a data series -- a data color and a context-band color must never collide.

Scope note (HORIZON v2.3, Phase 2 -- Colormap & Heatmap Modernization):
    CMAP is now "cividis" (proposal §6a). cividis is perceptually uniform AND
    deuteranopia-safe (a redesign of viridis by Nuñez, Anderton & Renslow 2018,
    doi:10.1371/journal.pone.0199239), so the sequential heatmaps read with full
    dynamic range and survive color-blind / grayscale reproduction. The viridis
    migration is no longer deferred -- it lands here, package-wide. Every
    sequential heatmap routes through CMAP; every heatmap calls
    ``cmap.set_bad(NO_DATA)`` so empty / NaN bins read as no-data, not blank.

Exported names (consumed by ~13 plotting modules -- all kept exported):
    TRUTH, MEAN, EDGE, REFERENCE, ACCENT     -- semantic role colors
    VARIANT_NO_MASS, VARIANT_WITH_MASS       -- headline comparison series
    PLANCK, SH0ES                            -- reserved cosmology-band colors
    CYCLE                                     -- ordered 7-color Okabe-Ito cycle
    CMAP                                      -- default sequential colormap name (str)
    DIVERGING_CMAP                            -- diverging colormap name for bias maps (str)
    NO_DATA                                   -- gray for set_bad / no-data bins (str)
    SEQUENTIAL_BLUES                          -- truncated Blues cmap object

Cycle source: Wong (2011) Nature Methods, doi:10.1038/nmeth.1618 (Okabe-Ito;
colorblind-safe for deuteranopia, protanopia, tritanopia).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Okabe-Ito cycle (7 colors; black excluded -- reserved for text/edges) ---
# Wong (2011) Table 1, columns 2-8. Unchanged in HORIZON v2.
CYCLE: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# --- Semantic role colors (HORIZON v2) ---
TRUTH: str = "#C2451E"  # HORIZON warm vermillion -- truth / injected rule ONLY
MEAN: str = "#D55E00"  # vermillion -- mean / summary lines
EDGE: str = "#1a1a1a"  # near-black -- histogram edges, outlines
REFERENCE: str = "#4F4F4F"  # HORIZON scaffold gray -- neutral secondary reference lines
ACCENT: str = "#E69F00"  # orange -- accent for annotations/highlights

# --- Variant comparison colors (without / with BH mass channel) ---
# HORIZON navy vs gold: strong lightness separation, grayscale + CB safe.
VARIANT_NO_MASS: str = "#1B2A4A"  # HORIZON observatory navy -- without M_z (headline)
VARIANT_WITH_MASS: str = "#E8A317"  # HORIZON signal gold -- with M_z

# --- Reserved cosmology-band colors (NEVER use for a data series) ---
PLANCK: str = "#3E7CB1"  # mid cyan-blue -- Planck / early-universe band
SH0ES: str = "#9A6FB0"  # muted purple -- SH0ES / late-universe band

# --- Sequential Blues (truncated 0.1-0.85 to avoid near-white/near-black) ---
_blues_base = plt.colormaps["Blues"]
SEQUENTIAL_BLUES: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "Blues_trunc", _blues_base(np.linspace(0.1, 0.85, 256))
)

# --- Default sequential colormap (cividis: perceptually uniform + CB-safe) ---
# cividis is the deuteranopia-optimized redesign of viridis
# (Nuñez, Anderton & Renslow 2018, doi:10.1371/journal.pone.0199239). Every
# sequential heatmap in the package routes through this name.
CMAP: str = "cividis"

# --- No-data color for set_bad (every heatmap masks NaN/empty bins to this) ---
# HORIZON palette token (proposal §3.2). Heatmaps do
# ``cmap = plt.get_cmap(CMAP).copy(); cmap.set_bad(NO_DATA)`` so empty bins read
# as neutral gray instead of vanishing into the lowest colormap color.
NO_DATA: str = "#D9D9D9"

# --- Diverging colormap for bias / residual maps centered on a reference ---
# Built-in matplotlib diverging cmap -- NO new runtime dependency. Pair with
# _helpers.diverging_norm(vcenter=0.73) so the H0 truth sits at the white
# midpoint. UPGRADE PATH: cmcrameri's "vik" is the preferred perceptually
# uniform diverging map; when a real bias map is built, ``uv add cmcrameri``
# and swap DIVERGING_CMAP = "cmc.vik". Do NOT add cmcrameri until then.
DIVERGING_CMAP: str = "RdBu_r"
