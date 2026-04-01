"""Centralized color palette for EMRI thesis plots.

Semantic role colors for special plot elements and an ordered cycle
for multi-line plots (e.g. individual posteriors).  Ad-hoc color
strings in plot modules should be replaced with these constants.
"""

# --- Semantic role colors (per D-05) ---
TRUTH: str = "#2ca02c"  # green — truth / reference lines
MEAN: str = "#d62728"  # red — mean / summary lines
EDGE: str = "#1a1a1a"  # near-black — histogram edges, outlines
REFERENCE: str = "#7f7f7f"  # gray — secondary reference lines

# --- Ordered color cycle for multi-line plots ---
# 8 perceptually distinct colors; first 4 match the default tab10 subset
CYCLE: list[str] = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
]

# --- Default colormap name ---
CMAP: str = "viridis"
