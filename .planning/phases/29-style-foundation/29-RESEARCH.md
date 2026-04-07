# Phase 29: Style Foundation - Research

**Researched:** 2026-04-07
**Domain:** Matplotlib style infrastructure — rcParams, color palettes, figure presets
**Confidence:** HIGH

## Summary

Phase 29 is a pure matplotlib configuration task: update `emri_thesis.mplstyle`, replace `_colors.py`, and tune `get_figure` presets. There are no new dependencies, no external services, and no physics changes. Everything needed is already in the installed matplotlib 3.10.8.

The current style has five concrete gaps relative to STYL-01/02/03: (1) top/right spines are still shown, (2) `pdf.fonttype` and `ps.fonttype` are both `3` (Type 3 bitmap fonts, confirmed by `pdffonts`), (3) font sizes are too large for REVTeX column widths (11/12/13pt vs the required 7-9pt range), (4) ticks point outward, and (5) legends have a semi-transparent frame. `_colors.py` uses the tab10-derived palette, not Okabe-Ito. The `get_figure` height presets use golden-ratio defaults and no per-preset font-size guidance exists.

The biggest test impact: `test_style.py::test_apply_style_default_unchanged` and `test_rcparams_snapshot` hard-pin 18 rcParam values. Both tests must be updated as part of this phase; they are intentional regression guards, and the commit updating them is the proof that the change was deliberate.

**Primary recommendation:** Update `emri_thesis.mplstyle` first (drives all downstream), then `_colors.py` (isolated module swap), then verify `get_figure` height calculations. Update the two regression tests last to lock the new values.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STYL-01 | Update `emri_thesis.mplstyle`: remove top/right spines, `pdf.fonttype: 42`, `ps.fonttype: 42`, font sizes 7-9pt, inward ticks, frameless legends | rcParam names verified; all settable in `.mplstyle` file; see Standard Stack section |
| STYL-02 | Replace `_colors.py` with Okabe-Ito cycle, truncated Blues (0.1-0.85), single accent color | Hex values verified from Wong (2011); truncation API confirmed for matplotlib 3.10; see Code Examples |
| STYL-03 | Update `get_figure` presets with font sizes tuned for REVTeX widths | Width constants already correct (3.375/7.0); only height and associated font guidance need updating |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **No physics changes:** v2.1 is visualization only. No formula, constant, or waveform parameter changes.
- **Typing:** All public functions need complete annotations. Use `list[T]`, `dict[k,v]`, `X | None` (no `Optional`). No bare `np.ndarray`.
- **Pre-commit gate:** ruff + mypy + pytest must pass before committing.
- **Dataclass default rule:** Not applicable here (no new dataclasses).
- **GPU guard:** Not applicable here (no GPU code in plotting).
- **`apply_style()` convention:** MEMORY.md records "always use `apply_style()`; session fixture handles tests" — this fixture calls `apply_style()` with no args. The `use_latex=True` path must still work after changes.
- **GSD workflow:** Changes must go through GSD phase execution, not ad-hoc edits.

## Standard Stack

### Core (all already installed)
| Library | Version | Purpose | Notes |
|---------|---------|---------|-------|
| matplotlib | 3.10.8 | rcParams, style, colormaps | `plt.colormaps[name]` API (not deprecated `get_cmap`) |
| numpy | installed | colormap sample arrays | `np.linspace(0.1, 0.85, 256)` for truncated Blues |

[VERIFIED: local environment — `uv run python -c "import matplotlib; print(matplotlib.__version__)"`]

### No new dependencies needed
All changes are configuration (`.mplstyle` file edits, Python constant replacements). No `pip install` required.

## Architecture Patterns

### Recommended Change Sequence

```
master_thesis_code/plotting/
├── emri_thesis.mplstyle    # STYL-01: rcParam changes (drives all tests)
├── _colors.py              # STYL-02: palette replacement
├── _helpers.py             # STYL-03: get_figure height adjustment (minor)
└── _style.py               # No change needed unless use_latex sizes also updated
master_thesis_code_test/plotting/
├── test_style.py           # Must update: test_apply_style_default_unchanged + test_rcparams_snapshot
└── test_colors.py          # Must update: test_cmap_is_viridis + CYCLE length assertion
```

### Pattern 1: mplstyle rcParam additions

All target rcParams are settable in `.mplstyle` files directly. Verified working syntax:

```ini
# Spines (matplotlib 3.10 supports these as rcParams)
axes.spines.top:    False
axes.spines.right:  False

# Font types (42 = TrueType/Type1; prevents Type 3 bitmap embedding)
pdf.fonttype:  42
ps.fonttype:   42

# Tick direction (valid values: in, out, inout)
xtick.direction:  in
ytick.direction:  in

# Frameless legend (frameon: False disables the box entirely)
legend.frameon:  False
```

[VERIFIED: `uv run python -c "matplotlib.rcParams['axes.spines.top'] = False"` succeeds]

### Pattern 2: Okabe-Ito cycle in mplstyle

The cycler syntax in `.mplstyle` files requires double-quoted hex values (single quotes cause a parse error in matplotlib 3.10):

```ini
# Correct syntax — double quotes required inside the cycler() call
axes.prop_cycle: cycler(color=["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"])
```

[VERIFIED: tested locally — single-quote syntax fails with "unterminated string literal"; double-quote works]

### Pattern 3: _colors.py replacement

Replace the entire module. Preserve the same exported names so all 10 importing modules (`bayesian_plots.py`, `evaluation_plots.py`, `fisher_plots.py`, `model_plots.py`, `simulation_plots.py`, `convergence_plots.py`, `catalog_plots.py`, `sky_plots.py`, `physical_relations_plots.py`, `paper_figures.py`) require no import changes.

Exported names that must remain: `TRUTH`, `MEAN`, `EDGE`, `REFERENCE`, `CYCLE`, `CMAP`

New addition: `SEQUENTIAL_BLUES` (truncated cmap object) for heatmaps/2D plots.

```python
# Source: Wong (2011) Nature Methods doi:10.1038/nmeth.1618
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Okabe-Ito cycle (7 colors, excluding black which is reserved for text/edges)
CYCLE: list[str] = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Semantic roles — drawn from Okabe-Ito palette for consistency
TRUTH: str = "#009E73"      # bluish green — truth / reference lines
MEAN: str = "#D55E00"       # vermillion — mean / summary lines
EDGE: str = "#1a1a1a"       # near-black — histogram edges (unchanged)
REFERENCE: str = "#56B4E9"  # sky blue — secondary reference lines
ACCENT: str = "#E69F00"     # orange — accent for annotations

# Sequential Blues (truncated 0.1-0.85 to avoid near-white and near-black extremes)
_blues_base = plt.colormaps["Blues"]
SEQUENTIAL_BLUES: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "Blues_trunc", _blues_base(np.linspace(0.1, 0.85, 256))
)

# Default colormap for heatmaps/scatter (keep CMAP for backward compat)
CMAP: str = "Blues_trunc"
```

[VERIFIED: `plt.colormaps["Blues"]` works in matplotlib 3.10; `LinearSegmentedColormap.from_list` confirmed]

Note: changing `CMAP` from `"viridis"` to `"Blues_trunc"` will break `test_colors.py::test_cmap_is_viridis`. That test must be updated. The `CMAP` string would also need the cmap registered before use — or keep `CMAP = "viridis"` and add a separate `SEQUENTIAL_BLUES` object. See Open Questions.

### Pattern 4: get_figure preset heights

Current heights use golden ratio (`width / 1.618`). For REVTeX column figures, a 1:1 or 4:3 ratio is often better for single-column subplots. The STYL-03 requirement says "font sizes tuned for REVTeX widths" — this is primarily about the `_style.py` `use_latex=True` branch which currently hardcodes 10pt. The preset heights themselves are already set correctly (golden ratio is standard); the main gap is font size guidance per preset.

```python
# Updated _style.py use_latex branch — smaller sizes for REVTeX single column
if use_latex:
    matplotlib.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 8,           # was 10 — REVTeX body is ~10pt, labels smaller
        "axes.titlesize": 9,      # was 12
        "axes.labelsize": 8,      # was 10
        "xtick.labelsize": 7,     # was 9
        "ytick.labelsize": 7,     # was 9
        "legend.fontsize": 7,     # was 9
    })
```

The non-latex defaults in `emri_thesis.mplstyle` should also be reduced (STYL-01 covers this):

```ini
font.size:          8
axes.titlesize:     9
axes.labelsize:     8
xtick.labelsize:    7
ytick.labelsize:    7
legend.fontsize:    7
```

[ASSUMED: The exact 7/8/9 pt split is reasonable for REVTeX but not verified against the actual paper template measurements. User should confirm before locking.]

### Anti-Patterns to Avoid

- **Don't change `figure.figsize` in mplstyle:** Current `6.4, 4.0` is the fallback default when no preset is used. Keep it. Callers use `preset="single"` or `preset="double"` explicitly.
- **Don't set `axes.prop_cycle` programmatically in `apply_style()`:** Set it in `.mplstyle` only. The test `test_rcparams_snapshot` verifies the mplstyle is the single source of truth.
- **Don't use deprecated `plt.get_cmap()`:** In matplotlib 3.10, use `plt.colormaps["name"]`. The `get_cmap` API shows a deprecation warning (removed in 3.11).
- **Don't register `Blues_trunc` as a side-effect of importing `_colors.py`:** `LinearSegmentedColormap.from_list` creates the object; `plt.colormaps.register` adds it globally. Registration at import time is acceptable for a small project but keep it idempotent (use `force=True`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Colorblind-safe palette | Custom palette via trial/error | Okabe-Ito (Wong 2011) | Peer-reviewed, widely cited in scientific publishing guidelines |
| Type 42 font embedding | Font subsetting code | `pdf.fonttype: 42` rcParam | matplotlib handles the embedding; one line in mplstyle |
| Truncated colormap | Linear interpolation from scratch | `LinearSegmentedColormap.from_list` | matplotlib's own API, handles gamma-correct interpolation |
| Spine removal | `ax.spines['top'].set_visible(False)` per-axis | `axes.spines.top: False` in mplstyle | rcParam applies globally; no per-axes boilerplate in every plot function |

## Common Pitfalls

### Pitfall 1: test_rcparams_snapshot hard-pins all 18 values
**What goes wrong:** Edit mplstyle, tests fail with "rcParam 'font.size' drifted: expected 11.0, got 8.0"
**Why it happens:** `test_style.py` lines 116-154 have an explicit `expected` dict with exact old values.
**How to avoid:** Update `test_rcparams_snapshot` and `test_apply_style_default_unchanged` as part of the same task that updates the mplstyle. They are intentional regression guards, not accidental hardcoding.
**Warning signs:** Any test failure in `test_style.py` after mplstyle edits.

### Pitfall 2: test_cmap_is_viridis will fail after CMAP rename
**What goes wrong:** `test_colors.py::test_cmap_is_viridis` asserts `CMAP == "viridis"`.
**Why it happens:** Hardcoded equality check on the string value.
**How to avoid:** Either (a) keep `CMAP = "viridis"` and add `SEQUENTIAL_BLUES` as a new export, or (b) change CMAP and update the test. Option (a) is safer — no consumer of `CMAP` currently expects a Blues colormap.
**Warning signs:** Test failure in `test_colors.py` after `_colors.py` edit.

### Pitfall 3: CYCLE length test (>= 6 entries)
**What goes wrong:** Okabe-Ito has 7 colors (excluding black) — passes the ">= 6" test. No issue here, but if someone removes yellow (#F0E442) to avoid low-contrast backgrounds, the count still passes but colorblind contrast may degrade.
**How to avoid:** Keep all 7 Okabe-Ito colors. Yellow is safe for line plots with sufficient weight.

### Pitfall 4: axes.prop_cycle syntax in mplstyle
**What goes wrong:** Using single quotes → `cycler('color', ['#...'])` → parse error "unterminated string literal"
**Why it happens:** The mplstyle parser uses a restricted eval; single-quoted strings inside nested calls fail.
**How to avoid:** Use double quotes: `cycler(color=["#E69F00", ...])` — verified working.

### Pitfall 5: Blues_trunc cmap not registered before use as string
**What goes wrong:** Setting `CMAP = "Blues_trunc"` and then passing `cmap=CMAP` to matplotlib calls fails with "Invalid colormap name"
**Why it happens:** `LinearSegmentedColormap.from_list` creates the object but doesn't register it in matplotlib's global registry.
**How to avoid:** Either (a) pass the cmap object directly (`cmap=SEQUENTIAL_BLUES`) or (b) call `plt.colormaps.register(SEQUENTIAL_BLUES, force=True)` once at import time. Recommend (a): pass object directly, keep `CMAP = "viridis"` unchanged.

### Pitfall 6: use_latex=True font size override
**What goes wrong:** `apply_style(use_latex=True)` currently overrides to 10pt. After mplstyle reduces base to 8pt, the latex path writes 10/12/10/9/9/9 pt — larger than the mplstyle. This creates inconsistency.
**Why it happens:** The `use_latex` override was designed relative to 11pt base; values are now wrong relative to 8pt base.
**How to avoid:** Update the `use_latex` block in `_style.py` to match the new 7-9pt range. `test_apply_style_latex_mode` only checks that `text.usetex is True` and `"serif" in font.family` — no font size assertions — so updating the sizes won't break that test.

## Code Examples

### STYL-01: Complete updated mplstyle

```ini
# emri_thesis.mplstyle — updated for Phase 29
# Source: matplotlib 3.10 rcParams documentation

# Backend — always non-interactive; set programmatically by apply_style()
# backend: Agg  (not set here; mplstyle files cannot change backend reliably)

# Figure defaults
figure.figsize: 6.4, 4.0
figure.dpi: 150
savefig.dpi: 300

# Font — reduced for REVTeX column widths (3.375" single, 7.0" double)
font.size:          8
axes.titlesize:     9
axes.labelsize:     8
xtick.labelsize:    7
ytick.labelsize:    7
legend.fontsize:    7

# PDF/PS font types — 42 = TrueType/Type1 (no Type 3 bitmap fonts)
pdf.fonttype: 42
ps.fonttype:  42

# LaTeX rendering (requires a TeX installation)
text.usetex: False

# Lines
lines.linewidth: 1.5

# Axes — remove top and right spines (publication style)
axes.grid:         False
axes.linewidth:    0.8
axes.spines.top:   False
axes.spines.right: False

# Ticks — inward ticks
xtick.direction: in
ytick.direction: in

# Legend — frameless
legend.frameon:    False
legend.framealpha: 0.8
legend.edgecolor:  0.8

# Color cycle — Okabe-Ito (Wong 2011, Nature Methods)
axes.prop_cycle: cycler(color=["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"])

# Rendering performance
agg.path.chunksize: 10000

# Default colormap
image.cmap: viridis

# Tight layout
figure.constrained_layout.use: True
```

### STYL-02: _colors.py replacement (conservative option)

Keep `CMAP = "viridis"` to avoid cascade test changes. Add `SEQUENTIAL_BLUES` and `ACCENT` as new exports.

```python
"""Centralized color palette for EMRI thesis plots — Okabe-Ito edition.

Palette source: Wong (2011) Nature Methods, doi:10.1038/nmeth.1618
Colorblind-safe: verified for deuteranopia, protanopia, tritanopia.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Okabe-Ito cycle (7 colors; black excluded — reserved for text/edges) ---
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

# --- Semantic role colors ---
TRUTH: str = "#009E73"      # bluish green — truth / reference lines
MEAN: str = "#D55E00"       # vermillion — mean / summary lines
EDGE: str = "#1a1a1a"       # near-black — histogram edges, outlines
REFERENCE: str = "#56B4E9"  # sky blue — secondary reference lines
ACCENT: str = "#E69F00"     # orange — accent for annotations/highlights

# --- Sequential Blues (truncated 0.1-0.85 to avoid near-white/near-black) ---
_blues_base = plt.colormaps["Blues"]
SEQUENTIAL_BLUES: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "Blues_trunc", _blues_base(np.linspace(0.1, 0.85, 256))
)

# --- Default colormap name (kept as viridis for backward compat; use SEQUENTIAL_BLUES for 2D plots) ---
CMAP: str = "viridis"
```

### pdffonts verification command

```bash
# Generate a test figure and verify no Type 3 fonts
uv run python -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, tempfile, os
fig, ax = plt.subplots()
ax.plot([0,1],[0,1])
ax.set_xlabel('test')
with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
    fname = f.name
fig.savefig(fname)
import subprocess
result = subprocess.run(['pdffonts', fname], capture_output=True, text=True)
print(result.stdout)
assert 'Type 3' not in result.stdout, 'Type 3 fonts found!'
print('PASS: no Type 3 fonts')
os.unlink(fname)
"
```

[VERIFIED: pdffonts is at `/usr/bin/pdffonts` version 26.03.0 — available on this machine]

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|-----------------|-------|
| `plt.get_cmap(name)` | `plt.colormaps[name]` | `get_cmap` deprecated in 3.7, removed in 3.11 |
| `cycler('color', ['...'])` in mplstyle | `cycler(color=["..."])` | Single-quote nested syntax fails in mplstyle parser |
| Type 3 fonts (default) | `pdf.fonttype: 42` | Required for journal submission (IEEE, APS, etc.) |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Recommended font sizes: 8pt base, 9pt title, 7pt ticks/legend for REVTeX | Architecture Patterns P4, Code Examples STYL-01 | Figures could look too small or too large in the paper. Planner should flag for user confirmation before locking sizes. |
| A2 | TRUTH/MEAN/REFERENCE semantic colors should be drawn from Okabe-Ito | Architecture Patterns P3 | If the user prefers the old green/red semantics, existing plots would change color. The current green TRUTH and red MEAN are not Okabe-Ito; changing them affects all 10 importing modules. |

## Open Questions

1. **Font size exact values (A1)**
   - What we know: REVTeX single col = 3.375in; 7-9pt range specified in requirements
   - What's unclear: Is the split `font.size=8, title=9, ticks/legend=7` the intended mapping, or does the user want a flat 8pt everywhere, or 9pt base?
   - Recommendation: Default to `font.size=8, axes.titlesize=9, axes.labelsize=8, xtick/ytick/legend=7` as a reasonable 7-9pt spread. Planner should note this as a tuning decision.

2. **CMAP change scope**
   - What we know: STYL-02 requires Blues as the "sequential emphasis" colormap; `CMAP` currently equals `"viridis"` and is tested by equality
   - What's unclear: Should `CMAP` be renamed to `"Blues_trunc"` (requires test update + potential breakage in 3 plot modules that pass `cmap=CMAP` to `imshow`/`scatter`/`hist2d`) or should `CMAP` stay as `"viridis"` and `SEQUENTIAL_BLUES` be a new addition?
   - Recommendation: Keep `CMAP = "viridis"` for backward compat; add `SEQUENTIAL_BLUES` as a new object. Downstream phase (PFIG-03 etc.) can switch heatmaps to use `SEQUENTIAL_BLUES` intentionally.

3. **Semantic color reassignment (A2)**
   - What we know: `TRUTH` is currently green (`#2ca02c`), `MEAN` is red (`#d62728`) — these are visually meaningful
   - What's unclear: Should semantic colors move to Okabe-Ito equivalents (bluish-green for TRUTH, vermillion for MEAN)?
   - Recommendation: Yes — use Okabe-Ito semantic assignments for consistency with the cycle. This changes truth/mean marker colors in existing figures but improves colorblind safety.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| matplotlib | Style changes | Yes | 3.10.8 | — |
| pdffonts | QUAL-03 verification | Yes | 26.03.0 (poppler) | — |
| numpy | Blues truncation | Yes | installed | — |

No missing dependencies.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest + pytest-cov |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest master_thesis_code_test/plotting/test_style.py master_thesis_code_test/plotting/test_colors.py -v --tb=short --no-cov` |
| Full suite command | `uv run pytest -m "not gpu and not slow" --tb=short -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STYL-01 | rcParams match new mplstyle values | unit | `uv run pytest master_thesis_code_test/plotting/test_style.py -v` | Yes (needs update) |
| STYL-01 | No Type 3 fonts in generated PDF | smoke | inline pdffonts check (see Code Examples) | No — Wave 0 gap |
| STYL-02 | CYCLE has >= 6 hex entries | unit | `uv run pytest master_thesis_code_test/plotting/test_colors.py -v` | Yes (needs update) |
| STYL-02 | TRUTH/MEAN/EDGE/REFERENCE are valid hex | unit | included in test_colors.py | Yes |
| STYL-03 | preset="single" returns 3.375in width | unit | `uv run pytest master_thesis_code_test/plotting/test_helpers.py::test_get_figure_preset_single_width -v` | Yes |
| STYL-03 | No-args returns mplstyle default size | unit | `uv run pytest master_thesis_code_test/plotting/test_helpers.py::test_get_figure_no_args_uses_mplstyle_default -v` | Yes (needs update — 6.4 x 4.0 default check may need adjustment) |

### Sampling Rate
- **Per task commit:** `uv run pytest master_thesis_code_test/plotting/ -v --no-cov`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow" --tb=short -q`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/plotting/test_style.py` — update `test_apply_style_default_unchanged` and `test_rcparams_snapshot` to reflect new rcParam values
- [ ] `master_thesis_code_test/plotting/test_colors.py` — update `test_cmap_is_viridis` if CMAP changes; add `test_sequential_blues_is_cmap_object` and `test_accent_is_hex`; update CYCLE entries to Okabe-Ito
- [ ] Add `test_no_type3_fonts_in_pdf` to `test_style.py` — saves a figure and calls `pdffonts` to assert no Type 3 fonts

## Security Domain

Not applicable. This phase contains no authentication, session management, input from external sources, cryptography, or network communication. All changes are local file edits to a Python visualization library.

## Sources

### Primary (HIGH confidence)
- Local codebase analysis — current `emri_thesis.mplstyle`, `_colors.py`, `_helpers.py`, `_style.py`, all 10 importing modules, test files
- Verified via `uv run python` — all rcParam names, cycler syntax, colormaps API, pdffonts output

### Secondary (MEDIUM confidence)
- Wong (2011) "Points of view: Color blindness", Nature Methods 8:441 doi:10.1038/nmeth.1618 — Okabe-Ito palette source [ASSUMED: hex values from training data, consistent with the well-known palette]
- matplotlib 3.10 rcParams — `axes.spines.top/right`, `xtick.direction`, `legend.frameon`, `pdf.fonttype` confirmed settable

### Tertiary (LOW confidence)
- A1: REVTeX 7-9pt font size guidance is a common recommendation for two-column journals; not verified against the specific paper template measurements

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — matplotlib 3.10.8 confirmed installed, all rcParam names verified
- Architecture: HIGH — all code paths inspected, test files read, syntax verified
- Pitfalls: HIGH — test pin values read directly from source; syntax error reproduced and fixed

**Research date:** 2026-04-07
**Valid until:** 2026-07-07 (matplotlib rcParams API is stable; no upcoming removals in 3.10)
