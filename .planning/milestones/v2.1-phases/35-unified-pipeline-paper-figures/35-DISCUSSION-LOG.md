# Phase 35: Unified Figure Pipeline & Paper Figures - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 35-unified-pipeline-paper-figures
**Areas discussed:** Pipeline merge strategy, Paper figure polish scope, CI calculation consistency, Output organization

---

## Pipeline Merge Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Integrate into manifest | Add paper_figures.py functions as entries 16-19 in existing manifest. Remove standalone main(). Data dir passed as parameter. | ✓ |
| Keep paper figs separate | Keep paper_figures.py as-is with own entry point. Add --generate_paper_figures flag. | |
| Replace manifest entirely | Rewrite generate_figures() with paper figures as primary, demote thesis figures. | |

**User's choice:** Integrate into manifest (Recommended)
**Notes:** None

### Data Path

| Option | Description | Selected |
|--------|-------------|----------|
| Same data_dir | Paper figure functions accept data_dir as parameter, same as thesis figures. | ✓ |
| Subdirectory convention | Paper figures look in <data_dir>/eval_corrected_full/ by default. | |

**User's choice:** Same data_dir (Recommended)

---

## Paper Figure Polish Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Style inheritance only | Just ensure apply_style() works. Fix regressions but don't redesign. | |
| Light touch-up | Style inheritance + minor tweaks: consistent font sizes, legend placement, axis labels. | |
| Full visual rework | Redesign layouts, adjust spacing, add annotations, reconsider color choices. | ✓ |

**User's choice:** Full visual rework

### KDE Smoothing

| Option | Description | Selected |
|--------|-------------|----------|
| Conservative KDE | Light Gaussian KDE, Scott's rule bandwidth. MAP preserved within one grid spacing. | ✓ |
| Aggressive KDE | Heavier smoothing, manually tuned bandwidth. MAP may shift. | |
| Interpolation instead | Cubic spline interpolation. Passes through all grid points exactly. | |

**User's choice:** Conservative KDE (Recommended)

### Grid Resolution Detection

| Option | Description | Selected |
|--------|-------------|----------|
| Auto-detect from data | Infer grid spacing from h_values array via np.diff. Works with any grid. | ✓ |
| Config parameter | Add grid_resolution parameter to figure functions. | |

**User's choice:** Auto-detect from data (Recommended)

---

## CI Calculation Consistency

| Option | Description | Selected |
|--------|-------------|----------|
| Extract to _helpers.py | Add compute_credible_interval() to _helpers.py. Both modules call it. Unit tested. | ✓ |
| Extract to _data.py | Same consolidation but in _data.py alongside data processing utilities. | |
| Keep duplicated | Leave implementations where they are. | |

**User's choice:** Extract to _helpers.py (Recommended)

### CI Unit Test Distribution

| Option | Description | Selected |
|--------|-------------|----------|
| Gaussian | Test against Gaussian where 68% CI = ±σ. | |
| Gaussian + uniform | Test both Gaussian (peaked) and uniform (flat) distributions. | ✓ |
| You decide | Claude picks appropriate test distributions. | |

**User's choice:** Gaussian + uniform

---

## Output Organization

| Option | Description | Selected |
|--------|-------------|----------|
| Single figures/ dir | All figures to <dir>/figures/. Paper figures use paper_ prefix. | ✓ |
| Subdirectories | Paper to <dir>/figures/paper/, thesis to <dir>/figures/thesis/. | |
| Keep paper/ separate | Paper figures to paper/figures/ (LaTeX convention). | |

**User's choice:** Single figures/ dir (Recommended)

### Interactive Figures

| Option | Description | Selected |
|--------|-------------|----------|
| Keep separate flags | Interactive stays behind --generate_interactive. Different format, different audience. | ✓ |
| Add --all flag | Add --generate_all that runs both. Keep individual flags for granularity. | |
| Merge into one flag | --generate_figures produces both PDF and HTML. | |

**User's choice:** Keep separate flags (Recommended)

---

## Claude's Discretion

- Figure numbering scheme within unified manifest
- Internal refactoring of data loaders to work with manifest pattern
- Exact KDE bandwidth selection details
- Specific visual rework choices for each paper figure

## Deferred Ideas

None — discussion stayed within phase scope
