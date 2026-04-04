# Phase 15: Style Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 15-style-infrastructure
**Areas discussed:** Figure sizing strategy, Color palette design, LaTeX label migration, apply_style(use_latex=True) behavior

---

## Figure Sizing Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Named constants in `_helpers.py` | `SINGLE_COLUMN = (3.5, 2.6)` passed manually to `get_figure(figsize=...)` | |
| Preset parameter on `get_figure()` | `get_figure(preset="single")` / `get_figure(preset="double")` with internal mapping | ✓ |
| Change mplstyle default | Set `figure.figsize` to single-column width in mplstyle | |

**User's choice:** Preset parameter on `get_figure()` (recommended by Claude)
**Notes:** Default (no preset) keeps mplstyle as-is so existing code/tests stay untouched. Raw `figsize` still accepted for one-offs.

---

## Color Palette Design

| Option | Description | Selected |
|--------|-------------|----------|
| Semantic names only | `TRUTH`, `MEAN`, `POSTERIOR`, `HIST_EDGE` tied to physics plot roles | |
| Ordered cycle + semantic names | Small ordered cycle for multi-line plots plus semantic names for special roles | ✓ |
| Minimal — just current roles | Only replace the 3-4 ad-hoc strings that exist now | |

**User's choice:** Ordered cycle + semantic names
**Notes:** Covers multi-line plots (individual posteriors) and special-role colors. Start with existing ~4 roles, grow in later phases.

---

## LaTeX Label Migration

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 15 updates all labels now | Migrate every xlabel/ylabel across all 6 plot modules in this phase | |
| Phase 15 infrastructure only, Phase 17 migrates | Establish convention/constants in 15; bulk label rewrite in 17 | ✓ |

**User's choice:** Phase 15 infrastructure only, Phase 17 migrates

**Follow-up: Notation depth**

| Option | Description | Selected |
|--------|-------------|----------|
| Physics symbols only | `$M_\bullet$`, `$d_L$` but units stay plain: `[Mpc]` | |
| Symbols and units | `$M_\bullet \, [M_\odot]$`, `$d_L \, [\mathrm{Mpc}]$` — fully typeset | ✓ |
| You decide | Claude picks per label | |

**User's choice:** Fully typeset — symbols and units in mathtext

---

## apply_style(use_latex=True) Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Just `text.usetex` | Flip usetex flag only, minimal change | |
| `text.usetex` + font swap | Also switch to serif/Computer Modern | |
| `text.usetex` + font + sizing | Full TeX engine, serif fonts, paper-matched font sizes | ✓ |

**User's choice:** Option 3 — full TeX engine, serif/CM fonts, paper-matched font sizes
**Notes:** Figures target arXiv paper (not thesis). REVTeX two-column widths (~3.375in / ~7.0in). User clarified output is for a paper, not the thesis document.

---

## Claude's Discretion

- Exact height calculations for figure presets (aspect ratios)
- Internal structure of `_colors.py` (constants, enum, or dict)
- Whether to provide label constants module or docstring convention
- Font size values for `use_latex=True` mode

## Deferred Ideas

None — discussion stayed within phase scope.
