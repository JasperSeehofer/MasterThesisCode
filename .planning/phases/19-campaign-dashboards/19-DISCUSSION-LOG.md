# Phase 19: Campaign Dashboards & Batch Generation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 19-campaign-dashboards
**Areas discussed:** Dashboard layout, Batch generation interface, File size optimization, Figure manifest
**Mode:** Auto (all areas auto-selected with recommended defaults)

---

## Dashboard Layout

| Option | Description | Selected |
|--------|-------------|----------|
| 2x2 grid via subplot_mosaic | H0 posterior, SNR, detection yield, sky map | ✓ |
| Asymmetric mosaic | Large H0 posterior + smaller panels | |
| Separate dashboard per topic | Multiple summary pages | |

**User's choice:** [auto] 2x2 grid via subplot_mosaic (recommended default)
**Notes:** Matches success criterion #1 requirements exactly. Standard thesis layout.

---

## Batch Generation Interface

| Option | Description | Selected |
|--------|-------------|----------|
| Implement generate_figures() stub | Use existing CLI --generate_figures flag | ✓ |
| New standalone script | Separate entry point in scripts/ | |
| Makefile-based | GNU Make with figure targets | |

**User's choice:** [auto] Implement generate_figures() stub (recommended default)
**Notes:** Stub already exists at main.py:608. Path of least resistance.

---

## File Size Optimization

| Option | Description | Selected |
|--------|-------------|----------|
| rasterized=True + size warning | Hybrid vector/raster, log warning >2MB | ✓ |
| Full rasterization | All plots as raster in PDF | |
| Post-hoc compression | ghostscript/mutool after save | |

**User's choice:** [auto] rasterized=True + size warning (recommended default)
**Notes:** Matches success criterion #3. Preserves vector text/axes.

---

## Figure Manifest

| Option | Description | Selected |
|--------|-------------|----------|
| Comprehensive thesis set | All 12+ figure types from factory functions | ✓ |
| Minimal results set | Only H0 posterior, SNR, sky map | |
| Configurable YAML manifest | External config file | |

**User's choice:** [auto] Comprehensive thesis set (recommended default)
**Notes:** Graceful degradation for missing data via skip-with-warning.

---

## Claude's Discretion

- Figure ordering in manifest
- Exact subplot_mosaic layout string
- Figure numbering scheme

## Deferred Ideas

None.
