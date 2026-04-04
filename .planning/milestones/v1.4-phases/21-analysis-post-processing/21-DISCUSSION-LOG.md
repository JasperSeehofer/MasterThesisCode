# Phase 21: Analysis & Post-Processing - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 21-analysis-post-processing
**Areas discussed:** Script interface, Diagnostic output, Zero-handling default, Output artifacts

---

## Script Interface

| Option | Description | Selected |
|--------|-------------|----------|
| CLI subcommand | Add --combine flag to __main__.py alongside --evaluate and --snr_analysis | ✓ |
| Standalone script | New top-level script (scripts/combine_posteriors.py) decoupled from main pipeline | |
| You decide | Claude picks best fit | |

**User's choice:** CLI subcommand (--combine flag)
**Notes:** User initially asked whether this should be part of a visualization CLI or stay separate. Agreed that combination is data processing (not visualization) and fits the existing __main__.py subcommand pattern.

---

## Diagnostic Output

| Option | Description | Selected |
|--------|-------------|----------|
| Generated markdown | Script writes diagnostic_report.md to working directory | ✓ |
| JSON + stdout | Machine-readable JSON plus console summary | |
| You decide | Claude picks most practical format | |

**User's choice:** Generated markdown
**Notes:** None

---

## Zero-Handling Default

| Option | Description | Selected |
|--------|-------------|----------|
| Option 1 (exclude zeros) | Exclude events with zero likelihood, loses 3-21% of events | |
| Naive (no handling) | Naive multiplication matching current behavior | |
| Option 3 (physics floor) | Physics-motivated floor from Phase 22 | ✓ (with fallback) |

**User's choice:** Option 3 as default with Option 1 as fallback
**Notes:** User specified "Option 3 with option 1 fall back" — physics floor when available, automatic fallback to exclude-zeros until Phase 22 delivers the floor implementation.

---

## Output Artifacts

| Option | Description | Selected |
|--------|-------------|----------|
| JSON only | combined_posterior.json with joint H0 posterior + metadata | ✓ |
| JSON + plot | JSON plus basic H0 posterior PNG | |
| JSON + plot + comparison | All above plus ANAL-02 comparison table as markdown | |

**User's choice:** JSON only
**Notes:** Plotting handled separately by existing/future plotting infrastructure.

---

## Claude's Discretion

- Internal module structure and function decomposition
- JSON schema for combined output
- Comparison table format for ANAL-02
- Log-shift-exp implementation details

## Deferred Ideas

None — discussion stayed within phase scope
