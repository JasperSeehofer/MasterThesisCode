# Phase 30: Baseline & Evaluation Infrastructure - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 30-baseline-evaluation-infrastructure
**Areas discussed:** Baseline capture, Comparison trigger, Output format, Script reuse

---

## Baseline Capture

| Option | Description | Selected |
|--------|-------------|----------|
| New CLI flag (Recommended) | Add --save_baseline to --evaluate. After full sweep, extract MAP h, 68% CI, bias %, event count and write baseline.json. | ✓ |
| Post-processing script | Standalone script reads existing posterior JSONs and extracts baseline metrics. Separate manual step. | |
| Automatic on every run | Every --evaluate run auto-saves metrics. First one becomes baseline, subsequent ones compared. | |

**User's choice:** New CLI flag
**Notes:** None

### Follow-up: Sweep requirement

| Option | Description | Selected |
|--------|-------------|----------|
| Full sweep only (Recommended) | Baseline requires full h-grid (e.g., 27 values). Single h-value runs don't produce enough for CI/bias. | ✓ |
| Either mode | Works with single h-value or full sweep. More flexible but single-point baselines not useful. | |

**User's choice:** Full sweep only

---

## Comparison Trigger

| Option | Description | Selected |
|--------|-------------|----------|
| New CLI flag (Recommended) | Add --compare_baseline <path> to --evaluate. Computes new posterior AND generates comparison report. | ✓ |
| Separate CLI command | New top-level flag --compare <baseline.json> <new_run_dir> for report-only. Decoupled from evaluation. | |
| Automatic detection | Every --evaluate checks if baseline.json exists. If found, auto-generates comparison. | |

**User's choice:** New CLI flag

### Follow-up: Standalone mode

| Option | Description | Selected |
|--------|-------------|----------|
| Yes, both modes | --compare_baseline works with --evaluate (run + compare) AND standalone (just compare two existing result sets). | ✓ |
| Only with --evaluate | Comparison only during evaluation. Must re-run full sweep to regenerate report. | |

**User's choice:** Yes, both modes

---

## Output Format & Location

### Fields

| Option | Description | Selected |
|--------|-------------|----------|
| Core metrics (Recommended) | MAP h, 68% CI, CI width, bias %, event count, h-grid + log-posteriors. | |
| Core + per-event summary | Same plus per-event: d_L, SNR, sigma(d_L)/d_L, condition number, quality filter pass/fail. | ✓ |
| Full reproduction | Everything above plus git commit, CLI args, timestamp, CRB catalog path. | |

**User's choice:** Core + per-event summary

### Location

| Option | Description | Selected |
|--------|-------------|----------|
| .planning/debug/ (Recommended) | Alongside existing debug artifacts. Committed to git for cross-phase reference. | ✓ |
| Working dir / evaluation/ | In evaluation output directory. Closer to data but scattered across runs. | |
| New baselines/ directory | Dedicated top-level dir. Clean but adds another directory for temporary tool. | |

**User's choice:** .planning/debug/

---

## Script Reuse

| Option | Description | Selected |
|--------|-------------|----------|
| Refactor into package (Recommended) | Extract comparison logic from scripts/compare_posterior_bias.py into main package. CLI flags call this module. | ✓ |
| Build fresh, keep old | New comparison tool from scratch. Leave old script as-is. Risk: two tools that drift. | |
| You decide | Claude picks approach. | |

**User's choice:** Refactor into package

---

## Claude's Discretion

- Exact module placement within the package
- Comparison report markdown formatting and ASCII chart style
- Whether old compare_posterior_bias.py is kept as wrapper or removed
- Implementation of h-grid sweep detection

## Deferred Ideas

None — discussion stayed within phase scope.
