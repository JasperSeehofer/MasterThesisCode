# Phase 31: Catalog-Only Diagnostic - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 31-catalog-only-diagnostic
**Areas discussed:** Catalog-only bypass mechanism, Per-event diagnostic logging, Result interpretation

---

## Catalog-Only Bypass Mechanism

| Option | Description | Selected |
|--------|-------------|----------|
| CLI flag --catalog_only | New boolean on Arguments, threaded to worker. Skip completion term integral entirely and force f_i=1.0. Matches existing CLI pattern and success criteria. | ✓ |
| Compute but override | Still compute L_comp for diagnostic logging, but force f_i=1.0 in the combination formula. Slightly slower but shows what L_comp would have contributed. | |
| Both: flag + log L_comp | CLI flag skips expensive integral; separate --diagnostic_log flag computes and logs L_comp without using it. Two orthogonal flags. | |

**User's choice:** CLI flag --catalog_only (Recommended)
**Notes:** Clean match to success criteria wording and existing CLI patterns.

---

## Per-Event Diagnostic Logging — Format

| Option | Description | Selected |
|--------|-------------|----------|
| CSV file | One row per (event, h_value). Columns: event_idx, h, f_i, L_cat_no_bh, L_cat_with_bh, L_comp, combined_no_bh, combined_with_bh. Written to working_dir/diagnostics/. | ✓ |
| JSONL file | One JSON object per line, same fields. More flexible for nested data. | |
| You decide | Claude picks based on downstream consumption patterns. | |

**User's choice:** CSV file (Recommended)

## Per-Event Diagnostic Logging — When to Log

| Option | Description | Selected |
|--------|-------------|----------|
| Always during --evaluate | Diagnostic CSV always written. Small overhead. Useful for all subsequent phases without extra flag. | ✓ |
| Separate --diagnostic_log flag | Only write when explicitly requested. | |
| Only with --catalog_only | Tied to catalog-only mode only. | |

**User's choice:** Always during --evaluate (after discussion about HPC performance)
**Notes:** User was concerned about HPC performance impact. Claude explained: ~16k rows of 8 floats (~100KB), no extra computation since values are already computed in the worker. Zero overhead on bottleneck (fixed_quad integrals). User confirmed.

---

## Result Interpretation

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse Phase 30 infra | Run --evaluate --catalog_only --compare_baseline. Existing comparison report shows delta automatically. | |
| Add diagnostic summary | Phase 30 infra plus aggregate per-event CSV: mean f_i, L_comp stats, fraction of events where L_comp pulls lower. | |
| Both | Phase 30 comparison for headline numbers + diagnostic summary for deeper insight. | ✓ |

**User's choice:** Both
**Notes:** Headline comparison via Phase 30 infrastructure plus diagnostic summary explaining WHY the bias changes.

---

## Claude's Discretion

- Diagnostic CSV writing placement (worker vs post-collection)
- Diagnostic summary format and aggregation metrics
- L_comp directional bias metric design

## Deferred Ideas

None.
