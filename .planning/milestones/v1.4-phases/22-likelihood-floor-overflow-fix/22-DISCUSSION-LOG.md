# Phase 22: Likelihood Floor & Overflow Fix - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 22-likelihood-floor-overflow-fix
**Areas discussed:** Floor derivation approach, Floor placement, Underflow detection, Validation approach

---

## Floor Derivation Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Catalog incompleteness | Floor = likelihood from unseen galaxy at catalog completeness edge | |
| Uniform prior fallback | Floor = flat prior over error volume, normalized by volume | |
| Minimum nonzero likelihood | Floor = smallest nonzero likelihood across all h-bins for this event | ✓ |

**User's choice:** Minimum nonzero likelihood (option 3), as a quick but thought-through stopgap. Catalog incompleteness will be added to the evaluation pipeline soon and may remove the problem entirely. Priority is no bias from the computation.

### Follow-up: Floor Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Per-event minimum | Min nonzero across h-bins for each event individually | ✓ |
| Global minimum | Single min nonzero across all events and h-bins | |
| Per-h-bin minimum | Min nonzero across events at each h value | |

**User's choice:** Per-event minimum (recommended)
**Notes:** Preserves relative event weights and avoids cross-event bias

---

## Floor Placement

| Option | Description | Selected |
|--------|-------------|----------|
| In combination (post-hoc) | Apply floor in combine_posteriors after JSONs loaded; single_host_likelihood unchanged | ✓ |
| Inside single_host_likelihood | Clamp return value at computation time | |
| Both layers | Floor in both places as belt-and-suspenders | |

**User's choice:** In combination (post-hoc) (recommended)
**Notes:** Keeps physics code clean, easy to swap out when catalog incompleteness arrives

---

## Underflow Detection

| Option | Description | Selected |
|--------|-------------|----------|
| Log warning only | Detect zeros, log with details, continue execution | |
| Switch to log-space in-place | Auto-switch to log-space when underflow detected | |
| Remove check_overflow entirely | Dead code since Phase 21 handles this at combination layer | ✓ |

**User's choice:** Remove check_overflow entirely
**Notes:** Log-space accumulation and floor handle numerical problems; check_overflow is dead weight

---

## Validation Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Compare MAP estimates | Quantitative MAP comparison against baselines | |
| Visual posterior comparison | Overlay plots for shape inspection | |
| Both MAP + visual | MAP comparison AND visual overlay | ✓ |

**User's choice:** Both MAP + visual

### Follow-up: MAP Shift Tolerance

| Option | Description | Selected |
|--------|-------------|----------|
| Within 0.05 of exclude | Moderate tolerance, realistic for 21% zero-events | ✓ |
| Within 0.02 of exclude | Tight tolerance | |
| No specific threshold | Visual inspection only | |

**User's choice:** Within 0.05 of exclude (recommended)

---

## Claude's Discretion

- Implementation details of per-event-min floor in combine_posteriors
- Floor logging/reporting format
- Test structure for edge cases
- Whether to modify existing physics-floor strategy stub or create new one

## Deferred Ideas

- Catalog incompleteness model (proper fix for zero-likelihood events)
- Full log-space accumulation inside evaluate pipeline itself
