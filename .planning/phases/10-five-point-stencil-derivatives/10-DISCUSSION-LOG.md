# Phase 10: Five-Point Stencil Derivatives - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md -- this log preserves the alternatives considered.

**Date:** 2026-03-29
**Phase:** 10-five-point-stencil-derivatives
**Areas discussed:** Timeout strategy, Condition number handling, Old method retention, Stencil API refactor, Epsilon values, GPU memory pressure

---

## Timeout Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Fixed 120s | 4x the current 30s, matching ~4x waveform increase | |
| Fixed 180s | 6x current value, extra headroom for slow waveforms | |
| CLI flag --crb_timeout | Configurable via arguments.py | |

**User's choice:** 90s (custom value via "Other")
**Notes:** User chose 3x the current value rather than the suggested 4x or 6x. Rationale: not all waveforms will be worst-case, so 3x provides sufficient headroom. Phase 11 validation will confirm.

---

## Condition Number Handling

### Round 1

| Option | Description | Selected |
|--------|-------------|----------|
| Log only | Log condition number at INFO, no threshold, no skipping | |
| Log + warn above threshold | Log all, WARNING above threshold (e.g., 10^12) | |
| Log + skip if singular | Log condition number, skip on LinAlgError | |

**User's choice:** Asked for explanation of condition numbers and why "log only" was recommended.
**Notes:** User wanted to understand what condition numbers are and why ill-conditioning matters. After explanation of kappa(Gamma) = lambda_max/lambda_min and its impact on CRB accuracy, user agreed that clearly erroneous results should be skipped.

### Round 2 (after explanation)

| Option | Description | Selected |
|--------|-------------|----------|
| Skip on inversion failure | Log cond number, skip on LinAlgError or negative CRB diagonals | ✓ |
| Skip above threshold + on failure | Same + skip when cond > 10^15 | |
| Log + warn, never skip | Always compute, let Phase 11 decide filtering | |

**User's choice:** Skip on inversion failure (Recommended)
**Notes:** No arbitrary threshold -- let numpy's LinAlgError and physical constraints (non-negative variances) be the gate.

---

## Old Method Retention

| Option | Description | Selected |
|--------|-------------|----------|
| Remove it | Dead code, git history preserves it | |
| Keep with deprecation note | Add docstring note, keep available | |
| Toggle like Phase 9 | Add use_five_point_stencil: bool field, default True | ✓ |

**User's choice:** Toggle like Phase 9
**Notes:** Enables regression comparison between forward-diff and 5-point stencil results.

---

## Stencil API Refactor

| Option | Description | Selected |
|--------|-------------|----------|
| Refactor to loop internally | Match forward-diff API: loop over all params, return dict | ✓ |
| Keep per-parameter, add wrapper | Keep existing method, add wrapper that loops | |
| Claude's discretion | Let Claude pick cleanest structure | |

**User's choice:** Refactor to loop internally (Recommended)
**Notes:** Clean swap in compute_fisher_information_matrix() -- both methods have identical signatures.

---

## Epsilon Values

| Option | Description | Selected |
|--------|-------------|----------|
| Keep current values | 1e-6 default, 2*epsilon still negligible vs parameter ranges | |
| Review per-parameter in Phase 11 | Keep 1e-6 now, log ParameterOutOfBoundsError frequency | ✓ |
| Reduce to 1e-7 | Smaller footprint but more numerical noise | |

**User's choice:** Review per-parameter in Phase 11
**Notes:** Data-driven approach -- log bounds violations during validation campaign, tune if significant.

---

## GPU Memory Pressure

| Option | Description | Selected |
|--------|-------------|----------|
| No special handling | H100 has 80 GB, peak ~4 waveforms, no concern | |
| Add per-parameter cleanup | Explicit del of intermediates after each parameter | |
| Log peak GPU usage | Use MemoryManagement to log before/after derivative loop | ✓ |

**User's choice:** Log peak GPU usage
**Notes:** Diagnostic data for Phase 11 without changing computation flow.

---

## Claude's Discretion

- Internal loop structure and intermediate waveform storage/deletion
- Cleanup of parameter_space mutation pattern in five_point_stencil_derivative()

## Deferred Ideas

None -- discussion stayed within phase scope.
