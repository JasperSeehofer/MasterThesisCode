# Phase 11: Validation Campaign - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-29
**Phase:** 11-validation-campaign
**Areas discussed:** Campaign parameters, Comparison methodology, d_L threshold recalibration, Timeout & epsilon tuning

---

## Campaign Parameters

### Campaign Size

| Option | Description | Selected |
|--------|-------------|----------|
| Same as v1.1 smoke test | 3 tasks, 25 steps, seed 42. Direct comparison with v1.1 baseline. | |
| Slightly larger (5 tasks, 50 steps) | More statistical power, ~250 total steps. | |
| Minimal (3 tasks, 10 steps) | Quick sanity check. | |

**User's choice:** Same as v1.1 smoke test (Recommended)
**Notes:** Selected for direct apple-to-apple comparison with v1.1 baseline.

### Seed Choice

| Option | Description | Selected |
|--------|-------------|----------|
| Same seed 42 | Same EMRI parameters as v1.1. Clean A/B comparison. | |
| Different seed (e.g. 100) | Fresh parameter draws. Can't directly attribute differences to physics changes. | ✓ |

**User's choice:** Different seed (e.g. 100)
**Notes:** Uses seed 100, which matches the most recent v1.1 run.

### Step Count

| Option | Description | Selected |
|--------|-------------|----------|
| 10 steps (matches latest v1.1 run) | Last successful v1.1 run used 3 tasks, 10 steps, seed 100. | ✓ |
| 25 steps (original v1.1 plan) | More data per task, ~75 total events. | |

**User's choice:** 10 steps (matches latest v1.1 run)
**Notes:** Matches the most recent v1.1 baseline run parameters exactly.

---

## Comparison Methodology

### Metrics

| Option | Description | Selected |
|--------|-------------|----------|
| Detection rate | Fraction of events passing SNR threshold. | ✓ |
| SNR distribution | Per-event SNR values comparison. | ✓ |
| CRB magnitudes & condition numbers | Cramer-Rao bounds and Fisher matrix condition numbers. | ✓ |
| Wall time per event | Per-event timing to validate 90s timeout. | ✓ |

**User's choice:** All four metrics selected.
**Notes:** Comprehensive comparison across all available dimensions.

### Pass/Fail Criteria

| Option | Description | Selected |
|--------|-------------|----------|
| Directional checks | Qualitative sanity: lower SNRs, different CRBs, detection rate >0, no crashes/NaN. | ✓ |
| Quantitative bounds | Specific acceptable ranges (detection rate >5%, SNR reduction <50%). | |
| You decide | Claude determines criteria from observed data. | |

**User's choice:** Directional checks (Recommended)
**Notes:** Qualitative validation — physics changes should move metrics in expected directions.

### Report Format

| Option | Description | Selected |
|--------|-------------|----------|
| Summary table in terminal | Quick side-by-side comparison. | |
| Markdown report file | Permanent record in run directory. | ✓ |
| Both | Terminal + saved report. | |

**User's choice:** Markdown report file
**Notes:** Saved for thesis reference and future comparison.

---

## d_L Threshold Recalibration

### Recalibration Method

| Option | Description | Selected |
|--------|-------------|----------|
| Data-driven from validation run | Examine delta_d_L/d_L distribution, report percentiles, recommend threshold. | ✓ |
| Keep 10% for now | Don't change, just report distribution. Defer to Phase 12. | |
| Tighten to 5% | Preemptive tightening. Risk of discarding too many events. | |

**User's choice:** Data-driven from validation run (Recommended)
**Notes:** Will examine the distribution and recommend a threshold value.

### Code Update

| Option | Description | Selected |
|--------|-------------|----------|
| Update code if clear | If data clearly supports a specific threshold, update constants.py. | |
| Document only | Report recommendation but don't change code. Let Phase 12 apply. | ✓ |

**User's choice:** Document only
**Notes:** Threshold recommendation goes into comparison report for Phase 12 to act on.

---

## Timeout & Epsilon Tuning

### Timeout Response

| Option | Description | Selected |
|--------|-------------|----------|
| Report and increase for Phase 12 | Log timeout rate, recommend new value. Don't change code. | |
| Increase in this phase if needed | If >10% timeout, update main.py in this phase. | ✓ |
| Remove timeout entirely | Let events run to completion. Risk of indefinite hangs. | |

**User's choice:** Increase in this phase if needed
**Notes:** Keep Phase 12 clean by fixing issues as they arise in validation.

### Epsilon Tuning

| Option | Description | Selected |
|--------|-------------|----------|
| Tune if needed | If ParameterOutOfBoundsError rate is high, adjust per-parameter epsilons. | ✓ |
| Document only | Report error rate, defer changes to Phase 12. | |

**User's choice:** Yes, tune if needed (Recommended)
**Notes:** Same philosophy as timeout — fix issues in this phase rather than deferring.

---

## Claude's Discretion

- Comparison report structure and formatting
- rsync flags and file selection
- Whether to generate comparison plots
- sacct polling interval

## Deferred Ideas

None — discussion stayed within phase scope.
