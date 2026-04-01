# Phase 14: Test Infrastructure & Safety Net - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-01
**Phase:** 14-test-infrastructure
**Areas discussed:** Test data strategy, Test granularity, rcParams snapshot scope, Test organization

---

## Test Data Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Inline synthetic data | Each test builds minimal fake data inline. No external files. | |
| Shared fixtures module | conftest.py with reusable pytest fixtures (fake_crb_dataframe, etc.) | ✓ |
| You decide | Claude picks per function | |

**User's choice:** Shared fixtures module
**Notes:** User noted existing conftest.py infrastructure should be extended.

---

## Test Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| One test per factory | 22 individual tests, assert (Figure, Axes) return | ✓ |
| One test per module | 7 parametrized tests, one per plot module | |
| Per-factory + basic content | 22 tests that also check axes labels/titles | |

**User's choice:** One test per factory (Recommended)
**Notes:** Pure smoke tests — no content assertions.

---

## rcParams Snapshot Scope

| Option | Description | Selected |
|--------|-------------|----------|
| All 16 from mplstyle | Pin every rcParam set by emri_thesis.mplstyle | ✓ |
| Critical subset only | Pin ~6 key params | |
| You decide | Claude picks the balance | |

**User's choice:** All 16 from mplstyle (Recommended)
**Notes:** None.

---

## Test Organization

| Option | Description | Selected |
|--------|-------------|----------|
| One file per plot module | test_bayesian_plots.py, test_catalog_plots.py, etc. | ✓ |
| Single test_plot_smoke.py | All 22 smoke tests in one file | |
| You decide | Claude organizes as best fits | |

**User's choice:** One file per plot module (Recommended)
**Notes:** Mirrors the source layout. rcParams regression stays in test_style.py.

---

## Claude's Discretion

- Exact synthetic data shapes/values for each factory's fixtures
- Whether to use `@pytest.mark.parametrize` for functions with similar signatures

## Deferred Ideas

None — discussion stayed within phase scope.
