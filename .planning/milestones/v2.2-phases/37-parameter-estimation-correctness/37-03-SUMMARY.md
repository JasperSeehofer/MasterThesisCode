---
phase: 37-parameter-estimation-correctness
plan: 03
type: verify
status: complete
completed: 2026-04-22
---

# 37-03 SUMMARY: Phase 37 Verification + State Advance

## Phase 37 Goal

Resolve all 6 parameter-estimation correctness findings (COORD-05, PE-01..PE-05) identified in the 2026-04-21 audit, verified against 7 success criteria.

## Commits Landed

| SHA | Message |
|-----|---------|
| `35cc79d` | `chore(37): software hygiene — Omega_m limits, SNR unification, C/1000, idempotency guard` |
| `55a6d99` | `[PHYSICS] PE-01: thread h_inj into set_host_galaxy_parameters` |
| `7429c6e` | `[PHYSICS] PE-02: per-parameter derivative_epsilon for all 14 EMRI parameters` |
| `16ce20f` | `[PHYSICS] PE-02: SC-3 regression tests for per-parameter derivative_epsilon` |

## SC Checklist

| SC | Criterion | Status | Evidence |
|----|-----------|--------|---------|
| 1 | h-ratio test: d_L(h=1.0) = 2× d_L(h=0.5), rtol=1e-10 | **PASS** | `test_parameter_space_h.py::test_set_host_galaxy_parameters_h_ratio` — 1 passed |
| 2 | TypeError raised when h omitted from set_host_galaxy_parameters | **PASS** | `test_parameter_space_h.py::test_set_host_galaxy_parameters_requires_h` — 1 passed |
| 3 | Fisher det change < 1% on 4 seeds after per-parameter epsilon | **PASS** | `pytest -k 'epsilon or pe02'` — 2 passed (SC-3 regression + epsilon stability) |
| 4 | Omega_m lower=0.04, upper=0.5 in LamCDMScenario | **PASS** | Python assertion: `LamCDMScenario().Omega_m.lower_limit == 0.04` ✓ |
| 5 | No literal 15 adjacent to snr_threshold in production code | **PASS** | `grep -rn --include='*.py' -E '\bsnr_threshold\s*[:=].*\b15\b'` — zero matches |
| 6 | SPEED_OF_LIGHT_KM_S = C / 1000 (derived, not hardcoded) | **PASS** | `abs(SPEED_OF_LIGHT_KM_S - C/1000) < 1e-9` ✓ |
| 7 | _map_angles_to_spherical_coordinates raises AssertionError on second call | **PASS** | Double-call Python test — AssertionError raised on second call ✓ |

## Phase 36 Regression

`uv run pytest master_thesis_code_test/test_coordinate_roundtrip.py -v` → **9 PASSED, 0 FAILED**

Phase 36 coordinate roundtrip tests remain GREEN after all Phase 37 changes.

## Full Test Suite

`uv run pytest -m "not gpu and not slow" -q` → **521 passed, 6 skipped** — no regressions.

## Physics Changes Summary

| REQ-ID | Change | Method |
|--------|--------|--------|
| PE-01 | `h_inj` threaded into `set_host_galaxy_parameters`; d_L now computed at injected h, making Fisher CRBs self-consistent | Limiting case: h=0.73 → identical; h-ratio test at 1e-10 rtol |
| PE-02 | Per-parameter `derivative_epsilon` for all 14 EMRI parameters | Vallisneri (2008) arXiv:gr-qc/0703086; Fisher det change < 1% on 4 seeds |

## Phase 40 VERIFY-02 Note

The Phase 36 regression pickle (`36-superset-regression.pkl`) was committed as a CRB anchor
before PE-02 was applied. The per-parameter epsilon change (PE-02) alters derivative step sizes
and may slightly perturb `fisher_sky_2x2` values. Phase 40 VERIFY-02 should use post-Phase-37
CRB values as its reference baseline, **not** the Phase 36 pickle.

## What This Unblocks

Phase 38 (Statistical Correctness — STAT-01..STAT-04): stable CRB inputs with per-parameter
epsilon landed; h-threading ensures Fisher bounds are computed at the correct cosmology. STAT-03
always routes to GPD; STAT-01 may require GPD if the L_cat equivalence proof requires a fix.
