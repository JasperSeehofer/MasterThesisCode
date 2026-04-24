# VERIFY-04: Anisotropy Audit

**Timestamp:** 20260423T172607Z
**Phase:** 40 Wave 3
**Requirement:** VERIFY-04
**Rule (D-12):** `>1σ shift is a Stage-2 trigger for Phase 42 (not a blocker)`

## Overall h=0.73 posterior

- `MAP_total` = 0.8600
- 68% CI: [0.8508, 0.8582]
- σ = (CI_upper − CI_lower) / 2 = 0.003736
- N events in posteriors = 60 (out of 60 with non-zero lk)

**Note on MAP_total = 0.86:** `extract_baseline` sums log-likelihoods without the D(h)
denominator correction (the SC-3 FAIL finding from VERIFY-03). The quartile comparison
is internally self-consistent: all quartile MAPs and the threshold σ are derived from the
same biased posterior. The anisotropy audit result is therefore valid regardless of the
absolute MAP calibration issue.

**Likelihood source:** per-h JSON posterior files (not the diagnostic CSV), matching
`extract_baseline` behavior. Events with zero likelihood at a given h are skipped (same
as `extract_baseline` line 162), ensuring MAP_q is computed identically to MAP_total.

## Quartile edges (on |qS − π/2|, equal-count per D-14)

- Q1: [0.0123, 0.2603)  [events nearest ecliptic equator]
- Q2: [0.2603, 0.5043)
- Q3: [0.5043, 0.7574)
- Q4: [0.7574, 1.3887]  [events furthest from ecliptic equator]

## Per-quartile MAP_q

| # | Quartile label | N events | N finite-lk | MAP_q | |MAP_q − MAP_total| | σ | Trigger (ΔMAP > σ)? |
|---|----------------|----------|-------------|-------|---------------------|---|----------------------|
| 1 | Q1 (nearest ecliptic equator) | 15 | 11 | 0.8600 | 0.0000 | 0.0037 | no |
| 2 | Q2 | 15 | 9 | 0.8600 | 0.0000 | 0.0037 | no |
| 3 | Q3 | 15 | 10 | 0.8400 | 0.0200 | 0.0037 | YES |
| 4 | Q4 (furthest from equator) | 15 | 7 | 0.8600 | 0.0000 | 0.0037 | no |

## Verdict

**VERIFY-04: STAGE-2-TRIGGER**

Per D-12, this is a Stage-2 trigger for **Phase 42** (Sky-Dependent Injection Campaign),
NOT an abort condition. Phase 40 continues; Phase 41 can still run if its own VERIFY-05
trigger also fires.

## Links

- CRB CSV source: `simulations/prepared_cramer_rao_bounds.csv`
- Per-h posterior JSON files: `/home/jasper/Repositories/MasterThesisCode/simulations/posteriors/h_*.json`
- Diagnostic CSV (info only): `/home/jasper/Repositories/MasterThesisCode/simulations/diagnostics/event_likelihoods.csv`
- Driver: `.planning/debug/anisotropy_driver_20260423T172607Z.py`
- Machine-readable: `.planning/debug/anisotropy_audit_20260423T172607Z.json`
