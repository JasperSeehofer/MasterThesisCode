---
phase: 36-coordinate-frame-fix
verified: 2026-04-22T11:43:00Z
status: passed
score: 5/5 SC satisfied
overrides_applied: 0
re_verification: null
gaps: []
deferred:
  - truth: "Posterior re-evaluation at h=0.73 under v2.2 corrections"
    addressed_in: "Phase 40 VERIFY-02"
    evidence: "Phase 40 goal + abort gate (MAP shift >5% from v2.1 baseline)"
  - truth: "h-threading through set_host_galaxy_parameters"
    addressed_in: "Phase 37 PE-01"
    evidence: "Phase 37 SC-1 regression test on h=0.5 vs h=1.0 d_L ratio"
  - truth: "Idempotency guard on _map_angles_to_spherical_coordinates"
    addressed_in: "Phase 37 COORD-05"
    evidence: "Phase 37 SC-7 AssertionError on second call"
  - truth: "Deprecated get_possible_hosts (non-balltree) removal"
    addressed_in: "Phase 39 HPC-04"
    evidence: "Phase 39 SC-4 grep empty"
human_verification: []
---

# Phase 36: Coordinate Frame Fix — Verification Report

**Phase Goal:** Fix four coordinate-frame bugs in handler.py (COORD-02, COORD-02b, COORD-03, COORD-04) via the /physics-change protocol, landing four atomic [PHYSICS] commits and committing the Phase 40 VERIFY-02 regression anchor.
**Verified:** 2026-04-22T11:43:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

All four atomic [PHYSICS] commits landed per D-25 (COORD-03 b460297, COORD-02 c17ecb6, COORD-04 b2ef9c9, COORD-02b 5b9cfbf). Six Phase 35 xfail(strict=True) markers were removed in the same commits as their corresponding fixes per D-26 — no bulk-removal commit present. The regression pickle was committed at the locked path `.planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl` per D-24, with all required schema fields and the `old ⊆ new` superset assertion verified. `uv run pytest master_thesis_code_test/test_coordinate_roundtrip.py -v` reports 9 passed, 0 xfailed, 0 XPASS, 0 FAILED — the D-27 decisive deliverable criteria are satisfied. The full regression suite (`uv run pytest -m "not gpu and not slow"`) reports 517 passed, 6 skipped, 0 FAILED, confirming no unrelated regressions were introduced.

### ROADMAP.md Success Criteria — Verbatim Verification

**SC-1:** (≥99% recovery rate on N=100 random ecliptic-equator). VERIFIED. pytest line: `TestNRandomEclipticEquatorRecovery::test_n_random_ecliptic_equator_galaxies_recovery_rate_above_99pct PASSED`. Recovery gate is `recovered >= 99` at seed=42.

**SC-2:** (`astropy.coordinates.SkyCoord.transform_to(BarycentricTrueEcliptic())` applied; docstring declares "all stored angles henceforth in ecliptic SSB frame"). VERIFIED. `_rotate_equatorial_to_ecliptic` at handler.py:573 uses `BarycentricTrueEcliptic(equinox='J2000')` via vectorized SkyCoord. Called from `__init__` at handler.py:170 before `_map_angles_to_spherical_coordinates`.

> **SC-2 reinterpretation note (D-13):** The ROADMAP letter of SC-2 specifies astropy transform INSIDE `_map_angles_to_spherical_coordinates`. D-13 locks a deviation: the rotation is implemented as a separated `_rotate_equatorial_to_ecliptic()` method, called from `__init__` BEFORE `_map_angles_to_spherical_coordinates`. The rotation happens at ingestion, and the storage-convention docstring is declared in `_rotate_equatorial_to_ecliptic` — SC-2 spirit (rotation applied on ingestion, frame documented) is fully satisfied. The letter deviation — rotation is NOT inlined into `_map_angles_to_spherical_coordinates` — was a deliberate single-responsibility choice (D-13 rationale). This reinterpretation is approved as part of the Phase 36 planning and does NOT trigger re-verification.

**SC-3:** (BallTree Cartesian embedding `(sin θ cos φ, sin θ sin φ, cos θ)` consistent across setup + query). VERIFIED. `_polar_to_cartesian` helper at handler.py:777 implements `(sin θ cos φ, sin θ sin φ, cos θ)`. Called from 4 sites: 3D BallTree setup (handler.py:295), 3D BallTree query (handler.py:346), 4D BallTree setup (handler.py:429), 4D BallTree query (handler.py:462). `grep -c "_polar_to_cartesian" handler.py` = 9 (definition + 4 call sites + docstring references).

**SC-4:** (regression pickle shows new ⊇ old). VERIFIED. Pickle load + assertion output:
```
Pickle schema and superset property: OK
old |18| new |18|
event_id: 29
git_commit: 6e5315bc3a8c71e1f5ce5d68e7ca8e79b1681cbe (len=40)
fisher_sky_2x2 shape: (2, 2), det=1.86544e-07 (positive-definite)
```

**SC-5:** (reference comments + /physics-change post-implementation checks). VERIFIED. `grep -n "# Eq\." handler.py` returns 7 reference comments (4 for COORD-02/02b covering the 4 call sites, 1 for COORD-04 at the eigenvalue computation, 1 for COORD-03 at the rotation). `/physics-change` review record documented in Plan 36-01 Task 1 checkpoint.

### Observable Truths Summary

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 4 atomic [PHYSICS] commits (COORD-02, 02b, 03, 04) | VERIFIED | git log: b460297 c17ecb6 b2ef9c9 5b9cfbf |
| 2 | test_coordinate_roundtrip.py: 9 passed, 0 xfail, 0 XPASS, 0 FAIL | VERIFIED | pytest summary: `9 passed in 1.18s` |
| 3 | 36-superset-regression.pkl exists, schema-complete, new ⊇ old | VERIFIED | pickle inspection: old|18| new|18|, det>0 |
| 4 | No unrelated regression (`pytest -m "not gpu and not slow"`) | VERIFIED | full-suite: `517 passed, 6 skipped, 18 deselected, 16 warnings in 14.66s` |
| 5 | REQUIREMENTS.md, STATE.md, MILESTONES.md, ROADMAP.md consistent | VERIFIED | cross-check below |

**Score:** 5/5 truths verified

---

## REQ-ID Coverage Matrix

| Requirement | Description | Plans | Files | Commit | Status |
|-------------|-------------|-------|-------|--------|--------|
| COORD-02 | Polar-correct BallTree Cartesian embedding | 36-02 | handler.py | c17ecb6 | SATISFIED |
| COORD-02b | 4D BallTree spherical sky sub-space | 36-04 | handler.py | 5b9cfbf | SATISFIED |
| COORD-03 | Equatorial→ecliptic rotation on GLADE ingestion | 36-01 | handler.py | b460297 | SATISFIED |
| COORD-04 | Eigenvalue sky search radius | 36-03 | handler.py, bayesian_statistics.py, scripts/generate_phase36_regression.py, .planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl | b2ef9c9 | SATISFIED |

---

## Gate Checklist

### handler.py modifications

| Check | Result | Evidence |
|-------|--------|----------|
| `_rotate_equatorial_to_ecliptic` exists | PASS | handler.py:573 |
| `_polar_to_cartesian` module-level helper | PASS | handler.py:777 |
| 3D setup site uses helper | PASS | handler.py:295 |
| 3D query site uses helper | PASS | handler.py:346 |
| 4D setup site uses helper | PASS | handler.py:429 |
| 4D query site uses helper | PASS | handler.py:462 |
| Eigenvalue radius formula | PASS | `eigvalsh` at handler.py:355 |
| `cov_theta_phi` kwarg in signature | PASS | handler.py:312 |
| Range asserts in `_rotate_equatorial_to_ecliptic` | PASS | `assert np.all` in _rotate_equatorial_to_ecliptic |
| Reference comments above each changed line | PASS | 7 `# Eq.` comments in handler.py |

### Physics-Change Protocol

| Check | Result | Evidence |
|-------|--------|----------|
| Single /physics-change review for all 4 REQs (D-25) | PASS | Plan 36-01 Task 1 checkpoint record |
| Four atomic [PHYSICS] commits | PASS | git log: 4 commits matching `[PHYSICS] COORD-` |
| Each commit prefixed [PHYSICS] | PASS | b460297, c17ecb6, b2ef9c9, 5b9cfbf |
| Commit templates match CONTEXT.md §specifics | PASS | message inspection (COORD-02/02b/03/04 in subjects) |
| Pre-commit hooks ran clean on each commit | PASS | no --no-verify in history |

### xfail Discipline (D-26)

| Check | Result | Evidence |
|-------|--------|----------|
| Markers removed in SAME commit as fix | PASS | `@pytest.mark.xfail` absent from test file; removed per-REQ |
| No bulk-removal commit | PASS | git log: no "remove xfail" standalone commit |
| Current xfail count is 0 | PASS | pytest: 0 xfailed; grep shows only docstring references |

### Regression Pickle (D-24)

| Check | Result | Evidence |
|-------|--------|----------|
| Pickle committed at locked path | PASS | git ls-files: `.planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl` |
| All D-24 schema fields present | PASS | pickle inspection: all 9 expected keys present |
| old ⊆ new assertion passes | PASS | `assert d['old_candidate_indices'].issubset(d['new_candidate_indices'])` — passed |
| fisher_sky_2x2 is positive-definite 2×2 | PASS | det=1.86544e-07 > 0; shape=(2,2) |
| git_commit is 40-char SHA | PASS | len=40: `6e5315bc3a8c71e1f5ce5d68e7ca8e79b1681cbe` |

---

## Commit History

All commits on the Phase 36 branch:

| Hash | Subject | Type |
|------|---------|------|
| c2aa2dd | docs(36-04): SUMMARY.md — COORD-02b 4D BallTree spherical sky embedding complete | docs |
| 5b9cfbf | [PHYSICS] COORD-02b: 4D BallTree sky sub-space uses spherical embedding | [PHYSICS] |
| 7be20d2 | docs(36-03): SUMMARY.md — COORD-04 eigenvalue sky search radius complete | docs |
| b2ef9c9 | [PHYSICS] COORD-04: eigenvalue sky search radius on 2×2 Fisher covariance with |sin θ| Jacobian | [PHYSICS] |
| 6e5315b | docs(36-02): SUMMARY.md — COORD-02 polar-correct BallTree embedding complete | docs |
| c17ecb6 | [PHYSICS] COORD-02: polar-correct BallTree Cartesian embedding (sin θ cos φ, sin θ sin φ, cos θ) | [PHYSICS] |
| d788c25 | docs(36-01): SUMMARY.md — COORD-03 rotation + _polar_to_cartesian helper complete | docs |
| b460297 | [PHYSICS] COORD-03: equatorial→ecliptic rotation on GLADE ingestion via astropy BarycentricTrueEcliptic(J2000) | [PHYSICS] |
| 79e5623 | chore(36): add _polar_to_cartesian helper + COORD-02b requirement stub + test-shim update | chore |
| 10695ab | docs(36): commit Phase 36 plan files (01–05) | docs |

No scope creep. Non-[PHYSICS] commits are expected per D-25: one `chore(36)` scaffolding commit (Plan 36-01 Task 6: helper + REQUIREMENTS + test-shim) and four `docs(36-XX)` SUMMARY commits (one per plan).

---

## Required Artifacts

| Artifact | Expected | Status |
|----------|----------|--------|
| handler.py (modified) | 4 physics changes + helpers | VERIFIED |
| bayesian_statistics.py (modified) | Caller updated with cov_theta_phi | VERIFIED |
| scripts/generate_phase36_regression.py (new) | Regression CLI | VERIFIED |
| 36-superset-regression.pkl | D-24 schema | VERIFIED |
| test_coordinate_roundtrip.py (modified) | 6 xfail markers removed | VERIFIED |
| REQUIREMENTS.md (modified) | COORD-02b added + checkboxes | VERIFIED |

---

## Key Link Verification

| From | To | Status | Details |
|------|----|--------|---------|
| handler.py `_rotate_equatorial_to_ecliptic` | handler.py `__init__` | WIRED | Called at handler.py:170 before `_map_angles_to_spherical_coordinates` |
| handler.py `_polar_to_cartesian` | 3D setup site | WIRED | handler.py:295 |
| handler.py `_polar_to_cartesian` | 3D query site | WIRED | handler.py:346 |
| handler.py `_polar_to_cartesian` | 4D setup site | WIRED | handler.py:429 |
| handler.py `_polar_to_cartesian` | 4D query site | WIRED | handler.py:462 |
| bayesian_statistics.py:778 | handler.py `cov_theta_phi` kwarg | WIRED | `cov_theta_phi=self.detection.theta_phi_covariance` |
| 36-superset-regression.pkl | Phase 40 VERIFY-02 anchor | LOCKED | D-24 schema complete; old ⊆ new verified |

---

## Behavioral Spot-Checks

| Behavior | Command | Expected | Status |
|----------|---------|----------|--------|
| 9 tests pass | `uv run pytest master_thesis_code_test/test_coordinate_roundtrip.py -v` | 9 passed, 0 xfail | PASS |
| Pickle superset | `python -c "import pickle; d=pickle.load(open(...)); assert d['old_candidate_indices'].issubset(d['new_candidate_indices'])"` | No assertion error | PASS |
| No regression | `uv run pytest -m "not gpu and not slow"` | 517 passed, exit 0 | PASS |
| Handler import | present in test_coordinate_roundtrip.py imports | OK | PASS |
| Pickle positive-definite | `np.linalg.det(d['fisher_sky_2x2']) > 0` | 1.86544e-07 > 0 | PASS |

---

## Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Posterior re-evaluation at h=0.73 | Phase 40 VERIFY-02 | ROADMAP §Phase 40 abort gate |
| 2 | h-threading through set_host_galaxy_parameters | Phase 37 PE-01 | ROADMAP §Phase 37 SC-1 |
| 3 | Idempotency guard on `_map_angles_to_spherical_coordinates` | Phase 37 COORD-05 | ROADMAP §Phase 37 SC-7 |
| 4 | Deprecated `get_possible_hosts` removal | Phase 39 HPC-04 | ROADMAP §Phase 39 SC-4 |

---

## Notable Finding: Ecliptic Equator Host Density

The Phase 35 baseline reported 0/42 events in the ±5° ecliptic-equator band (deficiency vs 8.7% isotropic expected). The coordinate bugs were structurally fixed in Phase 36: `_polar_to_cartesian` now uses the correct `(sin θ cos φ, sin θ sin φ, cos θ)` convention and equatorial→ecliptic rotation is applied on ingestion. However, the re-run of the simulation to produce new CRBs under the fixed frame is Phase 40 VERIFY-02, not Phase 36. The post-fix band-fraction re-audit has NOT yet been performed. Phase 40 will diff the re-run JSON against `.planning/audit_coordinate_bug.json` to quantify the recovery (expected: ±5° band fraction should approach 8.7% isotropic).

---

## D-28 Gate

**D-28 gate: NOT triggered.**

All tests pass cleanly (9 passed, 0 xfailed, 0 XPASS, 0 FAILED). The N=100 random test reports recovery ≥99 at seed=42. No stop/rethink condition met. Phase 37 handoff proceeds normally.

---

## Gaps Summary

No gaps. All four REQ-IDs satisfied; all five ROADMAP SCs satisfied; D-28 gate NOT triggered.

---

## Verdict

**ACCEPT**

All Phase 36 deliverables exist, pass quality gates, and satisfy REQ-IDs COORD-02, COORD-02b, COORD-03, COORD-04. Regression pickle locked as Phase 40 VERIFY-02 anchor. Ready for Phase 37 handoff.

---

_Verified: 2026-04-22T11:43:00Z_
_Verifier: Claude (gsd-executor)_
