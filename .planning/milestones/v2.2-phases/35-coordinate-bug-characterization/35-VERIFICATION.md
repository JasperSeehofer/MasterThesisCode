---
phase: 35-coordinate-bug-characterization
verified: 2026-04-21T22:30:00Z
status: passed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification: null
gaps: []
deferred:
  - truth: "Production handler.py _map_angles_to_spherical_coordinates applies equatorial->ecliptic rotation"
    addressed_in: "Phase 36"
    evidence: "Phase 36 goal: 'Fix both critical coordinate bugs — apply equatorial->ecliptic rotation on catalog ingestion'"
  - truth: "BallTree Cartesian embedding is corrected to (sin θ cos φ, sin θ sin φ, cos θ)"
    addressed_in: "Phase 36"
    evidence: "Phase 36 success criteria 3: 'BallTree Cartesian embedding is (sin θ cos φ, sin θ sin φ, cos θ)'"
human_verification: []
---

# Phase 35: Coordinate Bug Characterization — Verification Report

**Phase Goal:** Encode the two critical coordinate bugs in failing tests before touching the
production code, and capture the pre-fix baseline of how many events sit in the danger zone
(±5° ecliptic equator).
**Verified:** 2026-04-21T22:30:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

Phase 35 is the RED phase of the coordinate-bug TDD cycle. Its deliverables are scaffolding
and a pre-fix baseline — not the fix itself. The phase succeeds when: (1) executable tests lock
the correct post-fix behavior behind `xfail(strict=True)`, and (2) committed baseline numbers
give Phase 40 a quantitative diff target.

### ROADMAP.md Success Criteria — Verbatim Verification

**SC-1:** `master_thesis_code_test/test_coordinate_roundtrip.py` exists and contains at least three tests:
(a) synthetic galaxy at ecliptic Dec=0° retrieved by BallTree, (b) Dec=90° NCP round-trip,
(c) equatorial→ecliptic conversion matches astropy SkyCoord to <0.001°.

- VERIFIED. File exists at `master_thesis_code_test/test_coordinate_roundtrip.py` (346 lines).
  All three mandatory tests are present as separate classes: `TestBallTreeRecoversEclipticEquatorGalaxy`
  (SC-1a), `TestBallTreeRecoversNorthCelestialPole` (SC-1b), `TestEquatorialToEclipticGroundTruth`
  (SC-1c). Additionally contains four supplementary test classes (D-05 vernal, solstice, ecliptic-pole
  + D-06/D-07 N-random statistical gate). Total: 9 tests across 7 classes.

**SC-2:** The three new tests are RED against the current code.

- VERIFIED. Live test run confirms `3 passed, 6 xfailed in 0.59s`. The 6 xfailed tests include all
  three mandatory scenarios (plus three supplementary edge-case and statistical tests). Zero XPASS
  (no premature fix). Zero FAIL (no broken markers). The xfail markers use `strict=True` — CI will
  reject green state once Phase 36 fixes the bugs without marker removal.

**SC-3:** `.planning/audit_coordinate_bug.md` reports baseline count and percentage of events in
the production CRB CSV within ±5° of ecliptic equator (`|qS − π/2| < 5° × π/180`).

- VERIFIED. File exists at `.planning/audit_coordinate_bug.md`. The ±5° row reads:
  `| ±5° | 0 | 0.0000 | 0.0872 | -0.0872 |`. 0/42 events in the ±5° band (0.0% observed vs 8.7%
  isotropic expected). Band table includes all three required widths (±5°, ±10°, ±15°). Histogram
  embedded via `![...](audit_coordinate_bug_histogram.png)`.

**SC-4:** Baseline artifact is committed so subsequent phases can reference it.

- VERIFIED. `git ls-files` confirms all three artifacts are tracked:
  `.planning/audit_coordinate_bug.json`, `.planning/audit_coordinate_bug.md`,
  `.planning/audit_coordinate_bug_histogram.png`. Committed in `d0573b1`.

### Observable Truths Summary

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | test_coordinate_roundtrip.py exists with >= 3 mandatory tests | VERIFIED | 9 tests in 7 classes; file:346 lines |
| 2 | Tests are RED — 6 xfailed, 0 XPASS, 0 FAIL | VERIFIED | Live run: 3 passed, 6 xfailed in 0.59s |
| 3 | audit_coordinate_bug.md reports ±5° baseline count | VERIFIED | 0/42 events; deviation -0.0872 |
| 4 | Baseline artifact committed to git | VERIFIED | git ls-files shows all three artifacts tracked |

**Score:** 4/4 truths verified

---

## COORD-01 Requirements Coverage Matrix

| Requirement | Description (from REQUIREMENTS.md) | Plans | Files | Status |
|-------------|-------------------------------------|-------|-------|--------|
| COORD-01 | Failing-test-first characterization: round-trip tests for equatorial↔ecliptic and polar↔latitude conventions, plus baseline counts of events in ±5° ecliptic-equator band from existing CRB CSV | 35-01, 35-02, 35-03 | `fixtures/coordinate.py`, `test_coordinate_roundtrip.py`, `scripts/audit_coordinate_bug.py`, `.planning/audit_coordinate_bug.*` | SATISFIED |

COORD-01 is fully satisfied. The characterization exists in both executable (xfail tests) and
analytical (committed audit artifacts) form.

---

## Gate Checklist

### Fixtures Module (`master_thesis_code_test/fixtures/coordinate.py`)

| Check | Result | Evidence |
|-------|--------|----------|
| File exists | PASS | 207 lines, committed in d9a7f86 |
| Three public helpers exported | PASS | `equatorial_to_ecliptic_astropy`, `synthetic_catalog_builder`, `build_balltree` + `EclipticCoords` NamedTuple all in `__all__` |
| Typed fully (no bare `np.ndarray`) | PASS | `npt.NDArray[np.float64]` used; mypy: no issues |
| No `from __future__ import annotations` | PASS | Grep: absent |
| No `import cupy` / no GPU leakage | PASS | Grep: absent; pure astropy + numpy + pandas |
| `equinox="J2000"` pinned in astropy call | PASS | Line 72: `BarycentricTrueEcliptic(equinox="J2000")` |
| `object.__new__(GalaxyCatalogueHandler)` shim | PASS | Lines 196-198 |
| `np.random.default_rng(seed)` pattern | PASS | Line 121 |
| `already_rotated: bool` stage-choice knob | PASS | Lines 84, 141 |
| Imports from `master_thesis_code.galaxy_catalogue.handler` | PASS | Lines 35-38 |
| ruff check | PASS | `All checks passed!` |
| mypy | PASS | `Success: no issues found` |
| CPU import sanity (`beta_rad ≈ 0` at vernal equinox) | PASS | Live run: `beta_rad: -1.02e-07` |

### Test File (`master_thesis_code_test/test_coordinate_roundtrip.py`)

| Check | Result | Evidence |
|-------|--------|----------|
| File exists | PASS | 345 lines, committed in fa9ec00 |
| Both bugs covered (rotation + BallTree embedding) | PASS | Rotation bug: `TestBallTreeRecoversNorthCelestialPole`, `TestSummerSolsticeRotation`, `TestEclipticPoleIngestion`; BallTree embedding bug: `TestBallTreeRecoversEclipticEquatorGalaxy`, `TestVernalEquinoxRoundTrip`, `TestNRandomEclipticEquatorRecovery` |
| All xfail tests use `strict=True` | PASS | grep count: 7 occurrences (6 on test methods + 1 on shared constant line — all correct) |
| No premature xfail on ground-truth class | PASS | `TestEquatorialToEclipticGroundTruth` (3 tests) are NOT xfail — correct by construction |
| Tests import from fixtures module (not duplicated logic) | PASS | Lines 40-44: `from master_thesis_code_test.fixtures.coordinate import build_balltree, equatorial_to_ecliptic_astropy, synthetic_catalog_builder` |
| Shared `_XFAIL_REASON` constant | PASS | Line 47: single constant; all 6 xfail markers reference it |
| `seed=42`, `N=100`, `recovered >= 99` gate | PASS | Lines 322, 326, 339 |
| No `from __future__ import annotations` | PASS | Grep: absent |
| No `@pytest.mark.gpu` | PASS | grep count: 0 |
| ruff check | PASS | `All checks passed!` |
| mypy | PASS | `Success: no issues found` |
| Live run: 3 passed, 6 xfailed, 0 FAIL, 0 XPASS | PASS | Confirmed above |

### Audit Script (`scripts/audit_coordinate_bug.py`)

| Check | Result | Evidence |
|-------|--------|----------|
| File exists | PASS | 289 lines, committed in edd6f03 |
| CLI (argparse, `--csv`, `--output-dir`) | PASS | Lines 238-265; `parse_args()` + `main()` present |
| `if __name__ == "__main__": main()` | PASS | Lines 287-288 |
| Reuses `_get_git_commit_safe` from evaluation_report | PASS | Line 40: `from master_thesis_code.bayesian_inference.evaluation_report import _get_git_commit_safe` |
| `save_figure(..., formats=("png",))` explicit | PASS | Line 127: `save_figure(fig, output_path_no_ext, formats=("png",))` |
| `apply_style()` called once in `main()` | PASS | Line 277 |
| No `import matplotlib.pyplot` at module level | PASS | Line 35 imports `import matplotlib` only; `pyplot` not imported |
| Bands tuple `(5, 10, 15)` locked as constant | PASS | Line 45: `_BANDS_DEG: tuple[int, ...] = (5, 10, 15)` |
| No `from __future__ import annotations` | PASS | Grep: absent |
| No `import cupy` | PASS | Grep: absent |
| ruff check | PASS | `All checks passed!` |
| mypy | PASS | `Success: no issues found` |

### Committed Baseline Artifacts

| Check | Result | Evidence |
|-------|--------|----------|
| `.planning/audit_coordinate_bug.json` exists and git-tracked | PASS | `git ls-files` output; committed d0573b1 |
| `.planning/audit_coordinate_bug.md` exists and git-tracked | PASS | `git ls-files` output |
| `.planning/audit_coordinate_bug_histogram.png` exists and git-tracked | PASS | `git ls-files` output; `file` confirms PNG image data 1012x625 RGBA |
| JSON has `event_count: 42` | PASS | JSON line 2 |
| JSON has `band_counts` with keys "5", "10", "15" | PASS | JSON lines 3-7 |
| JSON has `band_fractions` with keys "5", "10", "15" | PASS | JSON lines 8-12 |
| JSON has `expected_fraction_5deg` ≈ 0.0872 | PASS | `0.08715574274765817` (sin(5°) to 16 sig figs) |
| JSON has `csv_source_path`, `git_commit`, `timestamp` | PASS | All present; git_commit = `edd6f03ad7e...` |
| Markdown embeds histogram PNG via relative path | PASS | `![...](audit_coordinate_bug_histogram.png)` |
| Markdown has ±5°, ±10°, ±15° band rows | PASS | Lines 10-14 of audit_coordinate_bug.md |
| `.gitignore` updated for `!.planning/*.png` exception | PASS | Added in d0573b1 (deviation from plan — required for PNG tracking) |

### Physics-Change Protocol

| Check | Result | Evidence |
|-------|--------|----------|
| No physics-gate files modified | PASS | Phase 35 commits (d9a7f86, fa9ec00, edd6f03, d0573b1) touch only: `fixtures/coordinate.py`, `test_coordinate_roundtrip.py`, `scripts/audit_coordinate_bug.py`, `.planning/audit_coordinate_bug.*`, `.gitignore`, `CHANGELOG.md`, `TODO.md`. None are physics-gate files per CLAUDE.md. |
| No `/physics-change` protocol required | PASS | Confirmed — this is pure test infrastructure + CLI tooling |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `master_thesis_code_test/fixtures/coordinate.py` | Fixtures module with 3 helpers + EclipticCoords NamedTuple | VERIFIED | 207 lines; all exports present; mypy/ruff clean |
| `master_thesis_code_test/test_coordinate_roundtrip.py` | 7 test classes (6 xfail + 1 ground-truth) | VERIFIED | 346 lines; live: 3 passed, 6 xfailed |
| `scripts/audit_coordinate_bug.py` | CLI audit generator | VERIFIED | 289 lines; argparse + main; mypy/ruff clean |
| `.planning/audit_coordinate_bug.md` | Human-readable baseline with band table | VERIFIED | Band table present; histogram embedded |
| `.planning/audit_coordinate_bug.json` | Machine-readable baseline for Phase 40 diff | VERIFIED | All D-11 schema fields present; 42 events |
| `.planning/audit_coordinate_bug_histogram.png` | PNG histogram | VERIFIED | Valid PNG 1012x625; git-tracked |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `test_coordinate_roundtrip.py` | `fixtures/coordinate.py` | `from master_thesis_code_test.fixtures.coordinate import ...` | WIRED | Lines 40-44 |
| `test_coordinate_roundtrip.py` | `master_thesis_code.galaxy_catalogue.handler` | `from master_thesis_code.galaxy_catalogue.handler import InternalCatalogColumns` | WIRED | Line 39 |
| `fixtures/coordinate.py` | `master_thesis_code.galaxy_catalogue.handler` | `from master_thesis_code.galaxy_catalogue.handler import GalaxyCatalogueHandler, InternalCatalogColumns` | WIRED | Lines 35-38 |
| `fixtures/coordinate.py` | `astropy.coordinates` | `from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord` | WIRED | Lines 32-33 |
| `scripts/audit_coordinate_bug.py` | `simulations/cramer_rao_bounds.csv` | `pd.read_csv(csv_path)` | WIRED | Line 150 (CSV path from CLI arg) |
| `scripts/audit_coordinate_bug.py` | `.planning/audit_coordinate_bug.*` | `json_path.write_text(...)` / `md_path.write_text(...)` / `save_figure(...)` | WIRED | Lines 179-185, 232-233 |
| `.planning/audit_coordinate_bug.md` | `.planning/audit_coordinate_bug_histogram.png` | Markdown image embed | WIRED | `![...](audit_coordinate_bug_histogram.png)` |
| `scripts/audit_coordinate_bug.py` | `evaluation_report._get_git_commit_safe` | `from master_thesis_code.bayesian_inference.evaluation_report import _get_git_commit_safe` | WIRED | Line 40 |

---

## Data-Flow Trace (Level 4)

Not applicable for this phase. All three deliverables are test infrastructure and a CLI audit
generator — no dynamic data rendered by a UI component. The audit script reads a CSV and writes
static files; the test file exercises production code in-memory. No data-flow trace required.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Fixtures CPU import + astropy round-trip | `python -c "from ... import equatorial_to_ecliptic_astropy; print(equatorial_to_ecliptic_astropy(0,0).beta_rad)"` | `-1.02e-07` (≈ 0, within tolerance) | PASS |
| Test run: 3 passed, 6 xfailed, 0 FAIL | `uv run pytest test_coordinate_roundtrip.py -v --tb=no` | `3 passed, 6 xfailed in 0.59s` | PASS |
| JSON schema complete | `jq -e '.event_count == 42 and (.band_counts | has("5","10","15")) ...' .planning/audit_coordinate_bug.json` | Verified manually from JSON content | PASS |
| PNG is a valid image | `file .planning/audit_coordinate_bug_histogram.png` | `PNG image data, 1012 x 625, 8-bit/color RGBA` | PASS |

---

## Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| COORD-01 | 35-01, 35-02, 35-03 | Failing-test-first characterization: round-trip tests + ±5° ecliptic-equator baseline | SATISFIED | All 4 roadmap success criteria verified |

---

## Anti-Patterns Found

No stubs, placeholders, or anti-patterns identified. All three files are substantive
implementations with real logic. The audit artifacts contain real data from the production CSV.

---

## Deferred Items

Items not yet met in Phase 35, explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | handler.py applies equatorial→ecliptic rotation in `_map_angles_to_spherical_coordinates` | Phase 36 | Phase 36 goal + SC-2: "handler.py:_map_angles_to_spherical_coordinates uses astropy SkyCoord.transform_to(BarycentricTrueEcliptic())" |
| 2 | BallTree Cartesian embedding corrected to `(sin θ cos φ, sin θ sin φ, cos θ)` | Phase 36 | Phase 36 SC-3: "BallTree Cartesian embedding is (sin θ cos φ, sin θ sin φ, cos θ) with θ ∈ [0, π] polar" |
| 3 | Post-fix re-audit diffs baseline JSON against fixed CRB CSV | Phase 40 | Phase 40 VERIFY-04: "anisotropy audit + P_det diagnostic" |

These are intentional deferrals. Phase 35 captures the pre-fix state; Phase 36 applies the fix.

---

## Notable Finding: Equatorial Underdensity, Not Pile-Up

The ±5° band shows 0/42 events (0.0% observed vs 8.7% isotropic expected). The coordinate bug does
not pile events *onto* the ecliptic equator as might be naively expected from a singular BallTree
embedding. Instead, the broken `cos(θ)` embedding collapses equatorial-plane galaxies to the
`(0, 0, 1)` pole point — queries near the ecliptic equator never find their true hosts, causing
a deficiency in that band in the recovered `qS` distribution. The ±10° band also shows a
deficit: 2/42 observed (4.8%) vs 17.4% expected.

**This is the first-order Phase 40 VERIFY-04 signal.** After the Phase 36 fix, the band fractions
should shift toward the isotropic prior. Phase 40 diffs the re-run JSON against
`.planning/audit_coordinate_bug.json` to quantify the correction.

---

## Gaps Summary

No gaps. All four roadmap success criteria are satisfied. All artifacts exist, are substantive,
are wired, and are committed.

---

## Verdict

**ACCEPT**

All Phase 35 deliverables exist, pass quality gates, and satisfy the COORD-01 requirement exactly
as specified. The TDD RED state is correctly established: 6 tests are `xfail(strict=True)`, 3
ground-truth tests pass, 0 XPASS, 0 FAIL. The committed baseline provides a quantitative anchor
for Phase 40 VERIFY-04.

**Ready for `/gsd-execute-phase 36`**

Phase 36 must: (1) remove the `@pytest.mark.xfail` markers from `test_coordinate_roundtrip.py`,
(2) apply equatorial→ecliptic rotation in `handler.py:_map_angles_to_spherical_coordinates`,
(3) correct the BallTree Cartesian embedding. The xfail(strict=True) mechanism ensures CI
enforces this handoff.

---

_Verified: 2026-04-21T22:30:00Z_
_Verifier: Claude (gsd-verifier)_
