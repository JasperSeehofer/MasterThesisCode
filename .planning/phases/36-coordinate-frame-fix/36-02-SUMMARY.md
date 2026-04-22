---
plan: 36-02
phase: 36-coordinate-frame-fix
status: complete
completed: 2026-04-22
commits:
  - c17ecb6  # [PHYSICS] COORD-02: polar-correct BallTree Cartesian embedding
req_ids: [COORD-02]
---

## Summary

Replaced the latitude-formula Cartesian embedding `(cos θ cos φ, cos θ sin φ, sin θ)` with the polar-correct `_polar_to_cartesian(θ, φ)` = `(sin θ cos φ, sin θ sin φ, cos θ)` at both BallTree call sites in `handler.py`. Removed all four COORD-02-dependent `xfail` markers. Phase 35 RED test suite now reads 9 passed, 0 xfailed.

## Key Deliverables

- **`setup_galaxy_catalog_balltree` (handler.py:~286)** — replaced latitude-formula inline embedding with `data = _polar_to_cartesian(theta, phi)`; renamed misleading `ra`/`dec` locals to `phi`/`theta`; added clarifying comment about post-Plan-36-01 column semantics.
- **`get_possible_hosts_from_ball_tree` (handler.py:~307)** — replaced three-line inline `(cos θ …)` formula with `query_point = _polar_to_cartesian(np.array([theta]), np.array([phi]))`.
- **xfail removal** — all four COORD-02-dependent markers removed from `test_coordinate_roundtrip.py`: `TestBallTreeRecoversEclipticEquatorGalaxy`, `TestBallTreeRecoversNorthCelestialPole`, `TestVernalEquinoxRoundTrip`, `TestNRandomEclipticEquatorRecovery`.
- **Unused import cleanup** — `import pytest` removed from test file (no longer used after xfail removal).

## Test State After Commit

```
9 passed, 0 xfailed, 0 XPASS, 0 FAIL  (in 1.17s)
```

All Phase 35 RED tests are now green. ROADMAP §Phase 36 SC-1 (≥99% ecliptic-equator recovery) and SC-3 (BallTree embedding consistent across setup + query) are satisfied.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing] Removed unused `pytest` import**

- **Found during:** Task 3 (ruff check after xfail removal)
- **Issue:** Removing all `@pytest.mark.xfail` decorators left `import pytest` unused; ruff F401 error.
- **Fix:** Removed the `import pytest` line from `test_coordinate_roundtrip.py`.
- **Files modified:** `master_thesis_code_test/test_coordinate_roundtrip.py`
- **Commit:** c17ecb6 (included in the atomic commit)

**2. [Scope note] Fourth xfail marker removed (`TestVernalEquinoxRoundTrip`)**

- The plan listed 3 tests to de-xfail; the test file actually had 4 xfail markers after Plan 36-01 (the vernal equinox BallTree test was also COORD-02-dependent and remained xfailed). Removing all four is consistent with D-26 ("remove xfail in the SAME commit as the REQ-ID fix") and D-27 (target: 0 xfail). No architectural impact.

## Verification Results

- `grep -nE "np\.cos\(theta\) \* np\.cos\(phi\)"` → 0 matches in handler.py
- `grep -nE "_polar_to_cartesian" handler.py` → 3 matches (definition + 2 call sites)
- `uv run pytest master_thesis_code_test/test_coordinate_roundtrip.py -v` → 9 passed, 0 xfail
- `uv run ruff check` + `ruff format --check` + `mypy` → all clean
- Pre-commit hooks (ruff + ruff-format + mypy) → Passed

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary changes introduced.

## Self-Check: PASSED

- `master_thesis_code/galaxy_catalogue/handler.py` — modified (old latitude formula replaced)
- `master_thesis_code_test/test_coordinate_roundtrip.py` — modified (xfail markers + unused import removed)
- Commit `c17ecb6` exists: confirmed via `git log`
- 9/9 tests passing: confirmed
