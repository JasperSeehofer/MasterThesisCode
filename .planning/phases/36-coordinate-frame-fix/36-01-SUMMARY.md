---
plan: 36-01
phase: 36-coordinate-frame-fix
status: complete
completed: 2026-04-22
commits:
  - 79e5623  # chore(36): scaffolding (helper + REQUIREMENTS + test-shim)
  - b460297  # [PHYSICS] COORD-03: equatorial→ecliptic rotation
req_ids: [COORD-03, COORD-02b]
---

## Summary

Applied equatorial→ecliptic rotation on GLADE catalog ingestion via astropy `BarycentricTrueEcliptic(J2000)`, and introduced the shared `_polar_to_cartesian` helper consumed by Plans 36-02 and 36-04.

## Key Deliverables

- **`_polar_to_cartesian(theta, phi)`** — module-level helper at `handler.py` returning `(sin θ cos φ, sin θ sin φ, cos θ)` unit vectors.
- **`GalaxyCatalogueHandler._rotate_equatorial_to_ecliptic()`** — vectorized astropy `SkyCoord.transform_to(BarycentricTrueEcliptic(J2000))`, with hard range asserts per D-15.
- **`__init__` call insertion** — rotation fires BEFORE `_map_angles_to_spherical_coordinates` per D-13.
- **REQUIREMENTS.md** — COORD-02b traceability entry added.
- **Test shim** — Phase 35 bypass-init tests now call `_rotate_equatorial_to_ecliptic()` before `_map_angles_to_spherical_coordinates()` per D-13 test/production parity.
- **xfail removal** — markers removed from `TestSummerSolsticeRotation` and `TestEclipticPoleIngestion` (flip XPASS under COORD-03 alone).

## Test State After Commits

```
5 passed, 4 xfailed
```

- `TestSummerSolsticeRotation` → PASS
- `TestEclipticPoleIngestion` → PASS
- `TestEquatorialToEclipticGroundTruth` (3 tests) → PASS
- 4 xfailed: BallTree-dependent tests (COORD-02: equator retrieval, NCP, N=100 random; vernal equinox depends on both COORD-02 + COORD-03)

## Two-commit structure

1. `chore(36): add _polar_to_cartesian helper + COORD-02b requirement stub + test-shim update` (79e5623)
2. `[PHYSICS] COORD-03: equatorial→ecliptic rotation on GLADE ingestion via astropy BarycentricTrueEcliptic(J2000)` (b460297)

## Self-Check: PASSED
