# Phase 35: Coordinate Bug Characterization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-21
**Phase:** 35-coordinate-bug-characterization
**Areas discussed:** RED-test encoding, Baseline audit depth, Extra edge-case tests, Fixtures vs inline

---

## Gray-Area Selection

| Option | Description | Selected |
|--------|-------------|----------|
| RED-test encoding | plain fail vs xfail(strict=True) vs xfail vs pin-the-bug | ✓ |
| Baseline audit depth | bands + histogram + expected/observed + JSON | ✓ |
| Extra edge-case tests | vernal, summer solstice, ecliptic pole, N-random, near-pole | ✓ |
| Fixtures vs inline | dedicated fixtures module vs inline vs conftest | ✓ |

**User's choice:** All four.

---

## RED-test Encoding

| Option | Description | Selected |
|--------|-------------|----------|
| xfail(strict=True) | CI green now; when Phase 36 fix lands, test → XPASS → CI red until marker removed. Forces conscious handoff. | ✓ |
| Plain xfail (non-strict) | CI green; when fix lands, test silently XPASS (visible in summary only). | |
| Plain assertions, CI red | No marker. Main stays red from Phase 35 commit until Phase 36 lands (~0.5–1 day). | |
| Pin the wrong behavior | Assert buggy output now; rewrite in Phase 36. More churn, no red test. | |

**User's choice:** xfail(strict=True).
**Notes:** Keeps main green during the ~1-day gap between Phase 35 and Phase 36, while the strict flag guarantees the handoff is not silent when the fix lands.

---

## Green Target (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Single RED test per bug | Each test asserts CORRECT behavior with xfail(strict=True). Phase 36 removes marker when fix works. | ✓ |
| Two tests per bug | One pins wrong behavior (green now, delete in Phase 36) + one asserts correct (xfail now). | |
| Only correct-behavior tests, no bug-pinning | Same as option 1. Leanest. | |

**User's choice:** Single RED test per bug.
**Notes:** Each test becomes the acceptance criterion Phase 36 must satisfy — minimal test-surface churn across the handoff.

---

## Baseline Audit Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Multiple bands (±5°/±10°/±15°) | Event count + % in each band. Shows falloff of risk. | ✓ |
| Histogram of |qS−π/2| | Bin 42 events; commit PNG. | ✓ |
| Expected-vs-observed vs isotropic prior | Under isotropic prior, ±5° → ~8.7% expected. Flag deviations. | ✓ |
| Structured JSON sidecar | Machine-readable `.planning/audit_coordinate_bug.json` for Phase 40 diff. | ✓ |

**User's choice:** All four.
**Notes:** No hesitation — full audit with JSON sidecar enables Phase 40's VERIFY-04 anisotropy diff without re-parsing markdown.

---

## Audit Scope

| Option | Description | Selected |
|--------|-------------|----------|
| CRB CSV only | 42 events in `simulations/cramer_rao_bounds.csv`. | ✓ |
| CRB CSV + spot-check 3 h-value JSONs | Sanity check on downstream filtering. | |
| All 27 h-sweep CSVs | Exhaustive but redundant (same events). | |

**User's choice:** CRB CSV only.
**Notes:** Same events underlie the h-sweep posteriors; scanning them adds no new sky information.

---

## Extra Edge-Case Tests

| Option | Description | Selected |
|--------|-------------|----------|
| Vernal equinox (RA=0, Dec=0) | On-axis point; only catches "no rotation at all" bugs. | ✓ |
| Summer solstice (RA=6h, Dec=+23.4°) | Off equatorial-equator, on ecliptic-equator. Isolates obliquity bugs. | ✓ |
| Ecliptic pole (RA=18h, Dec=+66.56°) | β=+90°, θ_polar=0. Pole handling + obliquity sign. | ✓ |
| N random ecliptic-equator galaxies | Statistical recovery gate. | ✓ |
| Near-pole Dec=±89° (not presented in final Q) | Numerical stability. | (folded into NCP test) |

**User's choice:** All four (near-pole folded into NCP).
**Notes:** Gives full coverage — single-point traps AND statistical claim for Phase 36 acceptance.

---

## Recovery Rate Threshold

| Option | Description | Selected |
|--------|-------------|----------|
| ≥99% | Matches Phase 36 success criterion verbatim. | ✓ |
| ≥95% | Looser gate; requires rewording Phase 36 criterion. | |
| 100% (exact) | Too tight under finite-precision rotations. | |

**User's choice:** ≥99%.
**Notes:** Keeps the handoff criterion identical on both sides of the xfail.

---

## Fixtures vs Inline

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated fixtures module | `master_thesis_code_test/fixtures/coordinate.py` imported by Phase 35 AND Phase 36. | ✓ |
| Inline helpers in test file | Private functions in `test_coordinate_roundtrip.py`. Mild duplication risk. | |
| conftest.py additions | Autoloaded fixtures. Grab-bag risk; less discoverable. | |

**User's choice:** Dedicated fixtures module.
**Notes:** Phase 36 will add regression tests that need the same helpers; centralizing them now avoids duplication.

---

## Random Seed Config

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcoded seed=42, N=100 | Deterministic, reproducible. ≤1 miss allowed under ≥99% gate. | ✓ |
| Parametrize over 3 seeds | Triple compute; catches seed-edge flakiness. | |
| pytest-randomly / time-seeded | Flaky; moving target for Phase 36. | |

**User's choice:** seed=42, N=100.
**Notes:** Matches project's seed-pinned regression pattern.

---

## Claude's Discretion

- JSON sidecar field names beyond the listed minimum (keep additive).
- Histogram bin count and range (sensible defaults for N=42).
- Test docstring phrasing.
- Return type of `equatorial_to_ecliptic_astropy` (tuple/dict/namedtuple).
- Whether the baseline audit generator is a standalone Python module, a CLI subcommand, or a script under `scripts/` — recommended: standalone module invocable via `python -m` or similar, NOT a pytest fixture.

## Deferred Ideas

- Eigenvalue-based sky-ellipse search radius → Phase 36 (COORD-04).
- Equatorial→ecliptic rotation in production code → Phase 36 (COORD-03).
- Idempotency guard on `_map_angles_to_spherical_coordinates` → Phase 37 (COORD-05).
- `flip_hx` verification → Phase 39 (HPC-05).
- CI integration + PubFigs Phase 35 naming collision → post-v2.2 when PubFigs resumes.
- Full h-sweep posterior re-audit → Phase 40 VERIFY-03.
- Near-pole Dec=±89° numerical stability → folded into Dec=90° NCP test.
