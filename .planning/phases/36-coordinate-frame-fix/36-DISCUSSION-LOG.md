# Phase 36: Coordinate Frame Fix - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-22
**Phase:** 36-coordinate-frame-fix
**Areas discussed:** Rotation integration point, BallTree fix shape + scope, Search-radius eigen form + superset, Physics-change cadence + xfail removal

---

## Rotation integration point (COORD-03)

| Option | Description | Selected |
|--------|-------------|----------|
| Separate method | New `_rotate_equatorial_to_ecliptic()` called from `__init__` BEFORE `_map_angles_to_spherical_coordinates`. Single-responsibility; easy [PHYSICS] review. | ✓ |
| Inline inside _map_angles | Add astropy rotation inside existing method between deg load and polar conversion. Fewer lines changed but mixes two physics concerns. | |
| At parse_to_reduced_catalog | Rotate once during GLADE+.txt reduction; reduced CSV stores ecliptic coords. Invalidates existing reduced CSV. | |

**User's choice:** Separate method (Recommended). Locked as D-13.

| Option | Description | Selected |
|--------|-------------|----------|
| Recompute on load | Vectorized `SkyCoord.transform_to` on ~230k galaxies is sub-second. No cache-invalidation landmines. | ✓ |
| Cache to reduced CSV | Write ecliptic columns to disk on first run. Creates frame dependency on disk. | |
| Cache to sidecar parquet | Keep reduced CSV as-is; write ecliptic to sidecar keyed by git hash. | |

**User's choice:** Recompute on load (Recommended). Locked as D-14.

| Option | Description | Selected |
|--------|-------------|----------|
| Vectorized array + hard asserts | One SkyCoord with full arrays; assert lon ∈ [0, 2π), lat ∈ [-π/2, π/2] post-rotation. | ✓ |
| Vectorized only, no guards | Trust astropy; skip runtime assertions. | |
| Row-wise loop | Iterate per-galaxy. >100× slower; against project conventions. | |

**User's choice:** Vectorized array + hard asserts (Recommended). Locked as D-15.

| Option | Description | Selected |
|--------|-------------|----------|
| Astropy round-trip on 3 known points | Vernal equinox, summer solstice, ecliptic pole agreeing <0.001°. Phase 35 D-05 encoded. | ✓ |
| Full Phase 35 suite green | Wait for complete suite XPASS. Slower signal. | |
| Mean `|qS-π/2|` shift on 42 events | Aggregate post-load distribution. Slow; aggregate; misses subset bugs. | |

**User's choice:** Astropy round-trip on 3 known points (Recommended). Locked as D-16.

---

## BallTree fix shape + scope (COORD-02, COORD-02b)

| Option | Description | Selected |
|--------|-------------|----------|
| Private helper | `_polar_to_cartesian(theta, phi)` shared across sites. Symmetry guaranteed structurally. | ✓ |
| Inline both sites | Replace three lines at each call site. Nothing prevents future drift. | |
| Helper + assertion parity test | Helper plus grep/import-based test that both sites use it. Belt and suspenders. | |

**User's choice:** Private helper (Recommended). Locked as D-17.

| Option | Description | Selected |
|--------|-------------|----------|
| Out of scope | REQUIREMENTS.md names only the two 3D balltree call sites. Defer to Phase 41 precondition. | |
| Fold 4D fix into Phase 36 | Same bug class, same fix pattern. Safe for v2.2 (only affects future simulations, not existing CRBs). | ✓ |
| Fix + add assert-only to 4D tree | Fold fix AND add pole-proximity assertion. Belt-and-suspenders. | |

**User's question (after option presentation):** "shouldnt the 4d embedding be used in the main pipeline? if not go with 1."

**Trace (discussion follow-up):** Confirmed 4D tree is used by `find_closest_galaxy_to_coordinates` → `get_hosts_from_parameter_samples`, called from `master_thesis_code/main.py:388` (simulation pipeline) and `scripts/quick_snr_calibration.py:53, 69`. It is **not** used in the evaluation pipeline (`bayesian_statistics.py`). v2.2 scoping = "no re-simulation" means the existing CRBs carry the 4D-tree bug forward inert; Phase 41/42 conditional campaigns would expose it if they fire.

**User's choice:** Fold 4D fix into Phase 36 (Recommended, after trace). Locked as D-18. New requirement COORD-02b to be added to REQUIREMENTS.md traceability.

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 35 D-04/D-06/D-07 flip green | Three mandatory tests become XPASS post-fix; remove xfail markers. Matches ROADMAP SC-1 verbatim. | ✓ |
| Direct BallTree unit test | 5-galaxy synthetic catalog in test; assert nearest-neighbor. Faster but ignores Phase 35 ground-truth work. | |
| Numerical parity against astropy SkyCoord separations | Float-tolerance assertion on great-circle distances. Mathematically rigorous but over-scoped. | |

**User's choice:** Phase 35 D-04/D-06/D-07 flip green (Recommended). Locked as D-19.

| Option | Description | Selected |
|--------|-------------|----------|
| Leave | Out of Phase 36 scope; Phase 39 HPC-04 handles dead-code removal. | ✓ |
| Remove in Phase 36 | Scope creep into code removal during physics-change phase. | |
| Leave + add deprecation warning | Adds softness without removing code; still scope creep. | |

**User's choice:** Leave (Recommended). Locked as D-20.

---

## Search-radius eigen form + superset (COORD-04)

| Option | Description | Selected |
|--------|-------------|----------|
| Add `cov_theta_phi` kwarg | Keep `phi_sigma`, `theta_sigma`; add optional `cov_theta_phi = 0.0`. Minimally invasive. | ✓ |
| Accept 2×2 ndarray | Replace args with `sky_covariance: NDArray`. Cleaner long-term but touches every call site. | |
| Compute eigenvectors in caller | Caller passes `(major_sigma, minor_sigma, angle)`. Scatters physics across files. | |

**User's choice:** Add `cov_theta_phi` kwarg (Recommended). Locked as D-21.

| Option | Description | Selected |
|--------|-------------|----------|
| √λ_max of scaled cov | `r = σ_mult × √λ_max(Σ')`, `Σ' = J Σ J^T`, `J = diag(|sin θ|, 1)`. Subsumes axis-aligned max at θ=π/2. | ✓ |
| max(eigenvalue, old_radius) | Conservative; guarantees strict superset for ALL events. Tiny over-selection near poles. | |
| Ellipse containment test | Initial BallTree query + post-filter via Mahalanobis. Tighter but adds complexity. | |

**User's choice:** √λ_max of scaled cov (Recommended). Locked as D-22.

| Option | Description | Selected |
|--------|-------------|----------|
| Worst `|qS-π/2|` (equator galaxy) | Event most affected by both coordinate bugs; Phase 35 audit lists them. | ✓ |
| Median-`|qS-π/2|` (mid-latitude) | Common-case superset check. Less dramatic signal. | |
| Largest old candidate set | Numerically large superset but doesn't target bug geometry. | |

**User's choice:** Worst `|qS-π/2|` (Recommended). Locked as D-23.

| Option | Description | Selected |
|--------|-------------|----------|
| Event + old/new indices + Fisher block | Full pickle schema; distinguishes "fix landed" from "upstream input changed". | ✓ |
| Minimal: old/new index sets | Passes superset assertion; loses Fisher context. | |
| Full candidate hosts dump | Heavy; most info derivable from catalog_index lookups. | |

**User's choice:** Event + old/new indices + Fisher block (Recommended). Locked as D-24.

---

## Physics-change cadence + xfail removal

| Option | Description | Selected |
|--------|-------------|----------|
| One review, four [PHYSICS] commits | Single /physics-change protocol pass covering all four; four atomic commits per REQ-ID. | ✓ |
| Four separate reviews + four commits | Four full protocol passes. More review rounds; slower. | |
| One review, one bundled commit | Single atomic commit across all four. Destroys bisectability. | |

**User's choice:** One review, four [PHYSICS] commits (Recommended). Locked as D-25.

| Option | Description | Selected |
|--------|-------------|----------|
| Per REQ as they land | xfail markers removed in the SAME commit as their corresponding REQ-ID. | ✓ |
| Bulk at phase end | All markers removed in one commit after all four REQs land. CI red during intermediate commits. | |
| Per REQ, non-strict during transition | Flip strict=True → strict=False temporarily. Fragile. | |

**User's choice:** Per REQ as they land (Recommended). Locked as D-26.

| Option | Description | Selected |
|--------|-------------|----------|
| Superset pickle + all Phase 35 tests green | `36-superset-regression.pkl` + `pytest test_coordinate_roundtrip.py` fully green. | ✓ |
| Posterior re-evaluation at h=0.73 | Duplicates Phase 40 VERIFY-02; confuses ownership. | |
| Green pytest suite + manual frame audit | Manual note unstructured; harder for Phase 40 to automate. | |

**User's choice:** Superset pickle + all Phase 35 tests green (Recommended). Locked as D-27.

| Option | Description | Selected |
|--------|-------------|----------|
| Phase 35 tests don't go green | If after all four commits xfail tests do NOT flip XPASS (or <99% recovery), STOP. | ✓ |
| Superset contract violates | If worst-equator reference event's new candidate set does NOT contain old set, STOP. | |
| Both of the above | Two independent gates. | |

**User's choice:** Phase 35 tests don't go green (Recommended). Locked as D-28.

---

## Claude's Discretion

- Exact signature of `_polar_to_cartesian` (scalar/array, tuple/NDArray) — keep vectorized-friendly.
- Exact 5D metric weights for 4D BallTree fix — planner/researcher recommends; user approves if physics-significant.
- BallTree distance metric post-fix (`euclidean` vs `haversine`) — sklearn API cleanliness.
- Docstring phrasing (NumPy-style per repo convention).
- Reference-comment citation per REQ (astropy docs / `_sky_localization_uncertainty` precedent / IAU 2006).

## Deferred Ideas

- Deprecated `get_possible_hosts` removal → Phase 39 HPC-04.
- Idempotency guard on `_map_angles_to_spherical_coordinates` → Phase 37 COORD-05.
- Per-parameter `derivative_epsilon` interaction with rotated frame → Phase 37 PE-02.
- `flip_hx` verification → Phase 39 HPC-05.
- 5D metric weights for COORD-02b → researcher/planner.
- Posterior re-evaluation at h=0.73 → Phase 40 VERIFY-02.
- CI integration / PubFigs Phase 35 naming collision → already recorded in Phase 35.
