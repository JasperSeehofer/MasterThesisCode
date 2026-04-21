# Phase 36: Coordinate Frame Fix - Context

**Gathered:** 2026-04-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the three critical coordinate-frame requirements locked in REQUIREMENTS.md — missing equatorial→ecliptic rotation on GLADE ingestion (COORD-03), latitude-vs-polar Cartesian embedding in both 3D BallTree call sites (COORD-02), and axis-aligned search radius on the 2×2 sky covariance (COORD-04) — plus a discussion-added fourth requirement: flat-metric bug on the 4D BallTree sky sub-space (COORD-02b), which is used by the simulation pipeline and would hit Phase 41/42 if those conditional campaigns fire.

All four changes are physics changes per CLAUDE.md's trigger list. Each requires a `/physics-change` review (old formula / new formula / reference / dimensional analysis / limiting case) and commits prefixed `[PHYSICS]`. Phase 35's xfail(strict=True) tests must flip XPASS and have their markers removed for each corresponding REQ-ID.

**Requirements:** COORD-02, COORD-02b (new — to be added to REQUIREMENTS.md traceability), COORD-03, COORD-04

**In scope:**
- New `_rotate_equatorial_to_ecliptic()` on `GalaxyCatalogueHandler`, called from `__init__` before `_map_angles_to_spherical_coordinates`
- New private `_polar_to_cartesian(theta, phi)` helper shared by `setup_galaxy_catalog_balltree` and `get_possible_hosts_from_ball_tree`
- Updated `setup_4d_galaxy_catalog_balltree` / `find_closest_galaxy_to_coordinates` to use spherical embedding on the sky sub-space (planner determines exact 5D metric balancing)
- Eigenvalue-based search radius `r = σ_mult × √λ_max(Σ')` where `Σ' = J Σ J^T`, `J = diag(|sin θ|, 1)`
- Extended `get_possible_hosts_from_ball_tree` signature with `cov_theta_phi: float = 0.0` kwarg; caller at `bayesian_statistics.py:773` starts passing `detection.theta_phi_covariance`
- Regression pickle `36-superset-regression.pkl` locked as Phase 40 VERIFY-02 anchor

**Out of scope (handled elsewhere):**
- Idempotency guard on `_map_angles_to_spherical_coordinates` → Phase 37 COORD-05
- Per-parameter `derivative_epsilon` / h-threading through `set_host_galaxy_parameters` → Phase 37 PE-01, PE-02
- Deprecated non-balltree `get_possible_hosts` removal → Phase 39 HPC-04 (dead-code cleanup)
- Posterior re-evaluation and >5% MAP-shift abort gate → Phase 40 VERIFY-02

</domain>

<decisions>
## Implementation Decisions

### Rotation Integration (COORD-03)

- **D-13:** Rotation lives in a new method `_rotate_equatorial_to_ecliptic()` on `GalaxyCatalogueHandler`, called from `__init__` BEFORE `_map_angles_to_spherical_coordinates`. The existing method becomes a pure `deg → rad → polar` transform on already-ecliptic angles. One method rotates, one converts. Single-responsibility; easy to `/physics-change`-review as a contained diff.
- **D-14:** Recompute rotation on every load. Vectorized `SkyCoord.transform_to(BarycentricTrueEcliptic(equinox='J2000'))` on ~230k galaxies is sub-second. Reduced CSV stays frame-agnostic; no cache-invalidation landmines if astropy or obliquity constants ever change.
- **D-15:** Vectorized array rotation: build one `SkyCoord(ra=ra_arr*u.deg, dec=dec_arr*u.deg, frame='icrs')`, call `.transform_to()` once, extract `lon` and `lat` back into the DataFrame. Hard assertions on the output range: `assert np.all((lon_rad >= 0) & (lon_rad < 2*np.pi))` and `assert np.all((lat_rad >= -np.pi/2) & (lat_rad <= np.pi/2))` before handing off to polar conversion. Fail loud; no silent coordinate drift.
- **D-16:** Fast falsifier during development = Phase 35 D-05 three-point round-trip: vernal equinox (RA=0,Dec=0)→(λ=0,β=0), summer solstice (RA=6h,Dec=+23.4°)→(λ=90°,β=0°), ecliptic pole (RA=18h,Dec=+66.56°)→(β=+90°), each agreeing to <0.001°. Already encoded in `master_thesis_code_test/test_coordinate_roundtrip.py`. Exercise these directly during dev.

### BallTree Fix Shape (COORD-02, COORD-02b)

- **D-17:** Introduce `_polar_to_cartesian(theta, phi)` as a module-level helper in `handler.py` that returns `(sin θ cos φ, sin θ sin φ, cos θ)` for `θ ∈ [0, π]`. Both `setup_galaxy_catalog_balltree` (line 286-288) and `get_possible_hosts_from_ball_tree` (line 307-309) call it. Symmetry guaranteed structurally — the two sites cannot drift in future edits.
- **D-18:** Fold the 4D BallTree flat-metric bug into Phase 36 as a new requirement **COORD-02b**. Same bug class (flat metric on spherical angles), same fix pattern (Cartesian embedding for sky + normalized z + normalized log M). This is safe for v2.2 ("re-evaluate existing CRBs, no re-simulation") because the 4D tree is used only by `find_closest_galaxy_to_coordinates` → `get_hosts_from_parameter_samples`, which runs in the **simulation** pipeline (main.py:388, scripts/quick_snr_calibration.py:53). No effect on existing CRBs; only affects future simulations. Without this fix, Phase 41/42 conditional injection campaigns would hit a latent second coordinate bug. Planner determines the exact 5D metric (3D sky Cartesian + z_norm + log_M_norm with axis weights TBD).
- **D-19:** Decisive regression-green signal for COORD-02 = Phase 35 D-04/D-06/D-07 tests flip XPASS: Dec=0° retrieval, Dec=90° NCP round-trip, and N=100 random ecliptic-equator galaxies with ≥99% recovery (seed=42). This is the verbatim ROADMAP §Phase 36 SC-1 criterion — no ambiguity at the handoff.
- **D-20:** Deprecated `get_possible_hosts` (handler.py:395, box-cut on unrotated (phi, theta), no BallTree) stays in place for Phase 36. Grep confirms no active callers. Defer removal to Phase 39 or a later cleanup. Out of physics-change scope.

### Search-Radius Eigenvalue Form (COORD-04)

- **D-21:** Extend `get_possible_hosts_from_ball_tree` signature with `cov_theta_phi: float = 0.0` kwarg alongside the existing `phi_sigma, theta_sigma`. Keeps backward compatibility; easy A/B to isolate the radius fix. Caller `bayesian_statistics.py:773-783` starts passing `detection.theta_phi_covariance`. Matches existing `_sky_localization_uncertainty(phi_error, theta, theta_error, cov_theta_phi)` precedent in `datamodels/detection.py:15`.
- **D-22:** Search-radius formula: `r = σ_mult × √λ_max(Σ')` where `Σ' = J Σ J^T`, `J = diag(|sin θ|, 1)` rescales the φ-component to great-circle distance on the unit sphere. `Σ = [[σ_φ², C_θφ], [C_θφ, σ_θ²]]` is the 2×2 sky covariance from the Cramér-Rao bound. `λ_max` is the larger eigenvalue. The BallTree query uses this radius as a chord-length search (`2 sin(r/2)` for angular-to-euclidean conversion on the unit sphere, if needed — planner decides the exact BallTree metric call). Reference: mirrors `_sky_localization_uncertainty`'s formulation of the sky error ellipse (Eq. in docstring at `detection.py:18-26`).
- **D-23:** Reference event for the superset pickle (ROADMAP §Phase 36 SC-4) is the event with minimum `|qS − π/2|` from `simulations/cramer_rao_bounds.csv` — the worst-case equator galaxy where both coordinate bugs bite hardest. Phase 35's `.planning/audit_coordinate_bug.json` already lists these events; pick the top of that list.
- **D-24:** Regression pickle `.planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl` contains: `{event_id, event_qS_pre_fix, event_phiS_pre_fix, event_qS_post_fix, event_phiS_post_fix, old_candidate_indices: set[int], new_candidate_indices: set[int], fisher_sky_2x2: ndarray, git_commit}`. Phase 40 VERIFY-02 can assert `old ⊆ new` AND `fisher_sky_2x2` unchanged to distinguish "fix landed" from "upstream input changed" (Phase 37 PE-02 epsilon changes could perturb the Fisher block — Phase 40 should not mis-attribute).

### Physics-Change Cadence + xfail Removal

- **D-25:** Single `/physics-change` protocol review at the start of Phase 36 execution covering all four REQ-IDs in one turn: for each of COORD-02, COORD-02b, COORD-03, COORD-04 the agent presents (a) old formula with file:line, (b) new formula, (c) reference (astropy docs, `_sky_localization_uncertainty` precedent, or IAU 2006 for obliquity), (d) dimensional analysis, (e) at least one limiting case. User approves all four together. Then four **separate atomic commits**, each prefixed `[PHYSICS]` and tagged with the REQ-ID: `[PHYSICS] COORD-02: polar-correct BallTree Cartesian embedding`, `[PHYSICS] COORD-02b: 4D BallTree sky sub-space metric`, `[PHYSICS] COORD-03: equatorial→ecliptic rotation on catalog ingestion`, `[PHYSICS] COORD-04: eigenvalue sky search radius`. Per-REQ bisectability preserved.
- **D-26:** Phase 35 `xfail(strict=True)` markers removed in the SAME commit as their corresponding REQ-ID. After `[PHYSICS] COORD-02`, remove xfail on the BallTree Dec=0°/Dec=90°/N-random tests. After `[PHYSICS] COORD-03`, remove xfail on the astropy ground-truth three-point test. This keeps CI catching regressions during execution — xfail(strict=True) stays as a live signal for REQs that haven't landed yet. NO bulk-removal commit at phase end.
- **D-27:** Decisive Phase 36 deliverable = two artifacts: (a) `36-superset-regression.pkl` committed under the phase dir; (b) `uv run pytest master_thesis_code_test/test_coordinate_roundtrip.py -v` passes with ≥9 passed, 0 xfail, 0 XPASS remaining. This is the handoff contract with Phase 40 VERIFY-02.
- **D-28:** **Stop/rethink gate (mandatory):** If after all four commits the Phase 35 tests do NOT flip XPASS (or do flip but with <99% recovery on the N=100 random test), STOP. Do not remove xfail markers. Do not advance to Phase 37. Ask the user before any further commits. This catches either an incomplete fix or a misframed test expectation before compounding errors.

### Claude's Discretion

- Exact signature of `_polar_to_cartesian(theta, phi)` — whether it accepts scalars, arrays, or both; whether it returns a tuple `(x, y, z)` or a stacked `NDArray[np.float64]`. Keep vectorized-friendly.
- Exact 5D metric weights for the 4D BallTree fix (D-18) — balancing sky Cartesian, z_norm, and log_M_norm scales. Planner/researcher recommends; user approves if physics-significant.
- BallTree distance metric for the 3D sky tree after fix: `"euclidean"` on unit sphere with chord-length radius, or explicit `"haversine"` — whichever the sklearn API handles cleanest. Document choice in the method docstring.
- Docstring phrasing for the four new/modified methods. Follow NumPy-style per repo convention.
- Whether the reference comment above each changed line cites the astropy docstring, `_sky_localization_uncertainty`, or IAU 2006 — per-REQ judgment.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirement / plan artifacts
- `.planning/REQUIREMENTS.md` §Coordinate Frame Correctness — COORD-02, COORD-03, COORD-04 specs. **Add COORD-02b** to this file during Phase 36 execution.
- `.planning/ROADMAP.md` §Phase 36 — success criteria 1–5 (including "≥99% recovery rate" and "superset" wording).
- `~/.claude/plans/i-want-a-last-elegant-feather.md` §Phase B — source plan for Phase 36 scope.
- `.planning/phases/35-coordinate-bug-characterization/35-CONTEXT.md` — fixtures module, xfail strategy, decisions D-01..D-12.
- `.planning/phases/35-coordinate-bug-characterization/35-VERIFICATION.md` — Phase 35 RED-test state.

### Bug loci (production code)
- `master_thesis_code/galaxy_catalogue/handler.py:281-293` — `setup_galaxy_catalog_balltree` with buggy latitude-formula embedding (COORD-02).
- `master_thesis_code/galaxy_catalogue/handler.py:295-315` — `get_possible_hosts_from_ball_tree` with same buggy embedding + axis-aligned search radius at line 313 (COORD-02, COORD-04).
- `master_thesis_code/galaxy_catalogue/handler.py:356-371` — `setup_4d_galaxy_catalog_balltree` with flat-metric bug on sky axes (COORD-02b).
- `master_thesis_code/galaxy_catalogue/handler.py:373-390` — `find_closest_galaxy_to_coordinates` uses the 4D tree (COORD-02b downstream).
- `master_thesis_code/galaxy_catalogue/handler.py:486-492` — `_map_angles_to_spherical_coordinates` converts deg→polar WITHOUT rotation (COORD-03).

### Caller / consumer surfaces
- `master_thesis_code/bayesian_inference/bayesian_statistics.py:773-783` — caller of `get_possible_hosts_from_ball_tree`, currently passes only `phi_sigma, theta_sigma`; must start passing `detection.theta_phi_covariance`.
- `master_thesis_code/datamodels/detection.py:107` — `Detection.theta_phi_covariance = parameters["delta_phiS_delta_qS"]` (source of the off-diagonal CRB element).
- `master_thesis_code/main.py:388` — simulation pipeline uses 4D tree via `get_hosts_from_parameter_samples`.
- `scripts/quick_snr_calibration.py:53, 69` — SNR calibration uses 4D tree.

### Physics / math precedent
- `master_thesis_code/datamodels/detection.py:15-40` — `_sky_localization_uncertainty(phi_error, theta, theta_error, cov_theta_phi)` uses `|sin θ|` Jacobian + off-diagonal covariance. Phase 36 eigenvalue radius mirrors this formulation.
- `astropy.coordinates.SkyCoord.transform_to(BarycentricTrueEcliptic(equinox='J2000'))` — authoritative equatorial↔ecliptic converter.
- IAU 2006 obliquity ≈ 23.4392911° — eyeball sanity check only; astropy is source of truth.

### Test scaffolding (from Phase 35)
- `master_thesis_code_test/fixtures/coordinate.py` — `synthetic_catalog_builder`, `equatorial_to_ecliptic_astropy`, `build_balltree`. Reuse in Phase 36 regression tests.
- `master_thesis_code_test/test_coordinate_roundtrip.py` — 9 tests, 6 `xfail(strict=True)`; Phase 36 removes xfail markers per REQ-ID as fixes land.
- `.planning/audit_coordinate_bug.json` — JSON sidecar identifying the worst-|qS−π/2| events. Pick reference event (D-23) from the top of this list.
- `master_thesis_code_test/conftest.py` — session `apply_style()` fixture; trust it.

### Project-level discipline
- `CLAUDE.md` §Math/Physics Validation Workflow — `/physics-change` protocol. Four applications required.
- `CLAUDE.md` §Git convention for physics changes — `[PHYSICS]` commit subject prefix.
- `CLAUDE.md` §Physics-change trigger files — includes `galaxy_catalogue/handler.py` via the galaxy distribution / cosmological model list (galactic catalog affects `cosmological_model.py` inputs).
- `CLAUDE.md` §HPC/GPU Best Practices — vectorized array operations; no Python loops in hot paths.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `master_thesis_code_test/fixtures/coordinate.py` — Phase 35 built the ground-truth astropy helper and synthetic-catalog builder. Phase 36 regression tests import directly; no duplication.
- `_sky_localization_uncertainty` at `datamodels/detection.py:15` — encoded `|sin θ|`-Jacobian + off-diagonal covariance in a sky ellipse already. Phase 36 eigenvalue radius uses the same J-scaling.
- `master_thesis_code/plotting/apply_style.py` + `_helpers.save_figure()` — if Phase 36 generates any debug plots, use these (no direct `pyplot` imports).
- `astropy` is a first-class dependency (already used in `constants.py`); no `pyproject.toml` changes needed.

### Established Patterns
- `_map_*` naming prefix for catalog-mutation methods on `GalaxyCatalogueHandler` — follow with `_rotate_equatorial_to_ecliptic`.
- Private module-level helpers use `_snake_case` (`_empiric_stellar_mass_to_BH_mass_relation`, `_polar_angle_to_declination` both exist at the bottom of handler.py). `_polar_to_cartesian` fits.
- Seed-pinned `np.random.default_rng(seed=42)` for reproducible random tests.
- `@pytest.mark.xfail(strict=True)` is idiomatic per Phase 35.
- Commit discipline: `[PHYSICS]` prefix for physics changes; pre-commit hooks (ruff+mypy) run automatically.

### Integration Points
- Handler `__init__` order (handler.py:163-173) is the insertion point for `_rotate_equatorial_to_ecliptic()`: after `_map_stellar_masses_to_BH_masses()`, before `_map_angles_to_spherical_coordinates()`.
- `get_possible_hosts_from_ball_tree` caller at `bayesian_statistics.py:773-783` must be updated alongside the signature extension in a [PHYSICS] COORD-04 commit.
- Phase 40 VERIFY-02 reads `36-superset-regression.pkl` from `.planning/phases/36-coordinate-frame-fix/` — keep the schema stable once committed.
- `master_thesis_code_test/integration/conftest.py:117-119` calls `setup_galaxy_catalog_balltree()` + `setup_4d_galaxy_catalog_balltree()` — integration tests will exercise the fix automatically.

</code_context>

<specifics>
## Specific Ideas

- Method sketch (all numerical details deferred to planner):
  ```python
  def _rotate_equatorial_to_ecliptic(self) -> None:
      """Rotate catalog RA/Dec from equatorial J2000 to ecliptic SSB.

      After this method, PHI_S column holds ecliptic longitude (deg, [0, 360))
      and THETA_S holds ecliptic latitude (deg, [-90, +90]). Subsequent call
      to `_map_angles_to_spherical_coordinates` converts degrees to polar
      angle θ ∈ [0, π] with θ=0 at the north ecliptic pole.

      References:
          astropy.coordinates.BarycentricTrueEcliptic(equinox='J2000')
      """
      ra = self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S].values
      dec = self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S].values
      coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
      ecl = coord.transform_to(BarycentricTrueEcliptic(equinox='J2000'))
      lon_deg = ecl.lon.to(u.deg).value % 360.0
      lat_deg = ecl.lat.to(u.deg).value
      assert np.all((lon_deg >= 0) & (lon_deg < 360))
      assert np.all((lat_deg >= -90) & (lat_deg <= 90))
      self.reduced_galaxy_catalog[InternalCatalogColumns.PHI_S] = lon_deg
      self.reduced_galaxy_catalog[InternalCatalogColumns.THETA_S] = lat_deg
  ```
- Polar-to-Cartesian helper:
  ```python
  def _polar_to_cartesian(theta: npt.NDArray[np.float64], phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
      """Map polar (θ, φ) to Cartesian unit vector. θ ∈ [0, π] with θ=0 at north pole.

      Returns stacked (N, 3) array. Scalars auto-broadcast.

      Reference: Eq. (derived); standard spherical convention.
      """
      return np.vstack((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))).T
  ```
- Eigenvalue radius sketch:
  ```python
  Sigma = np.array([[phi_sigma**2, cov_theta_phi], [cov_theta_phi, theta_sigma**2]])
  J = np.diag([abs(np.sin(theta)), 1.0])
  Sigma_scaled = J @ Sigma @ J.T
  lambda_max = np.linalg.eigvalsh(Sigma_scaled).max()
  radius = sigma_multiplier * np.sqrt(lambda_max)
  ```
- Commit messages (locked templates):
  - `[PHYSICS] COORD-02: polar-correct BallTree Cartesian embedding (sin θ cos φ, sin θ sin φ, cos θ)`
  - `[PHYSICS] COORD-02b: 4D BallTree sky sub-space uses spherical embedding`
  - `[PHYSICS] COORD-03: equatorial→ecliptic rotation on GLADE ingestion via astropy BarycentricTrueEcliptic(J2000)`
  - `[PHYSICS] COORD-04: eigenvalue sky search radius on 2×2 Fisher covariance with |sin θ| Jacobian`

</specifics>

<deferred>
## Deferred Ideas

- **Deprecated `get_possible_hosts` (non-balltree) method removal** — out of Phase 36 scope; Phase 39 HPC-04 covers dead-code cleanup, or a later standalone pass.
- **Idempotency guard on `_map_angles_to_spherical_coordinates`** — Phase 37 COORD-05 per REQUIREMENTS.md.
- **Per-parameter `derivative_epsilon` interaction with the rotated frame** — Phase 37 PE-02 lands on top of the fixed frame; verify Phase 36 tests stay green after Phase 37.
- **`flip_hx` verification against current `fastlisaresponse`** — Phase 39 HPC-05.
- **Exact 5D metric weights for the 4D BallTree (COORD-02b)** — planner researches; user approves during planning or execution if the weight choice is physics-significant.
- **Re-evaluation of posterior at h=0.73 under v2.2 corrections** — explicitly Phase 40 VERIFY-02, not Phase 36.
- **CI integration, PubFigs Phase 35 naming collision** — Phase 35 already recorded this deferral; still applies.

</deferred>

---

*Phase: 36-coordinate-frame-fix*
*Context gathered: 2026-04-22*
