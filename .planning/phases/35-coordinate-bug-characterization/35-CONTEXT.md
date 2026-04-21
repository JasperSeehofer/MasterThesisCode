# Phase 35: Coordinate Bug Characterization - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Encode the two critical coordinate bugs discovered in the 2026-04-21 pre-batch audit (missing equatorial→ecliptic rotation; latitude-vs-polar Cartesian embedding in BallTree) in failing tests **before any production code is touched**, and snapshot a pre-fix baseline of how many production events lie in the ±5° ecliptic-equator danger zone.

This is pure test infrastructure + one audit artifact — no physics formulas change. Phase 36 applies the fix; Phase 35 builds the scaffolding that verifies Phase 36 worked.

**In scope:**
- New test file `master_thesis_code_test/test_coordinate_roundtrip.py` with RED (xfail(strict=True)) tests for both bugs
- New fixtures module `master_thesis_code_test/fixtures/coordinate.py` reused by Phase 36
- Baseline audit artifact `.planning/audit_coordinate_bug.md` + structured JSON sidecar

**Out of scope (handled in Phase 36 under `/physics-change`):**
- `astropy.SkyCoord.transform_to(BarycentricTrueEcliptic())` rotation in `handler.py:_map_angles_to_spherical_coordinates`
- Polar-correct BallTree Cartesian embedding `(sin θ cos φ, sin θ sin φ, cos θ)`
- Eigenvalue-based sky-covariance search radius with `|sin θ|` Jacobian

</domain>

<decisions>
## Implementation Decisions

### RED-test Encoding (xfail strategy)

- **D-01:** Use `@pytest.mark.xfail(strict=True, reason="Phase 36 fixes coordinate frame bug — see .planning/milestones/v2.2-...")` on correct-behavior assertions. CI stays green through Phase 35; when Phase 36 lands the fix, tests become XPASS and CI fails until someone removes the xfail marker. This forces a conscious handoff and prevents silent regression.
- **D-02:** Single RED test per bug (no duplicate bug-pinning tests). Each test asserts the CORRECT behavior (e.g., "BallTree recovers the synthetic Dec=0° galaxy with ≥99% recovery rate"). Phase 36 success criterion = remove xfail markers and tests go GREEN.
- **D-03:** No plain-assert-on-wrong-behavior tests. All new tests describe the desired post-fix state; the xfail strict marker is the only signal of pre-fix status.

### Mandatory + Edge-Case Test Scenarios

The roadmap locks three mandatory scenarios; discussion added four more for obliquity/pole/statistical coverage:

- **D-04:** Mandatory (from ROADMAP.md):
  - (a) Synthetic galaxy at ecliptic Dec=0° (ecliptic equator after rotation) — BallTree must retrieve it for a query at the same position.
  - (b) Synthetic galaxy at Dec=90° (NCP) — round-trip recovery through catalog ingestion + BallTree query.
  - (c) `astropy.coordinates.SkyCoord` ground-truth comparison on RA/Dec → ecliptic (λ, β) to tolerance <0.001°.
- **D-05:** Added edge cases:
  - Vernal equinox (RA=0, Dec=0) — catches "no rotation at all" bugs. On-axis trap; not sufficient alone but useful as a reference.
  - Summer solstice (RA=6h, Dec=+23.4°) → ecliptic (λ=90°, β=0°) — OFF equatorial equator, ON ecliptic equator. Isolates obliquity sign/magnitude bugs cleanly.
  - Ecliptic pole (RA=18h, Dec=+66.56°) → β=+90°, θ_polar=0. Verifies pole handling + sign-of-obliquity.
  - N random ecliptic-equator galaxies — seed-pinned statistical recovery gate.

### Recovery-Rate Gate

- **D-06:** The N-random ecliptic-equator test gates on **≥99% recovery rate**, matching the Phase 36 success criterion verbatim (ROADMAP.md §Phase 36, criterion 1: "recovery rate ≥99% over randomized offsets"). Using the same threshold on both sides of the handoff removes ambiguity when Phase 36 flips xfail → green.
- **D-07:** Random sampling config: `np.random.default_rng(seed=42)`, `N=100` galaxies. Deterministic, reproducible, roomy enough that ≥99% means ≤1 miss tolerated. Matches the project's seed-pinned regression pattern.

### Test Fixtures Layout

- **D-08:** Create a dedicated module `master_thesis_code_test/fixtures/coordinate.py` (new directory following the existing `fixtures/` pattern). Exports at least:
  - `synthetic_catalog_builder(n, sky_band)` — builds an in-memory `GalaxyCatalogueHandler`-compatible DataFrame with controlled sky positions.
  - `equatorial_to_ecliptic_astropy(ra_deg, dec_deg)` — ground-truth astropy wrapper returning (λ_rad, β_rad) and θ_polar.
  - Lightweight `build_balltree(catalog)` helper that calls into the real `setup_galaxy_catalog_balltree`.
- **D-09:** Phase 36 imports from this same fixtures module for its regression tests (post-fix green assertions). No duplication across phases.

### Baseline Audit Artifact (`.planning/audit_coordinate_bug.md`)

- **D-10:** Audit scope = the 42-event production CRB CSV (`simulations/cramer_rao_bounds.csv`) **only**. The 27 h-value posteriors in `simulations/posteriors/` reuse the same event sky positions, so scanning them adds no new information.
- **D-11:** Audit contents (all four additions enabled):
  - Event counts + fractions in **multiple bands**: ±5°, ±10°, ±15° around the ecliptic equator (`|qS − π/2| < band × π/180`). Mandatory ±5° is kept; additional bands show how rapidly risk falls off.
  - **Histogram of `|qS − π/2|`** binned across all 42 events, committed as `.planning/audit_coordinate_bug_histogram.png` (matplotlib, Agg backend, `apply_style()` applied).
  - **Expected-vs-observed under isotropic prior**: under an isotropic sky prior, sin θ dθ gives expected fraction in ±5° ≈ sin(5°·π/180) ≈ 8.7%. Report observed vs expected; large deviations flagged as selection-effect candidates.
  - **Structured JSON sidecar** `.planning/audit_coordinate_bug.json` with machine-readable fields (`event_count`, `band_counts`, `band_fractions`, `expected_fraction_5deg`, `csv_source_path`, `git_commit`, `timestamp`). Phase 40 VERIFY-04 anisotropy audit can diff post-fix numbers directly.
- **D-12:** Baseline artifact + JSON sidecar + histogram PNG all committed to git. Phase 36/40 reference them by path, not by re-computation.

### Claude's Discretion

- Exact field names in the JSON sidecar (beyond the listed fields) — keep it minimal and additive.
- Histogram bin count and range — pick sensible defaults for 42 events (likely 10–15 bins, range [0, π/2]).
- Docstring phrasing in the new test file.
- Whether `equatorial_to_ecliptic_astropy` returns a namedtuple, dict, or tuple.
- How to implement the baseline audit generator — standalone pytest test that writes the artifact + JSON under a `[audit]` marker, or a CLI subcommand, or a script in `scripts/`. Recommended: keep it as a one-shot module invocable from the command line (e.g., `python -m master_thesis_code_test.audit_coordinate_bug`) rather than a stateful test fixture.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Plan artifacts
- `~/.claude/plans/i-want-a-last-elegant-feather.md` §Phase A — source plan for Phase 35 scope, test list, and audit artifact format.
- `.planning/REQUIREMENTS.md` — COORD-01 requirement spec (tests RED, baseline committed).
- `.planning/ROADMAP.md` §Phase 35 — success criteria 1–4.

### Bug evidence
- `master_thesis_code/galaxy_catalogue/handler.py:281-293` — `setup_galaxy_catalog_balltree` with buggy latitude-formula embedding.
- `master_thesis_code/galaxy_catalogue/handler.py:295-315` — `get_possible_hosts_from_ball_tree` with same buggy embedding + axis-aligned search radius at line 313.
- `master_thesis_code/galaxy_catalogue/handler.py:486-492` — `_map_angles_to_spherical_coordinates`: converts Dec→polar angle but applies NO equatorial→ecliptic rotation.

### Production data under audit
- `simulations/cramer_rao_bounds.csv` — 42 events, column `qS` holds recovered ecliptic polar angle. Primary audit target.
- `simulations/posteriors/h_*.json` — 27 h-value posteriors (not directly audited; same underlying events).

### Test scaffolding reference
- `master_thesis_code_test/conftest.py` — session-level fixtures incl. `apply_style()` call and repo-root discovery. Do NOT duplicate.
- `master_thesis_code_test/fixtures/` — existing fixtures directory pattern to follow.
- `master_thesis_code/plotting/_helpers.py` — `save_figure()` for the histogram PNG.

### Physics/coordinate ground truth
- `astropy.coordinates.SkyCoord` with `frame='icrs'` + `.transform_to(BarycentricTrueEcliptic(equinox='J2000'))` — authoritative equatorial↔ecliptic conversion.
- IAU 2006 obliquity ≈ 23.4392911° — documented for eyeball sanity checks only; astropy is the source of truth.

### Downstream phases that consume Phase 35 artifacts
- Phase 36 success criterion 1: the xfail tests from this phase must flip GREEN after the fix.
- Phase 40 VERIFY-04: anisotropy audit diffs the JSON sidecar against post-fix re-audit.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `master_thesis_code/plotting/apply_style.py` — call `apply_style()` in the histogram-generation code per project convention.
- `master_thesis_code/plotting/_helpers.py:save_figure()` — standard figure-saving utility.
- `master_thesis_code_test/conftest.py` — do NOT override the session `apply_style()` fixture; trust it.
- `master_thesis_code/constants.py` — already imports `astropy.constants` and `astropy.units`, so astropy is a first-class dependency. No pyproject.toml changes needed for `astropy.coordinates`.

### Established Patterns
- Test files live flat under `master_thesis_code_test/` as `test_<topic>.py`. The new file follows this: `test_coordinate_roundtrip.py`.
- Fixtures live under `master_thesis_code_test/fixtures/` (directory exists per plan). New module: `fixtures/coordinate.py`.
- `@pytest.mark.xfail(strict=True)` is idiomatic; pytest is configured with default strict_markers.
- Seed-pinned regression tests use `np.random.default_rng(seed=...)` — follow this pattern for the N-random test.
- Figures saved via `save_figure()` with matplotlib Agg backend; NEVER import `pyplot` at module level.
- `.planning/debug/` and `.planning/*.md` artifacts are committed to git for cross-phase reference (precedent: `.planning/debug/baseline.json`, `.planning/debug/comparison_*.json`).

### Integration Points
- `test_coordinate_roundtrip.py` imports from `master_thesis_code.galaxy_catalogue.handler` — the buggy functions are `setup_galaxy_catalog_balltree`, `get_possible_hosts_from_ball_tree`, `_map_angles_to_spherical_coordinates`.
- The `.planning/audit_coordinate_bug.md` + JSON are referenced by Phase 36's post-fix regression and Phase 40's verification gate — keep the JSON schema stable once landed.
- Baseline audit generator reads `simulations/cramer_rao_bounds.csv` via `pandas.read_csv`; column of interest is `qS`.

</code_context>

<specifics>
## Specific Ideas

- Test naming: single file `test_coordinate_roundtrip.py`, one class or top-level test functions with descriptive names (`test_ball_tree_recovers_ecliptic_equator_galaxy`, `test_equatorial_to_ecliptic_matches_astropy_to_0_001_deg`, `test_n_random_ecliptic_equator_galaxies_recovery_rate_above_99pct`, etc.).
- Astropy call pattern (for the ground-truth helper):
  ```python
  from astropy.coordinates import SkyCoord, BarycentricTrueEcliptic
  coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
  ecl = coord.transform_to(BarycentricTrueEcliptic(equinox='J2000'))
  lambda_rad = ecl.lon.to(u.rad).value
  beta_rad = ecl.lat.to(u.rad).value
  theta_polar_rad = np.pi/2 - beta_rad
  ```
- Baseline audit invocation pattern: keep the generator standalone (a module or script), NOT a pytest fixture — artifact must be reproducible outside the test suite for Phase 40 re-runs.
- The audit artifact's markdown should include the band-count table inline, the histogram image embedded via relative path, and a literal block containing the JSON sidecar path for cross-referencing.

</specifics>

<deferred>
## Deferred Ideas

- **Eigenvalue-based sky-ellipse search radius** (COORD-04) — Phase 36.
- **Equatorial→ecliptic rotation in `_map_angles_to_spherical_coordinates`** (COORD-03) — Phase 36.
- **Idempotency guard on `_map_angles_to_spherical_coordinates`** (COORD-05) — Phase 37.
- **`flip_hx` verification against current fastlisaresponse** (HPC-05) — Phase 39.
- **CI integration / PubFigs Phase 35 naming collision** — deferred to when PubFigs resumes post-v2.2 (roadmap note already documented it).
- **Full h-sweep posterior re-audit** — user explicitly scoped Phase 35 audit to the single CRB CSV; per-h rescans happen at Phase 40 VERIFY-03 re-evaluation.
- **Near-pole Dec=±89° numerical stability test** — subsumed by the Dec=90° NCP test (case b). Revisit if NCP test reveals numerical issues.

</deferred>

---

*Phase: 35-coordinate-bug-characterization*
*Context gathered: 2026-04-21*
