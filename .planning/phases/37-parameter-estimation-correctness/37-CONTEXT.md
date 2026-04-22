# Phase 37: Parameter Estimation Correctness - Context

**Gathered:** 2026-04-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove the hidden coupling between the injected Hubble constant and saved Fisher CRBs, replace the
uniform `derivative_epsilon=1e-6` with per-parameter values appropriate to each parameter's scale,
and close four latent hygiene holes: swapped `Omega_m` limits, scattered SNR threshold literals,
hardcoded `SPEED_OF_LIGHT_KM_S`, and missing idempotency guard on the angle-mapping method.

**In scope:**
- COORD-05: Idempotency guard on `_map_angles_to_spherical_coordinates` in `handler.py`
- PE-01: Thread injected `h` from the simulation loop into `set_host_galaxy_parameters`
- PE-02: Per-parameter `derivative_epsilon` on each `Parameter` instance in `ParameterSpace` (physics change)
- PE-03: Fix swapped `Omega_m` lower/upper limits in `LamCDMScenario` (one-liner)
- PE-04: Unify SNR threshold — fix stale `15` literals, add `PRE_SCREEN_SNR_FACTOR = 0.3`
- PE-05: `SPEED_OF_LIGHT_KM_S = C / 1000` derived from the existing `C` constant

**Out of scope:**
- Re-evaluation of existing CRBs under the fixed h-threading — Phase 40 VERIFY-02
- Statistical correctness fixes (L_cat form, P_det extrapolation) — Phase 38
- HPC/visualization wins (flush interval, FFT cache, CPU-importable module) — Phase 39
- Posterior computation or evaluation pipeline changes — no `set_host_galaxy_parameters` call there

</domain>

<decisions>
## Implementation Decisions

### PE-01: h-threading

- **D-01:** Add `h: float` as a **required keyword argument** (no default) to `set_host_galaxy_parameters`:
  ```python
  def set_host_galaxy_parameters(self, host_galaxy: HostGalaxy, h: float) -> None:
      ...
      self.luminosity_distance.value = dist(host_galaxy.z, h=h)
  ```
  Calling without `h` raises `TypeError` — satisfies SC-2 directly. No stored state on `ParameterSpace`.

- **D-02:** The one caller in the simulation pipeline (`main.py:394`) passes `h_value` from the simulation arguments:
  ```python
  parameter_estimation.parameter_space.set_host_galaxy_parameters(host_galaxy, h=h_value)
  ```

- **D-03:** The `h` here is the **simulation truth h** (h_inj), not the evaluation-side h sweep. The evaluation pipeline reads CRBs from CSV and never calls `set_host_galaxy_parameters`. The injection pipeline at `main.py:650-651` already bypasses this function via `dist(sample.redshift, h=h_value)` — the workaround comment there should be removed once PE-01 lands.

- **D-04:** Regression test `test_parameter_space_h.py` pins the h-ratio: `set_host_galaxy_parameters(host, h=0.5)` must yield exactly half the luminosity distance of `set_host_galaxy_parameters(host, h=1.0)`, within floating-point tolerance.

### PE-02: Per-parameter derivative_epsilon

- **D-05:** The `Parameter` dataclass at `parameter_space.py:28-38` already has `derivative_epsilon: float = 1e-6` per instance. `parameter_estimation.py:163` already reads `derivative_epsilon = parameter.derivative_epsilon` per-parameter — the architecture is in place.

- **D-06:** The fix is to set appropriate non-default `derivative_epsilon` values in each `Parameter`'s `default_factory` lambda in `ParameterSpace`. No dict needed; no new fields needed. The GPD executor will determine specific values for all 14 parameters under the `/physics-change` protocol.

- **D-07:** Scale guidance for the GPD agent (for context; final values to be derived under physics protocol):
  - `M` (~10^4–10^6 solar masses): absolute epsilon ~1–10 solar masses (not 1e-6)
  - `d_L` (~0.1–2 Gpc): absolute epsilon ~1e-3 Gpc (not 1e-6 Gpc = 1 pc)
  - `mu`, `a`, `e0`, `x0`, angles, phases: current 1e-6 may be adequate — verify under protocol

- **D-08:** Success criterion: Fisher determinant on a seed-pinned representative event (seed=42, one event from `simulations/cramer_rao_bounds.csv`) must differ by less than 1% after switching. Verified across at least 3 random-perturbation resamples to confirm non-degeneracy.

- **D-09:** This is a **physics change** (PE-02 + parts of PE-01). GPD executor must run `/physics-change` covering: old epsilon → new epsilon for each parameter, dimensional analysis on step-size stability, reference (Vallisneri 2008 arXiv:gr-qc/0703086 on stencil step selection for Fisher matrices), and at least one limiting case (e.g., diagonal-dominant Fisher stays diagonal-dominant under step-size change).

### PE-03: Omega_m limits (one-liner)

- **D-10:** `cosmological_model.py:319-322`: `LamCDMScenario.Omega_m` has `upper_limit=0.04, lower_limit=0.5` (swapped). Fix to `lower_limit=0.04, upper_limit=0.5`. Both values are in the physically plausible range [0.04, 0.5] — no physics-change protocol required (constraint range fix, no formula change).

### PE-04: SNR threshold unification

- **D-11:** Add `PRE_SCREEN_SNR_FACTOR: float = 0.3` to `constants.py` alongside `SNR_THRESHOLD`. This names the pre-screen heuristic at `main.py:421`.

- **D-12:** Four-site update:
  - `constants.py`: add `PRE_SCREEN_SNR_FACTOR = 0.3`
  - `cosmological_model.py:171`: `snr_threshold: int = 15` → `snr_threshold: float = SNR_THRESHOLD` (import `SNR_THRESHOLD` from constants)
  - `paper_figures.py:553`: `snr_threshold: float = 15.0` → `snr_threshold: float = SNR_THRESHOLD`
  - `main.py:421`: `cosmological_model.snr_threshold * 0.3` → `cosmological_model.snr_threshold * PRE_SCREEN_SNR_FACTOR`

- **D-13:** `bayesian_plots.py:370` has `snr_threshold: float = 20.0` — value is already consistent with `SNR_THRESHOLD`. Claude's discretion whether to import the constant or leave the literal. Lean toward importing for single-source-of-truth, but not required.

- **D-14:** Smoke test: grep for literal `15` adjacent to `snr` token returns empty in `master_thesis_code/` (excluding comments and test files), confirming no stale references remain. This is SC-5.

### PE-05: Speed of light (one-liner)

- **D-15:** `constants.py:21`: `SPEED_OF_LIGHT_KM_S: float = 300000.0` → `SPEED_OF_LIGHT_KM_S: float = C / 1000`. `C` (m/s) is already defined two lines above. Test: `comoving_volume_element` result must match to 14 decimal places — SC-6.

### COORD-05: Idempotency guard

- **D-16:** Add `_angles_mapped_to_ecliptic: bool = False` field (or equivalent flag) to `GalaxyCatalogueHandler`. At the top of `_map_angles_to_spherical_coordinates`:
  ```python
  assert not self._angles_mapped_to_ecliptic, (
      "_map_angles_to_spherical_coordinates called twice — "
      "angles are already in ecliptic frame"
  )
  self._angles_mapped_to_ecliptic = True
  ```
  Raises `AssertionError` on second call; existing single-call path unaffected. SC-7.

### Claude's Discretion

- Whether `bayesian_plots.py:370` imports `SNR_THRESHOLD` or keeps the matching `20.0` literal.
- Exact field name for the idempotency flag on `GalaxyCatalogueHandler` (any bool flag is fine).
- Whether to add the `derivative_epsilon` regression test in an existing test file or a new `test_parameter_space_h.py` (the h-ratio test must be in a dedicated file per SC-1; epsilon regression can be alongside).
- Commit ordering: hygiene fixes (PE-03, PE-04, PE-05, COORD-05) can land as one or multiple software commits; PE-01 is one commit; PE-02 is one `[PHYSICS]` commit after the physics-change review.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` §Parameter Estimation — COORD-05, PE-01..PE-05 specs
- `.planning/ROADMAP.md` §Phase 37 — success criteria SC-1..SC-7 (verbatim gate conditions)
- `~/.claude/plans/i-want-a-last-elegant-feather.md` §Phase C — source plan for Phase 37 scope

### Code loci (production code to change)
- `master_thesis_code/datamodels/parameter_space.py:28-38` — `Parameter` dataclass with `derivative_epsilon` (PE-02 target)
- `master_thesis_code/datamodels/parameter_space.py:144-148` — `set_host_galaxy_parameters` (PE-01 target)
- `master_thesis_code/parameter_estimation/parameter_estimation.py:163` — reads `parameter.derivative_epsilon` per-param (already correct architecture)
- `master_thesis_code/constants.py:21,48` — `SPEED_OF_LIGHT_KM_S` and `SNR_THRESHOLD` (PE-04, PE-05 targets)
- `master_thesis_code/cosmological_model.py:171,319-322` — stale `snr_threshold: int = 15` and swapped `Omega_m` limits (PE-03, PE-04 targets)
- `master_thesis_code/main.py:394,421,650-651` — caller of `set_host_galaxy_parameters` + pre-screen coefficient + injection workaround
- `master_thesis_code/plotting/paper_figures.py:553` — stale `snr_threshold: float = 15.0`
- `master_thesis_code/galaxy_catalogue/handler.py:612` — `_map_angles_to_spherical_coordinates` (COORD-05 target)

### Prior phase context
- `.planning/phases/36-coordinate-frame-fix/36-CONTEXT.md` — D-24 documents that the regression pickle `fisher_sky_2x2` may be perturbed by PE-02 epsilon changes; Phase 40 VERIFY-02 must account for this
- `.planning/phases/36-coordinate-frame-fix/36-VERIFICATION.md` — Phase 36 tests must remain GREEN after Phase 37 lands

### Physics method reference
- Vallisneri (2008) arXiv:gr-qc/0703086 — optimal step-size selection for numerical derivatives in Fisher matrices (cite in PE-02 `/physics-change` review)

### Existing tests
- `master_thesis_code_test/test_coordinate_roundtrip.py` — 9 tests, all GREEN after Phase 36; must remain GREEN after COORD-05 guard lands
- `master_thesis_code_test/conftest.py` — session fixtures; do NOT duplicate

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Parameter.derivative_epsilon: float` — already a per-instance field on the `Parameter` dataclass; `parameter_estimation.py` already reads it per-parameter. No structural change needed for PE-02 — only set different default values in the `ParameterSpace` factory lambdas.
- `dist(redshift, h=H, ...)` in `physical_relations.py:28` — already accepts `h` as a keyword argument. PE-01 just threads it explicitly.
- `SNR_THRESHOLD` in `constants.py:48` — already authoritative; PE-04 makes it the sole reference point.
- `C` in `constants.py:19` — already defined via astropy; PE-05 derives `SPEED_OF_LIGHT_KM_S` from it.

### Established Patterns
- `@pytest.mark.xfail(strict=True)` for regression gates; `np.random.default_rng(seed=42)` for reproducibility.
- `[PHYSICS]` commit prefix for any change to a formula or constant — required for PE-02.
- Pre-commit hooks (ruff + mypy) run automatically on commit; no `--no-verify`.
- Physics-change protocol: old formula → new formula → reference → dimensional analysis → limiting case.

### Integration Points
- `main.py:394` — the only real caller of `set_host_galaxy_parameters` in the simulation pipeline. Update to pass `h=h_value`.
- `main.py:650-651` — injection pipeline workaround; remove the workaround comment once PE-01 lands.
- `bayesian_statistics.py:313` — already imports and uses `SNR_THRESHOLD` from constants; that caller is correct.
- Phase 36 regression pickle at `.planning/phases/36-coordinate-frame-fix/36-superset-regression.pkl` — the `fisher_sky_2x2` block may shift slightly after PE-02 epsilon changes; Phase 40 VERIFY-02 should use the Phase 37 post-fix value as its reference, not the Phase 36 pickle.

</code_context>

<specifics>
## Specific Ideas

- PE-01 test signature (from success criteria SC-1):
  ```python
  host = HostGalaxy(z=0.1, ...)
  ps_half = ParameterSpace()
  ps_half.set_host_galaxy_parameters(host, h=0.5)
  ps_one  = ParameterSpace()
  ps_one.set_host_galaxy_parameters(host, h=1.0)
  np.testing.assert_allclose(
      ps_one.luminosity_distance.value / ps_half.luminosity_distance.value,
      2.0, rtol=1e-10
  )
  ```

- COORD-05 idempotency guard sketch:
  ```python
  # In GalaxyCatalogueHandler.__init__ or class body:
  _angles_mapped_to_ecliptic: bool = False

  def _map_angles_to_spherical_coordinates(self) -> None:
      assert not self._angles_mapped_to_ecliptic, (
          "_map_angles_to_spherical_coordinates already called — "
          "angles are in ecliptic frame"
      )
      self._angles_mapped_to_ecliptic = True
      # ... existing body ...
  ```

- Constants additions (PE-04, PE-05):
  ```python
  SPEED_OF_LIGHT_KM_S: float = C / 1000  # km/s, derived from C (m/s)
  SNR_THRESHOLD: float = 20
  PRE_SCREEN_SNR_FACTOR: float = 0.3  # pre-screen heuristic in main.py
  ```

</specifics>

<deferred>
## Deferred Ideas

- **Vallisneri optimal-step formula implementation**: automated per-run step-size selection based on parameter value and waveform curvature. Out of scope for Phase 37 — static per-parameter dict is sufficient. Could revisit in a future optimization phase.
- **Re-evaluation of existing CRBs under h-threading** — Phase 40 VERIFY-02 does this.
- **L_cat / P_det statistical correctness** — Phase 38.
- **HPC optimizations** — Phase 39.

</deferred>

---

*Phase: 37-parameter-estimation-correctness*
*Context gathered: 2026-04-22*
