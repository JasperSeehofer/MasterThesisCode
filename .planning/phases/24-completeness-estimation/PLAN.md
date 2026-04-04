---
phase: 24-completeness-estimation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - master_thesis_code/galaxy_catalogue/glade_completeness.py
  - master_thesis_code/physical_relations.py
  - tests/test_glade_completeness.py
interactive: false

conventions:
  units: "SI with dimensionless h = H0/(100 km/s/Mpc)"
  metric: "mostly-plus"
  coordinates: "spherical"
  distance: "luminosity distance in Gpc (dist() return value) and Mpc (catalog data)"

dimensional_check:
  f_z: '[dimensionless, in [0,1]]'
  d_L: '[Gpc] from dist(), [Mpc] in catalog data'
  dVc_dz: '[Mpc^3/sr]'

approximations:
  - name: "angle-averaged completeness"
    parameter: "f(z) independent of sky direction"
    validity: "LISA EMRI sky localization ~0.01-1 deg^2, much smaller than angular variation scale of GLADE+"
    breaks_when: "events in regions with anomalous catalog depth (e.g., SDSS footprint boundary)"
    check: "Compare f(z) values with Dalya et al. (2022) Fig. 2 digitized data"

  - name: "fiducial cosmology for distance-redshift mapping"
    parameter: "h=0.73, Omega_m=0.25 for digitized data conversion"
    validity: "The digitized data was produced at fixed cosmology; trial-h conversion is handled at lookup time"
    breaks_when: "never -- the interface accepts h as parameter and converts internally"
    check: "f(z=0, h=any) = 1.0; f(z) monotonically non-increasing"

contract:
  scope:
    question: "Can we provide f(z) as an interpolatable function of redshift that returns the GLADE+ B-band luminosity completeness fraction?"
  claims:
    - id: claim-fz-range
      statement: "f(z) returns values in [0, 1] for all z in the EMRI detection range (z=0.03-0.20)"
      deliverables: [deliv-completeness-module]
      acceptance_tests: [test-fz-bounds, test-fz-dalya]
      references: [ref-dalya2022]
    - id: claim-fz-correct
      statement: "f(z) reproduces known GLADE+ completeness: ~90% at z~0.029, declining to <<50% at z>0.11"
      deliverables: [deliv-completeness-module]
      acceptance_tests: [test-fz-dalya]
      references: [ref-dalya2022]
    - id: claim-h-dependence
      statement: "f(z, h) correctly maps distance-based completeness data to redshift for any trial h value"
      deliverables: [deliv-completeness-module]
      acceptance_tests: [test-fz-h-dependence]
      references: [ref-gray2020]
    - id: claim-vectorized
      statement: "f(z) is callable with scalar and array inputs, suitable for integration in bayesian_statistics.py"
      deliverables: [deliv-completeness-module]
      acceptance_tests: [test-fz-vectorized]
      references: []
    - id: claim-dVc
      statement: "comoving volume element dVc/dz is dimensionally correct and matches known limiting cases"
      deliverables: [deliv-volume-element]
      acceptance_tests: [test-dVc-dimensions, test-dVc-limit]
      references: [ref-hogg1999]
  deliverables:
    - id: deliv-completeness-module
      kind: code
      path: "master_thesis_code/galaxy_catalogue/glade_completeness.py"
      description: "Refactored GLADE+ completeness with f(z,h) interface returning fractions in [0,1]"
      must_contain: ["get_completeness_fraction", "get_completeness_at_redshift"]
    - id: deliv-volume-element
      kind: code
      path: "master_thesis_code/physical_relations.py"
      description: "Comoving volume element dVc/dz/dOmega function"
      must_contain: ["comoving_volume_element"]
    - id: deliv-tests
      kind: code
      path: "tests/test_glade_completeness.py"
      description: "Test suite for completeness function and comoving volume element"
      must_contain: ["test_completeness_fraction_bounds", "test_completeness_at_redshift"]
  acceptance_tests:
    - id: test-fz-bounds
      subject: claim-fz-range
      kind: consistency
      procedure: "Evaluate f(z) for z in np.linspace(0, 0.25, 100). Assert all values in [0, 1]."
      pass_condition: "0 <= f(z) <= 1 for all z"
      evidence_required: [deliv-completeness-module, deliv-tests]
    - id: test-fz-dalya
      subject: claim-fz-correct
      kind: benchmark
      procedure: "Evaluate f(z=0.029) and f(z=0.11) at h=0.73 and compare to Dalya et al. (2022) values."
      pass_condition: "f(z=0.029) in [0.85, 0.95]; f(z=0.11) < 0.50"
      evidence_required: [deliv-completeness-module, ref-dalya2022]
    - id: test-fz-h-dependence
      subject: claim-h-dependence
      kind: consistency
      procedure: "At fixed d_L, higher h -> lower z. So f(z, h=0.6) should differ from f(z, h=0.86). Verify d_L(z, h) mapping is used correctly."
      pass_condition: "f(z=0.05, h=0.6) != f(z=0.05, h=0.86) -- different h maps z=0.05 to different distances"
      evidence_required: [deliv-completeness-module]
    - id: test-fz-vectorized
      subject: claim-vectorized
      kind: consistency
      procedure: "Call f(z_array) with z_array = np.array([0.01, 0.05, 0.10, 0.15]). Verify output shape matches input."
      pass_condition: "Output is array of same shape; values match scalar calls"
      evidence_required: [deliv-completeness-module]
    - id: test-dVc-dimensions
      subject: claim-dVc
      kind: dimensional_analysis
      procedure: "Verify dVc/dz has units of Mpc^3/sr. At z=0: dVc/dz -> (c/H0)^3 * z^2 * ... -> check numerical value."
      pass_condition: "dVc/dz(z=0.01) ~ 4.3e4 * (c/H0)^2 * (1/H0) Mpc^3/sr (order of magnitude)"
      evidence_required: [deliv-volume-element]
    - id: test-dVc-limit
      subject: claim-dVc
      kind: limiting_case
      procedure: "At z<<1: dVc/dz/dOmega ~ (c/H0)^3 * z^2. Verify this scaling numerically."
      pass_condition: "dVc(0.001)/dVc(0.002) ~ (0.001/0.002)^2 = 0.25 within 5%"
      evidence_required: [deliv-volume-element]
  references:
    - id: ref-dalya2022
      kind: paper
      locator: "Dalya et al. (2022), arXiv:2110.06184, Section 3"
      role: benchmark
      why_it_matters: "GLADE+ completeness characterization provides the ground truth f(z) values we must reproduce"
      applies_to: [claim-fz-correct, claim-fz-range]
      must_surface: true
      required_actions: [compare, cite]
    - id: ref-gray2020
      kind: paper
      locator: "Gray et al. (2020), arXiv:1908.06050, Sec. II.3.1 and Appendix A.2"
      role: method
      why_it_matters: "Establishes the completeness-corrected likelihood framework; f(z,h) interface must be compatible"
      applies_to: [claim-h-dependence]
      must_surface: true
      required_actions: [read, cite]
    - id: ref-hogg1999
      kind: paper
      locator: "Hogg (1999), arXiv:astro-ph/9905116, Eq. (28)"
      role: definition
      why_it_matters: "Defines the comoving volume element dVc/dz/dOmega used in the completion term"
      applies_to: [claim-dVc]
      must_surface: true
      required_actions: [read, compare, cite]
  forbidden_proxies:
    - id: fp-no-raw-catalog
      subject: claim-fz-correct
      proxy: "Recomputing completeness from raw GLADE+ catalog data instead of using existing digitized Dalya et al. (2022) data"
      reason: "Raw catalog computation is a separate multi-day task; the digitized data already captures the B-band luminosity comparison result"
    - id: fp-no-angular
      subject: claim-fz-range
      proxy: "Implementing angular-dependent completeness f(z, Omega) instead of angle-averaged f(z)"
      reason: "Angular dependence is a second-order correction for LISA EMRI; adds complexity without addressing the primary bias"
    - id: fp-no-bayesian-mod
      subject: claim-vectorized
      proxy: "Modifying bayesian_statistics.py to wire in completeness in this phase"
      reason: "Likelihood modification is Phase 25 scope; this phase delivers the completeness function only"
  uncertainty_markers:
    weakest_anchors:
      - "The digitized completeness data assumes fiducial cosmology h=0.73 for the distance axis. The h-dependent redshift mapping introduces cosmology dependence that is second-order."
    disconfirming_observations:
      - "If f(z=0.029, h=0.73) is NOT approximately 0.90, the digitized data does not match Dalya et al. (2022) and should be re-examined."
      - "If f(z) is not monotonically non-increasing (after initial plateau), the interpolation may have artifacts."

estimated_execution:
  total_minutes: 40
  breakdown:
    - task: 1
      minutes: 25
      note: "Refactor completeness module + add redshift interface + comoving volume element"
    - task: 2
      minutes: 15
      note: "Test suite with physics validation"

patterns_consulted:
  insights: []
  error_patterns: []
  adjustments_made: []
---

<objective>
Refactor the existing GLADE+ completeness module to provide f(z, h) as an interpolatable function returning completeness fractions in [0, 1], and add the comoving volume element to physical_relations.py.

Purpose: Phase 25 (likelihood correction) needs f(z) to weight the catalog vs completion terms in the dark siren likelihood. This phase delivers the completeness function; Phase 25 wires it into bayesian_statistics.py.

Output: Refactored glade_completeness.py with redshift-based interface, comoving_volume_element() in physical_relations.py, comprehensive test suite.
</objective>

<execution_context>
@/home/jasper/.claude/get-physics-done/workflows/execute-plan.md
@/home/jasper/.claude/get-physics-done/templates/summary.md
</execution_context>

<context>
@.gpd/PROJECT.md
@.gpd/ROADMAP.md
@.gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md
@master_thesis_code/galaxy_catalogue/glade_completeness.py
@master_thesis_code/physical_relations.py
@master_thesis_code/constants.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Refactor GladeCatalogCompleteness with f(z, h) interface and add comoving volume element</name>
  <files>
    master_thesis_code/galaxy_catalogue/glade_completeness.py
    master_thesis_code/physical_relations.py
  </files>
  <action>
    **Part A: Refactor glade_completeness.py**

    The existing `GladeCatalogCompleteness` dataclass has hardcoded distance (Mpc) and completeness (%) arrays digitized from Dalya et al. (2022) Fig. 2. Refactor to add:

    1. `get_completeness_fraction(distance_mpc: float) -> float`: Returns completeness as a FRACTION in [0, 1] (not percentage). Wraps existing `np.interp` but divides by 100. For distance > max (795.7 Mpc), extrapolate with the last value (~21.3%) rather than returning 0. For distance < 0, return 1.0.

    2. `get_completeness_at_redshift(z: float | ndarray, h: float = 0.73) -> float | ndarray`: The key new method.
       - Converts z to luminosity distance in Mpc using `dist(z, h) * GPC_TO_MPC` (dist returns Gpc)
       - Calls `get_completeness_fraction(d_L_mpc)`
       - Must handle both scalar and array z input (use `np.atleast_1d`, compute, squeeze back)
       - Clamp output to [0, 1]

    3. Keep the existing `get_completeness(distance: float) -> float` for backward compatibility but add a deprecation note in the docstring.

    4. Add a `get_completeness_fraction_vectorized(distance_mpc: ndarray) -> ndarray` for efficient array operations (just `np.interp(...) / 100.0` with appropriate edge handling).

    5. Use `scipy.interpolate.interp1d` with `kind='linear'` and `fill_value=(1.0, last_value/100)` as an alternative to repeated `np.interp` calls. Store as a lazy-initialized attribute for the interpolator. Actually, `np.interp` is already vectorized and efficient -- keep it simple. Use `np.interp` with `left=1.0` (complete at d=0) and `right=last_completeness_value/100.0` (extrapolate flat beyond max distance).

    6. Add proper type annotations on all methods per CLAUDE.md conventions. Add NumPy-style docstrings with References section citing Dalya et al. (2022), arXiv:2110.06184.

    7. Add `ASSERT_CONVENTION` comment at the top of any new physics function: distance in Mpc, completeness as fraction [0,1], dist() returns Gpc.

    Import `dist` (or `dist_vectorized`) from `physical_relations` and `GPC_TO_MPC` from `constants`.

    **Part B: Add comoving_volume_element to physical_relations.py**

    Add function `comoving_volume_element(z, h, Omega_m, Omega_de)`:

    ```
    dV_c/dz/dOmega = d_com^2(z) * c / H(z)
    ```

    where:
    - `d_com = d_L / (1+z)` is comoving distance
    - `d_L = dist(z, h) * GPC_TO_MPC` in Mpc
    - `H(z) = h * 100 * E(z)` in km/s/Mpc (use `hubble_function` for E(z))
    - `c = SPEED_OF_LIGHT_KM_S` in km/s

    Result is in Mpc^3/sr.

    This is a PHYSICS CHANGE requiring the Physics Change Protocol:
    - Old formula: None (new function)
    - New formula: dV_c/dz/dOmega = d_com^2 * c / H(z) [Mpc^3/sr]
    - Reference: Hogg (1999), arXiv:astro-ph/9905116, Eq. (28)
    - Dimensional analysis: [Mpc]^2 * [km/s] / [km/s/Mpc] = [Mpc^3] per steradian -- correct
    - Limiting case: At z << 1, d_com ~ c*z/H0, H(z) ~ H0, so dV/dz ~ (c/H0)^3 * z^2

    Add reference comment above the function: `# Eq. (28) in Hogg (1999), arXiv:astro-ph/9905116`

    Must handle both scalar and array z input. Use existing `hubble_function()` which already handles both.
  </action>
  <verify>
    1. **Completeness fraction bounds**: `get_completeness_fraction(0) == 1.0`, `get_completeness_fraction(200) ~ 0.44`, `get_completeness_fraction(800) ~ 0.21`, `get_completeness_fraction(1000) ~ 0.21` (flat extrapolation)
    2. **Redshift interface at fiducial h=0.73**: `get_completeness_at_redshift(0.0) == 1.0`, `get_completeness_at_redshift(0.029) ~ 0.90` (Dalya et al. reference point), `get_completeness_at_redshift(0.11) < 0.50`
    3. **h-dependence**: `get_completeness_at_redshift(0.05, h=0.6) != get_completeness_at_redshift(0.05, h=0.86)` (higher h -> larger d_L at same z -> lower completeness)
    4. **Vectorized**: `get_completeness_at_redshift(np.array([0.01, 0.05, 0.1]))` returns array of shape (3,)
    5. **Backward compat**: `get_completeness(200)` still returns ~44 (percentage, not fraction)
    6. **Comoving volume element dimensional check**: dVc/dz at z=0.01 should be ~ (c/H0)^2 * z^2 * c/H0 ~ (4115 Mpc)^2 * 0.0001 * 4115 ~ 7e6 Mpc^3/sr (order of magnitude)
    7. **Comoving volume element z^2 scaling**: dVc(z=0.002) / dVc(z=0.001) ~ 4.0 (quadratic scaling at low z)
    8. **Type annotations**: mypy passes on both modified files
  </verify>
  <done>
    glade_completeness.py has get_completeness_fraction() and get_completeness_at_redshift(z, h) returning fractions in [0,1].
    physical_relations.py has comoving_volume_element(z, h) returning dVc/dz/dOmega in Mpc^3/sr.
    Both handle scalar and array inputs. All verification checks pass.
  </done>
</task>

<task type="auto">
  <name>Task 2: Test suite for completeness function and comoving volume element</name>
  <files>
    tests/test_glade_completeness.py
  </files>
  <action>
    Create a comprehensive test suite. Tests must run on CPU-only dev machine (no GPU marker needed).

    **Completeness fraction tests:**

    1. `test_completeness_fraction_bounds`: f(z) in [0, 1] for z in np.linspace(0, 0.25, 200) at h=0.73.
    2. `test_completeness_fraction_at_zero`: f(z=0) == 1.0 for any h.
    3. `test_completeness_fraction_monotonic`: f(z) is non-increasing (within small tolerance for interpolation noise -- the raw data has a small non-monotonicity around 200-240 Mpc where completeness rises slightly from 43.0 to 43.6; allow tolerance of 0.01 for this).
    4. `test_completeness_dalya_reference_points`: f(z~0.029, h=0.73) in [0.85, 0.95]; f(z~0.11, h=0.73) < 0.50. These are from Dalya et al. (2022) Section 3.
    5. `test_completeness_h_dependence`: At fixed z=0.05, f differs between h=0.6 and h=0.86. Higher h -> d_L is larger -> lower completeness. Assert f(z=0.05, h=0.86) < f(z=0.05, h=0.6).
    6. `test_completeness_vectorized`: Input array produces output array of same shape; values match scalar calls element-wise.
    7. `test_completeness_edge_cases`: z < 0 -> 1.0 (or raises); z very large (z=1.0) -> small positive value (flat extrapolation, not zero or NaN).
    8. `test_backward_compat_get_completeness`: Old `get_completeness(200)` returns ~44 (percentage units).

    **Comoving volume element tests:**

    9. `test_comoving_volume_element_positive`: dVc/dz > 0 for z > 0.
    10. `test_comoving_volume_element_z_squared_scaling`: At z << 1, dVc/dz scales as z^2. Check ratio dVc(0.002)/dVc(0.001) ~ 4.0 within 5%.
    11. `test_comoving_volume_element_at_z_zero`: dVc/dz(z=0) == 0 (or very close).
    12. `test_comoving_volume_element_dimensions`: Spot check at z=0.1, h=0.73. d_com ~ dist(0.1, 0.73)*1000/(1.1) Mpc. H(z=0.1) ~ 73 * sqrt(0.25*1.1^3 + 0.75). Compute expected dVc/dz and compare.

    Use `pytest.approx` for floating-point comparisons with appropriate tolerances (rtol=0.05 for physics comparisons, rtol=1e-10 for exact mathematical relations).

    Import from `master_thesis_code.galaxy_catalogue.glade_completeness` and `master_thesis_code.physical_relations`.
  </action>
  <verify>
    1. All tests pass with `uv run pytest tests/test_glade_completeness.py -v`
    2. No GPU imports or markers needed
    3. Tests cover all acceptance_tests from the contract
    4. `uv run mypy master_thesis_code/galaxy_catalogue/glade_completeness.py master_thesis_code/physical_relations.py` passes
    5. `uv run ruff check master_thesis_code/galaxy_catalogue/glade_completeness.py master_thesis_code/physical_relations.py tests/test_glade_completeness.py` passes
  </verify>
  <done>
    Test suite with 12+ tests covers bounds, Dalya reference points, h-dependence, vectorization, edge cases, backward compat, and comoving volume element physics.
    All tests pass. Linting and type checking pass.
  </done>
</task>

</tasks>

<verification>
- f(z) in [0, 1] for all z in EMRI range (0.03-0.20) -- contract claim-fz-range
- f(z=0.029) ~ 0.90, f(z=0.11) < 0.50 -- matches Dalya et al. (2022) -- contract claim-fz-correct
- f(z, h) correctly handles h-dependent distance-redshift mapping -- contract claim-h-dependence
- comoving_volume_element has correct dimensions (Mpc^3/sr) and z^2 low-z scaling -- contract claim-dVc
- Backward compatibility: existing get_completeness() unchanged
- No modifications to bayesian_statistics.py (that is Phase 25 scope)
</verification>

<success_criteria>
1. f(z) curve matches Dalya et al. (2022) figures: ~90% at z=0.029, declining to <<50% at z>0.11
2. f(z) returns values in [0, 1] for all z in EMRI detection range (z=0.03-0.20) with no extrapolation failures
3. f(z) is available as an interpolating function callable from bayesian_statistics.py without loading the full catalog each time
4. comoving_volume_element(z, h) available for Phase 25's completion term
5. All tests pass, mypy clean, ruff clean
</success_criteria>

<output>
After completion, create `.gpd/phases/24-completeness-estimation/24-01-SUMMARY.md`
</output>
