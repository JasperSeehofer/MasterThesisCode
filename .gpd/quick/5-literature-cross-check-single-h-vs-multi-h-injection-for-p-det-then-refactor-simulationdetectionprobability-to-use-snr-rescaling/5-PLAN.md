---
phase: quick-5
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - master_thesis_code/bayesian_inference/simulation_detection_probability.py
  - master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py
  - .gpd/quick/5-literature-cross-check-single-h-vs-multi-h-injection-for-p-det-then-refactor-simulationdetectionprobability-to-use-snr-rescaling/5-SUMMARY.md
interactive: false

conventions:
  units: "SI (Gpc for distances, solar masses for M, km/s/Mpc for H0)"
  cosmology: "flat LCDM, Omega_m=0.25, H=0.73 (project fiducial)"
  snr_scaling: "SNR proportional to 1/d_L (amplitude scaling only)"

dimensional_check:
  d_L: "[Gpc]"
  SNR: "[dimensionless]"
  P_det: "[dimensionless, in 0..1]"

approximations:
  - name: "SNR proportional to 1/d_L"
    parameter: "d_L enters only as overall amplitude in h(t)"
    validity: "For fixed source-frame parameters (M, mu, a, e0, etc.), the waveform frequency content is independent of d_L. Only the amplitude scales as 1/d_L."
    breaks_when: "Cosmology-dependent selection effects beyond simple distance scaling exist (e.g., redshift-dependent waveform morphology, population model coupling to h). Literature check in Task 1 will verify this."
    check: "Numerical spot-check: compute SNR at two h values using dist(), compare ratio with d_L ratio."

contract:
  scope:
    question: "Can P_det(d_L, M | h) be computed from a single-h injection campaign via SNR rescaling d_L(z, h), avoiding per-h injection runs and interpolation artifacts?"
  claims:
    - id: claim-snr-rescaling-valid
      statement: "For fixed source-frame EMRI parameters, SNR scales as 1/d_L(z,h), so a single injection campaign at h_ref suffices to compute P_det at any h by rescaling d_L = dist(z, h_target)."
      deliverables: [deliv-literature-note, deliv-refactored-class]
      acceptance_tests: [test-literature-support, test-numerical-consistency]
      references: [ref-gray2020, ref-laghi2021]
    - id: claim-backward-compat
      statement: "The refactored SimulationDetectionProbability maintains the same public API (detection_probability_with_bh_mass_interpolated, detection_probability_without_bh_mass_interpolated) and produces numerically consistent results."
      deliverables: [deliv-refactored-class, deliv-tests]
      acceptance_tests: [test-api-compat, test-numerical-consistency]
      references: []
  deliverables:
    - id: deliv-literature-note
      kind: note
      path: ".gpd/quick/5-literature-cross-check-single-h-vs-multi-h-injection-for-p-det-then-refactor-simulationdetectionprobability-to-use-snr-rescaling/5-SUMMARY.md"
      description: "Literature cross-check findings on single-h vs multi-h injection for P_det"
    - id: deliv-refactored-class
      kind: code
      path: "master_thesis_code/bayesian_inference/simulation_detection_probability.py"
      description: "Refactored SimulationDetectionProbability using SNR rescaling"
      must_contain: ["dist_vectorized", "snr_rescaling", "_h_ref"]
    - id: deliv-tests
      kind: code
      path: "master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py"
      description: "Updated tests including SNR rescaling validation"
  references:
    - id: ref-gray2020
      kind: paper
      locator: "Gray et al. (2020), arXiv:1908.06050"
      role: benchmark
      why_it_matters: "Foundational dark siren framework; establishes how P_det enters the likelihood. Their injection methodology informs whether single-h rescaling is standard."
      applies_to: [claim-snr-rescaling-valid]
      must_surface: true
      required_actions: [read, cite]
    - id: ref-laghi2021
      kind: paper
      locator: "Laghi et al. (2021), arXiv:2102.01708"
      role: benchmark
      why_it_matters: "LISA EMRI dark siren study; their P_det construction methodology is directly comparable."
      applies_to: [claim-snr-rescaling-valid]
      must_surface: true
      required_actions: [read, cite]
  acceptance_tests:
    - id: test-literature-support
      subject: claim-snr-rescaling-valid
      kind: literature
      procedure: "Check 3+ dark siren papers for how they handle P_det across h values. Confirm SNR ~ 1/d_L is the standard assumption or identify any caveats."
      pass_condition: "Literature supports SNR rescaling approach OR identifies specific physics subtleties that must be addressed."
      evidence_required: [deliv-literature-note]
    - id: test-numerical-consistency
      subject: claim-snr-rescaling-valid
      kind: consistency
      procedure: "For synthetic injection data at h_ref=0.70, rescale to h=0.80 and compare P_det grid against independently constructed grid at h=0.80. Grids must agree within statistical noise."
      pass_condition: "Max |P_det_rescaled - P_det_direct| < 0.05 in bins with >10 events."
      evidence_required: [deliv-refactored-class, deliv-tests]
    - id: test-api-compat
      subject: claim-backward-compat
      kind: regression
      procedure: "All existing tests pass. Public methods accept same arguments. Pickle safety preserved."
      pass_condition: "pytest passes, no API signature changes in public methods."
      evidence_required: [deliv-tests]
  forbidden_proxies:
    - id: fp-untested-rescaling
      subject: claim-snr-rescaling-valid
      proxy: "Implementing rescaling without numerical verification that it reproduces per-h grid results"
      reason: "Would hide potential physics errors (e.g., if waveform morphology depends on redshift in a way that breaks the 1/d_L scaling)"
    - id: fp-literature-skip
      subject: claim-snr-rescaling-valid
      proxy: "Skipping literature check and assuming SNR ~ 1/d_L without verification"
      reason: "Dark siren literature may reveal subtleties (population model coupling, redshift-dependent selection) that invalidate the simple rescaling"
  links:
    - id: link-lit-to-impl
      source: claim-snr-rescaling-valid
      target: deliv-refactored-class
      relation: supports
      verified_by: [test-literature-support, test-numerical-consistency]
    - id: link-compat
      source: claim-backward-compat
      target: deliv-tests
      relation: supports
      verified_by: [test-api-compat]
  uncertainty_markers:
    weakest_anchors:
      - "Assumption that EMRI waveform frequency content is independent of cosmology (d_L enters only as amplitude). True for source-frame parameters but should be verified against FEW waveform model."
    disconfirming_observations:
      - "If P_det grids built via rescaling differ by >5% from per-h grids in well-populated bins, the 1/d_L scaling assumption is violated and a deeper investigation is needed."
---

<objective>
Cross-check with dark siren literature whether single-h injection with SNR rescaling is valid for computing P_det(d_L, M | h), then refactor SimulationDetectionProbability to use this approach.

Purpose: Eliminate interpolation artifacts from the current multi-h grid approach, pool all injection data into a single larger sample (improving per-bin statistics), and enable exact P_det evaluation at any h without requiring dedicated injection campaigns.
Output: Literature findings note, refactored SimulationDetectionProbability class, updated tests with rescaling validation.
</objective>

<execution_context>
@/home/jasper/.claude/get-physics-done/workflows/execute-plan.md
@/home/jasper/.claude/get-physics-done/templates/summary.md
</execution_context>

<context>
@.gpd/PROJECT.md
@.gpd/STATE.md
@master_thesis_code/bayesian_inference/simulation_detection_probability.py
@master_thesis_code/physical_relations.py
@master_thesis_code/constants.py
@master_thesis_code/bayesian_inference/bayesian_statistics.py
@master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Literature cross-check and physics validation of SNR rescaling</name>
  <files>.gpd/quick/5-literature-cross-check-single-h-vs-multi-h-injection-for-p-det-then-refactor-simulationdetectionprobability-to-use-snr-rescaling/5-SUMMARY.md</files>
  <action>
    Search dark siren literature for how P_det is computed across Hubble parameter values:

    1. Gray et al. (2020), arXiv:1908.06050 -- Check Section III and Appendix for how they construct P_det(theta | H0). Do they run separate injection campaigns per H0, or rescale from a reference?

    2. Laghi et al. (2021), arXiv:2102.01708 -- LISA EMRI dark sirens. How do they handle the H0 dependence of P_det? Do they use SNR rescaling?

    3. Finke et al. (2021), arXiv:2101.12660 -- Another dark siren approach. Check their P_det methodology.

    4. Any EMRI-specific dark siren papers (e.g., Laghi et al. 2022 if exists, or Babak et al.).

    For each paper, extract:
    - How P_det depends on cosmological parameters
    - Whether they use single-injection + rescaling or multi-injection approach
    - Any caveats about redshift-dependent waveform morphology or population model coupling

    Physics validation of the rescaling assumption:
    - The GW strain amplitude h(t) ~ 1/d_L for a source at luminosity distance d_L
    - For EMRI waveforms generated by FEW: the source-frame parameters (M, mu, a, p0, e0, Y0) determine the waveform shape (frequency evolution). d_L only enters as an overall amplitude prefactor.
    - Therefore SNR = sqrt(<h|h>) scales as 1/d_L exactly.
    - Changing h changes d_L(z, h) = (c(1+z)/H0) * integral, so SNR_new = SNR_ref * d_L_ref / d_L_new.
    - The source-frame mass M in the injection CSV is h-independent (it's the intrinsic mass).
    - The redshift z in the injection CSV is also h-independent (it's the true source redshift).

    Identify any subtleties:
    - Does the EMRI population model (event rate per comoving volume) depend on h? If so, the injection DRAWS would need re-weighting, not just SNR rescaling. Check whether the current SimulationDetectionProbability marginalizes over the population or just computes P(det | source params).
    - Selection effects beyond SNR threshold: are there any cosmology-dependent cuts (e.g., frequency band, observation time) that would break simple rescaling?

    Document findings in the SUMMARY file under a "Literature Findings" section.
  </action>
  <verify>
    1. At least 3 papers consulted with specific section/equation references
    2. Clear conclusion on whether single-h rescaling is standard practice or novel
    3. Any caveats identified with assessment of their relevance to this project
    4. Physics argument for SNR ~ 1/d_L validated against the FEW waveform model structure
  </verify>
  <done>Literature cross-check complete with clear go/no-go recommendation for SNR rescaling approach, documented with paper references.</done>
</task>

<task type="auto">
  <name>Task 2: Refactor SimulationDetectionProbability to use SNR rescaling</name>
  <files>master_thesis_code/bayesian_inference/simulation_detection_probability.py, master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py</files>
  <action>
    Refactor SimulationDetectionProbability to store raw injection data and compute P_det grids on-the-fly via SNR rescaling. Proceed only if Task 1 confirms the approach is valid (if Task 1 finds showstoppers, document them and stop).

    **Constructor changes:**
    1. Load ALL injection CSVs regardless of h label. Pool events from all h values into a single DataFrame.
    2. Store the raw per-event data: z, M, SNR_raw, d_L_raw, h_inj (the h at which each event was injected).
    3. Compute a reference SNR for each event that is h-independent: since SNR_raw was measured at d_L = dist(z, h_inj), define SNR_ref = SNR_raw * d_L_raw. This is the "SNR at 1 Gpc" -- an h-independent intrinsic loudness.
       - Alternatively, just store (z, M, SNR_raw, h_inj) and rescale at query time as SNR(h) = SNR_raw * dist(z, h_inj) / dist(z, h_query).
       - The second approach avoids storing d_L_raw and is more transparent. Use this.
    4. Remove per-h grid pre-computation from __init__. Grids will be built lazily or at query time.

    **Query-time grid construction:**
    1. Add a method `_get_or_build_grid(self, h: float) -> RegularGridInterpolator` that:
       a. Checks a cache dict `_grid_cache: dict[float, RegularGridInterpolator]`
       b. If cache miss: for each event, compute d_L_new = dist_vectorized(z_array, h=h) and SNR_new = SNR_raw * dist_vectorized(z_array, h=h_inj_array) / d_L_new
       c. Build the 2D grid from (d_L_new, M, SNR_new) using the existing _build_grid_2d logic
       d. Cache and return
    2. Similarly for 1D grids.
    3. Use an LRU-style cache with a reasonable max size (e.g., 20 entries) to avoid unbounded memory growth.

    **Important implementation details:**
    - Use `dist_vectorized` from physical_relations.py for efficient batch d_L computation (avoid per-event scalar `dist()` calls).
    - The bin edges (dl_edges, M_edges) should be recomputed per h since d_L ranges change with h. This is correct behavior -- at different h, the d_L range of the source population shifts.
    - Maintain the existing `_build_grid_2d` and `_build_grid_1d` methods as internal helpers. They already take a DataFrame and SNR threshold -- just feed them the rescaled data.
    - Keep the IS weighting infrastructure intact (the `weights` parameter in `_build_grid_2d`).

    **Public API preservation:**
    - `detection_probability_with_bh_mass_interpolated(d_L, M_z, phi, theta, *, h)` -- unchanged signature. Internally calls `_get_or_build_grid(h)` instead of `_interpolate_at_h`.
    - `detection_probability_without_bh_mass_interpolated(d_L, phi, theta, *, h)` -- unchanged signature.
    - `quality_flags(h)` -- should work with the new grid cache.
    - Pickle safety must be preserved (cache is a plain dict, should be fine).

    **Backward compatibility:**
    - The constructor should still accept `injection_data_dir` and `snr_threshold`.
    - The `h_grid` parameter becomes optional/ignored (since we no longer need a predefined h grid).
    - Add a deprecation warning if `h_grid` is passed, noting it's no longer needed.

    **Test updates:**
    1. Update existing tests to work with new internals (they access `_interpolators` directly -- update to use `_get_or_build_grid` or public API).
    2. Add a new test: `test_snr_rescaling_consistency` -- create synthetic data at h=0.70 with known SNR values. Query P_det at h=0.70 (should match direct grid). Query at h=0.80 and verify the rescaled grid is consistent with what you'd get from independent h=0.80 injection data.
    3. Add a test: `test_pools_all_h_values` -- create CSVs at h=0.70 and h=0.80, verify the constructor pools all events (total count = sum of both files).
    4. Ensure all existing tests still pass (pickle, empty dir, threshold filtering, API signatures).

    **Do NOT:**
    - Change the public method signatures (except allowing h_grid to be ignored)
    - Remove the IS weighting infrastructure
    - Change the bin count constants (_DL_BINS, _M_BINS)
    - Touch bayesian_statistics.py in this task (it already calls the public API correctly)

    Follow the Physics Change Protocol (/physics-change):
    - Old formula: P_det(d_L, M | h) built from per-h injection data, interpolated between h grid points
    - New formula: P_det(d_L, M | h) built from pooled injection data with SNR rescaled via SNR(h) = SNR_raw * d_L(z, h_inj) / d_L(z, h)
    - Reference: Gray et al. (2020) + findings from Task 1
    - Dimensional analysis: SNR is dimensionless, d_L ratio is dimensionless, product is dimensionless. Correct.
    - Limiting case: When h = h_inj for all events, d_L(z, h) = d_L(z, h_inj), so SNR(h) = SNR_raw. Grid should be identical to original per-h grid.
  </action>
  <verify>
    1. Limiting case: P_det grid at h=h_inj matches the old per-h grid exactly (no rescaling applied when h equals injection h)
    2. Dimensional analysis: SNR remains dimensionless after rescaling, d_L in Gpc throughout
    3. Monotonicity: For fixed (z, M), increasing h decreases d_L, increases SNR, increases P_det. Verify this trend in the test.
    4. Bounds: P_det in [0, 1] for all queries (enforced by np.clip, existing behavior)
    5. Pool size: constructor logs total pooled event count = sum of all CSV files
    6. All existing tests pass (regression)
    7. New rescaling consistency test passes
    8. `uv run pytest -m "not gpu and not slow"` passes
    9. `uv run ruff check master_thesis_code/bayesian_inference/simulation_detection_probability.py`
    10. `uv run mypy master_thesis_code/bayesian_inference/simulation_detection_probability.py`
  </verify>
  <done>SimulationDetectionProbability refactored to pool all injection data and compute P_det via SNR rescaling at query time. All existing tests pass, new rescaling validation test passes, linting and type checking clean.</done>
</task>

</tasks>

<verification>
- Literature confirms SNR ~ 1/d_L rescaling is physically valid for EMRI dark sirens
- Refactored class produces identical P_det when queried at h = h_inj (no-rescaling identity)
- P_det monotonicity with h is correct (higher h -> lower d_L -> higher SNR -> higher P_det at fixed z)
- All public API methods maintain backward compatibility
- Full test suite passes including new rescaling validation tests
</verification>

<success_criteria>
1. Literature cross-check documents 3+ papers with clear conclusion on SNR rescaling validity
2. SimulationDetectionProbability pools all injection data regardless of h_inj label
3. P_det at any h computed via SNR rescaling without interpolation between pre-built grids
4. Identity test: P_det(h=h_inj) matches old per-h grid construction
5. All tests pass: `uv run pytest -m "not gpu and not slow"`
6. Code quality: ruff + mypy clean
</success_criteria>

<output>
After completion, create `.gpd/quick/5-literature-cross-check-single-h-vs-multi-h-injection-for-p-det-then-refactor-simulationdetectionprobability-to-use-snr-rescaling/5-SUMMARY.md`
</output>
