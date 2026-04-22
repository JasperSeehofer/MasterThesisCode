# Phase 38: Statistical Correctness — Context

**Gathered:** 2026-04-22 (inline code audit — no discuss phase needed; requirements fully specified)
**Status:** Ready for planning

<domain>
## Phase Boundary

Reconcile the catalog likelihood implementation with Gray et al. (2020) Eq. 24-25 (STAT-01/STAT-02),
and make the P_det extrapolation policy symmetric between the single-host integrand and the
completion-term denominator (STAT-03/STAT-04). All four requirements are well-scoped with exact
code loci identified via pre-planning audit.

**In scope:**
- STAT-01: L_cat form — prove equivalence or fix to canonical (1/N)Σ(N_g/D_g)
- STAT-02: Unit test with 3 synthetic galaxies comparing both L_cat forms
- STAT-03: P_det extrapolation policy made symmetric (both zero-fill)
- STAT-04: Per-event diagnostic logging fraction of quadrature weight outside P_det grid
- SC-5: End-to-end smoke test `--evaluate` on N≤5 events

**Out of scope:**
- Full posterior re-evaluation with corrected formulas — Phase 40 VERIFY-02
- HPC optimizations (flush interval, FFT cache, CPU importability) — Phase 39
- Cluster submission or new injection campaigns — Phases 41-42

</domain>

<decisions>
## Pre-Planning Code Audit Results

### STAT-01: L_cat Form Analysis — FIX REQUIRED (not just proof)

**Current code** (`bayesian_statistics.py:930-947`):
```python
likelihood_without_bh_mass = np.sum(numerator_without_bh_mass)    # ΣN_g
selection_effect_correction = np.sum(denominators)                   # ΣD_g
L_cat = likelihood_without_bh_mass / selection_effect_correction   # ΣN_g/ΣD_g
```

**Gray et al. (2020) Eq. 24-25 with uniform 1/N galaxy prior requires:**
```
L_cat = (1/N) Σ_g (N_g / D_g)
```

**Proof of non-equivalence:**
- D_g = ∫_{z_g - 4σ_g}^{z_g + 4σ_g} p_det(d_L(z,h)) × N(z|z_g, σ_g) dz
- D_g VARIES per galaxy because:
  1. Integration limits [z_g ± 4σ_g] are galaxy-specific (different z_g, σ_z,g)
  2. p_det is not flat (it falls off with d_L); galaxies at different z → different d_L → different P_det contribution
- Counterexample confirming non-equivalence: N=2, N_1=1, D_1=1, N_2=1, D_2=2:
  Code: 2/3. Gray: (1/2)(1 + 1/2) = 3/4 ≠ 2/3
- Therefore: ΣN_g/ΣD_g ≠ (1/N)Σ(N_g/D_g) → **fix required via /physics-change protocol**

**Canonical fix** (D-01): Replace lines 943-944 (without BH mass) and 955-957 (with BH mass) with:
```python
per_galaxy_ratios = [n / d for n, d in zip(numerators, denominators) if d > 0]
N_gal = len(per_galaxy_ratios)
L_cat = float(np.mean(per_galaxy_ratios)) if N_gal > 0 else 0.0
```
Or equivalently:
```python
L_cat = float(np.mean([result[0] / result[1] for result in results if result[1] > 0]))
```

This is a **physics change** (STAT-01 + parts of Gray formula derivation). GPD executor must run
`/physics-change` covering: old ΣN_g/ΣD_g → new (1/N)Σ(N_g/D_g), dimensional analysis (both
dimensionless), reference: Gray et al. (2020) arXiv:1908.06050 Eq. 24-25, and limiting case:
single galaxy → both forms give N_1/D_1.

### STAT-02: Unit Test Decision

**D-02**: Write `test_l_cat_equivalence.py` with 3 synthetic galaxies constructed to have
different D_g values (e.g., z = [0.1, 0.3, 0.5] with σ_z = 0.001). Directly compute both
L_cat forms and assert they are NUMERICALLY DIFFERENT (to document the divergence), then
assert the NEW canonical form is what the production function returns.

The test must:
1. Verify the old ΣN_g/ΣD_g and new (1/N)Σ(N_g/D_g) produce measurably different values (non-trivial D_g spread)
2. Verify the production code path uses (1/N)Σ(N_g/D_g) post-fix

### STAT-03: P_det Extrapolation Asymmetry — FIX REQUIRED (GPD)

**Asymmetry identified:**
- `precompute_completion_denominator` (D(h)): calls `detection_probability_without_bh_mass_interpolated_zero_fill` → **zero-fill** outside grid
- `single_host_likelihood` numerator + denominator integrands (lines 1192, 1210): calls `detection_probability_without_bh_mass_interpolated` → **NN-fill** (fill_value=None)

**Physics rationale for symmetric zero-fill** (D-03):
- P_det outside the injection grid is unmodeled — no injection data → P_det is unknown, not equal to the boundary value
- Setting P_det=0 beyond the grid is conservative: underestimates detection probability for distant events but does not fabricate detections beyond coverage
- NN-fill propagates the boundary P_det value (≈ the last-measured near-zero value at high d_L) indefinitely beyond the grid edge, potentially overestimating P_det for events at d_L >> d_L,max
- The completion-term denominator D(h) already uses zero-fill correctly; the single_host_likelihood integrands must match

**Fix** (D-04): In `single_host_likelihood` (lines 1185-1213), replace:
- `detection_probability.detection_probability_without_bh_mass_interpolated(...)` → `detection_probability.detection_probability_without_bh_mass_interpolated_zero_fill(...)`
- Applied in: `numerator_integrant_without_bh_mass` (line 1192), `denominator_integrant_without_bh_mass` (line 1210)
- Apply the same fix to the `with_bh_mass` variants if they exist (check for `_with_bh_mass` integrands)

**Edge case note**: The comment at `simulation_detection_probability.py:455` says "fill_value=0.0 caused 44% of events to lose completeness correction because high-SNR events' 4σ integration bounds exceed the injection grid." This applies to the L_comp (completion term) numerator integration, where the ±4σ d_L window for a nearby high-SNR event might exceed the injection grid. With zero-fill in the single_host_likelihood too, some numerator integral mass may be lost. This must be acknowledged in the physics-change documentation, and the STAT-04 diagnostic will quantify the impact.

This is a **physics change** (STAT-03). GPD executor must run `/physics-change` covering:
- Old: NN-fill in single_host_likelihood
- New: zero-fill in single_host_likelihood
- Reference: physics reasoning (no injection data beyond grid = unknown P_det, conservative zero assumed)
- Dimensional analysis: P_det is dimensionless, integrands remain physically valid
- Limiting case: if the entire integration window falls within the grid, the change has no effect

### STAT-04: Per-Event Diagnostic Decision

**D-05**: Instrument the `fixed_quad` integrations in `single_host_likelihood` to measure the fraction of quadrature nodes that fall outside the P_det grid:
```python
# After each fixed_quad call, log fraction of quadrature points outside grid
quadrature_weight_outside_grid_numerator = _fraction_outside_grid(
    z_range, d_L_from_z, h, dl_min, dl_max
)
```

**D-06**: Add `quadrature_weight_outside_grid_numerator: float` and `quadrature_weight_outside_grid_denominator: float` to the return dict (or add to galaxy_likelihoods dict at line 1084).

**D-07**: WARNING fires if either fraction > 5% for a single event. The WARNING must include the event ID and both fractions.

**D-08**: The diagnostic is added after the STAT-03 fix is in place (same plan, depends on the P_det method change).

### Claude's Discretion

- The exact mechanism for measuring "fraction of quadrature weight outside grid" — options:
  1. Post-hoc: use the zero-fill vs NN-fill difference to estimate weight lost (compare the two calls)
  2. Direct: pass z-array from fixed_quad to a grid-coverage checker
  3. Simplified: check integration limits vs d_L(z_min/max, h) against grid bounds
  Option 3 is simplest: if d_L(z_upper, h) > dl_max of grid → fraction outside ≈ (z_upper - z_grid_edge_inv)/(z_upper - z_lower)
  
- Whether to add the diagnostic columns to the existing per-galaxy return dict or create a parallel diagnostic dict — adding to existing dict is simpler (D-06 leans that way)

- Commit ordering: STAT-03 P_det fix is one `[PHYSICS]` commit; STAT-01 L_cat fix is one `[PHYSICS]` commit; STAT-02/STAT-04 tests can land as software commits before or after the physics fixes

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements and roadmap
- `.planning/REQUIREMENTS.md` §Statistical Correctness — STAT-01..STAT-04 specs with exact SC descriptions
- `.planning/ROADMAP.md` §Phase 38 — success criteria SC-1..SC-5 (verbatim gate conditions)

### Code loci (production code to change)
- `master_thesis_code/bayesian_inference/bayesian_statistics.py:930-961` — L_cat computation: `ΣN_g/ΣD_g` → `(1/N)Σ(N_g/D_g)` (STAT-01 target)
- `master_thesis_code/bayesian_inference/bayesian_statistics.py:1185-1213` — `single_host_likelihood` integrands: replace `_interpolated` → `_interpolated_zero_fill` (STAT-03 target)
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py:672-717` — `detection_probability_without_bh_mass_interpolated_zero_fill` (the zero-fill method to use consistently)
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py:719-760` — `detection_probability_without_bh_mass_interpolated` (NN-fill method; no longer called from single_host_likelihood after STAT-03 fix)

### Prior phase context
- `.planning/phases/37-parameter-estimation-correctness/37-CONTEXT.md` — Phase 37 complete; PE-01/PE-02 landed; per-parameter epsilon now in place
- `.planning/phases/36-coordinate-frame-fix/36-VERIFICATION.md` — Phase 36 tests must remain GREEN after Phase 38 lands

### Physics reference
- Gray et al. (2020), arXiv:1908.06050, Eq. 24-25 — canonical catalog likelihood form: L_cat = (1/N)Σ(p_GW(d|z_g, h) / p_det(z_g, h))
- Gray et al. (2020), arXiv:1908.06050, Eq. A.19 — D(h) denominator integral (already correctly implemented with zero-fill)

### Existing tests
- `master_thesis_code_test/test_coordinate_roundtrip.py` — 9 tests; must remain GREEN after Phase 38
- `master_thesis_code_test/test_parameter_space_h.py` — Phase 37 regression; must remain GREEN

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `detection_probability_without_bh_mass_interpolated_zero_fill` already exists (`bayesian_statistics.py:672`); STAT-03 fix just changes which method is called in `single_host_likelihood`
- `simulation_detection_probability.py:_build_grid_1d` builds both NN-fill and zero-fill interpolators; both are accessible via the same object
- The galaxy_likelihoods dict at `bayesian_statistics.py:1084` already collects per-event diagnostics; STAT-04 can add keys there

### Established Patterns
- `[PHYSICS]` commit prefix for STAT-01 and STAT-03 fixes
- `/physics-change` protocol gate required for both fixes
- `np.testing.assert_allclose` for numerical tests; `np.random.default_rng(seed=42)` for reproducibility
- Pre-commit hooks (ruff + mypy) run automatically

### Integration Points
- `BayesianStatistics.evaluate()` at `bayesian_statistics.py:840` → calls the pool that calls `single_host_likelihood` — that's the entrypoint for STAT-03 fix
- `precompute_completion_denominator` at line 70 uses `detection_probability_without_bh_mass_interpolated_zero_fill` — already correct; STAT-03 aligns the numerator to match
- `_LOGGER` at the module level can be used for STAT-04 WARNING logs; the diagnostic per-event return value can be added to the existing `results` list

### Known Interaction: STAT-01 with BH-mass channel
The "without BH mass" L_cat combines results from both `possible_host_galaxies_reduced` and `possible_host_galaxies_with_bh_mass` (lines 930-940). The fix must preserve this combination while changing the aggregation from ΣN_g/ΣD_g to (1/N)Σ(N_g/D_g). N in the denominator should be the total count of contributing galaxies.

### Potential Issue: zero-fill and integration window overlap
When using zero-fill in single_host_likelihood, if the GW event's ±4σ d_L integration window extends beyond the injection grid, the numerator integral will be truncated. The STAT-04 diagnostic must quantify this. If the truncation is large (>5% of integral mass), it indicates the injection grid needs extension (Phase 41 CAMP-01 trigger).

</code_context>

<deferred>
## Deferred Ideas

- **Adaptive integration limits**: only integrate where P_det > 0 — more accurate but complex. Deferred; STAT-04 diagnostic first.
- **BH-mass P_det asymmetry**: STAT-03 fix is for the "without BH mass" 1D P_det. The "with BH mass" 2D P_det (d_L × M) also uses `_interpolated` (NN-fill). If STAT-03 fix only covers 1D, the 2D case is an open item. Check the `with_bh_mass` integrand at `bayesian_statistics.py:1289`.
- **L_cat for BH-mass channel separately**: same ΣN_g/ΣD_g bug in `L_cat_with_bh_mass` at lines 955-957. Must fix both channels in the same STAT-01 commit.
- Full posterior re-evaluation — Phase 40 VERIFY-02

</deferred>

---

*Phase: 38-statistical-correctness*
*Context gathered: 2026-04-22 (inline code audit)*
