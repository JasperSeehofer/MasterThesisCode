---
phase: 25-likelihood-correction
verified: 2026-04-04T18:00:00Z
status: passed
score: 4/4 contract claims verified
consistency_score: 10/10 physics checks passed
independently_confirmed: 7/10 checks independently confirmed
confidence: high
comparison_verdicts:
  - subject_kind: claim
    subject_id: claim-gray-eq9
    reference_id: ref-gray2020
    comparison_kind: formula_match
    verdict: pass
    metric: "structural match to Gray et al. (2020) Eq. 9"
    threshold: "exact structural agreement"
  - subject_kind: claim
    subject_id: claim-completion-term
    reference_id: ref-gray2020
    comparison_kind: formula_match
    verdict: pass
    metric: "structural match to Gray et al. (2020) Eqs. 31-32"
    threshold: "exact structural agreement"
suggested_contract_checks: []
---

# Phase 25 Verification: Likelihood Correction

**Phase goal:** The dark siren likelihood combines catalog and completion terms weighted by f(z), implementing Gray et al. (2020) Eq. 9

**Verified:** 2026-04-04 | **Status:** PASSED | **Confidence:** HIGH

## Contract Coverage

| ID | Kind | Status | Confidence | Evidence |
|----|------|--------|------------|---------|
| claim-gray-eq9 | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Code lines 600-603 implement `f_i * L_cat + (1-f_i) * L_comp`; spot-checks pass for f=0, f=0.5, f=1 limits |
| claim-completion-term | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Code lines 531-578 implement ratio of integrals matching Eqs. 31-32; positivity and dimensionless checks pass |
| claim-completeness-threading | claim | VERIFIED | INDEPENDENTLY CONFIRMED | `GladeCatalogCompleteness` instantiated in `evaluate()` (line 227), threaded to `p_D()` (line 255), to `p_Di()` (line 362); `get_completeness_at_redshift(z_det, self.h)` called at line 508 |
| claim-reference-comments | claim | VERIFIED | INDEPENDENTLY CONFIRMED | 7 reference comments citing arXiv:1908.06050 in bayesian_statistics.py; 1 in physical_relations.py |

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| `bayesian_inference/bayesian_statistics.py` | PRESENT, SUBSTANTIVE | Contains GladeCatalogCompleteness (4 occurrences), L_comp (9), completion_numerator/denominator_integrand, get_completeness_at_redshift (1). Note: `completion_term` as an exact string is absent; the concept is implemented via `completion_numerator_integrand` / `completion_denominator_integrand` function names. |
| `physical_relations.py` | PRESENT, SUBSTANTIVE | Contains "Gray et al." and "1908.06050" in comoving_volume_element docstring (line 426) |
| `test_completion_term.py` | PRESENT, SUBSTANTIVE | 11 tests, all passing. Contains `test_f1_recovers_catalog_only` (2 occurrences), `test_f0_gives_completion_only` (1). Note: `test_combination_formula` exact string absent; implemented as `TestCombinationFormulaWeightedSum` class with 3 tests. |

## Computational Verification Details

### Spot-Check Results (5.2) -- INDEPENDENTLY CONFIRMED

```
Test 1 (f=1): combined=3.140000e-05, L_cat=3.140000e-05 -- PASS
Test 2 (f=0): combined=1.230000e-04, L_comp=1.230000e-04 -- PASS
Test 3 (f=0.5): combined=4.000000e-04, expected=4.000000e-04 -- PASS
```

All three test parameter sets produce exact agreement (machine precision).

### Limiting Cases Re-Derived (5.3) -- INDEPENDENTLY CONFIRMED

**f=1 limit (complete catalog):**
1. Start with combination formula: `p_i = f_i * L_cat + (1-f_i) * L_comp`
2. Set f_i = 1: `p_i = 1 * L_cat + 0 * L_comp = L_cat`
3. Completion term vanishes entirely, recovering catalog-only behavior
4. Verified numerically: combined = L_cat to machine precision

**f=0 limit (empty catalog):**
1. Set f_i = 0: `p_i = 0 * L_cat + 1 * L_comp = L_comp`
2. Catalog term vanishes, using completion term only
3. Verified numerically: combined = L_comp to machine precision
4. L_comp > 0 (positive integrand) and finite -- confirmed

**Low-z limit of comoving volume element:**
1. At z << 1: d_com ~ cz/H0, H(z) ~ H0
2. Therefore dVc/dz/dOmega ~ (c/H0)^3 * z^2
3. At z=0.1, h=0.73: exact = 6.41e+08, approx = 6.93e+08, ratio = 0.925
4. Agreement within 8% at z=0.1 (expected deviation from higher-order terms)

### Dimensional Analysis (5.1) -- INDEPENDENTLY CONFIRMED

**L_comp is dimensionless:**
- Numerator: integral of `p_GW * P_det * dVc/dz dz`
  - [p_GW] = [1/Gpc^3] (3D Gaussian in fractional coordinates, but dimensionless since d_L_frac is dimensionless)
  - [P_det] = dimensionless (probability)
  - [dVc/dz] = [Mpc^3/sr]
  - [dz] = dimensionless
  - Numerator units: [Mpc^3/sr] (up to dimensionless factors)
- Denominator: integral of `P_det * dVc/dz dz`
  - Same units: [Mpc^3/sr]
- L_comp = numerator / denominator: dimensionless

**Scale invariance test:** Multiplying dVc by constant factors (1.0, 100.0, 0.01) leaves L_comp unchanged:
```
L_comp(scale=1)=1.91770011e+03, L_comp(scale=100)=1.91770011e+03
ratio = 1.000000000000000 -- PASS
```

**Comoving volume element:** `[Mpc]^2 * [km/s] / [km/s/Mpc] = [Mpc]^3/sr` -- consistent.

### Completeness Threading (5.6 -- Symmetry/Consistency) -- INDEPENDENTLY CONFIRMED

```
f(z=0.1, h=0.6) = 0.3214
f(z=0.1, h=0.86) = 0.4037
```

Higher h gives smaller d_L at the same z, which maps to a closer (more complete) catalog region: f_high > f_low. Direction is physically correct. Both values in [0, 1].

### Cross-Check: Completion Term Positivity (5.11) -- INDEPENDENTLY CONFIRMED

With P_det=1 (constant):
```
numerator = 1.314224e+11
denominator = 6.853124e+07
L_comp = 1.917700e+03 > 0 -- PASS
```

Both integrands are products of non-negative quantities (Gaussian PDF, volume element, detection probability), so L_comp = ratio of positive integrals is guaranteed positive.

### Test Suite Execution (5.4) -- INDEPENDENTLY CONFIRMED

All 11 tests pass:
```
test_f1_recovers_catalog_only_without_bh_mass PASSED
test_f1_recovers_catalog_only_with_bh_mass PASSED
test_f0_gives_completion_only PASSED
test_f0_with_zero_L_cat_still_returns_L_comp PASSED
test_f05_gives_average PASSED
test_f03_gives_weighted_sum PASSED
test_combination_always_between_components PASSED
test_completion_term_positive_with_constant_pdet PASSED
test_lcomp_invariant_under_dvc_scaling PASSED
test_f_varies_with_h PASSED
test_completion_integrand_uses_detection_sky_position PASSED
```

## Physics Consistency Summary

| Check | Status | Confidence | Notes |
|-------|--------|------------|-------|
| 5.1 Dimensional analysis | CONSISTENT | INDEPENDENTLY CONFIRMED | L_comp dimensionless (scale invariance verified numerically); dVc units [Mpc^3/sr] traced |
| 5.2 Numerical spot-check | PASS | INDEPENDENTLY CONFIRMED | f=0, f=0.5, f=1 all match hand computation |
| 5.3 Limiting cases | PASS | INDEPENDENTLY CONFIRMED | f=1 recovers L_cat; f=0 recovers L_comp; low-z volume element ~ z^2 |
| 5.4 Cross-check | PASS | INDEPENDENTLY CONFIRMED | Test suite (11 tests) exercises combination formula, positivity, dimensionlessness, sky position |
| 5.6 Symmetry/Consistency | PASS | INDEPENDENTLY CONFIRMED | f_i varies correctly with h (direction verified) |
| 5.7 Conservation | N/A | - | Not applicable to likelihood computation |
| 5.8 Math consistency | PASS | STRUCTURALLY PRESENT | Code structure matches Gray Eqs. 9, 31, 32; integrand terms verified |
| 5.10 Literature agreement | PASS | INDEPENDENTLY CONFIRMED | Formula structure matches Gray et al. (2020) arXiv:1908.06050 Eqs. 9, 24-25, 31-32 |
| 5.11 Plausibility | PASS | INDEPENDENTLY CONFIRMED | L_comp > 0, finite; combination interpolates between L_cat and L_comp |
| Gate C Integration measure | PASS | STRUCTURALLY PRESENT | comoving_volume_element uses d_com = d_L/(1+z), verified at z=0.1 |

## Forbidden Proxy Audit

| Proxy ID | Status | Evidence |
|----------|--------|---------|
| fp-no-single-host-mod | REJECTED | `single_host_likelihood()` does not appear in git diff; no completion/completeness terms in that function |
| fp-no-galaxy-f-weight | REJECTED | f_i weighting occurs only in combination formula (lines 600-603), not inside galaxy sum |
| fp-no-worker-completion | REJECTED | Completion term computed in `p_Di()` (lines 500-578), not in worker processes |
| fp-no-hardcoded-f | REJECTED | `GladeCatalogCompleteness.get_completeness_at_redshift(z_det, self.h)` used at line 508; no hardcoded values |
| fp-no-catalog-term-change | REJECTED | Catalog term computation (lines 460-498) structurally unchanged from pre-phase code |

## Anti-Patterns Found

| Pattern | Severity | Location | Notes |
|---------|----------|----------|-------|
| `allow_singular=True` TODO | INFO | bayesian_statistics.py:212,217 | Pre-existing (not from this phase), related to Gaussian construction |
| `must_contain` naming mismatch | MINOR | Contract specifies `completion_term` and `test_combination_formula` as exact strings; actual names use `completion_numerator_integrand` and `TestCombinationFormulaWeightedSum` | Concepts fully implemented; only exact string match fails |

## Confidence Assessment

**HIGH confidence.** The implementation correctly matches Gray et al. (2020) Eq. 9 (combination formula) and Eqs. 31-32 (completion term). This is confirmed by:

1. **7 independently confirmed computational checks** spanning spot-checks, limiting cases, dimensional analysis, scale invariance, positivity, and completeness threading
2. **11 passing tests** covering the combination formula, limiting cases, positivity, dimensionlessness, h-dependence, and sky position correctness
3. **All 5 forbidden proxies rejected** -- the implementation is structurally correct (completion in p_Di, not in workers; catalog term unchanged; no hardcoded completeness)
4. **7 reference comments** citing arXiv:1908.06050 with specific equation numbers

The only gap is cosmetic: two `must_contain` exact-string matches fail because the code uses slightly different naming conventions. The physics content is fully present.
