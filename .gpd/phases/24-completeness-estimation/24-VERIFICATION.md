---
phase: 24-completeness-estimation
verified: 2026-04-04T15:29:00Z
status: gaps_found
score: 4/5 contract claims verified
consistency_score: 10/11 physics checks passed
independently_confirmed: 8/11 checks independently confirmed
confidence: medium
gaps:
  - subject_kind: claim
    subject_id: claim-fz-correct
    expectation: "f(z=0.029, h=0.73) ~ 0.90, matching Dalya et al. (2022) B-band luminosity completeness"
    expected_check: "Compare digitized data against Dalya et al. (2022) Fig. 2 B-band luminosity completeness"
    status: failed
    category: literature_agreement
    reason: >
      The digitized data gives f(z=0.029) = 0.50 (50% at d_L = 122 Mpc), not ~90%.
      The ~90% figure from Dalya et al. (2022) refers to cumulative B-band LUMINOSITY
      completeness (fraction of total B-band light captured by the brightest galaxies),
      whereas the digitized data appears to represent NUMBER completeness (fraction of
      galaxies found). These are fundamentally different quantities. For the dark siren
      likelihood (Gray et al. 2020), the relevant quantity depends on whether the host
      galaxy weighting is by number or by luminosity. The digitized data has been in the
      codebase since the original commit (0175a55, Aug 2024) and was not introduced by
      the Phase 24 executor.
    computation_evidence: >
      f(z=0.029, h=0.73) = 0.5013 (computed). d_L(z=0.029, h=0.73) = 121.9 Mpc.
      Digitized data at d_L = 122 Mpc gives 50.1%. Dalya et al. (2022) abstract states
      "90% of the total B-band and K-band luminosity up to d_L ~ 130 Mpc." The rapid
      drop from 100% at d=18 Mpc to 65% at d=53 Mpc is characteristic of number
      completeness, not luminosity completeness.
    artifacts:
      - path: master_thesis_code/galaxy_catalogue/glade_completeness.py
        issue: "Digitized data (_COMPLETENESS_PCT) appears to be number completeness, not B-band luminosity completeness as claimed"
    missing:
      - "Verify which curve in Dalya et al. (2022) Fig. 2 was actually digitized"
      - "Determine whether number completeness or luminosity completeness is correct for the dark siren likelihood"
      - "If luminosity completeness is needed, re-digitize the correct curve from Dalya et al. Fig. 2"
    severity: significant
suggested_contract_checks:
  - check: "Cross-check digitized data against the actual Dalya et al. (2022) Fig. 2 by visual comparison"
    reason: "The code's completeness values do not match the paper's stated B-band luminosity completeness, suggesting the wrong curve was digitized"
    suggested_subject_kind: acceptance_test
    suggested_subject_id: "test-fz-dalya-fig2-visual"
    evidence_path: "master_thesis_code/galaxy_catalogue/glade_completeness.py"
comparison_verdicts:
  - subject_kind: claim
    subject_id: claim-fz-correct
    reference_id: ref-dalya2022
    comparison_kind: benchmark
    verdict: tension
    metric: "f(z=0.029) absolute value"
    threshold: "~0.90 (Dalya et al. stated completeness at d_L < 130 Mpc)"
  - subject_kind: claim
    subject_id: claim-dVc
    reference_id: ref-hogg1999
    comparison_kind: benchmark
    verdict: pass
    metric: "relative_error vs astropy FlatLambdaCDM"
    threshold: "<= 0.01"
---

# Phase 24 Verification: Completeness Estimation

**Phase goal:** GLADE+ completeness fraction f(z) is computed from the actual catalog data and available as an interpolatable function.

**Verified:** 2026-04-04T15:29:00Z
**Status:** gaps_found
**Confidence:** MEDIUM

**Overall assessment:** The code correctly implements an interpolatable f(z, h) function
over the digitized data. The comoving volume element is dimensionally correct and matches
astropy to 0.07%. All interface contracts (range, vectorization, h-dependence) are satisfied.
However, the underlying digitized data appears to represent NUMBER completeness rather than
B-band LUMINOSITY completeness. Since Dalya et al. (2022) report 90% luminosity completeness
at d_L < 130 Mpc but the digitized data gives ~50% at that distance, the data provenance
must be clarified before this function is wired into the dark siren likelihood.

## Contract Coverage

| ID | Kind | Status | Confidence | Evidence |
|---|---|---|---|---|
| claim-fz-range | claim | VERIFIED | INDEPENDENTLY CONFIRMED | f(z) in [0.213, 0.497] for z in [0.03, 0.20] at h=0.73 |
| claim-fz-correct | claim | PARTIAL | STRUCTURALLY PRESENT | Interpolation works correctly on data; data provenance unclear |
| claim-h-dependence | claim | VERIFIED | INDEPENDENTLY CONFIRMED | f(z=0.05, h=0.86)=0.453 > f(z=0.05, h=0.60)=0.431 |
| claim-vectorized | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Array and scalar outputs match to 1e-12 rtol |
| claim-dVc | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Matches astropy to 0.07%, z^2 scaling confirmed |

## Required Artifacts

| Artifact | Status | Details |
|---|---|---|
| `master_thesis_code/galaxy_catalogue/glade_completeness.py` | EXISTS, SUBSTANTIVE, INTEGRATED | 319 lines, full implementation with 3 public methods |
| `master_thesis_code/physical_relations.py` (comoving_volume_element) | EXISTS, SUBSTANTIVE, INTEGRATED | 58 lines, properly documented with references |
| `master_thesis_code_test/test_glade_completeness.py` | EXISTS, SUBSTANTIVE, INTEGRATED | 23 tests, all pass |

## Computational Verification Details

### Spot-Check Results (Check 5.2)

Executed code block with actual output:

```python
# Evaluated at key redshifts with h=0.73
gc = GladeCatalogCompleteness()
gc.get_completeness_at_redshift(0.029, h=0.73)  # => 0.5013
gc.get_completeness_at_redshift(0.11, h=0.73)   # => 0.3466
```

| Expression | Test Point | Computed | Expected | Match |
|---|---|---|---|---|
| f(z=0, h=0.73) | z=0 | 1.0 | 1.0 | PASS |
| f(z=0, h=0.60) | z=0 | 1.0 | 1.0 | PASS |
| f(z=0, h=0.86) | z=0 | 1.0 | 1.0 | PASS |
| f(z=0.029) | z=0.029, h=0.73 | 0.5013 | ~0.90 (Dalya luminosity) | FAIL -- see gaps |
| f(z=0.11) | z=0.11, h=0.73 | 0.3466 | < 0.50 | PASS |
| dVc(z=0.01) | z=0.01, h=0.73 | 6.879e6 Mpc^3/sr | 6.941e6 (analytical) | PASS (0.9% off) |
| dVc(z=0) | z=0.0 | 0.0 | 0.0 | PASS |

### Limiting Cases Re-Derived (Check 5.3)

**Limit 1: z -> 0 for f(z)**

At z=0, d_L = 0, so the interpolation looks up d_L=0 in the digitized table, which has
100% completeness. `np.interp(0, distances, completeness, left=100.0)` returns 100.0,
so f = 100/100 = 1.0.

Verified: f(z=0, h) = 1.0 for h = 0.6, 0.73, 0.86. PASS.

**Limit 2: z^2 scaling of dVc/dz at low z**

At z << 1:
- d_com ~ cz/H_0 (Hubble law)
- H(z) ~ H_0 (E(z) ~ 1)
- dVc/dz/dOmega ~ d_com^2 * c / H_0 ~ (c/H_0)^3 * z^2

This predicts dVc(2z) / dVc(z) = 4.0 at low z.

Computed: dVc(0.002) / dVc(0.001) = 3.997. PASS (agrees to 0.08%).

**Limit 3: Dimensional check of dVc at z=0.01**

(c/H_0)^3 * z^2 where c/H_0 = 300000/(73) = 4109.6 Mpc.
Expected: (4109.6)^3 * (0.01)^2 = 6.94e6 Mpc^3/sr.
Computed: 6.88e6 Mpc^3/sr. Ratio = 0.991. PASS (0.9% offset at z=0.01 is from
higher-order cosmological corrections).

### Cross-Check: astropy (Check 5.4)

Independently computed dVc/dz/dOmega using `astropy.cosmology.FlatLambdaCDM(H0=73.0, Om0=0.25)`:

| z | Our value (Mpc^3/sr) | astropy (Mpc^3/sr) | Ratio |
|---|---|---|---|
| 0.01 | 6.879e6 | 6.874e6 | 1.000692 |
| 0.05 | 1.667e8 | 1.666e8 | 1.000692 |
| 0.10 | 6.208e8 | 6.204e8 | 1.000692 |
| 0.20 | 2.074e9 | 2.073e9 | 1.000692 |

The constant ratio of 1.000692 = 300000 / 299792.458 arises because `comoving_volume_element()`
uses `SPEED_OF_LIGHT_KM_S = 300000.0 km/s` (an approximation) in the numerator, while `dist()`
uses `C = 299792458.0 m/s` (exact astropy value) for the luminosity distance. The d_com^2 term
uses the exact c from dist(), but the explicit c/H(z) factor uses the approximate value.

**Severity:** INFO -- 0.07% is negligible for astrophysical purposes. This is a pre-existing
inconsistency in the codebase (Known Bug 8 in CLAUDE.md already notes outdated cosmology
constants).

### h-Dependence Direction (Check 5.6 -- Symmetry/Physics)

Physics: d_L = c(1+z)/H_0 * integral. Higher H_0 (= higher h) => smaller d_L at fixed z =>
catalog is more complete at smaller distances => f(z, h_high) > f(z, h_low).

Verified:
- d_L(z=0.05, h=0.60) = 259.8 Mpc, d_L(z=0.05, h=0.86) = 181.3 Mpc. Correct: d_L decreases with h.
- f(z=0.05, h=0.60) = 0.4314, f(z=0.05, h=0.86) = 0.4534. Correct: f increases with h.

PASS -- INDEPENDENTLY CONFIRMED.

### Monotonicity (Check 5.11 -- Plausibility)

f(z) should be non-increasing (completeness declines with distance). Checked over
z in [0.001, 0.20] with 500 points at h=0.73: zero violations above 0.01 tolerance.
Maximum increase between consecutive points: 0.0013 (from small non-monotonicity in
digitized data around 200-240 Mpc).

PASS -- INDEPENDENTLY CONFIRMED.

## Physics Consistency Summary

| Check | Status | Confidence | Notes |
|---|---|---|---|
| 5.1 Dimensional analysis | CONSISTENT | INDEPENDENTLY CONFIRMED | dVc: [Mpc^3/sr] traced through all terms |
| 5.2 Numerical spot-check | PASS | INDEPENDENTLY CONFIRMED | 7 test points evaluated |
| 5.3 Limiting cases | PASS | INDEPENDENTLY CONFIRMED | z=0, z^2 scaling, dimensional spot-check |
| 5.4 Cross-check (astropy) | PASS | INDEPENDENTLY CONFIRMED | Agrees to 0.07% (constant offset from c approximation) |
| 5.6 Symmetry (h-dependence) | PASS | INDEPENDENTLY CONFIRMED | Direction physically correct |
| 5.7 Conservation | N/A | -- | No time evolution |
| 5.8 Math consistency | PASS | INDEPENDENTLY CONFIRMED | Interpolation, unit conversion, clipping all correct |
| 5.9 Convergence | N/A | -- | Not a numerical solver |
| 5.10 Literature agreement | TENSION | STRUCTURALLY PRESENT | Digitized data does not match Dalya et al. luminosity completeness |
| 5.11 Plausibility | PASS | INDEPENDENTLY CONFIRMED | f in [0,1], monotonically non-increasing within tolerance |
| 5.14 Spectral/analytic | N/A | -- | Not applicable |

**Gate A (catastrophic cancellation):** Not applicable -- no subtraction of large quantities.
**Gate B (analytical-numerical cross-validation):** dVc cross-checked against astropy. PASS.
**Gate C (integration measure):** Not applicable -- no coordinate transformations in user code.
**Gate D (approximation validity):** Flat extrapolation beyond 796 Mpc is a conservative
approximation. Controlling parameter: z_max such that d_L(z_max) = 796 Mpc. At h=0.73,
z_max ~ 0.18. For z > 0.18, completeness is fixed at 21.3%. This is valid for the EMRI
range (z up to ~0.20) but would need refinement for higher redshifts.

## Forbidden Proxy Audit

| Proxy ID | Status | Evidence |
|---|---|---|
| fp-no-raw-catalog | RESPECTED | No GLADE+ catalog I/O; only digitized data arrays |
| fp-no-angular | RESPECTED | No sky-direction dependence |
| fp-no-bayesian-mod | RESPECTED | No changes to bayesian_statistics.py |

## Test Suite

All 23 tests pass (`uv run pytest master_thesis_code_test/test_glade_completeness.py`).
Test coverage for `glade_completeness.py`: 100%.

## Anti-Patterns

| Pattern | File | Severity | Notes |
|---|---|---|---|
| `type: ignore[assignment]` | glade_completeness.py:52-53 | INFO | Mutable default workaround; __post_init__ handles it |
| `SPEED_OF_LIGHT_KM_S = 300000.0` | constants.py:21 | INFO | Known approximation (0.07%), pre-existing |

## Discrepancies Found

| Severity | Location | Evidence | Root Cause | Fix |
|---|---|---|---|---|
| SIGNIFICANT | glade_completeness.py digitized data | f(d_L=122 Mpc) = 50% vs Dalya et al. stated 90% | Wrong completeness type digitized (number vs luminosity) | Verify which Fig. 2 curve; re-digitize if needed |
| INFO | comoving_volume_element vs astropy | 0.07% offset at all z | Two different c values in codebase | Use consistent c; not blocking |

## Confidence Assessment

**Overall: MEDIUM.** The code interface is correct and well-tested (HIGH confidence for
claim-fz-range, claim-h-dependence, claim-vectorized, claim-dVc). However, the underlying
digitized data has a significant tension with the published Dalya et al. (2022) completeness
values, which drops overall confidence to MEDIUM. The data appears to be number completeness
(~50% at 122 Mpc) rather than B-band luminosity completeness (~90% at 130 Mpc). Whether this
matters depends on how the completeness is used in the dark siren likelihood. This question
should be resolved before Phase 25 wires f(z) into the likelihood.

## Gaps Summary

**One gap identified:** The digitized completeness data gives values roughly half of what
Dalya et al. (2022) report for B-band luminosity completeness at the same distance. The most
likely explanation is that different curves from Fig. 2 measure different quantities (number
completeness vs luminosity completeness). The code correctly interpolates whatever data it
has; the question is whether the data represents the right physical quantity for the dark
siren analysis.

**Recommended actions:**
1. Open Dalya et al. (2022) Fig. 2 and identify which curve was digitized
2. Determine which completeness measure (number or luminosity) is appropriate for the
   Gray et al. (2020) dark siren framework
3. If luminosity completeness is needed, re-digitize the correct curve
4. If number completeness is correct, update the plan's expected values

**Sources consulted:**
- [Dalya et al. (2022), arXiv:2110.06184](https://arxiv.org/abs/2110.06184)
- [Gray et al. (2020), arXiv:1908.06050](https://arxiv.org/abs/1908.06050)
- [Dalya et al. (2025), arXiv:2502.14164](https://arxiv.org/abs/2502.14164) -- discusses number vs luminosity completeness distinction
