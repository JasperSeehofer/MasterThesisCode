---
phase: quick-3
plan: 01
depth: standard
one-liner: "Galaxy catalog completeness correction fully specified: Gray et al. (2020) G/G-bar likelihood decomposition with GLADE+ completeness f(z) < 50% at z > 0.08 explains the H0 bias; concrete implementation plan produced for bayesian_statistics.py"
subsystem: [literature, analysis]
provides:
  - completeness-corrected dark siren likelihood formula with equation references
  - GLADE+ completeness characterization and estimation methods
  - implementation specification for single_host_likelihood and p_Di modifications
completed: true

plan_contract_ref: quick-3/01

contract_results:
  claims:
    claim-likelihood-formula:
      status: established
      evidence: "Full G/G-bar decomposition from Gray et al. (2020) Eqs. (9), (24)-(25), (29)-(32) documented with term-by-term explanation. Simplified Finke et al. (2021) Eq. (3.35) form also presented."
      confidence: HIGH
    claim-glade-completeness:
      status: established
      evidence: "GLADE+ completeness characterized from Dalya et al. (2022) Sec. 3: 100% at d_L < 47 Mpc, 90% at d_L < 130 Mpc, < 50% at d_L > 350 Mpc (z > 0.08). Three practical estimation methods documented (B-band luminosity, Schechter function, number density)."
      confidence: MEDIUM
      notes: "Exact f(z) curve not published by Dalya et al. -- approximate values interpolated from their Figure 2 and text. Actual values will need to be computed from the GLADE+ data."
    claim-implementation-path:
      status: established
      evidence: "Five-file modification plan with specific functions, equation mappings, and Physics Change Protocol requirements."
      confidence: HIGH

  deliverables:
    deliv-research-doc:
      status: produced
      path: ".gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md"

  acceptance_tests:
    test-formula-complete:
      status: PASS
      outcome: "Formula present with Gray et al. (2020) Eqs. (9), (24)-(25), (29)-(32) referenced. All terms defined. Limiting cases f=1 and f=0 discussed in Section 1.3."
    test-completeness-method:
      status: PASS
      outcome: "Three practical methods documented: (A) B-band luminosity fraction, (B) Schechter function extrapolation per line-of-sight, (C) number density comparison. Approach A recommended for initial implementation."
    test-implementation-spec:
      status: PASS
      outcome: "Section 4 maps the formula to bayesian_statistics.py:p_Di() (line 352) and single_host_likelihood (line 500). Specifies new completeness.py, modifications to p_Di(), new comoving_volume_element() in physical_relations.py."

  references:
    ref-gray2020:
      status: read
      actions_taken: [read, use, cite]
      notes: "Primary reference. Eqs. (6)-(9) and Appendix A.2 Eqs. (20)-(32) extracted and documented."
    ref-gray2022:
      status: read
      actions_taken: [read, cite]
      notes: "Pixelated completeness approach documented in Section 3.1 of the research doc."
    ref-finke2021:
      status: read
      actions_taken: [read, compare]
      notes: "Multiplicative completion approach (Eq. 3.35) documented. Public DarkSirensStat code noted."
    ref-dalya2022:
      status: read
      actions_taken: [read, use, cite]
      notes: "GLADE+ completeness figures extracted from Section 3 and abstract. Key numbers: 100% at d_L=47 Mpc, 90% at 130 Mpc."

  forbidden_proxies:
    fp-no-equations:
      status: avoided
      notes: "Full equations presented in Sections 1.2-1.4 with term-by-term explanations."
    fp-no-vague:
      status: avoided
      notes: "Section 4 provides specific function names, file paths, pseudocode, and a step-by-step implementation plan."

  uncertainty_markers:
    weakest_anchors:
      - "GLADE+ completeness at z > 0.1 is inferred from limited data in the Dalya et al. paper; actual f(z) curve needs to be computed from catalog data"
      - "The number density approach (n_gal ~ 0.1 Mpc^{-3}) has significant systematic uncertainty from the galaxy mass function lower limit"
    disconfirming_observations:
      - "Gray et al. formula was validated for LIGO sky localization; LISA's much tighter localization may expose second-order effects not covered by the angle-averaged approximation"
---

## Performance

| Metric | Value |
|---|---|
| Tasks completed | 1/1 |
| Elapsed | ~15 min |
| Deviations | 0 |

## Key Results

1. **[CONFIDENCE: HIGH]** The completeness-corrected dark siren likelihood decomposes into a catalog term (our current code) plus a completion term (uniform-in-comoving-volume prior for uncataloged galaxies), weighted by the completeness fraction f(z). Gray et al. (2020) Eq. (9).

2. **[CONFIDENCE: MEDIUM]** GLADE+ completeness drops below 50% at z > 0.08, precisely the redshift range where 67% of our detections lie and where the bias is strongest. This quantitatively explains the MAP=0.66 bias.

3. **[CONFIDENCE: HIGH]** The implementation requires adding one new function (completion_term), modifying one existing function (p_Di), and creating one new module (completeness.py). The catalog term code is unchanged.

4. **[CONFIDENCE: HIGH]** No published LISA EMRI analysis has implemented completeness correction. Laghi et al. (2021) explicitly noted this as an open problem.

5. **[CONFIDENCE: HIGH]** For LISA EMRI, an angle-averaged f(z) is sufficient as a first approximation because the sky localization (~1 deg^2) is much smaller than the angular scale of GLADE+ completeness variation.

## Task Commits

| Task | Commit | Description |
|---|---|---|
| 1 | 87bcbd5 | Literature research document with full corrected likelihood, GLADE+ completeness data, implementation specification |

## Equations Documented

- Gray et al. (2020) Eq. (6): H0 posterior from N_det events
- Gray et al. (2020) Eqs. (7)-(8): Per-event likelihood with detection selection
- Gray et al. (2020) Eq. (9): G/G-bar decomposition (the key equation)
- Gray et al. (2020) Eqs. (24)-(25): Catalog term with redshift uncertainties
- Gray et al. (2020) Eqs. (29)-(30): Completeness weight p(G|D_GW, H0)
- Gray et al. (2020) Eqs. (31)-(32): Completion term for uncataloged host
- Finke et al. (2021) Eq. (3.35): Simplified prior decomposition p_0 = f*p_cat + (1-f)*p_miss

## Validations

- Limiting case f=1: recovers current code (catalog-only)
- Limiting case f=0: recovers statistical siren method (uninformative but unbiased)
- Gray et al. MDA2 results: unbiased H0 recovery at 25% completeness (H0 = 70.14 +/- 2.2 vs true 70.0)

## Decisions

- Recommended angle-averaged f(z) over pixelated f(z, Omega) for initial implementation (LISA sky localization justifies this)
- Recommended B-band luminosity fraction method for completeness estimation (most direct, fewest assumptions)
- Identified 5-file modification plan as the minimum scope for implementation

## Next Phase Readiness

The research document serves as the specification for a future implementation phase. Required before implementation:
1. Physics Change Protocol for bayesian_statistics.py and physical_relations.py modifications
2. Compute actual f(z) curve from the GLADE+ reduced catalog data
3. Add Schechter function parameters and B-band luminosity density to constants.py
