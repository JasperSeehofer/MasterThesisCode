---
phase: 14-first-principles-derivation
plan: 01
depth: full
one-liner: "Derived d_L-only dark siren likelihood term-by-term from Bayes' theorem, verified dimensionally and against Gray et al. (2020), resolving sky localization weight placement"
subsystem: derivation
tags: [dark-siren, bayesian-inference, gravitational-waves, H0, likelihood, Fisher-matrix]

requires: []
provides:
  - "Eq. (14.12): d_L-only single-host likelihood with all terms explicit"
  - "Sky localization resolved: inside 3D GW Gaussian, not separate factor"
  - "Dimensional analysis: entire integrand dimensionless"
  - "Code mapping: every term traced to bayesian_statistics.py lines 557-609"
  - "Gray et al. (2020) comparison: structurally consistent"
  - "Three explicit Plan 02 handoff questions (/(1+z), denominator, analytic marginalization)"
affects: [14-02-with-bh-mass-extension, 15-code-audit]

methods:
  added: [Bayesian-inference, Fisher-matrix-Gaussian-approximation, dark-siren-method]
  patterns: [fractional-parameterization, per-galaxy-denominator]

key-files:
  created:
    - derivations/dark_siren_likelihood.md

key-decisions:
  - "Sky localization weight is correctly inside the 3D GW Gaussian -- TODO at line 556 is resolved as non-issue"
  - "No dd_L/dz Jacobian in d_L-only channel because d_L_frac is a functional evaluation, not a change of integration variable"
  - "Per-galaxy denominator is the correct decomposition of the Gray et al. beta(H0) selection correction"

patterns-established:
  - "Fractional parameterization: d_L_frac = d_L(z,H0)/d_L_meas makes covariance dimensionless"
  - "Equation numbering: Eq. (14.N) for Phase 14 cross-referencing"

conventions:
  - "SI units (c, G, H0 in km/s/Mpc)"
  - "d_L in Gpc"
  - "Angles in radians"
  - "z dimensionless"
  - "Gray et al. (2020) notation as baseline"

plan_contract_ref: ".gpd/phases/14-first-principles-derivation/14-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-dl-likelihood:
      status: passed
      summary: "d_L-only dark siren likelihood derived term-by-term (Eq. 14.3, 14.12) with GW likelihood, galaxy prior, p_det, and volume element all explicit"
      linked_ids: [deliv-derivation, test-dim-analysis, test-gray-match, test-sky-weight, ref-gray, ref-schutz, ref-chen]
      evidence:
        - verifier: self-check
          method: code-mapping + dimensional analysis + literature comparison
          confidence: high
          claim_id: claim-dl-likelihood
          deliverable_id: deliv-derivation
          acceptance_test_id: test-gray-match
          reference_id: ref-gray
          evidence_path: "derivations/dark_siren_likelihood.md"
    claim-sky-weight:
      status: passed
      summary: "Sky localization weight is inside the 3D GW Gaussian (Section 2.7); appears in exactly one factor (numerator GW likelihood); not in denominator"
      linked_ids: [deliv-derivation, test-sky-weight, ref-gray]
      evidence:
        - verifier: self-check
          method: Fisher matrix structure analysis
          confidence: high
          claim_id: claim-sky-weight
          deliverable_id: deliv-derivation
          acceptance_test_id: test-sky-weight
          reference_id: ref-gray
          evidence_path: "derivations/dark_siren_likelihood.md#section-27"
    claim-dimensions:
      status: passed
      summary: "Every factor documented with dimensions; full integrand is dimensionless (Section 4, Appendix A)"
      linked_ids: [deliv-derivation, test-dim-analysis]
      evidence:
        - verifier: self-check
          method: dimensional analysis table + code path trace
          confidence: high
          claim_id: claim-dimensions
          deliverable_id: deliv-derivation
          acceptance_test_id: test-dim-analysis
          evidence_path: "derivations/dark_siren_likelihood.md#section-4"
  deliverables:
    deliv-derivation:
      status: passed
      path: "derivations/dark_siren_likelihood.md"
      summary: "6-section derivation document covering Bayesian framework, term-by-term likelihood, code mapping, dimensional analysis, Gray et al. comparison, and baseline expression"
      linked_ids: [claim-dl-likelihood, claim-sky-weight, claim-dimensions, test-dim-analysis, test-gray-match, test-sky-weight]
  acceptance_tests:
    test-dim-analysis:
      status: passed
      summary: "All factors have documented dimensions in Section 4.1 table; full integrand times dz is dimensionless; independent code-path cross-check in Appendix A confirms"
      linked_ids: [claim-dimensions, deliv-derivation]
    test-gray-match:
      status: passed
      summary: "Term-by-term correspondence with Gray et al. (2020) Eq. (2)-(6) documented in Section 5.1; structural differences (redshift integration, Fisher marginalization, per-galaxy denominator) identified and justified in Section 5.2"
      linked_ids: [claim-dl-likelihood, deliv-derivation, ref-gray]
    test-sky-weight:
      status: passed
      summary: "Sky localization traced through likelihood decomposition in Section 2.7; appears exactly once (3D GW Gaussian in numerator); absent from denominator (GW likelihood integrates out)"
      linked_ids: [claim-sky-weight, deliv-derivation]
  references:
    ref-gray:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "Gray et al. (2020) Eq. (2)-(6) compared term-by-term in Section 5; structural consistency confirmed"
    ref-schutz:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Schutz (1986) cited in Section 1.1 as foundational dark siren proposal"
    ref-chen:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "Chen et al. (2018) galaxy catalog framework referenced in Sections 1.2 and 5.3; treatment of selection effects consistent"
    ref-code-without-bh:
      status: completed
      completed_actions: [read]
      missing_actions: []
      summary: "Code lines 557-609 read and mapped term-by-term in Section 3; full variable correspondence table provided"
  forbidden_proxies:
    fp-handwave-volume:
      status: rejected
      notes: "Volume element explicitly addressed in Section 2.4 and 4.2; the absence of dd_L/dz is derived from the fractional parameterization, not handwaved"
    fp-skip-sky:
      status: rejected
      notes: "Sky weight placement derived from Fisher matrix structure in Section 2.7; TODO at line 556 explicitly resolved"
  uncertainty_markers:
    weakest_anchors:
      - "Galaxy catalog completeness assumption (noted in plan, not verifiable from derivation)"
      - "Gaussian approximation to GW posterior (valid at SNR >= 20)"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations: []

duration: 15min
completed: 2026-03-31
---

# Phase 14, Plan 01: d_L-Only Dark Siren Likelihood Summary

**Derived d_L-only dark siren likelihood term-by-term from Bayes' theorem, verified dimensionally and against Gray et al. (2020), resolving sky localization weight placement**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-31T09:04:00Z
- **Completed:** 2026-03-31T09:17:00Z
- **Tasks:** 2
- **Files created:** 1

## Key Results

- **Eq. (14.12):** Complete d_L-only single-host likelihood with every factor (GW likelihood, galaxy prior, p_det) explicit and mapped to code
- **Sky localization resolved:** The weight is correctly inside the 3D GW Gaussian -- the TODO at line 556 is a non-issue [CONFIDENCE: HIGH]
- **Dimensional consistency:** Entire integrand is dimensionless; no stray Jacobians in d_L-only channel [CONFIDENCE: HIGH]
- **Three Plan 02 handoff questions:** (1) Is /(1+z) at line 679 correct? (2) Is the "with BH mass" denominator consistent? (3) Does analytic marginalization introduce Jacobian issues?

## Task Commits

1. **Task 1: Derive d_L-only dark siren likelihood from Bayesian framework** - `c466e86` (derive)
2. **Task 2: Dimensional analysis and literature comparison** - `e0511fa` (derive)

## Files Created/Modified

- `derivations/dark_siren_likelihood.md` -- 6-section derivation: Bayesian framework, term-by-term likelihood, code mapping, dimensional analysis, Gray et al. comparison, baseline expression

## Next Phase Readiness

- Eq. (14.12) provides the clean d_L-only baseline for Plan 02's "with BH mass" extension
- Three explicit questions for Plan 02 are documented at the end of Section 6
- Code mapping (Section 3) provides the bridge for Phase 15's code audit
- Sky localization TODO at line 556 can be marked resolved

## Contract Coverage

- Claim IDs advanced: claim-dl-likelihood -> passed, claim-sky-weight -> passed, claim-dimensions -> passed
- Deliverable IDs produced: deliv-derivation -> derivations/dark_siren_likelihood.md
- Acceptance test IDs run: test-dim-analysis -> passed, test-gray-match -> passed, test-sky-weight -> passed
- Reference IDs surfaced: ref-gray -> read+compare, ref-schutz -> cite, ref-chen -> read+compare, ref-code-without-bh -> read
- Forbidden proxies rejected: fp-handwave-volume -> rejected, fp-skip-sky -> rejected

## Equations Derived

**Eq. (14.1):** Posterior for H0

$$p(H_0 \mid \{d_\text{GW}^i\}, \text{catalog}) \propto p(H_0) \prod_{i=1}^{N_\text{det}} p(d_\text{GW}^i \mid H_0, \text{catalog})$$

**Eq. (14.3):** Single-host likelihood structure (numerator/denominator)

$$\mathcal{L}_j(H_0) = \frac{\int dz \; p_\text{det}(z) \; p_\text{GW}(\phi_j, \theta_j, d_{L,\text{frac}}(z, H_0)) \; p_\text{gal}(z)}{\int dz \; p_\text{det}(z) \; p_\text{gal}(z)}$$

**Eq. (14.12):** Complete d_L-only baseline (boxed in derivation)

## Validations Completed

- Dimensional analysis: all 8 factors verified dimensionless (Section 4.1 table)
- Code mapping: numerator and denominator match `bayesian_statistics.py` lines 557-609 (Section 3)
- Literature comparison: term-by-term with Gray et al. (2020) Eq. (2)-(6), structurally consistent (Section 5)
- Sky localization: appears exactly once (3D GW Gaussian), not in denominator (Section 2.7)
- Independent code-path dimensional cross-check (Appendix A)

## Decisions & Deviations

**Decisions:**
- Sky localization weight placement resolved as correct (inside GW Gaussian) based on Fisher matrix structure
- No dd_L/dz Jacobian needed because d_L_frac is a functional evaluation, not a change of integration variable
- Sections 1-6 written as a single coherent document rather than split across tasks (no loss of content)

**Deviations:** None -- plan executed as specified.

## Open Questions

- Is the `/(1+z)` at line 679 in the "with BH mass" numerator correct? (Deferred to Plan 02)
- Is the "with BH mass" denominator (MC sampling over M and z) consistent with the numerator (analytic M_z marginalization)? (Deferred to Plan 02)

## Self-Check: PASSED

- [x] derivations/dark_siren_likelihood.md exists
- [x] Commit c466e86 exists (Task 1)
- [x] Commit e0511fa exists (Task 2)
- [x] All contract claims, deliverables, acceptance tests, references, and forbidden proxies have entries

---

_Phase: 14-first-principles-derivation, Plan: 01_
_Completed: 2026-03-31_
