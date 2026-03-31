---
phase: 14-first-principles-derivation
plan: 02
depth: full
one-liner: "Derived 'with BH mass' dark siren likelihood with explicit Jacobian chain; proved /(1+z) at line 646 is spurious (double-counted after Gaussian rescaling absorbs it)"
subsystem: derivation
tags: [dark-siren, bayesian-inference, gravitational-waves, H0, likelihood, Jacobian, BH-mass, Fisher-matrix]

requires:
  - phase: 14-first-principles-derivation
    plan: 01
    provides: ["Eq. (14.12): d_L-only single-host likelihood", "Sky localization resolved", "Fractional parameterization conventions"]
provides:
  - "Eq. (14.32): 'with BH mass' numerator integrand with NO /(1+z) factor"
  - "Eq. (14.41): complete 'with BH mass' likelihood (boxed, numerator + denominator)"
  - "Jacobian chain proof: M -> M_z_frac Jacobian absorbed by Gaussian rescaling (Eq. 14.21)"
  - "Limiting case: sigma_Mz -> infinity recovers d_L-only (Eq. 14.40)"
  - "Verdict: /(1+z) at line 646 is SPURIOUS"
  - "Code mapping: 12 terms checked, 1 discrepancy (line 646)"
  - "Denominator correct: no changes needed (lines 656-685)"
affects: [15-code-audit, 16-validation]

methods:
  added: [Gaussian-rescaling-identity, Bishop-2006-conditional-decomposition, Gaussian-product-identity]
  patterns: [Jacobian-absorption-in-Gaussian, conditional-marginal-decomposition]

key-files:
  modified:
    - derivations/dark_siren_likelihood.md

key-decisions:
  - "/(1+z) at line 646 is spurious: Jacobian M_z_det/(1+z) is absorbed by Gaussian rescaling identity when transforming galaxy mass prior to M_z_frac coordinates"
  - "Denominator needs no /(1+z): integrates over source-frame (z, M) directly, no coordinate change to M_z_frac"
  - "Methodology asymmetry (quadrature numerator, MC denominator) flagged as numerical concern for Phase 15"

patterns-established:
  - "Jacobian absorption: when a Gaussian prior is transformed via linear change of variables, the Jacobian is absorbed into the Gaussian normalization"
  - "Equation numbering continues from Plan 01: Eq. (14.13)-(14.41)"

conventions:
  - "SI units (c, G, H0 in km/s/Mpc)"
  - "Fractional parameterization: M_z_frac = M*(1+z)/M_z_det"
  - "Bishop (2006) PRML Eq. 2.81-2.82 for conditional decomposition"
  - "Gaussian product identity for analytic marginalization"

plan_contract_ref: ".gpd/phases/14-first-principles-derivation/14-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-mass-extension:
      status: passed
      summary: "Extended d_L-only to 4D with M_z_frac as 4th observable; galaxy mass prior transformed to M_z_frac coordinates via Gaussian rescaling (Eq. 14.21); conditional decomposition and analytic marginalization derived step-by-step"
      linked_ids: [deliv-derivation-ext, test-jacobian, test-dim-analysis-mass, test-limiting-case, ref-bishop, ref-gaussian-product]
      evidence:
        - verifier: self-check
          method: step-by-step Jacobian algebra + Gaussian rescaling identity
          confidence: high
          claim_id: claim-mass-extension
          deliverable_id: deliv-derivation-ext
          acceptance_test_id: test-jacobian
          reference_id: ref-bishop
          evidence_path: "derivations/dark_siren_likelihood.md#section-8"
    claim-jacobian-verdict:
      status: passed
      summary: "DEFINITIVE VERDICT: /(1+z) at line 646 is SPURIOUS. The Jacobian |dM/dM_z_frac| = M_z_det/(1+z) is absorbed by the Gaussian rescaling identity (Eq. 14.20) when transforming p_gal(M) to M_z_frac coordinates. The code double-counts it."
      linked_ids: [deliv-derivation-ext, test-jacobian, test-limiting-case, ref-bishop]
      evidence:
        - verifier: self-check
          method: explicit Jacobian chain (Eqs. 14.15-14.21) + limiting case verification
          confidence: high
          claim_id: claim-jacobian-verdict
          deliverable_id: deliv-derivation-ext
          acceptance_test_id: test-jacobian
          reference_id: ref-bishop
          evidence_path: "derivations/dark_siren_likelihood.md#section-84"
    claim-denominator-consistency:
      status: passed
      summary: "Denominator derived from first principles (Eq. 14.33); integrates p_det * p_gal(z) * p_gal(M) over source-frame (z, M); no /(1+z) needed; no mz_integral needed; code implementation (lines 656-685) is correct"
      linked_ids: [deliv-derivation-ext, test-denom-terms, test-dim-analysis-mass]
      evidence:
        - verifier: self-check
          method: first-principles Bayesian derivation + code mapping
          confidence: high
          claim_id: claim-denominator-consistency
          deliverable_id: deliv-derivation-ext
          acceptance_test_id: test-denom-terms
          evidence_path: "derivations/dark_siren_likelihood.md#section-11"
    claim-limiting-case:
      status: passed
      summary: "sigma_Mz -> infinity limit: conditional variance diverges, mz_integral becomes z-independent, 1/sqrt(2*pi*sigma_cond^2) cancels in numerator/denominator ratio, recovering d_L-only likelihood (Eq. 14.40)"
      linked_ids: [deliv-derivation-ext, test-limiting-case, ref-plan01]
      evidence:
        - verifier: self-check
          method: step-by-step limiting case analysis (Eqs. 14.35-14.40)
          confidence: high
          claim_id: claim-limiting-case
          deliverable_id: deliv-derivation-ext
          acceptance_test_id: test-limiting-case
          reference_id: ref-plan01
          evidence_path: "derivations/dark_siren_likelihood.md#section-12"
  deliverables:
    deliv-derivation-ext:
      status: passed
      path: "derivations/dark_siren_likelihood.md"
      summary: "Sections 7-14 appended: 4D extension, Jacobian chain, conditional decomposition, analytic marginalization, denominator, limiting case, dimensional analysis, code mapping with boxed final expression"
      linked_ids: [claim-mass-extension, claim-jacobian-verdict, claim-denominator-consistency, claim-limiting-case, test-jacobian, test-dim-analysis-mass, test-limiting-case, test-denom-terms]
  acceptance_tests:
    test-jacobian:
      status: passed
      summary: "Complete chain of variable transformations M -> M_z_frac traced in Eqs. 14.15-14.21. Jacobian |dM/dM_z_frac| = M_z_det/(1+z) explicitly shown to cancel with 1/a from Gaussian rescaling identity. Verdict: /(1+z) should NOT appear at line 646."
      linked_ids: [claim-jacobian-verdict, claim-mass-extension, deliv-derivation-ext, ref-bishop]
    test-dim-analysis-mass:
      status: passed
      summary: "Dimensional analysis table in Section 13 covers all new terms. Full 'with BH mass' integrand is dimensionless. Key insight: /(1+z) is dimensionless, so dimensional analysis cannot detect this bug; the verdict rests on the Jacobian algebra."
      linked_ids: [claim-mass-extension, deliv-derivation-ext]
    test-limiting-case:
      status: passed
      summary: "sigma_Mz -> infinity worked step-by-step in Section 12 (Eqs. 14.35-14.40). The 1/sqrt(2*pi*sigma_cond^2) prefactor is H0-independent and cancels in the posterior normalization, leaving the d_L-only expression exactly."
      linked_ids: [claim-limiting-case, deliv-derivation-ext, ref-plan01]
    test-denom-terms:
      status: passed
      summary: "Denominator derived from first principles in Section 11. Term comparison table (Section 11.3) explicitly lists what should and should not appear. Code implementation verified correct at lines 656-685."
      linked_ids: [claim-denominator-consistency, deliv-derivation-ext]
  references:
    ref-bishop:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "Bishop (2006) PRML Eq. 2.81-2.82 used for conditional decomposition (Eqs. 14.23-14.28); code implementation verified to match"
    ref-gaussian-product:
      status: completed
      completed_actions: [compare]
      missing_actions: []
      summary: "Gaussian product identity (Eq. 14.30) verified numerically at 3 test points; applied for analytic M_z marginalization (Eq. 14.31)"
    ref-code-with-bh:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "Code lines 611-692 read and mapped term-by-term in Section 14.1; 12 terms checked, 1 discrepancy found (/(1+z) at line 646)"
    ref-plan01:
      status: completed
      completed_actions: [read]
      missing_actions: []
      summary: "Plan 01 d_L-only baseline (Eq. 14.12) used as limiting case target; verified in Section 12"
  forbidden_proxies:
    fp-skip-jacobian:
      status: rejected
      notes: "Complete Jacobian chain derived step-by-step in Section 8 (Eqs. 14.15-14.21); not skipped"
    fp-handwave-mass:
      status: rejected
      notes: "Jacobian |dM/dM_z_frac| explicitly derived and tracked through Gaussian rescaling; absorption shown algebraically"
    fp-adhoc-denom:
      status: rejected
      notes: "Denominator derived from Bayesian framework in Section 11 (Eq. 14.33); term comparison table provided"
  uncertainty_markers:
    weakest_anchors:
      - "No single canonical reference for the 'with BH mass' extension -- constructed by extending the d_L-only framework"
      - "Gaussian approximation to GW posterior may not capture all features for mass parameters"
    unvalidated_assumptions: []
    competing_explanations:
      - "If the /(1+z) were somehow needed (e.g., if the galaxy mass prior were defined in M_z coordinates instead of M), the Jacobian chain would show it. But the code clearly defines p_gal(M) in source-frame coordinates (line 583: `norm(loc=possible_host.M, scale=possible_host.M_error)`)."
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: claim-jacobian-verdict
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-code-with-bh
    comparison_kind: baseline
    metric: algebraic_derivation
    threshold: "exact match between derived expression and code"
    verdict: fail
    recommended_action: "Remove /(1+z) from bayesian_statistics.py line 646 in Phase 15"
    notes: "Code has spurious /(1+z) factor that the derivation shows should not be present. This is the primary suspected cause of the h=0.600 bias."
  - subject_id: claim-limiting-case
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-plan01
    comparison_kind: baseline
    metric: functional_equivalence
    threshold: "sigma_Mz -> infinity recovers d_L-only up to H0-independent normalization"
    verdict: pass
    recommended_action: "None -- limiting case verified"
    notes: "The 1/sqrt(2*pi*sigma_cond^2) prefactor cancels in the posterior normalization"

duration: 15min
completed: 2026-03-31
---

# Phase 14, Plan 02: "With BH Mass" Likelihood Extension Summary

**Derived "with BH mass" dark siren likelihood with explicit Jacobian chain; proved /(1+z) at line 646 is spurious (double-counted after Gaussian rescaling absorbs it)**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-31T09:19:31Z
- **Completed:** 2026-03-31T09:35:00Z
- **Tasks:** 2
- **Files modified:** 1

## Key Results

- **VERDICT: /(1+z) at line 646 is SPURIOUS** -- the Jacobian |dM/dM_z_frac| = M_z_det/(1+z) is exactly absorbed by the Gaussian rescaling identity when transforming the galaxy mass prior to M_z_frac coordinates. The code double-counts it. [CONFIDENCE: HIGH]
- **Eq. (14.41):** Complete "with BH mass" single-host likelihood with all terms explicit -- no /(1+z) in numerator
- **Denominator is correct** -- integrates p_det * p_gal(z) * p_gal(M) over source-frame (z, M); no coordinate change needed, no /(1+z) [CONFIDENCE: HIGH]
- **Limiting case verified:** sigma_Mz -> infinity recovers d_L-only likelihood (Eq. 14.40) [CONFIDENCE: HIGH]
- **12 code terms mapped:** All correct except the /(1+z) at line 646

## Task Commits

1. **Task 1: Derive "with BH mass" likelihood with explicit Jacobian chain** - `8600350` (derive)
2. **Task 2: Denominator derivation, limiting case, and code mapping** - `caf8ce6` (derive; note: commit also includes unrelated branch cleanup from pre-existing merge conflict state)

## Files Created/Modified

- `derivations/dark_siren_likelihood.md` -- Sections 7-14 appended: 4D extension, Jacobian chain, conditional decomposition, analytic marginalization, denominator, limiting case, dimensional analysis, code mapping

## Next Phase Readiness

- Eq. (14.41) provides the complete reference expression for Phase 15 code audit
- Primary fix identified: remove `/(1+z)` from `bayesian_statistics.py` line 646
- Denominator confirmed correct: no changes needed
- Analytic marginalization confirmed correct: no changes needed
- Methodology asymmetry (quadrature vs MC) flagged as numerical concern for Phase 15

## Contract Coverage

- Claim IDs advanced: claim-mass-extension -> passed, claim-jacobian-verdict -> passed, claim-denominator-consistency -> passed, claim-limiting-case -> passed
- Deliverable IDs produced: deliv-derivation-ext -> derivations/dark_siren_likelihood.md
- Acceptance test IDs run: test-jacobian -> passed, test-dim-analysis-mass -> passed, test-limiting-case -> passed, test-denom-terms -> passed
- Reference IDs surfaced: ref-bishop -> read+compare, ref-gaussian-product -> compare, ref-code-with-bh -> read+compare, ref-plan01 -> read
- Forbidden proxies rejected: fp-skip-jacobian -> rejected, fp-handwave-mass -> rejected, fp-adhoc-denom -> rejected
- Decisive comparison verdicts: claim-jacobian-verdict vs code -> FAIL (code has spurious /(1+z)), claim-limiting-case vs Plan 01 -> PASS

## Equations Derived

**Eq. (14.21):** Jacobian absorption -- the key result

$$p_\text{gal}(M) \, dM = \mathcal{N}(M_{z,\text{frac}};\; \mu_\text{gal,frac},\; \sigma_\text{gal,frac}^2) \, dM_{z,\text{frac}}$$

No leftover Jacobian factor. The (1+z) dependence enters only through $\mu_\text{gal,frac} = M_\text{gal}(1+z)/M_{z,\text{det}}$ and $\sigma_\text{gal,frac} = \sigma_M(1+z)/M_{z,\text{det}}$.

**Eq. (14.32):** "With BH mass" numerator (no /(1+z))

$$\text{Num}_j^{(\text{mass})}(H_0) = \int dz \; p_\text{det} \; p_\text{GW}^{(3D)} \; \text{mz\_integral}(z) \; p_\text{gal}(z)$$

**Eq. (14.41):** Complete "with BH mass" likelihood (boxed in derivation)

## Validations Completed

- Jacobian chain: M -> M_z_frac change of variables traced step-by-step (Eqs. 14.15-14.21)
- Gaussian rescaling identity: derived from definition, verified algebraically (Eq. 14.20)
- Gaussian product identity: verified numerically at 3 test points (Eq. 14.30)
- Conditional decomposition: matches Bishop (2006) PRML Eq. 2.81-2.82 and code (lines 593-605)
- Limiting case: sigma_Mz -> infinity recovers d_L-only (Section 12, Eq. 14.40)
- Dimensional analysis: all new terms documented; integrand dimensionless (Section 13)
- Code mapping: 12 terms checked against derivation (Section 14.1)

## Decisions & Deviations

**Decisions:**
- Chose "Option A" from the plan: transform integration variable to M_z_frac (rather than transforming GW likelihood to source-frame)
- Proved Jacobian absorption via explicit algebra rather than appealing to general transformation rules
- Denominator derived in source-frame coordinates (z, M) rather than transforming to M_z_frac

**Deviations:**
- [Rule 1 - Code Bug] Task 2 commit (`caf8ce6`) inadvertently included unrelated branch cleanup files due to pre-existing merge conflict state (UU files). The derivation content is correct; the commit metadata is polluted. This is a git housekeeping issue, not a physics issue.

## Open Questions

- Will removing /(1+z) from line 646 fully resolve the h=0.600 bias? (Phase 15)
- Is the MC denominator (N=10000 samples) introducing significant noise compared to quadrature numerator? (Phase 15)
- Should the denominator also use quadrature for consistency? (Phase 15 optimization)

## Self-Check: PASSED

- [x] derivations/dark_siren_likelihood.md exists with Sections 7-14
- [x] Commit 8600350 exists (Task 1)
- [x] Commit caf8ce6 exists (Task 2)
- [x] All contract claims, deliverables, acceptance tests, references, and forbidden proxies have entries
- [x] Every decisive comparison has a comparison_verdicts entry

---

_Phase: 14-first-principles-derivation, Plan: 02_
_Completed: 2026-03-31_
