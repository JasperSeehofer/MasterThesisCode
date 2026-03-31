# Roadmap: "With BH Mass" Likelihood Bias Audit (v1.2.1)

## Overview

This milestone audits the "with BH mass" dark siren likelihood channel that shows a systematic H0 posterior bias (h=0.600 vs h=0.678 for the "without BH mass" channel). The work proceeds from first-principles derivation of the correct likelihood, through code audit and correction, to numerical validation that both channels converge.

## Contract Overview

| Contract Item | Advanced By Phase(s) | Status |
| --- | --- | --- |
| First-principles derivation of "with BH mass" likelihood | Phase 14 | Planned |
| Code audit showing term-by-term match | Phase 15 | Planned |
| Posterior convergence between channels (within ~0.01) | Phase 16 | Planned |
| Verdict on /(1+z) Jacobian (line 679) | Phase 15 | Planned |
| Sky localization weight placement audit | Phase 15 | Planned |
| Numerator/denominator consistency audit | Phase 15 | Planned |

## Phases

**Phase Numbering:** Continues from v1.2 (GSD-tracked, Phases 9-13). v1.2.1 starts at Phase 14.

- [x] **Phase 14: First-Principles Derivation** - Derive correct d_L-only and "with BH mass" dark siren likelihoods from literature
- [ ] **Phase 15: Code Audit & Fix** - Audit bayesian_statistics.py against derivation, implement all corrections
- [ ] **Phase 16: Validation** - Re-run evaluation on 22-detection dataset, confirm channel convergence

## Phase Dependencies

| Phase | Depends On | Enables | Critical Path? |
|-------|-----------|---------|:-:|
| 14 - First-Principles Derivation | -- | 15 | Yes |
| 15 - Code Audit & Fix | 14 | 16 | Yes |
| 16 - Validation | 15 | -- | Yes |

**Critical path:** 14 -> 15 -> 16 (strictly sequential, no parallelism)

## Phase Details

### Phase 14: First-Principles Derivation

**Goal:** The correct "with BH mass" dark siren likelihood is derived from first principles, with all terms, Jacobians, and volume elements made explicit
**Depends on:** Nothing (entry point)
**Requirements:** DERV-01, DERV-02
**Contract Coverage:**
- Advances: First-principles derivation of "with BH mass" likelihood
- Deliverables: Written derivation document covering both d_L-only and M_z-extended formulations
- Anchor coverage: Gray et al. (2020) for d_L-only baseline; Schutz (1986) for dark siren framework; Bishop (2006) Eq. 2.81-2.82 for Gaussian conditioning; Chen et al. (2018) for galaxy catalog method
- Forbidden proxies: Derivation that skips Jacobians or hand-waves the M -> M_z_frac transformation

**Success Criteria** (what must be TRUE when this phase completes):

1. The d_L-only dark siren likelihood is written down term-by-term (GW likelihood, galaxy prior, detection probability, volume element, sky localization weight), matching Gray et al. (2020) notation
2. The extension to include M_z as 4th observable is derived with explicit Jacobian |dM/dM_z| = 1/(1+z), the conditional Gaussian decomposition (Bishop 2006 Eq. 2.81-2.82), and the analytic mass marginalization via Gaussian product identity
3. Every term has documented dimensions (the likelihood integrand is dimensionless after proper normalization)
4. The derivation reduces to the d_L-only case when M_z measurement uncertainty -> infinity (no mass information limit)
5. The sky localization weight placement is unambiguous: it appears in exactly one factor of the likelihood decomposition

**Plans:** 2 plans

Plans:
- [ ] 14-01-PLAN.md — Derive d_L-only dark siren likelihood (Bayesian framework, dimensional analysis, Gray et al. comparison)
- [ ] 14-02-PLAN.md — Extend to "with BH mass" (Jacobian chain, conditional decomposition, /(1+z) verdict, denominator, limiting case)

### Phase 15: Code Audit & Fix

**Goal:** Every term in bayesian_statistics.py matches the Phase 14 derivation, with all identified discrepancies corrected
**Depends on:** Phase 14 (derivation provides the reference formula)
**Requirements:** AUDT-01, AUDT-02, AUDT-03, FIX-01
**Contract Coverage:**
- Advances: Code audit showing term-by-term match; verdict on /(1+z) Jacobian; sky weight placement audit; numerator/denominator consistency
- Deliverables: Corrected bayesian_statistics.py with reference comments linking each term to the derivation
- Anchor coverage: Phase 14 derivation document; commit 15b49a3 (analytic M_z marginalization); TODO flags at lines 556, 755; current /(1+z) at line 679
- Forbidden proxies: "Looks wrong, remove it" without derivation backing; ad-hoc numerical fix without term-by-term justification

**Success Criteria** (what must be TRUE when this phase completes):

1. The /(1+z) factor at line 679 has a definitive verdict (correct, spurious, or needs modification) backed by the Phase 14 derivation, with the corresponding code change (or explicit retention) documented
2. Sky localization weight (phi, theta) is verified to appear in exactly one factor of the likelihood -- the TODO flags at lines 556 and 755 are resolved
3. The "with BH mass" denominator (MC-sampled, lines 689-722) uses the same Jacobians, weights, and mass terms as the numerator
4. Every formula change follows physics change protocol: old formula, new formula, reference, dimensional analysis, limiting case
5. Reference comments are added above every modified line linking to the derivation

**Plans:** TBD

### Phase 16: Validation

**Goal:** Numerical evidence confirms both likelihood channels produce consistent H0 posteriors on the 22-detection dataset
**Depends on:** Phase 15 (corrected code)
**Requirements:** VALD-01, VALD-02
**Contract Coverage:**
- Advances: Posterior convergence between channels
- Deliverables: Evaluation output showing both channel posterior peaks and their difference
- Anchor coverage: h=0.600 baseline ("with BH mass" pre-fix); h=0.678 baseline ("without BH mass"); 22-detection dataset (simulations/cramer_rao_bounds.csv)
- Forbidden proxies: Channels "agree" but neither traces to the derivation; peak moves but in wrong direction; two errors cancelling to produce false agreement
- Stop/rethink: If channels still disagree by > 0.03 after fix, re-audit (may indicate structural issue per PROJECT.md stop condition)

**Success Criteria** (what must be TRUE when this phase completes):

1. "With BH mass" posterior peak shifts from h=0.600 toward h=0.678 by at least 0.01 (directional confirmation that the fix addresses the bias)
2. Both channel posterior peaks agree within ~0.01 (acceptance signal from contract)
3. Both channels remain biased relative to true h=0.73 (expected, since P_det=1 is still disabled -- this confirms we haven't introduced a compensating error)
4. The posterior shape is physically reasonable: unimodal, smooth, with width consistent with 22-detection statistical power

## Risk Register

| Phase | Top Risk | Probability | Impact | Mitigation |
|-------|---------|:-:|:-:|-----------|
| 14 | Derivation reveals current code structure is fundamentally wrong (not just Jacobian) | LOW | HIGH | Stop/rethink trigger: escalate to user before proceeding to Phase 15 |
| 15 | Multiple interacting bugs (/(1+z) + sky weight + denominator) make isolated testing hard | MEDIUM | MEDIUM | Fix one term at a time, re-run evaluation after each to isolate effect |
| 16 | Channels converge but by accident (two errors cancelling) | LOW | HIGH | Check each fix independently against derivation; verify limiting cases |

## Backtracking Triggers

- Phase 15: If the derivation (Phase 14) shows the code structure is fundamentally wrong (not a localized fix), escalate to user -- may need architectural changes beyond this milestone's scope
- Phase 16: If "with BH mass" peak moves away from "without BH mass" after fix, return to Phase 15 and re-audit
- Phase 16: If channels still disagree by > 0.03, re-audit numerator/denominator consistency (Phase 15 AUDT-03 may have missed something)

## Progress

**Execution Order:** 14 -> 15 -> 16

| Phase | Plans Complete | Status | Completed |
| --- | --- | --- | --- |
| 14. First-Principles Derivation | 2/2 | Complete | 2026-03-31 |
| 15. Code Audit & Fix | TBD | Not started | - |
| 16. Validation | TBD | Not started | - |
