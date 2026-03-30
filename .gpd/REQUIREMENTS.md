# Requirements: "With BH Mass" Likelihood Bias Audit

**Defined:** 2026-03-30
**Core Research Question:** Why does the "with BH mass" likelihood channel produce an H0 posterior biased to h=0.600, nearly 3x worse than the "without BH mass" channel?

## Primary Requirements

### Derivations

- [ ] **DERV-01**: Derive the standard d_L-only dark siren likelihood from literature (Gray et al. 2020 / Schutz 1986), explicitly showing all terms: GW likelihood, galaxy prior, detection probability, volume element, and sky localization weight placement
- [ ] **DERV-02**: Extend the derivation to include M_z as a 4th observable — derive the Jacobian from source-frame M to M_z_frac = M*(1+z)/M_z_det, the conditional Gaussian decomposition (Bishop 2006 Eq. 2.81-2.82), and the analytic mass marginalization via Gaussian product identity

### Audits

- [ ] **AUDT-01**: Audit the `/(1+z)` factor at `bayesian_statistics.py` line 679 against the DERV-02 derivation — determine whether it is a double-counted Jacobian after the analytic marginalization refactor (commit 15b49a3)
- [ ] **AUDT-02**: Audit sky localization weight placement (phi, theta) in the GW likelihood (TODO flags at lines 556, 755) — verify the weight appears in exactly one factor of the likelihood decomposition, not double-counted between GW likelihood and galaxy catalog term
- [ ] **AUDT-03**: Audit the "with BH mass" denominator (MC-sampled, lines 689-722) for consistency with the numerator formula — verify the same Jacobians, weights, and mass terms appear in both

### Fixes

- [ ] **FIX-01**: Implement all corrections identified by DERV-01/02 and AUDT-01/02/03 in `bayesian_statistics.py`, with reference comments linking each term to the derivation

### Validations

- [ ] **VALD-01**: Re-run H0 evaluation on 22-detection dataset; "with BH mass" posterior peak must shift toward "without BH mass" peak (currently h=0.600 vs h=0.678)
- [ ] **VALD-02**: Both channels must agree within ~0.01 in peak location (both still biased by P_det=1, but consistently so)

## Follow-up Requirements

### Detection Probability

- **PDET-01**: Build simulation-based P_det to replace KDE-based DetectionProbability (Phase 11.1, GSD-tracked)
- **PDET-02**: Re-enable P_det in both likelihood channels and validate h=0.73 within 90% CI

## Out of Scope

| Topic | Reason |
|-------|--------|
| Detection probability (P_det) | Handled separately in Phase 11.1 under GSD |
| Production simulation campaign | v1.2 Phase 12 |
| Full H0 posterior sweep | v1.2 Phase 13 |
| "Without BH mass" channel derivation correctness | Used as reference baseline only |
| wCDM dark energy model | Separate milestone |

## Accuracy and Validation Criteria

| Requirement | Accuracy Target | Validation Method |
|-------------|-----------------|-------------------|
| DERV-01 | Exact analytic derivation | Term-by-term match with literature |
| DERV-02 | Exact analytic derivation | Jacobian dimensional analysis + limiting cases |
| AUDT-01 | Binary (correct/incorrect) | Derivation comparison |
| AUDT-02 | Binary (single/double-counted) | Factor-by-factor audit |
| AUDT-03 | Consistent numerator/denominator | Same terms in both |
| FIX-01 | Code matches derivation | Line-by-line correspondence |
| VALD-01 | Peak shift > 0.01 toward h=0.678 | Comparison script on 22-detection dataset |
| VALD-02 | Peak agreement within 0.01 | Both channels evaluated at same h grid |

## Contract Coverage

| Requirement | Decisive Output / Deliverable | Anchor / Benchmark / Reference | Prior Inputs / Baselines | False Progress To Reject |
|-------------|-------------------------------|-------------------------------|--------------------------|--------------------------|
| DERV-01 | Written derivation with all terms | Gray et al. (2020), Schutz (1986) | Existing "without BH mass" code | Derivation that skips Jacobians |
| DERV-02 | Written derivation extending to M_z | Bishop (2006) for conditioning | DERV-01 result | Handwaving the M → M_z_frac change |
| AUDT-01 | Verdict on /(1+z) with proof | DERV-02 derivation | Commit 15b49a3 diff | "Looks wrong, remove it" without derivation |
| AUDT-02 | Sky weight placement audit | Standard likelihood decomposition | TODO comments at lines 556, 755 | Ignoring the TODOs |
| FIX-01 | Corrected code with ref comments | DERV-01/02 derivations | Current bayesian_statistics.py | Ad-hoc fix without derivation backing |
| VALD-01 | Posterior comparison plot/data | h=0.600 baseline (current) | 22-detection dataset | Peak moves but doesn't match derivation |
| VALD-02 | Channel convergence measurement | h=0.678 ("without BH mass" baseline) | Both channel outputs | Channels "agree" but neither matches derivation |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DERV-01 | — | Pending |
| DERV-02 | — | Pending |
| AUDT-01 | — | Pending |
| AUDT-02 | — | Pending |
| AUDT-03 | — | Pending |
| FIX-01 | — | Pending |
| VALD-01 | — | Pending |
| VALD-02 | — | Pending |

**Coverage:**
- Primary requirements: 8 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 8

---
_Requirements defined: 2026-03-30_
_Last updated: 2026-03-30 after initial definition_
