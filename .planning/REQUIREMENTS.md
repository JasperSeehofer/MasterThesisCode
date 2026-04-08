# Requirements — v2.1 H₀ Bias Resolution

**Defined:** 2026-04-08
**Core Value:** Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.

## v2.1 Requirements

Systematically diagnose and fix the per-event H₀ posterior bias, testing each fix in isolation.

### Diagnostics

- [x] **DIAG-01**: Evaluation can run with f_i=1.0 (catalog-only, no completion term) via CLI flag to confirm L_comp as bias source
- [x] **DIAG-02**: Per-event diagnostic output logs L_cat, L_comp, f_i, and combined likelihood at each h value
- [ ] **DIAG-03**: Baseline posterior snapshot (current MAP h, 68% CI, bias %) saved before any fixes

### Completion Term

- [ ] **COMP-01**: Completion term uses EMRI rate-weighted source population prior instead of bare dVc/dz
- [ ] **COMP-02**: Completion term h-dependence validated: L_comp ratio between h=0.66 and h=0.73 is physically reasonable

### P_det Grid

- [ ] **PDET-01**: P_det grid resolution configurable, increased from 30 to 60 d_L bins
- [ ] **PDET-02**: P_det grid coverage validated: 4-sigma integration bounds fall within grid for >95% of events

### Fisher Quality

- [ ] **FISH-01**: Degenerate Fisher matrices detected and handled (regularization or exclusion) instead of allow_singular=True
- [ ] **FISH-02**: Events with near-singular covariance flagged in diagnostic output with condition number

### Evaluation Infrastructure

- [ ] **EVAL-01**: Before/after comparison report generated automatically: MAP h, 68% CI width, bias %, number of events used
- [ ] **EVAL-02**: Each fix produces a comparison against the baseline, stored in a structured format for cumulative tracking

## Future Requirements

### Remaining Physics Bugs (deferred — not blocking bias resolution)

- **PHYS-01**: wCDM params w0, wa accepted but silently ignored in dist()
- **PHYS-02**: Pipeline A hardcodes 10% σ(d_L) instead of per-source CRB
- **PHYS-03**: Update from WMAP-era cosmology (Omega_m=0.25, H=0.73) to Planck 2018
- **PHYS-04**: Galaxy redshift uncertainty scaling (1+z)^3 has no reference

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full re-simulation campaign | Fix bias first, then re-run in v2.0 Phase 27 |
| Paper updates | v2.0 Paper milestone paused until bias resolved |
| wCDM equation of state | Known bug but not related to H₀ bias |
| Pipeline A (bayesian_inference.py) fixes | Pipeline B is production; Pipeline A is dev cross-check only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIAG-01 | Phase 31 | Complete |
| DIAG-02 | Phase 31 | Complete |
| DIAG-03 | Phase 30 | Pending |
| COMP-01 | Phase 32 | Pending |
| COMP-02 | Phase 32 | Pending |
| PDET-01 | Phase 33 | Pending |
| PDET-02 | Phase 33 | Pending |
| FISH-01 | Phase 34 | Pending |
| FISH-02 | Phase 34 | Pending |
| EVAL-01 | Phase 30 | Pending |
| EVAL-02 | Phase 30 | Pending |

**Coverage:**
- v2.1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-08*
*Last updated: 2026-04-08 — traceability filled after roadmap creation (Phases 30-34)*
