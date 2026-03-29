# Requirements: EMRI Parameter Estimation v1.2

**Defined:** 2026-03-29
**Core Value:** The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramer-Rao bounds for statistically meaningful Hubble constant posteriors.

## v1.2 Requirements

Requirements for production campaign and physics corrections. Each maps to roadmap phases.

### Physics Corrections

- [ ] **PHYS-01**: Fisher matrix uses 5-point stencil derivative instead of forward-difference
- [ ] **PHYS-02**: LISA PSD includes galactic confusion noise term (Babak et al. 2023)
- [ ] **PHYS-03**: CRB computation timeout increased to accommodate 5-point stencil (56 waveforms vs 15)

### Simulation Campaign

- [ ] **SIM-01**: Validation campaign (3-5 tasks) confirms detection rates and timing with corrected physics
- [ ] **SIM-02**: Production simulation campaign runs 100+ tasks on bwUniCluster with corrected physics
- [ ] **SIM-03**: d_L fractional error threshold recalibrated based on 5-point stencil accuracy

### H0 Analysis

- [ ] **H0-01**: Full H0 posterior sweep evaluates detection catalog over h in [0.6, 0.9]
- [ ] **H0-02**: Combined posterior plot shows H0 constraint with credible intervals

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Dark Energy Model

- **wCDM-01**: w0, wa parameters used in distance calculations (currently silently ignored)

### Cosmological Parameters

- **COSMO-01**: Planck 2018 cosmological parameters replace WMAP-era values

### Observational Uncertainty

- **UNCERT-01**: Galaxy redshift uncertainty scaling corrected to standard (1+z) form

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| wCDM dark energy model | Requires physics review + separate validation; not needed for LCDM thesis results |
| Planck 2018 cosmology update | Changes baseline; better done as separate milestone after thesis results |
| Per-source CRB d_L in Pipeline A | Pipeline A is dev cross-check only; Pipeline B already uses per-source bounds |
| Multi-node MPI distribution | Array jobs provide sufficient parallelism |
| GPU CI runners | Cluster testing is manual; GitHub Actions stays CPU-only |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PHYS-01 | Phase 10 | Pending |
| PHYS-02 | Phase 9 | Pending |
| PHYS-03 | Phase 10 | Pending |
| SIM-01 | Phase 11 | Pending |
| SIM-02 | Phase 12 | Pending |
| SIM-03 | Phase 11 | Pending |
| H0-01 | Phase 13 | Pending |
| H0-02 | Phase 13 | Pending |

**Coverage:**
- v1.2 requirements: 8 total
- Mapped to phases: 8
- Unmapped: 0

---
*Requirements defined: 2026-03-29*
*Last updated: 2026-03-29 after roadmap creation*
