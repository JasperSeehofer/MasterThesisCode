# Requirements — v1.5 Galaxy Catalog Completeness Correction

## Completeness Estimation

- [ ] **COMP-01**: Compute GLADE+ completeness f(z) from B-band luminosity comparison with Schechter function extrapolation
- [ ] **COMP-02**: Provide angle-averaged f(z) as interpolatable function usable by the likelihood computation

## Likelihood Correction

- [ ] **LIKE-01**: Add completion term integrating GW likelihood over uniform-in-comoving-volume prior for uncataloged hosts (Gray et al. 2020 Eq. 9)
- [ ] **LIKE-02**: Add comoving volume element dV_c/dz/dOmega to physical_relations.py
- [ ] **LIKE-03**: Modify p_Di() to combine catalog term + completion term weighted by f(z) and (1-f(z))
- [ ] **LIKE-04**: Thread completeness function through evaluate() -> p_D() -> p_Di() call chain

## Verification

- [ ] **VER-01**: Limiting case f=1 recovers current (uncorrected) code exactly
- [ ] **VER-02**: Limiting case f=0 produces broad posterior centered near true H0 (statistical siren)
- [ ] **VER-03**: Re-run 534-detection dataset with corrected likelihood showing MAP shift toward h=0.73
- [ ] **VER-04**: Sensitivity test: vary f(z) by +/-20% and verify posterior peak is stable (shift << 20%)

## Deployment

- [ ] **DEPL-01**: Deploy corrected code to bwUniCluster and run production evaluation with real P_det

## Future Requirements

- Angular-dependent completeness f(z, Omega) using per-event sky position (second-order improvement)
- "With BH mass" pathway completeness correction (requires additional mass-dependent completeness)
- Comparison with gwcosmo package results on same dataset

## Out of Scope

- Full pixelated completeness a la Gray et al. (2022) — angular variation within LISA error boxes is negligible
- Replacing GLADE+ with a different catalog — GLADE+ is the standard for this analysis
- Updating to Planck 2018 cosmology (Omega_m=0.3153) — separate physics change, tracked in CLAUDE.md

## Traceability

| REQ-ID | Phase | Plan | Status |
|--------|-------|------|--------|
| COMP-01 | Phase 24 | — | pending |
| COMP-02 | Phase 24 | — | pending |
| LIKE-01 | Phase 25 | — | pending |
| LIKE-02 | Phase 25 | — | pending |
| LIKE-03 | Phase 25 | — | pending |
| LIKE-04 | Phase 25 | — | pending |
| VER-01 | Phase 26 | — | pending |
| VER-02 | Phase 26 | — | pending |
| VER-03 | Phase 26 | — | pending |
| VER-04 | Phase 26 | — | pending |
| DEPL-01 | Phase 27 | — | pending |

## References

- Gray et al. (2020), arXiv:1908.06050 — primary framework (Eq. 9)
- Dalya et al. (2022), arXiv:2110.06184 — GLADE+ completeness data
- Finke et al. (2021), arXiv:2101.12660 — alternative approach comparison
- Research specification: `.gpd/quick/3-literature-research-galaxy-catalog-in/galaxy-catalog-completeness-research.md`
- Bias investigation: `scripts/bias_investigation/FINDINGS.md`
