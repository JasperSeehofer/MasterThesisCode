# Requirements: Injection Campaign Physics Analysis

**Defined:** 2026-03-31
**Core Research Question:** Can we improve injection campaign detection yield and P_det grid resolution through enhanced sampling?

## Primary Requirements

### Audit

- [ ] **AUDT-01**: Verify injection parameter distributions (M, z, spin, sky angles) match the simulation pipeline's Model1CrossCheck + ParameterSpace
- [ ] **AUDT-02**: Verify cosmological model consistency (dist() function, h-dependent d_L) between injection and evaluation pipelines
- [ ] **AUDT-03**: Quantify waveform failure rate by parameter region (z, M) from existing injection data or SLURM logs

### Yield Analysis

- [ ] **YELD-01**: Compute detection fraction (SNR >= threshold / total injections) per h-value from existing injection data
- [ ] **YELD-02**: Quantify compute waste breakdown: fraction of GPU time on (a) waveform failures, (b) undetectable events (SNR < threshold), (c) detected events
- [ ] **YELD-03**: Validate z>0.5 cutoff — confirm zero detections above z=0.5 for all 7 h-values, especially h=0.60

### Grid Quality

- [ ] **GRID-01**: Compute per-bin injection counts and Wilson score confidence intervals for current 30x20 grid
- [ ] **GRID-02**: Compare grid quality at 30x20 vs 15x10 resolution (bin occupancy, CI widths, interpolation error)
- [ ] **GRID-03**: Implement quality flags (unreliable bins with <10 injections) in SimulationDetectionProbability

### Enhanced Sampling

- [ ] **SMPL-01**: Design importance-weighted histogram estimator compatible with non-uniform proposal distributions
- [ ] **SMPL-02**: Design stratified sampling with Neyman allocation to concentrate injections on detection boundary bins
- [ ] **SMPL-03**: Design two-stage pilot approach (30% uniform pilot + 70% targeted) with combined importance weights

### Validation

- [ ] **VALD-01**: Verify enhanced P_det grid produces unbiased estimates (comparison with uniform baseline, round-trip consistency checks)

## Follow-up Requirements

### Production Deployment

- **PROD-01**: Run enhanced injection campaign on cluster with new sampling strategy
- **PROD-02**: Build production P_det grids from enhanced injection data
- **PROD-03**: Re-evaluate H0 posterior with improved P_det (feeds back to v1.2.1 Phase 16)

## Out of Scope

| Topic | Reason |
|-------|--------|
| Waveform generator improvements | few/fastlisaresponse reliability is upstream; we work around failures |
| "With BH mass" bias root cause | Deferred to v1.2.1 Phase 16, blocked on P_det data |
| Production simulation campaign | v1.2 Phase 12; this milestone provides the P_det infrastructure |
| Fisher matrix computation | Injection campaign uses SNR-only (D-07); Fisher is separate |

## Accuracy and Validation Criteria

| Requirement | Accuracy Target | Validation Method |
|-------------|----------------|-------------------|
| AUDT-01 | Exact match of parameter ranges and distributions | Code comparison, side-by-side parameter range check |
| AUDT-03 | Failure rate quantified to within 5% | Count from injection CSVs or SLURM logs |
| YELD-01 | Detection fraction per h to 3 significant figures | Direct computation from injection CSV data |
| GRID-01 | Wilson 95% CI per bin | astropy.stats.binom_conf_interval |
| GRID-02 | Interpolation error < 5% at grid cell centers | Leave-one-out cross-validation |
| SMPL-01 | Unbiased to O(1/N) | Comparison with uniform baseline |
| VALD-01 | P_det agreement within 2-sigma Wilson CI | Statistical test on per-bin P_det differences |

## Contract Coverage

| Requirement | Decisive Output | Anchor / Benchmark | Prior Inputs | False Progress To Reject |
|-------------|----------------|-------------------|--------------|--------------------------|
| AUDT-01 | Parameter consistency report | injection_campaign() vs data_simulation() code | main.py, cosmological_model.py | "Looks similar" without line-by-line comparison |
| YELD-01 | Detection yield table per h | Injection CSV data | Cluster injection results | Estimated yield without actual data |
| GRID-01 | Per-bin CI heatmap | Wilson score intervals | SimulationDetectionProbability grids | Average CI without per-bin breakdown |
| SMPL-01 | Weighted estimator formula + code | Tiwari (2018) IS estimator | Current histogram code | Proposal that biases P_det |
| VALD-01 | Comparison test results | Uniform baseline P_det | Both uniform and enhanced grids | Qualitative "looks right" without statistical test |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AUDT-01 | TBD | Pending |
| AUDT-02 | TBD | Pending |
| AUDT-03 | TBD | Pending |
| YELD-01 | TBD | Pending |
| YELD-02 | TBD | Pending |
| YELD-03 | TBD | Pending |
| GRID-01 | TBD | Pending |
| GRID-02 | TBD | Pending |
| GRID-03 | TBD | Pending |
| SMPL-01 | TBD | Pending |
| SMPL-02 | TBD | Pending |
| SMPL-03 | TBD | Pending |
| VALD-01 | TBD | Pending |

**Coverage:**

- Primary requirements: 13 total
- Mapped to phases: 0 (pending roadmap)
- Unmapped: 13

---

_Requirements defined: 2026-03-31_
_Last updated: 2026-03-31 after initial definition_
