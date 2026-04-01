# Research Digest: v1.2.2 Injection Campaign Physics Analysis

Generated: 2026-04-01
Milestone: v1.2.2
Phases: 17-20

## Narrative Arc

The injection campaign -- designed in Phase 11.1 to build simulation-based P_det(z, M | h) grids -- had been running on bwUniCluster but its physics consistency with the main simulation pipeline was unverified, and the detection yield was unknown. This milestone began with a line-by-line audit (Phase 17) confirming that all 14 EMRI parameters and the cosmological model are consistent between the injection and simulation code paths, with only intentional differences per the original design decisions D-01/D-04. The audit also cataloged 8 waveform exception types and revealed that the CSV format records only successful injections, making exact failure rate measurement impossible from data alone. Phase 18 then quantified the detection yield from ~165k injection events: detection fractions range from 0.22% to 0.81% across 7 h-values (at SNR >= 15), meaning >99% of GPU time produces sub-threshold events. Grid quality analysis showed the 30x20 grid has too many unreliable bins given the injection count; a 15x10 grid with 3.2x better confidence intervals was recommended. Recognizing the extreme inefficiency, Phase 19 designed an importance-sampling-enhanced estimator using Neyman-optimal allocation to concentrate future injections on the detection boundary (d_L < 1 Gpc, M > 2e5 M_sun). The IS estimator is exactly backward-compatible with the existing unweighted pipeline (max |diff| = 0.0), and achieves variance reduction factors of 11.8-24.9x in boundary bins -- meaning equivalent grid quality with ~5x fewer GPU hours. Phase 20 validated the full pipeline: VALD-01 confirmed zero statistically significant discrepancies across 916 bins via Benjamini-Hochberg FDR, and VALD-02 proved that the grid-integrated selection integral alpha(h) equals the direct MC sum exactly (an algebraic identity for the unweighted case), confirming zero binning artifacts.

## Key Results

| Phase | Result | Equation / Value | Validity Range | Confidence |
|-------|--------|-----------------|----------------|------------|
| 17 | All 14 EMRI parameters consistent | 4 intentional differences per D-01/D-04 | All parameter space | High (line-by-line audit) |
| 17 | d_L round-trip accuracy | max relative error 2.18e-13 | z in [0, 0.5], h in [0.60, 0.90] | Machine precision |
| 17 | z_cut=0.5 safety | SNR(z=0.5) ~ 3.2 << threshold 15 | h in [0.60, 0.90] | High (SNR scaling) |
| 17 | Detection rate | 0.22% (363 at SNR>=20; 663 at SNR>=15) | 165k events | From data |
| 18 | Per-h detection fraction | f_det = 0.22-0.81% | 7 h-values, SNR>=15 | 3 sig figs |
| 18 | Farr criterion | N_total/N_det >= 124 | All h | High |
| 18 | Grid recommendation | 15x10 over 30x20 (3.2x better CIs) | ~23k events/h | From Wilson CIs |
| 19 | IS estimator backward compat | max |diff| = 0.0 (exact) | All 7 h-values | Machine precision |
| 19 | Variance reduction factor | VRF 11.8-24.9x in boundary bins | All h, boundary region | From Phase 18 data |
| 19 | CI half-width improvement | 3.4-4.6x | Boundary bins | From VRF |
| 19 | Weight bound | max w = 1/alpha = 3.33 | Defensive mixture alpha=0.3 | Analytical proof |
| 20 | VALD-01 | PASS: zero BH discoveries / 916 bins | All h, 15x10 grid | BH FDR alpha=0.05 |
| 20 | VALD-02 | alpha_grid = alpha_MC exactly | All 7 h-values | Algebraic identity |

## Methods Employed

- **Phase 17:** code-path-tracing -- line-by-line parameter comparison between injection_campaign() and data_simulation()
- **Phase 17:** round-trip-numerical-test -- d_L(z, h) -> z -> d_L round-trip across 7 h x 100 z grid
- **Phase 17:** SNR-scaling-argument -- SNR ~ M^{5/6}/d_L to bound detectable z range
- **Phase 17:** code-path-audit, injection-csv-analysis -- exception catalog and data-driven failure analysis
- **Phase 18:** csv-aggregation, waste-decomposition -- per-h detection yield and GPU waste breakdown
- **Phase 18:** wilson-score-ci -- per-bin 95% confidence intervals for P_det grid
- **Phase 18:** grid-resolution-comparison -- 30x20 vs 15x10 CI widths and interpolation error
- **Phase 18:** farr-criterion-check -- N_eff > 4*N_det sufficiency check
- **Phase 19:** self-normalized-IS-estimator -- importance-weighted P_det grid construction
- **Phase 19:** kish-effective-sample-size -- per-bin N_eff diagnostic for IS weights
- **Phase 19:** neyman-optimal-allocation -- variance-minimizing injection budget allocation
- **Phase 19:** defensive-mixture-proposal -- q = 0.3*uniform + 0.7*targeted ensures full support
- **Phase 19:** variance-reduction-factor -- VRF from actual per-bin Phase 18 statistics
- **Phase 20:** wilson-ci-overlap-test -- per-bin comparison of standard vs IS P_det
- **Phase 20:** bh-fdr-correction -- Benjamini-Hochberg across 916 pooled tests
- **Phase 20:** isotonic-monotonicity-check -- per-column isotonic regression for P_det(d_L | M)
- **Phase 20:** direct-mc-selection-integral -- Mandel, Farr & Gair (2019) alpha(h) = sum(w_i)/N
- **Phase 20:** grid-vs-mc-comparison -- alpha_grid vs alpha_MC round-trip check

## Convention Evolution

| Phase | Convention | Description | Status |
|-------|-----------|-------------|--------|
| (pre-existing) | Metric signature: mostly-plus | GR sign convention | Active |
| (pre-existing) | Fourier convention: physics | e^{-iwt} convention | Active |
| (pre-existing) | Natural units: SI | Distances in Gpc, masses in M_sun | Active |
| (pre-existing) | Coordinate system: spherical | Sky coordinates (theta, phi) | Active |
| (pre-existing) | Index positioning: Einstein | Summation convention | Active |

No new conventions were introduced during v1.2.2 (analysis/validation milestone, no new physics formulas).

## Figures and Data Registry

| File | Phase | Description | Paper-ready? |
|------|-------|-------------|--------------|
| figures/injection_yield_waste_breakdown.pdf | 18 | Pie chart: detected / sub-threshold / failed fractions | Yes |
| figures/grid_wilson_ci_heatmap.pdf | 18 | Per-bin Wilson CI heatmap for 30x20 grid | Yes |
| figures/grid_30x20_vs_15x10_comparison.pdf | 18 | Side-by-side grid resolution comparison | Yes |
| analysis/injection_yield.py | 18 | Standalone yield computation script | N/A (code) |
| analysis/grid_quality.py | 18 | Grid quality assessment script | N/A (code) |
| analysis/importance_sampling.py | 19 | IS utility: weighted histogram, Kish N_eff, IS Wilson CI | N/A (code) |
| analysis/sampling_design.py | 19 | Neyman allocation, VRF, defensive mixture | N/A (code) |
| analysis/validation.py | 20 | Validation framework: CI overlap, monotonicity, Farr, grid-vs-MC | N/A (code) |
| .gpd/phases/17-*/audit-parameter-consistency.md | 17 | Line-by-line parameter comparison report | No (reference) |
| .gpd/phases/17-*/audit-cosmological-model.md | 17 | Cosmological model audit + round-trip results | No (reference) |
| .gpd/phases/17-*/audit-waveform-failures.md | 17 | Exception catalog and failure characterization | No (reference) |
| .gpd/phases/18-*/yield-report.md | 18 | Detection yield tables and waste analysis | No (reference) |
| .gpd/phases/18-*/grid-quality-report.md | 18 | Grid quality tables and comparison | No (reference) |
| .gpd/phases/19-*/sampling-design-report.md | 19 | Full sampling design specification | No (reference) |
| .gpd/phases/20-*/validation-report.md | 20 | Combined VALD-01 + VALD-02 results | No (reference) |

## Open Questions

### Resolved by v1.2.2

1. Are injection parameter distributions consistent with the simulation pipeline? **YES** -- all 14 parameters traced, 4 intentional differences documented
2. What is the detection yield? **0.22-0.81%** per h-value at SNR >= 15
3. What fraction of compute is wasted? **>99%** on sub-threshold events; failure rate bracketed at 30-50%
4. Can importance sampling improve P_det grid quality? **YES** -- VRF 11.8-24.9x in boundary bins; CI improvement 3.4-4.6x
5. Does the z > 0.5 cutoff introduce bias? **NO** -- zero detections above z = 0.204; SNR ~ 3.2 at z = 0.5

### Remaining (from v1.2.1, on hold)

1. What causes the remaining "with BH mass" low-h bias after /(1+z) fix? (Phase 16 scope, blocked on P_det data)
2. p_det in numerator uses detection.M rather than galaxy M*(1+z) at trial z -- documented as known approximation

## Dependency Graph

```
Phase 17 "Injection Physics Audit"
  requires: Phase 11.1 design decisions D-01 through D-09
  provides: 14-parameter consistency, d_L round-trip, z_cut safety, exception catalog, 165k-event analysis
  -> Phase 18

Phase 18 "Detection Yield & Grid Quality"
  requires: Phase 17 (parameter consistency verified)
  provides: Per-h yield table, waste decomposition, Wilson CIs, 15x10 grid recommendation, quality flags
  -> Phase 19

Phase 19 "Enhanced Sampling Design"
  requires: Phase 18 (grid quality diagnostics, boundary identification)
  provides: IS estimator, Neyman allocation, VRF 11.8-24.9x, two-stage design, defensive mixture
  -> Phase 20

Phase 20 "Validation"
  requires: Phase 18 (uniform baseline), Phase 19 (IS estimator)
  provides: VALD-01 PASS (916 bins), VALD-02 PASS (alpha_grid = alpha_MC)
  -> production campaign (next milestone)
```

## Mapping to Original Objectives

| Requirement | Status | Fulfilled by | Key Result |
|-------------|--------|-------------|------------|
| AUDT-01: Parameter distribution match | Complete | Phase 17-01 | All 14 parameters consistent; 4 intentional diffs per D-01/D-04 |
| AUDT-02: Cosmological model consistency | Complete | Phase 17-01 | dist() defaults identical; d_L round-trip error 2e-13 |
| AUDT-03: Waveform failure by region | Complete | Phase 17-02 | 8 exception types cataloged; detections at z<0.2, M~10^5.5-6 |
| YELD-01: Detection fraction per h | Complete | Phase 18-01 | f_det = 0.22-0.81% (3 sig figs, 7 h-values) |
| YELD-02: Compute waste breakdown | Complete | Phase 18-01 | >99% sub-threshold; failure rate 30-50% (bracketed) |
| YELD-03: z>0.5 cutoff validation | Complete | Phase 18-01 | Zero detections above z=0.5; all below z=0.204 |
| GRID-01: Per-bin Wilson CIs | Complete | Phase 18-02 | 95% Wilson CIs for 30x20 and 15x10 grids |
| GRID-02: 30x20 vs 15x10 comparison | Complete | Phase 18-02 | 15x10 recommended: 3.2x better CIs, 93% reliable bins |
| GRID-03: Quality flags | Complete | Phase 18-02 | n_total, n_detected, reliable arrays in SimulationDetectionProbability |
| SMPL-01: IS estimator design | Complete | Phase 19-01 | Self-normalized IS with exact backward compat (max diff = 0.0) |
| SMPL-02: Stratified sampling | Complete | Phase 19-02 | Neyman-optimal allocation; VRF 11.8-24.9x |
| SMPL-03: Two-stage pilot design | Complete | Phase 19-02 | 30% uniform pilot + 70% targeted; alpha=0.3 defensive mixture |
| VALD-01: Unbiased P_det verification | Complete | Phase 20-01 | PASS: zero BH discoveries / 916 bins; monotonicity OK; Farr OK |
| VALD-02: Grid vs MC alpha(h) | Complete | Phase 20-02 | PASS: alpha_grid = alpha_MC exactly (algebraic identity) |
