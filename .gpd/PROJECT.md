# EMRI Parameter Estimation -- Dark Siren H0 Inference

## What This Is

A gravitational wave parameter estimation pipeline for LISA Extreme Mass Ratio Inspirals (EMRIs). The Bayesian inference pipeline evaluates the Hubble constant posterior using a dark siren method: summing over candidate host galaxies weighted by GW measurement likelihood. The simulation-based detection probability (P_det) infrastructure -- including an importance-sampling-enhanced estimator validated to produce unbiased estimates -- is now complete and ready for production deployment.

## Core Research Question

What is the Hubble constant H0 as measured by dark siren inference from LISA EMRI detections, and how does the measurement precision depend on the number of detected events and the quality of the selection function estimate?

## Requirements

### Validated

- [x] Derive "with BH mass" dark siren likelihood from first principles -- v1.2.1 (Phase 14)
- [x] Audit and fix bayesian_statistics.py to match derivation -- v1.2.1 (Phase 15; /(1+z) fix applied, insufficient alone)
- [x] Verify injection/simulation parameter consistency -- v1.2.2 (Phase 17; all 14 parameters consistent)
- [x] Verify cosmological model consistency across pipelines -- v1.2.2 (Phase 17; dist() identical, d_L round-trip to 2e-13)
- [x] Characterize waveform failure modes -- v1.2.2 (Phase 17; 8 exception types cataloged)
- [x] Compute detection yield per h-value -- v1.2.2 (Phase 18; f_det 0.22-0.81%)
- [x] Quantify GPU compute waste -- v1.2.2 (Phase 18; >99% sub-threshold)
- [x] Validate z > 0.5 cutoff -- v1.2.2 (Phase 18; zero detections above z=0.204)
- [x] Compute per-bin Wilson CIs for P_det grid -- v1.2.2 (Phase 18; 15x10 recommended)
- [x] Quality flags in SimulationDetectionProbability -- v1.2.2 (Phase 18)
- [x] Design IS-weighted histogram estimator -- v1.2.2 (Phase 19; exact backward compat)
- [x] Design Neyman-optimal stratified sampling -- v1.2.2 (Phase 19; VRF 11.8-24.9x)
- [x] Validate enhanced P_det estimator -- v1.2.2 (Phase 20; VALD-01 + VALD-02 PASS)

### Active

- [ ] Resolve "with BH mass" posterior low-h bias (v1.2.1 Phase 16, blocked on P_det data)
- [ ] Run enhanced injection campaign on cluster with new sampling strategy
- [ ] Build production P_det grids from enhanced injection data
- [ ] Re-evaluate H0 posterior with improved P_det

### Out of Scope

- Waveform generator reliability improvements -- few/fastlisaresponse is upstream
- Cluster extensions (CDMFT) -- not applicable
- Superconducting order -- not applicable
- wCDM model parameters (w0, wa) -- LCDM hardcoded, known bug but low priority
- Pipeline A (bayesian_inference.py) 10% sigma(d_L) -- development cross-check only
- WMAP-era cosmology constants -- consistent within pipeline, updating is low priority

## Context

### Current Research State

**v1.2.1 (On Hold):** First-principles derivation of the "with BH mass" likelihood complete (Phase 14). Code audited and /(1+z) spurious Jacobian removed (Phase 15). Posterior still monotonically decreasing -- additional bias sources exist beyond the /(1+z) fix. Phase 16 (validation with real P_det) blocked on injection data.

**v1.2.2 (Complete):** Injection campaign physics fully audited -- all 14 EMRI parameters consistent between injection and simulation pipelines. Detection yield is extremely low (0.22-0.81% at SNR >= 15), with >99% of GPU time producing sub-threshold events. The P_det grid infrastructure is now enhanced with:
- 15x10 grid recommended over 30x20 (3.2x better Wilson CIs for ~23k events/h)
- Quality flags (n_total, n_detected, reliable) in SimulationDetectionProbability
- IS-weighted histogram estimator with exact backward compatibility
- Neyman-optimal allocation achieving VRF 11.8-24.9x in detection boundary bins
- Two-stage design (30% pilot + 70% targeted) with defensive mixture (alpha=0.3)
- Full validation: VALD-01 PASS (916 bins, BH FDR), VALD-02 PASS (alpha_grid = alpha_MC exactly)

### Key Numerical Results

| Quantity | Value | Source |
|----------|-------|--------|
| "Without BH mass" H0 peak | h = 0.678 (P_det=1) | Phase 15 |
| "With BH mass" H0 peak | h = 0.600 (P_det=1, post-fix) | Phase 15 |
| Detection yield | 0.22-0.81% per h (SNR>=15) | Phase 18 |
| Farr criterion | N_total/N_det >= 124 | Phase 18 |
| IS estimator backward compat | max |diff| = 0.0 | Phase 19 |
| VRF (boundary bins) | 11.8-24.9x | Phase 19 |
| CI half-width improvement | 3.4-4.6x | Phase 19 |
| Validation bins tested | 916 (zero BH discoveries) | Phase 20 |

### Literature Comparison

- Mandel, Farr & Gair (2019): LVK uses direct MC selection integral, not gridded P_det. Our grid-based approach is non-standard but validated (VALD-02: alpha_grid = alpha_MC exactly for unweighted case).
- Tiwari (2018): IS estimator for GW population inference. Our implementation follows this approach adapted to histogram-based P_det.
- Farr (2019): N_eff > 4*N_det criterion satisfied globally (min ratio 124x).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep analytic M_z marginalization (15b49a3) | Correct physics per Bishop (2006) | Good (v1.2.1) |
| /(1+z) spurious, removed | Phase 14 derivation proved Jacobian absorbed | Good, but insufficient for bias (v1.2.1) |
| v1.2.1 on hold pending P_det data | Need injection-based P_det before further bias investigation | Pending |
| 15x10 grid over 30x20 | 3.2x better CIs for ~23k events/h | Good (v1.2.2) |
| Preserve unweighted code path (weights=None) | Bit-for-bit backward compatibility | Good (v1.2.2) |
| alpha=0.3 defensive mixture | Bounds max weight at 3.33; full support proof | Good (v1.2.2) |
| Reliable mask on n_total >= 10, not N_eff | N_eff is diagnostic, not gating | Good (v1.2.2) |

## Current Milestone: v2.1 Publication Figures

**Goal:** Unify and modernize the visualization pipeline with publication-quality style, merge disconnected figure paths, exploit per-galaxy likelihood data for novel visualizations, and add interactive figures for GitHub Pages.

**Target results:**

- Unified `--generate_figures` manifest covering paper, thesis, and galaxy-level figures (~20+)
- Publication-refined style: LaTeX/REVTeX, sequential emphasis palette + accent color, colorblind-safe, minimal spines, modern contour aesthetics (inspired by arXiv:2406.09228v1)
- Galaxy-level visualizations from 580MB per-event per-galaxy likelihood data (ranking, dominant fraction, BH mass impact, sky map)
- Completeness f(z,h) standalone + P_det-integrated figures
- Contour variants for H0 posterior, P_det surface, parameter correlations, galaxy likelihood maps
- Interactive Plotly/Bokeh HTML figures for GitHub Pages (hover, zoom, pan)
- Jupyter widget notebooks for parameter exploration
- All static figures print-optimized and grayscale-safe
- Auto-detect h-grid resolution (works with current 15-pt and future finer grids)
- Galaxy data: pre-process to CSV for speed, direct-load fallback for high-memory machines

## Constraints

- **GPU:** CUDA 12 required; injection campaign runs on bwUniCluster gpu_h100 partition
- **Detection yield:** 0.22-0.81% means ~100k injections needed per h for ~500 detections
- **h=0.90:** Insufficient statistics even for 15x10 grid (47% unreliable bins)
- **CSV format:** Records only successful injections; exact failure rate requires SLURM logs
- **P_det=1 debug mode:** `_DEBUG_DISABLE_DETECTION_PROBABILITY = True` still active for bias investigation

---

_Last updated: 2026-04-07 after v2.1 milestone start_
