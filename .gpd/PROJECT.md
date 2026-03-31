# EMRI Parameter Estimation — Dark Siren H0 Inference

## What This Is

A gravitational wave parameter estimation pipeline for LISA Extreme Mass Ratio Inspirals (EMRIs). The Bayesian inference pipeline evaluates the Hubble constant posterior using a dark siren method: summing over candidate host galaxies weighted by GW measurement likelihood. This milestone analyzes the injection campaign (used to build simulation-based P_det grids) from a physics perspective, auditing parameter consistency and designing enhanced sampling strategies to improve detection yield and P_det grid quality.

## Core Research Question

Can we improve the injection campaign's detection yield and P_det grid resolution through importance sampling or enhanced sampling strategies, given that the current uniform approach wastes significant GPU time on undetectable events?

## Current Milestone: v1.2.2 Injection Campaign Physics Analysis

**Goal:** Audit the injection campaign physics, analyze detection yield, and design enhanced sampling strategies.

**Target results:**

- Physics consistency audit between injection and simulation pipelines
- Detection yield analysis with quantified waste fraction
- Enhanced/importance sampling design that concentrates injections in detectable parameter regions

## Scoping Contract Summary

### Contract Coverage

- **Decisive output:** Detection yield report quantifying current waste; enhanced sampling proposal with expected improvement factor
- **Acceptance signal:** Enhanced sampling design that provably reduces GPU time by >2x for equivalent P_det grid quality
- **False progress to reject:** Sampling strategy that introduces bias in P_det estimates; detection yield analysis without quantified improvement path

### User Guidance To Preserve

- **User-stated observables:** Fraction of injections detected vs wasted; P_det grid resolution in regions that matter for H0 posterior
- **User-stated deliverables:** (1) Injection physics audit, (2) detection yield analysis, (3) enhanced sampling strategy
- **Must-have references:** Phase 11.1 design decisions (D-01 through D-09), injection_campaign() in main.py
- **Stop / rethink conditions:** If the injection physics has fundamental inconsistencies with the simulation pipeline that invalidate existing P_det grids

### Scope Boundaries

**In scope**

- Audit injection parameter distributions vs simulation pipeline (Model1CrossCheck, ParameterSpace)
- Audit cosmological model consistency (dist() function, h-dependent d_L)
- Analyze detection yield from existing/incoming injection data
- Design importance sampling or stratified sampling for injection campaign
- Analyze P_det grid quality (bin counts, empty bins, interpolation accuracy)

**Out of scope**

- "With BH mass" likelihood bias (v1.2.1 Phase 16)
- Production simulation campaign (v1.2 Phase 12)
- Full H0 posterior evaluation (v1.2 Phase 13)
- Waveform generation improvements (few/fastlisaresponse)
- Fisher matrix computation changes

### Active Anchor Registry

- **Phase 11.1 design decisions (D-01 to D-09):** Injection campaign architecture
  - Why it matters: Defines current sampling approach that we're analyzing/improving
  - Carry forward: planning | execution
  - Required action: read | audit | extend

- **injection_campaign() in main.py:396-572:** Current implementation
  - Why it matters: Source of truth for what the injection actually does
  - Carry forward: execution
  - Required action: read | audit

- **SimulationDetectionProbability class:** P_det grid construction
  - Why it matters: Downstream consumer of injection data; grid quality determines P_det accuracy
  - Carry forward: execution | verification
  - Required action: read | analyze

### Carry-Forward Inputs

- `master_thesis_code/main.py` — injection_campaign() function
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py` — P_det grid builder
- `master_thesis_code/cosmological_model.py` — Model1CrossCheck EMRI sampling
- `.planning/phases/11.1-simulation-based-detection-probability/11.1-CONTEXT.md` — design decisions
- Injection CSVs from cluster (when available) — raw injection results

### Skeptical Review

- **Weakest anchor:** Actual detection yield numbers (injection data may not be rsynced yet)
- **Unvalidated assumptions:** That the uniform sampling is actually uniform in the relevant coordinates (Model1CrossCheck uses MCMC)
- **Competing explanation:** Low detection yield may be dominated by waveform generation failures, not parameter space coverage
- **Disconfirming observation:** If detection yield is already >50%, enhanced sampling provides diminishing returns
- **False progress to reject:** Sampling strategy that biases P_det estimates (importance weights must be correctly accounted for)

### Open Contract Questions

- What is the actual detection fraction from the existing injection campaign?
- What fraction of GPU time is wasted on waveform failures vs undetectable events?
- Can importance sampling be applied without modifying the P_det grid construction math?

## Research Questions

### Answered (v1.2.1)

- [x] Single-galaxy likelihood peaks at h=0.73 (formula is correct in isolation) — debug session 2026-03-30
- [x] Both channels biased by P_det=1 (galaxy catalog density dominates) — debug session 2026-03-30
- [x] BH mass Gaussian index [0] vs [1] has no effect under delta-function approximation — quick task 260330-twe
- [x] `/(1+z)` at line 679 is a double-counted Jacobian — Phase 14 derivation, removed Phase 15
- [x] Sky localization weight correctly inside 3D GW Gaussian — Phase 14
- [x] "With BH mass" denominator consistent with numerator — Phase 14, Phase 15
- [x] /(1+z) fix alone insufficient to resolve bias — Phase 15 quick validation

### Active (v1.2.2)

- [ ] Are injection parameter distributions consistent with the simulation pipeline?
- [ ] What is the detection yield (detected / total injections)?
- [ ] What fraction of compute is wasted on waveform failures vs undetectable events?
- [ ] Can importance sampling improve P_det grid quality with fewer injections?
- [ ] Does the z > 0.5 cutoff introduce any bias in P_det estimates?

### Out of Scope

- "With BH mass" bias root cause (deferred to v1.2.1 Phase 16)
- Waveform generation reliability improvements
- Production campaign execution

## Research Context

### Physical System

Dark siren measurement of the Hubble constant using LISA EMRI detections. Each GW detection provides d_L (and optionally M_z) measurements. Cross-matching with a galaxy catalog provides candidate host galaxies. The H0 posterior is obtained by marginalizing over all candidate hosts.

### Theoretical Framework

Bayesian hierarchical inference: H0 posterior from GW dark sirens with galaxy catalogs (Schutz 1986; Gray et al. 2020). The "with BH mass" extension adds M_z = M*(1+z) as a fourth observable alongside sky location and luminosity distance.

### Key Parameters and Scales

| Parameter | Symbol | Regime | Notes |
|-----------|--------|--------|-------|
| Hubble constant | h | 0.6-0.9 | True value h=0.73 (WMAP-era) |
| Redshift | z | 0.03-0.20 | Range of 22 validation detections |
| d_L uncertainty | sigma_dL/dL | ~few % | From Fisher matrix / CRB |
| M_z uncertainty | sigma_Mz/Mz | ~1e-7 | Near-delta function (very precise) |

### Known Results

- "Without BH mass" peaks at h=0.678 (P_det=1 baseline)
- "With BH mass" peaks at h=0.600 (P_det=1 baseline, still biased after /(1+z) fix)
- Injection campaign uses z > 0.5 cutoff for efficiency (all detections at z < 0.18)
- SimulationDetectionProbability builds 2D P_det(z, M | h) grids with 30 z-bins × 20 M-bins
- Waveform generation has ~30-50% failure rate (timeouts, parameter bounds, convergence)
- Design decisions D-01 through D-09 govern injection architecture

### What Is New

Physics analysis of the injection campaign: consistency audit, detection yield quantification, and enhanced sampling design to improve P_det grid quality with fewer GPU hours.

### Computational Environment

Local workstation (CPU-only dev machine). Evaluation pipeline runs in ~minutes on the 22-detection dataset.

## Notation and Conventions

- d_L: luminosity distance (Gpc)
- M: source-frame BH mass (solar masses)
- M_z = M*(1+z): redshifted (detector-frame) BH mass
- d_L_frac = d_L(z,h) / d_L_det: luminosity distance fraction
- M_z_frac = M_gal*(1+z) / M_z_det: redshifted mass fraction
- h: dimensionless Hubble parameter H0/(100 km/s/Mpc)

## Unit System

Natural units for GW quantities; cosmological distances in Gpc; masses in solar masses.

## Requirements

See `.gpd/REQUIREMENTS.md` for the detailed requirements specification.

## Key References

- Gray et al. (2020) — Standard dark siren H0 inference with galaxy catalogs
- Schutz (1986) — Original dark siren proposal
- Chen et al. (2018) — Galaxy catalog method for GW cosmology
- Bishop (2006) PRML Eq. 2.81-2.82 — Multivariate normal conditioning (used in analytic marginalization)

## Constraints

- **P_det=1:** Detection probability is disabled (`_DEBUG_DISABLE_DETECTION_PROBABILITY = True`). Both channels are biased by this, but the bias audit focuses on the mass-specific additional bias.
- **22-detection dataset:** Validation uses the Phase 11 detection catalog.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep analytic M_z marginalization (15b49a3) | Correct physics, replaces delta-function approximation | Confirmed (v1.2.1) |
| /(1+z) spurious, removed | Phase 14 derivation proved Jacobian absorbed | Confirmed, insufficient for bias (v1.2.1) |
| v1.2.1 on hold pending P_det data | Need injection-based P_det before further bias investigation | User decision |
| Analyze injection physics before next campaign | GPU time is expensive; improve yield first | v1.2.2 scope |

---

_Last updated: 2026-03-31 after milestone v1.2.2 initialization_
