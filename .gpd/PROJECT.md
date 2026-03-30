# EMRI Parameter Estimation — Dark Siren H0 Inference

## What This Is

A gravitational wave parameter estimation pipeline for LISA Extreme Mass Ratio Inspirals (EMRIs). The Bayesian inference pipeline evaluates the Hubble constant posterior using a dark siren method: summing over candidate host galaxies weighted by GW measurement likelihood. Two likelihood channels exist: "without BH mass" (3D: sky + distance) and "with BH mass" (4D: sky + distance + redshifted mass). This milestone audits the "with BH mass" channel, which shows a systematic bias.

## Core Research Question

Why does the "with BH mass" likelihood channel produce an H0 posterior biased to h=0.600 (offset -0.130 from true h=0.73), nearly 3x worse than the "without BH mass" channel (h=0.678, offset -0.052)?

## Scoping Contract Summary

### Contract Coverage

- **Decisive output:** First-principles derivation of the "with BH mass" likelihood, code audit showing term-by-term match, and posterior convergence between channels
- **Acceptance signal:** "With BH mass" and "without BH mass" posterior peaks agree within ~0.01 (both still biased by P_det=1, but consistently)
- **False progress to reject:** Ad-hoc numerical fix without derivation; posterior shift that doesn't trace to a specific term

### User Guidance To Preserve

- **User-stated observables:** "With BH mass" posterior peak location relative to "without BH mass"
- **User-stated deliverables:** First-principles derivation extending d_L-only literature to include M_z
- **Must-have references:** Gray et al. (2020), Schutz (1986), Chen et al. (2018) for baseline d_L-only formulation
- **Stop / rethink conditions:** If the derivation shows the current code structure is fundamentally wrong (not just a Jacobian fix)

### Scope Boundaries

**In scope**

- Derive correct "with BH mass" dark siren likelihood from first principles
- Fix spurious `/(1+z)` Jacobian (line 679 in `bayesian_statistics.py`)
- Audit sky localization weight placement (TODO flags at lines 556, 755)
- Audit numerator/denominator consistency for "with BH mass" channel
- Re-run evaluation to validate both channels converge

**Out of scope**

- Detection probability fix (P_det) — handled separately in Phase 11.1
- Production simulation campaign
- Full H0 posterior sweep over [0.6, 0.9]
- wCDM dark energy model
- Planck 2018 cosmology update

### Active Anchor Registry

- **Gray et al. (2020):** Standard dark siren formulation with galaxy catalog
  - Why it matters: Baseline d_L-only likelihood that must be correctly extended
  - Carry forward: derivation | execution | verification
  - Required action: read | derive from

- **Debug session (2026-03-30):** `.planning/debug/h0-posterior-residual-bias.md`
  - Why it matters: Established that single-galaxy test peaks at h=0.73, bias is from multi-galaxy summation
  - Carry forward: execution | verification
  - Required action: use as baseline

### Carry-Forward Inputs

- `simulations/cramer_rao_bounds.csv` — 22-detection validation dataset
- `.planning/debug/h0-posterior-residual-bias.md` — root cause analysis
- `.planning/phases/11.1-simulation-based-detection-probability/11.1-CONTEXT.md` — P_det context (out of scope but related)

### Skeptical Review

- **Weakest anchor:** The `/(1+z)` being the sole source of extra "with BH mass" bias — sky localization weight may also contribute
- **Unvalidated assumptions:** That the 3D ("without BH mass") channel is correct — it's used as the reference but hasn't been derived from first principles either
- **Competing explanation:** Sky localization weight double-counting could be the real issue; `/(1+z)` may be secondary
- **Disconfirming observation:** If removing `/(1+z)` doesn't move the "with BH mass" peak closer to "without BH mass"
- **False progress to reject:** Channels converging by accident (e.g., two errors cancelling)

### Open Contract Questions

- Is the sky localization weight (phi, theta) correctly placed in exactly one factor of the likelihood?
- Does the MC-sampled "with BH mass" denominator (lines 689-722) have the correct weighting?

## Research Questions

### Answered

- [x] Single-galaxy likelihood peaks at h=0.73 (formula is correct in isolation) — debug session 2026-03-30
- [x] Both channels biased by P_det=1 (galaxy catalog density dominates) — debug session 2026-03-30
- [x] BH mass Gaussian index [0] vs [1] has no effect under delta-function approximation — quick task 260330-twe

### Active

- [ ] Is the `/(1+z)` at line 679 a double-counted Jacobian after the analytic marginalization refactor?
- [ ] Is the sky localization weight placed correctly (in exactly one likelihood factor)?
- [ ] Is the "with BH mass" denominator consistent with the numerator?
- [ ] Does the analytic M_z marginalization (commit 15b49a3) introduce any other issues?

### Out of Scope

- P_det implementation — separate Phase 11.1
- "Without BH mass" channel correctness — used as reference only

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
- "With BH mass" peaks at h=0.600 (P_det=1 baseline)
- Analytic M_z marginalization (commit 15b49a3) didn't change peak location
- Debug session eliminated: covariance construction, integration limits, core formula, d_L nonlinearity

### What Is New

First-principles derivation of the "with BH mass" dark siren likelihood, extending the standard d_L-only formulation to include redshifted mass as a fourth observable, with all Jacobians made explicit.

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
| Keep analytic M_z marginalization (15b49a3) | Correct physics, replaces delta-function approximation | Confirmed by user |
| Derive from d_L-only literature + extend | User built "with BH mass" as natural extension; needs first-principles validation | — Pending |
| P_det out of scope | Already handled in Phase 11.1 | Confirmed by user |

---

_Last updated: 2026-03-30 after milestone v1.2.1 initialization_
