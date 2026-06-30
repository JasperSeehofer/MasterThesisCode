# HANDOFF — σ_z / σ_M precision-forecast heatmap (LISA EMRI dark-siren H₀)

**Created 2026-06-30. Start a fresh session for this — it needs new simulation/inference sweeps.**
Branch suggestion: `study/sigma-z-m-precision-forecast` off `main` (after PR #17 merges) or off the
current `physics/photoz-joint-normalisation` tip.

---

## 1. Where we are (why this study)

The in-catalogue **photometric** dark-siren channel is now demonstrated **information-starved** at
GLADE's regime (σ_z ≈ 0.035, σ_z/z ≈ 0.7 at z ≈ 0.05): the host photo-z is ~17× the GW redshift
precision, so no normalization recovers a peaked H₀. Full record:
- `.planning/derivation-photoz-incatalog/INCREMENT3-DSM-VERDICT.md` (the decisive result)
- `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md` and `docs/PIPELINE_BUGS_REPORT.md` (session reports)
- `docs/H0_BIAS_RESOLUTION.md` (cumulative catalog, photo-z chapter appended)
- `scripts/bridge_closure/BRIDGE-FINDINGS.md` (rung ladder), `_rungI_verify_B.py` (the arbiter closure)

Agreed honest deliverable (decided 2026-06-30 with the user): present **(a)** the honest result for the
*current* GLADE catalog (photometric → starved; spectroscopic-host forecast recovers h≈0.725), and
**(b)** a **forecast** over hypothetical redshift/mass precisions — a **heatmap of H₀ precision
σ(H₀) as a function of (σ_z, σ_M)** — so one can read off *what measurement precision is needed for a
promising LISA dark-siren H₀*. Hypothesis to test: the **with-BH-mass (2-D / 4-D) channel converges
faster** (tolerates larger σ_z) than the without-BH-mass (1-D) channel — a nice result for the paper.

## 2. The goal figure

A 2-D heatmap (and/or contour) with axes **σ_z** (host redshift error) and **σ_M** (host BH-mass error),
colour = **recovered H₀ precision** σ(H₀)/H₀ (posterior width) at fixed event count. Two panels:
**without_bh_mass (1-D)** and **with_bh_mass (2-D)**, plus optionally their ratio. Overlay: the GLADE
photometric operating point (σ_z≈0.035) and the spectroscopic point (σ_z≈0.0017); a contour at a
"promising" target precision (e.g. σ(H₀)/H₀ = 5% and 2%). Output: `docs/figures/sigma_z_sigma_M_precision_heatmap.png`.

## 3. Method (recommended)

Use the **self-consistent closure** as the engine (it is unbiased by construction, so the posterior
WIDTH is a clean measure of information content — exactly what a forecast needs; the bias question is
already settled separately). Two viable routes:

- **Route A (fast, recommended first): extend the bridge `rung_I` closure.**
  `scripts/bridge_closure/rung_I_prior_domination.py` / `_rungI_verify_B.py` already sweep σ_z in a
  clean no-sky closure. TODO to make it a forecast:
  1. Report the **posterior width** σ(h) (already computable from `logpost`/`hs`: `sqrt(Σ(h-E[h])²P)`),
     not just the MAP. Average over seeds (the MAP/width are noisy at small N — see §4).
  2. Add the **σ_M (BH-mass) axis**: the current closure is `with_bh_mass=False`. Build the 4-D /
     with-mass closure — inject a host BH mass, give it a measurement error σ_M, and add the mass
     likelihood term to the per-event factor (mirrors the production 2-D channel; see
     `bayesian_statistics.py` single_host_likelihood with-mass path and the §3.15 H3 fix in the story
     for the source-frame-vs-observer-frame mass convention — get that right).
  3. Use an **unbiased normalization** in the closure (the spec-z-limit / dV_c-once `p_red`; do NOT
     rely on `D_sm`, which de-biases-but-doesn't-peak at large σ_z — fine for the small-σ_z forecast
     region but it muddies the large-σ_z width). Simplest: the closure is self-consistent so the
     standard form is unbiased at small σ_z; verify the width is meaningful where σ_z is small.
  4. Sweep a grid, e.g. σ_z ∈ {0.0005 … 0.05} (log-spaced, ~8 pts) × σ_M ∈ {0.05 … 1.0}·M (~6 pts),
     multi-seed (≥8) per cell, fixed N_events. Record σ(h) per cell → heatmap.

- **Route B (paper-grade, slower): the production pipeline with synthetic spec-z-style hosts.**
  Re-simulate CRBs and run `--evaluate` with injected (σ_z, σ_M) per a grid. Sky-faithful, but
  expensive (cluster). Do this only after Route A maps the landscape and identifies the interesting
  region. Fold in the frame fix (#15) and dV_c-once interpretation (`CATALOG-INTERPRETATION.md`).

## 4. Pitfalls (learned this session — do not repeat)

- **The MAP is a poor estimator for broad posteriors.** Use σ(h) (posterior width) and E[h] with the
  caveat that the grid [0.60,0.87] midpoint (0.735) ≈ truth, so a flat posterior gives E[h]≈0.73
  *trivially*. For the forecast, **width is the headline**; verify unbiasedness via multi-seed
  centering separately. Consider widening the H₀ grid (e.g. [0.50,0.95]) so "flat" is distinguishable.
- **Multi-seed always.** Single-seed numbers mislead (we saw a single-seed "de-rail" that vanished
  under multi-seed). The scatter at small N is large (std ~0.04 on the MAP at n_ev~250).
- **Scaling:** the MAP *location* is N_obs-independent (numerator and denominator both ∝ N_obs); the
  *width* ∝ 1/√N_obs. So pick a realistic N_obs (LISA EMRI yield over the mission — get the rate from
  `master_thesis_code/emri_rate.py` / `Model1CrossCheck`) and report width at that N, OR report
  width×√N_obs (the per-event information) and scale.
- **Catalogue size:** D_sm-type levers are edge-galaxy-dominated; if you use a real-catalogue route,
  n_gal must sample the z~0.15–0.25 edge (n_gal=12k is enough for the lever but under-samples events).
- **Bridge is no-sky.** Good for the relative (σ_z,σ_M) forecast; for absolute paper numbers, validate
  one cell against the sky-faithful production pipeline.

## 5. Done this session (context for the next one)

- Frame bug fixed: `[PHYSICS]` use CMB-frame z_cmb (col 28) not heliocentric (col 27). **PR #17** open
  to `main`; **issue #15** (frame), **issue #16** (host PV treatment). Needs reduced-catalogue
  regeneration from `GLADE+.txt` to take effect in a run.
- Likelihood-vs-posterior fork settled (likelihood; dV_c applied once); dV_c branch inconsistency +
  num/denom photo-z smearing asymmetry documented (`CATALOG-INTERPRETATION.md`).
- Bridge prototype committed: `_rungI_verify_B.py` `hierarchical_shared_latent` flag (the `D_sm` global
  photo-z-smeared selection), `5ef8c6e`.
- Reports + flowchart: `docs/PIPELINE_BUGS_REPORT.md`, `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md`,
  `docs/PIPELINE_FLOWCHART.md`; story chapter in `docs/H0_BIAS_RESOLUTION.md`.

## 6. First steps next session

1. Read `INCREMENT3-DSM-VERDICT.md` + `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md` for the full picture.
2. Build the with-BH-mass closure (σ_M axis) in the bridge; add posterior-width reporting.
3. Run the (σ_z, σ_M) sweep (Route A), multi-seed; produce the heatmap; check the 1-D vs 2-D
   convergence hypothesis.
4. Identify the (σ_z, σ_M) contour for a target σ(H₀)/H₀; relate to real survey capabilities
   (spec-z surveys, future photo-z, EMRI BH-mass Fisher errors from the simulation CRBs).
5. If promising, validate one cell with the sky-faithful production pipeline (Route B).
