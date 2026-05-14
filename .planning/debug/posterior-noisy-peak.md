---
slug: posterior-noisy-peak
status: partial-fix-landed-second-mechanism-suspected
trigger: "Why is the posterior so noisy around the peak? I would expect a continuous smooth peak, not the up and down spikes we actually see. please investigate."
created: 2026-05-11T18:14:00Z
updated: 2026-05-14T13:30:00Z
---

# Posterior noisy / spiky around peak

## Symptoms

**Expected behaviour:** Smooth, continuous unimodal peak in the combined H₀ posterior near h_true=0.73. Visual signature: a single smooth bell-like curve when plotted on the 63-point production grid (Δh=0.001 dense core in [0.710, 0.750]).

**Actual behaviour:** Posterior shows up-and-down spikes around the peak rather than a smooth curve. User noticed in the refreshed Phase 48 figures (commits `5eb438e` + `b8500c5`, 2026-05-11).

**Error messages:** None (no crash, no warning — purely a quality-of-result issue).

**Timeline:** Surfaced after the Phase 48 production sweep (63-pt non-uniform grid at h=0.73 on phase46-merged 1473 events, jobs `4271862` + `4344777`). Previous Phase 47 21-point grids were too coarse to reveal noise at this resolution. The Δh=0.001 dense core now exposes fine-scale structure that was hidden when Δh=0.005.

**Reproduction:** Open any of:
- `paper/figures/h0_posterior_comparison.pdf` (combined posterior, both channels)
- `interactive/combined_posterior.html` (Plotly version with hover values per h-grid point)
- `results/figures_seed200/fig01_h0_posterior_combined.pdf`
- Or load directly: `simulations/cluster_run_production_h0p73_20260506/combined_posterior.json` and plot posterior vs h_values.

## Context — relevant code paths

- Joint posterior assembled in `master_thesis_code/bayesian_inference/posterior_combination.py` (`combine_log_space`, `compute_combined_posterior`).
- Per-event likelihoods come from `bayesian_inference/bayesian_statistics.py:single_host_likelihood`.
- p_det denominator and numerator queries in same file (Phase 47 H3 fix changed numerator query to hypothesis-frame; commit `f01595c`).
- p_det grid built from injection KDE in `bayesian_inference/simulation_detection_probability.py` (Phase 45 bridge fix `2b33cad`).
- D(h) selection-function normalization: post-Tier-3 the outer −N log D(h) was removed (commit `6754ddb` 2026-05-04). Combined posterior is Σ log L_i, no outer factor.
- Production posteriors written per-h to `posteriors/h_*.json` and `posteriors_with_bh_mass/h_*.json` (63 files each).

## Hypotheses (initial brainstorm — to be tested by debugger)

H1. **p_det grid stochasticity propagates to per-event likelihood**: the 1D and 2D p_det grids are KDE-based on a finite injection campaign (Phase 46 injections). The bridge-fix uses anchors at `(dl_min, p_edge)`. Per-h-trial, the p_det evaluation depends on the (h-dependent) d_L(z) and M_z(z) mapping; tiny differences in h shift the d_L grid mapping and pick up KDE first-bin noise (logged warnings during --combine: "P_det 1D grid first bin [0, 0.04xx Gpc) has only 7 injections ... may be noisy"). This could create per-h-trial likelihood fluctuations that add up to spiky posterior structure.

H2. **2D grid boundary-crossing discontinuities re-emerged at finer Δh**: pre-bridge-fix, ~57 events crossed the grid boundary at h-trial transitions (Δh=0.005 produced steps). Post-bridge-fix the principled extrapolation gave a smoother behaviour at Δh=0.005 (verified Phase 47 R1). At Δh=0.001, smaller fractions of events may still cross *interpolation knots* inside the grid — RegularGridInterpolator gives C0 only (linear), so the gradient is discontinuous at grid lines. Could produce slope changes that look like "spikes" when stacked across 1473 events.

H3. **Floating-point parabolic-refine instability near MAP**: parabolic refine in `parabolic_refine` (test_24 helper) uses 3-point local fit at the discrete argmax. If neighbouring points are nearly equal (which they should be at Δh=0.001 ≈ σ_boot/4), parabolic fit becomes ill-conditioned and the refined MAP jumps. *This affects MAP only, not the posterior values directly — so this probably doesn't explain spikes in the posterior plot itself, but worth elimination.*

H4. **Per-event posterior write rounding**: per-h posteriors are written to JSON. If `single_host_likelihood` is being saved with reduced precision (e.g., float→str with limited digits), reconstruction of log-posterior introduces noise. Unlikely (JSON is high-precision by default) but quick to check.

H5. **Combined posterior plotting uses linear interpolation between non-uniform grid points**: the 63-point grid has Δh=0.001 in the core, Δh=0.010 in the wings. If the plotter assumes uniform spacing or applies a smoother only inside the core, the visualisation could exaggerate small fluctuations. *This is a plotting artefact hypothesis, not a physics one — but it matches "the data is fine, the rendering is spiky".*

H6. **Bootstrap signature aliasing**: σ_boot is computed by resampling events with replacement. If the bootstrap distribution of (MAP_h_grid_value) is concentrated on a few specific h-grid points (because event-resample distributions tend to pick the same argmax), the *plotted posterior* (which is the un-bootstrapped Σ log L_i, not the bootstrap mean) should NOT show this — but worth confirming the plot is showing the right quantity.

## Current Focus

hypothesis: H1+H2 (combination — see Evidence). The dominant mechanism is the H1 family: the `RegularGridInterpolator` is rebuilt per-h on a histogram whose bin edges depend on `np.max(d_L_target(h)) * 1.1`. As h changes by 0.001, the bin centres drift smoothly but individual injections also cross integer-count bin boundaries, producing per-bin probability `detected/total` jumps that propagate identically into every event's per-h likelihood.

test: (DONE) Plot raw posterior (log-likelihood sum, NOT exponentiated) on the 63-point grid and compute first/second differences in log-L. Smooth curve → small second difference; spiky → large oscillations. Cross-check against the per-event log-likelihood matrix to see whether the spikes are: (a) a single rogue event contributing everything, (b) coherent across many events, or (c) noise washing in stochastically.

expecting: The second difference of log-Σ L_i should reveal whether the spikes have a coherent pattern (suggesting a code path issue) or are uncorrelated (suggesting statistical noise being amplified by visualisation).

next_action: Decide on fix path. Two principled options:
(F1) **Static / h-independent grid edges**: build a single `dl_edges = np.linspace(0, DL_GLOBAL_MAX, N+1)` shared across all h (where `DL_GLOBAL_MAX` is the max across all SNR-rescaled injections over the h-grid). Eliminates per-h bin-edge drift entirely. Cheap. Likely the correct first move.
(F2) **Reduce KDE-noise propagation**: increase injection count (deeper Phase 46 augmentation), use kernel density estimate with smoothed support instead of histogram, or apply a smoothing kernel (Gaussian, 1-bin σ) on the histogram before constructing the interpolator. More involved.

Both are *physics-adjacent* (touching `simulation_detection_probability.py`, which is on the physics-change protocol list per CLAUDE.md). Requires `/physics-change` gate.

## Evidence

- **timestamp: 2026-05-11T20:00:00Z — direct inspection of combined_posterior.json**
  - File: `simulations/cluster_run_production_h0p73_20260506/combined_posterior.json`
  - Posterior values in dense core 0.726–0.745 oscillate 5–10× between adjacent Δh=0.001 bins. Selected: `0.726→0.0143, 0.727→0.0363, 0.728→0.0948, 0.729→0.0542, 0.730→0.0228, 0.731→0.0587, 0.732→0.0236, 0.733→0.2523 (reported MAP), 0.734→0.2092, 0.735→0.0868, ... 0.737→0.0096, 0.738→0.0029, 0.739→0.0203, 0.740→0.0140, 0.741→0.0305, 0.742→0.0063`.
  - Conclusion: spikes are real in the saved JSON; not a plotting artefact. Hypothesis H5 ELIMINATED.

- **timestamp: 2026-05-11T20:05:00Z — joint log-L Δ² analysis (/tmp/diagnose_spikes.py)**
  - Joint Σ_i log L_i computed directly from per-event JSONs (1473 events × 63 h-bins). Δ² log-L in dense core ranges between −2.6 and +3.3 — clear sawtooth pattern.
  - **No single event** is responsible: top-10 per-h-pair contributors all move smoothly (≤0.05 per Δh=0.001). Mean |Δlog L| per event ≈ 3e-3, std ≈ 5e-3 — orders of magnitude smaller than the joint fluctuation.
  - **No "spike-like events"** exist (defined as: at some h, log L > 2 above both neighbours). Zero events match. Hypothesis "one rogue event" ELIMINATED.

- **timestamp: 2026-05-11T20:10:00Z — correlated residual analysis (/tmp/diagnose_correlated.py)**
  - For each event, compute residual = log L − Savitzky-Golay-smoothed(log L) across the dense core.
  - Cross-event correlation of residuals (sample of 50 events): mean 0.17, but distribution bimodal (41% > 0.5; 32% < −0.3). This means events are split into two groups, each strongly synchronised across the dense core.
  - At specific h pairs:
    - `corr(res@0.731, res@0.737) = +0.997`
    - `corr(res@0.732, res@0.743) = −0.994`
    - `corr(res@0.731, res@0.733) = −0.994`
    - `corr(res@0.732, res@0.738) = +0.964`
  - Σ_events residual (= joint log-L deviation from smooth) reproduces every spike: −1.28 at h=0.731, +1.02 at h=0.732, +1.08 at h=0.733, −1.49 at h=0.737, +1.17 at h=0.740, etc.
  - **Conclusion**: spikes are driven by a SHARED, h-dependent noise source affecting most events identically (with smaller subset anti-correlated). Hypotheses H3, H4, H6 ELIMINATED.

- **timestamp: 2026-05-11T20:15:00Z — D(h) selection function does NOT fully explain (/tmp/diagnose_Dh.py)**
  - D(h) from combined_posterior.json shows Δ log D oscillating between −4e-3 and +8e-3 in dense core — itself spiky. Std(residual log D) = 1.9e-3, so N · σ ≈ 2.8.
  - But correlation of per-event residual with −residual(log D(h)) is only 0.16 mean / 0.34 median; predicted N · residual(log D(h)) is *anti-correlated* with observed Σ residual at most h bins (ratio ≈ −0.3 typical). So D(h) is itself noisy because of the same root cause, but it is NOT the sole channel through which noise enters the per-event likelihoods.

- **timestamp: 2026-05-11T20:20:00Z — p_det(d_L=0.5, h) directly spikes (/tmp/diagnose_pdet_h.py)**
  - Loading `SimulationDetectionProbability` from `INJECTION_DATA_DIR` and evaluating `detection_probability_without_bh_mass_interpolated_zero_fill(d_L=0.5, h)` at h ∈ [0.710, 0.750, Δ=0.001]:
    - Smooth decay 0.00677 → 0.00619 (h: 0.710 → 0.718).
    - **Jump +5%** at h=0.719: 0.00619 → 0.00652.
    - Smooth decay 0.00652 → 0.00628 (h: 0.719 → 0.724).
    - **Jump +26%** at h=0.725: 0.00628 → 0.00794.
    - **Jump +16%** at h=0.743: 0.00659 → 0.00763.
    - **Jump +13%** at h=0.744: 0.00763 → 0.00860.
    - **Jump +12%** at h=0.747: 0.00831 → 0.00928.
  - Each "jump" coincides with an injection crossing a histogram bin boundary as `dl_max(h) = max(d_L_target(h)) * 1.1` shrinks with h (verified `/tmp/diagnose_dl_max.py`: dl_max decreases monotonically 0.14 % per Δh=0.001, edges drift smoothly).
  - The interpolated p_det value at fixed query d_L is therefore non-smooth in h because (a) the histogram bin centres drift continuously, but (b) when an injection crosses a bin, the integer `detected/total` ratio in that bin jumps by 1/N_bin — typically O(few %).

- **timestamp: 2026-05-11T20:25:00Z — synthetic-population test reproduces joint-log spikes (/tmp/diagnose_pdet_drives_spikes.py)**
  - Generated 1000 synthetic hosts at z ~ U(0.05, 0.5), computed `Σ_hosts log p_det(d_L(z; h), h)` across the dense core.
  - Result: jumps of +8 to +25 in log units at h=0.719, 0.725, 0.743, 0.744, 0.747 — same h-values where the production posterior shows spikes. Between bin-crossing events the curve is flat or slowly drifting.
  - **This closes the loop**: the source of the shared, correlated per-event log-L noise is `p_det` itself, evaluated against an h-rebuilt finite-injection histogram with linear interpolation. The fluctuations are real and structurally inevitable with the current grid construction.

## Eliminated

- **H3** (parabolic-refine instability): spikes are present in saved posterior JSON values themselves, so they pre-exist MAP refinement.
- **H4** (JSON write rounding): per-event likelihoods in JSON files are full float64 precision; the spikes appear in the raw sum, not in any rounded representation.
- **H5** (plotting artefact): direct posterior values in `combined_posterior.json` show the spikes (e.g. 0.733 → 0.2523 vs 0.732 → 0.0236 vs 0.731 → 0.0587). Not a rendering issue.
- **H6** (bootstrap aliasing): the plotted posterior is the non-bootstrap Σ_i log L_i; bootstrap doesn't enter the curve.
- **Single rogue event**: top-10 per-h-pair contributors all move smoothly; no event with a binary spike pattern exists.
- **D(h) alone**: D(h) is itself spiky (it inherits the same p_det noise), but Pearson correlation of per-event residual against −residual(log D) is only 0.34 (median) — D(h) contributes part of the noise but is not the dominant channel.

## Resolution

**Root cause**: The per-event likelihoods `L_i(h)` integrate `p_det(d_L, M_z; h)` from a `RegularGridInterpolator` whose grid is rebuilt for every h-value via:

```python
# simulation_detection_probability.py:404-405 (2D) and 540-541 (1D)
dl_max = float(np.max(dl_vals)) * 1.1
dl_edges = np.linspace(0, dl_max, self._dl_bins + 1)
```

`dl_vals` comes from `_rescale_snr(h)`, so `dl_max` is h-dependent. As h changes by 0.001:
1. Bin centres drift smoothly by ~0.14 % per step (no spikes from drift alone).
2. But the histogram counts `detected/total` per bin are quantized at integer resolution (`np.histogram2d` / `np.histogram`). When a single injection crosses a bin boundary as `dl_max(h)` shrinks, the integer `detected/total` ratio in the involved bins jumps by O(1/N_bin) — typically 5–25% jumps for low-d_L bins where N_total per bin is ~7–50.
3. These p_det jumps at the interpolator's grid level propagate **identically** into every event's `L_comp = num/D` and `L_cat` integrals — because all 1473 events query the same per-h interpolator. The shared noise then sums coherently into Σ_i log L_i, producing ±1–3 jumps in the joint log-likelihood across consecutive Δh=0.001 steps.

This explains every observed signature:
- Why a single event is not responsible (all events are nudged identically).
- Why per-event residuals correlate near ±1 across distant h-bins (the same handful of bin-crossing injections drive multiple h-bins).
- Why the spikes only emerged at Δh=0.001 (at Δh=0.005, jumps average out within one step).
- Why warnings "first bin [0, 0.05 Gpc) has only 7 injections" are relevant: the LOW-d_L bins, with smallest count, contribute the largest relative fluctuations.

This is a **physics-relevant modelling artefact** (the selection-function p_det is itself noisy as a function of h). It is NOT a plotting bug. Touching `simulation_detection_probability.py` invokes the **Physics Change Protocol** (per CLAUDE.md "physics-change trigger files").

**Fix direction (proposed; pending `/physics-change`)**:

**F1 (preferred first step): use h-independent bin edges**.
Replace lines 404–405 and 540–541 with edges built once from the GLOBAL max of `d_L_target` over the entire h-grid (or simply a fixed physical maximum, e.g. d_L=10 Gpc which trivially covers all hosts). This eliminates per-h bin-edge drift entirely. Effect: per-h grids still differ in their detected-count distribution (because `_rescale_snr(h)` produces a different SNR for each injection per h), but injections no longer hop bins. The remaining h-dependence is then smooth, governed by the SNR threshold sweeping the injection set continuously.

Caveats / open questions for `/physics-change`:
- Does fixing edges introduce systematic bias if `dl_max(h)` would otherwise have undercovered? Answer: no — we'd extend, not contract. Coverage strictly increases.
- Is it equivalent to the post-fix scheme at the limit Δh→0? Answer: yes (and the limit becomes well-defined, which it currently is not).

**F2 (independent, additive option): apply mild smoothing to the histogram before interpolation**.
After computing `p_det_grid = detected/total`, convolve with a small Gaussian kernel (σ ≈ 0.5 bin) along the d_L axis. This trades a tiny systematic bias (smoothing across bin widths ~50 Mpc) against a large variance reduction. Defensible if validated against an analytical/semi-analytic p_det reference.

**F3 (longer-term): increase injection density at low d_L** (the 7-per-first-bin warning is the loud signal). Already partially addressed in Phase 46.

A regression test against the current 1473-event production output will need to be revised once F1 lands (per-h likelihood numbers will change). Add a NEW regression test that asserts:
- Σ |Δ² log Σ L_i| in the dense core is below a threshold (e.g. < 0.5 in joint log units).
- p_det(d_L=fixed, h) varies smoothly: max |Δp_det / Δh|·Δh < 1e-3 per Δh=0.001 step.

**Status: AWAITING USER DECISION** — proceeding to fix requires `/physics-change` engagement. Surfacing to orchestrator.

---

## Follow-up — F1 LANDED but spikes NOT eliminated (2026-05-14)

F1 (h-stable `dl_edges`) landed in `[PHYSICS]` commit `87ea7a8` on 2026-05-12; cluster validation job `4662333` completed 2026-05-13 (14 tasks × ~21 min wall on cpu_il, archive of pre-F1 posteriors preserved at `archive/production_h0.73_20260512_175829/`). Re-ran `test_28` analyzer + direct inspection of `combined_posterior{,_with_bh_mass}.json`.

### Outcome: posterior shape changed, spikes NOT gone

| Metric | Pre-F1 (Phase 48) | Post-F1 PARTIAL (Phase 49) | Notes |
|---|---|---|---|
| 1D continuous MAP | 0.7324 | **0.7378** (+0.0054) | Further from truth |
| 2D continuous MAP | 0.7322 | **0.7378** (+0.0056) | Same |
| 1D bias / z_boot | +0.0024 / +1.16σ | **+0.0078 / +2.02σ** | Worse |
| 2D bias / z_boot | +0.0022 / +0.97σ | **+0.0078 / +229σ** | σ_boot collapsed to ≈0 (every bootstrap landed on same h=0.738) |
| 1D σ_boot | 0.0021 | 0.0039 | Wider, more honest |
| Max adjacent-bin ratio in dense core [0.725, 0.745] (1D) | ~11× (raw posterior values pre-fix) | **16×** | Comparable |
| Max adjacent-bin ratio in dense core (2D) | ~10× | **51×** (0.738→0.739 drops from 0.577 to 0.018) | Worse |

**Qualitative shape change**: the rising flank 0.730 → 0.738 is now mostly smooth and monotonic, but at the peak h=0.738 the posterior has a single huge discontinuity (1D: 0.148 → 0.009 across one Δh=0.001 step; 2D: 0.577 → 0.018, a 32× drop). The noise pattern is *different* from pre-F1 but the joint posterior is **not the smooth single-mode curve the user expects**.

### Second mechanism: SNR-threshold integer crossings

F1 addressed only one of two mechanisms by which finite-injection-sample noise enters `p_det(d_L; h)`. The remaining mechanism, **almost certainly dominant in the post-F1 spike pattern**:

For each injection at fixed `(z_inj, SNR_raw, h_inj)`:
- The rescaled SNR at trial cosmology h is `SNR(h) = SNR_raw · d_L(z_inj, h_inj) / d_L(z_inj, h)`.
- As h shifts by Δh=0.001, `d_L(z_inj, h)` shifts smoothly by ~0.14% per step.
- An injection whose `SNR(h)` is near the threshold (SNR=20) crosses the threshold at some specific h*. At h < h* it counts as "detected" in its bin; at h > h* it counts as "undetected".
- **Each threshold-crossing flips its bin's `detected/total` ratio by 1/N_bin** — typically 5–25% jumps for low-count bins.

This mechanism is **independent of bin edges** (F1's fix). It's a property of the histogram estimator itself: integer counts of "events passing a threshold" at finite resolution.

The same coherent-summation pathology applies: all 1473 events query the same per-h `p_det` interpolator, so a single threshold-crossing nudges every event's `L_i(h)` identically, summing into `Σ log L_i` as a single ±1–3 jump across that h-step.

### Why F1 was still the right first move

- F1 implements the *minimum form* of the consensus practice (h-stable support), per the literature audit at `.planning/debug/F1_literature_audit.md`. It removes one mechanism cleanly; the source change is small, well-tested, and defensible on its own.
- F1 is **necessary but not sufficient**. The full consensus practice (Farr 2019 / ICAROGW / gwcosmo) is **fixed injection set + analytic per-Λ reweighting**, which eliminates *both* mechanisms (bin edges AND threshold crossings) and matches what production pipelines actually do.
- The literature audit already flagged this — Caveat 3: "F1 is the minimum form vs. the Farr 2019 fixed-injection + reweighting form that production pipelines use."

### Proposed F4 — Farr 2019 reweighting

Replace the histogram-binned estimator with a per-injection-weight estimator:

```python
# For trial h, instead of histogramming detected/total per bin:
# 1. Compute per-injection weight w_i(h) accounting for the trial cosmology
#    (typically: p_pop(theta_i | h) / p_inj(theta_i), where theta_i are the
#    injection parameters and p_inj is the injection prior).
# 2. p_det at any query x is a smooth weighted KDE/GP evaluation over the
#    fixed injection set, NOT an h-rebuilt histogram.
# Standard form: Farr 2019 Eq. 2-7; n_eff > 4*N_obs condition for accuracy.
```

This is a non-trivial refactor of `simulation_detection_probability.py`: it removes `RegularGridInterpolator` + histogram estimator and replaces it with a per-injection-weight evaluation. The injection generation pipeline (`scripts/inject_*.py`) must also be reviewed to ensure `p_inj` is recoverable from the injection metadata (it should be — the campaign uses a known prior).

Cost: 1–2 day refactor + cluster re-validation. Benefit: consensus-grade smoothness; matches what reviewers will expect; eliminates the structural noise floor entirely.

### Handoff for next session

1. **Confirm SNR-threshold mechanism with a direct probe**: re-run `/tmp/diagnose_pdet_h.py`-style script (now F1 is landed) — does `p_det(d_L=0.5, h)` still show ±5–25% jumps at specific h-values? Where are those h-values? Cross-check against the injection set: which specific injections have `SNR(h) ≈ 20` at those h-values?

2. **Decision point on F4**:
   - (a) Implement Farr 2019 reweighting (full consensus form).
   - (b) Apply a coarser-grain fix: smooth the histogram p_det with a small Gaussian kernel before interpolation (F2 from the original session). Cheap, partially effective, less principled. Worth considering as an intermediate if F4 is too costly.
   - (c) Increase injection density at low d_L / near-threshold (F3): smaller per-bin integer increments → smaller jumps. Cheap on cluster wallclock; reduces but doesn't eliminate the floor.

3. **DATA_INVENTORY tier**: pre-F1 posteriors are stale (archived on cluster); post-F1 posteriors at `simulations/cluster_run_production_h0p73_20260506/` are PARTIAL — not paper-grade. Will need re-evaluation after F4.

4. **Pattern update**: the wiki entry `[[scientific-computing-validation#hyperparameter-dependent-discretization-in-monte-carlo-selection-functions-produces-coherent-noise]]` was filed treating F1 as the resolution. After F4, the pattern can be promoted to verified; until then the entry needs a note that F1 is necessary-but-not-sufficient and the SNR-threshold mechanism is a sibling case.

### Files

- `scripts/bias_investigation/outputs/phase46_merged/h3_production_sweep_verdict.json` — Phase 48 pre-F1 reference (restored from git).
- `scripts/bias_investigation/outputs/phase46_merged/F1_post_fix_verdict_PARTIAL.json` — Phase 49 post-F1 verdict (this run).
- `simulations/cluster_run_production_h0p73_20260506/posteriors{,_with_bh_mass}/` — post-F1 per-h posteriors (local).
- `simulations/cluster_run_production_h0p73_20260506/combined_posterior{,_with_bh_mass}.json` — combined post-F1 posteriors (showing the 0.738 peak + adjacent discontinuity).
- Cluster archive of pre-F1 posteriors: `archive/production_h0.73_20260512_175829/posteriors{,_with_bh_mass}/` (preserved by the `ARCHIVE_OLD=yes` re-run).
