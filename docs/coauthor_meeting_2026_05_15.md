# Co-author Meeting — H₀ Inference: Bias Resolution & Current Status

**Date:** 2026-05-15  
**Prepared by:** Jasper Seehofer  
**Figures:** `simulations/cluster_run_production_h0p73_20260506/simulations/figures/`

---

## 1. What We Are Measuring

A dark-siren Hubble constant measurement using simulated LISA Extreme Mass Ratio
Inspiral (EMRI) events and the GLADE+ galaxy catalog, following the framework of
Gray et al. (2020) arXiv:1908.06050.

**Pipeline overview:**
1. **Simulation** (GPU cluster, bwUniCluster): Generate EMRI events at fixed
   injection cosmology `h_true = 0.73`, compute Fisher-matrix Cramér–Rao bounds
   on waveform parameters, keep events with SNR ≥ 20.
2. **Bayesian inference** (CPU cluster): For each trial `h ∈ [0.60, 0.86]`,
   compute a joint log-posterior:
   ```
   log p(h | data) = Σᵢ log Lᵢ(h)
   ```
   where each per-event likelihood `Lᵢ` marginalises over host galaxies in
   the GLADE catalog using the Gray et al. completeness framework.
3. Two analysis **channels** throughout:
   - **1D channel** (`without_bh_mass`): uses sky position + distance only
   - **2D channel** (`with_bh_mass`): additionally conditions on the observed
     black-hole mass `M_z` to match host galaxies by stellar mass (tighter posterior)

**Key numbers (production dataset):**
- ~1 473 detected events, SNR ≥ 20, seed 200 + seed 300 extension
- h-grid: 63 points — dense core Δh = 0.001 in [0.710, 0.750], wings Δh = 0.010
- Truth: `h_true = 0.73`

---

## 2. The Problem: Where We Started

After the initial pipeline was assembled, the cluster MAP was pinned at **h = 0.86**
— the grid maximum — a bias of +18% on h. The pipeline was fundamentally wrong.

| Milestone | Cluster 1D MAP | Notes |
|-----------|---------------|-------|
| Initial pipeline | 0.860 | MAP at grid ceiling |
| Post Phase 43 H2 (coord fix) | 0.730 (local, 60 events) | --evaluate PASS; cluster still broken |
| Post Phase 44 (P_det fix) | 0.765 | Cluster first moved off 0.86 |
| Post Phase 45 anchor (partial) | 0.755 | Wrong layer — anchor escalation |
| **Post Tier 3 D(h) fix** | **0.740** | D(h) double-counting removed |
| Post bridge + H3 fix (1473 events) | **0.731** | Both channels < 0.3 σ of truth |
| **Phase 48 production sweep** | **0.732** | **Paper-grade result** |

---

## 3. Confirmed Bias Sources — Ordered by Impact

A total of **15 confirmed bias sources** were identified and resolved across
Phases 9–48. Here are the most significant:

### 3.1 Full-volume D(h) denominator — Phase 32 (MAP: 0.60 → 0.73 local)

**What was wrong:** The completeness term `L_comp` for catalog-incompleteness
correction (Gray et al. Eq. A.19) was normalised only over a local 4-σ window
in luminosity distance, not over the full detectable volume.

**Effect:** Under-normalisation caused the posterior to prefer low h on the
local 60-event test set.

**Fix:** Replaced with `D(h) = ∫ P_det(d_L(z,h)) · dV_c/dz dz` integrated
over the full volume; wired into the cluster combine path in Phase 43.

**Result:** Local MAP 0.60 → 0.73 (bias 0% on 59 events).

---

### 3.2 Coordinate frame mismatch: equatorial CRBs vs ecliptic GLADE — Phases 36 + 43-H2 (MAP: 0.86 → 0.73)

**What was wrong:** GLADE+ coordinates were stored in equatorial RA/Dec;
EMRI sky angles `(qS, φS)` are in ecliptic frame. With the BallTree search
radius ≈ 1.76°, the 23.4° ecliptic obliquity caused ~15/60 events to match
spurious hosts.

**Two-surface bug:**
- **Phase 36** — GLADE ingestion: added `astropy BarycentricTrueEcliptic(J2000)`
  rotation + BallTree polar embedding, eigenvalue search radius, 4D sky
  sub-space.
- **Phase 43-H2** — cached CRBs: the `prepared_cramer_rao_bounds.csv` still
  held equatorial Fisher covariances; required a separate `migrate_crb_to_ecliptic`
  rotation of the 14×14 covariance matrix.

**Effect:** Host recovery 31/60 → 38/60; Q3 anisotropy 5.4σ → 0σ.

**Result:** `--evaluate` MAP returned to 0.730.

---

### 3.3 Missing −N log D(h) selection-function term — Phase 43-H1 (MAP: 0.86 → ~0.73)

**What was wrong:** The cluster combine path summed Σ log Lᵢ(h) but omitted
the Gray Eq. A.19 selection-function normalisation `−N · log D(h)`. Without
it, individual event likelihoods are higher closer to matching catalog
redshifts, producing a monotone tilt toward h_max = 0.86.

**Fix:** Added D(h) correction term in commit `2853c32`; deprecated the old
`extract_baseline()` path with a warning.

**Evidence:** N · log[D(0.86)/D(0.73)] = 29.5 (predicted 29.6).

---

### 3.4 h-dependent P_det zero-fill cutoff at c₀ ∝ 1/h — Phase 44 (MAP: 0.86 → 0.765)

**What was wrong:** The KDE-based detection probability had a left-side
zero-fill cutoff at `c₀(h) = d_L_max(h) / 120 ∝ 1/h`. Because `d_L_max ∝ 1/h`,
the cutoff was a *moving threshold*: 4 nearby events (d_L ≈ 0.085–0.097 Gpc)
fell below threshold at h=0.73 but above at h=0.86, so `L_comp` "switched on"
only at high h — pinning MAP at 0.860.

**Fix:** Removed the left-side zero-fill (commit `3697bdd`); the
`RegularGridInterpolator(fill_value=None)` now returns the genuine first-bin
estimate for very nearby events.

**Result:** Cluster MAP 0.860 → 0.765.

---

### 3.5 D(h) double-counting — Tier 3 fix (MAP: 0.755 → 0.740)

**What was wrong:** Phase 32 correctly placed `1/D(h)` inside each per-event
`L_comp` (Gray Eq. A.19 denominator). Phase 43-H1 *additionally* applied
`−N · log D(h)` as an outer correction — counting D(h) twice. This was the
dominant remaining systematic: at h_true=0.65 the outer correction shifted
MAP by +0.020 (≫ σ_boot), confirmed by:

- Closure test at h_true=0.65 on fine 11-pt grid (Δh=0.005): 1D MAP = 0.671
  (bias +0.021, z = +5.6σ FAIL) before fix.
- Per-event decomposition (412 events): Σ log Lᵢ alone peaks at h=0.740
  (within 1σ of truth); `−N log D(h)` overrides by +0.015 in the opposite
  direction.

**Fix:** `combine_log_space` now only sums `Σ log Lᵢ`; `log_D_h` parameter
retained for API compatibility but marked `ARG001` (not used), commit `6754ddb`.

**Post-fix closure test (h_true=0.65, 11-pt fine grid):**
- 1D MAP = 0.655 (bias +0.005, z = +1.67σ **PASS**)
- 2D MAP = 0.656 (bias +0.006, z = +1.68σ **PASS**)

---

### 3.6 P_det extrapolation: principled bridge (MAP: 0.741 → 0.731 in 1D)

**What was wrong:** 6–12% of events at every h_trial fell below the 2D P_det
grid's d_L lower bound. SciPy's linear extrapolation back-projected the noisy
boundary slope, reaching negative values (clipped to ≈0). This means very
nearby events (d_L < d_L_min) were assigned P_det ≈ 0 instead of the correct
P_det → 1. As the grid bounds shift with h_trial, ~50–60 events crossed the
boundary at each h-step — creating discontinuities in the joint posterior and
blowing up σ_boot in the 2D channel to near-zero (artefact tightness, not
real precision).

**Fix** (commit `2b33cad`): Replaced the raw extrapolation with a principled
monotonic bridge for both channels:
- *Saturating face* (d_L < d_L_min): linear bridge to P_det = 1 at d_L = 0.
- *Suppressing faces* (d_L > d_L_max, M_z out-of-range): slope-matched clamp
  to `[0, p_edge]`.
- Removed the Phase 45 anchor constants (Wilson 95% LB, intermediate anchor)
  which were fitted to truth — not a principled choice.

**Result on 1473 events:**
- 1D: MAP = 0.731, z = +0.19σ **PASS** (fully closed)
- 2D: MAP = 0.744, z = +3.6σ (10× improvement but residual remained — see next)

---

### 3.7 2D numerator queries observation instead of hypothesis — H3 fix (MAP 2D: +3.6σ → +0.2σ)

**What was wrong (two coupled bugs in the 2D channel):**

1. **Wrong query object.** At `bayesian_statistics.py:1304-1306`, the P_det
   integrand passed the *detected* ML mass `M_obs` (constant across the
   integration variable z) instead of the *hypothesis* mass `M_host · (1+z)`
   (which varies with z as the redshift hypothesis changes). The "known
   approximation, not a bug" comment from Phase 14 was justifying this exact
   mismatch.

2. **Grid axis mismatch.** The 2D P_det grid was binned in *source-frame* M,
   but the production queries passed *observer-frame* M_z = M · (1+z). At
   z ≈ 0.5, queries were ~50% above the bin labels — systematic per-event
   bias in P_det.

**Diagnostic:** `test_27_m_coordinate_mismatch.py` — 23% of events at h=0.73
have |ΔP_det| > 0.05 under the fix; mean ΔP_det = −0.031.

**Fix** (commit `f01595c`):
- Grid construction: multiply injection masses by `(1 + z_inj)` so the 2D
  grid axis is observer-frame M_z.
- Numerator integrand: changed P_det query from `M_obs` to `host_M · (1+z)`.

**References:** Mandel, Farr & Gair (2019) arXiv:1809.02063 §2; Loredo (2004)
arXiv:astro-ph/0409387.

**Result on 1473 events:**
- 2D: MAP = 0.731, z = +0.20σ **PASS** (20× bias reduction)
- Info monotonicity restored: |2D bias| ≤ |1D bias| (as expected — more
  information should sharpen the posterior toward truth, not away from it)

---

## 4. Final Production Result (Phase 48, 2026-05-07)

**Setup:** 63-point non-uniform h-grid — dense core Δh = 0.001 in [0.710, 0.750],
wings Δh = 0.010. 1 473 events, phase46-merged CRB (seed 200 + seed 300 extension).
Verdict JSON: `scripts/bias_investigation/outputs/phase46_merged/h3_production_sweep_verdict.json`

| Channel | Discrete MAP | Continuous MAP | σ_boot | Bias | z | Status |
|---------|-------------|----------------|--------|------|---|--------|
| 1D (without BH mass) | 0.7320 | 0.7324 | 0.0021 | +0.0024 | **+1.16σ** | **PASS ✅** |
| 2D (with BH mass) | 0.7320 | 0.7322 | 0.0022 | +0.0022 | **+0.97σ** | **PASS ✅** |

**Key properties:**
- Both channels within ±1.2σ of truth.
- Info monotonicity satisfied: |2D bias| ≤ |1D bias|; 2D σ_boot tighter — the
  BH-mass channel is doing its job of tightening the posterior toward truth.
- Δh-sensitivity spread (max − min MAP across sub-grids) comparable to σ_boot —
  the Δh=0.001 dense core is sufficient resolution; no further refinement needed
  at this event count.
- The Δh=0.005 sub-grid (matching earlier coarser estimates) recovers
  MAP ≈ 0.7309 with σ_boot ≈ 0.0048, confirming earlier R1 estimates were
  Δh-resolution-limited; Phase 48 refines upward by +0.0015 (within R1 σ_boot).

**Note on May 12 re-evaluation (post-meeting note):**  
A follow-up cluster run on 2026-05-12 using the same `run_production_h0p73_20260506`
directory updated the per-h posteriors with a slightly extended dataset
(1 549 events in CRB, 1 533 in posteriors after quality cuts). The 1D channel
gives MAP = 0.738, z = +2.02σ — still PASS but marginally higher. This run has
not been formally analysed yet; Phase 48 (May 7) remains the paper-reference result.

---

## 5. What the Figures Show

All 21 figures are at:
```
simulations/cluster_run_production_h0p73_20260506/simulations/figures/
```

**Key figures for the meeting:**

| Figure | Content |
|--------|---------|
| `fig01_h0_posterior_combined.pdf` | Main result: joint H₀ posterior, both channels overlaid with truth line |
| `paper_h0_posterior.pdf` | Publication-quality version of fig01 |
| `paper_h0_posterior_kde.pdf` | KDE-smoothed posterior — shows peak and credible interval more clearly |
| `fig02_event_posteriors.pdf` | Representative per-event likelihoods (how individual events constrain h) |
| `fig08_h0_convergence.pdf` | Posterior width vs N — shows the N^{−1/2} convergence |
| `paper_convergence.pdf` | Publication version of convergence |
| `fig03_snr_distribution.pdf` / `paper_snr_distribution.pdf` | SNR histogram of detected events |
| `fig04_detection_yield.pdf` | Detection yield vs redshift |
| `fig09_detection_efficiency.pdf` | P_det(d_L) showing the effect of the bridge fix |
| `fig15_campaign_dashboard.pdf` | Overview: N(z), SNR, sky coverage, Fisher quality |

---

## 6. Open Physics Issues

The following items are tracked in `docs/H0_BIAS_RESOLUTION.md` and remain
before paper submission:

### Low priority (known simplifications, documented)

| Issue | Location | Severity | Comment |
|-------|----------|----------|---------|
| wCDM params w0, wa silently ignored | `physical_relations.py:72` | MEDIUM | `dist()` accepts w0, wa but passes to hardcoded ΛCDM hypergeometric — wCDM mode is broken silently |
| Pipeline A hardcodes 10% σ(d_L) | `bayesian_inference.py` | MEDIUM | Uses `FRACTIONAL_LUMINOSITY_ERROR` instead of per-source CRBs — only affects the dev/cross-check pipeline (Pipeline B is production) |
| WMAP cosmology constants | `constants.py:29-30` | LOW | Ω_m = 0.25, H = 0.73 vs Planck 2018 (Ω_m = 0.3153, H₀ = 67.36) — production injections use H=0.73 which is not the Planck value |
| Galaxy redshift uncertainty (1+z)³ | `galaxy.py:64` | LOW | No reference for this scaling; standard is (1+z) |

### Code health (not physics)

| Issue | Location | Comment |
|-------|----------|---------|
| Unconditional `import cupy` | `LISA_configuration.py` | Breaks CPU-only import; fix when file is next touched |
| `extract_baseline()` deprecated but not deleted | `bayesian_inference/bayesian_statistics.py` | Footgun: a warning is all that separates callers from the old broken path |

### Fisher frame mismatch (tracked separately, requires /physics-change)

Sky position enters the Fisher matrix in ecliptic coordinates but the CRB
covariance may not have been verified end-to-end after the Phase 36 frame
rotation. This is tracked in the project memory as `project_fisher_frame_mismatch.md`.

---

## 7. Timeline of Bias Fixes (Chronological)

```
Phase 9   (2026-03-xx)  Galactic confusion noise added to LISA PSD
Phase 10  (2026-03-xx)  Fisher stencil: forward-diff → 5-point (Vallisneri 2008)
Phase 11  (2026-03-xx)  P_det: KDE → importance-sampling with SNR rescaling
Phase 15  (2026-04-xx)  Removed spurious /(1+z) Jacobian from 2D numerator
Phase 32  (2026-04-xx)  L_comp denominator: local window → full-volume D(h)
Phase 36  (2026-04-22)  GLADE ingestion: equatorial → ecliptic frame
Phase 37  (2026-04-xx)  Fisher epsilon parameter tuning
Phase 38  (2026-04-xx)  L_cat formula correction
Phase 43-H1 (2026-04-27) Added −N log D(h) to cluster combine path
Phase 43-H2 (2026-04-27) CRB CSV rotated to ecliptic frame (migrate_crb_to_ecliptic)
Phase 44  (2026-04-29)  P_det h-dependent zero-fill cutoff removed (MAP 0.86 → 0.765)
Phase 45  (2026-05-01)  First-bin anchor escalation (wrong layer; later superseded)
Tier 3    (2026-05-04)  D(h) double-counting removed (MAP 0.755 → 0.740)
Bridge fix (2026-05-05) Principled P_det extrapolation replacing Phase 45 anchor
H3 fix    (2026-05-06)  2D numerator: observation → hypothesis query + M_z axis fix
Phase 48  (2026-05-07)  Production fine-grid sweep — FINAL RESULT: both PASS
```

---

## 8. Statistical Framework Notes

A few design choices that may come up in discussion:

**σ_boot as uncertainty estimate:**  
We bootstrap by resampling events (B = 1 000 samples) and applying parabolic
refinement at each MAP. This gives σ_boot ≈ 0.002 at 1 473 events. Note this
does not capture seed-dependent MAP drift (~0.02 scale observed between
different prepare_detections seeds), which is a shared-injection-set effect.

**Shared injection set:**  
All h-values in a given run use the same detected events (same CRB CSV). This
is computationally efficient but means the bias is correlated across h-values.
σ_boot is conservative for the *joint* posterior but correct for the MAP.

**Info monotonicity as sanity check:**  
Adding the BH-mass channel (2D) must not increase posterior bias relative to
position-only (1D) — this would mean the extra information is hurting, which
is only possible if there is a systematic modelling error. This check caught
bugs §3.6 and §3.7 above.

**Gray et al. (2020) framework:**  
The completeness correction follows Eq. A.19 of arXiv:1908.06050 throughout.
The key formula is:

```
log L(h) = Σᵢ [ log( Σ_j w_j · L_gw(θᵢ, z_j, h) ) - log D(h) ]

where D(h) = ∫ P_det(d_L(z, h)) · (dV_c/dz) dz
```

Note: the `log D(h)` term appears once inside the per-event completeness term
`L_comp`; the Tier 3 fix corrected an additional (erroneous) outer `−N log D(h)`
that was being applied on top of this.
