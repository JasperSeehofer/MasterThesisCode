# ESS-floor pool-sizing analysis for the campaign redesign (issue #51)

**Date:** 2026-07-28 · **Scope:** quantitative core of the campaign design doc
(fix3_zmz_catalog_selection.md Amendments 1+2). Offline, CPU-only; no waveforms
were run — ESS depends only on kernel weights over (u, m), so the entire
(measure × N) scan runs on synthetic (M, z) draws from density grids.
**Scripts/outputs in this directory:** `s1_sizing.py` → `sizing_results.json`;
`s1b_mixtures3.py` → `sizing_results_mix3.json`; `s2_cap_snr.py` →
`cap_analysis.json`. Seed 20260728 throughout; single-realization draws
(sampling noise on w̄ at these ESS levels is far below the digits quoted).

Labels: **[M]** = MEASURED (computed here from repo data/code), **[E]** =
ESTIMATED (extrapolated from measurements), **[A]** = ASSUMED (stated choice).

---

## 1. Conventions replicated (and the two deliberate deviations)

- Support (Amendment 2): source-frame M ∈ [10⁴, 10⁷], z ∈ (0, 1.5] ⇒
  detector-frame m = log₁₀ M_z ∈ [4, 7.39794], u = ln(1+z) ∈ [0, 0.91629]. [A, ratified]
- Kernel: product Gaussian in (u, m); **Scott d=2, σ_j = N^(−1/6)·std_j on both
  axes** ([RATIFY-Z2]); Abramson √-law on u only, pilot = 400-bin histogram KDE
  exactly as `sdp.py:_abramson_lambda_u` (taps cap, 1e-12 floor, geometric-mean
  normalization). ESS = (Σw)²/Σw² per node, d_L-independent (Kish; sdp.py:693–698,
  z2.py:131–132). [M — code replicated line-for-line]
- Deviations from the *probe* (not from the ratified object): (i) the probe's
  `build_surv_ulm` ran σ_u at the d=1 exponent N^(−1/5) with the λ pilot at that
  width; Z2 ratifies N^(−1/6), which is what this scan uses, with the pilot at the
  kernel σ_u (the sdp.py convention). §3.4's own re-audit shows the direction of
  this difference (wider σ_u → mildly better ESS). (ii) λ is recomputed per
  measure/N from that measure's own pilot — required, since λ is pool-derived.

**Grid noding [A, justified]:** 61 u-nodes on [0, ln 2.5] (identical support to
the probe — the u axis does not widen) × **69 m-nodes on [4, 7.398]** (0.050
dex spacing ≈ the probe's 0.048 dex over its truncated range; spacing-parity,
not count-parity). Node count does not affect ESS or w̄ — verified [M]: at
N=200k the 31-m-node (0.113 dex) grid gives catalogue w̄ identical to 4 decimal
places for every measure. Noding is constrained instead by interpolation
fidelity: spacing should stay ≲ σ_m, and σ_m at N=10⁶ is 0.05–0.09 dex
depending on measure, so 0.05 dex is the right scale and 0.113 dex is not.
Storage: 3000 d_L × 61 × 69 float64 ≈ **101 MB** (50 MB float32) — same order
as §3.3's 45 MB. The §3.3 121-u production-doubling rider stays available
(≈200 MB); it changes nothing here.

---

## 2. Structural findings (all [M], from `catalog_zw_profile.json` + repo code)

1. **Amendment 2 is quantitatively vindicated.** Of the catalogue's
   R_eff/(1+z) rate weight (the exact Σ_glob_wbh weighting), **81.4 % sits
   above the OLD reachable ridge m = 6 + log₁₀(1+z)** (source cap 10⁶) —
   i.e. the current pool cannot cover four-fifths of the catalogue's query
   weight even in principle. The new box covers **99.98 %** (weight above
   m = 7.398: 0.023 %).
2. **The unreachable triangle.** Even on the new support, m > 7 + log₁₀(1+z)
   (source M > 10⁷) is physically unpopulated by ANY injection measure.
   **5.04 %** of catalogue weight lies above that ridge. Queries there are
   box-clamped and served by kernel tails/shrinkage no matter what N is —
   **the w̄ → 1 acceptance criterion must be defined on the reachable 94.96 %
   of weight**, with the unreachable 5.04 % reported separately (it is a
   support property of the mass function, not a sizing deficiency). Both
   variants ("all_clamped" and "reachable") are tabulated in the JSONs; at
   the recommended design they differ by ≤ 0.0001 anyway.
3. **Where the catalogue actually queries:** 99.98 % of weight at z < 0.3
   (34.6 % below z = 0.1), 81.4 % at m ∈ (6, 7.4]. The starvation problem is
   a low-z/high-m corner problem; the M1 measure puts only **0.34 %** of its
   draws in (m > 6) ∧ (z < 0.3) even on the widened box — this, not total N,
   is why the status-quo measure fails (§3).
4. **The truncation footgun, measured in the pool.** The canonical 50k pool's
   M column (detector-frame M_z) has max lm = 5.99997: `main.py`'s symmetric
   M_z truncation (`redshifted_M > parameter_space.M.upper_limit → skip`)
   bites at exactly 10⁶ *detector*-frame. If the new campaign sets the
   parameter-space M upper limit to 10⁷ (source) without lifting the
   detector-frame ceiling to 10^7.398, the m ∈ (7, 7.4] wedge is silently
   truncated again and ~29 % of catalogue weight (m > 6.5·… above the new
   truncated ridge) loses coverage. **Campaign implementation requirement:**
   population bounds are source-frame [10⁴, 10⁷]; waveform/CRB M_z bounds
   must be [10⁴, 10^7.398].
5. **ESS caveat (scope of the acceptance metric).** Kish ESS measures weight
   *concentration*, not local support: a node covered only by distant kernel
   tails can still report moderate ESS while its survival is an extrapolation.
   This is inherent to the ratified convention (production has it too). It is
   why the unreachable-triangle weight must be excluded from w̄ rather than
   "fixed" by more injections.

---

## 3. Measure × N frontier

Sampling measures (densities over (log₁₀M_source, z); drawn by 601×601-grid
inverse-CDF + in-cell jitter — better controlled than emcee and free of the
documented burn-in seam):

| key | definition |
|---|---|
| a | status-quo Babak M1 `emri_distribution` (exact emcee target density in these coordinates), widened box |
| b_rate | log-uniform M × z-marginal of (a) |
| b_vol | log-uniform M × dV_c/dz/(1+z) |
| cat | catalogue-coverage: ∝ W_z_lm profile restricted to the reachable set |
| mix_aXX | α·a + (1−α)·cat |
| flat_um | uniform in (u, m) on the box (variance-uniformizing KDE design measure) |
| mix3_x_y_z | x·a + y·cat + z·flat_um |

Note on (a): `dN_dz_of_mass` linearly extrapolates its top mass-bin fit and
holds it constant above 10^6.5 [A — adequate for sizing; the physics choice of
the population density on [10^6.5, 10⁷] is a separate physics-change decision,
§8]. Measured negative-density fraction after the z ≤ 1.5 clip: 0.0.

**Catalogue-weighted mean shrinkage w̄ = E_W[ESS/(ESS+10)], reachable weight** [M]:

| measure | 50k | 100k | 200k | 500k | 1M | N @ w̄=0.99 [E] |
|---|---|---|---|---|---|---|
| a (status quo) | 0.780 | 0.839 | 0.869 | 0.922 | 0.947 | **not reached ≤ 10⁶** |
| b_rate | 0.927 | 0.952 | 0.970 | 0.983 | 0.990 | ≈ 1.0 M |
| b_vol | 0.948 | 0.966 | 0.978 | 0.988 | 0.992 | ≈ 0.69 M |
| cat | 0.993 | 0.996 | 0.998 | 0.999 | 0.999 | ≤ 50 k |
| mix_a75 | 0.994 | 0.996 | 0.998 | 0.999 | 0.999 | ≤ 50 k |
| mix_a50 | 0.997 | 0.998 | 0.999 | 0.999 | 1.000 | ≤ 50 k |
| flat_um | 0.992 | 0.995 | 0.997 | 0.998 | 0.999 | ≤ 50 k |
| mix3_50_25_25 | 0.996 | 0.997 | 0.999 | 0.999 | 1.000 | ≤ 50 k |

**Grid-wide reachable-node floor (min ESS over reachable nodes / frac of
reachable nodes with ESS < 500)** [M]:

| measure | 50k | 200k | 1M |
|---|---|---|---|
| a | 2.5 / — | 1.0 / — | 1.1 / — |
| b_vol | 26.8 | 68.9 | 1.8 |
| cat | 1.0 | 1.0 | 1.0 |
| mix_a50 | 11.4 | 20.6 | 14.7 |
| flat_um | **151 / 0** | **346 / 0** | **820 / 0** |
| mix3_50_25_25 | 78.8 / 0.22 | 171.5 / 0.05 | 400.9 / 0.001 |
| mix3_40_20_40 | 90.6 / 0.15 | 175.4 / 0.03 | 466.8 / 0.000 |

Catalogue-weighted median ESS and weight-fractions below thresholds are in the
JSONs; headline: mix3_50_25_25 at N=200k → median ESS **8160**, catalogue
weight-fraction ESS<100 = **0.000**, ESS<500 = **0.0001**, w̄(all-clamped) =
0.9984.

**Key mechanisms [M]:**
- ESS grows ≈ N^(2/3), not N (Scott σ ∝ N^(−1/6) per axis shrinks the kernel
  window as N grows): measure-a median ESS rises 51 → 298 for 20× N, exponent
  ln(298/51)/ln 20 = 0.59. Brute-force N on the wrong measure is hopeless —
  measure (a) would need ~10⁷ injections to reach w̄ = 0.99 [E].
- The two-component mixtures' *grid* floors (~10–20) sit at the corners the M1
  and catalogue measures both ignore (low-m/high-z, m≈4 wedge); the flat_um
  component is what buys the grid-wide floor. The b-measures crash at 1M
  because the shrinking bandwidth exposes their empty low-z corner.
- b_vol vs b_rate: the M1 z-marginal is close to comoving-volume weighting;
  neither fixes the low-z mass deficit (both put < 1.2 % of draws at
  m > 6 ∧ z < 0.3).

---

## 4. Changing the sampling measure is legal for the conditional, and how to keep the marginal legs

The survival estimand S(d_L | u, m) = P(d_hor ≥ d_L | u, m) conditions on
exactly the kernel coordinates; given (u, m), d_hor depends only on extrinsics,
which the campaign randomizes identically for every draw. So any (u, m)
sampling measure yields an (asymptotically) unbiased conditional estimator —
the measure only moves *where the variance lands* (plus the standard
second-order KDE boundary/bias weighting, common to all candidates). **No
reweighting needed for the joint grid.** [Derivation-level statement; the
same tower-identity logic as fix3 §3.1.]

The POOL-MARGINAL objects (pooled survival, FIX-2 S(d_L|z) tower identity,
the m-marginal (K5) shrinkage target, any leg averaging over the pool) DO
require the population measure. Two clean options:

1. **Stratified campaign (RECOMMENDED [A]):** draw the components separately
   (N_a from measure a, N_cat, N_flat), record a `stratum` column (plus the
   three component densities per row for future reweighting). Marginal legs
   use the a-stratum only — exactly unbiased, zero new estimator conventions.
   The joint grid uses all rows.
2. Importance weights v = ρ_a/ρ_mix on all rows: Kish efficiency measured [M]
   ≈ the a-mixture-weight (mix_a75: 0.751, mix_a50: 0.500, mix3_50_25_25 ≈
   0.5·) — i.e. the marginal legs behave like α·N population draws either way.
   Option 1 is preferred because it adds no weighted-estimator surface area.

With N = 200k and α_a = 0.5, the a-stratum is 100k — **2× the current
production pool** for every pool-marginal leg. No regression anywhere.

## 5. The d_L-axis / selection subtlety (does it change the sizing?)

Checked in code [M]: both the probe (`z2.py` suffix-cumsum) and production
(`sdp.py` `_survival_at` / suffix tables) build **exact suffix-survival in d_L
from ALL injections at the node** — kernel weights carry no d_L, each
injection contributes its full d_hor step function, and one ESS number per
node governs the whole curve: Var[Ŝ(d_L)] ≈ S(1−S)/ESS at every d_L. There is
no per-d_L binning anywhere, so **per-node ESS is the complete sizing
statistic and the answer to Q4 is: no change to the sizing.** Two residual
caveats: (i) the curve's deep tail resolves only to S ≳ 1/ESS (relative error
∝ 1/√(S·ESS)) — Σ_glob_wbh is linear in S and dominated by mid-S, so the
floor below covers it; (ii) d_hor spread *within* a node comes from extrinsic
randomization — the campaign must keep full extrinsic randomization in every
stratum (it does by construction).

---

## 6. Recommendation block

- **ESS floor: 1000 per catalogue-support node** (pre-registered as:
  catalogue-weighted median ESS ≥ 1000, catalogue weight-fraction on
  ESS < 500 nodes ≤ 1 %, and reachable-weight w̄ ≥ 0.99). Rationale: with
  n₀ = 10, ESS = 990 gives per-node w = 0.99, so a 1000 floor makes the (K5)
  shrinkage per-node inert (≤ 1 % blend), and Var[Ŝ] ≤ (0.5)²/1000 →
  SE ≤ 1.6 % at S = 0.5 — below the ~2–3 ln resolution of the §3.8 residual
  arithmetic when propagated through Σ_glob_wbh. Not a hard estimator
  threshold (the ratified K5 remains the safety net); it is the *campaign
  acceptance* number.
- **Sampling measure: stratified 3-component mixture, α = (0.50 a, 0.25 cat,
  0.25 flat_um)** — `mix3_50_25_25`. Rationale: the a-stratum keeps all
  pool-marginal legs exactly population-sampled at 2× current size; the cat
  stratum places variance on the catalogue's query support (the thing the
  acceptance criterion scores); the flat_um stratum is the principled
  (variance-uniformizing, coordinate-derived, no fitted constants) guard for
  everything else — fallback events, P–P universes, and future catalogue
  revisions — lifting the reachable-grid min ESS from ~10–20 (2-component)
  to ~170 at 200k. Trade-off: 50 % of draws are "off-population", costing √2
  in marginal-leg precision vs an all-a pool of equal N — recovered
  many times over by the ×4 pool-size increase. All component densities are
  derivable from repo objects (emri_distribution, W_z_lm, the (u, m) box);
  nothing is fitted to a desired answer.
- **Total N = 200,000** (SNR-only injections; no Fisher). At this design
  [M]: reachable w̄ = **0.9985** (shrinkage attenuation of the −15.6 ln
  joint-composition increment: (1−w̄)·15.6 ≈ **0.02 ln** — measured inert),
  all-clamped w̄ = 0.9984, catalogue median ESS = 8160, catalogue W-frac
  ESS<500 = 0.01 %, grid-wide reachable min ESS = 172 (every node clears
  17× n₀). Floor option if compute-constrained: N = 100k still passes the
  acceptance set (w̄ = 0.9974, median 4801, W<500 = 0.6 %, grid min 120) —
  but 200k is the design point with headroom against the unmodeled
  differences between synthetic draws and the real campaign (timeouts,
  waveform-failure attrition, seed variation). 500k buys w̄ = 0.9992 and
  grid min 283 — not needed for the stated criterion.
- **Grid noding: 61 u-nodes × 69 m-nodes** (0.05 dex spacing — probe-parity
  *density*, not count; §1). ESS/w̄ insensitive to noding [M]; the constraint
  is interpolation fidelity (spacing ≲ σ_m ≈ 0.08 at 200k). 121-u rider
  optional; storage 101 MB float64 either way at 61×69.
- **Acceptance protocol (pre-register before the campaign):** rebuild the
  joint grid from the delivered pool with the ratified Z2/Z4 estimator,
  publish the §3.4-style table (min ESS, ESS<10/100/500 fractions,
  catalogue-weighted w̄ on reachable weight, unreachable weight-fraction
  reported separately), and require the three floor numbers above. The
  5.04 % unreachable-ridge weight is exempted by construction and must be
  listed, not hidden.

## 7. Detectability-verified-narrowing: measured status and pilot design

**Measured on the canonical 50k pool (`cap_analysis.json`) [M]:** the
detection horizon d_hor = SNR·d_L/20 *plateaus* into the current cap rather
than dying: p90(d_hor) = 3.8 → 4.5 → 4.5 → 4.1 Gpc across lm ∈ [5.5, 6.0] in
0.1-dex bins (log-slope over the top half-dex: +0.20/dex), with bin maxima
6–7 Gpc. The top bin (lm 5.9–6.0) still detects 3.7 % of its pool draws and
has d_hor max = 6.2 Gpc ≫ d_L of the entire catalogue support (d_L(z=0.3,
h=0.6) = 1.82 Gpc). **Detections continue above 10⁶ with near-certainty;
narrowing the upper bound is NOT verified and must not be assumed.** The
prior expectation from the LISA band (f at ISCO ∝ 1/M_z) is a genuine
horizon decline somewhere in m ∈ [6.5, 7.4], but its location must be
measured, not assumed.

**Pilot (run FIRST, before full-campaign submission):**
- N_pilot = **2,000** SNR-only injections, log-uniform in source M over
  [10^5.8, 10⁷] (the 5.8–6.0 half-bin overlaps the existing pool for
  cross-validation), z from the measure-a z-marginal, full extrinsic
  randomization, detector-frame M_z ceiling 10^7.398 (§2 item 4). Cost ≈ 4 %
  of the 50k campaign — cheap because d_hor is measured from EVERY injection
  (SNR rescaling), not only from detections.
- **Decision rule (pre-registered, horizon-support based — the same logic as
  `_compute_dl_global_max`, h-invariant):** bin m into 0.2-dex bins; the
  campaign's source-mass upper bound may be narrowed to 10^lg* only if every
  bin wholly above lg* (detector m > lg* + log₁₀(2.5)) satisfies
  max d_hor < ½·d_L(z_min-cat, h=0.60) — i.e. p_det ≡ 0 for every possible
  query, with a factor-2 margin. Binomial backstop: 0 detections among
  n ≥ 1000 pilot draws above the candidate cap bounds P(det) ≤ 0.3 % (95 %
  CL, rule of three). Expected outcome given §7 measurements: **no
  narrowing**; the pilot then doubles as the campaign's first high-m batch
  (its rows are valid mix-stratum draws if drawn from the campaign measure
  restricted to that band — recommended implementation).
- The pilot simultaneously retires the open FEW-validity question (§8): if
  waveform generation fails systematically above some M_z, that is a
  *model* limit to document, distinct from a detectability limit.

## 8. Open items (explicitly not settled here)

1. **Population density on source M ∈ [10^6.5, 10⁷]:** `emri_distribution`
   extrapolates its top fitted bin; `emri_rate.R_EMRI` documents a valid band
   [10⁴, 10⁷]. Which density the widened sampler uses is a physics-change
   decision (protocol applies). Sizing conclusions are insensitive (the
   frontier is dominated by the cat/flat strata, and b-measures with totally
   different mass profiles reproduce the ordering).
2. **FEW waveform validity at M_z up to 10^7.398** — offline session, not
   checked against few docs; resolved by the pilot (§7).
3. **§3.3-C clamp convention for the unreachable triangle:** box-clamp
   (current analog) vs ridge-clamp changes which node serves the 5.04 %
   unreachable weight. Flagged for the Z3 interpolant-convention decision;
   does not affect sizing.
4. **Cluster resourcing** (GPU-hours for 200k SNR-only injections + 2k pilot)
   — /cluster skill territory; not estimated here.
5. Acceptance-metric amendment (reachable-weight w̄) needs author ratification
   since Amendment 1 says "w̄ → 1 on the catalogue's query support" without
   the reachability carve-out (§2 item 2).
