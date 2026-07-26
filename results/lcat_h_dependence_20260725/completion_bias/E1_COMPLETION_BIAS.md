# E1 — Completion-only (fallback) channel low-bias: mechanism attribution

**Venue:** seed1000 deep run (truth h=0.73, 3454 events, 1992 zero-host "fallback" events),
`results/campaign_phase2_runs/run_20260719_seed1000_exp40/`.
**Measured anomaly:** fallback-only ensemble `Σ log(B_num/D)` peaks at **h = 0.6118 ± 0.0176** (replicated here to 4 decimals, `s7_results.json:real_prod_over_D_prod`), ≈ 6.5σ below truth.
**Date:** 2026-07-26. All analysis read-only; scripts `s1..s7_*.py` in this directory; compact numbers in `E1_summary.json`.

---

## Executive verdict

The low peak is **not** a defect of the completion-integral mathematics and **not** a data-provenance
problem. The decisive self-consistency Monte-Carlo (#6) shows the estimator core **closes at truth**
for the dark channel, and the real deep events are **statistically indistinguishable from
self-consistent synthetic dark events** (identical peaks to 4 decimals on a membership-clean subset).
The 0.612 number is produced by the **zero-host *subset* itself**: "ball is empty" is a
sky-and-redshift–dependent selection that the per-event statistic `B_num/D` does not (and per its own
derivation, need not) condition on. Three quantified components:

| Component | Direction | Size (peak shift) | Evidence |
|---|---|---|---|
| **M-A** subset normalization: `p_i` carries `P(dark\|det,h)=β_Ḡ/D`, rising in h | **UP** | ≥ +0.12 (MC: B/β_Ḡ 0.732–0.745 → B/D railed ≥0.86) | s2, s4, s6 |
| **M-B** empty-ball membership reshapes z-composition (dark low-z events acquire impostor candidates and exit the subset) | **DOWN** | ≈ −0.11 (0.86+ → 0.754 when composition pinned to observed) | s4 |
| **M-C** ZoA per-pixel completeness in `B_num`: low-z fallback events sit in `f_k≈0` pixels; their flat `(1−f_k)=1` weight has a systematically more negative h-slope than the sky-mean counterpart | **DOWN** | **−0.133** (real events, f̄-numerator: 0.745; production pixel-numerator: 0.612) | s3, s6 |
| **M-D** pooled-survival horizon trick vs pool-local true detection rate | **DOWN** | −0.02…−0.05 | s3/s4 (V4), s7 |

Net: +0.12 −0.11 −0.13 −(0.02..0.05) ≈ −0.12 → 0.61. The in-silico ladder (s4) reproduces the real
peak: model-composition B/D rails ≥0.86 → observed composition 0.754 → + ZoA pixel weighting rails
≤0.60, bracketing 0.612.

**Gate consequence:** fallback-only closure at truth with `Σ log(B_num/D)` is *not a theorem* —
only full-mixture closure is. C1 must use a subset-conditioned statistic (FIX-1 below), otherwise it
will fail even for a perfect estimator (in MC it fails HIGH, on the real data it fails LOW because
M-B/M-C dominate M-A).

---

## 1. What was measured (chronological, with scripts)

### s1 — subset-normalization test (`s1_subset_normalization.py`, `s1_results.json`)
D(h), β_Ḡ(h) extracted from the 41 per-h run logs (sky-aware production values); fallback census
(1992) from the `evaluate_*.err.gz` "#29" warnings, identical to the diagnostics `L_cat==0` census
(±1 event).
- `Σ_fb log(B_num/D)`: **0.6118 ± 0.0176** (replicates the known result).
- `Σ_fb log(B_num/β_Ḡ)` (conditioned on the dark channel): **rails ≤ 0.60**.
- Mixture weight `P(dark|det,h)=β_Ḡ/D` **rises** with h: 0.809 (h=0.60) → 0.853 (0.73) → 0.882 (0.86);
  its tilt at truth is **+633 logL per unit h** (toward HIGH h).
  ⇒ The subset-normalization term *counteracts* the low bias; the low pull lives in the numerators.

### s2 — decisive self-consistency MC (`s2_mc_selfconsistency.py`, `s2_mc_results.json`)
Synthetic dark events drawn from the generator's own density `(1−f̄(z,0.73))·dVc/dz/(1+z)` (z ≤ 1.5,
`PixelCompleteness` from the frozen `m_th` cache — the identical C1 object the run loaded), detected
with the estimator's own survival `p_det(d_L)` (built by `SimulationDetectionProbability` from the
same injection pool, `dl_max = 7.7603 Gpc`), observed `d_L` scattered with per-event Fisher
`σ_frac` matched to the real fallback events (NN in d_L), then scored with a verbatim-structure copy
of the production `B_num` (4σ window, z ≤ 1.5 cap, `fixed_quad n=50`, Gaussian in `d_L`-fraction).
- **B/β_Ḡ: 0.7366 ± 0.0155** (n=3759) — and 0.7320 ± 0.0079 / 0.7453 ± 0.0140 in the two replicate
  runs (s3, s6; seed/N variations). **The dark-channel estimator core closes at truth.**
- **B/D: rails ≥ 0.86** — the production subset statistic pushes HIGH for self-consistent data.
- Noiseless vs scattered observations: 0.7353 vs 0.7366 — **σ-kernel/no-noise effects are negligible**.
- Per-event slope bookkeeping at truth: self-consistent `E[dlogB_num/dh] = −1.197` vs required
  `dlogD/dh = −1.523`; the **real** events measure `−1.650` (gap −0.45/event: −0.30 composition,
  −0.15 within-z).

### s3 — slope-at-fixed-z attribution (`s3_slope_attribution.py`, `s3_results.json`)
Real per-event `dlogB_num/dh` (from the shipped diagnostics, 41 h) vs MC, binned in z:
- **z ≥ 0.45: real ≡ MC** (e.g. 0.45–0.50: −1.559 vs −1.548; 0.65–0.70: −2.082 vs −2.082).
  The production numerator and the harness agree, and the real deep events behave exactly like
  self-consistent dark events.
- **z < 0.4: real slopes far more negative** (0.2–0.25: −1.05 vs +0.07; 0.1–0.15: −0.32 vs +0.50).
- Cause identified (measured): **real low-z fallback events live in empty/ZoA pixels** —
  fraction with `f_k(z_ev, event pixel) < 0.01`: 62% (z 0.1–0.2), **97%** (0.2–0.3), 91% (0.3–0.4),
  while the sky mean is f̄(0.2)=0.434, f̄(0.3)=0.192, f̄(0.4)=0.055. Production `B_num` correctly uses
  the per-pixel `f_k` (GMV-2022 Eq. 5), so these events carry a flat `(1−f_k)=1` population weight
  whose h-slope is more negative than the rising `(1−f̄(z))` counterpart.
- V4 (acceptance by the pool's *actual* `SNR≥20` rate `p_true(z)` instead of the survival model):
  conditioned closure moves 0.7366 → **0.674** (s3) / 0.683 (s4) ⇒ **M-D ≈ −0.05** on the conditioned
  statistic. Survival vs truth: `p_surv(dist(z))` overestimates at mid-z
  (z=0.31: 0.493 vs 0.345; z=0.41: 0.359 vs 0.261), crossover near z≈0.5.

### s4 — composition + reproduction ladder (`s4_reproduction_ladder.py`, `s4_results.json`)
Observed fallback z-profile vs model dark-detected profile (density ratios):
**×8–10 at z<0.05** (17 events), **0.32–0.57 at z∈[0.15,0.40)**, ~1.3–1.9 at z>0.4 (mostly
renormalization of the low-z deficit). This is the empty-ball membership map: at z<0.4 the catalogue
is dense outside the ZoA, so dark events there almost always have impostor candidates and leave the
subset (observed fallback fraction 0.577 vs model `P(dark|det)=0.853`; ≈27% of detected events are
model-side dark-with-impostor-candidates).
Ladder (production sky-aware D(h)):
`B(f̄)/D` model composition **≥0.86 (railed)** → composition pinned to observed **0.754** →
+ ZoA-flat pixels at the empirically measured per-bin ZoA fractions **rails ≤0.60**. Real: **0.612**.
The three ingredients are sufficient (the last rung slightly overshoots; see §4).

### s5 — membership-clean real-data closure (`s5_clean_subset_closure.py`, `s5_results.json`)
Subset `x = d_L^obs > dist(0.45, 0.73) = 2.416 Gpc` (n=1373; membership ≈ complete there, f̄≈0):
with an x-cut–conditioned dark denominator, **real peak = 0.6329, MC (true-rule) peak = 0.6328** —
*identical to 4 decimals*. Strongest possible statement that **no provenance anomaly** exists in the
real fallback data; every residual deviation is statistic-side (the cut-conditioned statistic itself
carries an O(−0.1) distortion; see §4).

### s6 — selection-conditioning variant (`s6_pdet_in_numerator.py`, `s6_results.json`)
Because detection in this pipeline is decided by the latent true-parameter SNR (oracle selection),
the textbook alternative `B̃ = ∫(1−f)·p_det(z,h)·L·dVc/(1+z)dz` (p_det inside the numerator) was
tested: full-range closure 0.7534 vs 0.7453 (production form) — **second order (+0.008) at ensemble
level**; deep-subset statistics move by ±0.1 in opposite directions between the two forms (open
subtlety, §4). Applied to the real events with the f̄ (sky-mean) numerator:
`real Σ log B(f̄) / D_prod` peaks at **0.7447** vs production (pixel-f_k) **0.6118** — isolating
**M-C = −0.133** as a direct real-data measurement.

### s7 — fix predictions (`s7_fixes_and_summary.py`, `s7_results.json`)
- FIX-2 (z-resolved survival `S_z(d_L)` in D): real production sum over `D_zres` peaks **0.631**
  (+0.019; 0.05-wide z-bins make D_zres(h) kinky — resolution-limited; the V4 estimate −0.05 is the
  robust size of M-D).
- FIX-3 (generator-consistent mixture denominator, estimate): model in-catalogue population fraction
  `F = 0.0175`; `P(cat|det) ≈ 0.115`; `dlog D_gen/dh = −1.02` vs production `−1.52` ⇒ would push the
  fallback-only subset further DOWN (rails ≤0.60) while pushing the host channel up — meaningful only
  on the full ensemble.

---

## 2. Mechanism verdicts (mission list)

1. **Generator/analysis population mismatch (dark channel): REFUTED.** Generator and inference use the
   same frozen `PixelCompleteness`; MC with exactly this density closes at truth (0.732–0.745).
   **CONFIRMED for the full mixture** (secondary here): total detected z-histogram vs
   `p_pop·p_true(z)` shows ×8.4 (z<0.05), ×2.3 (0.05–0.10), ×0.62–0.67 (0.10–0.25) — the rate-weighted
   catalogue channel does **not** equal `f̄(z)·p_pop` (Option-A / COM-03 violation). Host-channel issue.
2. **Flat-21% completeness tail: OBSOLETE/REFUTED.** The run used the pixel Schechter model
   (`m_th` cache load confirmed in logs); f̄ = 0.853 (z=0.05), 0.434 (0.2), 0.192 (0.3), 0.055 (0.4),
   0.010 (0.5), 0.001 (0.6), 0.000 (≥0.8). No flat tail; the fallback ensemble is dominated by z>0.4
   where (1−f)≈1 and the completeness shape is nearly irrelevant.
3. **d_L uncertainty model: REFUTED as driver.** Prepared-CSV scatter is honest
   ((d_L^prep−d_L^raw)/σ: mean 0.005, std 1.022); σ_frac quartiles 2.8/3.8/4.6% (rel-err<0.10 gate);
   noiseless-vs-scattered MC peak difference 0.001.
4. **D(h) shape: PARTIALLY CONFIRMED, subdominant (M-D).** Sky banding irrelevant
   (dlogD/dh −1.518 sky-aware vs −1.523 isotropic); zero p_det-grid coverage warnings; z_max caps
   inert. But the pooled-survival horizon trick mis-shapes selection vs the pool's true `SNR≥20` rate:
   −0.05 (conditioned), +0.02 when fixed in the production statistic.
5. **Missing evolution/(1+z)/Jacobian: REFUTED.** Exact-structure MC closes at truth.
6. **Self-consistency (decisive): DONE.** Synthetic fallback ensemble under the production statistic
   does **not** peak at 0.61 — it rails HIGH; the conditioned statistic closes at truth. The real
   0.612 is a subset effect (M-A/M-B/M-C/M-D), not an estimator-core systematic, and the real data
   match the generator (s5: real ≡ MC).

---

## 3. First-principles fix candidates (no fitting to truth)

**FIX-1 — subset-conditioned gate statistic (for C1).**
Test fallback-only closure with `Σ_fb log p_i − N_fb·log P_fb(h)`, where
`P_fb(h) = model probability that a detected event has an empty candidate ball` — computable with no
data by model MC: draw events from the model at each h (mixture, sky-pixelated), scatter observables
with the Fisher model, run the *actual* `get_possible_hosts_from_ball_tree` window against the
*actual* catalogue, count the empty fraction. Derivable purely from model + catalogue geometry
(the catalogue is fixed data; ball emptiness is a deterministic function of the observables — no new
physics enters the per-event likelihood, which remains `B_num/D`).
**Predicted:** restores the fallback-subset peak to 0.73 ± ~0.02 by construction (removes M-A and
M-B; M-C becomes correct physics on both sides because `P_fb(h)` inherits the ZoA correlation).

**FIX-2 — z-resolved survival `S_z(d_L)` (estimator physics).**
Replace the pooled `P(d_hor ≥ d_L)` with `P(d_hor ≥ d_L | z near)` in D, β_Ḡ, β_G and the p_det
grids. Generator-consistent: the `SNR ∝ 1/d_L` amplitude law is exact *within* fixed z (fixed
detector-frame masses/waveform), while pooling across z mixes intrinsic populations that do not sit
at that distance. **Predicted fallback-peak shift: +0.02 … +0.05** (V4: conditioned −0.049 removed;
direct production recompute +0.019 at 0.05-z-bin resolution).

**FIX-3 — generator-consistent selection denominator (host-channel workstream).**
`D(h) = ∫ p_det·p_pop` assumes the injected population is `p_pop` (constant comoving density,
Option A); the actual generator mixture is `F·p_cat + (1−F)·(1−f̄)·p_pop` and the measurement in §2.1
shows the difference is large at low z. All ingredients exist per-run (`Σ_glob(h)` catalogue sums,
β_Ḡ(h)). Estimated `dlogD_gen/dh = −1.02` vs `−1.52`. **Predicted signs:** fallback-only subset moves
further DOWN, host channel moves UP; must be evaluated on the full ensemble — flagged for the
host-found channel investigation, not adopted here.

Rejected on principle: per-event pixel-conditioned D (conditioning the selection normalization on
observed data — not derivable); any reweighting of the observed composition (fitting to data).

---

## 4. What remains unexplained / open

- **Deep-subset conditioning subtleties at the ±0.05–0.1 level.** Sharp x-cut conditioned statistics
  carry O(−0.1) distortions (s5: both real and MC at 0.633 under a denominator intended to close);
  the oracle-selection numerator variant (p_det inside `B_num`) moves deep-subset statistics +0.12
  and the full range +0.008 (s6). A dedicated derivation of per-event selection conditioning with
  heterogeneous σ (σ–z coupling: detected near-threshold events have larger σ) is needed before
  FIX-1/FIX-2 magnitudes are quoted better than ±0.03.
- **Ladder overshoot.** The ZoA rung (s4) rails ≤0.60 vs real 0.612 — the Bernoulli per-bin ZoA
  assignment and f≡0 flattening are crude by ~0.01–0.05 in peak position.
- **z<0.05 composition excess (×8–10, 17 events).** Consistent with ZoA balls being empty even at
  tiny z, but not explicitly modeled.
- **Census off-by-one** (1993 diagnostics `L_cat==0` vs 1992 log warnings) — bookkeeping, untraced.

---

## 5. Cross-implications for the host-found channel

- **Shared factor D(h) and w_G(h):** M-D (survival-trick error) enters every event identically;
  FIX-2 changes `w_G(h) = β_G/D` shape.
- **Membership complement:** the ≈27% (model-side) dark-with-impostor-candidate events are *inside*
  the host-found sample, their `L_cat` driven by impostors, concentrated exactly where the fallback
  subset is deficient (z 0.1–0.4, catalogue-dense sky). The host channel's own low rail (0.60,
  78% completion-dominated post-Variant-1) shares M-B in mirror image.
- **Option-A violation (COM-03 flavor)** measured directly: detected-event z-histogram vs
  `p_pop·p_true` off by ×8.4 (z<0.05) / ×0.62–0.67 (z 0.10–0.25) — the discrete rate-weighted
  catalogue sums and the continuous `f̄·p_pop` integrals are *not* interchangeable; this is the
  quantitative basis for FIX-3 and directly biases `β_G·L_cat/D`.

## Files

- `s1_subset_normalization.py` / `s1_results.json` — D/β_Ḡ tables from logs, subset-normalization test
- `s2_mc_selfconsistency.py` / `s2_mc_results.json` — decisive MC (#6), V0/V0n/V3
- `s3_slope_attribution.py` / `s3_results.json` — slope-vs-z, V4 true-rule, p_true vs p_surv
- `s4_reproduction_ladder.py` / `s4_results.json` — composition + ladder + mixture check
- `s5_clean_subset_closure.py` / `s5_results.json` — membership-clean real≡MC test
- `s6_pdet_in_numerator.py` / `s6_results.json` — selection-conditioning variant; M-C real measurement
- `s7_fixes_and_summary.py` / `s7_results.json`, `E1_summary.json` — fix predictions, compact summary
