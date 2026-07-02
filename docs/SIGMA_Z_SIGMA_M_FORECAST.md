# σ_z / σ_M precision forecast — when is LISA EMRI dark-siren H₀ useful?

**Scope.** The constructive companion to
[`docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md`](BIAS_RESOLUTION_ATTEMPTS_REPORT.md).
That report demonstrated that the in-catalogue **photometric** dark-siren channel at
GLADE's regime (σ_z ≈ 0.035, σ_z/z ≈ 0.7, z ≈ 0.05, p_det ≈ 1) is *information-starved*.
This document answers the forward question: **what host redshift precision σ_z and host
BH-mass precision σ_M are required for an informative H₀, and where is it futile?** —
i.e. the paper's feasibility figure (F5). It also tests the hypothesis that the
**with-BH-mass (2-D) channel converges faster** (tolerates larger σ_z) than the
without-BH-mass (1-D) channel.

**Date:** 2026-06-30 · **Branch:** `physics/photoz-joint-normalisation`
**Engine:** `scripts/bridge_closure/sigma_z_sigma_M_forecast.py` ·
**Figure:** `docs/figures/sigma_z_sigma_M_precision_heatmap.png`

---

## 1. Method — the self-consistent forecast closure

We use the **self-consistent closure** (unbiased by construction) as the forecast engine,
extending the bridge `rung_I` closure that settled the photo-z investigation. Because the
closure is unbiased, the posterior **RMSE-to-truth** is a clean measure of information
content — exactly what a forecast needs (the *bias* question is settled separately).

- **Population & injection (self-consistent).** Galaxies are drawn `z ~ dV_c/dz/(1+z)`
  (or, for the robustness pass, by resampling the **real GLADE n(z)**); host BH masses from
  the M1 mass function × R_eff. Events are injected at the host's **true** (z, M),
  rate-weighted `w_g = R_eff(M_g)/(1+z_g)`, and pass a smooth mock `p_det(d_L)`. The
  catalogue then reports **noisy** observables: `z_cat = z + N(0, σ_z)` and
  `M_cat = M·(1 + σ_M·N(0,1))`; the GW reports `d_L_meas = d_L + N(0, σ_dL)`
  (σ_dL = 5% d_L) and a near-exact detector-frame mass `M_z_meas` (σ_Mz = 0.1%, EMRIs
  measure the central MBH mass very precisely). The inference kernels use the **same**
  σ_z, σ_M — so the estimator is unbiased and the posterior width measures information.

- **Two channels** (mirroring the production `single_host_likelihood`):
  - **1-D (without BH mass):** per-event numerator
    `Ñ_i(h) = Σ_g w_g ∫ N(z; z_cat_g, σ_z) p_GW(d_L(z,h)) dz`, selection denominator the
    global Option-A catalogue selection `D(h) = Σ_g w_g p_det(d_L(z_cat_g, h))`.
  - **2-D (with BH mass):** the numerator gains the host-mass factor
    `m_g(z) = N(M_z_meas; M_g·(1+z), σ_Mz² + (σ_M·M_g·(1+z))²)`. This is the absolute-mass
    equivalent of the production fraction-coordinate Gaussian product
    (`bayesian_statistics.py:1773–1783`), with the **H3** source-frame→observer-frame
    convention (`M_g` source-frame, `M_g·(1+z)` observer-frame hypothesis). Because the
    mock `p_det` is mass-blind, the selection denominator is **identical** to the 1-D one
    (the production 2-D MC denominator `∫ p_det·p_gal(z)·p_gal(M)` collapses over M to the
    mass-blind form) — so the 2-D channel is a clean **pure-numerator information gain**.

- **Metric.** σ_eff(H₀)/H₀ = √⟨(h−h_true)²⟩ / h_true (RMSE-to-truth = the posterior
  squared-loss/Bayes risk = √(width² + bias²)). This is the honest forecast metric: the raw
  posterior **width** is *misleading* because a sharp rail at the wrong grid edge has small
  width but large error (demonstrated: a railed cell has width 0.011 but RMSE 0.222). The
  H₀ grid is widened to [0.50, 0.96] (symmetric about truth) so a flat/uninformative
  posterior reads ≈ **18.6%** and an edge-pinned rail ≈ **31.5%** — clearly separated from
  a useful peak (< 5%). We report the **median over seeds** plus a `rail_frac` diagnostic.

### Optimisations (exact)
- `d_L(z, h) = dist(z, 1)/h` **exactly** (the E(z) integral is h-independent;
  `physical_relations.dist:71–76`) → precompute `g(z)=dist(z,1)` once per event; no per-h
  distance calls.
- The photo-z kernel and the mass factor are h-independent, so the candidate-galaxy
  dimension is collapsed into an h-independent vector once per event and the h-loop is a
  single vectorised matmul `num(h) = dz·(gw(h) @ v)`. ~50× speedup.

---

## 2. Result — the feasibility heatmap

![σ_z/σ_M precision heatmap](figures/sigma_z_sigma_M_precision_heatmap.png)

**Panels:** (A) 1-D channel, (B) 2-D channel with the σ_M=σ_z/(1+z) frontier and 2%/5%
accuracy contours, (C) the accuracy gain (1-D)/(2-D). Colour = σ_eff(H₀)/H₀.

**1-D channel (without BH mass).** Informative (σ_eff < 5%) **only for σ_z ≲ 10⁻³**
(essentially spectroscopic); for σ_z ≥ 0.015 it is **railed** (rail_frac → 1) at
σ_eff ≈ 26–30%. At GLADE photometric σ_z (0.035–0.05) it is hopelessly uninformative.
This *quantifies* the demonstrated info-starvation: photometric → useless,
spectroscopic → ~5%.

**2-D channel (with BH mass).** The informative region extends to **~50× larger σ_z** —
but only along the **small-σ_M edge**. The frontier is **σ_M·(1+z) ≲ σ_z**:
- σ_M ≤ 0.5–1% → useful all the way to GLADE photo-z (σ_z = 0.05);
- σ_M ≈ 2% → useful to σ_z ≈ 0.015;
- σ_M = 1.0 (linear, ≈ 0.36 dex) → **no better than the 1-D channel**.

The useful-5% σ_M boundary *at the GLADE photo-z operating point* (σ_z ≈ 0.035–0.05) is
**σ_M ≈ 1.2–1.7%**.

**Mechanism (the convergence physics).** The GW measures the EMRI detector-frame mass
`M_z` precisely; the host source-frame mass `M_g` then provides an **h-independent
redshift anchor** `1 + z = M_z/M_g` of precision `σ_{z,mass} ≈ σ_M·(1+z)` (differentiating
`M_z = M_g(1+z)` at fixed precise `M_z`). At trial h the *h-dependent* GW-distance redshift
`z*(h)` must agree with this *h-independent* mass anchor → H₀ is constrained directly. The
2-D channel therefore rescues exactly when the mass anchor beats the photo-z,
`σ_M·(1+z) ≲ σ_z` (with the detected-event median z ≈ 0.15, 1+z ≈ 1.14). The frontier line
in panel B is this relation; the steeper 2%/5% contours are the *useful-accuracy*
boundaries (a different, stricter curve — they coincide only near GLADE).

> **This heatmap uses the smooth synthetic n(z) — it is the *idealised* mass-channel bound.**
> Under the **real GLADE n(z)** (§4.6, `…_realnz.png`) the 1-D channel is *more* tolerant
> (real density structure carries z-info) but the catalogue rails hard at σ_z ≈ 0.025–0.05,
> and **GLADE's actual σ_z ≈ 0.035 rails regardless of σ_M** — even σ_M = 0.5% does not rescue
> it. The realistic mass-channel gain is ~1–3×, not ~60×. The core conclusion (info-starvation
> at GLADE; no rescue at realistic σ_M) holds under both and is *strengthened* by the real n(z).

---

## 3. The realism verdict — why the 2-D channel does not rescue GLADE in practice

The host BH mass comes from the stellar-mass→BH-mass relation
(`handler.py:30–33, :1033`). Its **intrinsic scatter is ~0.3–0.5 dex**, which in linear
fractional terms (log-normal CV = √(exp((ln10·s)²)−1)) is:

| relation scatter s | fractional σ_M |
|---|---|
| 0.08 dex (the *fit* error in code) | 0.19 |
| 0.3 dex | 0.78 |
| 0.4 dex | 1.16 |
| 0.5 dex | 1.66 |

So the engine's grid maximum σ_M = 1.0 corresponds to only **~0.36 dex — the optimistic
*low* end** of the realistic scatter; a realistic 0.5 dex (CV ≈ 1.66) lies *off the top of
the grid*. The 2-D channel needs σ_M ≲ 1–2% to rescue GLADE photo-z — **~50–170× smaller**
than the relation's intrinsic scatter. Note the production code currently carries only the
relation's *fit* error (`d_α = 0.08 dex ≈ 19%`), **not** the intrinsic scatter — and even
that under-estimate is already ~10–20× above the ~1–2% needed.

**Conclusion (defensible, conservative).** The with-BH-mass channel is a *powerful lever in
principle* — it tolerates ~50× larger σ_z — but realising it requires host BH-mass
precision ≲ 1–2%, far beyond the stellar-mass→BH-mass relation. So it **does not rescue
GLADE photometric hosts in practice.** The forecast's idealisations all make the 2-D channel
look its *best* (see §4), so this verdict is if anything *understated*. The practical path to
LISA dark-siren H₀ remains **either** spectroscopic hosts (σ_z ≲ 10⁻³) **or** an independent
percent-level host-mass measurement — both demanding. This is the paper's "know where to
stop."

---

## 4. Validity, caveats, and idealisations (verified)

The result was adversarially verified across three lenses (faithfulness, statistics,
physics-realism); faithfulness = **sound**, the others **minor issues only**, all of which
*strengthen* the conservatism of the conclusion. The caveats to carry into the paper:

1. **Self-consistent closure** — unbiased by construction; it measures *information content*,
   not production bias mechanisms (Malmquist, MC-denominator noise, photo-z systematics).
   That is the intended forecast question.
2. **The 2-D advantage is a pure-numerator gain** — three deliberate idealisations make the
   2-D channel look its best, so "no rescue" is conservative: (a) the selection denominator is
   **mass-blind** (shared with 1-D), not production's 4-D mass-dependent MC `p_det`;
   (b) `M_z` is treated as an **independent** measurement, not marginalised from the
   correlated 4-D GW covariance; (c) a **linear-Gaussian** (not log-normal) host-mass kernel —
   but at realistic σ_M the anchor width σ_M(1+z) ≈ 1.1 ≫ z ≈ 0.15, so the anchor is
   uninformative regardless of kernel shape.
3. **Do not quote sub-≈1.4% absolute precision.** The closure removes the documented generic
   +1.4% σ²-distance-inversion floor (best-cell bias ≈ 5×10⁻⁶), and the h-grid step (0.01,
   ≈1.4% in h/h) floors the bright corner. Cells below ~1.4% are "grid-limited / very
   useful," not literal sub-percent forecasts. The 5% useful boundary and the 26–30% railed
   values are well above the floor and unaffected.
4. **Absolute % is at N_events = 400; σ ∝ N^{−1/2}.** The realistic LISA EMRI host-associated
   yield is uncertain (model-dependent, plausibly far below 400), so the **relative/structural**
   conclusions (1-D railed for σ_z ≥ 0.015; 2-D rescue gated on σ_M ≲ 1–2%; ~50× tolerance;
   the σ_M·(1+z) ≈ σ_z mechanism) are the robust headline; absolute precision and the 2%/5%
   contour *positions* must be re-quoted at the paper's adopted N.
5. **Saturation band.** RMSE values in the 18–31% band reflect reversion toward an
   uninformative-flat (≈18%) or edge-pinned-rail (≈31%) posterior, not a measurement. The
   non-monotonic dip (σ_z = 0.05 slightly *below* 0.025) is the documented photo-z
   bias-turnover (a broad kernel washes out the n(z) gradient) — both are firmly "not useful."
6. **n(z) robustness — the frontier IS population-shape-dependent (important).** A robustness
   pass with the **real GLADE n(z)** (`--population real_nz`, 16 seeds,
   `sigma_z_sigma_M_precision_heatmap_realnz.png`) changes the picture in two ways and is the
   more realistic of the two:
   - **The 1-D channel is *more* tolerant under the real n(z)** — useful (< 5%) out to
     σ_z ≈ 0.015 (vs ≈ 10⁻³ for the smooth synthetic), because the real GLADE density
     *structure* (peaked/declining) carries redshift information the smooth n(z) lacks.
   - **But the real catalogue rails hard at σ_z ≈ 0.025–0.05, regardless of σ_M.** At GLADE's
     actual photometric σ_z (≈ 0.035), *both* channels rail — even σ_M = 0.5% gives σ_eff ≈ 26%
     (partially railed). The dense real catalogue has **mass+photo-z confusers** (galaxies whose
     mass-implied redshift coincides with a wrong-H₀ GW redshift while their broad photo-z
     kernel overlaps) that defeat the mass anchor at large σ_z; the sparse smooth synthetic
     catalogue has fewer, so its 2-D rescue (to σ_z = 0.05) is an **idealised upper bound**.
   - Consequently the realistic **mass-channel gain is modest (~1–3×)**, not the synthetic's
     ~60×, and is concentrated just below the railing cliff (σ_z ≈ 0.008–0.025).

   This **cross-checks the documented seed-600 real-pipeline rail at σ_z ≈ 0.035** and
   **strengthens** the core conclusion: at GLADE's real operating point the in-catalogue channel
   is information-starved and **host masses do not rescue it even at unrealistically precise σ_M**.
   The synthetic forecast establishes the *mechanism* and the idealised lever; the real-n(z)
   forecast is the realistic limitation. (The realistic conclusion at the relevant *large* σ_M is
   identical in both: the 2-D channel collapses onto the 1-D channel.)
   [`outputs/sigma_z_sigma_M_forecast_realnz.json`]

---

## 5. Reproducibility

```bash
# headline synthetic sweep (32 seeds) + figure
OMP_NUM_THREADS=1 uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py \
    --sweep --workers 14 --seeds 32 --population synthetic
uv run python scripts/bridge_closure/_forecast_plot.py   # writes the heatmap (rmse_truth + width)

# real-GLADE-n(z) robustness pass (needs the local reduced catalogue)
OMP_NUM_THREADS=1 uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py \
    --sweep --workers 14 --seeds 16 --population real_nz \
    --out scripts/bridge_closure/outputs/sigma_z_sigma_M_forecast_realnz.json

# quick sanity gate (a few cells)
uv run python scripts/bridge_closure/sigma_z_sigma_M_forecast.py --smoke
```

Outputs: `scripts/bridge_closure/outputs/sigma_z_sigma_M_forecast*.json` (swept medians +
raw per-seed cells), `docs/figures/sigma_z_sigma_M_precision_heatmap*.png`.

## 6. Quantitative table

σ_eff(H₀)/H₀ (%), median over **32 seeds**, N_events = 400 (σ ∝ N^{−1/2}; synthetic n(z)).
The last column is the 1-D (without-BH-mass) channel; the others are the 2-D channel by σ_M.
Cells ≲ 1.4% are grid/floor-limited (§4.3); ≈ 18–31% = the uninformative/railed saturation band.

| σ_z \ σ_M | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 | 1.0 | **1-D** |
|---|---|---|---|---|---|---|---|---|---|
| **0.0005** | 0.1 | 0.1 | 0.4 | 0.7 | 0.9 | 1.4 | 2.0 | 4.4 | **4.3** |
| **0.001** | 0.1 | 0.2 | 0.4 | 0.8 | 1.0 | 1.3 | 2.3 | 3.7 | **6.5** |
| **0.002** | 0.1 | 0.4 | 0.6 | 1.0 | 1.3 | 1.7 | 3.4 | 6.5 | **8.2** |
| **0.004** | 0.4 | 0.6 | 0.8 | 1.3 | 1.7 | 2.3 | 4.1 | 7.4 | **12.8** |
| **0.008** | 0.7 | 1.0 | 1.4 | 2.2 | 3.1 | 4.3 | 9.3 | 14.2 | **16.6** |
| **0.015** | 1.0 | 1.7 | 2.9 | 5.2 | 8.2 | 11.5 | 20.8 | 21.6 | **26.1** |
| **0.025** | 1.2 | 2.1 | 5.7 | 16.8 | 25.7 | 28.4 | 28.8 | 29.7 | **29.8** |
| **0.05** | 1.6 | 3.5 | 10.3 | 25.7 | 28.2 | 28.6 | 28.6 | 28.6 | **29.3** |

Reading: the 1-D channel crosses 5% between σ_z = 5×10⁻⁴ and 10⁻³ (spec-z) and is railed
(≈26–30%) for σ_z ≥ 0.015. The 2-D channel keeps σ_eff < 5% out to σ_z = 0.05 when
σ_M ≤ 0.01, and the useful region tracks the σ_M ≲ σ_z frontier; at σ_M = 1.0 (≈0.36 dex)
it collapses onto the 1-D channel. (32-seed values match the 8-seed run to ≲1%, confirming
the headline is seed-robust.)

