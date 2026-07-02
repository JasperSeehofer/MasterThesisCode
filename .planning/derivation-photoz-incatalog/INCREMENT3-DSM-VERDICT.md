# Increment-3 verdict: global photo-z-smeared selection `D_sm` — and the information-starvation conclusion

Status: **NEGATIVE / DEFINITIVE.** The global photo-z-smeared same-kernel selection `D_sm`
(the only candidate that survived the literature comparison + derivation + adversarial verification)
**de-biases the global density gradient but does NOT recover a peaked H₀.** The in-catalogue
photometric channel at GLADE's regime (σ_z ≈ 0.035, σ_z/z ≈ 0.7, z ≈ 0.05, p_det ≈ 1) is
**information-starved**: the photo-z error (~17× the GW redshift precision) destroys per-event host
localization, so no normalization tested recovers an H₀ peak. All bridge experiments are the clean
no-sky `rung_I` closure (truth h = 0.73), prototype in `scripts/bridge_closure/_rungI_verify_B.py`
(`hierarchical_shared_latent` flag, committed `5ef8c6e`).

## The candidate
Numerator: dV_c-once regularised `p_red = N(z;z_g,σ_z)·p_bg/Z_g`. Denominator: replace the frozen
point-eval global selection `D(h)=Σ_g w_g p_det(z_cat_g,h)` with the **global photo-z-smeared**
`D_sm(h) = Σ_g w_g ∫ p_det^GW(d_L(z,h)) p_red(z|z_cat_g) dz` — same kernel in numerator and
denominator, over the full catalogue. Gate proof (σ_z→0 → standard Option-A) confirmed
(`DERIVATION-HIERARCHICAL.md`).

## Evidence (all measured)
| Test | Result |
|---|---|
| **Gate** σ_z=0.002, multi-seed | `D_sm` ≈ standard (median ~0.73). PASS. |
| **Single-seed** de-rail σ_z=0.035, seed=1 | 0.693 interior — looked like a win. **It was a favourable draw.** |
| **Multi-seed** σ_z=0.035, n_ev=250 | 6 interior / 4 rail-up (0.87) / 2 rail-down (0.60); std 0.11. |
| **More events** n_ev=2000 | std 0.097 — **did NOT shrink** (so scatter is not event noise). |
| **Lever** `d/dh log(D_sm/D)@0.73` vs n_gal=12k→400k | **+0.19 ± 0.014 → ±0.007, D_sm/D ≈ 0.920** — deterministic already at 12k (so scatter is not catalogue/edge under-sampling either). |
| **Posterior shape** n_ev=2000, per seed | peaks at 0.64, 0.64, 0.69, 0.87 — **never 0.73**; several multimodal/bumpy. |
| `E[h] ≈ 0.735` | **Artifact**, not de-bias: the grid [0.60,0.87] midpoint is 0.735 ≈ truth, so a flat posterior gives E[h]≈0.73 trivially. The *shape* (flat/multimodal) is the real readout. |

## Why (mechanism, now fully resolved)
`D_sm`'s lever is real and deterministic — the global selection denominator, smeared over the photo-z
kernel, is edge-galaxy-dominated (z~0.15–0.25, where p_det varies and σ_dL > the p_det logistic
width), giving a stable rising `D_sm/D(h)`. That **cancels the global density-gradient rail**
(standard's deterministic 0.60). But a single global per-h scalar **cannot track the LOCAL numerator
gradient at z*(h)**, and with σ_z ≫ σ_z^GW there is no per-event localization to track anyway. The
residual is the bare discrete-catalogue + photo-z noise structure → a flat/multimodal posterior with
no peak at the truth. This is the structural obstruction (global scalar vs local gradient) made
manifest, plus irreducible information loss.

## What was ruled out (this investigation + prior)
standard (rail-down 0.60) · all numerator-only cleans (rail-up 0.87) · local consistent-denominator
(gate FAIL) · **global same-kernel `D_sm`** (de-biases, no peak — this doc). Remaining untested:
the full hierarchical cross-event Monte-Carlo `(★)` — but `DERIVATION-HIERARCHICAL.md` + the verifier
predict it is **null** at p_det≈1 via the 1/N_gal "space is big" suppression (the cross-event
coherence vanishes regardless). Optional to run for definitive closure; low expected value.

## Conclusion & recommendation
**In-catalogue photometric dark sirens at GLADE σ_z/z≈0.7, z≈0.05 are information-starved.** This is a
demonstrated methodological result, not an assertion. Recommended project direction: pivot the H₀
headline to the **spectroscopic forecast arm** (self-consistent spec-z hosts recover h≈0.725, per
`scripts/bridge_closure/BRIDGE-FINDINGS.md`) plus a rigorously-characterized **GLADE-photometric
limitation**. The frame fix (#15) and the likelihood-vs-posterior / dV_c-once interpretation
(`CATALOG-INTERPRETATION.md`) remain valid and feed the forecast arm.
