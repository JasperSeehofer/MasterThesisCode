# G7 — Systematic-error budget (soundness gate, 2026-07-02)

Paper-ready inventory of every known systematic on the H₀ inference, with magnitude, status, and
provenance. "FIXED" = landed on `physics/derail-completion-4pi` (PR #18) with tests; "QUOTED" =
deliberately not absorbed, to be stated in the paper; "CAMPAIGN" = quantified/re-measured during
the Phase-2 production campaign.

| # | Source | Magnitude on H₀ | Status | Evidence |
|---|---|---|---|---|
| 1 | Missing dt² DFT normalization in the inner product | SNR ×1/10, CRB σ ×10; population depth z ≤ 0.11 instead of ≲ 1.5 | **FIXED** `fcc49c4` | `docs/derivations/G8_dt2_inner_product_derivation.md` |
| 2 | Bare-Gaussian photo-z numerator (Eddington-in-z, ∝ σ_z²) | −2.4% MAP bias at σ_z = 0.035; ≈0% P–P coverage | **FIXED** (volume_deconv default `235b783`; kernel 6d4c4e1) | d2 coverage test; `docs/derivations/G2b_host_z_volume_prior.md` |
| 3 | Completion-term peak-density sky evaluation | ~5000× B_num inflation → rail at grid edge | **FIXED** (cb16142, 1/(4π) marginal) | `docs/derivations/G2a_...md`; de-rail matrix |
| 4 | Completion-term solid-angle Jacobian sin θ_det | median 1.15× (mean π/2) per-event completion over-weight | **FIXED** `4a259b7` | G2a Eq. (10); `test_completion_sky_marginal.py` |
| 5 | Global-denominator Option-A cancellation (Σ_global vs h³β_G) | −17% end-to-end h-shape tilt → 'global' mode rails | **AVOIDED** (local modes; 'global' deprecated + warned) | `docs/gates/G1_beta_g_check.md`; ablation cube `G3_ablation_cube.json` |
| 6 | Ω_m fiducial vs true universe | population-matched 0.2726 (`bdf5339`); if truth is Planck 0.3153: +0.3% (z≈0.1) → +2.6% (z≈1.0) → +3.3% (z≈1.5) on H₀ | **QUOTED** (population-weighted ≈+1.5–2.5% post-dt² pop; Ω_m marginalization = future work) | this gate (brentq d_L matching); Barausse 2012 arXiv:1201.5888 |
| 7 | Heliocentric vs CMB-frame host redshifts | ≈+0.15% net (measured, §3.19) | **CAMPAIGN**: rebuild catalogue with z_cmb (col 28; fix on main `7021f6f`) — the current live 8-col catalogue used z_helio | commission caveat; docs/H0_BIAS_RESOLUTION.md §3.19 |
| 8 | Waveform-timeout selection (30 s inj / 90 s sim asymmetry) | 0.6–1.25% of events per stage removed; does not cancel in p_det; bounded sub-% on H₀ | **CAMPAIGN**: params now logged at all catch sites `4d1c27a` → bin by (M, e₀, p₀); expect higher rates at the deeper post-dt² population | `docs/gates/G9_timeout_scan.md` |
| 9 | Eddington-in-M (mass channel not volume/rate-deconvolved) | second-order: ∝ σ_M² d ln(dn/dM · R_eff)/dM; unquantified | **QUOTED** caveat on the 2-D channel (or fix analogously to #2 pre-campaign) | G2b deviation 3 |
| 10 | Unseeded MC in the 4-D denominator | ~1% per-likelihood run-to-run noise, non-reproducible | **FIXED** `2f094d3` (deterministic per-host streams; --seed threaded) | CHANGELOG G4 entry |
| 11 | Fisher matrices with κ > 10¹⁴ | stored CRBs were numerical noise (<2 significant digits) | **FIXED** `d17230d` (hard gate skips event); count the exclusions in the campaign | G10 commit |
| 12 | h-grid truncation/interpolation | expected negligible on the Δh = 0.001 core schedule (posterior width ≈ 0.03) | **CAMPAIGN**: finer-grid robustness check on the final posterior | inference-audit gap item |
| 13 | σ_z model 0.013(1+z)³ in `datamodels/galaxy.py` | none — production reads per-galaxy `z_err` from the catalogue; formula is legacy synthetic-catalog code only | **NON-ISSUE** (downgraded from known-bug 9; delete with the legacy class post-paper) | grep: `HostGalaxy.z_error` ← `REDSHIFT_ERROR` column |
| 14 | wCDM parameters silently ignored in `dist()` | none for the ΛCDM paper (w₀ = −1, w_a = 0 self-consistent) | **QUOTED** as scope limit; fix before any dark-energy extension | known-bug 6 |
| 15 | GLADE completeness at z ≳ 0.3 (binding after dt²) | forecast-defining, not a bug: completion term dominates the deep population | **CAMPAIGN**/paper: report w_G(z) split; F5 σ_z×σ_M frontier covers the information content | G5 notes; F5 forecast |
| 16 | M1 population-shape approximation (dN/dz extracted at h = 0.704, injected truth 0.73) | population-model choice; affects rates/shape, not estimator calibration (P–P closes at injected truth) | **QUOTED** as forecast assumption | Barausse 2012; `M1_model_extracted_data/` |

## Reading of the table

- Every **estimator-calibration** systematic found by the gate (#1–#5, #10, #11) is fixed and
  regression-tested; the estimator's remaining error budget is dominated by **model-scope
  choices** (#6, #16) that the paper states rather than absorbs — standard forecast practice.
- Rows #7, #8, #12 are cheap campaign actions: rebuild catalogue with z_cmb, read the new timeout
  histograms, one finer-grid pass.
- Row #9 is the only open estimator-side item: fix analogously to #2 or carry as a stated 2-D
  channel caveat. Decision before the campaign.

## Numbers behind row #6 (fixed-Ω_m mis-specification, h_true = 0.73)

| z | assume 0.2726, true 0.3153 | assume 0.25, true 0.3153 (pre-G11) |
|---|---|---|
| 0.05 | +0.16% | +0.25% |
| 0.1 | +0.32% | +0.49% |
| 0.3 | +0.94% | +1.45% |
| 0.5 | +1.50% | +2.33% |
| 1.0 | +2.60% | +4.08% |
| 1.5 | +3.31% | +5.25% |

(h′ solving d_L(z; h′, Ω_assumed) = d_L(z; 0.73, Ω_true), repo `dist()`.)
