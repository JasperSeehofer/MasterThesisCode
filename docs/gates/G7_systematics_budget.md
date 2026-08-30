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
| 16 | M1 population-shape approximation (dN/dz extracted at h = 0.704, injected truth 0.73) | **MEASURED, calibration-affecting** (re-graded 2026-08-22, ledger row #159 D4): row #138 measured a calibration effect, contradicting the former "affects rates/shape, not estimator calibration" clause | **MEASURED** systematic (was QUOTED forecast assumption) | Barausse 2012; `M1_model_extracted_data/`; ledger rows #138, #159 |
| 17 | V2 measure prefactor (M_z-obs density-vs-ratio, 1/(σ_M·M(1+z)) class, incl. D-ii ratio-form GW factor) | completion leg: ≲1e-6 tilt-neutral deviation at measured σ_cond p50 = 8.8e-8 (immaterial); catalogue leg (σ_M ~ 60–200%): unquantified, deferred with [P3] | **QUOTED** (G2 ruling, ledger row #118: ratio convention retained in BOTH legs; re-opens with the Gray-convention paper task, row #110) | `PROPOSAL_2D_SELECTION_FUSION_VERIFIER_ADDENDUM_20260817.md` MAJOR-4; L6-DER2 addendum bound |

## Reading of the table

- Every **estimator-calibration** systematic found by the gate (#1–#5, #10, #11) is fixed and
  regression-tested; the estimator's remaining error budget is dominated by **model-scope
  choices** (#6, #16) that the paper states rather than absorbs — standard forecast practice.
- Rows #7, #8, #12 are cheap campaign actions: rebuild catalogue with z_cmb, read the new timeout
  histograms, one finer-grid pass.
- Row #9 is the only open estimator-side item: fix analogously to #2 or carry as a stated 2-D
  channel caveat. Decision before the campaign.

## Appended note, 2026-08-29 (orchestrator decision, charter node B3; append-only, row 16 not edited)

**Launched under rows #222/#223 — charter node B3.** Docs-only; row 16 above is left unedited
(append-only). This note sharpens row 16's content per
`results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md`
§12 ("Paper-facing caveat (G7 row 16)"), which is itself downstream of §F's finding that the
mock's production generator and its estimator already share one population law — row 16's
"MEASURED, calibration-affecting" grade (re-graded 2026-08-22, ledger row #159 D4, citing row
#138) described a mismatch that §F shows does not exist **in the mock**; the quantified
sensitivity below is a **real-data-facing systematic**, not a mock-calibration defect.

**Quantified population-shape sensitivity (§12, lines 622–624, quoted verbatim):**

> a shape change of the size between the two M1 implementations in this repository (r(z) from
> 0.53 to 1.39 over z ∈ [0.17, 1.5]; ×0.65 across the band z = 0.39 → 0.9) moves the dark-class
> per-event score at truth by **−0.60** (bins 2–5), a summed slope of ≈ **−290 nats per unit h**
> over 484 events — comparable to the whole measured production tilt and sufficient to rail a
> completion-dominated posterior (row #137: the pure completion class rails at 0.60 from a −0.635
> score).

Source: `PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md:622-624` (§12), 2026-08-29. The r(z)
values themselves (crossover z ≈ 0.17, r(0.392)/r(0.9) = 0.653) are §14-provenanced to
`scratchpad/b32_T_table.json` via `b3_1_pop_measure.{w_true_of_z,w_model_of_z}`, 2026-08-29
(`PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md:666-684`, §14 table).

**Reading for the paper (§12, continued):** for the **mock**, the population prior is exactly the
generator's law — the mock carries **no** population-shape systematic of the row-#138 kind
(self-consistent closure on this axis). For **real data** the EMRI population redshift shape is
unknown; the sensitivity above quantifies how large an unresolved model-scope uncertainty this is
— O(1) relative to the whole production tilt in the completion-dominated regime — and the paper's
honest treatment is fork (b) of `docs/derivations/population_mismatch_dark_score.md` §6
(hierarchical marginalisation over rate-evolution parameters), not a bias-removal fix.

**Branch verdict of record: B3 CLOSED — PREMISE-REFUTED (provenance, zero compute)** — row 16's
re-grade to MEASURED stands for the real-data systematic quantified above, but no longer for a
mock-calibration defect (none exists, §F).

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

## Row 16 re-grade, 2026-08-30 (launched under row #255 -- tree 2 node A5; append-only, row 16 table entry above not edited)

Per ledger row #255, item A5: row 16 is re-graded exactly as ruled --

mock: zero by construction (the injected dark-host law is the estimator prior, section F of
results/campaign51_20260728/realistic_20260729/fanout1_20260829/PHYSICS_CHANGE_POPULATION_PRIOR_M1_20260829.md);
real data: O(1) degeneracy with the population z-evolution -- hierarchical marginalisation
required (shape sensitivity: r(z) 0.53 to 1.39 moves the dark-class score by -0.60 on bins 2-5).

Rows #137/#138 are retired as citations (kept on the record, no longer the basis for row 16's grade).
