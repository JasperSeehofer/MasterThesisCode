# m-highz-completion — chair re-derivation + booking (2026-09-04 ~16:20 CEST)

Read of record: READ_RECORD_rev4.md + highz_completion_result_read.json (disjoint reader, real mode
once, after four build/gate rounds; gates: computability GREEN, formula rev3 GREEN, rev4 GREEN on the
PIN CORRECTION 4 tolerance change). Every pin passed: file md5/sha256, population sha256s (606/144/231;
493/111/191), harness manifest (67), G-1 closure max 3.6e-15, G-1d ≤ 1e-6, G-2(i) mean_h anchors,
G-2(ii) Δ_K = 0.0861064 (anchor 0.086106), G-2(iii) 0 exclusions, G-3 a–e.

## Re-derived by the chair from the JSON (MATCH)
| quantity (iiib 2D) | value |
|---|---|
| Δ_F (all terms of the 144 K_dark events frozen to the R-median profile) | +0.064320 |
| Δ_B (B_num term alone) / Δ_g (g term alone) | +0.059585 / +0.001912 |
| shares s_B = Δ_B/Δ_F, s_g | 0.9264 (chair: 0.059585/0.064320 = 0.9264 ✓), 0.0297 |
| non-additivity r = Δ_F − Δ_B − Δ_g | 0.002824 (4.4 % of Δ_F; band 0.6) ✓ |
| Δ_D (completion-denominator identity) | 0.0 exactly |
| Δ_K (159 leave-out) / Δ_K,dark (144) / concordance | 0.086106 / 0.078911 / 0.916 |
| replicates | iiib 1D s_B = 1.0; joint_r1 2D s_B 0.928, Δ_F 0.047; joint_r1 1D s_B 1.0 |
| stencil score excess (nats/h): B / g, iiib | −0.747 ± 0.027 (Z −27.7) / −0.0369 ± 0.0010 (Z −37.7) |
| harness (67 universes, 2D): pooled S_B / S_g | −0.776 ± 0.0098 / −0.0353 ± 0.0005 |
| S_F^harn, Z_harn; ρ_S = S_F^harn / S_F^prod | −0.811 ± 0.010, Z −81.3; ρ_S = 1.035 (B: 1.039; g: 0.958) |
| s_t^harn (B) | 0.956 |
| replicate rule | TERM-OWNS(B) in all four families; no downgrade |

## Dispositions (registration §5, script-evaluated, chair-confirmed)
Tier 1 (production): **TERM-OWNS(B)** — the B_num term (the fused-selection completion numerator,
which carries p_det × S̄_φ for a zero-candidate event) owns 93 % of the high-z pull; g (the
completeness fraction) is immaterial (3 %); the denominator cancels identically. Tier 2 (harness):
**ESTIMATOR-INTERNAL candidate** — the self-consistent harness universe reproduces the same pull with
the same owning term (ρ_S 1.04, s_B^harn 0.96, |Z_harn| 81).

## Booking (chair-derived; returns as fresh RULE R24)
The Graph 2 mechanism node closes its first pass: the 2D offset carried by the 144 high-z
zero-candidate dark events is, to 93 %, the h-dependence of the B_num completion-numerator term for
those events — and the estimator does the same thing on a universe where it is correct by
construction. Under R12's convention (harness reproduces ⇒ estimator-internal) this is the
ILLEGITIMATE candidate; under the alternative reading flagged by author D (R-b: a calibrated
estimator satisfies E[∂_h ln L | z] = 0, so a harness that reproduces the pull may simply be showing
the correct likelihood's h-dependence for information-poor high-z dark events) it is
FLOOR-CONSISTENT. **The mapping is the author's ruling (R24), not the chair's.** Facts that bear on
it: (i) freezing the 144 events' term profiles to the low-z reference removes the ENTIRE 2D offset
(+0.0643 vs −0.0641); (ii) the harness's S3 coverage was itself a DEFECT-SIGNATURE in the
catalogue-hosted class (row #335) while its dark class was score-clean — the dark-class pull here
is not a score-zero violation but an h-slope of B_num; (iii) row #347's matched-channel score put
only 26 % of the dark-class residual in the harness — a different statistic (score at truth vs
h-slope of the leave-out); reconciling the two is the next registered question.
Next node (proposal, not launched): b-highz-bnum-factor (author D's conditional follow-up) — split
B_num into its p_det factor and its S̄_φ factor at the integrand level (catalogue-dependent) to name
which factor's z-weighting carries the slope.

## NOTE D16 (end-verification, 2026-09-04): the G-1 closure identity holds on P_dark only (max 3.6e-15); over all 1588 events (hosted included) the two-term identity fails by up to 16.7 nats, as expected — the gate is correctly scoped to zero-candidate events; stated explicitly here.
