# END VERIFICATION — wave 3 (2026-09-03, top-tier end-verifier, charter §1.14)

Every number below re-derived from raw inputs (checkpoints, CSVs, logs), not from reader tables.
Evidence for the chair, not authority. Scripts run in the session scratchpad; nothing else written.

## 1. rd-s3-readout (92 checkpoints, ladder reference, `kstwo`, catalogue md5)

| quantity | reader/chair | mine | match | source |
|---|---|---|---|---|
| KS crit n=67 / n=25 | 0.163221 / 0.264041 | 0.163221 / 0.264041 | Y | `scipy.stats.kstwo.isf` |
| F S no/with, T no/with | 11.44/11.38, 11.33/11.44 | 11.4394/11.3797, 11.3322/11.4405 | Y | median SD ÷ floor |
| PIT-KS D S / T | .3217/.3340, .3060/.3459 | .321731/.334016, .306030/.345870 | Y | `posterior.*.pit` |
| HPD hits S no/with | 36,39,58,61 / 25,31,54,60 | same | Y | `hpd*` flags |
| HPD hits T no/with | 10,14,22,22 / 10,12,21,22 | same | Y | |
| mean(MAP)−h, Z (harness form) | S .04187/5.89, .05022/6.48; T .0388/2.82, .0488/3.58 | 5.892/6.482; 2.825/3.584 | Y | std(MAP)/√n |
| same Z, **registered form** SEM=σ̄_post/√n_U | — | S 6.00/7.27; T 3.48/4.34 | see D1 | |
| score-zero Z S: cat/dark/all | 9.76,1.26,4.93 / 7.15,1.76,4.26 | 9.759,1.264,4.929 / 7.149,1.756,4.264 | Y | pooled per §1571 |
| score-zero Z T | 3.48,.871,2.49 / 2.94,1.92,3.10 | 3.481,.871,2.493 / 2.935,1.918,3.103 | Y | |
| rail fractions (hi/lo) | S 10/0, 14/0; T 4/1, 6/0 | same (grid .60–.86) | Y | `map_h` |
| byte-pin with_bh vs ladder | S 63/63, T 20/20, 0 diffs | 0 field diffs, max_rel 0.0 | Y | 9 fields incl. ln_post |
| catalogue md5 | c52c13b5… | c52c13b5cab61f6b3f04bbe202550969 | Y | md5 of file |
| g-population | 0 mixed, N=200 | 0 mixed, N=200, cells OK | Y | |
| floor "floor(200)=0.00518915" (reg §2.1) | 0.00518915 | = floor(**180**); floor(200)=0.004923 | see D2 | median n_scored=180 |

## 2. m-s0b-production (5 node CSVs, N=1588)

| quantity | reader/chair | mine | match |
|---|---|---|---|
| score_b_re no/with: mean, SEM, Z | −.6822/.1293/−5.274; −.7412/.1195/−6.204 | identical to 1e-9 | Y |
| score_lns no/with | −.032661/.004544/−7.188; −.036824/.005022/−7.333 | identical | Y |
| Es_null, PA-HIER-33 Z | −.0003762±.0001637, −7.101; −.0001236±.0001282, −7.306 | −.0003762±.0001647, −7.101; −.0001236±.0001301, −7.306 | Y (boot SD seed-noise) |
| curvature: b̂, σ_b, ln ŝ, σ | −.011365, .003239, −1.165182, .149885 | same | Y |
| B0-B / B0-M / B0-P | LIVE / MIXED / POWERED | same under PA-HIER-31(e) text | Y |
| GATE ENG b+/b−/s+/s− | .5447/.4861/.508/.476 | same | Y |
| class split, C-C identity | 1139/449, max Δ 0 | same | Y |
| 157 moved: combined_no_bh ≤1.6e-7 rel | 3.73e-9 abs / 1.58e-7 rel | same | Y |
| "physics identical" (DIAG headline) | asserted | **false on matched class**: vs `c0prime_off_iiib` (= 2026-08-27 baseline bit-for-bit) combined_no_bh max_rel 0.734, 562/1588 events >1e-6, ALL shifted up (+2.58 nats total); with_bh max_rel 1.65, 392 events | see D3 |
| cost 9.98 CPU-h | 2246 s×16 | 2246 s×16 | Y |

## 3. Addendum reads

| record | quantity | reader | mine | match |
|---|---|---|---|---|
| parity-decomp | buckets 19/442, max 5.2324, seed max_rel .44682/.03882/.21601/.04348, wbh 5.9–12% | — | identical | Y |
| 2D bootstrap | row #302 4/4 ≤5e-7; SE .016505/.017434/.012914/.021017; k=82/94/72/46 | — | script rerun identical (seeded) | Y |
| 2D bootstrap §5 | N=1588 SE 0.01358 vs §2 0.01651 | — | 18% apart, >3× 200-draw noise | see D6 |
| timeout | M/e0/p0 counts + rates, 1196/4523, 12.19σ, 6.84σ | — | identical from raw logs; edges reproduce from stated rule (union incl. timeouts) | Y |
| timeout | p0 gradient | 1.24σ (§) / **1.28σ** (band table); denom "4521" | 1.24σ; 4523 | cosmetic |
| c0prime parity | max_abs no_bh 4.846 / 0.5503 (num_log); mass_aware of named comparand = `auto` | — | confirmed (`run_metadata_21.json`: auto); STOP correct | Y |

## 4. m-cone-loss (census rebuilt, scores from raw CSV)

| quantity | reader/chair | mine | match |
|---|---|---|---|
| OUT/IN | 10/66 | 10/66 (idx 231,271,298,656,816,883,900,946,1251,1545) | Y |
| Δh 1D, SE, Z, φ, M | −2.7313e-4, 8.756e-4, −.312, .00434, 9.14 | −2.7316e-4, 8.757e-4, −.312, .00434, 9.14 | Y |
| Δh 2D, SE, φ, M | −3.091e-4, 9.058e-4, .00482, 8.83 | same | Y |
| MAD-SD / plain / top-2 | .8401/7.170; 889, 474 | same | Y |
| leave-out Δmean_h | −0.004904 | −0.004903779 | Y |
| "18× non-linearity" | asserted | like-for-like linear = −Σ_OUT s_e/I = −0.00347 (SE 8.2e-4) → 1.75·SE from leave-out | see D4 |

## 5. d-photoz-leverage dossier — arithmetic traceable (b̂, ln ŝ, ŝ=0.31, 3.4×, Z's); two claims not (D3, D5).

## DISCREPANCIES

1. **S3 mean-MAP Z uses std(MAP)/√n, registration §2.1 says SEM=σ̄_post/√n_U.** Registered-form Z: S 6.00/7.27, T 3.48/4.34. Number only (all still OUTSIDE; T no_bh flips 2.82→3.48 but T is unbanded).
2. **Registration §2.1 "floor(200)=0.00518915" is floor(180)** (median n_scored). floor(200)=0.004923 → F would be 12.06/12.00. The delivered F is "at 180 scored events", not N=200. Number only; Branch G must not label it N=200.
3. **DIAG "physics identical"/chair §5 "DIAGNOSTIC-ONLY" is overbroad.** True for the 157 moved events only. The S0-B truth node differs from the mass-aware-matched production baseline on 562/1588 matched events (combined_no_bh up to 73 % rel, every shift positive; with_bh up to 165 %). Cause unexplained (θ-hook site 2.2 is identity-forced at (0,1) per `bayesian_statistics.py:5987`; run had 606 dirty files). Secants are intra-run and unaffected; but PA-HIER-31(d) "truth = C0 baseline gate" and §3.1 GATE T-ID (0 ULP) are **not satisfied** against any local comparand, and the dossier omits it. Severity: changes a claim; the C0/T-ID control status must be disclosed in the dossier.
4. **Cone-loss INTERMEDIATE trigger rests on a mismatched comparison.** Δh_cone is the excess over s̄_IN; the leave-out removes OUT events wholesale. Like-for-like: −0.00347±0.00082 vs −0.00490 → agrees within 2·SE. Chair fact 2 ("non-linearity") is unsupported. Under the registration as literally written the row still fires; but §2 also says the read "is booked on the leave-out number with the flag" (−0.0049, φ=7.8 %) — which is itself IMMATERIAL-FLOOR-SHARE. Severity: changes a booking (INTERMEDIATE ← design artefact).
5. **Dossier "config = row #287 certified"** — row #287 certified b0i, divisor-on, zwin-zk4; S0-B ran iiib, divisor-off, zwin-off (readout §1 lists 3 DEVIATIONS). Claim discrepancy.
6. Bootstrap §5 N=1588 SE (0.01358) vs §2 (0.01651): 18 % apart, exceeds 200-draw noise; no band change.
7. **Pre-flip "pilot" cell T is not pre-flip.** Ladder T files (commit 6c43f8f9, 2026-08-31T21:57Z) post-date the flip (5e7fda16, 11:46Z); T no_bh is byte-identical ladder↔postflip on 20/20 seeds. Registration §0/§2.2 and readout §5 comparands (F_no_bh(T)=11.27, T/S no_bh=1.517) are mislabeled. Booking unchanged.
8. **No checkpoint carries `catalogue_leg_1d_mass_aware`** (registration §1: token is part of population identity). g-population "green" is half-checked. S checkpoints carry 13 different commit stamps, 1112–1117 dirty paths.
9. Cosmetic: timeout p0 1.28σ→1.24σ; "4521"→4523; Es_null boot SD seed-dependent.

## Existence contract — unopenable sources
None decisive. Absent as the readers said: byte-pin script (S3), `global_denom_*`, `run_metadata.json` (S0-B), S0-B dirty diff, seed3000 injection symlinks, correct-venue candidate dump, HPD n=100 orientation derivation.

## Bookings not literally supported
- S3 chair §2: "g-byte-id criterion of row #291 DISCHARGED" — registration §2.3 says *proposed* discharge, routed to d-s4-review; T pin is a same-code rerun (D7), carrying no flip-invariance information.
- S3 chair §4 "F delivered at N=200" — D2.
- S0-B chair §4 "population-ROBUST … physics identical" — D3 (labels robust: yes; physics identical to baseline: no).
- Cone chair "two rows fire → INTERMEDIATE" — literal §4 yes; §2's own resolution rule (book on leave-out) not applied; D4.
- Dossier item 1 presents charter-clause conflict but not the failed C0/T-ID parity (rd-s0b-parity STOP) — D3.

---
## ADDENDUM — m-completion-residual (rev 4), appended 2026-09-03 late

Re-derived from raw: production re-baseline CSV + `seed61000/prepared_cramer_rao_bounds.csv` (`host_galaxy_index<0` = dark), 67 harness `seed9010NN_S/simulations/{diagnostics/event_likelihoods.csv, prepared_cramer_rao_bounds.csv}`, 67 checkpoint `resolved_flags` blocks. Formula per draft §2.1 (β̄ = D_tilde_phi − alpha_G_phi; stencil 0.725/0.735).

| quantity | reader/chair | mine | match |
|---|---|---|---|
| per-event closure max | 2.56e-13 | 8.5e-14 (s_e vs ln combined ≤1.0e-11) | Y |
| T_prod, SE, Z (n=1512) | −0.19663662, 0.01943993, −10.1151 | −0.19663662, 0.01943993, −10.1151 | Y |
| S_G, S_dark, π_G, S_all, residual | 1.207935, −0.114203, 0.047859, −0.05092649, 1.4e-17 | same, residual 0.0 | Y |
| T_harn, SD_U, SE, Z (67 U) | −0.05054134, 0.059931, 0.00732173, −6.9029 | same; range [−0.1901, +0.0883], 79.1 % negative, n_dark 165–184, Σ 11 525 | Y |
| byte-id (full dark score vs checkpoint) | 67/67 | 67/67 to <1e-12; T_full 0.0082159 / SE 0.0063142 | Y |
| ρ | 0.25703 | 0.25703 (δρ ≈ 0.045) | Y |
| δh_M | −0.091323 | −0.091323 | Y |
| resolved-flags = REGISTERED (13 tokens) | 67/67 | 1 distinct block, all 13 values as listed | Y |
| disposition literal row | INTERMEDIATE (b) | draft §4 row reads **0.2 < ρ < 0.5** (strict); chair wrote "0.2 ≤ ρ < 0.5" — immaterial at ρ=0.257 | cosmetic |

**Column-semantics check (chair's flag):** production and harness diagnostics CSVs carry the identical 19-column header; both grids contain 0.725/0.73/0.735; `B_num`, `num_log_term_no_bh`, `den_log_term`, `D_tilde_phi`, `alpha_G_phi` are written by the same `bayesian_statistics.py` diagnostics block in both. β̄(h) = D̃φ − αφ is numerically IDENTICAL production vs every harness universe at all three nodes (893324910 / 888403790 / 883510540) — a catalogue-level global, so s_M is the same quantity in both venues. Harness event_idx (176 scored) ⊂ CRB rows (200); dark labels join cleanly.

**Discrepancies**
- D10 (number/cosmetic, non-gating): §2.9 g-precision "within_tolerance False, rel 5.5e-3" is a filename bug — `f"{h:.2f}"` maps 0.725→`_0_72`, 0.735→`_0_73`, so h=0.72/0.73 full-precision tables were compared to the 0.725/0.735 CSV values. Against the correct `selection_tables_h_0_725/0_735.json`: rel 5.5e-8 / 3.5e-8 → within tolerance. The disclosure text should be corrected; no booking effect.
- D11 (label): row-text "0.2 < ρ < 0.5" in draft vs "0.2 ≤ ρ" in chair; row (c) is "ρ ≤ 0.2", so the boundary at exactly 0.2 belongs to (c) — chair's paraphrase is the wrong side. Immaterial here.
- Note (context, not a discrepancy): δh_M = −0.091 magnitude > rail −0.063 — as the chair says, linear-response over-prediction; same class of mismatch as D4 (cone).

Bookings: INTERMEDIATE (b) is the literal sole-firing row — supported. Rail disclosure 14.9 %/20.9 % carried (disposition_role None) — consistent with registration.

---
## BATCH 2 — m-offset-subset (appended 2026-09-04)

Re-derived from raw: iiib re-baseline `event_likelihoods.csv` (`combined_with_bh`, 41 nodes, gradient weights, no floor needed — 0 zeros), `covariate_table_iiib.csv`, own Mann–Whitney/Fisher(Haldane)/Holm, own leave-outs and null draws (seed 20260904).

| quantity | reader/chair | mine | match |
|---|---|---|---|
| influence vector (2D) | `influence_iiib.csv` | identical to 1e-15 (CSV sign = mean_h(−i) − full; ranks identical); top-10 = 576,94,46,172,201,160,1176,158,1482,55 | Y |
| minimal k / oracle Δ_S | 82 / 0.046234 | 82 / 0.046234 | Y |
| C4 AUC, p_raw | 0.8722, 6.24e-30 | 0.8722, 6.24e-30 | Y |
| C10 / C7 / C3c AUC | 0.7410 / 0.2669 / 0.2923 | same | Y |
| C2 OR (15,67,967,539) / C1 OR / C3 OR | 0.1280 / 2.0546 / 0.5595 | same | Y |
| Holm p (C4,C10,C7,C3c,C2,C1) | 6.24e-29, 1.48e-12, 1.68e-12, 4.05e-10, 2.20e-15, 0.219 | reproduced with **m=10** | Y (see D12) |
| top-z decile Δ, null CI99 | +0.08611, [−0.00909, +0.01075] | +0.086106, [−0.009089, +0.010751] | Y |
| C2 stratum Δ (n=606) | +0.15568 | +0.155678 (remove level False); removing level True (982) gives −0.0655 | Y |
| median z S / bulk | 0.85 / 0.48 | 0.849 / 0.481 | Y |

**(4) C2 stratum — the code is RIGHT, the chair's F1 is WRONG.** `C2_hosted_exact=True` means hosted (75/76 in-catalogue events are True; False count = 606 = the registered exact-zero dark class). S is 15 True / 67 False → **S is 82 % exact-zero DARK vs bulk 36 %**, i.e. enriched in the False level. The script's rule (OR=0.128 < 1 → level False) removed exactly the registered enriched level; Δ_strat = +0.1557 is the registered number. The chair's line "S 18 % dark vs bulk 64 %" and the coordinator's premise invert the label (18 % is the HOSTED fraction). Consequences: F1 should be withdrawn; the chair's physical picture ("events the estimator labels hosted by exact-zero support") is inverted — S is dominated by exact-zero dark events (consistent with C7 fewer candidates, C3c lower f_cat, C1 null). Severity: changes a claim/interpretation, not the disposition.

**(5) Booking.** Literal §5 INTERMEDIATE triggers: "primary 2D and 1D iiib families disagree in disposition" fires only because iiib_1d has no ln L matrix (materiality empty by contract) — vacuous. The reader's second trigger ("C10 SEPARATES but not MATERIAL") is a misreading: the row says "SEPARATES but **no** stratum is MATERIAL", and four strata are material. So INTERMEDIATE is the literal sole-firing outcome via a data-contract artefact; "SUBSET-IDENTIFIED before the trigger" is a fair disclosure, and iiib_1d's separation (C4 AUC 0.98) agrees in kind. Replicate rule 4.3: 3/3 for C2, C3c, C4, C7 — supported.

**Discrepancies**
- D12 (number): Holm run at m=10 (C10b untested, excluded); draft §4.1/§5 say m=11. With m=11: C1 0.325, others unchanged in verdict. No booking effect.
- D13 (claim): chair F1 and the "hosted by exact-zero support" narrative — label inverted (above).
- D14 (cosmetic): C7 and C3c bottom deciles are the SAME 159 events (overlap 159/159: n_cand=0 ⇒ f_cat=0), explaining the identical Δ=0.03431 the reader flagged as unexplained.
- Note: C2 Δ=+0.156 with captured fraction 3.4 means removing the 606 dark events overshoots truth (0.666→0.822): the dark class as a whole pulls h down far beyond S — a population-level statement, not an S-specific one.
