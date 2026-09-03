# m-offset-subset — chair re-derivation + booking (2026-09-04 ~01:55 CEST)

Read of record: READ_RECORD_rev5.md + offset_subset_result_read.json (disjoint reader, real mode
once, after five build/gate rounds; gates: computability GREEN, byte-id 30/30, formula rev3 40/40,
rev4/rev5 GREEN; sha256 both tables + md5 both logL tables matched; 1588/1588 joined per venue).
Blindness: the covariate table was hashed before the join; the join was performed once, by the
reader. The influence ranking is public by registration (row #342 anchors).

## Re-derived by the chair from the built CSVs (iiib, k = 82; MATCH)
| covariate | reader | chair | note |
|---|---|---|---|
| C4 z (AUC S > bulk) | 0.872, p 6e-30 | 0.872 | median z: S 0.85 vs bulk 0.48 |
| C10 log10 M (AUC) | 0.741 | 0.741 | median 5.90 vs 5.75 |
| C2 exact-zero dark (OR) | 0.128 | S 18 % dark vs bulk 64 % | S is ENRICHED in "hosted by exact-zero" |
| S influence sign | — | 100 % of S pull h DOWN; ΣS = +0.0285 vs Σall = +0.0019 (directional units) | the 82 carry ~15× the net |

## Separation table (iiib 2D primary; 1D and joint_r1 agree in kind)
SEPARATES (Holm): C4 redshift (AUC 0.87 / 0.98 1D / 0.91 jr1), C10 mass (0.74), C7 candidate count
(AUC 0.27: FEWER candidates), C3c log10 f_cat (AUC 0.29: LOWER catalogue-leg fraction), C2
exact-zero label (OR 0.13: enriched in hosted-by-exact-zero). WEAK: C5 cone area (0.65), C6
retention. NULL: C1 TRUE in-catalogue status (OR 2.1, p_holm 0.22), C3 relative label (2D; SEPARATES
in 1D/jr1), C8 P6-outside. NOT-TESTED: C10b (n = 5). REPORTED-ONLY: C11 SNR (AUC 0.23, lower SNR).
Materiality (top/bottom decile leave-out, 159 events, frozen T0; null 99 % band ±0.01):
C4 top-z decile Δmean_h = +0.086 (1.9× the oracle 0.046 of leaving out S itself) · C3c bottom
decile +0.034 · C7 bottom decile +0.034 · C10 +0.005 (immaterial) · C2 (see flag).

## Booking (chair-derived; returns as fresh RULE R15)
Primary family iiib_2d: SUBSET-IDENTIFIED before the 1D-agreement trigger; the trigger fires
INTERMEDIATE only because iiib_1d has no ln L matrix under the current data contract (its
materiality is empty by construction), not because 1D disagrees on separation (it separates the
same covariates, more strongly). **Booked INTERMEDIATE by the literal table, with the primary
reading SUBSET-IDENTIFIED disclosed.** R14 line: (a) exact-zero label and (c) continuous f_cat
separate; (b) the relative label does not (2D).

## Physical picture (facts for the decider, no ruling)
The 82 events that carry the 2D offset are HIGH-z (median 0.85 vs 0.48), slightly higher-M, LOW-SNR,
with FEW catalogue candidates and a small-but-nonzero catalogue-leg fraction — i.e. events the
estimator labels "hosted" by exact-zero support while the catalogue leg is negligible; their TRUE
in-catalogue status is irrelevant (C1 null). Removing the top-z decile alone moves mean_h from
0.667 by +0.086 — past truth (0.73): the high-z tail over-pulls h DOWN. This is the depth-skew of
row #287-era findings (73 % of catalogue-leg weight below true z) made event-level, and it is
consistent with rows #335 (S3 defect in the catalogue-hosted class) and #347 (74 % of the
completion residual production-only). Candidate mechanism for Graph 2: at high z the catalogue
is incomplete → few candidates → the completion leg should dominate but the tiny catalogue leg
still enters with a low-z-biased weight (the WBHZERO/completion-denominator balance).

## Flags
F1 C2's materiality stratum removed the level == False (606 dark) events although S is enriched in
True (hosted) — the "enriched via OR ≥ 1.0" rule mislabels enrichment when OR < 1 encodes S
depletion in the True level; Δ = +0.156 there is the effect of removing the dark class, not the
registered stratum. Verifier to rule; does not touch C4/C3c/C7.
F2 iiib_1d and joint_r1 have no ln L matrices under the launch block → their materiality is empty
and the 2-of-3 replicate rule for materiality could not be evaluated (separation replicates 3/3).
Data-contract amendment for the author.

## ERRATUM (end-verification D13/D12/D14, 2026-09-04 ~02:20 CEST) — the chair inverted the C2 label
`C2_hosted_exact = True` means HOSTED (75/76 true in-catalogue events are True; False = the 606
exact-zero dark class). S = 15 True / 67 False → **S is 82 % exact-zero DARK vs bulk 36 %**. The chair's
"S 18 % dark" read the hosted fraction as the dark fraction. Consequences:
- Flag F1 is WITHDRAWN: the code removed exactly the registered enriched level (False = dark, n = 606);
  Δ_strat = +0.156 is the registered number (removing the whole dark class moves mean_h 0.666 → 0.822,
  past truth — a statement about the dark class, not about S). Removing the hosted level instead gives −0.066.
- The physical picture is CORRECTED: the 82 offset-carrying events are high-z (median 0.85), low-SNR
  DARK events with NO catalogue candidate in the cone (C7 bottom decile = C3c bottom decile = the
  159 n_cand = 0 events, hence the identical Δ = 0.0343; D14). They are pure completion-leg events.
  This is directly the dark-class completion-leg residual of row #347 (74 % production-only),
  localized to the high-z, zero-candidate tail — and consistent with cone loss being immaterial
  (those events have no host to lose) and with C1 (true in-catalogue status) being null.
- D12: Holm was applied at m = 10 (C10b NOT-TESTED excluded) vs the draft's m = 11; C1 p_holm 0.325
  either way; no verdict changes.
- Booking unchanged: INTERMEDIATE by the literal (vacuous) 2D/1D trigger; primary SUBSET-IDENTIFIED
  disclosed; separation replicates 3/3 for C2, C3c, C4, C7.
Candidate mechanism for Graph 2 (corrected): the completion leg at high z, where the catalogue has
no candidate at all, pulls h down — the completion-denominator / missing-completion balance for
zero-candidate cones, not a catalogue-leg weight issue.
