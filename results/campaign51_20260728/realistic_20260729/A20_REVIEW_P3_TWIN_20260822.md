<!-- A20 clean-context adversarial review of the [P3-IMP] catalogue-leg twin measurement (Opus,
artifacts + registration text only, banked verbatim 2026-08-22 ~09:35). Orchestrator note: the
decisive re-referenced primary (+0.015524 ± 0.003657, banked trapezoid fleet bias −0.108302) was
independently re-derived through the committed scoring path before the amendments were adopted —
exact match on all three pairings. -->

# A20 ADVERSARIAL REVIEW — [P3-IMP] "catalogue-leg twin"
**Recommendation: BANK-WITH-AMENDMENTS.** The label (REPORT-BOUND) survives every correction I can construct. The headline *number* does not.

## FATAL
None. No gate invalidated; the instrument cell is correct; the pairing is genuine (banked and φ CSVs have identical event_idx sets for all 12 seeds).

## MAJOR-1 — Δ̄(12) pairs a corrected-convention arm against a superseded-convention reference [MEASURED]
mean_h(φ) uses the corrected `trapezoid` weights (row #145, 24921db3); the banked JSONs' mean_h is the superseded `legacy_gradient` value (eb1328de predates the fix; verified by exact reproduction to 9 s.f. on all 12 seeds; systematic offset +0.003733). Recomputation: as-used +0.019257 ± 0.003704 (reproduces the verdict exactly — the arithmetic is sound, the reference is wrong); consistent trapezoid/trapezoid **+0.015524 ± 0.003657 (4.24σ)**; legacy/legacy +0.015769 ± 0.003854 (4.09σ). The trapezoid convention is the one of record (banked trapezoid fleet bias −0.108302 = the −0.1083 headline). Band verdict unchanged under every pairing ⇒ MAJOR, not FATAL. Gate gap: R-P3 verified diagnostic COLUMNS, never the reference STATISTIC — new A17 rule (re-derive the baseline statistic through the arm's own scoring path and gate it, before any Δ).

## MAJOR-2 — the verdict quotes seed-900101 values as fleet figures [MEASURED]
Impostor decomposition under φ: fleet +0.06366 ± 0.0090 (sd 0.0312, range 0.0248–0.1128) = 80.6% of the coded −0.079, not "+0.076 / ~96%". Score-at-truth full: fleet −0.21145, not −0.197. Directions unchanged.

## MAJOR-3 — "misses MATERIAL by 0.0007" is a smuggled materiality claim [NARRATIVE]
Under the corrected reference the gap is 0.00448 (6.4×); REPORT-BOUND was registered "first-class … no label" and the verdict supplied the label rhetorically. Distance-to-anchor phrasing withdrawn, not restated.

## MINOR-1 — K-flat's level/slope split is constant-choice-dependent and the constant is not h-constant [MEASURED]
c = 0.26986/0.27408/0.27963 at h = 0.60/0.73/0.86 (unweighted grid-mean) vs operative per-event median 0.353 (range 0.017–0.941); c(h) itself h-sloped +3.6% (d ln c/dh ≈ +0.137 vs the real factor's median ≈ +0.150) — the "slope" residual is per-event/per-host heterogeneity, not the factor's h-tilt. Offset-robust: re-referenced level +0.039283, slope −0.023639 (slope stands as published).

## MINOR-2 — GATE-LEV compares a score-scale prediction to a mean_h-scale threshold [MEASURED]
Realized primary 7.9× smaller than the "prediction"; the gate's resolvability job was discharged correctly (4.2σ realized), but the LEV↔completion-twin +0.122/+0.125 coincidence must not be carried as corroboration. The two disclosed substitutions undermine nothing the verdict relies on.

## MINOR-3 — fired-gate evidence not banked [PROVISIONAL]
The AMENDMENT-2 fired work root was overwritten by the re-run; the discriminator lived in an ephemeral scratchpad. Chain sound in substance (independently reproduced: banked CSV recomputes to 1.3e-14 via the current canonical path); record to be made artifact, not prose.

## MINOR-4 — the R-P3 fallback justification is contradicted by its own evidence [MEASURED]
Two independent runs on different commits agree to 17 digits (L_cat 1.3476160126670065e-14; combined 5.206628746432402e-15) — the runs are deterministic; the "multiprocessing float order" cause is wrong; the 1.3e-14 residual vs the banked CSV has an unidentified deterministic source (immaterial in size). This cross-run bit-identity is the strongest available evidence for the off-default byte-identity.

## PASS items
(1) A21 identity across all three amendments — git ordering vs artifact mtimes consistent; AMD-3 + band freeze after pilot data existed but outcome-insensitive (the max() resolved to the pre-registered 0.02 anchor under every realized SEM) and "may only TIGHTEN" respected; fleet seeds carry the post-freeze commit. (2) Independent recomputation done (exact). (3) The twin cell: same table object (:2078→:4187→:4694), off default byte-identical (R-P3 + MINOR-4), the with-BH r[0] concatenation handled at both call sites, all four dispatch branches covered, phi_flat rewritten pre-dispatch. (4) K-flat arithmetic internally exact. (5)/(6) covered above.

## Verdict recommendation
**BANK-WITH-AMENDMENTS** — amendments 4–7 as adopted verbatim into PREREGISTRATION_P3_TWIN_20260822.md.
