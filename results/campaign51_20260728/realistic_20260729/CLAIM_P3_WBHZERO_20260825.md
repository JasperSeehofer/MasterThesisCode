# CLAIM [P3-WBHZERO] — the with-BH candidate mass pre-filter asymmetrically excludes true hosts (stage 0; PRODUCTION-DEFECT-CANDIDATE, [AGENT] pending Gate-B verification)

**Opened:** row #191 (the b0i2d pilot's 8.3% exact-zero with-BH numerators, both arms).
**Forensic (2026-08-25, [AGENT] — banked verbatim below the fold):** the zeros are a
COMBINATORIAL EXCLUSION upstream of all numerics: `get_possible_hosts_from_ball_tree`
(`galaxy_catalogue/handler.py:634-642`) applies `sigma_multiplier` (1.5) to the GW mass
uncertainty but NEVER to the galaxy's own `BH_MASS_ERROR` — an asymmetric ±1.5σ-vs-±1σ
window. Event-113 walkthrough: the candidate sits at −1.13…−1.27σ of its own catalogue mass
error (an ordinary 13–18% tail) yet is rejected; the would-be mass-kernel value is an
unremarkable 0.0082 (NOT underflow). All 7 pilot events share the signature
(n_cand_nomass = 1, n_pass_mass_filter = 0).

**Production relevance ([AGENT], zero-compute recounts):** the zero-with-BH-live-no-BH class =
5.0% of the banked b0i fleet rows (unlinked venue — NOT a linked-venue artifact) and
**43.3% (run_20260804_postfix/iiib) / 30.7% (joint_r1) of real production diagnostics rows**
at h = 0.73. Direction: silently starves the with-BH catalogue channel toward
completion/no-BH — magnitude and h-dependence UNMEASURED.

**Refute by (Gate B, the cheapest decisive tests):** (i) independent re-read of
`handler.py:634-642` + the documented intent of `sigma_multiplier` (is the asymmetry a
DESIGN CHOICE anywhere on record?); (ii) independent recount of the production 43.3%/30.7%
and the per-event candidate-set forensics on 3 fresh events; (iii) the counterfactual
symmetric-window count (how many exclusions would a ±1.5σ-both-sides window retain) —
zero-compute from the catalogue columns.

**Interaction with [P3-2D] ([ORCH-DECIDE], sequencing):** the filter is COMMON to both arms
and to the RHS scorer (all score through production paths), so the registered identity test
remains internally consistent GIVEN the filter — but running it before the filter ruling would
calibrate the twin against an eligibility model that may be about to change. **[P3-2D] fleet
stays HELD until the author rules on this claim.** The M2-LINK(iii) monster clause is
re-attributed by this forensic (filter exclusions, not unlinked masses) — A21 amendment to the
[P3-2D] prereg queued with the ruling.

**Exoneration check:** no prior exoneration covers the candidate mass filter (searched the
claim files' Exonerated lists + ledger §2 — the with-BH channel's candidate ELIGIBILITY has
never been an arm in any campaign).

---

## GATE-B VERIFICATION (2026-08-25; clean-context adversarial, inherit/xhigh; banked verbatim summary — full findings in the ledger row #196 statement)

**Adjudication: DEFECT (candidate-confirmed; the claim stands).** Highlights of the
independent re-derivation: (1) the asymmetry real as characterized; design intent
UNDOCUMENTED (thesis-era commits 75b5e2a6/33d0082a, pre-gate-ledger, no rationale anywhere;
MATH_REVIEW F5 + IDEALIZATION_LEDGER I4/I7 flag the window but never ratify the asymmetry;
the neighboring z-filter shares the same unwritten convention — consistency, not a record).
(2) **iiib: 688/1588 = 43.3% confirmed AND fully attributed — 688/688 fall exactly in the
reconstructed "mass filter emptied a non-empty z-passed ball" class, zero residue; the
empty-ball alternative is the DISJOINT 606-row both-zero class; a symmetric ±1.5σ window
retains ≥1 candidate in 689/689.** (3) joint_r1: 30.7% numerically confirmed; structural
attribution UNDETERMINED (runs on observed-catalogue realization r1; deciding artifact = the
r1 catalogue CSV, cluster-side). (4) Pilot 7/7 reproduced (event 113 at 1.12σ of its own mass
error); symmetric retains 5/7. Fleet: 127/129 analyzable zeros filter-emptied (2 = a distinct
rare kernel-zero class, ~1.6%; 20 lack CRB artifacts). (5) **No normalization counterpart
anywhere** (Σ^4D sums all masses event-independently; B_num_wbh none) — unmodeled ONE-SIDED
numerator selection: a bias-mechanism class for the with-BH mixture weight (toward
completion/no-BH), magnitude and h-dependence unmeasured. **Claim AMENDS:** the exoneration
check corrected (CODE_INVENTORY.md §7 touches this filter on the membership question — a
different axis, not covering starvation); the "149" quotation decomposed (140 fleet + 9
eb0a/replica). Verification scripts + JSONs preserved in the session scratchpad
(counterfactual_symmetric.py, prod_reconstruct.py + outputs).

**STATUS: Gate-B-verified DEFECT-CANDIDATE — the disposition (retroactive ratification of the
asymmetry as a design choice vs a physics-change-gated fix) is the author's [RULE].**
