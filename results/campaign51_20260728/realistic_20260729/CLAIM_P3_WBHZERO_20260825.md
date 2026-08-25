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
