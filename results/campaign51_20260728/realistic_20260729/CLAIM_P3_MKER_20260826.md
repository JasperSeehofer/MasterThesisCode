# CLAIM [P3-MKER] — the with-BH mass kernel's uncertainty budget is incomplete, and the eligibility window has no derivation: "kernel first, window second" (stage 0)

**Opened:** 2026-08-26, author grant (verbatim): *"please open it as suggested by you"* — in
response to the orchestrator's succession proposal for the kernel-zero problem (row #205's
in-fleet exhibit; the proposal's structure is restated in §3 below). Thread tag `[P3-MKER]`
(mass-kernel consistency). **Sequenced AFTER the [P3-2D] verdict** (§5) — the 2D twin's
`mz_sel` object is where this kernel lives; re-deriving the kernel mid-calibration would
invalidate the twin's comparison frame.

## 1. The claim (two parts, both correctness-class, NOT bias-driver-class)

**(a) Kernel:** the with-BH mass likelihood weights candidates by a width dominated by the
GW-conditional σ_cond (production p50 fractional ~1e-8) and does NOT convolve the full
uncertainty budget of the CATALOGUE-side mass: the R&V15 mass-relation intrinsic scatter
(~0.55 dex) is omitted [DOC: [[mass-relation-reines-volonteri]]; 3 related bugs [PHYSICS]-fixed
in 555f018; the log-normal refactor DEFERRED — this thread subsumes that deferral]. A
candidate compatible at ~1σ of its own catalogue error can therefore carry kernel weight
~e^{-k²/2} with k ~ O(10) — physically wrong as a statement about mass compatibility.

**(b) Window:** the eligibility window's k = 1.5 (now symmetric, rows #198–#202) has no
derivation on record (Gate-B row #196: undocumented). It is a physics choice where none
should exist: the window ought to be a TRUNCATION BOUND on the correct kernel — k derived
from a stated tolerance ε (excluded weight < ε of the numerator; Gaussian: k = √(2·ln(N/ε)) on
σ_eff), making the filter an ε-controlled instrument constant with a limiting case (ε→0 ⇒ no
filter ⇒ the exact model).

## 2. Evidence at intake

- **[LOCAL] the in-fleet exhibit (rows #205):** seed 900121 event 20 —
  `L_cat_with_bh = 1.39e-85` with n_sym = 2 window-passed candidates (~1.4σ_g inside the
  window, ~19σ_kernel under the narrow kernel; −176.6 nats). Re-measured this session from
  the fleet artifacts + the zero-compute reconstruction
  (`p3_2d_fleet_20260825/m2link_iii_reattribution_check.json`).
- **[DOC] the scatter omission:** the R&V15 intrinsic scatter (~0.55 dex) omitted from
  host-mass errors (memory + commit 555f018 record; "host-mass errors ~3–7× too tight").
- **[DOC] the window's non-derivation:** Gate-B row #196 (design intent undocumented;
  MATH_REVIEW F5 + IDEALIZATION_LEDGER I4/I7 flag the window, never ratify a k).
- **[DOC] the row-#196 fleet forensic:** ~1.6% of analyzable zeros were the kernel-zero
  class (2/129) — the class this thread names.

## 3. Succession structure (the author-granted proposal, restated)

1. **Kernel first:** derive the convolved with-BH mass kernel — GW width σ_cond ⊕ catalogue
   σ_g ⊕ the mass-relation intrinsic scatter (log-normal, subsuming the deferred refactor) —
   as a 6-item physics-change package (derivation, dimensional analysis, limiting cases:
   scatter→0 recovers the current kernel; σ_cond→0; validity conditions incl. the R&V15
   regime check per the Stage-L assumption register).
2. **Window second:** re-derive k from ε on the ratified kernel's σ_eff — the window ceases
   to be physics.
3. **Measure-first throughout** (the [P3-WBHZERO] pattern transfers): counterfactual flag,
   byte-identical default → mirror-venue paired measurement → production counterfactual
   read → package → author [RULE]. No adoption before measurement.

## 4. Delimitation against the standing exonerations (hard-rule-1 check, PASSED with scope)

Ledger §2 item 1 exonerates the **mass-kernel FAMILY as the 2D-bias driver** (twice: Δ2D
+0.0029 wrong sign #72; 4-cell A/B MAP-unmoved #89; bounded +0.002). **[P3-MKER] does NOT
re-open that claim.** This thread's claim is model-consistency/correctness (the author's
standing values ruling: correctness outranks bias-removal); its H₀ effect may well be small
and is NOT the motivation. Any H₀-effect statement this thread produces must be checked
against that exoneration's bound before banking. Also honored: the honest caveat from intake —
the convolved kernel WIDENS the with-BH channel (σ_M forecast: the with-BH H₀ rescue needs
σ_M ≲ 1–2%), so the correct kernel likely makes the channel more honest AND less informative.

## 5. Sequencing and cheapest decisive measurements

- **HELD behind the [P3-2D] verdict** (the twin calibrates against the current kernel; the
  kernel change re-enters through the twin's own machinery afterwards).
- **Cheapest decisive reads available NOW (zero-compute, rule A1/9):** (i) the fleet-wide
  census of the kernel-zero class (extend the row-#205 scan to all window-passed candidates:
  distribution of kernel-pulls vs window-pulls — quantifies how often σ_window ≫ σ_kernel
  bites); (ii) recompute the 900121:20 kernel value under a convolved σ_eff (analytic, one
  formula) to confirm the exhibit dissolves.
- **Refute by (the claim's own falsifier):** (a) produce a documented derivation showing the
  current narrow kernel is the CORRECT conditional likelihood given the pipeline's
  generative model (i.e., the catalogue mass is treated as exact BY DESIGN elsewhere in the
  chain, consistently in numerator AND normalization) — that would convert (a) from defect
  to design choice; (b) show the σ_M forecast's regime makes the convolved kernel's effect
  numerically indistinguishable (< the twin's ε₂ scale) in EVERY consumer — that would
  demote the thread to documentation-only.

## 6. Stage-L R0 sweep (mandatory at stage 0)

Launched at intake (lightweight): re-read of the already-cited mass-relation and dark-siren
host-weighting papers (R&V15 itself; Gray et al. 2020's host-weighting treatment; the
fastemriwaveforms/EMRI mass-precision references) for stated validity conditions on
catalogue-mass uncertainty treatment. Results append below this line when banked.

---

## R0 SWEEP RESULTS (2026-08-26, [AGENT] sonnet, symptom-card-only; quote-verification per Stage L; banked verbatim summary)

- **[LIT-1, HIGH]** R&V15 §IV.1 (ar5iv full text, quote-verified): "The rms deviation of the
  BH mass measurements from the relation is 0.55 dex, and incorporates both our adopted
  measurement errors of 0.50 dex and a best-fit intrinsic scatter of 0.24 dex (added in
  quadrature)." Sample validity: 262 broad-line AGN, z < 0.055, 10⁸ ≤ M_*/M☉ ≤ 10¹².
  Cross-confirms `docs/MASS_RELATION_ASSESSMENT.md` §2. The single most decisive
  already-cited fact for part (a).
- **[LIT-2, MEDIUM]** Gray 2020 G20-d (already two-fetch-verified in
  `docs/LITERATURE_WARNINGS.md`): host-weighting validated only at 25–75% completeness; our
  venue at 4.79% in-catalogue share — out of the source's validated range.
- **[LIT-3, REPORTABLE ABSENCE]** No cited dark-siren methodology paper (Gray 2020/2023,
  MFG19) treats mass-covariate deconvolution / error-in-variables at all — Gray's
  completeness formalism is magnitude/luminosity-threshold-only. The kernel design must be
  argued from first principles or NEW literature (an R2/R3 ring sweep is the stage-2-time
  follow-up), never assumed literature-compliant.
- **[LIT-4, REPORTABLE ABSENCE]** No cited selection-cut/truncation-bias warning has ever
  been checked against the k = 1.5 hard pre-filter (the symmetric proposal's §3 already
  records the no-derivation fact; the cut-on-observed-vs-cut-on-true question is untouched).
- **Bridge to the Refute-by:** Gray 2023 G23-b (§2.1.3) — truncation/renormalization is
  harmless ONLY under numerator/normalization consistency — is status UNCHECKED against our
  mass-kernel code: checking it IS the §5 Refute-by(a) path (if the current narrow kernel is
  consistently exact-mass on both sides, part (a) demotes to a design choice).
