# ch06_FLAGS.md — Chapter 6 ("Opening the Black Box: What the Waveform Actually Measures")

Raised by the ch06 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, **stop and flag; do not silently reconcile in
either direction**."*

Every item below is also visible to the **reader**, in both forms, on `ch06-black-box.html`.
Nothing here blocks the chapter. Items F-ch06-1 and F-ch06-2 touch numbers that **other
chapters already ship** (Ch 1, Ch 4 dossiers) and need integrator attention.

---

## F-ch06-1 — EMRI-889's distance precision: `σ_dL/dL = 8.0×10⁻⁵` (spec) vs `8.98×10⁻⁴` (measured) — **CROSS-CHAPTER**

- **Spec value.** `BOOK_DESIGN.md` §1 carries "`σ_dL/dL = 8.0×10⁻⁵`" **three times**: the
  Ch 1 card (dossier opening), the Ch 6 card ("the dossier gains `σ_dL/dL = 8.0×10⁻⁵`"),
  and by inheritance the Ch 4 page, which already ships it in its dossier
  (`ch04-loud-half.html`, row `d_L`). `BOOK_PEDAGOGY.md` Q6.5 repeats it ("the distance
  shell is thin to 8×10⁻⁵").
- **Measured by `gen_ch06.py`**, directly from the cited artifact
  (`.../seed61000/prepared_cramer_rao_bounds.csv`, row 889):
  - `sqrt(delta_luminosity_distance_delta_luminosity_distance)` = **7.984×10⁻⁵ Gpc**
  - `luminosity_distance` = **0.0888792 Gpc** (= 88.879 Mpc, matching the spec's "88.9 Mpc")
  - therefore `σ_dL/dL` = 7.984e-5 / 0.0888792 = **8.983×10⁻⁴**.
- **Reading.** `8.0×10⁻⁵` is the **absolute** 1-σ distance error in **Gpc**, printed under a
  *fractional* label. The two numbers are the same measurement in different units
  (`8.0×10⁻⁵ Gpc` = `0.0798 Mpc`); the ratio between them is exactly `d_L` in Gpc.
- **Independent corroboration that 8.98×10⁻⁴ is the fraction.** Two, from different
  directions:
  1. Over all 1590 rows the product `(σ_dL/dL) × SNR` has median **1.040** and p5–p95
     range 0.970–1.250 — i.e. the fractional distance error is ≈ 1/SNR to within 30%
     across three decades of SNR. At 889's SNR of 1424.7 that predicts 7.3×10⁻⁴,
     consistent with 8.98×10⁻⁴ and inconsistent with 8.0×10⁻⁵ (which would require
     SNR ≈ 13,000).
  2. **The project's own readout says the same thing.**
     `IDEALIZED_BASELINE_READOUT.md:47-48` builds its per-event budget from
     "σ_H0/H0 ≈ **0.38 %** / √76 = 0.044 %" — i.e. it treats the *per-event GW distance
     precision of the 76 information-carrying hosts* as ≈ 4×10⁻³. Measured here, those 76
     rows have median σ_dL/dL = 5.3×10⁻³ and an inverse-variance-effective 3.2×10⁻³. All
     three figures are of order 10⁻³. None is of order 10⁻⁵.
- **Disposition (ch06).** Neither number is dropped and neither is silently corrected. The
  chapter prints **both**, labelled: `σ_dL = 8.0×10⁻⁵ Gpc = 0.0798 Mpc` and
  `σ_dL/dL = 9.0×10⁻⁴`, and says in the dossier and in the Q6.5 answer which is which.
  Both are emitted by the generator as `sigma_dL_Gpc` and `sigma_u`.
- **For the integrator.** Ch 1 and Ch 4 carry the mislabelled form. The minimal repair is a
  unit, not a value: `σ_dL = 8.0×10⁻⁵ **Gpc**`. Ch 6 does not edit either page.

## F-ch06-2 — Q6.5's "≈6000× larger" follows from F-ch06-1 and does not survive it

- **Spec.** `BOOK_PEDAGOGY.md` Q6.5 answer: *"The distance contributes ~0.008% to the H₀
  error; a 49% fractional redshift error contributes ~49%, roughly **6000×** larger."*
  0.008% is exactly 8.0×10⁻⁵ read as a fraction.
- **With the measured fraction** `σ_dL/dL = 8.983×10⁻⁴` (0.0898%) the same comparison gives
  0.49 / 0.000898 = **≈550×**, not ≈6000×.
- **Disposition.** The question text is used **verbatim** (rubric D). The answer states the
  measured ratio **≈550×**, shows the arithmetic, names the unit slip explicitly, and points
  at this flag. The *conclusion* Q6.5 exists to deliver is unchanged and if anything
  under-stated by neither figure: the GW distance is between two and three orders of
  magnitude better known than the catalogue redshift, so the redshift is the error budget.
- **Note.** The 49% figure itself is not in question: `IDEALIZED_BASELINE_READOUT.md:50-52`,
  median `σ_z/z ≈ 49%` for the information-carrying hosts. Ch 6 quotes it as a
  forward-reference only; Ch 7 owns it.

## F-ch06-3 — Trap 6.A's "largest exactly for the loud events" is **not** what this run measures

- **Spec.** `BOOK_PEDAGOGY.md` Trap 6.A dismantle: *"the error is largest exactly for the
  loud, nearby, information-carrying events."*
- **Measured** over all 1590 rows, with
  `r_sky,u ≡ sqrt(r_{φu}² + r_{θu}²)` (the sky↔distance correlation magnitude):
  - Spearman rank correlation of `r_sky,u` against SNR: **−0.019** (i.e. none).
  - Quartile medians by SNR: 0.045 / 0.039 / 0.040 / 0.041 (quietest → loudest) — flat.
  - Event 889, the **loudest** in the run, sits at `r_sky,u = 0.035`, *below* the
    population median 0.041.
  - `|r_{θφ}|` (the sky↔sky correlation) likewise: Spearman vs SNR **+0.006**.
- **What the run *does* support**: the *in-catalogue* subsample carries a higher sky↔distance
  correlation than the dark one — median **0.058 vs 0.041** — so the "information-carrying"
  half of the sentence has measured support; the "loud" half does not.
- **Disposition.** The trap is dismantled with the measurement, not with the spec's
  sentence. The chapter states the flat SNR dependence and the in-cat/dark split, and does
  not repeat "largest for the loud events".

## F-ch06-4 — `get_redshift_outer_bounds`'s `sigma_multiplier` argument is dead code (informational)

- `physical_relations.py:546-567` accepts `sigma_multiplier: float = 3.0` but the body
  hardcodes `3 *` in both bounds (`:563`, `:566`). Production passes `sigma_multiplier=2.0`
  (`bayesian_statistics.py:2805`) and the candidate-ball call passes `sigma_multiplier=1.5`
  (`:2820`, which *is* used, by `handler.get_possible_hosts_from_ball_tree`).
- **Consequence:** the redshift *window* is always ±3σ_dL, never ±2σ_dL. This is not a
  physics defect (a wider window is conservative and the window is not a likelihood
  weight — it only bounds which galaxies enter the sum) and it is not in any ledger row.
- **Disposition:** `gen_ch06.py` calls the production function, so it inherits the real ±3σ
  behaviour rather than re-implementing the documented one. Stated in the chapter's
  GW-reader stratum. Recorded here so no other chapter quotes "±2σ redshift window".

## F-ch06-5 — Ch 8's "`M_z` at 10⁻⁴" vs the CRB table's ~10⁻⁷ (informational, for ch08)

- **Spec.** `BOOK_DESIGN.md` §1 Ch 8 card, learning goals: *"M_z = M(1+z) at 10⁻⁴ as a
  second z-handle"*.
- **Measured** on `seed61000/prepared_cramer_rao_bounds.csv`, `σ_Mz/M_z ≡
  sqrt(delta_M_delta_M)/M`: median **8.8×10⁻⁸**, p5–p95 2.5×10⁻⁸–3.0×10⁻⁷; the product
  `(σ_Mz/M_z) × SNR` has median 2.57×10⁻⁶. Event 889: 1.36×10⁻⁹.
- **Not adjudicated here.** Ch 6 owns E3/E4 (the covariance), not M1–M4, and reports only
  what the shipped CRB table contains, chipped. The `10⁻⁴` in
  `mass_marginal_2d_kernel.md:638` is a *test tolerance* (`σ_cond = 10⁻⁴` in the
  N-A-vs-N-B limit gate), not a measured precision, so the spec line may be a
  transcription of that. Flagged for the ch08 agent and the integrator.

## F-ch06-6 — Line anchors: two `bayesian_statistics.py` citations in the spec are from an older revision (informational)

- `BOOK_PEDAGOGY.md` Ch 6 card and `G2a_completion_sky_marginal_4pi.md` §1 cite
  `bayesian_statistics.py:1052` (the 3-D mean `[det.phi, det.theta, 1]`) and
  `:982-1000` (the 3×3 covariance assembly).
- In **both** current trees those objects live at **`:2459`** and **`:2389-2409`**
  respectively (`:2410-2437` for the 4×4, `:2492-2510` for the Bishop conditional).
  `:982-1000` in the current file is inside `_wbh_z_kwargs`, an unrelated FIX-3 helper.
- Per `BOOK_DESIGN.md` §3.2 line numbers are re-grep anchors, not immutable. The chapter
  chips the **G2a-era anchor as the derivation's own citation** and additionally prints the
  current-tree anchor, so a reader who greps finds the code either way. Nothing is invented.
- Every other anchor this chapter uses was re-verified against the deployable tree and is
  correct as written: `parameter_estimation.py:335, :396, :399, :430, :447, :488`;
  `bayesian_statistics.py:1856, :3532, :4014`; `handler.py:505, :519, :575-578, :584-592,
  :594-603`; `physical_relations.py:546-567`.

## F-ch06-7 — Injection pool: 200,807 vs 200,100, independently reconfirmed (no new conflict)

- `gen_ch06.py` recounts the pool used by I6.2 and finds **200,100 data rows in 707 files**
  (200,100 + 707 header lines = 200,807 lines), reproducing exactly the arithmetic the ch04
  agent recorded in `ch04_FLAGS.md` F-ch04-1. Emitted as `meta.n_data_rows`,
  `meta.n_files`, `meta.n_lines_with_headers`.
- Strata: a = 99,014, b = 50,947, c = 50,139. Only stratum `a` carries the population
  measure (`simulation_detection_probability.py:364-401`), and I6.2 uses `a` only.
- `dl_max`-adjacent fingerprint check: the largest detection horizon in stratum `a` is
  **8.33181 Gpc**, matching the 8.3318 that ch04 reports as the pre-headroom value behind
  `dl_max(0.73) = 9.164987 Gpc` (× 1.1). No conflict.

## F-ch06-8 — I6.3 exists in the pedagogy but not in the build spec (resolved by precedence)

- `BOOK_PEDAGOGY.md` §4.1 lists three Ch 6 interactives: I6.1, I6.2 and **I6.3 "The 4π
  Marginal"**. `BOOK_DESIGN.md` §1 lists only I6.1 and I6.2 for Ch 6 and assigns the 4π
  marginal to **Ch 5 as I5.3**.
- `BOOK_DESIGN.md` §0 preamble: *"Where this file and those disagree, this file wins."*
  Ch 6 therefore ships I6.1 and I6.2, and forward-/back-references the 4π marginal to Ch 5.
  Recorded so a reviewer working from the pedagogy document does not score a missing widget.

---

# REVISION 2026-07-31 — post-review pass (`REVISION_WORKLIST.md` §C-ch06)

Appended, not rewritten: everything above is the record as it stood at build time. This
section records what the revision pass changed, what it measured, and what it could not
reach. Two new flags are opened (F-ch06-9, F-ch06-10).

## F-ch06-1 — RESOLVED by author mandate: σ_dL/d_L = 8.98×10⁻⁴ is the spec value

`REVISION_WORKLIST.md` §A-D1 adopts the six-chapter measured value book-wide. The
measurement recorded above was correct and is unchanged; only its **status** changed, from
"open disagreement with the spec" to "the spec was wrong and has been corrected."

- §4's boxed flag is gone. D1 requires a footnote, not a boxed OPEN dispute, so the
  `.ch06-flag` block (and its page-local CSS) was replaced by a `.ch06-note` carrying the
  **canonical erratum line verbatim** (`BOOK_CANON.sigmaDL.erratum`, `js/manifest.js`) plus
  the one-sentence lesson.
- **Both arbitration checks survive in the main column**, per the worklist AC and expert A's
  P1 / Tomas's P2 PRAISE: σ_uρ ≈ 1.04 ⇒ 7.3×10⁻⁴ at ρ = 1424.7 (the discarded reading needed
  ρ ≈ 13,000), and the readout's own 0.38 %/√76 ⇒ order 10⁻³ against the measured 5.3×10⁻³
  over the 76 in-catalogue rows. The step-by-step arithmetic for both moved into an adjacent
  `details.num-view`; the claims themselves did not.
- Dossier: the `d_L` row is now the canonical string
  `d_L  88.9 Mpc · σ_dL/d_L = 8.98×10⁻⁴` (copied from `BOOK_CANON.sigmaDL.dossierRowHTML`,
  not re-worded). The "distance precision" row keeps the absolute Gpc figure, correctly
  labelled, because the Gpc value is a real quantity in real units — it is not a second
  candidate value for the fraction.
- The generator was **not** changed for this: it always emitted `sigma_dL_Gpc` (Gpc) and
  `sigma_u` (dimensionless) as two separate, correctly-named keys. Its module docstring's
  scope note 1 was updated to record the resolution.
- `qa_gates.py` D1 gate: **PASS** on `ch06-black-box.html` (4 hits before this pass, at
  `:696`, `:700`, `:704`, `:980`).

## F-ch06-2 — RESOLVED with F-ch06-1: Q6.5 now answers ≈550×

- The stem keeps the disputed number verbatim (rubric D) and gains **Tomas m3's dagger**:
  "† the units in this question are the point — see the answer". A reader who reads the stem
  and skips the answer is now warned on the stem.
- The answer states **≈550×** as the answer. The retired ≈6000× survives only as an
  explicitly-labelled historical clause ("exactly what you get by reading that Gpc figure as
  a fraction — an eleven-fold slip that does not change the verdict, which is the only reason
  it survived as long as it did"). It is no longer offered as a co-equal reading.

## F-ch06-9 — NEW: the 14×14 conditioning, and the chapter's most extreme number priced

Raised by Tomas M8, executed here. **Every number below was recomputed by `gen_ch06.py`
from the CRB table, not copied from the review**; the reviewer's independent values are
quoted after each as a cross-check.

- **The gate's actual operand.** `FISHER_CONDITION_NUMBER_MAX = 1e14` acts on the 14×14
  *Fisher* at simulation time (`parameter_estimation.py:447`). The chapter previously showed
  only the derived 3×3 and 4×4 blocks. Since Σ = Γ⁻¹ and both are symmetric positive
  definite, `cond₂(Σ₁₄) = cond₂(Γ₁₄)` in exact arithmetic, so the stored covariance answers
  the gate's question directly. New generator code: `CRB_PARAMS_14`, `_cov14()`, and a
  `cond14*` block inside `_conditioning_stats()`; per-event `cond14` on the four featured
  events.
- **Measured over all 1590 rows:** κ₁₄ min 4.82×10⁸ · p5 9.49×10⁸ · **median 2.63×10⁹** ·
  p95 **1.36×10¹⁰** · max **3.85×10¹²**; **0** rows above the 10¹⁴ gate; all 1590 matrices
  positive definite. (Tomas measured median 2.6×10⁹ / p95 1.4×10¹⁰ / max 3.9×10¹² — agreement
  to the printed precision, from an independent implementation.)
- **The price of that conditioning**, now stated on the page: float64 carries ≈16 significant
  digits and a condition number costs ≈log₁₀κ of them, so the worst-conditioned event in this
  run has ≈**3.4** trustworthy digits left in its inverse
  (`cond14_float64_digits_left_worst_case`). The chapter says so immediately before §4.1
  reports a mass precision to three.
- **Babak plausibility check (§4.1).** Measured median (σ_Mz/M_z)·ρ = 2.572×10⁻⁶ ⇒
  **1.29×10⁻⁷ at ρ = 20**. Babak et al. (2017), arXiv:1703.09722 — this project's own EMRI
  population reference — quotes Δ(ln M_z) of order 10⁻⁵–10⁻⁶ at comparable SNR, i.e. this run
  is **7.8× to 78× better than the published range** (the page rounds to "roughly 8 to 80
  times"). The literature figure is carried in the generator as
  `BABAK_2017_MASS_PRECISION`, explicitly marked `"literature citation, not recomputed here"`
  — the book does not pretend to have re-derived somebody else's forecast.
- **Verdict, stated on the page:** the gap could be the AK-vs-numerical-derivative difference,
  the mission baseline, or high-SNR Fisher optimism, and it is **not tested here**. Nothing in
  the book leans on the mass precision's width — Ch 8's result turns on what the mass channel
  does to the sum. This is the chapter's own "measure before you generalize" discipline
  applied to its own most extreme number, which is exactly what M8 asked for.

## F-ch06-10 — NEW: the main-column budget is not reachable by relocation alone

The worklist's [P2] ped-M5 item asks for §4.1 and §5's matrix bookkeeping to move into
`details.gw-reader`, with the acceptance criterion "main column ≤ ~1.7× budget (from 2.37×)"
and the binding constraint "no argument lost, only relocated". Those two cannot both be met
for this chapter, and the arithmetic is worth recording rather than fudging.

- **Measured, ped's own metric** (`<main>` minus everything inside `<details>`):
  **5,933 → 5,392 words**, i.e. **2.37× → 2.16×** of the 2,500-word budget. Target for 1.7×
  is 4,250.
- **Relocated (nothing deleted; every sentence is still on the page, in a fold):** §4.1's
  conditional-decomposition derivation → `details.gw-reader`; §3's two derivative repairs
  (stencil, per-parameter ε) → `details.gw-reader`; §5's two repair surfaces + the frame-stamp
  bookkeeping → `details.gw-reader`; §4's two Jacobian ledger entries → the existing §4
  `details.gw-reader`; I6.1's two provenance paragraphs, I6.2's counterfactual-scope box, the
  four-event table's computation note, the erratum's step-by-step arithmetic, and the new
  conditioning table → `details.num-view`; the 15-item provenance panel → a collapsed
  `<details>`. Total relocated ≈ **845 words**.
- **Added in the same pass** (required by other worklist items, all in the main column):
  the 14×14 conditioning paragraph, the Babak plausibility paragraph, the ch03 cross-reference,
  the signature-default warning, the persona nudge, and the reframed erratum ≈ **305 words**.
  Net −541.
- **Why 1.7× is not reachable.** ped's own named ch06 fix list (§4.1 + §5's matrix
  bookkeeping) is worth ≈400 words; applied alone it would land at 2.21×. Everything
  additional in the list above was found by re-reading the page against ped's own diagnosis
  ("lists, captions, readout legends, verification asides and provenance narration promoted
  into the main column"), and that vein is now mined out. The remaining 5,378 words are the
  cold open, §1–§6's narrative, two widget captions, five self-check stems, one dossier and
  two traps. Closing the last 1,142 words means deleting argument, which the same acceptance
  criterion forbids.
- **One measurement correction offered to the integrator, not claimed as a pass.** ped's
  metric counts `<noscript>` static fallbacks, which no reader with JavaScript ever sees and
  which exist only so the page degrades honestly. ch06 carries **440** such words. The
  main-column load an actual reader meets is therefore **4,952** (1.98×). Both numbers are
  reported here; the pass is scored against the first.
- **PRAISE guard.** Two praised elements were checked back into the main column after an
  earlier draft folded them: Tomas P2's "F6.1's spread used as the licence bound on I6.1's
  rescaling" (*exact to ~30% for σ_u, only indicative for the sky widths*) and expert A P1's
  "two independent arbitration checks". Expert A's protected sentence, *"A normalization that
  multiplies a threshold changes which data exists"*, is untouched at §2.

## Other worklist items, dispositioned

- **Ball cross-reference to the regenerated census (worklist P1, synth-D2).** The four-event
  table's caption now states that Ch 3 measures the same rule at the same production
  multiplier, and that 889's ball is the same two galaxies at the same 0.757′ radius on both
  pages. The two *areas* are named as the two different objects they are (Ch 3's search disc
  π r² = 5.00×10⁻⁴ deg² vs this chapter's localization ellipse ΔΩ = 3.29×10⁻⁴ deg²), chipped
  to `ch03_FLAGS.md F-ch03-14`; the dossier's sky-ellipse row says the same. **No number was
  changed to make the two pages agree** — ch06's ball facts were already at the production
  multiplier (`gen_ch06.py`'s `SIGMA_MULTIPLIER = 1.5`), which is why ch03 regenerated and
  ch06 did not.
- **The signature-default trap, recorded on this side too.** ch06's §4 GW-reader fold now
  names it explicitly: production passes n_σ = 1.5 at the one ball-search call site
  (`bayesian_statistics.py:2838`), the *signature default* of
  `get_possible_hosts_from_ball_tree` is 2 (`handler.py:568`) and production never uses it,
  and the 2.0 fifteen lines above at `:2823` parameterizes the redshift window. The same
  warning is now in `gen_ch06.py`'s `SIGMA_MULTIPLIER` comment. This chapter's two drifted
  citations for those call sites (`:2820`, `:2805`) were corrected to `:2838` and `:2823`.
- **Traps relocated in-line (ped-M3).** Trap 6.B ("numerical factors are bookkeeping") now
  fires in §2, immediately after I6.2 — *after* the predict reveal, deliberately, because the
  trap's punchline is the 168× the reader is asked to commit to (D4). Trap 6.A ("sky and
  distance are separate measurements") now fires in §4, immediately after I6.1's measurement
  of event 361, where the misconception is created. Both texts are unchanged; only their
  position moved.
- **F-ch06-5's pointer text (worklist P2, tomas-B3 site).** Verified against the revised
  `ch08-mass-channel.html`: ch08 carries the both-values pair with median **8.8×10⁻⁸** and
  889 at **1.36×10⁻⁹**, cited to `ch06_FLAGS.md F-ch06-5` at six sites plus its provenance
  panel and its own F-ch08-10. F-ch06-5's numbers are exactly those. **No change needed**;
  ch06 remains the measuring side and adjudicates nothing.
- **Anchor re-grep (worklist P2, expA-m1).** Re-grepped against the current tree; the spec
  anchors are kept as the citation per `BOOK_DESIGN.md` §3.2 and a current-line `title=`
  tooltip added beside each, as ch03 did:
  `IDEALIZED_BASELINE_READOUT.md:47-48 → :59-60` (the "σ_H0/H0 ≈ 0.38 %/√76 = 0.044 %"
  sentence; 3 citation sites on this page) and `:50-52 → :64-66` (the "median σ_z/z = 49 %"
  sentence; 2 sites). Also re-grepped and tooltipped while here:
  `bayesian_statistics.py:2389-2409 → :2392-2410`, `:2459 → :2462`, `:2492-2510 → :2494-2510`,
  `handler.py:575-578 → :610-617`, `:584-592 → :623-632`, `:594-603 → :634-644`.
  **Re-verified as exact, unchanged:** `parameter_estimation.py:335, :396, :399, :430, :447,
  :488`; `physical_relations.py:546-567`; `handler.py:568`. F-ch06-6's own two anchors
  (`:1052`, `:982-1000`) are a different, older drift and its disposition stands.
- **Persona nudge (ped-m3, §D item 9, integrator-owned P3).** ch06 is one of the three named
  sites. The chapter's line is now in §3, after the "derivatives of the waveform" paragraph;
  the integrator owns the shared mechanism, this is only the sentence.

## Generator

`gen_ch06.py` re-runs clean in **7.5 s** and is byte-deterministic (two consecutive runs,
identical SHA-256 for both outputs). `ch06_fisher.json` 211.6 → 212.8 KB. New keys:
`population.conditioning.cond14{,_max,_min,_n_above_1e14,_equals_fisher_cond,
_all_positive_definite,_float64_digits_left_worst_case}`,
`population.mass_precision_plausibility`, and per-event `cond14`. `_cov14()` raises rather
than degrading if a CRB column is missing, so a schema change fails the build.

## Left for the integrator (not this agent's ownership)

- **Page-local rail pips (tomas-m2, worklist §D item 6).** `ch06-black-box.html` still appends
  `.ch06-pip` elements to `#bias-rail` from its own script block (page CSS at the top of the
  file, JS beside the `Book.biasRail` call). Converting them to `Book.biasRail({pips})` — so
  the amber does not change colour between Ch 5, Ch 6 and Ch 7 — is the integrator's §D-6 item
  and was deliberately not done here. The page-local CSS is left in place *because* the JS
  still emits those elements; removing one without the other would break the rail.
- **Cumulative rail rows (ped-M7, worklist §D item 4).** ch06's rail shows two rows where ch05
  shows three, so a reader stepping Ch 5 → Ch 6 sees it go backwards. The fix is
  `BOOK_BIAS_ROWS` + `from_chapter` in the manifest, which is integrator-owned.
- **Persona `apply("mara")` force-closing folds (ped-m3 tail, §D item 9 [P3]).** This pass moved
  more content into `details.gw-reader` / `details.num-view` on ch06 than any other chapter, so
  the "only ever open, never force-close" change is now worth more here than it was at review
  time. Shared-file work; flagged, not done.
