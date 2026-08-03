# ch02_FLAGS.md — Chapter 2 ("Bayes, Once and For All")

Raised by the ch02 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, **stop and flag; do not silently reconcile in
either direction**."*

Neither item blocks the chapter. Both are presented on the page in **both** forms.

---

## F-ch02-1 — EMRI-889's "σ_dL/dL = 8.0×10⁻⁵" is the **absolute** σ_dL in Gpc, not a fraction — BLOCKING FOR OTHER CHAPTERS

- **Spec value (three places, all identical):**
  - `BOOK_DESIGN.md` §1 Ch 1 running example: "σ_dL/dL = 8.0×10⁻⁵";
  - `BOOK_DESIGN.md` §1 Ch 6 dossier: "σ_dL/dL = 8.0×10⁻⁵, correlated with sky";
  - `BOOK_PEDAGOGY.md` Part 2 beat B4: "fractional distance precision **σ_dL/dL = 8.0×10⁻⁵**";
  - `BOOK_PEDAGOGY.md` Part 3 **Q1.2** builds an answer on it: "σ_H0/H0 ≈ σ_dL/dL ≈ **0.008%**".
  - It is already rendered on a shipped page: `book/site/ch04-loud-half.html` dossier row.
- **Measured by `gen_ch02.py`** from
  `results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`,
  row 889:
  - `delta_luminosity_distance_delta_luminosity_distance` = 6.3748×10⁻⁹ — a **covariance**
    entry, i.e. a variance, in the parameter's own units
    (`parameter_estimation.py:430-480`: the columns are the entries of Σ = Γ⁻¹).
  - ⇒ **σ_dL = 7.9843×10⁻⁵ Gpc = 0.0798 Mpc**.
  - `luminosity_distance` = 0.0888792 **Gpc** (the FEW/`few` convention; 88.9 Mpc).
  - ⇒ **σ_dL/dL = 8.983×10⁻⁴ ≈ 9.0×10⁻⁴ (0.090%)**, *not* 8.0×10⁻⁵.
- **Three independent confirmations that the fraction is ~9×10⁻⁴:**
  1. **The repo says so itself.** `results/campaign51_20260728/idealization_audit/IDEALIZATION_LEDGER.md:31`:
     "The **3 loudest** (SNR 995–1425, z ≈ 0.016–0.021, **σ_dL/dL = 0.09–0.11%**) carry
     **46%**…". `gen_ch02.py` reproduces exactly 0.090% / 0.101% / 0.107% for events
     889 / 1536 / 118.
  2. **Order of magnitude.** For a matched-filter amplitude parameter σ_dL/dL ≈ 1/ρ. Over
     all 1590 rows the measured median of (σ_dL/dL)·SNR is **1.040** (IQR 1.006–1.097).
     With ρ = 1424.7 that gives 7.3×10⁻⁴; 8.0×10⁻⁵ would require ρ ≈ 1.3×10⁴.
  3. **Dimensional.** 8.0×10⁻⁵ is dimensionally the Gpc number: 7.9843×10⁻⁵ Gpc to
     2 s.f. is 8.0×10⁻⁵. The two quantities differ by exactly the factor d_L/1 Gpc.
- **Disposition in this chapter:** the Ch 2 dossier prints **both** — "σ_dL = 7.98×10⁻⁵ Gpc
  = 0.0798 Mpc ⇒ σ_dL/dL = 9.0×10⁻⁴ (0.090%)" — names the design docs' `8.0×10⁻⁵` as the
  absolute-σ reading, shows the arithmetic that relates them, and links this flag. Neither
  value is dropped and neither is asserted to supersede the other by fiat.
- **For the integrator (action needed, not by me):** Ch 1, Ch 4 and Ch 6 all carry the
  spec's fractional reading, and `BOOK_PEDAGOGY.md` Q1.2's *answer* ("0.008%") is used
  verbatim by Ch 1. That answer's arithmetic changes by 11× if the fraction is 9.0×10⁻⁴.
  This is a cross-chapter consistency item and is above a single chapter agent's authority.

## F-ch02-2 — "3 golden events carry 46%" reproduces **only** under the project's own 3-point curvature metric

- **Spec value:** `BOOK_DESIGN.md` §1 Ch 2 ("3 carry 46%"), `BOOK_PEDAGOGY.md` Q2.2, both
  citing `IDEALIZED_BASELINE_READOUT.md:42-47`.
- **Reproduced exactly (0.46996 → 46%)** using the metric declared in
  `results/campaign51_20260728/realistic_20260729/score_realistic.py:14-21` and reused
  verbatim by `gen_ch02.py`:
  `curv_k = ln(L_k(0.73)/L_k(0.725)) + ln(L_k(0.73)/L_k(0.735))`, evaluated on the
  **canonical idealized directory** `run_seed61000/posteriors_fixed` (§4.2 rule 1). Total
  238.332, implied σ_h = dh/√Σcurv = 3.239×10⁻⁴. In-catalogue share 101.3%, dark −1.3%
  (readout: "100%" and "~1%"). Golden set = {1536, 889, 118}, SNR 1068 / 1425 / 995
  (readout: "SNR 995–1425"). ✓
- **It does NOT reproduce under other natural metrics**, and the chapter says which metric
  it is using every time it prints the number:
  - quadratic fit of ln L over the **zoom** grid → top-3 share **52.5%**;
  - CRB-only Fisher weights 1/(σ_dL/dL)² over the 76 in-catalogue events → **41.9%**.
- **Disposition:** no contradiction with the spec — the spec number is right and the metric
  is now pinned. Logged so that no later chapter recomputes "46%" a different way and
  reports a conflict that is really a metric change. **Ch 10 and Ch 11 use the same
  statistic and should import this definition, not invent one.**

## F-ch02-3 — the realistic-venue information shares are **not quotable** (carried, not a disagreement)

- `REALISTIC_READOUT.md:110-113` (the artifact's own words): *"the percentages are
  ill-conditioned and should not be quoted … that is why 'dark share' reaches 140% and one
  run's golden share goes to −159%. Quote the signed sums, never the ratios."*
- `gen_ch02.py` therefore emits the realistic-r1 **signed** curvature sums and the absolute
  curvature mass, sets `quotable_ratios: false` in `ch02_information.json`, and the page's
  Event Stacker refuses to display a golden/in-catalogue *share* in the realistic venue,
  saying why. This is a live constraint on the chapter, not a numerical disagreement.

---

# REVISION 2026-07-31 — post-review pass (`REVISION_WORKLIST.md` §C-ch02)

Appended, not rewritten: everything above is the record as it stood at build time. This
section records what the revision pass changed, what it decided where the worklist left a
choice, and one new flag.

## F-ch02-1 — RESOLVED by author mandate: σ_dL/d_L = 8.98×10⁻⁴ is the spec value

`REVISION_WORKLIST.md` §A-D1 adopts the measured value book-wide, superseding every
reviewer's "print both" proposal (§B-1). The measurement recorded above was correct and is
unchanged; only its status changed — from "open disagreement with the spec" to "the spec was
wrong and has been corrected".

- The dossier now carries the canonical row `d_L 88.9 Mpc · σ_dL/d_L = 8.98×10⁻⁴`
  (`BOOK_CANON.sigmaDL.dossierRowHTML`, quoted verbatim), and the canonical erratum line
  sits under the table as a `.note` (`BOOK_CANON.sigmaDL.erratumHTML`, verbatim).
- The "distance precision" row keeps the arithmetic that *relates* the two readings
  (7.98×10⁻⁵ Gpc on 0.0888792 Gpc ⇒ 0.090%) — the order-of-magnitude check that caught the
  slip survives as the proof — but no longer presents the spec figure as a live alternative.
- `gen_ch02.py`: the JSON key `event889.spec_quoted_fraction = 8.0e-5` is retired into
  `event889.erratum {status, note, retired_spec_fraction}`, so the retired figure ships only
  inside an erratum block. The D1 build gate (`qa_gates.py`) passes on this page and on both
  of its data files.

## F-ch02-4 — NEW (and the reason BLOCKER-1 was possible): ch02 quoted a census it did not read

- **What was wrong.** The chapter asserted "tens of thousands of candidates" three times
  (`:373`, `:829`, and — worse — inside Q2.5's *graded answer*), inherited verbatim from
  `BOOK_PEDAGOGY.md:696`. Chapter 3's census measures a median of **6** candidates after the
  redshift window. A student (mara BLOCKER-1) answered "tens of thousands", was marked
  correct, turned the page, and found the book refuting its own answer key.
- **Fixed.** All three sites now state the distribution, and Q2.5's answer is rewritten from
  Ch 3's **regenerated** census (worklist §B-3 adopts mara's shape and rejects her literals,
  which were measured at the retired 2σ radius): median **888** in the sky ball / **6** after
  the window, 95th **2725**, max **245,334**, **607 of 1590** with no candidate at all,
  EMRI-889 with **two**. Chipped to `bayesian_statistics.py:2838` (n_σ = 1.5) and labelled as
  Ch 3's measurement.
- **Gated, not just corrected — new gate G8.** `gen_ch02.py` now re-reads Ch 3's own shipped
  `book/site/data/ch03_candidates.json` and refuses to write if any figure Q2.5 quotes has
  drifted (`CH03_CENSUS_QUOTED`). If that file is absent (cold clone; `make_all.py` runs
  gen_ch02 before gen_ch03) the check downgrades to a printed advisory. The class of bug —
  one chapter asserting another chapter's measurement from memory — is now build-visible.

## Other worklist items, dispositioned

- **The central predict is graded (mara MAJOR-3, [P1]).** The reveal opens
  "**(d) — a handful. Measured: 24 of 1588.**" and now answers every wrong option, including
  the near miss "~100" (right instinct — a few percent of 1588 is the right order for the 76
  in-catalogue events — wrong conclusion, because the participation ratio over the 1122
  positive-curvature events is 12.1). `data-predict-correct="few"` is tagged on the widget and
  on the predict row for the integrator's §D-8 mechanism; the prose grading does not depend
  on it.
- **62% → 52% (expert A M5, [P1]).** r1's own pair divides to **51.6%**; the readout's 62% is
  the *ensemble* figure (mean 0.076 / mean 0.123 over the ten runs). Both now ship, each
  labelled with its scope, in the §4 adjudicator block. New gate **G7** recomputes the ratio
  and fails the generator if the printed 52% stops being the division of the two numbers
  beside it. `ch02_information.json.realistic_r1` gains `signed_over_absolute`,
  `signed_over_absolute_printed`, `signed_over_absolute_readout_ensemble` and a scope string.
- **Denominators named (expert A m5, [P2]).** §4's cumulative series is now quoted against
  the **signed total** throughout (47 / 71 / 87 / 91%), with the in-catalogue set (46 / 70 /
  86 / 90%) given once in the same breath and the readout's 46% identified as the
  in-catalogue figure. Same fix in the widget's `<noscript>`, in the dossier's "rank in the
  information budget" row, in the "101.3% / −1.3% *of the signed total*" paragraph, and in
  Q2.2's stem.
- **Rung violation fixed (pedagogy M6, [P1]).** §1's `β(h)^N` derivation box (D(h), ledger
  #20, Mandel/Farr/Gair) is **moved**, not deleted: the main column keeps one sentence
  ("…already conditional on detection, so the joint posterior is a plain product…") plus a
  `⏭ Ch 4` chip, and the box's full argument now lives in this chapter's existing
  `details.gw-reader` fold, where it says in as many words why it is below the fold. **Routing
  note for the ch04 agent:** the worklist's preferred destination was ch04 §4 "Counted exactly
  once", which no ch02 agent may edit. If ch04 wants it, the text is self-contained in ch02's
  §1 fold and can be lifted verbatim; ch02's column sentence and forward chip stand either
  way, so nothing breaks if it stays here.
- **Traps de-spoiled (pedagogy M9 / B1, [P1]).** Trap 2.A now states the *phenomenon* only
  ("an estimator can be stable, reproducible and carry zero information about the parameter it
  reports") and hands #49a's number, verdict and exhibit to Ch 10. Trap 2.B keeps the √N
  mechanism and drops the recorded 2D figures and the "Chapter 8's cold open" signpost for a
  bare `⏭ Ch 8`. The same de-numbering was applied to **Q2.1's answer** and to the two
  provenance-panel bullets that carried #49a's verdict and the 2D pull's magnitude — the
  acceptance criterion is "ch08/ch10 reveals are no longer pre-announced with numbers from
  ch02", and a hidden answer or a provenance line is still ch02 printing them.
- **DECISION — I2.2's fourth preset is now a round toy value, not the recorded 2D offset.**
  The worklist asks only for a relabel ("so it doesn't name the recorded 2D offset's value"),
  but the button dialled **+0.077** into a visible readout and the note beneath it attributed
  that number to campaign #53's 2D channel — i.e. the value survived any relabel, and §D4
  scopes +0.077 to Ch 8 and later. The preset now dials **+0.080** ("badly biased, honest σ"),
  chosen as 4σ and nothing else; the note says so and forward-references Ch 8 without a
  number. Two side benefits: the widget's "this widget contains no project data" claim is now
  true (it was self-contradictory), and the lesson is unchanged — coverage 0.5%, pull s.d.
  still 1.09. The static fallback's third row was **recomputed**, not edited: the page's own
  mulberry32 + Box–Muller draws were replayed to get mean pull **+3.95** (was +3.80 at 0.077);
  coverage and pull s.d. are unchanged at 0.5% and 1.09.
- **Q2.3 re-aimed (pedagogy M1, [P2]).** It was verbatim recall of the §3 definition box.
  It now asks the reader to *construct* an estimator failing exactly one of the three using
  I2.2's dials, and to say which failures a single `combined_posterior.json` can diagnose
  (none). The answer is built from I2.2's own measured numbers (17.5% coverage / pull s.d.
  4.37; 0.5% / +3.95), so it is checkable in the widget rather than recoverable from the box.
- **Lazy Plotly init (ux MAJOR, [P2]).** All **7** `Book.themedPlot` call sites now go through
  a page-local `lazyPlot()` with the same signature and the same `{ update(traces) }` return
  shape: an `IntersectionObserver` (ch03's precedent, 300px root margin, observing the
  enclosing `.widget`) constructs each plot the first time it approaches the viewport, and
  `update()` before that only records the traces to draw at construction. No call site
  changed. Degrades to the eager path without `IntersectionObserver`. Swap for `Book.lazyPlot`
  if the integrator promotes it.

## Verification notes (what was and was not checkable here)

- `gen_ch02.py` re-runs clean in **0.32 s** with gates G1–G8 (G8 gated against the shipped
  ch03 census, `ball_rule` = n_σ 1.5 from `bayesian_statistics.py:2838`).
- `qa_gates.py`: D1 / ROW / DNR pass; the one TNS violation is `index.html:148` — the
  integrator's file, integrator pass 2's item, untouched here.
- The page's script passes `node --check`; the HTML parses with no unclosed or crossed tags;
  a headless `file://` DOM dump renders KaTeX (132 nodes) and runs I2.2 end-to-end (readouts
  −0.0009 / 64.5% / 1.09) through the new `lazyPlot` path, including the update-before-build
  case. **Not checkable in this environment:** a served click-through — loopback networking is
  blocked in the sandbox, so the lazy plots could not be observed constructing on scroll.
  BUILD_REPORT gap #1 (browser click-through) is inherited by integrator pass 2 and this page
  is one of the pages that needs it.

## For the other chapters

Ch 2 no longer prints: #49a's verdict or MAP, the 2D channel's mean pull, or +0.077 in any
form. Ch 8 and Ch 10 own those reveals outright. The census figures Ch 2 quotes are Ch 3's
(median 888 / 6, p95 2725, max 245,334, 607 of 1590, EMRI-889 = 2) and are gated against Ch
3's data file — if Ch 3's census moves again, Ch 2's generator fails rather than Ch 2's
answer key going quietly wrong.
