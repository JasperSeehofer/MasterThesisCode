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
