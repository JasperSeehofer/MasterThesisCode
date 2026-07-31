# ch05_FLAGS.md — Chapter 5 ("The Galaxy You Cannot See")

Raised by the ch05 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction"*, and §4.3 item 4.

Nothing here blocks the chapter. Each item is presented on the page in both forms where it
is visible to the reader.

---

## F-ch05-1 — The 4π defect: "~5000×" vs the ratified analytic factor at the SAME σ_sky — UNRECONCILED, carried on the page

- **Spec / ledger value:** `BOOK_DESIGN.md` §1 Ch 5 ("the 5000× defect"), `BOOK_SOURCES_MAP.md`
  §3 C3 and `BIAS_HISTORY_LEDGER.md` row **#46**, and `docs/gates/G7_systematics_budget.md`
  row 3: *"~5000× `B_num` inflation → rail at grid edge"*. **No localization is attached to
  that figure in any of those sources.**
- **Ratified derivation value:** `G2a_completion_sky_marginal_4pi.md` §7 limiting case 5 gives
  the analytic old/new ratio in closed form,
  `old/new = 4π/(2π σ_φ σ_θ) = 2/(σ_φ σ_θ)`, and quotes it as
  **≈1.6×10³ at σ_sky = 2°** (asserted to 5% by
  `test_completion_sky_marginal_reduces_magnitude`) and **≈1.8×10⁵ at the median 0.2 deg²
  EMRI localization**. It concludes only that a mechanism claim of "~10³–10⁵" is
  *"arithmetically consistent"*.
- **Third value, in the code:** the fix-site comment
  (`bayesian_statistics.py`, completion-numerator block, the `[PHYSICS]` comment above
  `p_gw = norm.pdf(...) * np.sin(theta) / (4π)`) pins the two together at one point:
  *"over-counted the completion term by ~4π·(peak sky density) (**~5000× at σ_sky≈2°**)"*.
  At σ_sky = 2° the derivation's own factor is **1.6×10³**, not 5×10³.
- **Reproduced by `gen_ch05.py` / the page's I5.3 widget** (closed form, in-browser):
  `2/(σ_φσ_θ)` = 1.64×10³ at 2.00°, 1.82×10⁵ at 0.19° (= 0.197 deg² under the project's own
  area convention `Δ Ω = 2π sinθ σ_φ σ_θ`, `detection.py:25`, at sinθ ≈ 0.87). Both anchors
  reproduce G2a exactly.
- **Why they are not obviously the same quantity:** `cb16142` changed *two* things at once
  (G2a §3, first bullet) — the sky factor (peak density → 1/4π) **and** the width of the
  distance Gaussian (the old peak evaluation carried the *conditional* precision
  `(Σ⁻¹)₂₂` instead of the *marginal* variance `Σ₂₂`, i.e. a too-narrow, too-tall
  `u`-Gaussian). The measured pipeline-level `B_num` inflation therefore contains a second
  factor that the sky ratio alone does not. **No artifact in the repo states that
  decomposition**, so the book does not compute one.
- **Disposition:** the chapter shows **both**: the recorded ~5000× as the measured
  pipeline-level inflation (chipped `ledger #46`, `G7 rows 3–4`), and the analytic
  `2/(σ_φσ_θ)` law with G2a's own two anchors (chipped `G2a §7 case 5`), in a
  `.voice-adjudicator` block that names the code comment's single-point pinning as the
  discrepancy and explicitly declines to reconcile it. Neither number is dropped, adjusted
  or averaged.
- **For the integrator / other chapters:** Ch 6 (I6.3 in the pedagogy doc, folded into
  Ch 5 as I5.3 by `BOOK_DESIGN.md` §1) and the museum's C3 exhibit both quote this factor.
  Quote "~5000× measured in the pipeline" or "1.6×10³–1.8×10⁵ analytic, depending on
  localization" — do not quote "5000× at 2°".

## F-ch05-2 — "493 events" is a zero-CATALOGUE-LEG count, not a zero-HOST count — precision flag

- **Measured by `gen_ch05.py`** on `seed61000/real_r1/diagnostics/event_likelihoods.csv`:
  **493 of 1588** events have `L_cat_no_bh == 0.0` at **every** one of the 41 h-values; the
  other 1095 are non-zero at every h; **0** events are zero at only some h.
- **What C4 / ledger #54 is about** is the *zero-host* population — events whose BallTree
  lookup returned no candidate at all, which the pre-`8db6c6e` code silently dropped. The
  delivered diagnostics carry no column distinguishing "no candidates returned" from
  "candidates returned, none contributed a non-zero numerator", and the run log lines that
  were staged (`mixture_leg_log_extract.txt`) carry only `D(h)`, `beta_Gbar(h)` and the
  4-d.p. `w_G` line — not the `_n_zero_host` counter.
- **Disposition:** the chapter quotes **493 events with an identically-zero catalogue leg**
  and says in the GW-reader stratum that this is not the same as a zero-host count. The
  **58%** figure is quoted as what it is: the *deep* Phase-2 venue (seed 1000, depth-1.5
  pool), chipped `ledger #54`, never as a property of this campaign.
- **Side observation, recorded not resolved:** 2 of the 493 carry `in_catalog = True` — the
  generator did place their host in the catalogue and the candidate search still returned
  nothing usable. Stated on the page; not diagnosed here (candidate-window mechanics are
  Ch 6/Ch 8 material).

## F-ch05-3 — I5.2's combined MAP is 0.740, where the design card writes "~0.73" — NO CONFLICT, recorded

- `BOOK_DESIGN.md` §1 Ch 5 I5.2 static fallback says the three curves peak at
  "0.86 / 0.64 / ~0.73"; `BOOK_PEDAGOGY.md` Q5.4 says "with the combined answer at 0.73".
- **Measured:** in-catalogue (76) → **0.86**, dark (1512) → **0.64**, all 1588 → **0.740**
  (mean 0.7321), which is this run's published row (`REALISTIC_READOUT.md` §1) and the same
  number Ch 4 reports for the identical event set.
- The design text carries a tilde and the pedagogy sentence is referring to the headline
  region, so this is a rounding of the combined value toward the injected truth (0.73), not
  a competing measurement. **Disposition:** the chapter prints the measured 0.740 and the
  truth 0.73 as two distinct numbers everywhere, including inside the Q5.4 stem (the only
  place the question text was altered from the pedagogy doc's wording). Recorded here so the
  edit is visible to a reviewer diffing against `BOOK_PEDAGOGY.md` Part 3.

## F-ch05-4 — Trap 5.A's "84%+" is printed as an r1 partition, not as the finding — deliberate deviation from the pedagogy text

- `BOOK_PEDAGOGY.md` Trap 5.A reads *"it is where 84%+ of the 2D channel difference lives"*.
- `BOOK_DESIGN.md` §3.3 item 3 and `BOOK_SOURCES_MAP.md` §7 item 5 are binding the other
  way: **"84.2%" is r1-specific**; off-r1 replication gives 84.2%–112.5% (mean 91.6%); what
  replicates is *dark ≫ in-cat, dark always positive*; and §4.3 item 5 lists "'84%' as the
  replicated finding" as a **banned sentence**.
- **Disposition:** the chapter's Trap 5.A states the qualitative replicated finding first,
  then gives 84.2% explicitly labelled as the r1 partition alongside the 84.2–112.5% /
  mean 91.6% replication range. The binding rule wins over the pedagogy wording, as
  `BOOK_DESIGN.md` §0 requires.

## F-ch05-5 — Line-number anchors: spec chips vs the working tree

- `BOOK_DESIGN.md` §3.2 says line numbers are re-grep anchors copied from
  `BOOK_SOURCES_MAP.md`, not invented. The chapter uses the spec's chips verbatim
  (`bayesian_statistics.py:3006-3009, :1042-1048`, `:3309-3311`, `:3210-3238`,
  `:2832-2844`).
- In the working tree inspected on 2026-07-31 the corresponding blocks sit at slightly
  different offsets — e.g. the completion-numerator integrand at ~`:3279-3320`, the
  zero-host fallback comment at ~`:2843-2866`, and the `w_G = beta_G / D_h` assignment at
  `:3392`. **Not a conflict**: every cited block was located and read, and every semantic
  claim matches. Recorded so a reviewer re-grepping does not read the offset as an error.
