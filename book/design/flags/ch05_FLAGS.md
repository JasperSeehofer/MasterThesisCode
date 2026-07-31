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

---

# REVISION 2026-07-31 — post-review pass (`REVISION_WORKLIST.md` §C-ch05)

Appended, not rewritten: everything above is the record as it stood at build time. This
section records what the revision pass changed, and opens one new measurement flag.

## F-ch05-6 — RESOLVED by author mandate: σ_dL/d_L = 8.98×10⁻⁴ is now the spec value

`REVISION_WORKLIST.md` §A-D1 adopts the six-chapter measured value book-wide. Ch 5's only
site was the closing dossier row, which now carries the canonical string
`d_L  88.9 Mpc  ·  σ_dL/d_L = 8.98×10⁻⁴`, followed by the canonical one-line erratum note.
Ch 5 never carried the dispute in prose, so nothing else on the page changed. The retired
`8.0×10⁻⁵` appears on this page in exactly one place — inside that erratum note — and the
build's D1 grep gate passes on ch05.

## F-ch05-7 — CORRECTION: C10's attribution was mis-scoped in the block whose job is to enforce it

- **What was wrong.** §2's `.voice-adjudicator` ("What the flatten-the-slope number is, and
  is not") read *"only 39.1% of them have a positive completion tilt; ΣΔln L^comp = −3.11
  over the same window"* in a sentence about **dark** events. Both halves were mis-scoped:
  −3.11 is the **all-event** total, and 39.1% is the fraction positive on
  **(1−w_G)·L^comp**, not on `L^comp` alone. The same two numbers appeared in the
  provenance panel. `ch08_FLAGS.md` F-ch08-6 already establishes the second scoping, so
  Ch 5 was inconsistent with Ch 8 as well as with C10 itself.
- **Caught by:** expert A M1 (independent recomputation from the r1 diagnostics CSV).
- **Re-measured by `gen_ch05.py`** over C10's own window h = 0.73 → 0.81, emitted as
  `ch05_mixture.json.c10_Lcomp_scoping` and printed in the generator's console gates:

  | quantity | value |
  |---|---|
  | ΣΔ ln L^comp, **dark (1512)** | **−22.717** |
  | ΣΔ ln L^comp, all 1588 | −3.109 |
  | ΣΔ ln L^comp, in-catalogue (76) | +19.609 |
  | dark fraction positive, **L^comp alone** | **27.71%** |
  | dark fraction positive, **(1−w_G)·L^comp** | **39.09%** |
  | ΣΔ ln[(1−w_G)·L^comp], dark | **+7.327** |
  | Δ ln(1−w_G) per event | +0.0198705 |

  Every figure reproduces expert A's recomputation to the digit, and `N·Δln(1−w_G)` is
  unchanged at +31.554 against C10's +31.55.
- **Fixed:** the adjudicator block and the provenance panel now print the dark sum −22.72
  with the all-event −3.11 named as such, both percentages with the object each counts, the
  F-ch08-6 pointer, and the sign flip −22.72 → +7.33 that *is* C10's claim. Nothing was
  reconciled away; the numbers were re-scoped, not changed.

## F-ch05-8 — NEW measurement: I5.1's κ dial is non-monotonic, and the 0.86 plateau has a measured mechanism

- **Raised by:** Mara MAJOR-7 (she read the shipped JSON and found the midrange unnarrated;
  the page's generic `other` verdict said only "watch how far the MAP travels").
- **Measured by `gen_ch05.py`**, emitted as `prefactor_tilt_by_kappa`,
  `map_by_kappa_with_leg` and `map_by_kappa_no_leg`. The dial's regimes:

  | κ | MAP (total) | MAP, 1095 with a catalogue leg | MAP, 493 without | N·Δln(1−w_κ) over the grid |
  |---|---|---|---|---|
  | 0 | 0.755 | 0.86 | 0.60 | +0.0 |
  | 0.05 → 0.7 | 0.61 → 0.71 | 0.82 → 0.76 | 0.60 | +7.1 → +90.1 |
  | 1 (shipped) | 0.740 | 0.76 | 0.60 | +123.7 |
  | 3 → 60 | 0.85 → **0.86** | 0.85 → 0.86 | 0.85 → 0.86 | +295 → +878 |
  | 120 | 0.86 | 0.61 | 0.86 | +926.5 |
  | 250 → 3000 | 0.63 → 0.600 | 0.60 | 0.86 | +954 → +979 |
  | ∞ | 0.600 | 0.60 | — (493 silenced) | — |

- **The two mechanisms, as measured — not as guessed.** (a) *Below κ = 1 the dial is
  effectively discontinuous at the origin*: for an event like EMRI-889 the catalogue leg is
  ≈5×10⁷ times the completion leg at h = 0.73 (both numbers are already on the page's
  dossier), so any weight above ~10⁻⁸ leaves the catalogue branch — and its rail at 0.600 —
  in charge. Only at exactly κ = 0 does the term vanish and the MAP jump to 0.755.
  (b) *The 0.86 plateau is the (1−w_G) prefactor, not the in-catalogue class.* Mara's
  proposed narration ("you are watching the 76 in-catalogue events' own preference
  dominate") is **not** what the data says and was not adopted: over κ = 3…60 the 1095
  events **with** a catalogue leg and the 493 **without** one peak at 0.86 *together*, and
  by κ = 120 the with-leg group has already fallen back to 0.61 while the total is still
  0.86. What rises monotonically across the whole stretch is C10's prefactor tilt, from
  +124 nats at the shipped κ = 1 to +979 at κ = 3000. The plateau is an invented weight
  slope out-shouting both branches — which is the same lever the chapter's own
  flatten-the-slope beat is about, seen from the other side.
- **Disposition:** three new `V51` verdict states (`below`, `plateau`, `wall`) render the
  live per-κ numbers, so every reachable dial position has a measured verdict; a prose
  paragraph before the widget and an extended `<noscript>` fallback give static readers the
  same content; the `Show me the numbers` fold gains the per-κ table above. `n_zero_by_kappa`
  is 0 at every finite κ and this is now stated on the page — the 493 fall out only at
  κ = ∞ exactly.

## Other items from the worklist, dispositioned

- **[mara MAJOR-1] w_G's type before its value.** The line-of-sight-average sentence was
  **moved** (not copied) out of the `▸ For the GW reader` fold into the narrator flow
  immediately before "First: 12%"; the fold now points back at it and keeps only the
  β_G/D mechanics. The 12%-vs-5% paragraph additionally labels **both** realized-rate
  figures with their samples — 76/1588 = 4.8% (seed 61000, one realization) and
  164/3135 = 5.2% (two seeds pooled, the claim file's figure) — which was the page's
  internal denominator contradiction.
- **[expA M2 / mara MAJOR-6 — D5]** Q5.4's answer no longer carries "1500–2400×" alone: it
  now carries the adjudicated 1500–2400× **and** the recomputed 142–2458×, median 197×,
  with the `ch11_FLAGS.md` F-ch11-1 pointer, plus the sentence that the argument does not
  depend on which is right. The 0.025 Poisson headline (which reproduces to 1e-6) leads.
- **[mara MAJOR-4 — D4]** §3 de-spoiled: the heading is now "The completion branch, and the
  step where you must integrate rather than evaluate" and the opening paragraph says "costs
  orders of magnitude" instead of "a factor of several thousand". The ~5000× now first
  appears in the reveal, so the predict is answerable wrongly. No data changed.
- **[ped M3]** Trap 5.B moved to §2, immediately after the "12%. Not 5%" paragraph that
  creates the misconception; Trap 5.A moved to §3, immediately after `L^comp` is defined.
  Neither trap remains below the self-check.
- **[ped M9 / D4]** Trap 5.B's C9 measurement numbers stripped: the binomial z = −11.86 and
  the "2.3–2.5× the realized in-catalogue rate" framing are Ch 9's, and D4 names z = −11.86
  explicitly. The trap keeps what Ch 5 owns — 0.920 vs 0.1215 locally, 0.1215 vs 4.8%
  globally — and names C9 as the phenomenon without its verdict. The provenance panel's C9
  entry was softened the same way. **Consequence for D5:** with the ratio gone, no half of
  a flagged pair is left alone on the page.
- **[ped M1] Q5.3 / Q5.4 overlap, re-aimed per option (b)**, not kept. Q5.3 now asks *which*
  sanity limit is discontinuous on the dial and what property of this run's branches makes a
  vanishing weight fail to vanish (answerable from the dossier; it consumes F-ch05-8's
  measurement). Q5.4 now asks the reader to **design** the leverage test — what to perturb,
  what to hold fixed, what response falsifies reading A — before comparing with C5.
  Re-measured 5-gram overlap against the chapter body, same method as the pedagogy review:
  **Q5.3 25% → 7.9%, Q5.4 27% → 0.4%** (Q5.1 6.8%, Q5.2 15.2%, Q5.5 0.0%). All five now
  sit under the 22% bar.
- **[synth / D2]** §4's GW-reader fold now consumes Ch 3's **regenerated** census
  (n_σ = 1.5, `bayesian_statistics.py:2838`): **607 of 1590** events with no catalogue
  candidate, **983** with at least one — replacing nothing on this page (Ch 5 never quoted
  the retired 552) but closing the gap a reader would otherwise have to guess across. The
  expA-P7 discipline is stated explicitly: it is a *reconstruction* count on the truth
  catalogue and a different event list, never a drop count, and never this run's 493.
- **Not changed, deliberately:** F-ch05-1's two numbers for the 4π defect stay carried and
  unreconciled (that is a §D-5 both-values item and the adjudicator block already obeyed
  it); F-ch05-2's "493 is a zero-catalogue-leg count" discipline stands; F-ch05-3 and
  F-ch05-4 are untouched; the I5.1 sandbox's `Has this been tried?` box (ledger #61/#64)
  and the museum interlude are PRAISE-protected and were not edited.
