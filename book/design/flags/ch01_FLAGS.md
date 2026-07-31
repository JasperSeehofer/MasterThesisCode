# ch01_FLAGS.md — Chapter 1 ("A Ruler That Needs No Ladder")

Raised by the ch01 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

Nothing here blocks the chapter. Every item is shown to the reader in **both** forms, on
the page, with the arithmetic that relates them.

---

## F1 — EMRI-889's distance precision: `8.0×10⁻⁵` (spec, labelled *fractional*) vs `8.98×10⁻⁴` (recomputed fractional) — **UNRESOLVED, both printed**

- **Spec values.** `BOOK_DESIGN.md` §1 Ch 1 running example: *"σ_dL/dL = 8.0×10⁻⁵"*.
  `BOOK_PEDAGOGY.md` beat B4 and question **Q1.2** repeat it: *"a fractional distance
  precision of 8.0×10⁻⁵"*, and Q1.2's model answer converts it to *"σ_H0/H0 ≈ σ_dL/dL ≈
  0.008%"*. `BOOK_DESIGN.md` §1 Ch 6 repeats it a third time.
- **Measured by `gen_ch01.py`** from the row the spec names
  (`seed61000/prepared_cramer_rao_bounds.csv`, row 889 — verified byte-identical to the
  spec-named `real_r1/` copy on every column read):

  | quantity | value |
  |---|---|
  | `luminosity_distance` | 0.088879221 Gpc = 88.879 Mpc |
  | `sqrt(delta_luminosity_distance_delta_luminosity_distance)` | **7.9842729×10⁻⁵ Gpc** = 0.0798 Mpc |
  | ratio of the two | **8.983284×10⁻⁴** |
  | `1 / SNR` (SNR = 1424.7236) | 7.0189×10⁻⁴ |

- **What the two numbers appear to be.** `7.98×10⁻⁵` and `8.0×10⁻⁵` agree to two digits.
  The stored σ is in **Gpc** — the same units as the `luminosity_distance` column, since
  `dist()` returns Gpc — so the spec's figure is very plausibly the **absolute** σ_dL in
  Gpc carried forward with a **fractional** label, i.e. a missing division by
  d_L = 0.0888792 Gpc (a factor 11.25). **The generator does not assert this** and does not
  substitute one for the other.
- **The discriminating check (stated on the page, not used to overrule anything).** A
  matched-filter amplitude measurement cannot beat the ~1/ρ scale. Measured:
  σ_dL/dL = **1.28 / ρ** — a physically ordinary degeneracy-inflated distance error.
  The spec's fractional reading would be **0.11 / ρ**, i.e. ~9× *better* than the
  signal-to-noise scale. Independently, `IDEALIZED_BASELINE_READOUT.md:42-47` records a
  per-event budget of σ_H0/H0 ≈ 0.38% for the golden events, within a factor ≈4 of
  8.98×10⁻⁴ and a factor ≈47 from 8.0×10⁻⁵.
- **Disposition on the page.** The dossier prints **σ_dL = 7.98×10⁻⁵ Gpc (0.0798 Mpc) =
  8.98×10⁻⁴ of d_L**, both readings side by side, with the ×11.25 arithmetic and a pointer
  to this flag. Q1.2 is kept as the spec's estimate question, and its answer is the
  order-of-magnitude check *plus* an explicit editor's note carrying both values and the
  Q1.2 model answer's own 0.008% figure. No number was adjusted, dropped or rounded.
- **For the integrator and for Ch 2 / Ch 6 / Ch 10.** `BOOK_DESIGN.md` §1 Ch 6 also lists
  *"σ_dL/dL = 8.0×10⁻⁵"* as a dossier row, and Ch 2/Ch 10 quote the golden-event
  information budget. **Whoever resolves this should resolve it once, for all of them.**
  Both quantities ship in `data/ch01_event889.json`
  (`distance_precision.sigma_dL_Gpc`, `.sigma_dL_over_dL`, `.one_over_snr`,
  `.spec_quoted_as_fraction`, `.spec_disagrees_with_recomputation`).

## F2 — `1588` events (spec) vs `1590` stored CRB rows — **RESOLVED-BY-ARITHMETIC, logged**

- **Spec:** `BOOK_DESIGN.md` §1 and `BOOK_PEDAGOGY.md` throughout — "1588 events",
  "76 of 1588". `BOOK_PEDAGOGY.md` beat B4 separately calls 889 "the loudest of **1590**".
- **Measured:** the CRB CSV has **1590** rows, of which **76** carry `in_catalog = True`
  (4.78%); `real_r1/posteriors/combined_posterior.json` reports
  `n_events_total = n_events_used = 1588`, `n_events_excluded = 0`, `n_events_empty = 0`.
- **Disposition:** both are correct for what they count — 1590 stored Cramér–Rao rows,
  1588 events with a delivered per-event posterior. The chapter says "one of 1590 stored
  events; 1588 of them reach the combined posterior" and chips both. The `76 / 1588` =
  4.8% in-catalogue statistic is quoted with `1588` exactly as the spec does, and the
  page notes that the same 76 rows are 4.78% of 1590.

## F3 — `plunge_window` placement: Ch 1 sidebar (design) vs "Chapter-2 sidebar at most" (sources map §8)

- `BOOK_DESIGN.md` §1 Ch 1 assigns the plunge-window sidebar to this chapter, citing
  `docs/derivations/plunge_window_initial_conditions.md` (RATIFIED 2026-07-28) and
  "sidebar at most, per sources map §8". `BOOK_SOURCES_MAP.md` §8 actually says
  *"`plunge_window` belongs in a **Chapter-2** sidebar at most"* — but §8's chapter numbers
  refer to its own §6 partition (11 chapters), which `BOOK_DESIGN.md` §0 **supersedes**;
  §6's "Chapter 2" is the one-event/waveform chapter, which the adopted arc splits between
  Ch 1 and Ch 6.
- **Disposition:** followed `BOOK_DESIGN.md` (it wins where the two disagree, per its own
  header). The plunge window is a collapsed sidebar, ~180 words, no new machinery, and
  answers Q1.3 — which the spec assigns to this chapter.

## F4 — No disagreement found where one was plausible: G7 row 6 reproduces exactly

Recorded as a **passing** cross-check, not a conflict. `gen_ch01.py` recomputes the Ω_m
mis-specification systematic from scratch — h′ solving
`d_L(z; h′, 0.2726) = d_L(z; 0.73, 0.3153)` via `brentq` on the repo's own `dist()` — and
reproduces `docs/gates/G7_systematics_budget.md` ("Numbers behind row #6") at **all six**
redshifts: +0.16 / +0.32 / +0.94 / +1.50 / +2.60 / +3.31 %. The generator treats this as a
**hard gate** (tolerance 0.01 percentage points) and refuses to write a file if it drifts.

---

# REVISION PASS — 2026-07-31 (ch01 agent, `REVISION_WORKLIST.md` §C-ch01)

Appended, not rewritten: everything above is the record as it stood when the chapter was
built. This section records what the revision changed and why.

## F1 — **RESOLVED 2026-07-31 by author mandate (worklist §A-D1)**

- **Disposition changed.** The book-wide spec value for this row is now the measured
  **σ_dL/d_L = 8.98×10⁻⁴** (absolute σ_dL = 7.98×10⁻⁵ Gpc). The old spec figure
  `8.0×10⁻⁵` is the *absolute* σ_dL in Gpc carried under a *fractional* label — the missing
  division is by d_L = 0.0888792 Gpc, a factor **11.25**. Chapters stop printing dual values
  for this item; every page carries the corrected value plus the canonical erratum line.
  (Worklist §B-1: this supersedes the five reviewers' "print both" fixes — mara BLOCKER-3,
  tomas B2, expA B2, expB BL-5, ped B2 — whose page inventories remain the fix-site list.)
- **The pedagogical beat is kept and reframed.** The §2 block is no longer an `OPEN`
  arbitration that "prints both and picks neither"; it is a `RESOLVED 2026-07-31` erratum
  block that leads with the canonical erratum line and then keeps the two order-of-magnitude
  checks *as the proof*: 8.98×10⁻⁴ = 1.28/ρ (ordinary) vs the retired reading's 0.11/ρ (9×
  better than the 1/ρ bound — an impossibility), and the idealized per-event budget
  σ_H0/H0 ≈ 0.38%, within ≈4× of the corrected value and ≈47× from the retired one.
- **Q1.2 rewritten as the erratum lesson** (ped-P3's beat kept, now with a resolution). The
  stem no longer hands the reader a fraction: it hands σ_dL = 7.98×10⁻⁵ **Gpc** and asks for
  the conversion, the 1/ρ sanity check, and the reason the book will not get near it. The
  answer derives 8.98×10⁻⁴ → σ_H0/H0 ≈ 0.09%, runs both checks, and closes with a dated
  erratum note in which the retired `0.008%` model answer appears **only as history**. A
  `†` on the stem flags that the units are the point (device borrowed from tomas-m3).
- **Generator.** `gen_ch01.py` no longer refuses to choose. `SPEC_DOSSIER` carries
  `sigma_dL_over_dL = 8.98e-4` (spec) and `sigma_dL_over_dL_retired = 8.0e-5` (history), and
  two **hard gates** now run: (i) the measured fraction must equal the D1 spec value to 1%,
  and (ii) the retired figure must equal the measured *absolute* σ_dL in Gpc to 1% — so the
  erratum's diagnosis is verified, not asserted. `data/ch01_event889.json`'s
  `distance_precision` block gained `dossier_row`, `erratum`, `resolved`,
  `retired_spec_value_as_fraction`, `retired_spec_value_is_the_absolute_Gpc`,
  `retired_spec_value_missing_division_by_dL_Gpc` (11.25) and
  `retired_reading_in_units_of_one_over_snr` (0.114); the old key
  `spec_quoted_as_fraction` / `spec_disagrees_with_recomputation` pair is retired.
- **Dossier.** The distance row is now the canonical string, read at runtime from the
  integrator's one definition (`js/manifest.js` → `BOOK_CANON.sigmaDL.dossierRow`), with the
  full stored numbers (d̂_L, σ_dL in Gpc and Mpc, 1.28/SNR) kept on a second row. The
  `<noscript>` fallback carries the same canonical string literally.
- **Passes** `book/generators/qa_gates.py` gate D1 (`8.0×10⁻⁵` survives on this page only
  inside the two erratum notes, §2 and Q1.2).

## F5 — σ_Mz: `~10⁻⁴` (claim file) vs `8.8×10⁻⁸` (measured) — **BOTH VALUES, not a correction**

New this pass (worklist §C-ch01 [P1] tomas-B3, policy §D5 / §B-8). `ch01-ruler.html`'s
&ldquo;an EMRI is just a long binary&rdquo; trap asserted the redshifted mass is measured to
`~10⁻⁴` with no flag, one chapter before Ch 6 measures ~10⁻⁷–10⁻⁹ from the same table.

- **Claim-file value:** σ_Mz/M_z ≈ 10⁻⁴ — `CLAIM_2D_BIAS_20260730.md:172`, repeated on
  `BOOK_DESIGN.md` §1's Ch 8 card. Chipped `CLAIM C4` on the page.
- **Measured, recomputed here** by `gen_ch01.py` from `seed61000/prepared_cramer_rao_bounds.csv`
  (`sqrt(delta_M_delta_M)/M`): median **8.797×10⁻⁸**, p5–p95 2.47×10⁻⁸–2.99×10⁻⁷, event 889
  **1.365×10⁻⁹** — reproducing `ch06_FLAGS.md#F-ch06-5` exactly. The generator **hard-gates**
  its own recomputation against F-ch06-5's published median and 889 value (2% tolerance).
- **Disposition.** Unlike F1 there is **no author mandate**, so this is a both-values item:
  the trap prints both readings with the `CLAIM C4` chip and the F-ch06-5 pointer, states
  that neither is substituted for the other, and notes that no argument in the book turns on
  which is right (the channel is astonishingly sharp on either reading). Emitted as
  `mass_precision` in `data/ch01_event889.json`. Amending the claim file itself is the
  author's (worklist §F-2).

## Other revision items (no new flag)

- **[P2] tomas-m4 — the dossier's mass row is now `M_z`.** The stored `M` column is the
  **detector-frame** mass (`ch08_FLAGS.md#F-ch08-8`; `gate_b_20260730/c8_reparam.py:59` reads
  it as `M_z = crb["M"]`). Ch 1's dossier row, its `<noscript>` fallback, and the 14-parameter
  num-view now say so, with the ~2% source-frame difference at z ≈ 0.02 stated as below the
  displayed precision. **For the ch08 agent / integrator:** F-ch08-8's disposition line
  ("Ch 1's dossier row should say which frame it means") is now satisfied on Ch 1's side —
  ch08_FLAGS.md is that agent's file, so this pass did not append there.
- **[P2] mara-MINOR-4 — Q1.3 is answerable from the default read path.** The sidebar's
  two-sentence core (SNR accumulates near plunge; the rate model is itself a plunge rate; the
  generator/population-model mismatch) is now a narrator paragraph in §2, before the sidebar;
  the history (the retired `p0 ~ U[10,16]` draw) stays in the collapsed sidebar.
- **[P2] mara-MINOR-6 — the standard-siren caveat is out of the fold.** One narrator
  paragraph after the derivation box: it is the circular-binary quadrupole form, printed to
  show where d_L enters; the real EMRI (e₀ = 0.167 for the running example) is generated
  numerically by `few` + `fastlisaresponse`; nothing in the chapter depends on the closed form.
- **[P2] ped-m2 — Q1.4's transfer chain now routes through Ch 2**: "…and before any of that
  you need a machine that turns 1,588 distances into a statement about h, which is Chapter 2."
- **[P2] expA-m1 — anchor drift re-grepped.** `IDEALIZED_BASELINE_READOUT.md` gained a
  closure row on build day; the "76 of 1588 / 3 loudest carry 46% / per-event 0.38%" text
  cited as `:42-47` now lives at **`:54-60`** (§ "Where the information comes from
  (MEASURED)"), verified by grep on 2026-07-31. Both ch01 citations (Q1.2's answer, the
  provenance panel) updated; the provenance entry names the section and records the old
  anchor, so the drift is visible rather than silently patched.
- **D4 spoiler discipline:** the two forward references that carry numbers were checked —
  the Ch 8 pointers name the phenomenon (a mass number's frame/units moving a result) and
  never Ch 8's reveal value; the trap no longer leads with `10⁻⁴` as the single truth.
