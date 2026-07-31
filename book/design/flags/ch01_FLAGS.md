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
