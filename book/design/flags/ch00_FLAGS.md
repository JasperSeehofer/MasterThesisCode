# ch00_FLAGS.md — Chapter 0 ("Two Numbers That Should Be One", prologue)

Raised by the ch00 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

**No project-artifact conflict arose.** Chapter 0 makes no pipeline claim (its card:
*"No pipeline claims are made in this chapter"*), and its single contact with the code —
`H = 0.73` — reproduced exactly. The items below are literature-vs-design-doc rounding
and one frozen-file workaround, recorded so a reviewer does not have to re-derive why the
page prints what it prints. Nothing here blocks the chapter.

---

## F-ch00-1 — SH0ES central value: 73.0 (design docs) vs 73.04 ± 1.04 (source paper)

- **Spec value:** `BOOK_PEDAGOGY.md` Part 4 §Ch 0 (I0.1): *"whether it separates 67.4 from
  **73.0** at 3σ"*. `BOOK_DESIGN.md` §1 Ch 0 says only "Planck-like, SH0ES-like".
- **Source value:** Riess et al. (2022), arXiv:2112.04510 — H₀ = **73.04 ± 1.04**
  km s⁻¹ Mpc⁻¹.
- **Disposition:** the page and `gen_ch00.py` carry **73.04 ± 1.04**, the published pair,
  because the widget needs the σ as well as the central value and 73.0 is the same number
  to the precision the pedagogy doc quotes it at. This is a rounding difference, not a
  disagreement; no reconciliation is asserted in either direction and the arXiv id is
  chipped everywhere the number appears.

## F-ch00-2 — Tension significance: 5σ (quoted) vs 4.89σ (recomputed)

- **Spec/paper value:** `BOOK_PEDAGOGY.md` Part 2 §Ch 0 says the two disagree *"at ~5σ"*;
  arXiv:2112.04510 quotes **5.0σ** for its full SH0ES-vs-Planck comparison.
- **Recomputed by `gen_ch00.py`:** the two-number Gaussian combination of the two
  *quoted* uncertainties alone gives 5.64 / √(0.5² + 1.04²) = **4.888σ**.
- **Why they differ (stated, not resolved):** the published 5.0σ is not the arithmetic on
  two summary numbers — it comes from the full SH0ES error budget and covariance treatment.
  The page therefore prints **both**, labelled: "4.89σ on the two quoted uncertainties
  alone (the paper quotes 5.0σ)". Neither figure is dropped and neither is presented as
  correcting the other. Both are emitted into `data/ch00_tension.json`
  (`tension.n_sigma`, `tension.published_n_sigma`).

## F-ch00-3 — The chapter's own numbers are a declared toy, and say so

- The arbitration budget (σ_tot² = σ₁²/N + σ_sys², the T_A significance, the 3σ/2σ
  thresholds, the per-event scale σ₁ = 10 km s⁻¹ Mpc⁻¹) is **defined in
  `gen_ch00.py`**, not taken from any project artifact. It is chipped
  `toy: analytic` on the widget, in the equation callout, and in the provenance panel.
- The derived numbers the prose quotes from it — ceiling 1.57 km s⁻¹ Mpc⁻¹ (2.3% of H₀),
  41 events at zero systematic, the 2.50σ ceiling at σ_sys = 2 — are all emitted by the
  generator and **re-computed in the browser**, which cross-checks itself against five
  generator check points at page load (max |Δ| = 0.0, shown in the widget's numbers view).
  Recorded here because "toy" and "unverified" are not the same thing.

## F-ch00-4 — `<!DOCTYPE html>` + `<meta charset>` workaround (not a new request)

- Already filed by the ch07 agent as **R-ch07-3** in `book/design/WIDGET_REQUESTS.md`:
  the frozen `_template.html` / `index.html` omit the doctype, so pages render in quirks
  mode and KaTeX refuses to run.
- `ch00-two-numbers.html` carries the same page-local workaround (doctype + charset as its
  first two lines, with a comment pointing at R-ch07-3). **No duplicate request was
  appended** to `WIDGET_REQUESTS.md`. Verified headless (Chromium 141, served over HTTP):
  56 KaTeX spans render, 0 parse errors, 0 console errors.

---

# REVISION — 2026-07-31 (post-review pass, `REVISION_WORKLIST.md` §C-ch00)

Appended, not rewritten: everything above is the build-day record and still stands.
This section records what the revision pass changed on `ch00-two-numbers.html`,
`gen_ch00.py` and `data/ch00_tension.json`, and the judgement calls behind it.

## R-ch00-1 — [P1, ped-M5] Budget trim: 2,801 → **2,040** main-column words (2.33× → **1.70×**)

Measured with the reviewer's own metric (`<main>` minus every `<details>` subtree,
`<script>`/`<style>` excluded, `<svg>` text and `<noscript>` fallbacks included) — the
baseline reproduces their 2,801 exactly, so the two numbers are comparable.

Relocations (nothing deleted, everything still on the page):

- **§2's step-by-step σ_tot algebra** → `details.num-view` *"Show me the algebra — the
  arbitration budget, written out"*: the RATIFIED-style definition callout (σ_tot², T_A,
  the 3σ/2σ thresholds) and the three-step ceiling derivation. The column keeps the
  result (1.57 km s⁻¹ Mpc⁻¹ = 2.3% of H₀) and the both-branches sentence. This is the
  reviewer's own prescription; I0.1 recomputes all of it live.
- **The "How to read the pages that follow" legend** (stamps, badges, voices) → a
  `details.num-view` fold, with a three-clause summary of the conventions left in the
  column. It is a legend, not narration.
- **I0.1's σ₁ = 10 provenance note** → into I0.1's existing numbers view.
- **The provenance panel's roster** → a fold inside the panel. The panel keeps its
  heading and gains a one-line lede ("Seven recorded literature values, one code site,
  four forward references, and one toy. Nothing on this page is a pipeline result.").
  *Judgement call, flagged for the integrator:* this is the only structural deviation
  from the other thirteen pages. It was needed because the reviewer's metric counts the
  panel as main column while their own fix advice ("gate tolerances belong in the
  provenance panel") routes text *into* it — the two cannot both be satisfied without
  folding. If the integrator prefers panel-uniformity across the book, reverting this
  one fold costs ~240 words and puts ch00 at ~1.9×; that is the trade.
- The rest is line-level tightening (hedges, doubled clauses, over-long static
  fallbacks). Protected content untouched in substance: the two figures, the ladder
  schematic, I0.1, the time-delay 2%→8% paragraph, the contract callout, all three
  self-checks, the GW-reader fold, every provenance chip.

## R-ch00-2 — [P1, tomas-M3.1] GWTC-3 dark sirens added to the third-methods figure

`gen_ch00.py` `MEASUREMENTS` gains `gwtc3`: **H₀ = 68.0 (+8.0/−6.0)**, *Abbott et al.
(2023), arXiv:2111.03604*, family `siren`, plotted at the bottom of the cast (the row
order is `reverse()`d, so appending puts it directly under GW170817). It carries its
citation in the table, the noscript fallback, the widget's `rec:` chip and the
provenance roster.

New generator block `genre_anchor` computes, rather than types, the two numbers the
prose quotes: `frac_of_own_H0` = 10.3% → printed **10%**, and `ratio_to_ceiling` =
4.47 → printed **4.5×** (against §2's 1.5661 ceiling). One paragraph after the figure
says what the row is and how far it sits from the arbitration ceiling — which is the
prologue's whole argument aimed at the book's own genre.

**Honesty note (deliberate, and the reason the wording is careful):** the quoted
interval is the collaboration's dark-siren analysis *combined with the GW170817
counterpart*; the catalogue events alone constrain far more weakly. The page therefore
never presents it as a before/after against the 2017 counterpart-only row (70.0
+12.0/−8.0) — that is a different analysis of different data, and the `note` field in
the JSON says so. The book asserts only what the citation supports: this is the
published state of the art in the genre, and it is ~4.5× wider than the ceiling.

## R-ch00-3 — [P2, ped-M9 / D4] Trap 0.A de-spoiled

Ledger #49a's verdict number (`h = 0.86`) and #9's (`h = 0.60`) are gone from the trap,
and "reparametrization-dependent" (C8's reveal) with them. Each of the three failures is
now named as a *phenomenon* plus its chapter chip: walks off the bottom of its prior
`⏭ Ch 4`; reports the same answer whatever universe it is given `⏭ Ch 10`; moves when its
variable is redefined `⏭ Ch 8`. The ledger chips stay, so a reader who wants the number
can still look it up — they just will not be handed the punchline ten chapters early.

The same spoiler lived a second time in the provenance panel's forward-reference item
(*"the h=0.86 H₀-independent estimator"*). That is fixed too — the AC is "no #49a verdict
text before ch10", and the panel is on the page.

## R-ch00-4 — [P2, expA-m6] "The most generous case imaginable" → the named branch

Expert A is right: the arithmetic uses σ_A = 1.04, i.e. the method parked on the
early-universe anchor excluding the late one, and that is the *more demanding* of the
two placements (parked on the late-universe anchor, σ_A = 0.5 and the cap is 1.81).
Prose and arithmetic now name the same branch, and the generator emits both:

    ceiling_rhs                          1.8800   (= gap / 3)
    ceiling_sigma_total                  1.5661   parked on planck, far anchor shoes
    ceiling_sigma_total_parked_on_late   1.8123   parked on shoes, far anchor planck

Downstream numbers are unchanged and reproduce exactly: ceiling 1.5661 (2.32% of H₀),
41 events at zero systematic, static-fallback rows 3.01 / 2.50 / 4.49σ, σ_sys = 2
asymptote 2.502σ, in-browser cross-check max |Δ| = 0.0 on all five check points.

## R-ch00-5 — [P2, mara-MINOR-5] Q0.3: **re-aimed** (decision recorded, as the AC asks)

Mara is right that the old stem ("name the one thing you still need") was answered four
times on the page. Rather than accept it as a warm-up, it is re-aimed one step past the
body: the stem now *gives* the redshift gap (which is what keeps the Q0.3 → Ch 1 link in
ped-m2's transfer chain intact) and asks the transfer question instead —

> In §2's budget, which side does an inferred redshift land on — and can more events buy
> you out?

The answer applies the chapter's own theorem to the book's subject: the per-event guess
is noisy and averages down, but the machinery the guess is made *with* is common to every
event, so it does not; and §2 says a systematic above the ceiling is fatal at any N.
Phenomenon-level only — no catalogue numbers, no chapter's reveal, D4-clean.

## R-ch00-6 — cross-chapter consistency touches (not on ch00's worklist, done here)

- The legend's *"Three of the estimator's measured inconsistencies are live"* → **Four**,
  matching ch11's post-cell-B board (C5, C7, C8, C9 — worklist §C-ch11 BL-6) and the
  index's corrected sentence (§C-index MJ-7). The old "three" disagreed with the board
  even before cell B landed.
- ch00's two `IDEALIZED_BASELINE_READOUT.md:42-47` chips (GW-reader fold, provenance
  roster) carry a `title="current lines: :54-60"` tooltip — the same treatment ch03
  already ships, per §D item 10's anchor-drift policy and expA-m1's drift table, which
  lists ch00 as a citer although §C-ch00 does not. The cited anchor string is left
  intact; only the current-line tooltip is added. If the integrator's §D-10 pass
  re-greps generators instead, this is compatible.

## R-ch00-7 — verification

- `gen_ch00.py` re-runs clean and deterministically (0.015 s; JSON 8,802 bytes).
- `qa_gates.py`: D1 / ROW / DNR / TNS — ch00 clean on all four (the one repo-wide TNS
  failure is `index.html:148`, integrator-owned).
- Headless Chromium over HTTP: 4 Plotly figures, 173 KaTeX spans (including the maths
  inside the new algebra fold), **zero console errors**, in-browser cross-check
  max |Δ| = 0.0, and every scripted readout matches its static placeholder
  (1.57 / 2.3% / 1.81 / 1.88 / 1.04 / 0.50 / 10% / 4.5× / 2.50 / h = 0.73).
- Grep: no `0.86`, `0.60`, "reparametrization-dependent", "most generous case",
  "five published determinations" or "third and fourth rows" survives anywhere on the
  page (the only `0.86` left in the file is a CSS `font-size`).

## Still open / not ch00's to fix

- §D-2's shared `Book.loadJSON` failure surface: ch00's `.catch()` still only
  `console.error`s (ux-MAJOR-1). Integrator item; the page will adopt the wrapper.
- §D-1's Symbol Passport gating and §D-6's canonical strings: nothing on this page
  depends on them, but ch00 is where the passport's `H0` and `h` terms are first used.
