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
