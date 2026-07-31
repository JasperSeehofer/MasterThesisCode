# ch11_FLAGS.md — Chapter 11 ("The State of the Art, Honestly")

Raised by the ch11 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, **stop and flag; do not silently reconcile in
either direction**."*

Nothing here blocks the chapter, and nothing here changes any verdict. Each item is shown
to the reader **in both forms**, on the page, with both provenance chips.

---

## F-ch11-1 — the C5 leverage ratio `dh*/dε`: **1500–2400×** (adjudicated) vs **142–2458×** (recomputed)

- **Spec / document value.** `BOOK_DESIGN.md` §1 Ch 11 ("leverage 0.025; dh*/dε 1500–2400×
  idealized"), `BOOK_SOURCES_MAP.md` §3 X1, `BOOK_PEDAGOGY.md` §2.1 Ch 11 and Q5.4 — all
  tracing to `CLAIM_2D_BIAS_20260730.md` C5 (line 290) and
  `gate_b_20260730/ADJUDICATION_20260730.md` §1 C5 (line 119), which read:
  *"dh\*/dε leverage **1500–2400×** idealized"*.
- **Measured by `gen_ch11.py`** from the adjudicator's **own** output file
  `gate_b_20260730/c5_leverage_results.json`, dividing each realistic run's
  `dh_deps_incat` by its *own seed's* idealized baseline:

  | seed | r1 | r2 | r3 | r4 | r5 |
  |---|---|---|---|---|---|
  | 61000 (ideal 1.012e-4) | 170.8× | 279.2× | **2457.8×** | 203.9× | 470.6× |
  | 62000 (ideal 1.539e-4) | 172.5× | **141.8×** | 192.8× | 159.5× | 201.5× |

  Range **141.8×–2457.8×**, median **197.2×**. Only one of the ten runs (61000/r3) lands
  inside the quoted 1500–2400× band, and it lands just above it.
- **Internal cross-check that makes the recomputation the likelier reading.** The same
  adjudication sentence states, two clauses earlier, that class slopes at the MAP are
  **5× smaller** and total curvature **~1000× smaller** than idealized. Since
  `dh*/dε = −S'_in(h*)/S''_tot(h*)` (`attack_c5_leverage.py:196`), those two numbers imply a
  leverage ratio of **≈ 1000/5 = 200×** — which is the measured median, not 1500–2400×.
- **Disposition — NOT reconciled.** The chapter prints the adjudicated range
  *as the adjudicated range*, chipped to the claim file, **and** the per-run recomputed
  ratios in I11.1's numbers view, chipped to `c5_leverage_results.json`, with a visible
  note pointing here. Both are emitted into `ch11_runaways.json`
  (`published.claim_leverage_ratio_range` vs `published.measured_leverage_ratio_range`).
  No number is dropped, rounded toward the other, or silently preferred.
- **What is NOT affected.** The claim C5 rests on this only as a corroborating magnitude.
  Everything else in the leverage block reproduces **exactly** and is gated in the
  generator:
  - the **±1/√N Poisson class reweight moves the combined MAP by up to 0.025** —
    recomputed **0.024904** (seed61000 r5), against **0.000006 / 0.000015** idealized;
  - the full λ-scan (12 venues × 5 λ) reproduces `c5_class_weight_results.json` to
    < 1e-6, including λ = 0 → 0.635–0.644 and the in-cat/dark argmaxes 0.86 / 0.64;
  - all six Poisson combinations and every base MAP reproduce `c5_rail_results.json` to
    < 1e-6.
  So the *qualitative* finding the chapter is built on ("your own hands move the headline
  number, and the same gesture does nothing idealized") is measured, not inherited.
- **For the integrator / other chapters:** Ch 5's I5.2 plants this same object (the Two
  Runaways) and Q5.4's answer quotes "1500–2400×". If Ch 5 quotes the ratio it should
  carry this flag too, or quote the 0.025 Poisson figure instead, which reproduces exactly.

---

## F-ch11-2 — the "0.12–0.51 σ_h" gloss on the same Poisson shift does not reproduce

- **Document value.** `ADJUDICATION_20260730.md` §1 C5 line 121 and
  `CLAIM_2D_BIAS_20260730.md` C5 line 291: the ±1/√N reweight moves the combined MAP
  "by up to 0.025 **(0.12–0.51 σ_h)**".
- **Measured by `gen_ch11.py`** (each run's `poisson_max_shift` from
  `c5_rail_results.json` divided by that run's own `sigma_h` from
  `c5_leverage_results.json`): **0.085 – 1.124**, with seed61000/r5 — the very run that
  supplies the headline 0.0249 — at **1.12 σ_h**, not 0.51.
- **Disposition — NOT reconciled.** The chapter quotes the shift **in h** (0.025), which
  reproduces to 1e-6, and does not quote the σ_h gloss at all. Recorded here so that a
  later chapter or the paper does not pick it up unchecked. Both quantities are present in
  `ch11_runaways.json` (`runs[].poisson_max_shift`, `runs[].leverage.sigma_h`) so any
  reader can redo the division.

---

## F-ch11-3 — event 889's r2 channel swing: −2.04 (document) vs −2.0347 (recomputed)

- **Document value.** `gate_b_20260730/c3c4_allruns_summary.md:123` — event_idx 889's own
  channel difference "swings from **+1.98** (real_r1) to **−2.04** (real_r2) to
  **−3.30** (real_r3)".
- **Measured by `gen_ch11.py`** from each run's `diagnostics/event_likelihoods.csv`
  (`ln combined_with_bh − ln combined_no_bh`, differenced over h = 0.73 → 0.81):
  **+1.9832 / −2.0347 / −3.3004**.
- **Disposition.** r1 and r3 agree to the printed precision; r2 differs by 0.0053, i.e. the
  summary rounded −2.0347 to −2.04 **away from** 2 d.p. (−2.03). This is a rounding
  presentation difference, not a numeric disagreement, and is logged only because the
  chapter prints the swing as a headline. The chapter prints the **recomputed** values to
  4 d.p. with the summary's rounded triple beside them. Gate tolerance in the generator is
  0.011 so the discrepancy cannot grow silently.

---

## Not flags — checked and consistent

Recorded so a reviewer does not have to redo them:

- **C1 / C2 / C3 budgets** re-measured from `seed61000/real_r1/diagnostics/event_likelihoods.csv`:
  in-cat **+2.48**, dark **−11.77**, total **−9.30**; channel difference in-cat **+2.97**,
  dark **+15.83**, 2D total **+9.51** — all match the claim file to ≤ 0.02 nats (gated).
- **C9's realized in-catalogue rate** re-counted from both seeds' CRB tables:
  76/1590 + 88/1545 = **164/3135 = 0.052313** — matches the claim file exactly (gated to 1e-9).
- **Poisson σ fractions** back out to N = 76 / 1512 (seed61000) and 88 / 1454 (seed62000),
  consistent with `host_galaxy_index >= 0` counts.
- **The h grid** shipped in `ch11_runaways.json` is the production 41-point non-uniform grid
  (0.01 on [0.60,0.65] ∪ [0.80,0.86], 0.005 between). No second difference is taken anywhere
  on the page; the two sub-grid MAP refinements are the attack scripts' own, transcribed and
  gated.

---

## Observation (not a disagreement) — the two sub-grid MAP refinements differ on this surface

The project used **two** sub-grid MAP refinements in the same Gate-B session:
`attack_c5_class_weight.py:55-63` fits a parabola to the **five** points around the grid
argmax (the λ-scan), while `attack_c5_rail.py:185-193` uses the **three** nearest points (the
Poisson reweight). Both are reproduced verbatim by `gen_ch11.py` and by the chapter's JS,
and each reproduces its own published table to < 5×10⁻⁷.

They do **not** agree with each other on the near-flat realistic profiles. Taking the
largest ±1/√N shift per run: the 3-point convention (published) gives 0.0012–0.0249, the
5-point convention gives 0.0023–0.0234, and the per-run values differ by up to a factor 3
(seed62000/r1: 0.0108 vs 0.0035). This is not an error in either script — it is what
"near-flat" means, and it is arguably a sharper illustration of C5 than the numbers
themselves.

The chapter states this on the page rather than choosing silently: the live λ readout uses
the 5-point convention (λ is the control being dragged), the quoted ±1/√N figure is labelled
as the published 3-point measurement, and the numbers view tabulates both per run.
