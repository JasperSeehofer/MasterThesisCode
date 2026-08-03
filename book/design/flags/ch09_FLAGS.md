# ch09_FLAGS.md — Chapter 9 ("Building a Universe to Break Your Estimator")

Raised by the ch09 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

**Nothing here blocks the chapter.** Every hard gate in `gen_ch09.py` passed on first run:
`w_G(0.73) = 0.1215037`, `r(0.73) = 0.39248`, mass-aware `w_G = 0.05149`, realized
`164/3135`, binomial `z = −11.86` / `+0.21`, the four C9 counterfactual MAP/mean pairs, all
twelve published Ω_m mis-specification cells, G1's own end-to-end tilts, and the
`derail_matrix_results.json ≡ G3_ablation_cube.json` identity on all four shared modes.

The items below are (a) two places where a *downstream* number inherits an already-registered
open dispute, (b) three provenance clarifications that would silently corrupt another
chapter's figure if inherited, and (c) one observation about a number Ch 1 / Ch 4 already
shipped. Each is presented on the page in both forms wherever it is visible to the reader.

---

## F-ch09-1 — the ×2.19 / ×2.45 mixture-weight ratio inherits the §7.7 dispute — BOTH SHOWN

- **Spec value:** `CLAIM_2D_BIAS_20260730.md` C9 — *"The #51 → #53 switch changed the delivered
  mixture weight by **×2.19**"*. That factor is `0.1215037 / 0.0555`, i.e. it uses C9's own
  quoted `generator_marginal` value at truth.
- **Measured by `gen_ch09.py`:** the `generator_marginal` value at truth measured in the
  `sig0_control` diagnostics CSV is **0.0496786** (verified to 5e-7 by the generator, and
  identical to the value recorded in `BOOK_SOURCES_MAP.md` §7 item 7), giving **×2.446**.
- **Status:** this is not a new disagreement — it is the *arithmetic consequence* of the
  already-registered OPEN item, sources map §7 item 7 ("the exact curve attribution is OPEN").
- **Disposition:** the page prints **both** ratios, side by side, labelled with which input each
  uses, inside the §4 adjudicator block that names the dispute; and the JSON carries
  `event_889.ratio_53_over_51_claimed` and `..._measured`. Neither is asserted as *the* number.
- **For the integrator / Ch 11:** any chapter quoting "×2.19" should carry the same pairing, or
  quote it as "C9's value" rather than as the measurement.

## F-ch09-2 — the two `generator_marginal` w_G point sets — BOTH PLOTTED

- Restatement of sources map §7 item 7, recorded here because ch09 is the chapter that plots it.
- C9 quotes `0.0774 / 0.0692 / 0.0555 / 0.0427` at h = 0.60/0.64/0.73/0.86; the diagnostics CSV
  measures `0.0686001 / 0.0614573 / 0.0496786 / 0.0385580`. `gen_ch09.py` re-measured the CSV
  values from `seed61000/sig0_control/diagnostics/event_likelihoods.csv` and they match the
  recorded set exactly (gate: 5e-7).
- **Disposition:** both series appear in I9.2 as distinct marker styles (filled = measured,
  open = as quoted in C9), the widget legend names the dispute, and the estimand caveat
  (sources map §7 item 6 — `sig0_control` carries the `generator_marginal` estimand) is stated
  in prose wherever the file is used. The CSV is used **only** to verify values that the design
  documents already record; no number in the chapter originates from it.

## F-ch09-3 — the 4-dp `w_G` log field: quoted floor 4.8e-4, measured max relative deviation 3.9e-4

- **Spec value:** `BOOK_DESIGN.md` §4.2 rule 4 / sources map §7 item 19(e) — never use the 4-dp
  `w_G` log line (`bayesian_statistics.py:2335`) for residual-level work, *"its 4.8e-4 noise floor
  is comparable to the entire 2D residual structure"*.
- **Measured by `gen_ch09.py`** over the 41-point grid, comparing that field against the
  full-precision `(D − β_Ḡ)/D`: **max absolute deviation 4.54e-5, max relative deviation
  3.95e-4**.
- **Assessment:** consistent in kind (4-decimal rounding of a quantity of size ~0.1 is a few
  times 1e-4 in relative terms); the spec's 4.8e-4 reads as a rounding *bound*, the 3.95e-4 as
  the *realized* maximum on this particular grid. Not treated as a contradiction, and neither
  number is adjusted.
- **Disposition:** the chapter states the rule and the reason; the numbers view carries the
  measured relative deviation, explicitly labelled as "measured on this run's grid". The 4.8e-4
  figure is not restated as a measurement anywhere on the page.

## F-ch09-4 — `c9_darkdraw_results.json` has THREE KS statistics; only one is current

- The file's **top-level** block reports `KS D = 0.1579` (full range) and `0.1197` (the
  z ≤ 0.722 "credible sub-test"). These come from a **local-pool approximation** and are
  explicitly superseded inside the same file.
- The **`production_pool`** block — fingerprint-verified against the run log
  (`dl_max = 9.164987` Gpc vs 9.165, rel. err 1.4e-6) — reports **`KS D = 0.0863`,
  `p = 1.08e-19`**, which is the value carried by `CLAIM_2D_BIAS_20260730.md` C9, by
  `BOOK_SOURCES_MAP.md` §3 N6, and by this chapter. The claim file states the direction of the
  correction explicitly: removing the pool-depth confound **narrows** the effect (median offset
  +0.047 → +0.037) rather than eliminating it.
- **Disposition:** `gen_ch09.py` reads the `production_pool` block only. Recorded so that Ch 11
  (which also carries the dark-side extension) does not pick up the superseded top-level number.

## F-ch09-5 — the injection pool has two writer eras; 6,000 rows carry no initial conditions

- `injection_pool_mix200k_20260728` concatenates rows from two `code_rev` values. The earlier
  era (`a9f29e82…`, 6,000 rows) predates the `t_plunge_yr` / `p0` columns; the later era
  (`f6449051…`, 194,100 rows) carries them.
- **Disposition:** the I9.1 population panel uses all stratum-'a' rows (99,014); the p₀ panel
  uses the **96,105** stratum-'a' rows that carry the columns, and the page and the generator
  docstring both say so. Rows are excluded, never imputed.
- **Related, already recorded:** `ch04_FLAGS.md` F-ch04-1 — the pool is 200,100 **data rows** in
  707 files = 200,807 **lines**. This chapter quotes the data-row count and the line count
  together, consistently with ch04.
- **Also:** the pool's `M` column is the **detector-frame** `M_z = M(1+z)` (`main.py:1291-1293`),
  and only stratum 'a' carries the population measure
  (`simulation_detection_probability.py:356-407`). Any chapter histogramming this pool without
  the stratum mask is plotting an importance-sampled mixture, not a population.

## F-ch09-6 — "σ_dL/dL = 8.0×10⁻⁵" appears to be σ_dL in Gpc, not the ratio — INFORMATIONAL

- **Spec value:** `BOOK_DESIGN.md` §1 Ch 1 dossier and `BOOK_PEDAGOGY.md` describe EMRI-889 as
  carrying `σ_dL/dL = 8.0×10⁻⁵`, sourced to
  `seed61000/real_r1/prepared_cramer_rao_bounds.csv`. Ch 4 propagates the same label.
- **Observed while reading that CSV for the dossier:** row 889 has
  `delta_luminosity_distance_delta_luminosity_distance = 6.37e-9` and
  `luminosity_distance = 0.0888792` Gpc, so `sqrt(diag) = 7.98e-5` **Gpc** — which is the
  quoted number — while the dimensionless *ratio* `sqrt(diag)/d_L` is **8.98e-4**.
- **Disposition:** ch09 does **not** recompute or requote this quantity — its dossier row omits
  σ_dL entirely and defers to Ch 1 / Ch 4. Raised for the integrator because two shipped
  chapters carry the ratio label on what appears to be an absolute width; the fix (if any) is a
  label change in Ch 1/Ch 4, and is not this chapter's to make.

## F-ch09-7 — G1's "end-to-end tilt" is a difference of normalized shapes — CONVENTION NOTE

- `docs/gates/G1_beta_g_check.json` reports `end_to_end_tilt_h3_corrected = −0.171917`. That is
  `shape(0.86) − shape(0.60)` on the h³-corrected shape normalized at `h_ref = 0.72`
  (1.085395 → 0.913479), i.e. "+8.7% falling to −8.7%", matching the prose in
  `G1_beta_g_check.md`. It is **not** a ratio: `shape(0.86)/shape(0.60) − 1 = −15.8%`.
- Likewise `end_to_end_tilt_raw = +0.9285` is the same difference on the *uncorrected* shape,
  while the ratio growth quoted in the prose is `ratio(0.86)/ratio(0.60) = ×2.478`.
- **Disposition:** `gen_ch09.py` gates on the file's own value with the file's own convention and
  the page quotes **−17.2%** and **×2.48** exactly as G1 does. Recorded so that no later chapter
  recomputes a ratio and reports a different "end-to-end" number.

---

# REVISION PASS — 2026-07-31 (post-review, `REVISION_WORKLIST.md` §C-ch09)

Appended, not rewritten: everything above is the record as it stood when the chapter
shipped. This section records what the revision pass changed, what it measured, and the
judgment calls it made. Generator re-ran clean (all pre-existing hard gates OK, one new
gate added); only `ch09_bench.json` changed on disk.

## F-ch09-8 — the pre-registered `w_G` equality: RE-MEASURED, exact — NEW HARD GATE

- **Registered** (`PREREGISTRATION_2x2_cellB.md`, before submission): *"w_G(h): expected
  bit-identical to the #53 runs (pure quadrature, no catalogue input). If it differs, that
  itself is a finding."*
- **Delivered** (the 2×2 cell B landed 2026-07-31; evaluate 6103219 / combine 6103220, the
  resubmission of the registered 6101146/6101147): the book does **not** copy the readout's
  claim. `gen_ch09.py` re-measures it two independent ways —
  (a) cell B's per-h `w_G` column vs campaign #53 r1's, element-wise over all 41 grid
  points: **max|Δ| = 0.0**, element-wise equality `True`;
  (b) the two runs' own `D(h)` and `beta_Gbar(h)` log legs: **identical numbers**.
  Reads: **0.1625175 / 0.1215039 / 0.1038732** at h = 0.60/0.73/0.81 — equal to the
  readout's three quoted values to 5e-8 (gate).
- **Disposition:** a new hard gate. If that equality is ever anything but exact the
  generator stops, quoting the pre-registration's own consequence clause. §6 carries the
  result as a dated readout block; the numbers view carries the measured `max|Δ|` and the
  grid-point count.
- **Precision note (kin to F-ch09-3):** the three quoted reads are the *diagnostics
  column's* full-precision values. The page's own curve is built the design-mandated way,
  `(D − β_Ḡ)/D` from the 7-s.f. log legs, which gives 0.1625174 / 0.1215037 / 0.1038737 —
  the two routes differ by at most **5.5e-7**, i.e. by the log lines' own rounding. Both
  appear on the page (0.1215037 in the cold open and the gates, 0.1215039 in §4 and §6),
  and §4 already states why they differ. Nothing was reconciled silently.

## F-ch09-9 — C9's cell-B gate is RELEASED; C9 itself is unchanged and live

- Page changes (worklist [P0] expB-MJ-5): the re-litigation guard, the I9.2 `aware|aware`
  verdict string, the provenance panel (OPEN → dated FINDING) and the rail pip now say
  that the **gate** was released on 2026-07-31 and that what remains is the author's
  leg-adjudication plus a `/physics-change` derivation, binding that **C9 and C8 are fixed
  jointly** (their counterfactuals act on different terms of the same mixture and are not
  additive — `CELLB_READOUT_20260731.md` Consequences; `ADJUDICATION_20260730.md` §5 item 6).
- **No verdict on this page changed.** C9 stays live and unfixed, the ledger-#61 fix form
  stays exonerated-as-a-form and must not be re-tried, and the adjudicator's leverage
  discount still travels with the counterfactual.
- The rail pip is now the canonical one (`BOOK_CANON.cellB.pipLabel`, read at runtime with
  the identical string as a literal fallback), so ch07/ch09/ch10/ch11 read alike.

## F-ch09-10 — "residual" was two different objects in adjacent boxes — RENAMED, no number moved

- §5's I9.4 reveal and the "Has this been tried?" box both said *residual* for quantities
  10× apart and opposite-signed (tomas M6). They are now **"the shape residual"**
  (−17.2% end-to-end, a tilt in Σ_glob/β_G across G1's 14 h-values) and **"the 1D bias
  residual"** (+1.667%, i.e. +0.017 in h), with one clause stating they are different
  objects and different measurements.
- The `(0.73/0.81)³ − 1 = −26.80%` parenthetical is **re-scoped, not deleted**: it belongs
  to the exonerated β_G/Σ_glob drift measured on campaign #53's own legs over
  h = 0.73 → 0.81 (measured −26.84% on the with-BH leg), *not* to the ×2.48 growth over
  0.60 → 0.86 that it sat beside. Both statements are now attached to the h-pair they were
  measured on. No printed number changed value.

## F-ch09-11 — the `global` deprecation is now venue-scoped (it was a bare claim about a published method)

- §3 previously deprecated `global` — the faithful transcription of Gray (A10) — in one
  clause ("~0% coverage, rails to the grid edge") with no venue (tomas M9). The page now
  states: the coverage figure comes from the 2026-07-01 commission's **from-scratch
  synthetic** calibration test (the real GLADE catalogue was never loaded), whose own note
  says the rail there needs the synthetic's discrete density to reproduce β_G and is
  therefore *"a limitation of the synthetic, NOT evidence about the real code"*
  (`results/commission_20260701/scratch/d2/NOTE_calibration_findings.md`); that on real
  data `global` has **never been run in a configuration matching the published machinery**
  (its two archived runs are de-rail steps 1–2, carrying this pipeline's ±3σ ball, the D2
  smooth-completeness deviation, the bare host-z kernel, G2c's C6 numerator/denominator
  photo-z asymmetry, and — step 1 — the pre-4π completion numerator); and that the
  on-catalogue evidence is §5's non-constant C, which indicts *this* bridge, not (A10)'s
  algebra.
- **gwcosmo is now named** as the reference implementation (tomas M3.2), with `G5a`'s
  inspection result (same LOS prior in numerator and selection denominator; catalogue
  z-pdfs declared posteriors) and the honest limit: G5a is a **code inspection**, and the
  commission's external numerical cross-check (item D7) is still recorded NOT-ATTEMPTED
  (sources map §7 item 15). The ~0% coverage of Ch 10's **bare host-z kernel** is
  explicitly disambiguated from this one — a different variable.

## F-ch09-12 — D1 (σ_dL) on ch09: the flag is answered; the dossier row is now the canonical one

- `REVISION_WORKLIST` §A-D1 lists ch09 as touching D1 "only via F-ch09-6 flag text", and
  F-ch09-6 above is exactly the observation D1 adopts: **8.0×10⁻⁵ is the absolute σ_dL in
  Gpc, the fraction is 8.98×10⁻⁴**. Recorded as **RESOLVED book-wide 2026-07-31** by the
  author's mandate; the flag text above stands as the record.
- **Judgment call (agent's, recorded here):** ch09's dossier row printed `d_L 88.9 Mpc`
  with no width at all, because F-ch09-6 deferred the label question to Ch 1 / Ch 4. With
  D1 decided, the row now carries the **canonical dossier string**
  (`d_L 88.9 Mpc · σ_dL/d_L = 8.98×10⁻⁴`, `BOOK_CANON.sigmaDL.dossierRowHTML`) so that the
  card reads identically on every chapter. **No erratum note was added to the page**: ch09
  never printed the retired value, so there is nothing on this page to correct — the
  erratum belongs on the pages that carried it. The D1 grep gate passes on ch09 either way.

## Coordination notes (other agents' files — not touched here)

1. **Σ_glob / n_w passport tags.** Ch 9 §3–§5 now tag `Sglob`, `nw`, `Wcat`, `Vf`,
   `Fincat`, `Lcat`, `Ng`, `Dg` (integrator's §D-7 entries; all resolve, 0 unknown keys).
   The same symbols still leak untagged into **ch07 §6** (`Δ ln Σ_glob`) and **ch11's C8
   dimension count** — those are ch07's and ch11's to tag (tomas M5).
2. **The 2×2 cell B naming (MJ-3)** is used throughout ch09. Museum ledger row #88's
   "Cell B" disambiguation clause is the museum agent's.
3. **Honest scoring of the pre-registration (D3 / expB BL-4)** lives in ch11 §5. Ch 9
   reports only its own read (the `w_G` equality) and points forward to the scoring,
   naming the miss as a phenomenon without its number (D4). Ch 9 does **not** repeat the
   readout's "confirmed on every pre-registered read" sentence, and does not print the 2×2
   table — that is ch11's centrepiece.
4. **BW3 tag.** I9.2 carries `data-hypothesis="61"` plus
   `data-hypothesis-verdict="inline"` (the §D-5 opt-out): the widget's verdict is already
   hard-coded in the re-litigation guard, so the shared inline chip must not double-report
   it.

## Items considered and deliberately not done

- **Q9.1** was re-aimed to ped-m1's harder transfer form (name a quantity the same
  argument would *not* license, and why). Q9.2–Q9.5 were left alone: the pedagogy review's
  full-21-question rewrite is rejected in §G-6 and none of ch09's others is on the named
  list.
- **§3's ledger-#49a paragraph** still states the H₀-independent verdict. It is the step-1
  state of the very matrix this section measures, and no worklist item asks for it; D4's
  enumerated list (decks, index blurbs, Traps 2.A/2.B/5.B, ch05 §3, the passport) does not
  include it. Flagged here so the integrator can overrule if the ch10 interlude wants the
  first telling.
- **Plot lazy-init** was implemented page-locally (ch03's `IntersectionObserver`
  precedent), since `Book.lazyPlot` is an integrator pass-2 item: all **8** Plotly
  instances now construct only when their enclosing `.widget` approaches the viewport —
  verified 0 built on load, 8 built on intersection, and no-JS/no-IntersectionObserver
  falls back to eager construction. The observer watches the `.widget`, not the plot div,
  because three of these plots sit inside a `display:none` `.reveal` that never intersects.
