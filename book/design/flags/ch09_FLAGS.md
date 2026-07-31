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
