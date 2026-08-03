# ch03_FLAGS.md — Chapter 3 ("Which Galaxy?")

Raised by the ch03 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

Nothing here blocks the chapter. Every item is presented on the page in **both** forms
wherever the reader can see it.

Everything below is produced by `book/generators/gen_ch03.py` from
`master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv` (baseline reduced
GLADE+, 20,834,171 rows after the production prune) and
`results/campaign51_20260728/realistic_20260729/seed61000/`.

---

## F-ch03-1 — EMRI-889's distance precision: 8.0×10⁻⁵ (spec) is the **absolute** σ in Gpc, not the fraction

- **Spec value:** `BOOK_DESIGN.md` §1 Ch 1 and Ch 6, and `BOOK_PEDAGOGY.md` B4, all write
  "**σ_dL/dL = 8.0×10⁻⁵**". `ch04-loud-half.html` (shipped) carries the same string.
- **Measured:** from `seed61000/prepared_cramer_rao_bounds.csv` row 889,
  `sqrt(delta_luminosity_distance_delta_luminosity_distance)` = **7.98427×10⁻⁵**, and the
  CSV's `luminosity_distance` column is in **Gpc** (0.08887922 Gpc = 88.879 Mpc,
  `Detection.d_L` docstring: "Gpc"). So
  - σ_dL = 7.984×10⁻⁵ **Gpc** = 0.0798 Mpc, and
  - σ_dL/d_L = **8.983×10⁻⁴** (0.0898 %), an order of magnitude larger than the quoted
    ratio.
- **Reading:** the spec's number is the *absolute* 1σ in Gpc with a fractional label. The
  two are the same measurement; only the label is wrong.
- **Disposition:** Chapter 3 quotes **both** — "σ_dL = 7.98×10⁻⁵ Gpc = 0.0798 Mpc, i.e.
  σ_dL/d_L = 8.98×10⁻⁴" — and never prints "8.0×10⁻⁵" as a fraction. Both values ship in
  `ch03_skyball.json` (`sigma_d_L_Gpc`, `sigma_d_L_over_d_L`). No reconciliation is
  asserted beyond the unit reading, which is checkable from the CSV.
- **For other chapters:** Ch 1 (I1.2 dossier), Ch 2, Ch 6 (I6.1) and Ch 11 all print this
  dossier row. `BOOK_PEDAGOGY.md` Q3.4 also says "0.01% distance precision"; the measured
  figure is 0.0898%.

## F-ch03-2 — "tens of thousands of galaxies in the ball" is the tail, not the typical event

- **Spec value:** `BOOK_DESIGN.md` §1 Ch 3 opening hook ("The localization ball holds tens
  of thousands of galaxies and you cannot tell which is the host") and
  `BOOK_PEDAGOGY.md` Q3.4's answer ("Tens of thousands.").
- **Measured** (production ball rule `handler.py:519`, production candidate z-window
  `physical_relations.py:546`, all 1590 CRB rows):

  | quantity | 5% | 25% | median | 75% | 90% | 95% | 99% | max | mean |
  |---|---|---|---|---|---|---|---|---|---|
  | galaxies inside the sky ball | 1 | 232 | **1616** | 6759 | 20,322 | 35,466 | 68,411 | 546,158 | 7330 |
  | after the candidate z-window | 0 | 0 | **12** | 271 | 1708 | 4891 | 19,594 | 431,670 | 1330 |

  79 events have an **empty** sky ball; **552 of 1590** have zero surviving candidates.
  **EMRI-889's own ball holds 3 galaxies, 2 after the z window.**
- **Reading:** "tens of thousands" is correct for the top ~10% of events (and event 1121,
  featured in I3.1, really does hold 57,535 → 26,652), and is *not* the typical event.
- **Disposition:** the chapter opens on the **measured distribution** rather than the
  slogan, features both extremes (889: 3; 1121: 57,535), and quotes the percentile table
  in full. Q3.4's answer text is amended on the page to name the measured median and to
  keep the "tens of thousands" statement scoped to the tail — the mechanism the answer
  teaches (angular volume dominates; stacking, not identification) is unchanged.
- **Venue caveat:** measured on the **baseline (truth)** catalogue; see F-ch03-3.

## F-ch03-3 — the reconstruction venue is the TRUTH catalogue; the run's own `L_cat` is not

- The seed-61000 evaluation ran `absolute_marginal` × `volume_deconv` against **observed
  catalogue realizations** `realizations_20260729/observed_catalogue_seed90000{1..5}.csv`
  (`REALISTIC_READOUT.md:1-11`). Those CSVs are **not in this checkout** (nor is the #53
  parent catalogue — `BOOK_SOURCES_MAP.md` §7.19d).
- The injection side, which is where `Detection.host_galaxy_index` comes from, used the
  **baseline** catalogue (campaign #53 convention (A),
  `realistic_host_observation_model.md` §1.2; `main.py` refuses to pair an observed
  catalogue with any generative stage). Confirmed numerically — see F-ch03-6.
- **Consequence, measured:** rebuilding event 889's catalogue leg with the project's own
  `single_host_likelihood` on the truth catalogue and the run's own Σ_global(h)
  reproduces the run's `L_cat_no_bh` to a factor **0.815–1.319 in the grid wings** but
  puts the peak at **h = 0.77** where the run's own leg peaks at **h = 0.75**.
- **Disposition:** the page shows both curves, side by side, labelled by venue, and states
  that they are *not* the same measurement. The leading candidate explanation — the
  observed realization's scattered host redshifts — is named **as a candidate, not a
  verified cause**; verifying it needs the realization CSV, which is Chapter 9's territory.
  Emitted as `production_889.vs_measured` in `ch03_ratio.json`.

## F-ch03-4 — ledger #26 bundles **two** changes; "a rearrangement worth 0.010" is half the story

- **Spec phrasing:** `BOOK_DESIGN.md` §1 Ch 3 I3.2 AHA — "an 'obviously equivalent'
  rearrangement is worth 0.010 in h"; `BOOK_PEDAGOGY.md` Q3.3 — "fixing it to Gray's
  A.9/A.10 ratio-of-sums **halved** the 1D bias, 0.750 → 0.740 (ledger #26)".
- **Ledger row #26 verbatim** (`BIAS_HISTORY_LEDGER.md:42`): *"`L_cat` departs from Gray
  A.9/A.10: **spurious p_det in numerator + mean-of-ratios**"* → *"fixed, sign-test
  PASSED: 1D 0.750→0.740 (halved), 2D 0.7375→0.7350"*, artifact `H0R:1119-1139`,
  commit `816f904`.
- **Reading:** the measured 0.010 is the **joint** effect of removing the spurious
  numerator `p_det` *and* replacing mean-of-ratios by ratio-of-sums. No artifact in this
  checkout splits the two.
- **Disposition:** the chapter states both components of the fix in the same sentence as
  the number, every time, and never attributes 0.010 to the rearrangement alone. It backs
  the rearrangement's *own* size with a direct measurement instead (F-ch03-9).

## F-ch03-5 — the chapter teaches the local ratio of sums; the run it quotes elsewhere uses the **global** denominator

- `weighted_ratio_of_sums` (`bayesian_statistics.py:804`) —
  `L_cat = Σ_ball w_g N_g / Σ_ball w_g D_g` — is selected by the `local_ratio` /
  `volume_deconv` modes (`G2c` §4.2/§4.3).
- The seed-61000 run's mode, `absolute_marginal`, takes the **other** branch
  (`bayesian_statistics.py:3190`): `L_cat = Σ_ball w_g N_g / Σ_global w_g D_g`, the
  full-catalogue selection sum of `precompute_global_catalog_selection` — `G2c` §4.1, the
  *literal* (A10) denominator.
- **Disposition:** Chapter 3 teaches the ratio of sums as the chapter card requires, and
  says explicitly, in the GW-reader stratum and the provenance panel, that the campaign
  the book reads elsewhere runs the global-denominator variant, with a forward chip to
  Ch 9 (the normalization-modes chapter). Both denominators are emitted in
  `ch03_ratio.json` (`production_889.L_cat_ratio_of_sums` vs `.L_cat_absolute_marginal`,
  the latter using the run's own logged Σ_global).

## F-ch03-6 — `host_galaxy_index` is an injection-frame (truth-catalogue) position — verified

- Not a disagreement; recorded because it is the assumption every host-identification
  beat in the book rests on, and `resolve_host_recovery_position` (`handler.py:659`)
  documents that the two frames are **not** positionally interchangeable once an observed
  realization is in play.
- **Gate (hard, in `gen_ch03.py`):** baseline pruned row **859360** carries
  `M = 709,540.71 M☉`, `z = 0.021268457`; the detector-frame lift `M(1+z) = 724,631.54 M☉`
  reproduces the CRB row's `M` to a relative **2.96×10⁻¹⁰**. The generator raises if this
  ever exceeds 1×10⁻⁵. Emitted as `host_frame_gate` in `ch03_candidates.json`.

## F-ch03-7 — this chapter writes `book-predict:ch03-host-guess`

- Per `WIDGET_REQUESTS.md` R-ch11-2, Ch 11 probes `ch03-host-guess` **first** when
  re-surfacing the reader's Chapter-3 host guess. This chapter uses exactly that string,
  on both the `.widget` and the `.predict-row` (the `predictReveal` container quirk noted
  in R-ch04-1). Values written: `"impostor"` / `"host"` / `"cannot-tell"`.

## F-ch03-8 — the production h-grid **undersamples** EMRI-889's own distance shell

- Measured: one 0.005 h-step moves 889's distance shell by 0.68% in `d_L`, i.e.
  **7.6 σ_dL**. On the 41-point production grid the point-kernel (delta-redshift)
  numerator of a *single* galaxy is therefore invisible between samples — the generator
  measures `n_in_window = 0` at every production grid point for this event.
- **This is not a defect in the pipeline**: production never evaluates a delta kernel on
  this grid; the host-z kernel (σ_z = 0.0044 for 889's host, ≈233× the GW distance width
  in the same units) smears each galaxy's contribution across ~0.15 in h. It *is* a
  constraint on the chapter's own figure, so I3.1 draws its own uniform 0.0005 display
  grid and says so; every number compared with the pipeline stays on the production grid.

## F-ch03-9 — how much the rearrangement is worth **per event**, measured here

- Ratio-of-sums and mean-of-ratios are algebraically identical when every `D_g` is equal,
  so the size of the difference is set by the spread of `D_g` across a candidate set.
  Measured in the Gray (A.9) point-kernel limit with the production `p_det`, on three real
  candidate sets:

  | event | candidates | max/min `D_g` at h = 0.60 | spread of ln(ratio-of-sums / mean-of-ratios) across the grid |
  |---|---|---|---|
  | 889 | 2 | 1.001 | 8.05×10⁻⁴ nats |
  | 1121 | 26,652 | 1.145 | 3.44×10⁻² nats |
  | 676 | 18,839 | 3.353 | 7.84×10⁻² nats |

- Event 676 was selected by measurement (largest `p_det` spread across candidates of any
  event in this run with ≥ 50 candidates), not by taste.
- **No stack-level extrapolation is made from these numbers** — the per-event differences
  do not all carry the same sign, and this checkout has no artifact that isolates the
  rearrangement's stack-level effect from the numerator-`p_det` half of ledger #26
  (F-ch03-4). The chapter reports the per-event measurement and the recorded joint
  verdict, and says so.

## F-ch03-11 — Trap 3.B's "84%+" clause is dropped, per the binding amendment rule

- `BOOK_PEDAGOGY.md` Trap 3.B ends: *"and in the 2D channel they end up carrying 84%+ of
  the channel difference."*
- `BOOK_DESIGN.md` §3.3 item 3 forbids exactly that: *"'84.2%' is r1-specific (print the
  replicated qualitative claim, not the number, as the finding)."*
- **Disposition:** the chapter prints Trap 3.B without the percentage. The qualitative
  claim it supports (dark events dominate the 2D channel difference) belongs to Ch 8 and is
  not previewed here at all, so no replacement number is needed at this rung.

## F-ch03-12 — new measurement: how concentrated a candidate sum actually is

- Not a disagreement — a measurement this chapter needed and the spec does not carry, so
  other chapters should know it exists before quoting anything adjacent.
- **Measured** (Gray A.9 point-kernel numerator, truth catalogue, evaluated at the injected
  $h = 0.73$, over the **1038** of 1590 events that have at least one candidate):

  | quantity | 5% | 25% | median | 75% | 95% |
  |---|---|---|---|---|---|
  | largest single galaxy's share of $\sum_g w_g N_g$ | 0.0125 | 0.135 | **0.563** | 0.995 | 1.000 |
  | effective number of contributing galaxies $(\sum x)^2/\sum x^2$ | 1.00 | 1.01 | **2.34** | 15.1 | 196 |

  One galaxy carries **>50%** of the weighted numerator for **52.6%** of those events, and
  **>90%** for 35.3%.
- **Scope, stated on the page:** this is the **point-kernel** (delta-redshift) limit. The
  production host-z kernel is much wider than the GW distance window (for EMRI-889's host by
  a factor ≈233 in the same units, F-ch03-8), so the production sum is far less
  concentrated than these numbers. The chapter uses them only to dismantle Trap 3.A
  ("one galaxy often does dominate — just not the nearest one, and only because you
  pretended the redshift was a number") and never as a statement about the production
  estimator.

## F-ch03-10 — 552 of 1590 events have **no** catalogue candidate in the truth-catalogue reconstruction

- Recorded for Ch 4 (§5's zero-host plant) and Ch 5 (C4, the zero-host fallback,
  ledger #54/#55/#57). It is a *reconstruction* number on the truth catalogue and is not
  the run's own drop count: the run reports 76 in-catalogue events of 1588 and **zero**
  zero-likelihood events (`real_r1/posteriors/diagnostic_report.md`), because the
  completion branch carries events whose ball is empty. Chapter 3 quotes it only as
  "candidates present", never as "events dropped".

---

# REVISION 2026-07-31 — post-review pass (`REVISION_WORKLIST.md` §C-ch03)

Appended, not rewritten: everything above is the record as it stood at build time. This
section records what the revision pass changed and why, and opens two new flags.

## F-ch03-2 — CORRECTION: the census above was measured at the WRONG search radius

- **What was wrong.** `gen_ch03.py:160` read `SIGMA_MULTIPLIER = 2  # handler.get_possible_
  hosts_from_ball_tree default`. `2` is the *signature default* of
  `handler.get_possible_hosts_from_ball_tree` (`handler.py:568`) and production never uses
  it. The **only** production ball-search call site is `bayesian_statistics.py:2838`, which
  passes `sigma_multiplier=1.5` explicitly. The `2.0` six lines above (`:2823`) is an
  argument to `get_redshift_outer_bounds` — a different multiplier for a different cut (the
  candidate z-window) — and is almost certainly how the two got crossed. Ch 6 had it right
  all along (`gen_ch06.py:165-166`), so the book carried two different ball populations for
  its own running example.
- **Caught by:** expert A B1, Tomas B1, Mara BLOCKER-2 (three independent reviewers,
  three independent recomputations of 889's radius). Adjudicated in `REVISION_WORKLIST.md`
  §A-D2 / §B-2; Mara's "two live candidate-search call sites" reading is rejected (§G-1) —
  there is one.
- **Fixed:** `gen_ch03.py` now sets `SIGMA_MULTIPLIER = 1.5` pinned to the production call
  site, with the trap written out in the comment; `ch03_candidates.json.meta.ball_rule`
  names the call site and says the signature default is not production; §1's RATIFIED box
  states `n_σ = 1.5` and spends a sentence on the signature-default trap, so the error
  cannot recur silently.
- **The census, re-measured at n_σ = 1.5** (all 1590 seed-61000 rows, 20,834,171-row pruned
  catalogue; 2σ column shown for the record):

  | quantity | 5% | 25% | median | 75% | 90% | 95% | 99% | max | mean | zero |
  |---|---|---|---|---|---|---|---|---|---|---|
  | in the sky ball (**1.5σ, production**) | 0 | 118 | **888** | 3811 | 11,272 | 19,851 | 37,653 | **311,430** | 4116 | **95** |
  | in the sky ball (2σ, retired) | 1 | 232 | 1616 | 6759 | 20,322 | 35,466 | 68,411 | 546,158 | 7330 | 79 |
  | after the z-window (**1.5σ, production**) | 0 | 0 | **6** | 149 | 954 | 2725 | 11,323 | **245,334** | 751 | **607** |
  | after the z-window (2σ, retired) | 0 | 0 | 12 | 271 | 1708 | 4891 | 19,594 | 431,670 | 1330 | 552 |

- **EMRI-889:** ball radius **0.7568′** (0.012613°), **2** galaxies in the ball, **2** after
  the z-window — i.e. the window removes nothing for this event. Was 1.009′ / 3 / 2. The two
  candidates are the *same* rows (760983, 859360), so every 889-specific number on the page
  is unchanged: separations 0.22841′/0.22842′, weights 88.50/92.14, h\* 0.7294/0.7675,
  σ_h 0.155/0.0089 vs 0.00066, ratio 236×, peak-lnN gap 0.005 nats, peak-ln(wN) gap 0.035
  nats, the ratio-of-sums vs mean-of-ratios spread 8.05×10⁻⁴ nats, the D_g factor 1.001, and
  the F-ch03-3 reconstruction (wings 0.815–1.319, peak 0.77 vs the run's 0.75).
- **Event 1121:** 32,825 in ball → **15,242** candidates (was 57,535 → 26,652); radius
  3.7217° = 223.30′ (was 4.963°); n_eff between **369 and 544** (was 436–635); largest single
  share ≤ **0.81%** (was 0.74%).
- **Event 676 (I3.2's third set):** **10,829** candidates (was 18,839); form-separation
  7.78×10⁻² nats (was 7.84×10⁻²); D_g factor 3.353 at h = 0.60, unchanged.
- **Event 1121's form-separation:** 3.42×10⁻² nats (was 3.44×10⁻²).

## F-ch03-10 — CORRECTION: the zero-candidate count is **607 of 1590**, not 552

Same cause, same fix. The discipline around the number is unchanged and still binding: this
is a *reconstruction* count on the truth catalogue ("events with no catalogue candidate"),
**never** a drop count — the run itself reports 76 in-catalogue events of 1588 and zero
zero-likelihood events, because the completion branch carries events with an empty ball.
Ch 4 §5 and Ch 5 §4 must consume **607**, not 552.

## F-ch03-12 — CORRECTION: the concentration statistics, re-measured

Gray (A.9) point-kernel numerator, truth catalogue, at the injected h = 0.73, over the
**983** of 1590 events with at least one candidate (was 1038):

| quantity | 5% | 25% | median | 75% | 95% |
|---|---|---|---|---|---|
| largest single galaxy's share of Σ w_g N_g | 0.0134 | 0.146 | **0.5516** | 0.996 | 1.000 |
| effective number of contributors (Σx)²/Σx² | 1.00 | 1.008 | **2.2888** | 12.85 | 170.4 |

One galaxy carries **>50%** for **52.0%** of those events (was 52.6%) and **>90%** for
**35.1%** (was 35.3%). The scope statement above (point-kernel limit only; the production
host-z kernel is ~233× wider than the GW distance window for 889's host, so the production
sum is far less concentrated) is unchanged and still the only licensed reading.

## F-ch03-1 — RESOLVED by author mandate: σ_dL/d_L = 8.98×10⁻⁴ is now the spec value

`REVISION_WORKLIST.md` §A-D1 adopts the measured value book-wide. The page no longer prints
the dispute: the dossier row is the canonical string
`d_L  88.9 Mpc  ·  σ_dL/d_L = 8.98×10⁻⁴`, and §1 carries the canonical one-line erratum.
The measurement recorded above is unchanged and was correct; only its status changed from
"open disagreement with the spec" to "the spec was wrong and has been corrected".

## F-ch03-13 — NEW: the local catalogue's `z_error` column is not the campaign parent's

- **Observed.** Every width-sensitive number on this page is computed from
  `reduced_galaxy_catalogue.csv`'s `z_error` column in *this* checkout. The realization
  sidecar `real_r1/posteriors/realization_provenance.json` records
  `parent_csv_sha256 = 7af3f4f4a2…` (22,641,048 rows) — the cluster copy — while the local
  file hashes to `623527929d…`. `BOOK_SOURCES_MAP.md` §7.19(d): they differ in **exactly one
  column, `z_error`** (the cluster copy carries the #40b counted-once PV width), and it says
  in as many words: *use the cluster parent for width-sensitive work*.
- **Raised by:** expert A M6. Ch 6 carries an equivalent caveat; Ch 3 did not, and Ch 3 is
  where the ±0.155 / 236× punchline lives.
- **What is affected, precisely.** Position-derived quantities are parent-independent: the
  ball radii, the in-ball counts, the separations, h\*(g), the sky panels. Width-derived
  quantities are indicative: the σ_h figures in §4 (±0.155 host, ±0.0089 impostor, the 236×
  ratio), **and — less obviously — the after-window census itself**, because the redshift
  window is a `z ± σ_z` overlap test (`handler.py:623-632`). The in-ball column of the census
  is safe; the candidate column, the 6-galaxy median and the 607 zero-candidate count all
  ride on the stale column to some degree.
- **What is NOT affected.** The qualitative claim the chapter makes — a photometric catalogue
  redshift is worse than this event's distance measurement by ~2.4 orders of magnitude, a
  spectroscopic one by ~1.1 — survives any plausible PV width. The campaign runs themselves
  were evaluated against the true parent, so no result quoted from the project is at risk;
  only this page's recomputed digits are.
- **Disposition:** stated in the §4 note beside the three numbers and in the venue box of the
  GW-reader fold, both chipped to §7.19(d) and to this flag. Not reconciled — reconciling it
  needs the cluster parent, which is not in this checkout.

## F-ch03-14 — NEW: Ch 3's "solid angle" and Ch 6's ΔΩ are two different objects

- Raised by this revision, while checking the D2 acceptance criterion "ch03 and ch06 print
  the same radius, solid angle and ball population for EMRI-889". The radius (0.757′) and the
  ball population (2) do now agree exactly. **The solid angles do not, and should not.**
- Ch 3 quotes the **search disc**: π r² with r = 1.5√λ_max(Σ') — the area the ball search
  actually sweeps. For 889: **5.00×10⁻⁴ deg²**.
- Ch 6 quotes **ΔΩ = 2π|sin θ|√(σ_φ²σ_θ² − C_φθ²) = 2π√det Σ'** (`gen_ch06.py`, key
  `d_omega_sr`) — a localization-*ellipse* solid angle that does not contain the search
  multiplier at all. For 889: 1.0036×10⁻⁷ sr = **3.2945×10⁻⁴ deg²** = 1.186 arcmin².
  Reproduced independently here from row 889 of the CRB table.
- The synthesis's "n_σ = 1.5 ⇒ ΔΩ 3.29×10⁻⁴ deg²" (worklist §A-D2, from expert A's table)
  conflates the two: scaling Ch 3's own 2σ figure 8.885×10⁻⁴ by (1.5/2)² gives 5.00×10⁻⁴,
  which is what the regenerated data says. **No number was changed to make the two agree.**
- **Disposition:** Ch 3 prints its search-disc area and names Ch 6's ΔΩ beside it, in §1 and
  in the dossier row, saying which is which. A reader who meets both cannot conclude the book
  contradicts itself. Nothing is asked of Ch 6.

## Other items from the worklist, dispositioned

- **Q3.4 rewrite (worklist P1, mara BLOCKER-1 tail).** The answer no longer opens
  "Thousands to tens of thousands" and then measures otherwise. It opens with the honest
  shape — *there is no single number; the answer is a distribution spanning five orders of
  magnitude* — concedes that thousands is the right order for the kind of event the stem
  describes, then gives the regenerated census. Ch 2's BLOCKER-1 fix should consume the same
  numbers (median 888 / 6, 95th 2725, 607 zero-candidate, max 245,334).
- **`Sig` passport gloss (mara MINOR-7).** Ch 3's side only: §1's first use of Σ now carries
  an inline rung-safe appositive ("the measurement's covariance: how big the error ellipsoid
  is and which way it tilts"), so the sentence stands without the hover and without Γ. The
  `firstChapter` gating itself is the integrator's (§D-1).
- **Anchor drift (mara MINOR-8, expA m1; worklist P3).** Current-line `title=` tooltips added
  beside the spec anchors this page cites: `handler.py:505→554`, `:519→558/:617`,
  `:584-592→623-632`, `:594-603→634-644`, `:592→632`, `:605→644`, `:765→912`, and
  `IDEALIZED_BASELINE_READOUT.md:42-47→:54-60` (re-grepped: the "76 in-catalogue / 3 carry
  46%" text is now at :55-60). The spec anchors are kept as the citation, per §3.2.
- **"No single event's argmax moves" (found by this pass, not by a reviewer).** The I3.2
  verdict asserted this as a fixed string. At the production radius it is false for event
  1121: ratio-of-sums peaks at h = 0.67, mean-of-ratios at h = 0.86. It is not a real
  disagreement — 1121's leg spans 0.30 nats across the *entire* grid and the two peaks are
  3×10⁻⁴ nats apart, i.e. the argmax is degenerate. The verdict text is now computed from the
  data and says which case it is; the noscript fallback says the same. Events 889 and 676
  agree under both forms (0.73 and 0.60).
- **Event 676's "selected by measurement" claim, made true again.** The pick was made on the
  retired 2σ census. `gen_ch03.d_spread_scan` now re-derives it at the production radius and
  **raises** if 676 is not the argmax: of the 568 events with ≥ 50 candidates, 676 has the
  largest max−min spread in p_det at h = 0.60 (0.246→0.824), runner-up event 392
  (0.196→0.762) — a narrow margin, and by the *ratio* max/min event 392 would win, so the
  criterion is stated on the page as a difference. Emitted as
  `ch03_ratio.json.meta.d_spread_selection`.

## For the other chapters — the numbers to consume

Anything quoting Ch 3's census must use these, not the values in the build-time section
above: **median in ball 888 · median candidates 6 · 95th 2725 · max 245,334 · 607 of 1590
zero-candidate · 983 events with ≥1 candidate · EMRI-889: radius 0.757′, 2 galaxies in the
ball, both surviving the window · event 1121: 15,242 candidates · event 676: 10,829.**
