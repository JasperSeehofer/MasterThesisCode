# ch10_FLAGS.md — Chapter 10 ("Is It Calibrated?")

Raised by the ch10 agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, stop and flag; do not silently reconcile in
either direction."*

Nothing here blocks the chapter. Every item is presented **on the page in both forms**,
and both forms are emitted into the chapter's data files so a reviewer can check either.

---

## F-ch10-1 — C11's two quoted bias bands: lower endpoints reproduce exactly, upper endpoints do not — **OPEN**

- **Spec / cited value.** `CLAIM_2D_BIAS_20260730.md` C11 (and, verbatim,
  `gate_b_20260730/ADJUDICATION_20260730.md` §1): *"pp_coverage extension to comp_frac
  0.008–0.234 … bias **+0.0008..+0.0097** at comp_frac 0.06–0.09 and **+0.0034..+0.0181**
  at 0.13–0.24 … **6–16×** below +0.077."* Carried into `BOOK_SOURCES_MAP.md` §3 X4 and
  `BOOK_DESIGN.md` §1 Ch 10.
- **Measured by `gen_ch10.py`.** Re-running the archived harness cells that span exactly
  that completion-fraction window — `results/pp_coverage_deepvenue_20260730/`
  (`z_support` 0.38/0.39/0.41/0.43) plus `results/pp_coverage_deepvenue_20260710/`
  (`z_support` 0.2/0.3/0.5/1.0), all `kernel="volume"`, `mixture_mode="two_branch"`,
  σ_z ∈ {0.015, 0.035}, truths {0.62, 0.72, 0.84} — gives

  | band | claim | recomputed | agreement |
  |---|---|---|---|
  | comp_frac 0.06–0.09 | +0.0008 … +0.0097 | **+0.0008 … +0.0078** | lower endpoint exact; upper differs |
  | comp_frac 0.13–0.24 | +0.0034 … +0.0181 | **+0.0034 … +0.0157** | lower endpoint exact; upper differs |

  The window endpoints themselves reproduce exactly (measured comp_frac range
  0.00847–0.2337 vs the claim's "0.008–0.234").
- **Strength of the recomputation.** This is not an independent re-implementation: the
  generator re-runs each cell **from its own archived `config` block**, with the archived
  `seed`, and asserts **bit-equality** of `coverage`, `map_bias`, `rail_fraction` and
  `completion_fraction` against the stored JSON before writing anything. All 16 cells ×
  3 truths pass. The archives' own `.log` files carry the same values
  (e.g. `pp_zs0.38_sz0.035_volume.log`: `h_true=0.7200 … bias=+0.0078`;
  `h_true=0.8400 … bias=+0.0157`). So the recomputed numbers *are* the archive's numbers.
- **Where the claim's upper endpoints may come from.** `+0.00963` at comp_frac 0.0847
  does exist — in `results/pp_fullpower_20260727/pp_cat_lcat_zs0.43_sky1e-4_h0.84.json`,
  which is a **different harness family** (`catalogue_mode=True`, `mixture_mode="lcat"`,
  the impostor-ball universe, n_realizations 2000). If C11's band pools the continuum
  `two_branch` cells with the catalogue-mode cells, `+0.0097` is accounted for. A scan of
  every `results/pp_*/**.json` found **no** archived cell reproducing `+0.0181` inside
  comp_frac 0.13–0.24 (nearest: `pp_cat_lcat_zs0.30_sky2e-4_h0.62`, comp_frac 0.2142,
  bias +0.0174 — again catalogue-mode, and a 120-event cell).
- **Does it change the verdict?** No, in either direction. C11's conclusion is that the
  completion leg's calibration is far too small to own the +0.077 2D bias. The recomputed
  maximum (+0.0157) is *smaller* than the claimed one (+0.0181), so the exoneration is if
  anything stronger. **This flag is about provenance, not about the verdict.**
- **Disposition on the page.** §4 quotes C11's numbers verbatim in the adjudicator's voice
  with its badge and chip, then shows the archive-gated recomputation immediately beside
  it and names the disagreement as a disagreement. `data/ch10_pp.json`
  (`c11_window.bands`, `c11_window.band_disagreement`) carries both, plus the pointer to
  the `pp_fullpower_20260727` candidate. **Nothing is reconciled.**
- **For the author / integrator.** Worth an explicit note in the claim file recording which
  harness families the C11 band pools, and re-deriving the "6–16×" ratio from the named
  cells. As stated, "6–16×" corresponds to a bias band of roughly [0.0048, 0.0128], which
  is neither of the two quoted bands.

---

## F-ch10-2 — "the 3 loudest carry 46%": denominator unstated — **AMBIGUITY, both carried**

- **Spec / cited value.** `IDEALIZED_BASELINE_READOUT.md:42-47` and
  `idealization_audit/IDEALIZATION_LEDGER.md` §1: *"The **3 loudest** (SNR 995–1425,
  z ≈ 0.016–0.021) carry **46%** of the total information by themselves."* Carried into
  `BOOK_SOURCES_MAP.md` §3 R1 and `BOOK_DESIGN.md` §1 Ch 10.
- **Measured by `gen_ch10.py`** on the canonical `run_seed61000/posteriors_fixed`, using
  the audit script's own statistic (signed 3-point second difference of Σᵢ ln Lᵢ at
  h ∈ {0.725, 0.730, 0.735}):
  - in-catalogue curvature **+241.335**, dark **−3.003**, signed total **+238.332**
    (ledger: 241.3 / −3.0 / 101% / −1% — **exact match**, gated);
  - σ_h = 3.24×10⁻⁴ → **σ_H0 = 0.0324 km/s/Mpc** (ledger: 0.032 — **match**, gated);
  - the three loudest in-catalogue events (889, 1536, 118; SNR 1425 / 1068 / 995) sum to
    **112.006**, which is **46.41 %** of the in-catalogue curvature and **47.00 %** of the
    signed total.
- **The ambiguity.** The ledger writes "46% of the total", but 46% is the *in-catalogue*
  share; the *total* share is 47.0%. Since the same paragraph reports the in-catalogue
  share as "101% of the total", the denominator convention is not uniform in the source.
- **Disposition.** The page quotes the ledger's "46%" with its chip and immediately gives
  both recomputed denominators. `data/ch10_closure.json` (`golden3.share_of_in_catalog`,
  `golden3.share_of_total`, `golden3.ledger_quotes`) carries all three numbers. No
  substitution is made.

---

## F-ch10-3 — I10.1's card label says "run 200 universes"; the archived ensembles are 120 — **LABEL, corrected on the page**

- **Spec value.** `BOOK_DESIGN.md` §1 Ch 10 and `BOOK_PEDAGOGY.md` Part 4 both describe
  I10.1's control as *"press 'run 200 universes' (precomputed grid)"*.
- **Measured.** Every archived cell used by the widget has `n_realizations = 120`
  (`pp_coverage_deepvenue_20260710`, `_20260730`). The 500-realization archives
  (`pp_coverage_absolute_20260726`) are `mixture_mode="absolute"` cells — a different
  estimator, not the two_branch ladder C11 rests on — and the 2000-realization ones are
  the catalogue-mode family.
- **Disposition.** The widget's button reads the count out of the data
  (`cells[].n_realizations`) and says **"run the 120-universe ensemble"**. The card's
  round number is not carried into the page. Binomial standard errors on the coverage
  numbers (±0.043 at 68% for n = 120) are shown next to them so the reader is not invited
  to over-read a third digit.

---

## Non-flags (checked, consistent — recorded so a reviewer need not re-check)

- σ→0 byte-identity md5s `1e81ba22` (1D) / `733c8d32` (2D): quoted verbatim from
  `HANDOFF_20260730.md` §1 and `REALISTIC_READOUT.md` §2 P5; not recomputed (the control
  posteriors are cluster-side). Presented as a **recorded** measurement, chipped.
- The 0.67 closure (MAP 0.670, 1343 events, σ_h = 4.42e-4, peak 0.670053, +0.12σ) is
  quoted from `HANDOFF_20260730.md:15-23`, **with** that source's own caveats (fitted bins
  sit 11.3σ out; the GPU array timed out at 1345 vs the baselines' 1590 detections).
- The n-scaling ladder (cov68 0.63 → 0.38 → 0.12 at h_true = 0.72) is read straight from
  the archives: 0.6333 (`pp_coverage_exactmode_20260711/pp_exact_zs0.3_sz0.035.json`,
  n = 250), 0.3833 and 0.1250 (`pp_coverage_noisemodel_20260711/pp_nscale_constsig_n{1000,4000}`).
  Matches `pp_coverage_noisemodel_20260711/SUMMARY.md:80-86`.
- `sig0_control` is **not** used anywhere in this chapter (sources map §7.6: it carries the
  `generator_marginal` estimand). It is *named* in §5 only as the object Gate A1 read to
  confirm C6, which is exactly what the claim file does.
- `REALISTIC_READOUT.md` §6's struck-out sentence "the 1D channel is the defensible one"
  is **not** used (sources map §7.3). §5 of the chapter follows C5 and the readout's own
  2026-07-30 amendment.

---

# REVISION pass — 2026-07-31 (ch10 agent, per `design/REVISION_WORKLIST.md` §C-ch10)

Append-only. Nothing above this line was rewritten; the pre-revision record stands.
Every item below names its worklist entry and what changed on the page.

## R-ch10-1 — [P0] expB BL-7: §5's forward promise now points at a landed control

`CELLB_READOUT_20260731.md` (jobs **6103219 / 6103220**, the resubmission of the
registered 6101146 / 6101147 after a pure-plumbing symlink failure; code `7fd60bb`)
landed the 2×2's cell B on 2026-07-31. §5 promised Chapter 11 an *unresolved*
prediction. Applied, exactly per BL-7's scope (the design lesson and the "design the
missing control yourself" beat are untouched, and the landing sentence still sits
**after** that exercise so the reader designs blind):

- `§5` closing paragraph: *"It is in flight. Chapter 11 shows you that prediction from
  the inside, unresolved, because the project has not resolved it either."* →
  *"It was submitted, and it landed on **2026-07-31** — after the pre-registration, and
  after this chapter's prediction was written down. Chapter 11 shows you the prediction
  from the inside first, and then what it got right and what it missed."*
  The registered prediction ("estimator owns it", cell B ≈ cell C in both channels) is
  quoted **verbatim above it**, unedited (D3's verbatim rule).
- Closing callout: *"one confounded attribution whose decisive control has not landed"*
  → *"one attribution that was confounded until the control was run."*
- Provenance panel: the `PREREGISTRATION_2x2_cellB.md` item **OPEN → FINDING**, dated,
  and now cites `CELLB_READOUT_20260731.md` + the result job IDs + the one-sentence
  resubmission note (D3's job-ID split; expB MJ-2's `:1150` site is the rail pip, below).
- Rail pip: the grey *"cell B (the decider) not landed"* pip → the **canonical** amber
  cell-B pip, copied verbatim from `BOOK_CANON.cellB.pipLabel` / `.pipNote`
  (`js/manifest.js`, §D item 6). Verified byte-identical to the canonical strings;
  `qa_gates.py`'s pip advisory no longer lists ch10.
- **Q10.5 is kept as history, not overwritten.** The original answer ("you can conclude
  **nothing** about which one") was correct at the state the question describes and is
  intact; a visually separated **dated postscript** was appended: 2D MAP 0.7300 → 0.7900
  = 72% of that channel's displacement, "which one" is now the estimator configuration,
  and "what could you conclude *before* the control" is still **nothing** — the
  transferable half.

**Judgement recorded (not a deviation, but worth stating):** the §5 heading *"The control
that was never run"* and the C6 `CONFOUNDED` adjudicator box are **kept**. BL-7 scopes the
edit to the last two sentences and forbids spoiling the design beat; the box is the state
the section's argument reasons *from*, it is chipped to the dated
`CLAIM_2D_BIAS_20260730.md`, and the resolution is stated two paragraphs later. The only
addition is a dated scope clause on the provenance panel's C6 entry, so nothing static at
the bottom of the page lets a resolved claim look open (museum MJ-8's mirror rule). C6's
badge flip to RESOLVED is ch11's item (BL-6), not ch10's.

## R-ch10-2 — [P1] tomas M10: the deck no longer answers the cold-open predict

Subtitle *"It recovers truth at −0.24σ. **That is not the same as being right.**"* → *"It
recovers truth at −0.24σ, twice. **What does that establish?**"* The discovery statement
moved **below** the predict widget, opening the "this chapter is about instruments"
paragraph. An honest wrong answer at `#ch10-predict-1` (Yes / No / can't tell) is now
possible: the deck poses the question the box asks instead of answering it (D4).

## R-ch10-3 — [P1] tomas M3.4: the §3 scenario table has an external scale

Caption added under the A–D table, citing the book's already-used LISA EMRI dark-siren
forecast — **Laghi et al. 2021, arXiv:2102.01708** (the same paper ch07 cites for the
comparable host-z kernel) — as **percent-level on H₀**. Row D's 3.6 km s⁻¹ Mpc⁻¹ is
**≈5%** of H₀ (3.6/73 = 4.9%): same order as the forecast, a little worse. Row A's 0.032
is **0.04%**, two orders below anything the literature claims — which is itself an
argument for reading it as a consistency baseline. The caption states the comparison is an
order-of-magnitude anchor only (different event counts, mission duration, population model
and catalogue assumptions; this row is 76 hosts from one seed). No repo number changed;
the anchor is external and carries an arXiv chip, matching ch00's convention for
literature values.

## R-ch10-4 — [P2] ped m5: I10.2's num-view

**Finding first: the AC was already met.** `details.num-view` → `#ch10-closure-table` has
been on I10.2 since the build (page `:555`, filled by the widget's own `table()` call with
the K-ladder: K, events used, % removed, curvature at truth, σ_H0, MAP). ped-m5's "the
real gaps are ch10's I10.2" does not hold against the shipped page; recorded here rather
than silently "fixed".

The *substance* of the review's complaint does hold — the widget argues from a class
decomposition (76 in-catalogue events carrying 101% of the curvature, dark −1%, the three
loudest ~46%, 54 of the top 60 in-catalogue) that appeared only in prose and in the
`<noscript>` fallback. A **second table** was added inside the same num-view
(`#ch10-closure-decomp`), computed live from `ch10_closure.json`: in-catalogue / dark /
signed total with shares, the three loudest with **both** denominators (47.00% of the
signed total, 46.41% of the in-catalogue curvature — F-ch10-2's ambiguity carried, not
resolved), the top-60 subset with its in-catalogue count, and σ_H0 at K = 0. Every number
is read from the gated data file; none is typed into the page.

## R-ch10-5 — [P2] mandate D1: the dossier row, and a generator key carrying the same slip

- Dossier `:735` → the canonical row, verbatim from `BOOK_CANON.sigmaDL.dossierRowHTML`:
  `d_L | 88.9 Mpc · σ_dL/d_L = 8.98×10⁻⁴`, followed by the canonical erratum note
  (`.erratumHTML`, verbatim). A page-local `.dossier .note` rule styles it as a footnote
  (`css/book.css` is frozen; promote in integrator pass 2 if wanted).
- **`gen_ch10.py` carried the same units slip and is fixed.** `sqrt(Σ_dLdL)` from the CRB
  is the **absolute** σ_dL in Gpc (same units as the `luminosity_distance` column), and it
  was emitted under the key `sigma_dL_over_dL` in `ch10_closure.json` —
  `event889.sigma_dL_over_dL = 7.98e-05` and 60 `top_meta` rows. Renamed to
  **`sigma_dL_Gpc`**, with a genuine **`sigma_dL_over_dL = σ_dL / d_L`** added beside it
  (same treatment the worklist mandates for `ch04_denominator.json`). Regenerated:
  event 889 now reads `sigma_dL_Gpc 7.98e-05` / `sigma_dL_over_dL 0.000898` — i.e. the
  generator independently reproduces D1's adopted spec value **8.98×10⁻⁴** from the CRB
  row. No page reads these keys (verified by grep), so nothing on the page moved.
- `qa_gates.py` D1 gate: **0 ch10 violations**; the retired string appears on the page
  only inside the erratum note.

## R-ch10-6 — [P2] expB MJ-2: job IDs

The only ch10 site that named job IDs is the rail pip (`:1150`), which reports a
*result* → `6103219 / 6103220` via the canonical pip note. The provenance entry carries
the full split (registered 6101146/6101147 → resubmitted 6103219/6103220) with the
one-sentence plumbing-failure note. §5 quotes the pre-registration but names no IDs, so
nothing there needed changing.

## R-ch10-7 — [D5, book-wide policy] the C7 threshold is a both-values item

Not in ch10's item list, applied because D5 is book-wide and ch10 quoted one half twice.
§1's G2b↔C7 collision sentence and the C7 rail-pip tooltip printed **0.256** alone; both
now carry **0.2644** (solving the same artifacts' corrected law for the 0.86 edge, which
is also where the delivered per-host measurement puts the crossing) with a pointer to
`ch07_FLAGS.md` FLAG-2. Nothing is adjudicated: ch10 states that the project has not
reconciled them.

## Verification (this pass)

- `gen_ch10.py` re-ran clean end to end: **61.5 s**, 16 archive cells × 3 truths all
  bit-gated PASS, the K = 0 MAP gate and the audit-script decomposition gate both pass,
  and the two pre-existing C11 band disagreements (F-ch10-1) still print as FLAGGED —
  unchanged, as they should be.
- `qa_gates.py`: ch10 clean on **D1** (retired σ_dL only inside the erratum), on **TNS**
  (no "not landed" / "in flight" / "still running" anywhere on the page) and on the
  canonical-pip advisory. Remaining gate failures are other chapters' pages.
- Page structure re-checked after editing: HTML tag balance clean, the single inline
  `<script>` passes `node --check`, and the four canonical BOOK_CANON strings
  (`dossierRowHTML`, `erratumHTML`, `pipLabel`, `pipNote`) are present **verbatim**.
