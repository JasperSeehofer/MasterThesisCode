# REVISION_WORKLIST.md — authoritative revision spec (post-review synthesis)

**Synthesizer pass, 2026-07-31.** Inputs: the six reviews in `book/design/reviews/`
(`student_mara_ch00-05`, `student_tomas_ch06-11`, `expert_A_ch00-06_museum`,
`expert_B_ch07-11_cellB`, `pedagogy`, `ux_robustness`), `BUILD_REPORT.md` §5, and
`results/campaign51_20260728/realistic_20260729/CELLB_READOUT_20260731.md`.
This file is the single revision spec: chapter agents and the integrator work from it,
not from the raw reviews. Where this file and a review disagree, this file wins
(§B records why). Flag files remain the historical record and are **appended to,
never rewritten** — except where an item below explicitly updates a flag's status.

Priorities: **[P0]** ship-blocking / mandated · **[P1]** major, fix this round ·
**[P2]** minor, fix this round if the page is open · **[P3]** polish, optional.
Source tags: `[mandate]` author/orchestrator instruction · `[mara]` `[tomas]`
`[expA]` `[expB]` `[ped]` `[ux]` reviews · `[synth]` synthesizer adjudication.

---

## A. Book-wide decisions (read these before any per-chapter item)

### D1 — σ_dL units slip: DECIDED. Spec value is now σ_dL/d_L = 8.98×10⁻⁴ `[mandate]`

The six-chapter measured value is adopted as the spec value:
**σ_dL = 7.98×10⁻⁵ Gpc (absolute); σ_dL/d_L = 8.98×10⁻⁴ (fractional).**
The old spec figure `8.0×10⁻⁵` was the absolute Gpc value carried under a fractional
label (×11.25 slip). Consequences, book-wide:

- **Chapters STOP printing dual values for this item.** Every page prints the corrected
  fractional value with a **one-line erratum note**; the flag files remain the record.
- **Canonical dossier row** (single string, used identically on every dossier card
  ch01–ch11 + museum):
  `d_L  88.9 Mpc  ·  σ_dL/d_L = 8.98×10⁻⁴`
- **Canonical erratum line** (once per page that previously carried the dispute or the
  bad value; a footnote or small `.note`, not a boxed OPEN dispute):
  *"Erratum: the spec card carried σ_dL/dL = 8.0×10⁻⁵ — that is the absolute σ_dL in
  Gpc under a fractional label. Corrected book-wide 2026-07-31; record: ch01 flag F1 /
  BUILD_REPORT §5.1 item 1."*
- **OPEN badges on this item become erratum notes.** The pedagogical beats built on the
  dispute (ch01 F1 block, Q1.2 editor's note, ch06 §4 arbitration box) are **kept as
  lessons but reframed**: no longer "the book prints both and picks neither" — now
  "here is how the order-of-magnitude check caught a units slip, and here is the
  resolution." The 1/ρ discriminating check stays; it is now the *proof*, not an open
  arbitration.
- **Downstream numeric casualties to fix:** Q1.2's "0.008%" answer (rewrite as the
  erratum lesson); Q6.5's "6000×" → measured **≈550×**; `ch04_denominator.json` key
  `event889.sigma_dL_over_dL = 7.98e-05` renamed to `sigma_dL_Gpc` with a new
  `sigma_dL_over_dL = 8.98e-4` (gen_ch04).
- **Build gate (integrator):** after the pass, a repo grep must show the string
  `8.0×10⁻⁵` (and `8.0e-05`-style variants labelled as σ_dL/dL) appearing **only**
  inside erratum notes / flag files. No dossier, stem, table, JSON value or answer may
  carry it as a live value.

Affected pages: ch01, ch02, ch03, ch04, ch05, ch06, ch07, ch08, ch10, ch11, museum
(ch09 touches only via F-ch09-6 flag text). This retires review items: mara BLOCKER-3,
tomas B2, expA B2, expB BL-5, ped B2 — all of which proposed "print both"; D1
supersedes them (see §B-1).

### D2 — Production ball-search multiplier: DECIDED. n_σ = 1.5; ch03 regenerates `[synth, verified in-tree]`

Verified against the tree: `handler.get_possible_hosts_from_ball_tree` has signature
default `sigma_multiplier: int = 2` (`handler.py:568`); the **only** production
ball-search call site is `bayesian_statistics.py:2838` → `sigma_multiplier=1.5`.
The `2.0` at `:2823` is an argument to `get_redshift_outer_bounds` — a different
multiplier for a different cut (and per F-ch06-4 the handler's z-window multiplier is
dead code; the window is hardcoded ±3σ). Mara's "two live candidate-search call
sites" reading is wrong (§B-2).

- `gen_ch03.py:160` → `SIGMA_MULTIPLIER = 1.5`, comment pinned to
  `bayesian_statistics.py:2838 (production call site; handler signature default is 2 — do not use)`.
- **Every ch03 census number regenerates**: median in-ball / after-window counts, p95,
  max, empty-ball count, the zero-candidate count (currently 552/1590), concentration
  stats, the featured extreme, Q3.4's answer, EMRI-889's ball (becomes **2** galaxies,
  radius **0.757′**, ΔΩ 3.29×10⁻⁴ deg²), the `ball_rule` meta string, and flags
  F-ch03-2/-10/-12 (append the correction, do not rewrite history).
- ch03 §1's RATIFIED box states n_σ = 1.5 **and adds one sentence** naming the
  signature-default trap (that is how the error happened; saying so prevents recurrence).
- **Dependency:** ch02's BLOCKER-1 fix and any page quoting census figures (ch04 §5,
  ch05 §4 framing, ch06 cross-refs) must consume the **regenerated** numbers, not the
  2σ literals quoted in the reviews. ch03 runs first (§E wave 0).

### D3 — Cell B landed: the 2×2 is filled, C6 is RESOLVED, the arc becomes pre-registration → control → scored readout `[mandate, expB]`

`CELLB_READOUT_20260731.md` (jobs **6103219/6103220**, resubmission of 6101146/6101147
after a pure-plumbing symlink failure; code `7fd60bb`, same as cells A and C):

|             | point / generator_marginal | volume_deconv / absolute_marginal |
|-------------|---------------------------|-----------------------------------|
| unscattered | A = #51: 1D 0.7299, 2D 0.7300 | **B: 1D MAP 0.7450 (mean 0.7320, σ 0.026), 2D MAP 0.7900 (mean 0.7962, σ 0.019, edge/peak 1.2e-2)** |
| scattered   | forbidden by guard        | C = #53 r1: 1D MAP 0.7400 (mean 0.7321), 2D 0.8133 |

**B − A (estimator) = +0.0151 / +0.0600 · C − B (scatter) = −0.0050 / +0.0233 ·
C − A (total, r1) = +0.0101 / +0.0833.** The estimator owns **72% of the 2D**
displacement (2D-only figure — never print it as a both-channel summary, see MJ-1).
Catalogue-leg rail 90.7% (B) vs 89.2% (C); combined 1D in-cat argmax at 0.86:
69.7% vs 57.9% vs 5.3% (#51). w_G(h) bit-identical to #53 across all 41 grid points.
Dark channel difference +18.00 nats unscattered (vs +15.83 r1). C6 → **RESOLVED
2026-07-31: THE ESTIMATOR OWNS IT**; the realism layer (#53 scatter) largely exonerated;
C9's cell-B gate released (C9 stays live, fix author-gated, joint C9+C8 only); the C7
fix must supersede G2b and must not be the exonerated "p_det inside the numerator
alone" form.

**Honest scoring is part of the arc** (expB BL-4): the registered prediction had three
numeric reads — 2D ∈ [0.78, 0.82] → 0.7900 ✓; in-cat class argmax ≈ 0.86 → 0.860 ✓;
**1D ∈ [0.70, 0.74] → 0.7450 ✗ by one grid step** (the 1D *mean* 0.7320 is inside the
band, but the band was written in MAPs). The book scores 2/3 + one one-grid-step miss
and says why that makes the pre-registration stronger, and explicitly does **not**
copy the readout's "confirmed on every pre-registered read" sentence. Editing rules:

- Pre-registration blocks are kept **verbatim** — never edited into hindsight. Results
  are appended as visually distinct, dated readout blocks.
- Where the *pre-registration* is quoted, job IDs stay 6101146/6101147; where the
  *result* is reported, cite 6103219/6103220 with the one-sentence resubmission note.
- Answers that were correct at the pre-cell-B state (Q10.5, Q11.6) are kept and get
  dated postscripts — history, not overwrite.
- Naming: the 2×2 object is called **"the 2×2 cell B"** everywhere; museum ledger #88's
  "Cell B" (seed1000 deep venue, = BIAS_HISTORY_LEDGER §3's A′) gets a disambiguation
  clause (MJ-3).
- Canonical rail pip, identical wording on ch07/ch09/ch10/ch11 (integrator supplies,
  agents place): `cell B (2026-07-31): estimator owns +0.060 of the 2D +0.083`.

Affected: ch07, ch08, ch09, ch10, ch11, museum, index (+ data files
`ch07_c7.json`, `ch11_board.json`, `museum_ledger.json`).

### D4 — Spoiler discipline (forward references and chrome) `[ped, mara, tomas]`

A forward reference names the *phenomenon* and the chapter, never the number or the
verdict (Ch 2's "note it and move on" is the canonical template). Applied to: chapter
decks (ch07/ch08/ch10), index journey blurbs, Traps 2.A/2.B/5.B, ch05 §3's heading,
and the Symbol Passport (chapter-gated notes, §D-INT). Numbers that are a chapter's
own reveal (+0.077, 0.256, "estimand-dependent", #49a's verdict, z = −11.86) appear
**at or after** their home chapter only.

### D5 — Both-values policy for the still-open disputes (everything except σ_dL)

D1 resolves item 1 of BUILD_REPORT §5.1 only. Items 2–6 (C7 threshold 0.256 vs 0.2644,
C5 leverage 1500–2400× vs ~197×, C11 endpoints, the 4π "5000×", C9 ×2.19 vs ×2.446)
and the σ_Mz 10⁻⁴ vs 8.8×10⁻⁸ pair **remain both-values items**: any page quoting one
half must carry the other with the flag pointer. The reviews found pages quietly
preferring one half (ch05 Q5.4, ch08 ×5, ch11 opening table); those are fixed below.

---

## B. Conflicts between reviewers — adjudicated

1. **σ_dL handling — everyone vs the mandate.** All five reviewers who hit it
   (mara BLOCKER-3, tomas B2, expA B2, expB BL-5, ped B2) proposed "print both values
   everywhere," which was the correct fix under the build contract at review time.
   **The author's mandate supersedes it**: adopt 8.98×10⁻⁴, single corrected value +
   erratum note (D1). The reviewers' page inventories (which dossiers carry the bad
   string) remain the authoritative fix-site list. Pedagogy B2's *mechanism* (one
   shared canonical dossier string) is adopted as the implementation.
2. **Ball multiplier — expA/tomas vs mara.** Expert A and Tomas win, verified in-tree
   by this synthesis: `:2823`'s 2.0 parameterizes `get_redshift_outer_bounds` (z-window
   cut), not the ball search; there is exactly one production ball-search call site
   (`:2838`, 1.5). Mara's fix option 2 ("both genuinely live in different legs → print
   889's ball both ways") is **rejected**; her option 1 (regenerate at production
   value) is what D2 does, with a one-sentence signature-default warning.
3. **ch02 census replacement text — mara's literals vs D2.** Mara's suggested Q2.5/
   §-text rewrites embed the 2σ census numbers (1616 / 12 / 4891 / 552). Her *shape*
   (the answer is a distribution spanning orders of magnitude; chip to ch03) is
   adopted; every literal is replaced by the regenerated 1.5σ value. ch02's fix is
   sequenced after ch03's regen.
4. **BW3 "Has this been tried?" — ped B4 vs the integrator's deliberate choice**
   (BUILD_REPORT §2: no per-widget auto-reveal, to avoid double-reporting and
   predict-lock pre-emption). Adjudication: both are right at their scope. The two
   **false advertising claims** (museum §7, index) are corrected now [P0]; the
   scoped inline-chip mechanism (render `⚖ #N — verdict` inside a widget when a
   `data-hypothesis`-tagged control becomes active, with `data-hypothesis-verdict="inline"`
   opt-out for widgets that already hard-code the verdict) is implemented [P1] — it
   honors the integrator's double-report concern while delivering the advertised
   behavior; ch08–ch11 agents add the missing tags.
5. **Q11.6's model answer — BUILD_REPORT gap #4 vs tomas P9 / ped P10.** Both
   reviewers who examined it independently say keep it (the discipline answer is
   genuinely known; the five unanswerable questions sit below with no key). **Kept**,
   with expB BL-7's dated postscript (the control was run; a confirmed prediction is
   the weakest kind of confirmation — which is why the fix stays author-gated).
6. **Museum backlinks — BUILD_REPORT "acceptable" vs tomas m5 / ped m6.** Reviewers
   win: both independent walk-throughs got lost by Exhibit 8. One "referenced by
   Ch N §… · ← back" line per exhibit [P2].
7. **"Confirmed on every pre-registered read" — the readout artifact vs expB BL-4.**
   ExpB wins for the book: the book scores the 1D band miss explicitly (D3). The
   artifact itself is out of book scope → author list (§F-3).
8. **σ_Mz 10⁻⁴ — correction vs both-values.** Tomas B3's own adjudication is adopted:
   this is a **both-values case** (the claim file really does say 1e-4 at
   `CLAIM_2D_BIAS_20260730.md:172`), not a book-side correction — unlike σ_dL there is
   no author mandate. Ch08/ch01/ch07 carry both with the F-ch06-5 pointer. Whether the
   claim file itself gets amended is the author's (§F-2).
9. **2×2 table convention — expB MJ-1.** Uncontested but load-bearing: MAPs
   throughout (A 0.7299/0.7300 · B 0.7450/0.7900 · C 0.7400/0.8133), means in a
   footnote, and the explicit note that the 1D estimator share exceeds 100% of the 1D
   total because scatter pushes the other way — "72%" is 2D-only.

---

## C. Per-chapter worklists

Every item: `[priority] [source] what-to-change — **AC:** acceptance criterion.`

### ch00 — two numbers

- [P1] [ped-M5] Trim to budget: move §2's step-by-step σ_tot algebra into a
  `details.num-view` (I0.1 recomputes it live); keep the two figures, the arbitration
  budget, the contract, and the time-delay 2%→8% paragraph. — **AC:** main-column word
  count ≤ ~1.7× the 1,200 budget (from 2.33×); no argument lost, only relocated.
- [P1] [tomas-M3.1] Add the LVK dark-siren state of the art to the third-methods
  figure: GWTC-3 dark-siren H₀ = 68⁺⁸₋₆ (Abbott et al. 2023, arXiv:2111.03604), one
  row, gen_ch00. — **AC:** a reader finishing ch00 knows what the book's own genre
  currently achieves; provenance chip on the row.
- [P2] [ped-M9] Trap 0.A: strip #49a's verdict/number if present; phenomenon + `⏭ Ch 10`
  only (D4). — **AC:** no #49a verdict text before ch10.
- [P2] [expA-m6] "Most generous case imaginable" wording: name the anchor the
  hypothetical method parks on (Planck, σ_A = 1.04 → cap 1.566) or say "the more
  demanding of the two placements" (SH0ES branch caps at 1.812). — **AC:** prose and
  arithmetic name the same branch; downstream numbers unchanged (they reproduce).
- [P2] [mara-MINOR-5] Q0.3 is a giveaway (answered four times on-page). Re-aim one step
  past the body or accept as warm-up — agent's call, low stakes. — **AC:** explicit
  decision recorded in the flag file.

### ch01 — the ruler

- [P0] [mandate-D1] Apply D1: F1's OPEN arbitration block → resolved-erratum block
  (keep the 1/ρ discriminating check as the proof); Q1.2's answer rewritten as the
  erratum lesson (the order-of-magnitude check *caught* the slip — keep ped-P3's beat,
  now with a resolution); dossier row → canonical string. — **AC:** D1 grep gate
  passes on ch01; Q1.2's answer no longer says "0.008%" as a live reading; the lesson
  sentence survives.
- [P1] [tomas-B3] `ch01-ruler.html:290`'s σ_Mz ≈ 10⁻⁴: add the measured counterpart
  (median 8.8×10⁻⁸; 889: 1.36×10⁻⁹) with the F-ch06-5 pointer (D5). — **AC:** both
  values + flag chip on the page.
- [P2] [tomas-m4] Dossier mass row: `M` → `M_z` (7.246×10⁵ M☉ is the detector-frame
  CRB column; ch08 establishes M_z = M(1+z) is the measured quantity). — **AC:** label
  matches ch08's dossier; F-ch08-8 appended.
- [P2] [mara-MINOR-4] Q1.3 is only answerable from the collapsed events sidebar: fold
  its two-sentence core ("SNR accumulates near plunge; the rate model is a plunge
  rate") into §2 narrator flow; history stays in the sidebar. — **AC:** Q1.3 answerable
  from the default-persona read path.
- [P2] [mara-MINOR-6] One narrator sentence after the standard-siren equation: this is
  the circular-quadrupole form, printed to show where d_L enters; the real EMRI
  waveform is numerical and richer. — **AC:** sentence present outside any `<details>`.
- [P2] [ped-m2] Q1.4's transfer chain: add the one clause routing through Ch 2
  ("…a machine that turns 1,588 distances into a statement about h"). — **AC:** chain
  Q1.4 → Ch 2 unbroken.
- [P2] [expA-m1] Re-grep the `IDEALIZED_BASELINE_READOUT.md` anchors ch01 cites
  (`:42-47` → current lines). — **AC:** cited lines hold the cited text.

### ch02 — Bayes  *(sequenced after ch03 regen)*

- [P0] [mara-BLOCKER-1] "Tens of thousands of candidates" ×3 (`:373`, `:829`, Q2.5
  answer `:918`): amend all three to the distribution truth using **regenerated**
  census numbers; Q2.5's answer becomes "the answer is a distribution — Chapter 3
  measures it," with the new median/p95/zero-candidate figures and a ch03 chip.
  — **AC:** no ch02 assertion contradicts ch03's census; the graded answer matches
  the measured numbers; F-ch02 entry appended.
- [P1] [mara-MAJOR-3] Grade the central predict (`:598-665`): reveal opens with the
  correct option bald ("**(d) — a handful.** Measured: 24 of 1588."), plus one clause
  per wrong option including "~100" (participation-ratio argument, number already
  on-page). — **AC:** every option has a graded response; consistent with the ch00/
  ch04/ch05 reveal style.
- [P1] [expA-M5] `:§4` "62%" → **52%** for r1's pair (0.0851/0.1650), with "62%
  averaged over the ten runs" as the readout's ensemble figure. — **AC:** the printed
  ratio equals the division of the two numbers beside it.
- [P1] [ped-M6] Rung violation: §1's "no extra β(h)^N" derivation box (D(h), ledger
  #20, Mandel/Farr/Gair) → one forward-reference sentence in the column + move the box
  into `details.gw-reader` or to ch04 §4 ("Counted exactly once" — coordinate with
  ch04 agent; preferred destination ch04). — **AC:** no L4 tool used in ch02's main
  column; the theorem sentence + `⏭ Ch 4` chip remain.
- [P1] [ped-M9/B1] Trap 2.A: strip #49a's full verdict → phenomenon + `⏭ Ch 10`.
  Trap 2.B: keep the √N mechanism, drop +4.04 / +0.077 / "Chapter 8's cold open"
  signpost → bare `⏭ Ch 8`; relabel the I2.2 preset button so it doesn't name the
  recorded 2D offset's value. — **AC:** ch08/ch10 reveals are no longer pre-announced
  with numbers from ch02.
- [P2] [expA-m5] §4 concentration figures: state the denominator (in-cat vs signed
  total) per figure, or quote one set throughout. — **AC:** every percentage names its
  denominator.
- [P2] [ped-M1] Q2.3 (verbatim recall): re-aim — "construct an estimator that fails
  exactly one of biased/noisy/mis-calibrated and passes the other two, using I2.2's
  dials." — **AC:** answer no longer 5-gram-recoverable from the defbox.
- [P2] [mandate-D1] Dossier row + F-ch02-1 → canonical string + erratum note.
- [P2] [ux] Lazy-init the 7 Plotly instances behind ch03's IntersectionObserver
  pattern (or `Book.lazyPlot` if the integrator ships it). — **AC:** no plot
  constructs before its container approaches the viewport.

### ch03 — which galaxy  *(wave 0 — everything census-dependent waits for this)*

- [P0] [expA-B1, tomas-B1, mara-BLOCKER-2, mandate-D2] Apply D2: regenerate at
  n_σ = 1.5; fix the RATIFIED box, `ball_rule` meta, census prose/widget caption,
  Q3.4, the featured extreme, the dossier ball facts, and append corrections to
  F-ch03-2/-10/-12. Add the signature-default-trap sentence pinned to
  `bayesian_statistics.py:2838`. — **AC:** ch03 and ch06 print the same radius
  (0.757′), solid angle, and ball population (2) for EMRI-889; the words "production
  ball rule" are true; a cross-chapter flag records the incident.
- [P1] [expA-M6] The ±0.155 / 236× / ±0.0089 σ_z-derived numbers: add the sources-map
  §7.19(d) staleness caveat to the venue box (local `z_error` ≠ #53 parent; cluster
  parent for width-sensitive work), chip the three numbers as parent-dependent, open
  F-ch03-13. Note: cell B ran against the true parent — the qualitative claim
  (photo-z dominates by ~2.4 orders) is safe; only digits are indicative. — **AC:**
  no width-sensitive number on the page without the parent-column caveat.
- [P1] [mara-BLOCKER-1 tail] Q3.4's own answer must not open "Thousands to tens of
  thousands" and then measure otherwise in the same paragraph — rewrite from the
  regenerated census. — **AC:** answer internally consistent.
- [P2] [mandate-D1] Dossier row → canonical string + erratum note (ch03 currently
  prints both; collapse to D1 form).
- [P2] [mara-MINOR-7] `Sig` passport gloss is circular before Ch 6 (needs the
  integrator's `firstChapter` gating; ch03 side: ensure the §1/§4 uses read naturally
  with the rung-safe gloss "the measurement's covariance: how big the error ellipsoid
  is and which way it tilts"). — **AC:** hovering Σ in ch03 defines it without Γ.
- [P3] [mara-MINOR-8] Stale `handler.py:519` chip: add current-line tooltip per the
  §5.5-23 drift table (integrator policy item; ch03 applies it here).

### ch04 — the loud half

- [P0] [mandate-D1] Dossier `:627` → canonical string + erratum note; gen_ch04 JSON
  key rename (`sigma_dL_Gpc` + corrected `sigma_dL_over_dL`). — **AC:** D1 grep gate
  passes; JSON key name matches its content.
- [P1] [tomas-M2] State what p_det is a function of: two sentences in §2 + `⏭ Ch 9`
  chip — p_det(d_L) is marginalized over the injection population's intrinsics; exact
  only if applied events are drawn from that population; Ch 9 measures a case where
  they are not (C9's seed). — **AC:** the words "marginal"/"intrinsic" appear in §2;
  the LVK-reader gap is closed before C9 is met.
- [P1] [mara-MAJOR-5] Q4.3/Q4.4 answers use machinery the chapter never introduces:
  Q4.3's ESS clause → a mechanism the chapter owns (support mismatch / ledger #8
  extrapolation example); Q4.4 ends at "…is the selection correction," with the
  G1/Σ_glob sentence moved to a `⏭ Ch 9` chip. — **AC:** both answers derivable from
  ch00–ch04 content only.
- [P1] [ped-M6 receiver] Receive ch02's β(h)^N/ledger-#20 box into §4 "Counted exactly
  once" (or its gw-reader fold) if the ch02 agent routes it here. — **AC:** the
  double-count history lives at the rung that owns D(h).
- [P2] [mara-MINOR-3] Guess-marker desync: disable the slider after lock (ch00's
  `setLocked` pattern) or re-apply on input when locked; prefer converging on
  `Book.predictValue`. — **AC:** readout, stored value, and drawn marker cannot
  disagree after locking.
- [P2] [synth] §5's zero-candidate framing: update the 552 figure to the regenerated
  value (D2 dependency) while keeping the "reconstruction count, never a drop count"
  discipline (expA-P7). — **AC:** number matches ch03's regenerated census.

### ch05 — the unseen galaxy

- [P0] [mandate-D1] Dossier `:863` → canonical string + erratum note. — **AC:** D1
  grep gate passes.
- [P1] [mara-MAJOR-1] w_G's type before its value: move (not copy) the
  line-of-sight-average sentence from the GW-reader fold into the narrator flow
  immediately before "First: 12%" (mara's proposed wording is good). Also add the
  one-clause sample note distinguishing 76/1588 = 4.8% (one seed) from
  164/3135 = 5.23% (two seeds) where both appear. — **AC:** a default-persona reader
  can state what w_G *is* (population-level, detection- and volume-weighted, one
  number per h) before hitting the 12%-vs-5% shock; the two realized-rate figures are
  each labelled with their sample.
- [P1] [expA-M1] Fix C10's attribution in the adjudicator block + provenance panel:
  dark ΣΔln L_comp = **−22.72** (all-event −3.11); **27.7%** of dark events positive
  on L_comp alone, **39.1%** with the (1−w_G) prefactor — the form C10 quotes.
  — **AC:** numbers match F-ch08-6's scoping and expA's recomputation; the block that
  enforces C10's rule obeys it.
- [P1] [expA-M2, mara-MAJOR-6] Q5.4's "1500–2400×" alone: apply D5 — either quote the
  0.025 Poisson figure instead (reproduces to 1e-6), or add the recomputed
  142–2458× / median 197× + F-ch11-1 pointer. — **AC:** no lone half of the flagged
  pair on the page.
- [P1] [mara-MAJOR-7] I5.1's unnarrated κ midrange: add a third `V51` state for
  1 < κ ≲ 200 (in-catalogue class taking over → 0.86 plateau; push further → dark
  zero-legs reassert → wall), plus one prose sentence for static readers. — **AC:**
  every reachable dial regime has a narrated verdict; the 0.86 plateau is explained
  on-page.
- [P1] [mara-MAJOR-4] De-spoil §3: heading loses "factor of 5000"; first paragraph's
  "several thousand" → "orders of magnitude"; the 5000× stays in the reveal (D4).
  — **AC:** the predict question is answerable wrongly.
- [P2] [ped-M3] Relocate Trap 5.B to §2 (after "12%. Not 5%"), Trap 5.A to §3 (where
  L^comp is introduced); per ped-M9 strip the binomial z / 2.3–2.5× from 5.B (Ch 9's
  measurement). — **AC:** traps fire where the misconception forms; no C9 numbers
  before ch09.
- [P2] [ped-M1] Q5.3/Q5.4 overlap: re-aim per ped's option (b) once the leverage fix
  above lands. — **AC:** <22% 5-gram overlap or an explicit keep decision in the flag
  file.
- [P2] [synth] §4's zero-candidate framing: consume regenerated census numbers (D2).

### ch06 — the black box

- [P0] [mandate-D1] §4 flag box → erratum form; Q6.5's "6000×" → measured ≈550×; Q6.5
  stem keeps the units-lesson with tomas-m3's dagger ("† the units in this question
  are the point — see the answer"); dossier row → canonical string. — **AC:** D1 grep
  gate passes; the arbitration checks (σ_u·ρ ≈ 1.04; 0.38%/√76) survive as the proof.
- [P1] [tomas-M8] Add the 14×14 condition-number distribution to §3 (gen_ch06
  computes; expB/tomas measured median 2.6×10⁹ / p95 1.4×10¹⁰ / max 3.9×10¹² as the
  expected values — generator must recompute, not copy) and the Babak et al. 2017
  plausibility clause on §4.1's σ_Mz (10–100× better than published forecasts; honest
  "not tested here" verdict). — **AC:** the gate's actual operand (14×14) is shown;
  the chapter's most extreme number carries the chapter's own discipline.
- [P1] [synth-D2] ch06 is the correct side of the ball dispute — add one cross-ref
  sentence to the four-event table noting ch03's census now measures the same rule
  (closes the loop for readers who hit the old contradiction). — **AC:** 889's ball
  facts identical in ch03/ch06.
- [P2] [ped-M5] Move §4.1 ("the fourth coordinate") and §5's matrix bookkeeping into
  `details.gw-reader`; keep §5's story (0.860 → 0.730, host recovery 31→38) in the
  column. — **AC:** main column ≤ ~1.7× budget (from 2.37×); ch05→ch07 pause no longer
  a stall.
- [P2] [ped-M3] Relocate Trap 6.A → §4/§5, Trap 6.B → §2 (dt²). — **AC:** in-line.
- [P2] [tomas-B3 site] Nothing to fix for σ_Mz here (ch06 is the measuring side);
  ensure F-ch06-5's pointer text still matches after ch08's edit.
- [P2] [expA-m1] Re-grep `IDEALIZED_BASELINE_READOUT.md` anchors (`:47-48`, `:50-52`).

### ch07 — redshift  *(cell-B set)*

- [P0] [expB-BL-3] §6 "Both sides, and who decides": replace "It has not landed." with
  the dated landed-block (90.7% vs 89.2% vs 5.3%; class argmax 0.860 as registered;
  scatter damps the combined rail 69.7%→57.9%), followed by the scope constraint: cell
  B settles C7's **magnitude and attribution**, not the G2b collision; the fix must
  supersede G2b and must not be the exonerated "p_det inside the numerator alone"
  form. Same update at: provenance `:1194` OPEN→dated FINDING; closing `:1155`;
  Trap 7.B ("…pre-registered to decide — and did, 2026-07-31"). Include the honest
  staleness nuance: stale column predicted 98.7% (different statistic), staleness-free
  delivered 90.7% — "resolves in the confirming direction, somewhat weaker," never
  "98.7% confirmed." Keep the registered readings verbatim above the block.
  — **AC:** zero "not landed" assertions remain; P1-praised structure (both boxes,
  binding rule, decider) untouched; the G2b constraint sentence present.
- [P0] [expB-MN-1/MN-2] `ch07_c7.json`: `conflict.decider` → landed values;
  `hosts` gains `resolved_by_cellB {date, lcat_rail_frac 0.907, n "68/75",
  comparison_scattered 0.892}`; noscript fallback `:597-603` gains the same numbers
  (static readers currently get the stale prediction with no resolution). — **AC:**
  page, data file, and noscript agree.
- [P1] [tomas-M4 ch07 side] Define φ_cat at first use in §6 (selected number density,
  `f(z,Ω)·(dV_c/dz)/(1+z)`); keep Trap 7.B's two-priors sentence (its promotion target
  is ch11 §4); one clause noting standard-practice kernels (Laghi/Turski/gwcosmo) do
  not deconvolve at all — which is what makes D8 the thing under test. Does **not**
  resolve C7. — **AC:** φ_cat defined; the axis named; no adjudication.
- [P1] [expB-MN-5] Rail pip → canonical cell-B pip (integrator wording). [expB-MN-6]
  Ledger-#88 citation in §5: one scoping sentence (per-leg vs whole-estimator, deep vs
  campaign venue) so 86.7% and 72% don't read as a disagreement; use "the 2×2 cell B"
  naming (MJ-3). — **AC:** both decompositions scoped.
- [P1] [ped-M2/M1] Q7.1 (verbatim re-ask of the cold-open predict): convert to the
  transfer form (what sets the sign; what would make symmetric widening right).
  Q7.5's 62% overlap: fold the answering passage or re-aim. — **AC:** no "predict"
  framing after the reveal; overlap <22% or keep-decision recorded.
- [P2] [ped-M3] Relocate Trap 7.A → after I7.1 stage-1 reveal in §2; Trap 7.B →
  immediately before §6's twist. — **AC:** the chapter-defining trap fires when the
  misconception forms (cold open), not forty minutes later.
- [P2] [ped-B1 tail] Deck "does not blur the answer — it moves it" collapses the
  3-way predict → soften to "…and the central value is not the safe choice" (D4).
- [P2] [mandate-D1] Dossier/provenance FLAG-1 → erratum form (ch07 already prints
  8.98×10⁻⁴; reframe from open dispute to resolved erratum).
- [P2] [tomas-B3 site] Q7.6 σ_Mz: both values + F-ch06-5 pointer (D5).

### ch08 — the mass channel  *(cell-B set + spoiler inversion)*

- [P0] [tomas-B3] σ_Mz ≈ 10⁻⁴ at five sites (deck-adjacent claim, §1 RATIFIED display
  equation, §2 ×2, Q8.1): both-values treatment per D5 — keep the claim-file value
  chipped `CLAIM C4`, add the measured 8.8×10⁻⁸ (889: 1.36×10⁻⁹) with F-ch06-5
  beside it, note the direction of the argument survives (10⁻⁷ ≪ 1.28 too). No silent
  substitution. — **AC:** no lone 10⁻⁴ on the page; the RATIFIED box carries the
  flag.
- [P0] [ped-B1, tomas-M10] De-spoil the chapter: deck → "The mass channel should
  sharpen everything. Watch what it does instead."; +0.077 leaves the deck and every
  pre-reveal position; prefer ch04's inversion pattern if the agent restructures the
  beat. (Index blurb: integrator.) — **AC:** an honest wrong prediction is possible at
  `#ch08-predict-1`.
- [P1] [expB-MJ-4] Cell B into §4/§5: the +18.00-nats-unscattered block (channel
  difference exists with zero realized noise — estimator-borne, closing the
  scatter-induced question); C4 partition configuration-scoped ("on r1: 534 survivors /
  98.5%, 487 zeroed / 1.5%; on cell B: 219 / 80.7%, 688 / 19.2% — deletion is a
  minority carrier in both; the precise share is not a constant of the mechanism"),
  **marked as recomputed-for-the-book** (reviewer-computed, not in any adjudicated
  artifact) and raised in ch08_FLAGS for the author (§F-4); §4 closing pointer →
  "…was confounded — until the 2×2's cell B separated them ⏭ Ch 11 §5". — **AC:**
  98.5% never printed as the mechanism's signature; the reviewer-computed numbers
  carry their provenance marker.
- [P1] [tomas-M7, ped-M2] Q8.1: replace the wrong mechanism (M_z ≥ M vacuity) with
  §2's correct one (σ_M ≳ M_g ⇒ lower edge negative ⇒ upper leg vacuous), and
  convert from a re-ask to the transfer form ("name a source population for which the
  upper leg would not be vacuous; does the tilt survive?"). One rewrite serving both.
  — **AC:** answer's mechanism matches §2; not answerable by scrolling up.
- [P1] [tomas-M3.5] Spectral-siren disambiguation sidebar in §1: this is per-host
  BH-mass *association* via a catalogue proxy, not the population-mass-function
  method (Farr / Mastrogiovanni / Ezquiaga–Holz); failure modes differ. — **AC:** the
  predictable misreading is blocked at the point it forms.
- [P2] [ped-M3] Traps 8.A/8.B → the sections that create them. [ped-m8] If the
  duplicated completion-trap is deliberate (C10), note that in ch11, not here.
- [P2] [mandate-D1] Dossier `:801` → canonical string + erratum note.
- [P2] [ux] Lazy-init the 8 Plotly instances (ch03 pattern / `Book.lazyPlot`).
- [P2] [synth] Rail pip conversion to `Book.biasRail` is the integrator's (tomas-m2);
  agent only verifies no page-local pip CSS remains.

### ch09 — the universe factory  *(cell-B set)*

- [P0] [expB-MJ-5] Re-litigation guard `:898-906` + verdict string `:1300`: C9 stays
  **live**, cell-B gate **released 2026-07-31**; remaining gate = author's
  leg-adjudication + `/physics-change`; the fix must be the joint C9+C8
  mass-consistent mixture (counterfactuals act on different terms, not additive).
  Provenance OPEN → dated FINDING; rail pip → canonical cell-B pip. — **AC:** no
  "gated on cell B" as a live blocker anywhere on the page.
- [P1] [expB-MJ-5 win] §6: add the bit-identical w_G payoff — registered "expected
  bit-identical," delivered max|Δ| = 0.0 across all 41 grid points
  (0.1625175 / 0.1215039 / 0.1038732 at h = 0.60/0.73/0.81), dated. — **AC:** the
  book's cleanest pre-registration hit is stated as such.
- [P1] [tomas-M5, ped-M8] §4 symbol load: three-sentence physical picture before the
  Option-A box (catalogue = list, integrals = volumes, exchange rate that cancels);
  `n_w` gets units + one sentence before first use; `Σ_global` defined in words
  first; fold the mode-by-mode bookkeeping (n̄_w, D_gen, Ŵ_cat, V_f) into
  `details.gw-reader`. Passport keys (`nw`, `Sglob`, `Wcat`, `Vf`, `Fincat`) are the
  integrator's; agent tags the terms (+ tag `Lcat`/`Lcomp`/`Ng`/`Dg` where used, and
  Σ_glob's leaks in ch07 §6 / ch11 C8 — coordinate). — **AC:** every §4 symbol has a
  picture-first introduction or lives in a fold; hovers resolve.
- [P1] [tomas-M6] §5's two "residuals": rename ("the shape residual" −17.2% vs "the 1D
  bias residual" +1.667%), one clause stating they are different objects; drop or
  re-scope the `(0.73/0.81)³ = −26.80%` parenthetical (states a different comparison
  than the ×2.48 drift beside it). — **AC:** no two quantities share the name
  "residual" on the page.
- [P1] [tomas-M9, M3.2] §3's `global`-mode deprecation: name the venue, name the
  co-varying deviations present (±3σ ball, volume_deconv numerator, D2 smooth
  completeness), state whether the mode was ever run in a configuration matching the
  published machinery (if not, say that); name gwcosmo as the reference implementation
  of the literal A10. — **AC:** a reader can tell whether the pathology indicts the
  published method or this pipeline's configuration of it.
- [P2] [ped-m1] Q9.1 (Ω_m recall): ask the harder version ("name a quantity for which
  the same argument would NOT license a non-Planck value"). — **AC:** not answerable
  from ch01 §4 alone.
- [P2] [tomas-m7] Dossier row: lead with 889's own w_G under both estimands (already
  on the row) rather than "not special here." — **AC:** the beat lands like the other
  chapters'.
- [P2] [ux] Lazy-init the 8 Plotly instances. [P2] [expB-MJ-3] "the 2×2 cell B" naming.

### ch10 — calibration  *(cell-B set)*

- [P0] [expB-BL-7] §5: keep the design lesson intact; last two sentences of `:873-881`
  → "submitted, and landed 2026-07-31 — after the pre-registration… Chapter 11 shows
  you the prediction from the inside first, and then what it got right and what it
  missed." Callout `:883-891` → "…confounded until the control was run." Provenance
  `:1069` → dated FINDING; rail pip → canonical. **Q10.5:** keep the answer (correct
  at the state described) + dated postscript (2D 0.7300→0.7900, 72%; "what could you
  conclude before the control" is still nothing — the transferable half). — **AC:**
  no unresolved-forward-promise to ch11; Q10.5's original answer intact above its
  postscript.
- [P1] [tomas-M10] De-spoil the deck: move "That is not the same as being right" below
  the predict box (D4). — **AC:** the Yes/No/can't-tell predict is honest.
- [P1] [tomas-M3.4] §3 scenario-table caption: one external anchor — Laghi et al. 2021
  (already cited in the book) forecasts ~1% for LISA EMRI dark sirens; row D (3.6
  km/s/Mpc ≈ 5%) sits against it. — **AC:** the reader has an external scale for the
  table.
- [P2] [ped-m5] `num-view` for I10.2 (argued from a number the reader cannot see
  tabulated). — **AC:** table view present.
- [P2] [mandate-D1] Dossier `:735` → canonical string + erratum note.
- [P2] [expB-MJ-2] Job IDs at `:1150` per D3's rule (result → 6103219/6103220).

### ch11 — the honest state  *(cell-B centerpiece; largest agent)*

Work in expB's order: BL-1 + BL-2 + BL-4 as ONE editing pass over §5 + the closing
block, then BL-6 (board), then the rest — the page must read as a single arc:
confound → registered prediction → control → two hits and a one-grid-step miss.

- [P0] [expB-BL-2, BL-4, MJ-1] §5: retitle "The confound, and the control that
  resolved it"; keep the pre-registration block verbatim; append the distinct dated
  readout block with the filled 2×2 **in MAPs throughout** (A 0.7299/0.7300 ·
  B 0.7450/0.7900 · C r1 0.7400/0.8133), means as a footnote, B−A / C−B / C−A rows,
  the 72%-is-2D-only warning, and the note that the 1D estimator share exceeds 100% of
  the 1D total because scatter pushes the other way. Score the registered prediction
  explicitly: 2D ✓ · in-cat argmax ✓ · **1D ✗ by one grid step** (mean inside band;
  band written in MAPs) — plus the sentence that the readout's "confirmed on every
  read" is the one sentence this book should not copy. Badge OPEN → RESOLVED
  2026-07-31 (or paired badges); chips → CELLB_READOUT + C6-as-amended. — **AC:**
  a reader can recompute every cell and both difference rows from the table; the miss
  is scored, not softened.
- [P0] [expB-BL-1] Closing block: item 1 leaves the no-answer list via a dated
  resolution card; the question text stays visible (struck/"what this chapter asked");
  renumber to four; closing line → "Four of the five… The fifth was answered by
  running the control — which is the only reason it is no longer on this list."
  — **AC:** the no-answer contract is true again; the arc question→prediction→answer
  is on the page.
- [P0] [expB-BL-6, MJ-7] Regenerate `ch11_board.json` (gen_ch11): C6 status = the
  amended heading verbatim, badge/tag/live/adjudication/refute_by per BL-6's spec;
  C9 gate-released wording; C7 adjudication appended (cell-B magnitude check + G2b
  supersede + no-p_det-in-numerator form); live count 5 → **4** (C5, C7, C8, C9) in
  the JSON, the widget count line, and the noscript at `:297`. — **AC:** "verbatim
  from the claim file" is true against the file's current version; all three count
  surfaces agree on four.
- [P1] [expB-MJ-6] Opening table 2D pull row → `+2.47 … +4.74 (mean +4.04)` with the
  F-ch08-1 footnote (readout prints +3.4…+4.5; mean and 10/10 reproduce). — **AC:**
  no lone half of a flagged pair in the most quotable table.
- [P1] [tomas-M4] §4's C7 block: promote Trap 7.B's two-priors framing (prior over
  *where an EMRI host is* vs prior over *a row in a flux-limited catalogue*,
  φ_cat ∝ f·(dV_c/dz)/(1+z)); name the axis, adjudicate nothing. — **AC:** the
  collision is framed as two priors over two random variables, not a σ_z scale
  artefact.
- [P1] [ped-B3, mara-MINOR-1/2] The recall beat: map slugs to human labels
  (`impostor → "A, the spectroscopic galaxy"` etc.), rewrite for the delivered
  three-button interaction, and do not let the slug state the verdict; update the
  stale R-ch11-2 comment. — **AC:** the reader sees their own choice's label; the
  verdict lands after, in prose.
- [P1] [expB-MJ-2, MJ-3, MN-3, MN-4] Job-ID split per D3; "the 2×2 cell B" naming +
  ledger-#88/A′ clause at `:991-1001`; `<meta description>` + subtitle → "…the control
  that landed the night the book was built"; §7 closing → expB MN-4's ending
  ("…landed: the estimator configuration owns 72% of the 2D displacement… What to do
  about it is still a position."). — **AC:** zero "in flight"/"still running" strings
  on the page.
- [P1] [expB-BL-7 tail] Q11.6: keep the discipline answer; add the dated postscript
  (control run; a confirmed prediction is the weakest confirmation — why the fix
  program stays author-gated). — **AC:** history preserved, state current.
- [P2] [tomas-m1] `:1154` "3130 detections" → **3135** (1590 + 1545).
- [P2] [ped-M1/M5] The 5-of-6 recall answers + §4 recitation: collapse §4's C7/C8/C9
  detail to one-line status + folded detail (Ch 7/8/9 argued each), which also fixes
  most of the overlap; re-aim Q11.2 per ped's construct-a-statistic form if time
  allows. — **AC:** §4 no longer duplicates §1's scoreboard in the main column.
- [P2] [mandate-D1] Dossier `:1018` → canonical string + erratum note (the *closing*
  dossier — highest-visibility D1 site). [ped-m5] num-view for I11.1. [ped-m8] If the
  twice-stated completion trap is deliberate, say so ("the one trap the book states
  twice, because C10 exists to retire it").

### museum

- [P0] [expA-M3] `gen_museum.py:286` separator class → `[,/·;]` (or any non-`#`
  punctuation): recovers #41/#52/#43/#44 into `do_not_retry_rows` (26 → 30); update
  the three printed counts (census caption `:189`, noscript `:1290`, M.4's answer
  `:1397`). — **AC:** "do-not-re-try only" + search *starvation* returns Exhibit 12's
  rows; count = 30 everywhere; `Book.ledger` badges the four recovered rows book-wide.
- [P0] [expA-M4] Row #68 parsing: split 7-cell rows from both ends (or unescape `\|`
  and tolerate a bare `|`) + hard gate: every parsed row has exactly 7 cells and
  non-empty `documented`. Restores the `[AMBIG] see #69` residual and the correct
  citation. — **AC:** row #68 round-trips verbatim; the gate fails the build on any
  malformed row.
- [P0] [expB-MJ-8] Date-scope the two cell-B statements (`:586-591`, `:1305-1310`):
  "…the 2026-07-30 adjudication left open pending the cell-B control; that control
  landed 2026-07-31 and released the gate — C9 remains live and unfixed, and w_G stays
  off the exonerated list." — **AC:** nothing static lets a resolved question look
  open (the meta-rule's mirror image).
- [P1] [expB-MJ-3] Ledger row #88's verdict string gains the disambiguation clause
  ("this 'Cell B' is the three-way A/B's leg B on the seed1000 deep venue — not the
  2026-07-31 2×2 cell B; see Ch 11 §5"), via gen_museum (mark as a book-added
  annotation, not ledger text). — **AC:** searching "cell B" cannot mislead.
- [P1] [ped-B4 step 1] §7's claim about BW3 → the scoped truth ("…backs the search box
  carried on every chapter page, and the dead-end verdicts the sandboxes hard-code" —
  or, after the integrator ships inline chips, the stronger true sentence). — **AC:**
  the museum describes the instrument that exists.
- [P1] [mandate-D1] Dossier `:1316` → canonical string + erratum note.
- [P2] [expA-m2] "twenty-one were real defects that were fixed and did not fix the
  symptom" → the ledger's own qualifier ("…and landed — most of them, in the ledger's
  own words, *insufficient alone*"); rows #9/#12 contradict the current sentence.
- [P2] [expA-m3/m4] M1 flag box "1.7–2.3%" → "0.2–2.3%"; static fallback "≈10⁻⁸⁶" →
  "10⁻⁷⁹–10⁻⁸⁶ — 0.0000 at every printed precision, essentially flat in n".
- [P2] [tomas-m5, ped-m6] Per-exhibit backlinks: "referenced by: Ch N §… · ← back".
- [P2] [ped-M5] Ship the five interlude exhibits expanded, the other nine collapsed
  (`<details>`, summary = hypothesis + verdict badge). — **AC:** the room stays
  complete; the visit gets a path.

### index.html  *(integrator-owned)*

- [P0] [expB-MJ-7] `:145-149` → "three live, measured inconsistencies… plus a fourth
  (C5), and an attribution that was confounded until the control was run on
  2026-07-31." — **AC:** consistent with ch11's board count of four.
- [P1] [ped-B1/m7, tomas-M10] "The journey": rewrite Ch 4's and Ch 8's blurbs as
  failures, not answers ("…and the answer moves the wrong way"); wrap the per-chapter
  discovery statements in a `<details>` ("spoiler: what each chapter discovers") so
  the front door maps without pre-answering four cold opens. — **AC:** no chapter's
  reveal number appears un-collapsed on the front door.
- [P1] [ped-B4 step 1] The BW3 promise ("the book will tell you") → the scoped truth,
  matching the museum §7 fix / the shipped mechanism. — **AC:** promise matches
  behavior.

---

## D. Shared-file work — ONE consolidated integrator item

All changes to `js/book.js`, `js/manifest.js`, `css/book.css`, `index.html`,
`book/README.md`, and cross-page policy. File new WIDGET_REQUESTS entries for each.

1. **[P0] Symbol Passport chapter gating** [mara-MAJOR-2, ped-M4]: add `firstChapter`
   to `Book.SYMBOLS` entries; before the home chapter render sym/units + rung-safe
   gloss only, suppress formula-bearing `meaning` and `note` (page index from
   manifest). Rung-safe glosses needed at minimum: `wG` ("mixture weight — the
   probability the host is catalogued; how it is computed is Ch 9"), `eps`, `Cscale`,
   `Sig` ("the measurement's covariance: size and tilt of the error ellipsoid" — Γ⁻¹
   from Ch 6 on). — **AC:** hovering w_G in ch05 shows no β_G/D, no "ESTIMAND-
   DEPENDENT", no C9; ch09+ shows the full card.
2. **[P0] UX: fetch failure surface** [ux-MAJOR-1]: shared `.catch()` on
   `Book.loadJSON` consumers (a `Book.widget(id, url, render)` wrapper is fine) that
   swaps the container for the adjacent `<noscript>` text or a one-line pointer to the
   static fallback. — **AC:** killing the server / `file://` in Chromium yields
   readable fallbacks in every widget container, zero silent blank boxes.
3. **[P0] UX: dark-mode badge contrast** [ux-MAJOR-2]: mirror the badge/chip/note
   `data-theme="dark"` overrides inside
   `@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){…} }`.
   — **AC:** first-visit OS-dark badges measure ≥ 4.5:1 (the four failing colors were
   3.05–3.97:1).
4. **[P1] Cumulative bias rail** [ped-M7]: `BOOK_BIAS_ROWS` in manifest with
   `from_chapter`; pages render all rows with `from_chapter ≤ n` + own pips. — **AC:**
   the rail never loses rows moving forward; ch11 shows the full five-row history
   under its amber pips.
5. **[P1] BW3 inline chips** [ped-B4 steps 2–3, per §B-4]: on a `data-hypothesis`
   control becoming active, render `⚖ #N — <verdict>` inside the widget;
   `data-hypothesis-verdict="inline"` opt-out. Chapter agents add tags: ch08 I8.2 →
   #71/#72/#89; ch09 I9.2 → #61; ch10 I10.x → #49a; ch11 I11.1 → #61/#64. — **AC:**
   the ch04 denominator switch flipped to `local window (recorded)` volunteers ledger
   #9 without a page reload; museum/index claims (C-museum, C-index items) become
   literally true.
6. **[P1] Canonical strings**: publish in one place (manifest or a shared JS/data
   constant) the D1 dossier row + erratum line and the D3 cell-B pip; convert ch06/
   ch08's page-local rail pips to `Book.biasRail` [tomas-m2]. — **AC:** grep finds one
   definition; all four cell-B pips identical; rail amber consistent across all
   chapters.
7. **[P1] SYMBOLS additions** [tomas-M5]: `nw`, `Sglob`, `Wcat`, `Vf`, `Fincat` (+
   review `phcat` for ch07's φ_cat), with units + one-line pictures; verify 0 unknown
   keys after chapter agents tag. — **AC:** ch09 §4 / ch07 §6 / ch11 C8 hovers all
   resolve.
8. **[P2] Predict grading support** [mara-MAJOR-3]: optional `data-predict-correct`
   on `predictReveal` so widgets can grade uniformly (ch02 is the customer). — **AC:**
   ch02's reveal opens with the graded verdict.
9. **[P2] `Book.lazyPlot`** (or documented ch03-pattern recipe) for ch02/ch08/ch09
   [ux-MAJOR-3]; **[P3]** consolidate the per-plot `MutationObserver`s into one
   [ux-MINOR]; **[P2]** merge the two `@media print` blocks [ux-MINOR]; **[P3]**
   `_template.html` placeholder-JSON guard comment [ux-MINOR]; **[P3]** persona
   switch only ever opens, never force-closes [ped-m3 tail]; add the three persona
   nudge sentences (ch06 Fisher, ch08 mass measure, ch11 adjudication) — one line
   each, coordinate with those agents [ped-m3].
10. **[P2] Anchor-drift policy** [mara-MINOR-8, expA-m1]: for the chips enumerated in
    BUILD_REPORT §5.5 item 23 + expA-m1's IDEALIZED/CLAIM drift table, add
    current-line `title=` tooltips beside the spec anchors (or re-grep in
    generators). — **AC:** following any listed chip lands on the cited text or shows
    the current line.
11. **[P2] `book/README.md`**: drop the retired `ch00-demo` references (frozen-list
    exception granted by this worklist) [tomas-m6, BUILD_REPORT §7.5]. Also note the
    vendor-scoped external-ref grep policy for future CI [ux-MINOR].
12. **[P0] Build gates** (make_all or a QA script): (a) the D1 grep gate (§A-D1);
    (b) museum row-shape gate (§C-museum); (c) do-not-retry count = 30; (d) no page
    contains "not landed"/"in flight" adjacent to cell-B references. — **AC:** gates
    run in `make_all.py` and fail loudly.

---

## E. Revision fan-out plan

### Wave 0 — unblockers (parallel)
| agent | scope |
|---|---|
| **ch03 agent** | D2 regeneration + all ch03 items. Publishes the regenerated census numbers in its flag file for ch02/ch04/ch05 to consume. |
| **integrator (pass 1)** | §D items 1–3, 6, 7, 12 (gating, catch, contrast, canonical strings, SYMBOLS, build gates) — everything other agents depend on. |

### Wave 1 — chapter agents (parallel; ch02 waits for ch03's numbers)
| agent | scope (from §C) | notes |
|---|---|---|
| **ch00** | small: M5 trim, GWTC-3 row, m6 wording, traps | gen_ch00 touch |
| **ch01** | D1 arc (F1→erratum, Q1.2), σ_Mz note, M_z label, minors | |
| **ch02** | BLOCKER-1 (regen numbers), grading, 52%, rung move, traps, D1, lazy-init | **after ch03**; coordinates β(h)^N box handoff with ch04 |
| **ch04** | D1 + JSON rename, p_det marginal, Q4.3/Q4.4, marker desync, census consume | receives ch02's box |
| **ch05** | w_G type, C10 attribution, Q5.4, κ dial, de-spoil §3, traps, D1 | |
| **ch06** | D1 (Q6.5 550×), 14×14 κ + Babak, trims, traps, cross-ref | gen_ch06 touch |
| **ch07** | cell-B set (BL-3, MN-1/2/5/6), φ_cat, Q7.1, traps, D1, D5 | |
| **ch08** | σ_Mz ×5, de-spoil, MJ-4 cell-B, Q8.1, spectral-siren, D1, tags, lazy-init | new F-ch08 flag for reviewer-computed numbers |
| **ch09** | cell-B set (MJ-5 + w_G win), symbols, residuals, global-mode scoping, D1, tags, lazy-init | gen_ch09 touch |
| **ch10** | cell-B set (BL-7 + Q10.5 postscript), de-spoil, Laghi anchor, D1, num-view | |
| **ch11** | cell-B centerpiece (BL-1/2/4/6, MJ-1/2/3/6/7, MN-3/4), two-priors, recall slug, 3135, D1, §4 collapse | gen_ch11 regen; largest scope — start first in the wave |
| **museum** | parser fixes (M3/M4), cell-B date-scoping, #88 clause, BW3 claim, D1, counts, backlinks, collapse | gen_museum regen |

### Wave 2 — integrator (pass 2) + close-out
index.html items; §D items 4, 5, 8–11; place the four canonical cell-B pips; full
`make_all.py` regen (13/13 + new gates); QA sweep re-run (the BUILD_REPORT §4 table +
the ux review's mechanical checklist); consolidate the appended flag entries into a
BUILD_REPORT §5 addendum; browser click-through of every changed widget (BUILD_REPORT
gap #1 still open — this pass inherits it).

### Frozen (no agent touches)
`vendor/` · `_template.html` (except the §D-9 comment) · the spec docs
(`BOOK_PEDAGOGY.md`, `BOOK_DESIGN.md`, `BOOK_SOURCES_MAP.md` — historical spec; the
book corrects pages, flags record) · all `MasterThesisCode` (main repo) files — book
agents never edit production code or project artifacts (§F routes those) · existing
flag-file text (append-only) · pre-registration blocks quoted in pages (verbatim rule,
D3) · everything praised in the reviews' PRAISE sections — explicitly including
ch07 §6's structure, ch08's C4 refutation, ch09's I9.2 refusal-to-invent, ch10 §4's
exoneration, the museum's F-museum-1 non-adjudication, Ch 2's rung-guard sentence, and
the no-answer-key block's *form*.

---

## F. Back to the AUTHOR (not fixed by this revision)

1. **F-museum-1's production code comments** — `bayesian_statistics.py:384` and
   `:3670` carry the falsified scalar-collapse mechanism attribution. Main-repo
   change, `/physics-change` protocol. Also: whether F-museum-1 and F-museum-2 become
   BIAS_HISTORY_LEDGER rows (BUILD_REPORT §5.2 already asks).
2. **`CLAIM_2D_BIAS_20260730.md:172`'s σ_Mz ≈ 1e-4** — measured 8.8×10⁻⁸ (F-ch06-5;
   plausibly a test-tolerance transcription). The book carries both values (D5);
   amending the claim file is the author's.
3. **`CELLB_READOUT_20260731.md`'s "confirmed on every pre-registered read"** — the 1D
   MAP (0.7450) missed the registered 0.70–0.74 band by one grid step (band written in
   MAPs; the mean is inside). The book scores it honestly (D3); annotating the project
   artifact is the author's. (expB BL-4.)
4. **Expert B's reviewer-computed unscattered C4 partition** (219 survivors / 80.7%,
   688 zeroed / 19.2%; dark zero-leg fraction 0.855; ×8.39 de-weighting) — used in
   ch08 with a "recomputed for the book" marker; whether it is promoted into an
   adjudicated artifact / claim-file amendment is the author's.
5. **`BIAS_HISTORY_LEDGER.md:88`'s unescaped pipes** (row #68) — one-character
   upstream fix; the book's parser now tolerates it either way. Also the §3
   "Cell B"/A′ naming collision with the 2×2 — a one-line upstream annotation would
   help future readers (the book disambiguates on its side regardless).
6. **C7 rail threshold 0.256 vs 0.2644** — origin still unreconstructed
   (BUILD_REPORT §5.1-2); no book action beyond the existing flags; project-side
   archaeology is the author's if wanted.
7. **σ_dL upstream** — D1 corrects the book; the spec cards/design docs and any
   project artifact carrying 8.0×10⁻⁵ as a fraction remain uncorrected historical
   record unless the author amends them.
8. **The two `/physics-change` gates cell B unblocked** (joint C9+C8 mass-consistent
   mixture; C7 kernel fix superseding G2b, not the exonerated p_det-in-numerator
   form) — the book now states these as the fix surface; running them is the thesis
   work, not the book's.

---

## G. Rejected review items (with why)

1. **[mara BLOCKER-2, fix option 2]** Print EMRI-889's ball at both 2σ and 1.5σ as
   "both genuinely live" — rejected; there is one production ball-search call site
   (verified, §B-2). The 2σ census is not a live production configuration and would
   be a counterfactual dressed as fact.
2. **[mara BLOCKER-1's literal replacement text]** — superseded: embeds 2σ census
   numbers that D2 regenerates. Shape adopted, literals rejected (§B-3).
3. **[All reviewers' "print both σ_dL values" fixes]** — superseded by D1's mandate
   (§B-1). The fix-site inventories are retained.
4. **[tomas M3.3]** Add Chen, Fishbach & Holz 2018 as an external anchor for Ch 2/
   Ch 10 — rejected this round: lowest payoff of the five literature items (a
   citation anchor for a result the book measures internally), two-page blast radius,
   and the book's sourcing discipline (repo artifacts) makes decorative external
   citations a new precedent. The other four literature items are accepted. Revisit if
   a citations pass happens for other reasons.
5. **[ped M5's "everywhere" sweep]** (~40 gate-tolerance sentences book-wide) —
   scoped down: the four targeted relocations (ch00, ch06, ch11, museum) are accepted;
   the book-wide sweep is rejected this round (churn across all 14 pages, every
   sentence needing individual judgment, against a delivery-polish payoff). Agents
   already touching a page apply the rule opportunistically ("gate tolerances live in
   provenance panels / num-views").
6. **[ped M1's full 21-question rewrite]** — scoped down: the named worst offenders
   (Q2.3, Q7.1, Q7.5, Q8.1, Q11.2 + ch11's §4-collapse side effect) are in per-chapter
   items; a mechanical rewrite of all 21 is rejected (several "overlaps" are the
   Examiner-persona's legitimate access path, and ped's own diagnosis is that the
   overlap is a side effect of good prose — option (a)'s fold-relocation is folded
   into the M5-scoped trims instead).
7. **[ped B4 as originally framed]** ("the instrument is worthless as shipped") —
   partially rejected: the integrator's no-auto-reveal reasoning stands (§B-4); we fix
   the claims and ship the scoped inline-chip version, not a page-wide auto-reveal.
8. **[BUILD_REPORT gap #4]** (remove Q11.6's model answer) — rejected; both reviewers
   who audited it say keep (§B-5).
9. **[ux MINOR: vendor Plotly external-URL strings]** — no action beyond the README
   note (§D-11); the reviewer's own finding is that they are unreachable.
10. **[tomas m7's implied restructure of ch09's dossier into a special beat]** —
    accepted only as the minor reframing in §C-ch09; a redesigned beat is rejected
    ("889 is not special here" is itself honest venue discipline).

---

*Synthesizer, 2026-07-31. No book page was edited by this pass; this file is the
only artifact written.*
