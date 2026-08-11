# museum_FLAGS.md — the Defect Museum annex

Raised by the museum agent, 2026-07-31, per `BOOK_DESIGN.md` §4.1: *"if a generator's
recomputation disagrees with a spec number, **stop and flag; do not silently reconcile in
either direction**."*

Both items below are presented **on the page in both forms**, side by side, with the
disagreement named. Nothing was adjusted, dropped, or averaged.

---

## F-museum-1 — ⚠⚠ The flagship exhibit's stated mechanism does not reproduce: `volume_trunc`'s "`fixed_quad(n=50)` aliases the GW peak to 0.0000" is a **scalar-collapse artifact of the diagnostic script**, not a property of `fixed_quad(n=50)`

**This is the sharpest flag in the annex. It does not overturn any project verdict, and the
museum does not adjudicate it — but it must not be shipped silently.**

### What the spec and the artifacts say

- `BOOK_SOURCES_MAP.md` §4 exhibit 1, `BOOK_PEDAGOGY.md` B5 (Ch-7 interlude) and Part 4 M1,
  and `BOOK_DESIGN.md` §1 (Ch 7 interlude + the museum card) all state the flagship AHA as:
  *"at n = 50 the integral reads **0.0000** where the exact value is 0.24–0.65 — the peak
  falls between nodes"*, and *"two independent causes at once"*.
- The primary artifact is `results/volume_trunc_ab_20260712/FINDING.md:1-58`
  ("Mechanism (two compounding effects; `quadrature_diagnostic.py`)"), whose table reads:

  | h | numerator, GW window (n=50) | numerator, host window (n=50) | numerator, host window (exact quad) |
  |---|---|---|---|
  | 0.60 | 0.0003 | **0.0000** | 0.2417 |
  | 0.73 | 0.0005 | **0.0000** | 0.4314 |
  | 0.86 | 0.0007 | **0.0000** | 0.6537 |

  and concludes "**Quadrature aliasing (dominant).** … the sparse Gauss-Legendre nodes
  straddle the narrow GW peak and miss it (n=50 → 0.0 vs exact 0.24–0.65)".
- The same attribution is carried into production source comments:
  `bayesian_statistics.py:384` and `:3670` ("fixed_quad n=50 aliases the narrow GW peak over
  the wide host window").

### What `gen_museum.py` measures

Running `results/volume_trunc_ab_20260712/quadrature_diagnostic.py` unchanged reproduces its
published table **exactly** (verified, this session). Re-computing the *same integrand* with
the GW leg written the way **production** writes it does not:

| h | host-window `fixed_quad(n=50)`, GW leg via `dist()` (the diagnostic's form) | host-window `fixed_quad(n=50)`, GW leg via `dist_vectorized()` (production's form) | exact |
|---|---|---|---|
| 0.60 | 0.0000 | **0.2376** | 0.2417 |
| 0.73 | 0.0000 | **0.4412** | 0.4314 |
| 0.86 | 0.0000 | **0.6524** | 0.6537 |

n = 50 is accurate to **1.7–2.3%**, not to 100%.

### The mechanism of the disagreement (measured, not inferred)

`darksiren_emri/physical_relations.py:132 dist()` is **scalar-only**: given an array it
returns a 0-dimensional array holding the value at the array's *first* element.
`scipy.integrate.fixed_quad` passes the whole node array in one call, so inside the
diagnostic the GW factor becomes a **constant** — the likelihood at the window's lower
limit — and the node count then cannot matter.

That hypothesis is quantitative and it reproduces the diagnostic's **entire** table:

- host window, lower limit z = 0 ⇒ `d_L = 0` ⇒ distance fraction 0 ⇒
  `exp(−0.5·(1/0.05)²) ≈ 1.4×10⁻⁸⁷` ⇒ prints as `0.0000` at every h and **every n**
  (measured ladder, h = 0.73: 6.5e−79 at n = 10 … 1.1e−86 at n = 600 — flat in n);
- GW window, lower limit `z(d_L − 4σ_dL)` ⇒ fraction 0.8 ⇒ `exp(−8)` ⇒ predicted column
  **0.000265 / 0.000468 / 0.000701** at h = 0.60/0.73/0.86, i.e. **0.0003 / 0.0005 / 0.0007**
  to the printed 4 dp — the published GW-window column, digit for digit
  (`museum_quadrature.json.gates`, all `match_4dp: true`).

`darksiren_emri/bayesian_inference/bayesian_statistics.py:3806`
(`numerator_integrant_without_bh_mass`) and `:3826` (`denominator_integrant_without_bh_mass`)
both use `dist_vectorized`, so **the production path does not have this issue**, and the
diagnostic's zeros were never production numbers.

### What is and is not affected

- **NOT affected — the verdict.** `volume_trunc` is FALSIFIED on the seed600 494-event A/B
  gate (`gate_result.json`: 1D mean 0.7450 → 0.8000, 2D 0.7681 → 0.8000), which was run with
  production code. The museum presents that verdict unchanged.
- **NOT affected — mechanism (2).** "Even the exact host-window numerator tilts high" is
  reproduced here exactly (0.2417 → 0.4314 → 0.6537, monotone in h), and it is by itself a
  sufficient explanation of the collapse onto h = 0.80.
- **Affected — mechanism (1) and the "two independent causes" framing.** The evidence
  offered for the aliasing cause is the `0.0000` column, and that column is explained by the
  scalar collapse.
- **Still true — aliasing is real, at a different order.** The vectorized ladder at h = 0.73
  reads 1.504 (n = 10), 0.147 (15), 0.068 (20), 0.458 (25), 0.604 (30), 0.382 (40), 0.441
  (50), 0.4314 (75, converged) against an exact 0.4314 — erratic by factors 3.5× high and
  6.3× low, in both directions, exactly the "which nodes happen to catch the peak" behaviour
  the FINDING describes. The lesson ("quadrature is physics") is intact; only the node count
  at which it bites has moved.

### Disposition

- `museum.html` exhibit `#ex-volume-trunc` and interactive **M1** carry a two-state
  evaluation switch — *scalar `dist()` (2026-07-12 diagnostic)* vs *`dist_vectorized()`
  (production)* — so the reader sees both columns and the flat-in-n signature themselves.
  The recorded FINDING.md table is drawn as a labelled `rec` overlay and is never replaced.
- The page states in the narrator's voice that the book does **not** adjudicate this, links
  this flag, and leaves the project's verdict on `volume_trunc` standing.
- `gen_museum.py` enforces both reproductions as hard gates: it raises rather than writing a
  file if the exact column, or the scalar-collapse prediction of the GW-window column, ever
  stops matching FINDING.md to 4 dp.
- **For the author / integrator:** this is a candidate ledger entry in its own right (a
  falsified *mechanism attribution* inside a correct falsification), and it touches two
  production source comments (`bayesian_statistics.py:384`, `:3670`) that propagate the
  attribution. It is raised here, not resolved.

---

## F-museum-2 — commission injection scan (#49a): "tracks the truth exactly" vs a re-run that is one grid step low in 2 of 5 cells

- **Recorded:** ledger row **#49a** and `synthesis/WF2_DIGEST.md:26-30` /
  `synthesis/DRAFT_REPORT.md:24-27` — *"catalog_only MAP tracks truth **EXACTLY**
  (0.63→0.63 … 0.77→0.77); PRODUCTION MAP = 0.86 for EVERY injected truth"*.
- **Re-run by `gen_museum.py`** (importing `results/commission_20260701/injection_scan.py`
  and calling its own functions with its own seeds — `default_rng(2024)` for the catalogue,
  `default_rng(int(h·1000))` per injection):

  | injected truth | 0.63 | 0.67 | 0.70 | 0.73 | 0.77 |
  |---|---|---|---|---|---|
  | `catalog_only` MAP | 0.630 | **0.660** | **0.690** | 0.730 | 0.770 |
  | production MAP | 0.860 | 0.860 | 0.860 | 0.860 | 0.860 |

- **Assessment.** The headline — the production estimator's MAP is *independent of the
  injected truth* — reproduces exactly, and it is what the exhibit is about. The
  `catalog_only` control lands one 0.01 grid step below truth at two of the five injections,
  so "tracks the truth exactly" is a **grid-step-level** overstatement, not a different
  result. The digest's own parenthetical (`0.63→0.63 … 0.77→0.77`) quotes only the two
  endpoints, both of which are exact.
- **Disposition:** the page prints the **re-run** table with all five cells, quotes the
  digest's wording next to it, and says which two cells differ and by how much. No number
  was adjusted. The exhibit's claim is stated as *"the production MAP does not move with the
  truth; the control does"* — which both the record and the re-run support.
- **Venue note carried on the page:** this scan is the commission's own synthetic harness
  (20,000-galaxy catalogue, moderate completeness `f(z) = exp(−(z/0.3)²)`, an `erfc`
  detection horizon at 3.0 Gpc) — **not** the production catalogue, and the museum says so.

---

## F-museum-3 — build portability: two of the museum's sources are untracked

- `results/commission_20260701/**` (the WF digests, `DRAFT_REPORT.md`, `injection_scan.py`)
  is **not git-tracked** — it exists only in the working tree of the main checkout, so it is
  absent from this worktree and from a fresh CI clone. The ledger, the claim file,
  `results/volume_trunc_ab_20260712/`, `results/mass_trunc_ab_20260713/`,
  `docs/gates/G6_starvation_postmortem.md` and `docs/H0_BIAS_RESOLUTION.md` **are** tracked
  and present.
- **Disposition:** `gen_museum.py` resolves every artifact from this repo root first, then
  from a sibling `MasterThesisCode` checkout; if `injection_scan.py` is missing it prints a
  NOTICE and leaves the already-committed `museum_h0_independent.json` untouched rather than
  failing the build or writing a degraded file (the same pattern as `gen_ch04.py`, see
  `ch04_FLAGS.md` F-ch04-5).
- **For the integrator:** identical to F-ch04-5. Either those artifacts get committed, or
  every generator keeps the tracked-first / sibling-fallback / keep-committed-output pattern.

---

## F-museum-4 — no disagreement found (recorded for the audit trail)

Checked and reproduced without discrepancy: the ledger's own row count (**98** rows, ids
1–94 plus 49a/49b/49c/49d — matching the spec's "98 hypotheses"); the ledger §2 DO-NOT-RE-TRY
union (17 items, back-referencing 26 distinct ledger rows) and the claim file's 15-entry
Exonerated list, both parsed from the artifacts rather than transcribed; the `volume_trunc`
A/B posteriors (`gate_result.json`); the FINDING.md exact-quadrature column (see F-museum-1).

---

# REVISION — 2026-07-31 (post-review pass, `REVISION_WORKLIST.md` §C-museum)

Appended, not rewritten: everything above is the build-pass record and stands as
written. Two of the entries above are **corrected in state** by items below —
F-museum-4's "26 distinct ledger rows" is superseded by F-museum-5 (the count was a
parser defect, not a property of §2), and the museum's own §7 claim about the BW3
instrument is superseded by F-museum-6.

## F-museum-5 — the DO-NOT-RE-TRY union was **26 rows, not 30**: a separator-class defect in this generator (expA-M3, expA-M4 in the same parser)

**Two independent parser defects in `build_ledger`, both found by review, both fixed
2026-07-31. Neither changed a project number; both changed what the museum *said* the
project had ruled out — which, in a room whose meta-rule is "no dead hypothesis may look
alive", is the failure mode that matters.**

### (a) the separator class — four rows silently unflagged

`gen_museum.py` read §2's ledger back-references only when the parenthesised group was
**comma-separated** (`#\d+[a-z]?(\s*,\s*#\d+[a-z]?)*`). §2 writes item **13** as
`(#41/#52)` and item **15** as `(#43/#44)`, with slashes. Both groups were rejected whole,
so rows **#41, #52** (information starvation) and **#43, #44** (heliocentric / PV frame)
came out of the parser with `do_not_retry: false`.

- **Measured before:** `do_not_retry_rows` = 26. **After:** 30, from the same 17 §2 items.
- **The page contradicted itself on click:** Exhibit 12 is badged *"DO NOT RE-TRY — ledger
  §2 item 13"*, while ticking *do-not-re-try only* in §7's browser and searching
  `starvation` returned **0 rows**. Verified fixed: the same search now returns row #52,
  `heliocentric` returns #43, `pv` returns #44.
- **Blast radius beyond this page:** `js/book.js` `Book.ledger` badges from the same field,
  so information starvation and the PV frame were reachable from any chapter's ledger panel
  **without** their verdict badge. Recovering the rows in the data fixes every page at once;
  no chapter edit was needed.
- **Three printed counts corrected** (all said 26): the census caption, the census `<noscript>`
  fallback, §7's browser `<noscript>`, M.4's answer, and the provenance panel — five sites,
  one more than the review found.
- **Guard added:** the generator now raises if §2 does not resolve to exactly
  `EXPECTED_DNR_ROWS = 30` rows, or if it back-references a row id absent from the table.
  `book/generators/qa_gates.py` (integrator) gates the same number from the other side.
- The guard that **rejects** `(#30 option b)` — a GitHub issue number, not a ledger row — is
  unchanged and still correct.

### (b) row #68 — an unescaped pipe shifted six cells and deleted an `[AMBIG]` marker

`BIAS_HISTORY_LEDGER.md:88` writes `(trimming top-|tilt| GROWS it)` with **unescaped** pipes
inside the VERDICT cell. (The three other cell-internal pipes in the table, rows #49d and
#82, are escaped `\|` and always parsed correctly.) A naive split gave 9 cells, so the row
shipped with the verdict truncated at `(trimming top-`, `documented` = `"tilt"`, and
`residual` = `"GROWS it)"`.

**The destroyed cell was `[AMBIG] see #69`** — the marker recording that #68's attribution
was *reopened* by the h1_zclamp re-attribution, i.e. exactly the open thread Exhibit 1 leans
on ("unowned since 2026-07-13 — ledger #69"). The searchable ledger dropped the ambiguity
flag from the one row that owns it, in the annex whose contract is verbatim transcription.

- **Fix:** `_split_row` folds any surplus cells back into the VERDICT column (the table's only
  free-text column) after honouring `\|` escapes.
- **Hard gate (the review asked for it; it is in the generator, so it fails the build, not the
  review):** a data row that does not recover to exactly 7 cells raises; every one of the
  seven cells must be non-empty; and `documented` must match a citation shape
  (`:\d`, `.md`, `.py`, `.json`, `§`, `#\d`) — the specific way this defect presented.
  `qa_gates.py` re-checks the shipped JSON independently.
- **Verified round-trip** against the source row: verdict ends `(trimming top-|tilt| GROWS it)`,
  `documented` = `pp_coverage_shallowvenue_20260711/SUMMARY.md:12-76`,
  `residual` = `[AMBIG] see #69`. Census unchanged (the truncated verdict classified as
  `measured`, and so does the full one).
- **Upstream is the author's** (worklist §F-5): one character in
  `BIAS_HISTORY_LEDGER.md:88`. The parser now tolerates the source either way, so the fix is
  optional, not blocking.

## F-museum-6 — the museum described a BW3 instrument the build does not have (ped-B4, §B-4)

§7 said the ledger *"will back the book-wide **Has this been tried?** instrument, which
volunteers a verdict whenever a sandbox anywhere in this book is dragged into a configuration
the project has already killed."* As shipped there is no auto-reveal: `Book.ledger` gives every
chapter page a **search box** over these rows plus a seeded list of the dead hypotheses that
page's sandboxes can reach, and each sandbox **hard-codes its own** dead-end verdict at the
control. That is a deliberate integrator choice (BUILD_REPORT §2: a second automatic reveal
would double-report and pre-empt the predict-locks), and the worklist upholds it (§B-4) — so
the defect is the *claim*, not the build.

- **Corrected on the page** to the scoped truth, with the correction dated and stated in the
  open rather than quietly swapped. The same over-promise on `index.html` is the integrator's
  (worklist §C-index).
- **If integrator pass 2 lands the scoped inline chips** (§D item 5), §7's paragraph can be
  strengthened; it is written so that it stays *true* either way, not so that it needs
  rewriting. The museum's §1 meta-rule is normative ("must volunteer") and is untouched —
  it is the requirement the chips would satisfy more completely.

## Cell B landed — the museum's two stale statements, date-scoped (expB MJ-8)

Both sentences were **correct about the 2026-07-30 adjudication and false about the current
state**, which is the mirror image of the meta-rule this room enforces: nothing static may let
a *resolved* question look open. Kept and date-scoped rather than rewritten, at **four** sites
(the review found two):

1. Exhibit 2's "still live downstream" note — C9 "which the adjudication leaves open pending
   the cell-B control" → 2026-07-30 left it open; the control landed 2026-07-31 and released
   the gate; **C9 remains live and unfixed** and `w_G` stays off the exonerated list.
2. The binding-union adjudicator block — same treatment; the load-bearing omission of `w_G`
   from the exoneration list is unchanged, only its tense is.
3. *(not in the review)* Exhibit 8's closing note — "is *live* and gated on the cell-B control".
4. *(not in the review)* Self-check **M.3**'s answer — "until it lands, the honest state is
   'measured, unexplained, gated'". Rewritten to state what the control actually delivered
   (`w_G(h)` bit-identical to the realistic run at all 41 grid points, so the mis-calibration
   is estimator-borne) and what it did **not** do: the honest state moved from *gated* to
   *attributed and waiting on a derivation*, not to *closed*.

No exhibit's verdict changed. Cell B is cited as `CELLB_READOUT_20260731.md`; the museum
reports no cell-B number of its own, so D3's job-ID split does not arise here.

## Ledger row #88's "Cell B" is not the 2×2's cell B — book-added annotation (expB MJ-3)

Row #88's verdict cell says *"Cell B (broadened `volume_deconv` numerator + generator norm):
1D = 0.73 truth, 2D = 0.80 INTERIOR HIGH"* — the three-way A/B's leg B on the **seed1000 deep
venue**. The 2026-07-31 2×2's cell B is a different object on a different venue with a nearly
identical 2D number (0.7900). A reader who finishes Ch 11 §5 and types "cell B" into the ledger
search lands on #88 first.

- **Implementation:** a new `book_note` field per row, populated only for #88, carried in
  `museum_ledger.json` and rendered **below** the verdict in its own italic, rule-marked block
  that names itself *"Book note (not ledger text)"*. The quoted verdict string is untouched —
  the annex's contract is verbatim transcription, and an annotation merged into the quote would
  break it more quietly than the collision it fixes.
- The note is also in the browser's **search haystack**, so the disambiguation surfaces on the
  query that causes the confusion.
- **Open handoff — integrator:** `js/book.js` `Book.ledger._fmtRow` (the per-chapter panel)
  renders `r.verdict` only, so a "cell B" search *from a chapter page* still shows row #88
  without the clause. One line (`+ book_note`) in that formatter closes it book-wide. Filed
  here rather than edited: `js/book.js` is integrator-owned.

## D1 — σ_dL, applied (mandate)

Dossier row → the canonical string (`d_L 88.9 Mpc · σ_dL/d_L = 8.98×10⁻⁴`), and the card's
remaining facts (SNR 1425, host index 859360, in catalogue) moved to their own row rather than
being smuggled into the distance line. The canonical erratum note follows the dossier, with one
museum-specific sentence: a units slip that survived six chapters is itself a defect with a
date, so this room keeps it visible instead of silently overwriting it. Verified: `8.0×10⁻⁵`
now appears exactly once on the page, inside that erratum.

## Minor corrections (expA m2 / m3 / m4)

- **m2 — the cold open overclaimed one class.** *"twenty-one were real defects that were fixed
  and did not fix the symptom"* is false for rows #9 ("MAP 0.60 → 0.73, bias −17.8% → 0.0%")
  and #12 ("PRIMARY mover per A5"), and it contradicts this page's own closing Trap. Replaced
  with the ledger's own qualifier: *"fixed and landed — most of them, in the ledger's own
  words, **insufficient alone**"* (the phrase appears 4× in the source).
- **m3 — M1's flag box said n = 50 is accurate to "1.7–2.3%".** Recomputed from the shipped
  ladder: **1.70%** (h = 0.60), **2.26%** (0.73), **0.21%** (0.86). The stated range excluded
  the best of the three. Now "0.2–2.3%", with the three values printed.
- **m4 — M1's static fallback said the scalar ladder returns "≈10⁻⁸⁶" at every order.** The
  shipped ladder runs **6.53×10⁻⁷⁹** (n = 10) → **1.11×10⁻⁸⁶** (n = 600). The claim being made
  (flat in n, prints as 0.0000) is right; the exponent was right only at the high-n end. Now
  states the range, both endpoints, and "0.0000 at every printed precision".

None of these touches F-museum-1's verdict, its gates, or the recorded FINDING.md overlay.

## Room navigation: nine exhibits collapsed, per-exhibit backlinks (ped-M5, ped-m6, tomas-m5)

The museum's main column was **2.79× budget** (6,981 words vs 2,500) — the worst ratio in the
book — and two independent walk-throughs got lost by Exhibit 8 with no labelled way back.

- **Collapsed (9):** every exhibit no chapter interlude sends the reader to —
  `absolute_marginal`, numerator-only, Gray mixture, HA, zero-host fallback, `w_G` bookkeeping,
  HB, the index bug, archaeology. Summary = the exhibit number/ledger row, the hypothesis
  headline and the verdict badge, so the *room* is still fully readable at a glance.
- **Expanded (5):** the interlude targets — `volume_trunc` (Ch 7), `mass_trunc` (Ch 8),
  the p_det anchor (Ch 4), information starvation (Ch 5), the H₀-independent estimator (Ch 10).
- **Nothing is hidden from anyone who asked for it.** A page-local script opens the fold that a
  deep link (from a chapter, or from the ledger browser) targets — on load *and* on
  `hashchange` — then re-runs the jump, because the anchor resolves before the fold expands.
  `beforeprint` and a `matchMedia("print")` listener open all of them, so the printed annex is
  complete. Without JS, `<details>` still opens on click natively.
- **Backlinks:** one `.mus-backlink` line per exhibit naming its referring chapter and section
  with a link back (e.g. *"referenced by Ch 9 §6 · ← back"*). Five exhibits have no chapter
  referrer at all; those say so — *"Not cited from a chapter — museum-only"* — rather than
  inventing one. Chapter headings carry no `id`s, so the links are page-level with the section
  named in text; if a chapter agent adds heading anchors, these can be sharpened.
- No plot lives inside an exhibit block (all six `widget-plot` containers sit between exhibits),
  so no Plotly instance is constructed inside a closed `<details>`. Verified before collapsing.

## Verification

- `gen_museum.py` re-runs clean end to end (~16 s): 98 rows, census unchanged
  (21/21/19/8/8/8/7/4/2), do-not-re-try union 30 rows from 17 §2 items, the FINDING.md exact
  and GW-window columns still reproduce to 4 dp, and the injection scan re-runs to the same
  five cells.
- `qa_gates.py`: **ROW** and **DNR** pass; **D1** and **TNS** report no museum hits (the
  remaining failures are ch09 / ch11 / index.html, other agents' pages).
- `museum.html` parses with balanced tags after the 14-block restructure (checked
  programmatically, zero mismatches, zero unclosed).
- Grepped: no `8.0×10⁻⁵` outside the erratum; no `26`/`twenty-six` do-not-re-try count anywhere;
  no `1.7–2.3%`; no `≈10⁻⁸⁶`.

## Note on expA-M3's search wording

The review's acceptance wording is *"search `starvation` returns Exhibit 12's rows"*. Under the
do-not-re-try filter it now returns **#52** (previously zero rows, against a badge that said
otherwise); **#41**'s source text says *"information-starved"*, not *"starvation"*, so it answers
to `starv`/`starved`. Both rows are now flagged, which is the defect the review found; the search
was **not** stemmed to force a literal match, because the ledger cells are quoted verbatim and
fuzzy-matching them would be a second, quieter kind of edit. §7's browser now also searches the
row **id**, so `#41` finds it directly.
