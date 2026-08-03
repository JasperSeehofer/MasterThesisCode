# Pedagogy review — *A Dark Siren Discovery Book*

**Reviewer:** pedagogy / interactive-learning-design
**Scope:** all 14 pages (`index`, `ch00`–`ch11`, `museum`), audited against
`book/design/BOOK_PEDAGOGY.md` (Parts 1–6) and against instructional-design best practice.
**Method:** read every page's extracted narrative; traced `js/book.js` (`predictReveal`,
`predictValue`, `getPrediction`, `ledger`, `persona`, `biasRail`, `SYMBOLS`) and every page's
inline scripts; programmatic measurements of word budgets, answer↔body n-gram overlap,
trap/interlude placement, dossier continuity, and prediction-id plumbing.
**Not audited:** the numeric disputes in `BUILD_REPORT.md` §5 (out of scope by instruction;
I flag only *policy inconsistency* in how the book presents one of them, never the value).

**Headline.** The discovery arc is real and, chapter by chapter, unusually well executed —
this is the best-argued piece of physics pedagogy I have reviewed in a while. The problems are
almost all *delivery* problems on beats the design got right: a chapter deck that prints the
answer to its own predict-lock, a running-example card that contradicts itself, a payoff
mechanic that renders an internal slug, an instrument that advertises behaviour it does not
have, and a self-check set written for a book half this length.

---

## [BLOCKER]

### B1 — `ch08-mass-channel.html`, chapter deck vs `#ch08-predict-1`: the chapter prints the answer to its own predict-lock, one screen above the lock

**Observed.** The `<h1>` deck reads *"The mass channel should sharpen everything — and the
bias goes to +0.077."* Roughly 40 lines later, the first widget says *"Commit before you
scroll"* and offers `Sharper, same centre / Barely moves / Moves the centre`. The reveal then
announces "+0.077". The reader has already been given +0.077 by the page title.

**Expected.** BOOK_PEDAGOGY §2.2 B2 names this exact beat as the book's best predict payoff
("the mass channel adds information — will the bias go up, down, or stay?") and §4.2 rule 2
requires the reveal to be *locked*. A lock is worthless when the answer is above the fold. The
deck lines are the spec's **Part 5 dashboard** — an internal build table for the writer — which
has been transcribed onto the reader-facing page.

**Aggravating.** `index.html` §"The journey" repeats it verbatim ("*the mass channel should
sharpen everything, and the bias goes to +0.077*"), and `ch02` Trap 2.B hands over the same
result plus a signpost ("*10 of 10 runs … mean pull of +4.04 … and it is Chapter 8's cold
open*") together with an I2.2 preset button literally labelled *"set b to the recorded 2D
offset"* which dials +0.077. By the time Mara reaches Ch 8, she has met the answer three times.

**Suggested fix.** Ch 4 already solved this problem and should be the template: its deck also
gives away the rail, so the chapter **inverted the beat** — it shows the rail in the cold open
and then asks the reader to predict where the *repaired* estimator lands (`#ch04-predict-1`),
which is both spoiler-proof and a better question (0.740, not 0.730). Do the same in Ch 8:
- Deck → *"The mass channel should sharpen everything. Watch what it does instead."*
- Index blurb → the failure, not the number: *"…and the answer moves the wrong way."*
- Trap 2.B → keep the mechanism (coherent tilt ⇒ pull grows as √N), drop the +4.04 / +0.077
  figures and the "Chapter 8's cold open" signpost; replace with a bare `⏭ Ch 8`.
- Same treatment for Trap 2.A, which currently delivers exhibit #49a's full verdict (see M8).
- Ch 7's deck (*"does not blur the answer — it moves it"*) collapses `#ch07-predict-1` from a
  3-way to a 2-way; soften to *"…and the central value is not the safe choice"*.

---

### B2 — EMRI-889's dossier contradicts itself across chapters: 5 of 11 cards print the disputed σ_dL/dL unflagged

**Observed.** The running example's headline precision number changes chapter to chapter:

| page | dossier row | flagged? |
|---|---|---|
| ch02 | `σ_dL/d_L = 9.0×10⁻⁴ (0.090%)` + "both readings are carried" | yes (F-ch02-1) |
| ch03 | `σ_dL = 7.98×10⁻⁵ Gpc … i.e. σ_dL/d_L = 8.98×10⁻⁴` | yes |
| ch06 | `→ as a fraction, σ_dL/d_L = 9.0×10⁻⁴ (see the flag in §4)` | yes |
| ch07 | computed live + a dedicated `ch07-flag` provenance note | yes |
| **ch04** | `d_L 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)` | **no** |
| **ch05** | `d_L 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)` | **no** |
| **ch08** | `d_L 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)` | **no** |
| **ch10** | `d_L 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)` | **no** |
| **ch11** | `d_L 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)` | **no** |

**Expected.** Two contracts are broken at once. (i) BUILD_REPORT §5.1 item 1 states the
resolution policy — *"Chapters print both values everywhere"* — and the reviewer's guide asks
explicitly to "verify no page quietly prefers one". Five dossiers quietly prefer one, and it is
the one four other chapters have already shown to be a label slip. (ii) B4 (the dossier) is the
book's continuity spine; its entire pedagogical function is that the reader can watch **one
object** accumulate facts. An object whose first row silently oscillates between two values a
factor 11.25 apart is not a continuity device, it is a proofreading hazard — and it lands in a
book whose thesis is *"a measurement is a number plus a stated list of the ways it could be
wrong."* Ch 1's own adjudicator block says the page "prints both and picks neither"; five later
pages pick one.

**Suggested fix.** Make the dossier row a single shared string emitted by the generators —
`σ_dL = 7.98×10⁻⁵ Gpc ⇒ σ_dL/d_L = 8.98×10⁻⁴ (spec quotes 8.0×10⁻⁵ as a fraction — flag F1)` —
and use it in all eleven cards. If the dossier row must stay short, use
`σ_dL/d_L = 8.98×10⁻⁴ ⚑` with the flag on the badge. Do not resolve the dispute; resolve the
*inconsistency*.

---

### B3 — `ch11-honest-state.html#ch11-recall`: the book's showcase payoff renders an internal slug, and the slug gives the answer away

**Observed.** `ch03-which-galaxy.html:286-288` stores the host guess with
`data-predict="impostor" | "host" | "cannot-tell"`. `ch11-honest-state.html:1936` reads it and
renders:

> *"In Chapter 3 you put a marker on the galaxy you thought hosted this event (**impostor**)."*

Three failures in one sentence: (a) the reader sees a developer slug, not their choice
("A — the spectroscopic one"); (b) the slug **states the verdict** — a reader who picked A is
told "impostor", a reader who picked B is told "host" — so the recall does the reader's
reasoning for them at the exact moment the beat is supposed to make them own it; (c) *"you put
a marker on the galaxy"* mis-describes the delivered interaction, which is a three-button
choice, not a marker (the phrasing is inherited from the spec's I3.1, which was re-designed).

**Expected.** BOOK_PEDAGOGY §2.2 B2: *"re-surfacing their Ch 3 guess in Ch 11 is one of the
strongest devices available."* The device must echo the reader's own words and let the
consequence land.

**Suggested fix.** Store a human label alongside the slug (`data-predict-label="A — the
spectroscopic one"`), or map in Ch 11:
`{impostor:"A, the spectroscopic galaxy", host:"B, the photometric one", "cannot-tell":"neither — you said you could not tell"}`.
Rewrite the sentence for the delivered interaction: *"In Chapter 3 you were shown two galaxies
and asked which hosted this event; you chose **A, the spectroscopic one**."* Then keep the
existing (excellent) follow-on about weights, unchanged.

---

### B4 — BW3 "Has this been tried?" does not volunteer anything, and two pages tell the reader it does

**Observed.** The shipped instrument (`book.js` `Book.ledger.init`) injects a **collapsed
`<details>` search box immediately before the provenance panel** — i.e. at the very bottom of
the page — plus a row of `#id` chips. It never fires on sandbox state. Concretely:

- `ch04-loud-half.html:516-518` tags the three-way denominator switch
  `data-hypothesis="9"` on `local window (recorded)` — the historically shipped defect. Nothing
  in ch04's inline script reads `data-hypothesis`; flipping to the dead configuration volunteers
  nothing.
- `ch07-redshift.html:564, :1087` write full verdict strings into `data-hypothesis-verdict=…`.
  A repo-wide grep finds **no reader anywhere** — the attribute is dead data.
- `ch08`, `ch09`, `ch10`, `ch11` carry **zero** `data-hypothesis` tags, so even the bottom-of-page
  chip seed is empty on the four chapters whose sandboxes are the most re-litigable
  (I8.2 reparametrization walk, I9.2 consistency bench, I10.1/I10.2, I11.1).

Meanwhile `museum.html` §7 tells the reader the ledger backs *"the book-wide **Has this been
tried?** instrument, **which volunteers a verdict whenever a sandbox anywhere in this book is
dragged into a configuration the project has already killed**"*, and `index.html` promises
*"If your clever fix has a name, a date, and an obituary, the book will tell you."*

**Expected.** BOOK_PEDAGOGY §4.0 BW3 is explicit that the *confrontation* is the whole value:
*"Lost if static: the confrontation. A read-only ledger is a document; a ledger that interrupts
you is a research supervisor."* As shipped, BW3 is precisely the document the spec says is
worthless. Separately — and this is why it is a blocker rather than a major — a book whose
central claim is that it never lets a dead hypothesis look alive currently **describes an
instrument it does not have**, on two pages, in the present tense.

I accept the integrator's reasoning (BUILD_REPORT §2) that a *second* auto-reveal would
double-report and pre-empt predict-locks. That argues for scoping, not for shipping the claim.

**Suggested fix (cheapest first).**
1. **Correct the two claims today.** Museum §7 → *"…backs the **Has this been tried?** search
   box carried on every chapter page, and the dead-end verdicts the sandboxes hard-code."*
   Index → *"…every sandbox page carries a search box, and every sandbox that can reach a dead
   configuration says so where you reach it."*
2. **Make the tags live, once, in `book.js`.** On any control carrying `data-hypothesis`,
   render an inline `⚖ #N — <verdict>` chip *inside the widget* when that control becomes the
   active state. Chapters that already hard-code a verdict for the same state opt out with
   `data-hypothesis-verdict="inline"`. That is ~15 lines and it turns 10 dead attributes into
   the instrument the book advertises.
3. **Tag ch08–ch11's sandboxes** (I8.2 → #71/#72/#89/HB; I9.2 → #61; I10.x → #49a;
   I11.1 → #61/#64) so the chip seed is non-empty where re-litigation risk is highest.

---

## [MAJOR]

### M1 — 21 of 61 self-check answers restate the chapter body; design rule (a) says "no recall questions"

**Observed.** Measured 5-gram overlap between each hidden answer and its own chapter's body
text (body = everything before the `Self-check` heading):

| chapter | questions with >22% verbatim overlap |
|---|---|
| ch11 | **5 of 6** — Q11.1 30%, Q11.2 45%, Q11.4 25%, Q11.5 37%, Q11.6 32% |
| ch07 | 3 of 6 — Q7.1 **50%**, Q7.4 28%, Q7.5 **62%** |
| ch08 | 3 of 6 — Q8.1 **44%**, Q8.2 28%, Q8.6 28% |
| ch02 | 3 of 5 — Q2.1 32%, Q2.3 **40%**, Q2.4 28% |
| ch05 | 2 of 5 — Q5.3 25%, Q5.4 27% |
| ch04, ch06, ch09, ch10 | 1 each — Q4.2 27%, Q6.1 32%, Q9.4 28%, Q10.3 28% |

Cleanest cases: Q2.3 asks the reader to *"distinguish, with a one-line operational test each,
a biased / noisy / mis-calibrated estimator"* — §3's `ch02-defbox` states all three definitions
*and their operational tests* verbatim, in a bordered box, thirty lines earlier. Q0.1 and Q0.2
are each answered by a named section of Ch 0 (*"So the honest answer to 'how precise must it
be' is 1–2%…"*, and the §"Why the systematics must be uncorrelated" heading).

**Diagnosis — this is a length side-effect, not sloppiness.** The question set was written
against a spec in which Ch 0 was ~1,200 words and Ch 2 ~2,200. The delivered chapters make
every argument fully in the body (see M5), so questions designed to *force* the argument now
merely *recall* it. Good questions were made into recall questions by good prose.

**Suggested fix.** Do not rewrite 21 questions. Change the *relationship*: for each flagged
question, either
- (a) move the body passage that answers it into a `<details class="answer">` **after** the
  question ("Show the argument"), leaving the body with the claim and not the derivation — this
  preserves Tomas/Examiner access while restoring the gap for Mara; or
- (b) re-aim the question one step past what the body says. Q2.3 → *"Which of the three would a
  single run's `combined_posterior.json` let you diagnose, and which two does it structurally
  cannot?"* (the answer — scatter and calibration need realizations; a single run diagnoses
  neither — is in the chapter's adjudicator block but never stated as such). Q11.2 → *"Construct
  a different statistic of the same 76 events under which the rail would look like noise, and
  say why the project did not use it."*

### M2 — Q7.1 and Q8.1 are verbatim re-asks of in-chapter predict widgets, still framed "predict before the reveal"

**Observed.**
- `ch07#ch07-predict-1` (cold open, line 127) offers `(a) It widens symmetrically / (b) It
  shifts low / (c) It shifts high`. Q7.1, ~1,000 lines later, reads *"**Predict, before the
  reveal**: … Does the H₀ posterior (a) widen symmetrically, (b) shift low, (c) shift high?"*
  Same three options, same wording, after §2 has explained the mechanism and §6 has extended it.
- `ch08#ch08-predict-2` (line 381) offers `Symmetric / One-sided low / One-sided high`. Q8.1
  reads *"**Predict** the shape of the resulting mass window's rejection: symmetric, one-sided
  low, or one-sided high?"* — identical, after §2–§4 have measured 193 vs 1.

**Expected.** B2 requires prediction *before* evidence; B7 requires the self-check to be a
check, not a replay. A "predict" instruction issued after the reveal teaches the reader that
the book's prediction framing is decorative.

**Suggested fix.** Drop the "predict" framing and convert both to the transfer form the
chapter has now earned. Q7.1 → *"You measured Δh = −Cσ_z² with C ≈ 17–20. Without re-reading
§2: what feature of the volume prior sets the **sign**, and what would have to be true of the
prior for a symmetric widening to be the right answer?"* Q8.1 → *"σ_Mz/M_z ≈ 10⁻⁴ makes the
window's upper leg vacuous. Name a source population for which it would not be, and predict
whether the tilt survives."*

### M3 — 13 of 26 misconception traps sit *after* the self-check, including the one Ch 7 is built around

**Observed** (position measured as a fraction of body length, self-check as the cut):

| chapter | traps in-line | traps **after** the self-check |
|---|---|---|
| ch01, ch03, ch04, ch09, ch11 | all | — |
| ch02 | 3 | — |
| ch05 | 0 | **2** (5.A completion-as-fudge, 5.B w_G-as-completeness) |
| ch06 | 1 | **2** (6.A sky⟂distance, 6.B numerical factors) |
| **ch07** | **0** | **2** (7.A "bigger σ_z just means wider", 7.B "more deconvolution is better") |
| ch08 | 1 | 2 |
| ch10 | 1 | 2 |
| museum | 0 | 2 |

**Expected.** BOOK_PEDAGOGY §2.2 **B6: "1–2 per chapter, *in-line*"**, *"stated as a plausible
sentence the reader is likely to be thinking"*. A trap's whole mechanism is temporal — it must
fire while the misconception is forming.

Ch 7 is the acute case. The spec says of the symmetric-widening belief: *"Symmetric widening is
what almost every reader predicts, and it is **the trap this chapter is built around**."* In the
delivered chapter, Trap 7.A appears at the very bottom of a 6,750-word page, after the
self-check, after the interlude — roughly forty minutes of reading after the reader formed the
belief in the cold open, and about twenty minutes after §2 has already dismantled it. It lands
as a summary, not a trap. Trap 5.B ("w_G is roughly the catalogue's completeness fraction") has
the same problem: the misconception is created and correctly attacked in §2's *"First: 12%. Not
5%…"*, but the labelled trap sits 800 lines later.

**Suggested fix.** Relocate: 7.A → immediately after the I7.1 stage-1 reveal in §2. 7.B →
immediately before §6's twist (it is the sentence the reader thinks *entering* §6). 5.B → §2,
right after the "12%. Not 5%" paragraph. 5.A → §3, where `L^comp` is introduced. 6.A → §4/§5
where the correlations are shown. 6.B → §2 (the dt² section). 8.A/8.B and 10.A/10.B → the
sections that create them. Keeping a trap in-line costs nothing and is the difference between
inoculation and a footnote.

### M4 — The Symbol Passport (BW2) is chapter-blind and leaks three later chapters' climaxes on hover

**Observed.** `Book.SYMBOLS` is global and carries a `note` field surfaced on every hover:

- `eps` (σ_z/z): *"the C7 variable; **rail threshold quoted at 0.256**"* + *"**C7 is a live
  FINDING** — see Ch 7 §6 / Ch 11"*. Both `data-term="eps"` tags in `ch07-redshift.html` sit at
  byte offsets ~4.4k and ~5.4k — the deck and §1 — while §6, whose entire dramatic function is
  to *arrive at* 0.256, begins at offset ~37.5k. A reader who hovers the chapter's own subject
  symbol on the first screen is handed the last screen's number.
- `wG`: *"mixture weight β_G/D — **ESTIMAND-DEPENDENT; always name the mode**"* +
  *"C9 is a live FINDING — see Ch 9 §6"*. Both `data-term="wG"` tags are in **ch05**.
  "Estimand-dependent" is precisely Ch 9 §4's reveal (*"two different estimands, neither
  corrupt"*), and Ch 5's own text is careful to stay at its rung.

**Expected.** BOOK_PEDAGOGY §1.3's rung-guard rule and §2.2 B2's discovery contract. The
BUILD_REPORT reviewer's guide asks this question directly ("does the caveat land before Ch 9
confuses you, or does it spoil Ch 9's reveal?"). Answer: it spoils it.

**Suggested fix.** `manifest.js` already knows chapter order. Gate the `note` and any
climax-bearing clause on the current page's index: before the establishing chapter render
*"status: established in Ch 9 — not yet used here"*; from that chapter on, render the full
caveat. Definition + units + code site (the passport's actual job for Mara and Tomas) stay
unconditional. ~10 lines in `Book.passport`.

### M5 — Cognitive load: every chapter's main column is 1.4–2.8× the design's own word budget

**Observed.** Main narrative column = `<main>` minus everything inside `<details>` (so folds,
answers and numbers-views are already excluded), against `BOOK_DESIGN.md`'s per-chapter budget:

| page | main-column words | budget | ratio |
|---|---:|---:|---:|
| museum | 6,981 | 2,500 | **2.79×** |
| ch06 | 5,933 | 2,500 | **2.37×** |
| ch00 | 2,801 | 1,200 | **2.33×** |
| ch11 | 6,826 | 3,000 | **2.28×** |
| ch02 | 4,987 | 2,200 | **2.27×** |
| ch10 | 5,996 | 2,800 | 2.14× |
| ch09 | 5,456 | 2,800 | 1.95× |
| ch03 | 4,842 | 2,500 | 1.94× |
| ch05 | 5,886 | 3,200 | 1.84× |
| ch08 | 6,777 | 3,700 | 1.83× |
| ch01 | 3,375 | 2,000 | 1.69× |
| ch07 | 5,878 | 3,500 | 1.68× |
| ch04 | 4,014 | 2,900 | 1.38× |

Total ≈ **83,700 words** ≈ 6 hours of reading before a single control is touched. Widget count
tells the same story: **59 widget blocks, 47 with controls**, against the spec's 24 chapter
interactives + 2 museum — roughly double, in a Part 4 that opens *"the book should have fewer,
better interactives, not more."*

Pure `<p>` prose is much closer to budget (0.81–1.58×). The overrun is therefore almost
entirely **lists, captions, readout legends, verification asides and provenance narration
promoted into the main column**. Representative sample, `ch07` §6 main column:

> *"…taking ln L^cat_i at h = 0.86 and h = 0.73 straight out of
> real_r1/diagnostics/event_likelihoods.csv and adding Δ ln Σ_glob = +0.027597 reproduces the
> stored per-event tilts to 7.2×10⁻¹⁶ (gate G4 in gen_ch07.py)."*

That is a *build-QA* sentence at rung L7, addressed to nobody in the persona set except the
Examiner — who has a `num-view` fold and a provenance panel for exactly this.

**Where the load hurts most.** Ch 6 is the arc's *designed* pause: Ch 5 plants the splinter and
says "hold that thought for six chapters," and Ch 6 is the methods detour before Ch 7 resumes.
At 2.37× budget (≈34 min) the designed pause becomes a stall — the single place in the book
where a reader is most likely to put it down. Ch 0 at 2.33× is the second worst place to be
long: a prologue's only job is to get the reader to Ch 1.

**Suggested fix (concrete, ~12k words of relocation, no content lost).**
- **Ch 6:** move §4.1 ("The fourth coordinate, and why it is nearly degenerate") and the
  numerical frame-audit detail of §5 into `details.gw-reader`. Keep §5's *story* (0.860 → 0.730,
  host recovery 31→38) in the column; the matrix bookkeeping does not belong at L6-first-contact.
- **Ch 0:** cut to the two figures + the arbitration budget + the contract. The
  time-delay-cosmography 2%→8% paragraph is the best thing on the page — keep it. The
  step-by-step σ_tot algebra of §2 can live in a `num-view` since I0.1 recomputes it live.
- **Ch 11:** §4's per-claim recitation duplicates §1's scoreboard; make §4's C7/C8/C9 blocks
  collapse to a one-line status with the detail folded, since Ch 7/8/9 already argued each one.
- **Museum:** 14 exhibits × six fields in one column is a reference work, not a read. Ship the
  five interlude exhibits expanded and the other nine collapsed by default (`<details>` on the
  exhibit, summary = hypothesis + verdict badge). The room stays complete; the visit gets a path.
- **Everywhere:** any sentence containing a gate tolerance (`7.2×10⁻¹⁶`, `bit-equal`,
  `refuses to write a file if it drifts`) belongs in the provenance panel or a `num-view`.
  There are ~40 of these and they are collectively worth several thousand main-column words.

### M6 — `ch02-bayes.html` §1 opens the selection denominator at rung L2, three rungs early

**Observed.** §1's third "design decision" is *"Why there is no extra β(h)^N"*, followed by a
`voice-derivation` box: *"Conditioning on the observed number of detections makes the β(h)^N
factor of the unconditional (Poisson) likelihood cancel against the rate prior. Applying
−N log D(h) again on top of per-event likelihoods that already carry it is a double count —
and the project shipped exactly that for a while, then measured what removing it was worth: it
eliminated a +0.020…+0.025 shift in h (ledger #20)."*

**Expected.** BOOK_PEDAGOGY §1.3, rung-guard rule: *"A chapter may not use a tool from a higher
rung even in an aside."* `D(h)`, the selection integral, and the meaning of "conditional on
detection" are **L4** — Ch 4's entire discovery. At L2 the reader has no `D(h)`, no `β`, no
selection concept and no way to evaluate a double-count argument; the passage can only be taken
on authority, which is the one thing this book asks readers never to do. It also spends part of
Ch 4's payoff two chapters early.

**Suggested fix.** Keep one sentence in the column — *"Each L_i is already conditional on
detection, so the joint posterior is a plain product with no extra population factor on top;
why that is a theorem and not a convenience is Chapter 4 ⏭ Ch 4."* — and move the
Mandel/Farr/Gair box, the double-count history and ledger #20 into `details.gw-reader` (where
Tomas, who *does* have the rung, will find it) or forward into Ch 4 §4 "Counted exactly once",
which is where it actually belongs.

### M7 — The bias rail is not cumulative; it *loses* rows at Ch 6 and again at Ch 11

**Observed.** `Book.biasRail` entries per page:

| page | rows |
|---|---|
| ch04 | 2 — `cat-only, no D(h)` −0.178 · `full-volume D(h)` 0.000 |
| ch05 | 3 — + `two branches disagree (0.86 / 0.64) — unresolved` |
| **ch06** | **2** — ch05's third row is gone |
| ch07 | 2 + a live row set by I7.1 |
| ch09 | 4 — + `volume_deconv` −0.002 · `2D, mass channel` +0.077 |
| ch10 | 4 |
| **ch11** | **2** — `1D (contingent)` 0.000 · `2D (+0.077)` +0.077 |

**Expected.** BW1 is called *"the book's single most important structural device"*, and its AHA
is *"the estimator is an object under construction with a live scorecard, **and the scorecard
does not monotonically improve**"*. That AHA is only legible if the history is still on screen.
As delivered, the reader watches the rail grow to four rows by Ch 9, then arrives at the honesty
chapter — the one place where the whole build history is the argument — and finds the rail
shrunk to two rows with the −0.178 origin deleted. Ch 6 has the same problem in miniature: a
reader stepping Ch 5 → Ch 6 sees the rail go *backwards*.

**Suggested fix.** Define the canonical row list once in `manifest.js` (`BOOK_BIAS_ROWS`) with a
`from_chapter` field; each page renders every row with `from_chapter ≤ n`, plus its own live/
amber additions. Ch 11 then shows all five rows with the amber pips on top — which is exactly
the "honest amber state" the spec describes, and it is currently the only chapter that cannot
show it.

### M8 — Ch 9 §3–§4 introduces six symbols in two boxes with no physical picture first

**Observed.** Inside two consecutive `voice-derivation` blocks the reader meets
`n_G(z,Ω)`, `n̄_gal`, `Σ_global`, `n̄_w`, `D_•`, `D_gen`, `Ŵ_cat`, `V_f(h)`, and a
`Bernoulli(F)` channel split — nine new objects, all in equations, none introduced with a
picture or units beforehand.

**Expected.** §1.1's design consequence for the primary persona: *"Every new symbol is
introduced with units and a physical picture **before** it appears in an equation."* This is the
one place in the book where that rule is broken at scale, and the BUILD_REPORT flagged it as a
danger zone in advance. (By contrast Ch 5 §2 — the other flagged zone — handles it beautifully;
see P8.)

**Suggested fix.** Precede the Option-A box with the picture, which is a genuinely intuitive one
and takes three sentences: *"The catalogue is a list of galaxies. The selection integrals are
volumes. To multiply them you need an exchange rate — how many catalogue rows per Mpc³ — and
Option A is the claim that you never have to know it, because it cancels."* Then define
`Σ_global` in words ("what the catalogue channel expects to detect") before the symbol appears.
Fold `n̄_w / D_gen / Ŵ_cat / V_f` — the mode-by-mode bookkeeping — into `details.gw-reader`;
Mara needs *"two modes, two estimands, both internally consistent"*, which §4's prose already
delivers well.

### M9 — Ledger #49a and claim C9 are each spent three to five times before reaching their designated exhibit

**Observed.**
- **#49a** (the H₀-independent estimator, MAP 0.86 for every injected truth) appears in ch00's
  Trap 0.A, ch02's Trap 2.A *with its full verdict and mechanism*, ch09 §3's de-rail matrix
  commentary, ch10's designated interlude, and the museum. Five exposures; the interlude that
  BOOK_PEDAGOGY B5 places after Ch 10 to *inoculate against a mistake Ch 11 invites* arrives as
  the reader's fourth telling.
- **C9 / z = −11.86** appears in ch05 (Trap 5.B and §2), ch08 (Q8.6's answer and Trap 8.A) and
  only then in ch09 §6, its home. By then the reveal *"the generator and the estimator describe
  different universes"* is confirming something the reader has been told twice.

**Expected.** B5's placement logic and the discovery contract: a chapter's reveal must be a
reveal. Some forward-referencing is right and the book does it well elsewhere (Ch 2's *"Note it
and move on — you are not equipped to interpret it yet"* is the model). What is wrong here is
forward-referencing **with the punchline attached**.

**Suggested fix.** Adopt Ch 2's own discipline as a rule: a forward reference names the
*phenomenon* and the chapter, never the number or the verdict. Trap 2.A → *"An estimator can be
stable, reproducible and carry zero information about the parameter it reports. Chapter 10 shows
you one. ⏭ Ch 10"*. Trap 5.B → keep "w_G is not the completeness fraction" and the 0.1215 vs
4.8% contrast (which Ch 5 legitimately owns), drop the binomial z and the 2.3–2.5× framing,
which are Ch 9's measurement.

---

## [MINOR]

- **m1 — Q9.1 is recall.** *"Ω_m = 0.2726: bug or design choice, and how would you tell?"* is
  fully answered by **Ch 1 §4** ("the first example in this book of the difference between a bug
  and a decision") and then again by Ch 9 §1.2, which reopens with the identical framing
  ("every reader's first instinct is that it is a bug"). BOOK_DESIGN assigns the topic to both
  chapters, so this is a spec-level duplication — but the *question* should move with the first
  telling, or Ch 9 should ask the harder version: *"Name a quantity in this pipeline for which
  the same argument would **not** license a non-Planck value, and say why."*

- **m2 — Q1.4 is the only broken link in the transfer chain.** Q0.3→Ch1, Q2.5→Ch3, Q3.6→Ch4,
  Q4.5→Ch5, Q5.5→Ch6, Q6.5→Ch7, Q7.6→Ch8, Q8.6→Ch9, Q9.5→Ch10, Q10.5→Ch11 all land exactly on
  the next chapter's opening failure — a genuinely impressive chain. Q1.4 forwards to
  "Chapters 3, 5 and 7", skipping Ch 2 entirely. Ch 1 §5's closing prose *does* set up Ch 2
  ("what 'infer' even means…"), so the fix is one clause in the answer: *"…and before any of
  that you need a machine that turns 1,588 distances into a statement about h, which is
  Chapter 2."*

- **m3 — Persona nudges are absent.** §1.2 requires Mara to be *"nudged to [open Tomas's
  stratum] at three specific points (Ch 6 Fisher, Ch 8 mass measure, Ch 11 adjudication)"*.
  No page contains such a nudge; the folds are unlabelled beyond `▸ For the GW reader`. Add one
  inline sentence at each of the three sites (*"If you have met Fisher matrices before, the GW
  stratum below is where the 14-parameter version lives."*). Also: `Book.persona.apply("mara")`
  force-closes every `gw-reader`/`num-view` the reader had opened — defensible as a reset, but
  the spec's *"it never hides content"* would be better honoured by only ever opening.

- **m4 — Interludes sit past the self-check.** All five (ch04, ch05, ch07, ch08, ch10) are
  correctly placed *between* chapters per B5 — good — but that puts them in the highest-
  abandonment zone on the page, after the reader's natural stopping point. A one-line bridge
  above the self-check (*"One more thing before Chapter 8 — a fix that failed for two reasons at
  once."*) costs nothing and recovers the inoculation.

- **m5 — `num-view` coverage is short of the accessibility contract.** §4.2 principle 7:
  *"every widget has a 'show me the numbers' table view."* ch10 has 2 for 4 widgets, ch11 3 for
  5, ch04/ch06 3 for 4. Some of those are predict-rows that need none; the real gaps are
  ch10's I10.2 and ch11's I11.1, both of which are argued *from* a number the reader cannot see
  tabulated.

- **m6 — The museum has no per-chapter backlinks.** BUILD_REPORT gap #6 rates this "acceptable";
  pedagogically it is a little worse than that. An interlude ejects the reader from the arc mid-
  build and offers no labelled return, so the museum reads as a terminus. One line per exhibit —
  *"You arrived here from Chapter 7 §6 · ← back"* — closes the loop.

- **m7 — The index's chapter list is the spec's internal dashboard.** `index.html` §"The
  journey" reproduces BOOK_PEDAGOGY Part 5's discovery statements verbatim, including two that
  are answers rather than failures (Ch 4's *"it runs to the edge of the prior"*, Ch 8's
  *"+0.077"*). Ch 4 survives it because the chapter re-designed its beat (see P2); Ch 8 does not
  (see B1). Rewrite both blurbs as failures.

- **m8 — Trap "The completion term pulls the answer up" appears identically in ch08 (66% of
  body) and ch11 (68%).** Given Writing Rule 5 makes retiring that sentence binding, the
  repetition is arguably deliberate; if so, say so in Ch 11 (*"this is the one trap the book
  states twice, because C10 exists to retire it"*) so it reads as insistence rather than
  copy-paste.

---

## [PRAISE] — what works, and must survive any edit

- **P1 — `ch03` §1's opening predict is the best interactive question in the book.** The reader
  is asked how many galaxies are in a typical localization ball and offered
  `<10 / dozens / thousands / tens of thousands`; the reveal is *"**All four answers are right,
  for different events** — and that is the finding"* (median 1,616 → 12 after the z-window;
  worst 431,670; 552 with none; EMRI-889 has three). It refuses the question's false premise and
  converts a would-be recall item into a lesson about distributions. This is a model the rest of
  the book could copy in two or three more places.

- **P2 — `ch04` inverted its predict beat to defeat its own spoiler.** The deck gives the rail
  away, so the chapter shows the rail first and asks the reader to predict where the *repaired*
  estimator lands (`#ch04-predict-1`, numeric slider). The reveal — 0.740, not 0.730, plus
  *"if you put the marker exactly on 0.730 you made the assumption this book is built to
  attack"* — is a better beat than the specified one. Ch 8 should do exactly this (B1).

- **P3 — `ch01` Q1.2's editor's note.** Faced with a disputed number in its own question stem,
  the answer redoes the estimate both ways, shows that the corrected value lands within a factor
  4 of the measured per-event budget while the spec value is a factor 47 away, and concludes
  *"the order-of-magnitude check you were asked to do is what catches the units slip — which is
  the real lesson."* Turning a build defect into the pedagogical payload is exemplary, and it is
  the single best argument that this book's honesty posture is not decoration.

- **P4 — Ch 2's explicit rung guard.** After the loudest-first excursion to 0.86: *"That is a
  real, measured feature of this run, it has a name, and it is the subject of Chapter 5. **Note
  it and move on — you are not equipped to interpret it yet, and being handed the interpretation
  now would cost you the discovery.**"* Naming the pedagogical contract out loud, to the reader,
  is rare and it works. It is also the exact discipline M9 asks the traps to adopt.

- **P5 — The dossier arc genuinely runs.** Ch 1 opens it with `z: unknown`; Ch 3 adds the host
  *and the impostor* with their rate weights; Ch 4 adds "it could have been seen 71× further
  away"; Ch 5 adds dark 606 as a counterpart and the 50-million-fold branch asymmetry; Ch 6 adds
  the sky ellipse and correlations; Ch 9 adds "how it was born"; Ch 11 closes it with
  *"what this chapter added: **a warning label**"* and the +1.98 → −2.04 → −3.30 swing. The
  spec promised "the whole book in one object" and the delivery earns it. (Fix B2 so the card
  does not undercut itself.)

- **P6 — Ch 2's lurch is located to a single event.** *"222 events have accumulated and the
  width has moved 13%. The 223rd is EMRI-889, and the width halves in one step, from one galaxy,
  arriving at no distinguished place in the queue."* This is the cleanest possible demonstration
  that N is not the currency, and it is only possible because the running example was chosen
  well.

- **P7 — Ch 11 §3 is a textbook payoff.** The `⏮ Ch 5` back-reference, both readings restated
  in the reader's terms, the leverage identity derived from stationarity, the predict-lock, and
  a reveal that contrasts 0.024904 (realistic) against 0.000006 (idealized) — *"That contrast is
  the measurement."* Six chapters of deferred gratification, discharged in about 400 words.

- **P8 — Ch 5 §2 holds its rung under maximum temptation.** *"Everything about w_G that matters
  **at this rung** is measurable, and it is measured… First: 12%. Not 5%… the gap between them
  is a live, adjudicated problem — Chapter 9 measures it and Chapter 11 puts it on the board."*
  The BUILD_REPORT flagged this as a rung-violation danger zone; the chapter converted the
  danger into a plant. Ch 9 §3–§4 (M8) should be rewritten to this standard.

- **P9 — Ch 7 §6's "Both sides, and who decides".** RATIFIED G2b and MEASURED C7 set
  side by side in their own typographic voices, with the historical record noted as cutting
  against the proposed fix, and *"The book therefore does not resolve this, because the project
  has not."* This is what Writing Rule 4 looks like when it is honoured rather than announced.

- **P10 — Ch 11's "The questions with no answer key."** Five genuinely open questions, no
  disclosure control, and *"adding one would be the single most dishonest thing this book could
  do."* This fully answers BUILD_REPORT gap #4: Q11.6 legitimately keeps a model answer (the
  discipline answer *is* known) while the unknowable questions are separated and left bare. Best
  ending available to this project.

- **P11 — Ch 6's cold open.** Four real events, their SNRs, sky areas and actual candidate
  counts (2 / 52 / 4,758 / 145) with the note *"Two events a factor 33 apart in loudness, and the
  candidate list differs by a factor of 3,555."* It converts "we need the Fisher matrix" from an
  assertion into a felt need, which is the hardest thing to do for a methods chapter.

- **P12 — The predict-lock implementation is correct and safe.** `.reveal{display:none}` until
  `.shown`; the class is only ever added by JS, so a no-JS reader is never locked out and gets a
  `<noscript>` static answer instead. I could not find a path to the reveal without committing,
  and I could not construct a dead lock. Exactly the right trade.

- **P13 — Ch 3 arms the bias rail with an `IntersectionObserver` on §5's failure callout.** The
  rail's first live entry appears at the precise moment the failure is planted. Small, and the
  right kind of interaction: state changes only because the reader moved.

- **P14 — Museum §1's rule for writing an exhibit.** *"The hypothesis — stated as it was
  believed at the time, in its strongest form. **If it sounds stupid, the exhibit is written
  badly.**"* Together with the standing venue-scoping rule stated up front, this is what stops
  the museum being a morality play, and it is the reason the room is worth visiting.

---

## Priority for the next pass

1. **B1** (Ch 8 deck) and **m7** (index blurbs) — one hour, restores the book's best beat.
2. **B2** (dossier σ_dL) — generator-side, mechanical, removes a self-contradiction from the
   continuity spine.
3. **B3** (Ch 11 recall slug) — ~10 lines, restores the showcase payoff.
4. **B4 step 1** (correct the two BW3 claims) — immediate; **B4 steps 2–3** when there is time.
5. **M3** (relocate 13 traps) — pure cut-and-paste, largest pedagogical gain per minute.
6. **M4** (chapter-gate the passport notes) and **M7** (cumulative rail) — both ~15 lines in
   `book.js` + `manifest.js`.
7. **M1/M2** (question set) and **M5** (load) — the largest jobs; M5's relocations should be
   done first, since roughly a third of M1's overlap problem dissolves once the derivations move
   into folds.
