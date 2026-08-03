# Student review — "Mara" (physics MSc, zero GW background)

**Reviewer persona:** physics MSc, comfortable with Bayes and undergrad GR, first contact
with gravitational-wave astronomy and with dark sirens.
**Method:** read `index.html` → `ch00` → `ch05` in order, deeply, committing a written
prediction at every predict-then-reveal *before* reading on and attempting every self-check
before opening its answer. Interactives simulated by reading `js/book.js`, each chapter's
inline script, and the corresponding `data/*.json` (I could not run a browser; where a claim
depends on rendering I say so). Then a ~10-minute skim of ch06–ch11 for the on-ramp.
**Date:** 2026-07-31. Read-only throughout; nothing outside this file was modified.

I have deliberately not re-litigated the numeric disputes already indexed in
`BUILD_REPORT.md` §5 — where I mention one, it is only because a *specific page* handles it
differently from its siblings, which is a presentation defect the flag index does not cover.

---

## Summary of my predictions vs. the reveals

| # | Page | Prediction I committed | Measured | Verdict |
|---|---|---|---|---|
| 1 | ch00 I0.1 | (d) no number of events works | (d), saturates at 2.50σ | ✅ right; reveal added the ceiling arithmetic |
| 2 | ch02 §4 | "a minority — ~100" | 24 events | ⚠️ **ungraded** — see BLOCKER-2 |
| 3 | ch03 census | "thousands" | median 1616 in ball / 12 after window | ⚠️ **ungraded**, and contradicts ch02 — see BLOCKER-1 |
| 4 | ch03 host | "Neither — you cannot tell" | "Neither" is correct; host is B | ✅ right; the *why* (5% in z is 5% in h) was new to me |
| 5 | ch04 §3 | MAP marker at 0.730 | 0.740 | ❌ **wrong, and the reveal named my exact error** — best beat in the book |
| 6 | ch05 §3 | "thousands" | ~5000× | ✅ right, but the AHA was already printed twice above the question — see MAJOR-4 |
| 7 | ch05 §5 | (b) in-cat near truth, dark biased | (c) 0.86 / 0.64, neither near truth | ❌ **wrong, and I understood why immediately** — the single best reveal in ch00–05 |

Two of seven reveals actually moved me. Two more were dead on arrival (spoiled or
ungraded). That ratio is the biggest single lever available to this book.

---

# [BLOCKER]

## BLOCKER-1 — ch02 tells me "tens of thousands of candidates" and marks me correct; ch03 measures 12, and ch02 was never amended

**Where:** `ch02-bayes.html:373`, `:829`, and the hidden answer to Q2.5 at `:918`;
against `ch03-which-galaxy.html` census reveal (median **1616** in the ball, **12** after
the redshift window) and `ch03-which-galaxy.html:889` (Q3.4).

**What I observed.** Ch 2 asserts, three separate times, that a typical event's
localization volume holds "tens of thousands" of candidate galaxies. Its transfer question
Q2.5 asks me to *predict* the number; I answered "tens of thousands" (because ch02 had
already told me twice), opened the answer, and was told: *"typically tens of thousands of
candidates."* Marked correct. I then turned the page and ch03's opening census measured the
median at **1616 in the ball and 12 after the z-window** — a factor ~1000 off the number ch02
had just graded me right on. EMRI-889, the book's running example, has **3 and 2**.

**What I expected.** The chapter I trusted three screens ago should not be refuted by the
next chapter without an on-page correction. `flags/ch03_FLAGS.md` F-ch03-2 shows the ch03
agent noticed exactly this, amended *its own* Q3.4, and scoped the slogan to the tail —
but ch02 kept the pedagogy's verbatim text (`BOOK_PEDAGOGY.md:696`) unamended. The result is
that the book's own graded answer key is wrong, in a book whose thesis is that graded answer
keys must be checkable.

**Why this is a BLOCKER and not a MINOR.** It is not a stylistic inconsistency; it is a
*hidden self-check answer that the book itself measures to be false one chapter later*.
Rubric D says answers must name the mechanism and carry provenance; this one carries neither
and is contradicted by chipped data.

**Suggested fix.** Amend ch02 in three places, keeping the mechanism intact:
- `:373` → "a discrete latent variable with anywhere from zero to hundreds of thousands of
  candidate values per event."
- `:829` (the failure-the-next-chapter box) → "For some events that volume contains tens of
  thousands of them; for others, three. Either way, not knowing which is the host is not an
  error bar — it is a sum."
- Q2.5's answer → "The host redshift; and a number of candidates that spans five orders of
  magnitude — the median event offers 12 after the redshift window, the 95th percentile 4891,
  and 552 of 1590 offer none at all `<chip>` ch03 census. The point of the question is that
  the answer is a *distribution*, and Chapter 3 measures it."

Also fix Q3.4's own answer (`ch03:889`), which still opens "**Thousands to tens of
thousands.**" and then measures 1616/12 in the very next sentence — the answer contradicts
itself inside one paragraph.

## BLOCKER-2 — ch03 and ch06 disagree about the production search radius (2σ vs 1.5σ), and therefore about how many galaxies are in EMRI-889's ball

**Where:** `ch03-which-galaxy.html:199` (ratified voice box: `r = n_σ√λ_max(Σ')`, **n_σ = 2**)
and `:239`, `:779` (radius 1.009′, solid angle 8.885×10⁻⁴ deg², "**3 galaxies** inside it")
vs `ch06-black-box.html:136-137` ("the **production** search radius
`1.5√λ_max(JΣ_skyJᵀ)`"), the four-event table there (889 → **2** galaxies in the ball,
ΔΩ = 3.29×10⁻⁴ deg²), `:795` ("search radius 0.757 arcmin"), and `:843-848` ("Production
passes n_σ = 1.5 `<chip>` bayesian_statistics.py:2820").

**What I observed.** Both chapters call their multiplier "the production" value, in ratified
/ chipped voice, and they differ. 0.757′ / 1.009′ = 0.75 = 1.5/2 exactly, so this is not a
rounding artefact. It changes the book's most-repeated concrete fact about its running
example from "three galaxies in the ball" to "two".

**What I checked.** In the source tree, `handler.get_possible_hosts_from_ball_tree` has
`sigma_multiplier: int = 2` as its *signature default*
(`galaxy_catalogue/handler.py:568`) and uses it at `:617` for the radius. Production has
**two** call sites: `bayesian_statistics.py:2823` passes `2.0` and `:2838` passes `1.5`.
`gen_ch03.py:160` hardcodes `SIGMA_MULTIPLIER = 2  # handler...default`; `gen_ch06.py:166`
hardcodes `1.5  # Production BallTree call (bayesian_statistics.py:2837)`. So ch03 quoted
the *function default* and called it production; ch06 quoted one of two call sites and called
it production. Neither says there are two.

**What I expected.** The chapter that introduces the ball (ch03) is the one that states the
rule, in a `RATIFIED` box, as though it were settled. It is not settled, and ch06 — three
chapters later — silently overrides it and *also* discloses (`:848-858`) that the redshift
window's `sigma_multiplier` argument is dead code (window always ±3σ). A student who reads
both chapters carefully cannot reconcile them, and the one who reads only ch03 carries a
wrong parameter.

**Suggested fix.** Adjudicate once and propagate. Concretely:
1. State in ch03's ball box that the pipeline has two candidate-search call sites
   (`bayesian_statistics.py:2823` → 2.0, `:2838` → 1.5), name which one the campaign the book
   reads actually ran, and re-run `gen_ch03.py` with that multiplier.
2. If both are genuinely live in different legs, say so on the page and print EMRI-889's ball
   at both (3 galaxies at 2σ, 2 at 1.5σ) — the book's own standard.
3. Either way, ch03 must not carry `n_σ = 2` inside a `RATIFIED` block without the
   qualification; that block currently reads as a theorem.
4. File it as a cross-chapter flag; it is not in `BUILD_REPORT.md` §5 at all.

## BLOCKER-3 — ch04's and ch05's dossiers quietly prefer the disputed σ_dL/dL, which the reviewer's guide explicitly asks not to happen

**Where:** `ch04-loud-half.html:627` and `ch05-unseen-galaxy.html:863`, both:
`d_L | 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)`.

**What I observed.** The book's flagship honesty device is that every page carrying the
EMRI-889 distance-precision number prints **both** readings and picks neither (BUILD_REPORT
§5.1 item 1). ch01 does this beautifully (`:391-408`, an `OPEN`-badged block with the 1/ρ
discriminating check). ch02's dossier does it (`:874-886`). ch03's dossier does it
(`:854`). ch06 does it and flags it (F-ch06-1). ch07's opener uses the recomputed
`9.0×10⁻⁴` (`ch07-redshift.html`, "failure this chapter answers"). **ch04 and ch05 print
only the disputed spec value, with no flag, no both-values, no chip.** Neither chapter's flag
file raises it.

**What I expected.** Rubric B §3.3.1 ("no number without provenance") plus the build
report's own instruction to reviewers: *"Each chapter claims to show both values — verify no
page quietly prefers one."* Two pages do. As a reader I hit the ch04 dossier immediately
after ch03 had shown me the two readings side by side, and read the single value as ch04
having settled the dispute — which it has not.

**Suggested fix.** Copy ch03's dossier cell verbatim into ch04:627 and ch05:863:
`88.879 Mpc (σ_dL = 7.98×10⁻⁵ Gpc = 0.0798 Mpc, i.e. σ_dL/d_L = 8.98×10⁻⁴)` plus the F1 flag
chip. Zero other content changes needed. Then add a one-line grep gate to the build:
`8.0×10⁻⁵` must never appear without `8.98×10⁻⁴` within the same block.

---

# [MAJOR]

## MAJOR-1 — ch05 §2 teaches me w_G as a per-event probability, then immediately shows me it isn't, and the one sentence that would rescue me is inside a collapsed `<details>`

**Where:** `ch05-unseen-galaxy.html` §2 — the master-equation paragraph ("The weight w_G is
the probability that *this detection's* host is in the catalogue, and it is a probability, not
a flag"), then four lines later "**First: 12%.** Not 5%, which is the fraction of events whose
host really is catalogued in this run (76/1588 = 4.8%)."

**What I observed.** This is the paragraph I read three times. I had just been told w_G is a
per-event probability; a per-event probability that reads 12% when the realized rate is 4.8%
is, to a Bayesian student, simply *wrong*, and the page's response is to defer it to Ch 9
and Ch 11. The sentence that dissolves the confusion — *"w_G is a selection- and
volume-weighted average of f along the whole line of sight, not f at any one redshift"* —
exists, but only inside the `▸ For the GW reader — what w_G is made of` details block, which
the default (Mara) persona does not pre-open. I only found it because I was reading the HTML.

**What I expected.** The narrator to give me the *type* of the object before showing me a
number that violates my assumed type. The 12%-vs-5% gap is genuinely a live finding (C9) and
must stay — but "this is a population-level, detection- and volume-weighted quantity, and it
is *not* the same object as the realized in-catalogue fraction; whether it *should* agree with
it is claim C9" is a rung-level sentence, not a Ch 9 tool.

**Suggested fix.** Move (do not copy) one sentence from the GW-reader block into the narrator
flow, immediately before "First: 12%": *"Read that carefully: w_G is not a per-event flag and
not a per-event count. It is a single number per h, averaged over the whole line of sight
against detectability and volume — so there is no arithmetic reason it must equal the fraction
of this run's events that happened to have a catalogued host. Whether it nevertheless
should is a live claim."* Then the 12%-vs-5% shock lands as a finding rather than as
apparent nonsense.

**Related, same section:** the trap box at `:1093-1102` quotes the realized rate as
**164/3135 = 0.0523**, while §2 quotes **76/1588 = 4.8%**. These are different samples (two
seeds vs one), but the page does not say so, and a student reading both concludes the book
cannot keep its own denominator straight. One clause fixes it.

## MAJOR-2 — the Symbol Passport spoils Ch 9 from Ch 5 and violates the rung-guard it was built to protect

**Where:** `js/book.js:567` — `wG: { meaning: "mixture weight β_G/D — ESTIMAND-DEPENDENT;
always name the mode", note: "C9 is a live FINDING — see Ch 9 §6 / Ch 11" }`, surfaced on
`ch05-unseen-galaxy.html`'s two `data-term="wG"` spans.

**What I observed.** BOOK_DESIGN §1 Ch 5 explicitly says w_G is to be described "at rung level
— *the probability the host is catalogued*, β_G/D mechanics **deferred to Ch 9** per
rung-guard." The passport hands me `β_G/D` on hover in Ch 5, four chapters early, plus the
words "ESTIMAND-DEPENDENT" and a pointer to a live finding called C9. The build report asks
whether this "lands before Ch 9 confuses you, or spoils Ch 9's reveal". My honest answer:
**neither — it did something worse.** It made me distrust w_G *before Ch 5 had earned my
trust in it*, which flattened Ch 5's own hero-graphic moment. By the time I reached the
12%-vs-5% paragraph I had already been told there was a scandal attached, so it read as
confirmation rather than as discovery.

The same issue affects `eps` ("rail threshold quoted at 0.256… C7 is a live FINDING") and
`Cscale` ("the arbitrary mass-coordinate rescale of the C8 walk"), both of which are pure
Ch 7/Ch 8 spoilers if a term is tagged earlier.

**What I expected.** Shared chrome must not be able to defeat the rung-guard that binds every
chapter agent.

**Suggested fix (small, in `book.js` only).** Add an optional `firstChapter` field to each
`Book.SYMBOLS` entry and a per-page chapter index (already available via `manifest.js`).
Before the symbol's home chapter, render only `sym`, `units` and a rung-safe gloss, and
suppress `meaning`'s formula and the `note`. For `wG` that means Ch 5 hover shows
*"mixture weight — the probability the host is catalogued; how it is computed, and what it is
normalized to, is Ch 9"*, and Ch 9 onward shows the current full card. Three lines of
rendering logic; it preserves the instrument and restores the guarantee.

## MAJOR-3 — ch02's central predict-then-reveal never grades me

**Where:** `ch02-bayes.html:598-602` — options `~1400 / ~800 / ~100 / ~10`; reveal at
`:649-665` opens "Measured: **24 events**."

**What I observed.** I committed to "~100". The reveal reports 24 and then addresses only the
opposite extreme: *"If you chose '~1400' you were applying √N reasoning…"*. Nothing tells me
whether "~100" was right, and 24 sits awkwardly between two of the four options (2.4× from
"~10", 4.2× from "~100"). There is no `correct` marker in the markup and `predictReveal`
has no notion of a right answer, so the widget cannot grade. Compare ch00, whose reveal opens
with a bald "**(d).**", and ch04's, which names my specific error ("If you put the marker
exactly on 0.730 you made the assumption this book is built to attack"). ch05's class-crossing
reveal opens "**(c).**". ch02 is the outlier, and it is the chapter's payload beat.

**Suggested fix.** Open the reveal with `**(d) — a handful.** Measured: 24 of 1588.` and add
one clause per wrong option: *"'~100' is the instinct that there must be a healthy
sub-population carrying the answer; the participation ratio says the effective count is 12."*
(That number is already computed on the page.) Optionally add an `data-predict-correct`
attribute to `predictReveal` so every widget can grade uniformly — 5 lines in `book.js`.

## MAJOR-4 — ch05's ~5000× predict is spoiled by its own section heading, twice, before the question is asked

**Where:** `ch05-unseen-galaxy.html` §3 heading — "The completion branch, **and a factor of
5000 hiding in a sky integral**" — and the section's first paragraph — "the step that costs a
**factor of several thousand** if you get it wrong". Then the predict block: *"…inflates the
completion branch by a factor of —"* with options `about 1.5× / about 20× / thousands`.

**What I observed.** I "predicted" correctly, from the heading, without thinking. The
prediction instrument is the book's declared analogue of pre-registration; here it registers
nothing. The information *needed* to reason (the peak-vs-flat-prior ratio 2/(σ_φσ_θ)) is
present and I could have derived ~10³–10⁵ honestly — the chapter simply didn't let me.

**Suggested fix.** Rename §3 to "The completion branch, and the step where you must integrate
rather than evaluate", and cut "several thousand" from the first paragraph (the sentence
still works as "…the step that costs orders of magnitude if you get it wrong"). Move the
5000× to the reveal, where it already is. Zero data changes.

## MAJOR-5 — ch04's Q4.3 and Q4.4 hidden answers use machinery the chapter never introduces

**Where:** `ch04-loud-half.html` Q4.3 answer ("…the estimator's effective sample size
collapses in exactly the cells that matter, which is why the project's ratified campaign
design imposes a **minimum ESS per node**") and Q4.4 answer ("…the project's G1 gate checked
one such cancellation (**Σ_glob ≡ n̄ β_G**) and found a −17.2% end-to-end residual, which is
now part of **claim C9**").

**What I observed.** I could produce one of Q4.3's two required mechanisms (extrapolation
policy, which the chapter's defect-shelf table teaches). The second requires "effective
sample size", "per node", and the campaign design document — none of which appear anywhere in
ch00–ch04. Q4.4 likewise: I could answer the first half from §3, but Σ_glob, β_G, n̄, G1 and
C9 are all Ch 5/Ch 9/Ch 11 objects. Per rubric A this is the "relies on a linked doc to
complete an argument" failure, delivered inside the answer key.

**Suggested fix.** Q4.3: replace the ESS clause with a mechanism the chapter owns — e.g.
*"(ii) support mismatch: if the pool has few injections where the catalogue's candidates
actually live, p_det there is estimated from a handful of horizons, and a systematically
low estimate in one distance band is a systematic tilt on every event that overlaps it —
which is what ledger #8's `fill_value=0.0` did in its extreme form."* Q4.4: end the answer at
"…is the selection correction", and move the G1/Σ_glob sentence into a `⏭ Ch 9` chip rather
than the answer body.

## MAJOR-6 — ch05's Q5.4 answer carries the disputed C5 leverage figure alone, while ch11 carries both halves

**Where:** `ch05-unseen-galaxy.html:968` — "…with dh*/dε leverage **1500–2400×** — strong
evidence for reading B", no flag, no alternative.
Compare `ch11-honest-state.html:613-615`: *"1500–2400×; recomputing it from the adjudicator's
own … runs, median **197×**. This book does not reconcile the two: both are…"* and again at
`:1302-1304`.

**What I observed.** BUILD_REPORT §5.1 item 3 names this dispute and says it "touches Ch 5's
Q5.4 too". Ch 11 obeys the both-halves rule; ch05 does not. Same defect shape as BLOCKER-3.

**Suggested fix.** Append ch11's own clause to the Q5.4 answer: *"(the adjudicated figure is
1500–2400×; recomputing it from the adjudicator's output gives 142×–2458×, median 197× —
`flags/ch11_FLAGS.md` F-ch11-1. The argument does not depend on which is right.)"*

## MAJOR-7 — I5.1's κ dial is wildly non-monotonic in the middle, and the page narrates only its two endpoints

**Where:** `ch05-unseen-galaxy.html` I5.1; data `data/ch05_mixture.json`
(`kappa_grid`, `summary_by_kappa`).

**What I observed (by reading the data, not the browser).** Dragging κ up from the shipped
1.0, the MAP walks: `0.740 → 0.755 (κ=1.4) → 0.78 (2) → 0.85 (3) → 0.86 (5 … 120) → 0.63
(250) → 0.60 (500+)`. That is a rail *high* for roughly a third of the dial's upper range and
then a discontinuous flip to a rail *low*, with `n_zero_by_kappa` = 0 at every finite κ (the
493 silenced events only appear at κ = ∞). The page's prose narrates κ→0 (0.755) and κ→∞
(0.600, 493 silenced) and nothing in between; the generic `other` verdict text says only
"watch how far the MAP travels". A student who drags the dial sees the answer pin to the top
of the grid for a long stretch and reasonably concludes the widget is broken.

**What I expected.** The book is scrupulous everywhere else about naming what a control's
extreme states mean. This one has an unlabelled 0.86 plateau that looks exactly like the
`ledger #49a` H₀-independent pathology the book warns about in ch02's Trap 2.A.

**Suggested fix.** Add a third `V51` state for `1 < κ ≲ 200` — something like: *"Counterfactual
— **the in-catalogue class taking over.** As κ rises the mixture is increasingly the catalogue
branch, and you are watching the 76 in-catalogue events' own preference (I5.2's 0.86 curve)
dominate. Push further and the 1512 dark events' zero legs reassert and drag it back to the
wall. Neither end is a measurement; the round trip is the point."* And add one sentence to
the prose before the widget so no-JS/static readers get it too.

---

# [MINOR]

## MINOR-1 — ch11 re-surfaces my Chapter-3 guess as an internal slug that also spoils it

`ch11-honest-state.html:1946` renders `esc(ch3.value)`, and ch03's buttons store
`impostor` / `host` / `cannot-tell` (`ch03:286-288`). So the payoff sentence reads *"In
Chapter 3 you put a marker on the galaxy you thought hosted this event (**impostor**)."* —
which (a) is a developer slug, not the label I clicked ("A — the spectroscopic one"), and
(b) grades me by naming the answer inside the parenthesis. **Fix:** store or look up the
button's text content, or map `{impostor:"A, the spectroscopic one", host:"B, the photometric
one", "cannot-tell":"neither"}` in ch11.

## MINOR-2 — ch11's recall code still carries the pre-integration fallback

`ch11:1926-1934` comments "The Chapter 3 host-guess id is not yet fixed book-wide
(WIDGET_REQUESTS R-ch11-2)" and probes four ids. The integrator resolved this; ch03 writes
`ch03-host-guess`. Harmless, but the comment now misdescribes the build. **Fix:** keep the
defensive probe, update the comment.

## MINOR-3 — ch04's guess marker desynchronises if you drag after locking

`ch04:1118-1122`: `guessSlider`'s `input` listener updates `data-predict` but the plot marker
only moves inside `applyGuess`, which runs on click. Drag after locking and the readout, the
stored value and the drawn marker disagree. **Fix:** either disable the slider after lock
(consistent with ch00's `setLocked`) or call `applyGuess` from the input listener once
locked. Also: ch04 hand-rolls this instead of using `Book.predictValue`, which the integrator
built for exactly this request (R-ch04-1) — worth converging.

## MINOR-4 — Q1.3 is only answerable from a collapsed sidebar

`ch01-ruler.html` Q1.3 ("why is an event defined as plunging inside the mission span?") is
answerable *only* from the `Sidebar — what counts as an "event"` `<details>`, which the Mara
persona does not pre-open. I skipped the sidebar on first read (it is chipped RATIFIED and
looked like specialist material) and could not answer. **Fix:** either open the sidebar by
default for all personas, or fold its two-sentence core ("SNR accumulates near plunge; the
rate model is itself a plunge rate") into §2's narrator flow and leave the history in the
sidebar.

## MINOR-5 — three self-checks are recall, not transfer

- **Q2.3** ("distinguish biased / noisy / mis-calibrated with a one-line test each") is a
  verbatim restatement of the definition list three screens above, including the operational
  tests. Per rubric D, "recall questions present" scores 1.
- **Q5.4** asks for "two mutually exclusive readings and what would distinguish them"; §5
  supplies Reading A, Reading B, *and* names the distinguishing test ("the test is leverage")
  in the paragraph directly above. Nothing is left for me to do.
- **Q0.3** ("name the one thing you still need") is answered four times on the ch00 page.

None of these is harmful; all three are free points that dilute the instrument. **Fix for
Q2.3:** ask instead for an estimator that fails exactly one of the three and passes the other
two, and require the reader to construct the settings — I2.2's three dials make that a
real, checkable exercise the widget already supports.

## MINOR-6 — ch01's standard-siren equation is the circular-quadrupole form, and the caveat is in a collapsed block

`ch01:113-124` prints `h_+ = (A/d_L)((1+cos²ι)/2)cosΦ` as *"the entire content of the phrase
standard siren"*. For an EMRI (eccentric, inclined, Kerr, `e0 = 0.167` per this event's own
dossier) that is a caricature, and the page says so — inside `▸ For the GW reader` ("The
pipeline never manipulates h(t) in the form printed above"). As Mara I carried the wrong
mental model for two chapters. **Fix:** one narrator sentence after the box: *"This is the
circular-binary quadrupole form, printed because it shows where d_L enters; the real EMRI
waveform is generated numerically and is far richer — nothing below depends on the closed
form."*

## MINOR-7 — ch03 introduces Σ, Σ₃, λ_max and u before Ch 6

`ch03:168-171` (the `JΣJᵀ` rescaling and `λ_max`) and `:471-477` (the "three-dimensional
Fisher Gaussian" with `Σ₃` and `u = d_L/d̂_L`). Per BOOK_DESIGN §3.3.7 these are Ch 6 tools.
The Symbol Passport tags `Sig`, `u` and `phth`, which genuinely saved me — I hovered `u` and
got "fractional distance d_L/d̂_L (mean 1)" and moved on. But the passport card for `Sig`
reads "Cramér–Rao covariance Γ⁻¹", and Γ has not been defined either, so the definition is
circular at that point. **Fix:** give `Sig` a rung-safe first gloss (see MAJOR-2's mechanism)
— "the measurement's covariance matrix: how big the error ellipsoid is and which way it
tilts" — and keep Γ⁻¹ for Ch 6 onward.

## MINOR-8 — ch03's line anchor for the ball search is stale in a way that misleads

`ch03:160` chips `galaxy_catalogue/handler.py:519` for the ball search. In the current tree
`:519` is `read_reduced_galaxy_catalog`; the function is at `:556`. BUILD_REPORT §5.5 item 23
records this drift as expected (anchors are re-grep hints), so this is not a defect of the
build — but a student who follows the chip lands on catalogue loading and concludes the
citation is wrong. **Fix:** since the drift is known and enumerated, print the current line
in a `title=` tooltip alongside the spec anchor, on the chips listed in §5.5 item 23.

---

# [PRAISE]

**P-1 — ch04 §3's nats comparison is the best single paragraph in the book.**
"Per event that is ln D(0.60) − ln D(0.86) = 0.4025. Multiply by 1588 events and the
denominator is worth **639 nats** of monotone pull … the entire data-driven preference is
**18.8 nats** (low edge) and 20.8 (high edge). The denominator you are about to delete is
worth roughly thirty times the whole signal." I re-derived both numbers on paper in under a
minute and *understood the rail* rather than being told about it. This is what "shown, not
asserted" is supposed to mean. Every chapter should aim for one of these.

**P-2 — ch03 §4's h*(g) = d_L(z_g; h=1)/d̂_L.**
"Each candidate galaxy makes exactly one statement about the Hubble parameter: *if I am the
host, then h is this.*" One line, fully re-derivable, and it converts "a catalogue constrains
H₀" from a slogan into a mechanism. I predicted the shape of the answer before the widget
told me. Keep the sentence "If the catalogue were structureless … the sweep would carry no
information at all" — it is the load-bearing corollary and it is placed exactly right.

**P-3 — ch05's class-crossing reveal (prediction #7).**
This is the only place I was confidently, instructively wrong. I reasoned that in-catalogue
events must be the well-behaved ones; they rail at 0.86. The follow-up — Reading A vs Reading
B, and "it is not centred on truth, it is *balanced* there" — is the sharpest conceptual move
in ch00–05, and refusing to resolve it for six chapters is the right call. The honesty of
`summary_incat.edge_high_over_peak = 1.0` being disclosed on-page ("its highest point is the
last grid point, so the peak is wherever the grid stops") is exemplary.

**P-4 — ch04's predict marker names *my* error, not a generic one.**
"If you put the marker exactly on 0.730 you made the assumption this book is built to
attack." I did. That single clause did more pedagogical work than the widget it gates.

**P-5 — I5.1's "Has this been tried?" box fires on every off-shipped state.**
`ch05:1470-1496`: `triedBox.hidden = (cls === "shipped")`, with a per-state verdict from
`V51`. I could not find a way to reach a dead configuration in ch00–05 without being handed
its obituary (`ledger #61`, `#64`, `#54`, `#57`). The museum meta-rule holds where I tested
it. Likewise ch01's `EXONERATED ledger #53/#59` block volunteers the Ω_m story before the
reader can propose it.

**P-6 — ch01's F1 flag block is a model of how to carry an unresolved dispute.**
It states both readings, gives an *independent discriminating check* (σ/d_L = 1.28/ρ vs
0.11/ρ), and then explicitly declines to settle it, citing the build rule that forbids
silent reconciliation. This is the tone the whole book should hold — which is exactly why
BLOCKER-3 stings.

**P-7 — the ch05→ch06 on-ramp is airtight.**
Q5.5 asks for two properties of the ellipsoid ("size" and "orientation/correlation"); ch06's
failure box opens by quoting all three prior chapters' unexamined phrase and then hands me a
four-event table where 889's ball holds 2 galaxies and 606's holds 7109. The handoff answers
the question I was left holding, with data, in the first screen. On-ramp: **holds**.

**P-8 — venue discipline.**
ch04's "What the live numbers are, and are not" block (−0.178/0.000 is Phase-32; 0.740 is one
realization of one seed; "bias is a property of an estimator over realizations") is the
distinction I most expected a popular-science treatment to smudge, and it is stated three
separate times across ch02/ch04/ch05. Do not cut it for length.

**P-9 — static fallbacks are real content, not stubs.**
Every `<noscript>` I read carries the actual numbers and the actual conclusion (ch05's
predict blocks even hand over the measured answer rather than dead-ending). A JS-off reader
loses the instrument and keeps the physics. That is a genuinely hard thing to get right.

---

# Grades — "could I re-derive this, and did I build intuition?"

| Ch | Grade | Why (one sentence) |
|---|---|---|
| **0** | **A−** | The quadrature ceiling is fully re-derivable in three lines and "σ_sys does not shrink with N" landed permanently; docked only because Q0.3 is a giveaway and the chapter teaches one idea. |
| **1** | **A** | I can now re-derive d_L(z;h,Ω_m), argue from its structure why h is a pure prefactor and Ω_m a shape parameter, and reconstruct the (d_L, ι) amplitude degeneracy unaided — and the "running the ruler backwards is a restatement of your assumption" pivot is unforgettable. |
| **2** | **A−** | Grid vs sampler, log-additivity, the three failure modes and the curvature statistic are all re-derivable, and the lurch is a genuine surprise; docked for the ungraded predict (MAJOR-3) and the "tens of thousands" answer key (BLOCKER-1). |
| **3** | **A** | h*(g) = d_L(z_g;1)/d̂_L is the cleanest idea in the book and I derived the ratio-of-sums vs sum-of-ratios divergence myself before reading the reveal; the σ-multiplier contradiction (BLOCKER-2) is a fidelity problem, not a comprehension one. |
| **4** | **A** | d_hor = ρd_L/ρ_thr, the h-invariance of the horizon set, and the 639-vs-19-nats comparison mean I could reconstruct this chapter's whole argument from memory on a whiteboard. |
| **5** | **B** | The master equation is legible and the class-crossing is the book's best beat, but I cannot re-derive w_G — its definition is quarantined in a `<details>`, the 12%-vs-5% shock arrives before I can absorb the object's type, and a third of I5.1's dial is unnarrated. |

**Ship-gate read (rubric A, student comprehension), my scoring:** ch00 3, ch01 3, ch02 3,
ch03 3 (2 if BLOCKER-2 is judged comprehension rather than fidelity), ch04 3, **ch05 2**.
All ≥ 2, so the gate passes on axis A — but ch05 is the one chapter where a motivated
student can finish the page and still not know what w_G *is*, and it is the chapter the rest
of the book leans on hardest.

---

*Reviewer: "Mara". Read-only pass; the only file written is this one.*
