# Review — "Tomas" (GW researcher, LVK background, new to this pipeline)

**Scope:** skim of Ch 0–5, deep read of Ch 6 → Ch 11 → museum, in order, as a Ch-5 reader
would arrive. Predictions made at every predict-then-reveal beat before scrolling;
self-check questions attempted before opening the answers; interactives simulated by
tracing the page JS against `book/site/data/*.json`; a dozen numbers re-derived from
`results/campaign51_20260728/realistic_20260729/seed61000/prepared_cramer_rao_bounds.csv`
with the project venv.

**Headline.** This is the best-argued piece of scientific writing I have read out of a
thesis project. Chapters 6, 7, 9, 10 and 11 would each survive a referee. The physics
fidelity is real — every number I independently recomputed came back exact (see PRAISE).
The problems below are almost all *consistency* problems between chapters, not errors of
physics, and three of them are the kind that a hostile examiner will find in ten minutes.

**On the crown question (does Ch 11 equip me to have an opinion on C7/C8/C9?):** yes,
and I formed one — see the note after [MAJOR-4]. It is not a recitation of verdicts.

---

## [BLOCKER]

### B1 — Ch 3 §1 uses a **non-production** sky-ball radius and calls it production; it contradicts Ch 6 on the running example

**Page/anchor:** `ch03-which-galaxy.html` §1 "The candidate ball", the
`.voice-derivation` box (line ~199) and the census widget above it; vs
`ch06-black-box.html` §4 gw-reader box and I6.1 static fallback.

**Observed.** Ch 3 states the search radius as
$r = n_\sigma\sqrt{\lambda_{\max}(\Sigma')}$ with **$n_\sigma = 2$**, in a RATIFIED
derivation box chipped `handler.py:519`, and the census widget's caption says the counts
are "measured over all 1590 rows **with the production ball rule**". Ch 6 states
"Production passes **$n_\sigma = 1.5$** (`handler.py:575-578`, `bayesian_statistics.py:2820`)".

The generator makes it explicit:

```python
# book/generators/gen_ch03.py:160
SIGMA_MULTIPLIER = 2  # handler.get_possible_hosts_from_ball_tree default
```

…and the production call site is:

```python
# master_thesis_code/bayesian_inference/bayesian_statistics.py:2838
sigma_multiplier=1.5,  # type: ignore[arg-type]
```

`2` is the *function signature default* (`handler.py:568`), which production overrides.
Ch 3 has documented the default and labelled it the production rule.

I recomputed EMRI-889's radius from its own CRB row both ways:

| multiplier | chord → radius |
|---|---|
| 1.5 (production) | **0.7568 arcmin** — exactly Ch 6's "0.757 arcmin" |
| 2.0 (Ch 3) | **1.0090 arcmin** — exactly Ch 3's "1.009 arcmin" |

**Consequence.** Every candidate-census number in Ch 3 is measured at a 33% larger radius,
i.e. **1.78× the solid angle**: median ball 1616, p95 35,466, max 546,158, "552 of 1590
events have no catalogue candidate", and the running example's "**three** galaxies in its
ball". Ch 6, on the same event and the same catalogue, prints "**2** catalogue galaxies in
the ball". The book's flagship running example has two different ball populations in two
chapters, with no flag on either page.

**Expected.** Either the production 1.5 everywhere, or — if the 2σ census is deliberate —
a flag box saying so, per `BOOK_DESIGN §4.1` ("a generator that disagrees with the spec
stops and flags"). This is the one numeric dispute in the book that is *not* in §5 of the
BUILD_REPORT and *not* in `ch03_FLAGS.md`.

**Fix.** Set `gen_ch03.py:160` to `1.5`, regenerate, and update §1's derivation box and
the 889 numbers; or add `F-ch03-12` and print both, as Ch 1/6/7 do for σ_dL. Also fix the
`.voice-derivation` box: it currently asserts a wrong production constant under a RATIFIED
badge, which is a rubric-B automatic zero by the spec's own §3.4.

---

### B2 — Four pages print the disputed `σ_dL/dL = 8.0×10⁻⁵` as fact, unflagged

**Page/anchor:** the EMRI-889 dossier row in `ch08-mass-channel.html:801`,
`ch10-calibration.html` (§3 dossier), `ch11-honest-state.html` ("Event dossier — EMRI-889,
closed"), `museum.html` ("Event dossier — EMRI-889 in the museum").

**Observed.** Ch 1 (§2 + Q1.2 editor's note), Ch 6 (§4 flag box + Q6.5) and Ch 7 (dossier
provenance note) all establish, three independent ways, that `8.0×10⁻⁵` is the **absolute**
σ_dL in Gpc wearing a fractional label, and that the fraction is `8.98×10⁻⁴`. I confirmed:
σ_dL = 7.984×10⁻⁵ Gpc, d_L = 0.0888792 Gpc → 8.983×10⁻⁴; and σ_u·ρ for row 889 is 1.2799,
inside the population's own p95 of 1.250 — the fractional reading is the only one
consistent with the CRB table.

Ch 8, Ch 10, Ch 11 and the museum then print `d_L 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)` with no
flag, no both-values, no chip. Ch 11's is on the **closing** dossier, the last time the
reader sees the running example.

**Expected.** The BUILD_REPORT's reviewer's guide says explicitly: "Each chapter claims to
show both values — verify no page quietly prefers one." Four pages quietly prefer the
one the rest of the book calls a units slip.

**Fix.** One-line edit on each of the four dossier rows to match Ch 6's dossier wording:
`σ_dL = 8.0×10⁻⁵ Gpc = 0.0798 Mpc → as a fraction 9.0×10⁻⁴ (see Ch 1 §2 / Ch 6 §4)`.

---

### B3 — Ch 8 asserts `σ_Mz/M_z ≈ 10⁻⁴` five times, unflagged, after Ch 6 measured 8.8×10⁻⁸ and filed the flag *for Ch 8*

**Page/anchor:** `ch08-mass-channel.html` §"The promise this chapter starts from" (line
124), §1 display equation (line 238: `σ_Mz/M_z ≈ 10⁻⁴` inside a RATIFIED derivation box),
§2 (line 317, the "needle vs barn door" ratio), §2 predict box (line 376), Q8.1 (line
1044). Also `ch01-ruler.html:290` and `ch07-redshift.html` Q7.6 / §closing.

**Observed.** `ch06_FLAGS.md#F-ch06-5` says, verbatim: *"Measured on
`seed61000/prepared_cramer_rao_bounds.csv`: median **8.8×10⁻⁸**, p5–p95
2.5×10⁻⁸–3.0×10⁻⁷. Event 889: 1.36×10⁻⁹. … Flagged for the ch08 agent and the
integrator."* I reproduced it: median `sqrt(delta_M_delta_M)/M` = **8.8×10⁻⁸**, row 889 =
**1.365×10⁻⁹**. Ch 6 §4.1 prints 1.36×10⁻⁹ on the page. Ch 8 never carries the flag.

So the book states, in two adjacent chapters, that the same stored quantity is 10⁻⁴ and
10⁻⁷–10⁻⁹, and one of them puts it in a display equation under a RATIFIED badge.

**Why it is a blocker and not a nit.** It is the *numerator* of Ch 8's central mechanism
("the comparison of a needle with a barn door", the 193:1 one-sidedness, the sieve's
97–99% rejection). The argument's direction survives (10⁻⁷ ≪ 1.28 too), but a reader who
has just read Ch 6 hits a three-orders-of-magnitude contradiction at the exact point where
Ch 8 asks them to trust a ratio.

**Fix.** Ch 8 §1 keeps the claim-file value chipped `CLAIM C4` and adds Ch 6's
recomputation beside it with a pointer to `F-ch06-5`, exactly as Ch 1/6/7 handle σ_dL.
Same for `ch01-ruler.html:290` and `ch07` Q7.6. Do not silently substitute — the claim
file really does say 1e-4 (`CLAIM_2D_BIAS_20260730.md:172`), so this is a both-values case,
not a correction.

---

## [MAJOR]

### M1 — Ch 11's opening table quotes a pull range that Ch 8 measured and flagged as non-reproducing

**Page/anchor:** `ch11-honest-state.html:161` (the delivered-state table, 2D row);
vs `ch08-mass-channel.html:203` and `ch08_FLAGS.md#F-ch08-1`.

**Observed.** Ch 11 prints 2D pull `+3.4 … +4.5 (mean +4.04)`. Ch 8 prints
`+2.47…+4.74 (mean +4.04)` and its flag file states the readout's range does not
reproduce from `realistic_scores.csv` (low end seed61000/r2 = +2.474, high end r5 =
+4.735). Same ten runs, same quantity, two ranges, one flagged and one not — and the
unflagged one is in the *first table of the closing chapter*, which is the most quotable
object in the book.

**Expected/fix.** Carry F-ch08-1's both-values note into Ch 11's table (`+3.4…+4.5`
readout / `+2.47…+4.74` recomputed). Mean and the 10/10 count agree and can stand alone.

---

### M2 — Ch 4 never says that `p_det` here is `p_det(d_L)` — a marginal over intrinsic parameters — which is both the deviation from LVK practice and the seed of C9

**Page/anchor:** `ch04-loud-half.html` §2 "p_det: a horizon you can measure" and its
gw-reader box.

**Observed.** The survival estimator is introduced as
`p_det(d_L) = Pr(d_hor ≥ d_L)` over the injection pool, and the chapter's elegance
argument (h-invariance of the horizon set) is beautiful and correct. But the words
"marginal", "intrinsic", "masses" appear **nowhere** on the page (grep: zero hits). A GW
reader arrives from LVK where the selection function is `p_det(θ)` — a function of masses,
spins, inclination, sky — and is used *per event*. Here it is a function of `d_L` alone,
i.e. it has been marginalised over the population's intrinsic-parameter distribution, and
that marginalisation is a modelling assumption, not a definition.

That assumption is exactly what Ch 9's C9 measures as broken: "β_G weights the completeness
f(z) by the **pool-marginal, population-mass** p_det, but the catalogue's hosts are
Malmquist-selected and carry heavier M–σ black holes". The reader meets the consequence 5
chapters after the assumption, and the assumption was never stated.

**Fix.** Two sentences in Ch 4 §2 plus a `⏭ Ch 9` chip: *"Note what this p_det is a
function of: distance, and nothing else. The intrinsic-parameter dependence has been
marginalised over the population the injections were drawn from — which is exact only if
the events you apply it to are drawn from that same population. Chapter 9 measures a case
where they are not."* This costs 60 words and closes the book's longest open loop.

---

### M3 — The external literature contrast is thin in exactly the places a GW reader needs it

**Page/anchor:** book-wide; acutely `ch08-mass-channel.html` §1, `ch09` §3,
`ch10` §3, `ch00` "Third methods have been tried".

**Observed.** The pipeline's own lineage is cited well (Gray 2020/2023, GMV 2022, MFG
2019, Laghi 2021, Turski 2023, Dálya 2022, Babak 2017, Barausse 2012, Vallisneri, Finn,
Cutler–Flanagan). What is absent, everywhere:

1. **No LVK dark-siren result.** `Abbott et al. 2023, arXiv:2111.03604` (GWTC-3 cosmology,
   H₀ = 68⁺⁸₋₆ dark-siren) never appears. Ch 0's "third methods" figure plots TRGB, time
   delay and the GW170817 *bright* siren but not the dark-siren number the entire book is
   about. A reader finishes Ch 0 without knowing what the state of the art in this book's
   own genre currently achieves.
2. **No `gwcosmo` / `icarogw`.** The book repeatedly says "the pipeline deviates from Gray
   et al." (D1, D2, D4, D6, D8, C1) without once naming the reference implementation those
   deviations are deviations from. Ch 9 §3 then deprecates the `global` mode — the literal
   A10 transcription, i.e. what gwcosmo actually does — as "empirically mis-calibrated,
   ~0% coverage, railing to the grid edge" (see M9).
3. **No Chen, Fishbach & Holz 2018 (arXiv:1712.06531).** The canonical dark-siren
   information-budget paper; Ch 2's "information concentration" and Ch 10's idealization
   ledger both beg for it as an external sanity anchor.
4. **No forecast anchor for Ch 10 §3.** The scenario table (0.032 → 0.22 → 0.30 → 3.6
   km/s/Mpc) is excellent, but a reader has no external scale. Laghi et al. 2021 — already
   cited elsewhere in the book — forecasts ~1% for LISA EMRI dark sirens; putting that
   beside row D (3.6, i.e. ~5%) would instantly tell the reader whether row D is
   pessimistic or realistic.
5. **The mass channel will be misread as spectral sirens.** "spectral siren" appears zero
   times. Ch 8's premise — "M_z is a redshift handle" — is one sentence away from the
   population-mass-function method (Farr, Mastrogiovanni, Ezquiaga–Holz). It is *not* that
   method: this is a per-host BH-mass **association**, using a catalogue proxy, not a
   population feature. Every GW reader who skims Ch 8 will make this mistake.

**Fix.** One sidebar in Ch 8 §1 ("this is not the spectral-siren method, and here is why
the failure modes differ"); one row added to Ch 0's third-methods figure; one paragraph in
Ch 9 §3 naming gwcosmo as the reference implementation of `global`; one external number in
Ch 10 §3's table caption.

---

### M4 — Ch 11 §4 loses the framing that would let me *decide* the C7↔G2b collision, which Ch 7's Trap 7.B already had

**Page/anchor:** `ch11-honest-state.html` §4 "C7 — the host-redshift kernel integrates
against the wrong prior" and the two-column RATIFIED/FINDING block; vs
`ch07-redshift.html` Trap 7.B.

**Observed.** Ch 11 §4 frames the collision as *"They are answers to different questions.
G2b establishes which weight is uniquely consistent with the rate model … C7 measures what
that weight does at finite width when the numerator's integration window is proportional
to z."* That framing makes the tension look like a scale-dependence artefact, and it hands
the whole question to cell B.

Ch 7's Trap 7.B says the sharper thing, in half a sentence and then drops it: *"the domain
here is set by whether the weight you deconvolve against is the **population prior** (which
G2b establishes it is) or the **selected population prior** (which C7 measures it is not)."*
That is the actual disagreement, and it is not about σ_z at all. The two boxes are priors
over **different random variables**:

- G2b's `w_pop = (dV_c/dz)/(1+z)` is the prior over *where an EMRI host is*, given that an
  EMRI happened. Correct, unique, and h-independent — G2b's proof is fine.
- The object being deconvolved in `N_g` is *a row in a flux-limited catalogue*. Its prior
  is the catalogue's own selected number density, `φ_cat(z) ∝ f(z,Ω)·(dV_c/dz)/(1+z)`, and
  for the numerator of a **detected** event it should further carry `p_det`. That prior
  turns over; the bare one does not, which is the whole of C7's `+3(σ_z/z)²`.

`φ_cat` appears **exactly once** in the book (Ch 7 §6, inside a quoted C7 statement) and is
never defined. Once the two objects are named, the collision stops being "two correct
answers" and becomes "two answers to two questions, and the numerator asked the wrong one"
— which is close to the standard-practice reading (gwcosmo does not deconvolve galaxy
redshifts against a bare cosmic volume prior at all).

**My opinion, which the book did equip me to form:** I would fix **C8 first** (Ch 11's
Q11.4 gives both the case and the counter-case, and I agree with the project's reason for
not shipping it: the measure repair and the +19-nat population tilt must be separated).
On the leg question I land where the book says a reader should not be allowed to land
without cell B — but I land there for a reason the book underplays: the numerator prior is
a prior over a *catalogued* object, and the burden of proof is on the bare-volume weight,
not on C7. **This is a compliment to Ch 11 and a request:** the chapter is good enough
that the missing framing is the only thing between it and a reader forming the *right*
opinion rather than merely an opinion.

**Fix.** Promote Trap 7.B's two-priors sentence into Ch 11 §4's collision block, define
`φ_cat` at first use in Ch 7 §6, and say in one clause that the standard-practice kernels
(Laghi/Turski/gwcosmo) do not deconvolve at all — which is what makes this project's D8
deviation the thing under test. This does **not** resolve C7 and does not violate the
binding rule; it names the axis the resolution will lie on.

---

### M5 — Ch 9 §4 dumps seven undefined symbols into two display equations; none is in the Symbol Passport

**Page/anchor:** `ch09-universe-factory.html` §4 "Two estimands",
`.voice-derivation` "The two absolute normalizations".

**Observed.** The block introduces `n_w`, `n̄_w`, `n̂_w`, `Σ_glob`, `W_cat`, `V_f(h)`,
`D_gen` and `F` inside two displayed formulas. `n_w` appears in the master formula
*before* it is described, and is never given units or a physical picture ("effective
rate-weighted catalogue number density" would do it). I checked
`Book.SYMBOLS` in `js/book.js:534`: **none of these keys exists**, so hovering them gives
nothing, and `ch09` tags only 9 terms overall (`betaG betaGbar Dh h Mz Om pdet wg wG`) —
it does not even tag `Lcat`/`Lcomp`/`Ng`/`Dg`, which it uses.

This is the rung-violation zone the BUILD_REPORT predicted, and it is the only place in
Chapters 6–11 where I had to go back and re-read a box twice. `Σ_glob` also leaks into
Ch 7 §6 ("adding Δln Σ_glob = +0.027597") and Ch 11's C8 dimension count, where it is
likewise untagged.

**Fix.** Add `nw`, `Sglob`, `Wcat`, `Vf`, `Fincat` to `Book.SYMBOLS` (integrator file —
so this is a WIDGET_REQUESTS item), tag them in Ch 7/9/11, and give `n_w` one sentence of
physical meaning + units before the formula that uses it.

---

### M6 — Ch 9 §5: two different quantities are both called "the residual", 10× apart and opposite-signed, in adjacent boxes

**Page/anchor:** `ch09-universe-factory.html` §5 "The identity that fails", the I9.4 reveal
vs the "Has this been tried?" box directly beneath it.

**Observed.** The reveal: *"Divide it out and a smooth monotone **residual** survives:
**−17.2%** end-to-end, +8.7% at h = 0.60 falling to −8.7% at h = 0.86."* The box
immediately below: *"**What remains after removing it** is a 1D-only **residual** of
**+1.667%**, i.e. +0.017 in h."*

These are different objects — a shape residual in `Σ_glob/β_G` across the h grid, and a
bias residual in h — but "what remains after removing it" reads as the same quantity, and
nothing on the page says otherwise. I stopped and re-read three times.

Second problem in the same box: *"it is exactly the h⁻³ volume Jacobian,
(0.73/0.81)³ − 1 = −26.80%"* is offered as the arithmetic justification for a ×2.48 drift
measured over h = 0.60 → 0.86. The pure-h³ expectation over that range is
(0.86/0.60)³ = 2.944, stated correctly two paragraphs earlier. The −26.80% is a carried-over
number from a different comparison and does not connect to anything on the page.

**Fix.** Rename one of them ("the shape residual" vs "the 1D bias residual"), state in one
clause that they are different quantities, and either drop the `(0.73/0.81)³` parenthetical
or say which pair of h values it compares.

---

### M7 — Ch 8's Q8.1 gives a *different and wrong* mechanism for the one-sidedness than §2 does

**Page/anchor:** `ch08-mass-channel.html` §2 predict-reveal vs Q8.1 answer (line ~1044).

**Observed.** §2 (correct): *"once σ_M ≳ M_g the lower edge M_g − σ_M goes negative: the
upper leg of the test becomes vacuous. A galaxy can always be 'too light for this
redshift'; it can essentially never be meaningfully too heavy."* That is right — the
overlap test `[M_g−σ_M, M_g+σ_M] ∩ [(M_z−2σ)/(1+z_max), (M_z+2σ)/(1+z_min)]` can only fail
high if `M_g − σ_M` exceeds the upper bound, and with σ_M ≈ M_g it never does.

Q8.1 (different, and not right): *"the physically-allowed condition M_z = M(1+z) ≥ M makes
the upper leg vacuous"*. `M_z ≥ M` bounds the GW-allowed source-frame mass from above at
≈ M_z; it says nothing about why the *galaxy's* upper test cannot fire. Nothing prevents a
GLADE galaxy with M_g up to 10⁷ M☉ being "too heavy" for an M_z ≈ 7×10⁵ event — except
exactly the σ_M ≳ M_g argument §2 gives.

Per the spec's rubric D, an answer that names the outcome with the wrong mechanism scores
2, and this is the answer to the chapter's own predict box.

**Fix.** Replace Q8.1's clause with §2's.

---

### M8 — Ch 6 states a nine-significant-figure mass precision without the plausibility check the chapter itself preaches, and never reports the 14×14 condition number

**Page/anchor:** `ch06-black-box.html` §4.1 and §6 ("Its detector-frame mass is known to
nine significant figures").

**Observed.** The chapter reports `σ_Mz/M_z = 1.36×10⁻⁹` for 889 and a population median
`(σ_Mz/M_z)·ρ = 2.57×10⁻⁶`, which implies **1.3×10⁻⁷ at ρ = 20**. Published EMRI Fisher
studies (Babak et al. 2017, arXiv:1703.09722 — the book's own population reference) quote
Δln M ≈ 10⁻⁵–10⁻⁶ at comparable SNR. This run is ~10–100× better than the literature, and
the chapter presents it as a fact with no comment.

Related gap: §3 reports condition numbers for the derived **3×3** (median 53.2, max 2813)
and **4×4** (median 2.3×10¹¹) blocks, and quotes the κ > 10¹⁴ gate — but the gate acts on
the **14×14** Fisher, whose conditioning is what actually limits the inverse, and the
chapter never shows its distribution. I measured it: over all 1590 rows the 14×14
covariance has **median κ = 2.6×10⁹, p95 = 1.4×10¹⁰, max = 3.9×10¹²**. That is comfortably
inside the gate — so the answer is reassuring — but the chapter that opens the black box
should be the chapter that shows it, especially since it is the number that decides how
many of those "nine significant figures" survive float64 inversion (at κ = 3.9×10¹²,
roughly three to four).

**Fix.** One sentence in §3 with the 14×14 κ distribution (it is a two-line addition to
`gen_ch06.py`), and one clause in §4.1 comparing 2.57×10⁻⁶ against Babak 2017's quoted
range with the honest verdict ("higher than the published forecasts by a factor of a few to
ten; whether that is the AK-vs-numerical-derivative difference or optimism of the
high-SNR Fisher is not tested here"). This is the chapter's own "measure before you
generalize" discipline applied to its own most extreme number.

---

### M9 — Ch 9 deprecates the literature-faithful normalization without saying whether the pathology is the mode's or this pipeline's

**Page/anchor:** `ch09-universe-factory.html` §3, the three-mode list.

**Observed.** *"`global` — the faithful discrete transcription of Gray's (A10).
**Deprecated and warned**: on photometric catalogues it is empirically mis-calibrated, ~0%
coverage, railing to the grid edge."*

That is a strong claim about the published method, stated in one clause, with no venue and
no diagnosis. From Ch 10 §1 I know the ~0% coverage figure belongs to the **bare host-z
kernel** at large σ_z — a different variable. So a reader cannot tell whether `global`
rails because A10 is wrong, or because A10 is being run with this pipeline's ±3σ ball,
its `volume_deconv` numerator, and its smooth-completeness deviation D2 — which is the
much more likely story given the museum's own Exhibit 3 (numerator/denominator asymmetry)
and the standing "equations are not modular" lesson from Exhibit 4.

**Fix.** Two sentences: name the venue, name which of the co-varying deviations were
present, and say explicitly whether the mode has ever been run in a configuration matching
the published machinery. If it has not, say *that* — it is a better sentence than the
current one and it is honest.

---

### M10 — Several predict-then-reveal beats are pre-spoiled by the chapter's own discovery statement and by the front door

**Page/anchor:** `ch08-mass-channel.html` (subtitle "and the bias goes to +0.077", ~100
words above `#ch08-predict-1` whose options are "sharper same centre / barely moves /
**moves the centre**"); `ch10-calibration.html` (subtitle "That is not the same as being
right", above a predict box whose options are "Yes / **No** / I can't tell");
`index.html` "The journey", which prints every chapter's discovery statement.

**Observed.** I could not make an honest prediction at either beat: the answer was on
screen. The Ch 6 dt² beat, the Ch 10 top-K beat, the Ch 11 Poisson-leverage beat and the
museum's quadrature beat are all genuinely blind and are excellent — the contrast is
stark.

**Fix.** For Ch 8 and Ch 10, move the discovery statement *below* the predict box (they
are both single-sentence subtitles; the cold-open widget can carry the chapter's tension
by itself). For `index.html`, consider a `details`-wrapped "spoiler: what each chapter
discovers" so the front door still works as a map without pre-answering four cold opens.

---

## [MINOR]

- **m1 — Ch 11 "The state of the art, in one paragraph" says "3130 detections"**
  (`ch11-honest-state.html:1154`); everywhere else, including line 791 and the generator's
  own gate, it is **3135** (1590 + 1545). This is the single most quotable paragraph in the
  book; fix the digit.

- **m2 — the bias rail looks different on Ch 6 and Ch 8 than everywhere else.**
  Ch 5/7/9/11 use the shared `Book.biasRail({pips})` (standardized amber `#D55E00`, with the
  "LIVE / UNRESOLVED" separator). Ch 6 (`ch06-black-box.html:76-84`) and Ch 8
  (`ch08-mass-channel.html:90-98`) still append page-local `.ch06-pip` / `.ch08-pip`
  elements styled `background: var(--planck)` and with no separator heading. Reading in
  order, the rail's amber dot changes colour at Ch 6, changes back at Ch 7, changes again at
  Ch 8. Convert both to `Book.biasRail({pips})` as the integrator did for the other four.

- **m3 — Ch 6's Q6.5 stem asserts the disputed number** ("the distance shell is thin to
  8×10⁻⁵") and the answer then dismantles it. I think this is deliberate and good pedagogy,
  but a reader who reads the question and skips the answer takes the wrong number away. A
  dagger on the stem ("† the units in this question are the point — see the answer") would
  cost nothing.

- **m4 — Ch 1's dossier calls the CRB mass column `M`, Ch 8 establishes it is `M_z`.**
  `ch01-ruler.html` prints "M = 7.246×10⁵ M☉" from the same column Ch 8's dossier prints as
  "M_z = 7.25×10⁵ M☉"; `ch08_FLAGS.md#F-ch08-8` records the discrepancy but Ch 1's page does
  not. Since Ch 8's whole point is that `M_z = M(1+z)` is the *measured* quantity, the
  running example should be labelled correctly from Ch 1.

- **m5 — no per-exhibit backlinks in the museum.** Arriving at `#ex-volume-trunc` from
  Ch 7's interlude, there is no way back except the global nav; by Exhibit 8 I had lost
  which chapter sent me. A "referenced by: Ch 7 §interlude" line per exhibit would fix it
  (BUILD_REPORT gap 6 — agreeing with it after actually walking the path).

- **m6 — `book/README.md` still names the retired `ch00-demo` files** (BUILD_REPORT §3;
  frozen to the integrator, so it is the orchestrator's one-liner).

- **m7 — Ch 9's dossier row is the weakest running-example beat in the book.** "889 is not
  special here" is honest, but the chapter *does* have a 889-specific number available
  (its own `w_G` under both estimands, which the row already prints) and the framing buries
  it. Minor, and only noticeable because every other chapter's dossier row lands.

---

## [PRAISE] — keep exactly as-is

- **P1 — Ch 7 §6 "Both sides, and who decides."** The best piece of scientific writing in
  the book and, I think, the reason the book exists. Two boxes, RATIFIED and FINDING, both
  sourced, neither softened; then the binding rule; then the pre-registered decider; then
  "What C7 is not" narrowing the scope twice. I have referee'd papers that would be
  improved by copying this structure verbatim. Do not let a later revision tidy it.

- **P2 — Ch 6's F6.1 is a real measurement used as a real bound.** I recomputed it from the
  CRB table: σ_u·ρ p5/median/p95 = **0.970 / 1.040 / 1.250**, spread **1.29**; 76
  in-catalogue rows with median σ_dL/dL = **5.3×10⁻³**. Exact, to every digit printed. And
  crucially the chapter then *uses* the spread as the licence bound on I6.1's SNR
  rescaling ("exact to ~30% for σ_u, only indicative for the sky widths") instead of
  leaving it as decoration. That is what rubric C's "lost-if-static" is supposed to mean.

- **P3 — Ch 6 §2, the dt² story.** "A normalization that multiplies a threshold changes
  which data exists" is the best single sentence in the book. The dimensional check as the
  *detection method*, the five evidence lines with L2 named as decisive because it removes
  the PSD and the Fourier convention from the argument, and the honest "an internally
  consistent pipeline has no way to notice a global rescaling" — this is how to write up a
  four-month-old factor-100 error without either minimizing or flagellating. My prediction
  at the reveal was "~1000× fewer" and the reveal told me precisely why my instinct was
  right and my geometry wrong (168×, z-cut at 1.5, (1+z) thinning). Perfect beat.

- **P4 — Ch 8 §5, the C4-amended budget.** `+15.83 = 0 (completion, cancels identically)
  + 19.10 − 3.27`, with the exactness argument (`p = C(1+R)`, `ln C` cancels between
  channels) *before* the numbers, and then "487 events zeroed at every h is arresting, true,
  and causally almost irrelevant; 534 events down-weighted by 5.8 is dull, true, and carries
  98.5% of the effect." That the refuted mechanism was the author-agent's own, and that it
  is said so plainly, is the most persuasive thing on the page.

- **P5 — Ch 9's I9.2 refuses to invent the cell that does not exist.** Tracing the JS
  (`ch09-universe-factory.html:1302-1314`), the `blind|blind` and `blind|aware` states
  return "No such run exists … the bench will not invent one", the realized readout prints
  the literal string `no realization`, and the binomial z is reported undefined. Best
  interactive in the book, and the only one I have seen anywhere that models *absence of
  data* as a first-class state. The verdict text even explains *why* the dial could not be
  turned ("could not be turned without rebuilding the catalogue the mock is made of").

- **P6 — Ch 10 §2c, the P1–P6 scorecard including the rows that lost.** P3 "PREDICTION
  MISSED" and P6 "NOT SCORABLE — no such counter is written" are the two rows a normal
  paper deletes. Printing them, and then saying "a prediction you did not write down is a
  prediction you will remember having made correctly", is the book's thesis in two lines.

- **P7 — Ch 10 §4's C11 block, and specifically the "Recomputed, and the upper endpoints do
  not reproduce" note.** The chapter re-runs the archived cells, finds +0.0078/+0.0157 where
  the claim says +0.0097/+0.0181, names the plausible pooling explanation, and then says
  the discrepancy *strengthens* the exoneration rather than weakening it — and still does
  not reconcile it. That is the correct handling and the correct tone.

- **P8 — the museum's F-museum-1.** An exhibit that reproduces its own flagship's
  published mechanism table digit-for-digit, then shows that the mechanism attribution is
  an artefact of `dist()` being scalar-only under `fixed_quad`, then ships **both**
  evaluation modes on the dial and adjudicates nothing — while explicitly preserving the
  falsification verdict, which rests on a production A/B. I went looking for the museum to
  quietly resolve this in its own favour and it does not. This is the sharpest thing in the
  build and the BUILD_REPORT is right to say so.

- **P9 — Ch 11 §5, the pre-registered cell B with all three readings registered in
  advance**, including "this is the predicted outcome, so it would be the weakest kind of
  confirmation" and "note this is the outcome that would teach the most". Followed by the
  no-answer-key block with "There is no *Show answer* control on this block, and adding one
  would be the single most dishonest thing this book could do." On the BUILD_REPORT's
  gap #4 (Q11.6 carries a model answer): I think the ch11 agent's call is right — "run the
  control" is a *discipline* answer, the project genuinely has it, and the five genuinely
  open questions sit immediately below it with no key. Leave it.

- **P10 — the arithmetic is honest.** Spot-checks that came back exact: binomial
  z for w_G (n = 3135, p = 0.1215037, k = 164 → **−11.86**); the C7 rail threshold solved
  from the corrected law for h_eff/h_true = 0.86/0.73 → **0.2645**, matching the page's
  "0.2644" and its refusal to round it into agreement with the artifacts' 0.256; F6.1's
  five quantiles; 889's production ball radius **0.7568 arcmin** (which is also how I
  caught B1). Rubric B is being earned, not asserted.

- **P11 — the predict-locks degrade correctly.** `Book.predictReveal`
  (`js/book.js`) gates only via a JS-applied `.is-predict-locked` class and every locked
  widget carries a `<noscript>` static fallback containing the answer. With JS off you get
  the answer and never a dead lock — which is the right trade, and the `data-hypothesis-verdict`
  attributes on the sandboxes (e.g. `ch07-redshift.html:564`, `:1087`) mean the museum
  meta-rule is enforced structurally rather than by discipline.

---

## Comprehension log (where I actually stumbled, in reading order)

| where | what happened |
|---|---|
| Ch 6 §4.1 | `σ_Mz/M_z = 1.36×10⁻⁹` — I stopped and checked the literature. See M8. |
| Ch 7 §6 | Followed cleanly on first read. The `φ_cat` in the C7 quote is undefined; I inferred it. See M4. |
| Ch 8 §1 | Hard stop: 10⁻⁴ here vs 10⁻⁹ one chapter earlier. See B3. |
| Ch 8 §6 | The "consistent unit change is exactly invariant / the measure is the problem" distinction is subtle and the page nails it. No stumble. |
| Ch 9 §3–§4 | The predicted danger zone. §3 is fine; §4's symbol load is the one place I re-read twice. See M5. |
| Ch 9 §5 | Genuine confusion: −17.2% vs +1.667% both called "the residual". See M6. |
| Ch 10 | No stumbles. The cleanest chapter in the book after Ch 7. |
| Ch 11 §1–§7 | No stumbles. §3's leverage derivation (`dh*/dε = −S'_in/S''_tot`) plus the idealized-venue contrast (0.024904 vs 0.000006) is the payoff Ch 5 promised and it lands. |
| Museum | No stumbles. Lost track of which chapter sent me by Exhibit 8. See m5. |

**Predictions I got wrong** (and would have gotten wrong in a seminar too): Ch 6 dt²
(said 10³×, measured 168×); Ch 10 top-K (said ~200, measured 40); Ch 11 Poisson leverage
(said ~0.005, measured up to 0.0249). All three reveals explained *why* the intuition
failed rather than just correcting it. That is the book working.

---

*Reviewer: "Tomas". Read-only pass; no book page or repository file was modified other than
this review file.*
