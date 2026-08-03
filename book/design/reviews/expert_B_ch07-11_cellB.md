# Expert review B — chapters 7–11 (+ museum), against the state as of 2026-07-31

Reviewer: Physics Expert B. Scope: ch07–ch11, the museum, and cross-page consequences.
Read-only on every book page and on `MasterThesisCode`. No git operations.

**The single fact this review exists to deliver: cell B landed after the book was built.**
`results/campaign51_20260728/realistic_20260729/CELLB_READOUT_20260731.md` (dated
2026-07-31) fills the 2×2, and `CLAIM_2D_BIAS_20260730.md` C6 was rewritten the same night
to **RESOLVED — THE ESTIMATOR OWNS IT**. Every page that says the control is "in flight",
"not landed", or that item 1 of the last question block has no answer is now **factually
false**, not merely stale. That is why most of the BLOCKER section is one story.

I re-verified the readout independently rather than trusting it (see §0), and I have
re-run at least three published gates per chapter. Where the flag index
(`BUILD_REPORT.md` §5) already records a dispute I did **not** re-litigate it, except where
cell B changes its status or where a chapter *silently prefers one side* of an
already-flagged pair — which BUILD_REPORT §8 explicitly asks the expert reviewer to check,
and which I found in two places.

---

## §0 — Independent re-verification of cell B (so the revision can be written from measured numbers)

Run with `/home/jasper/Repositories/MasterThesisCode/.venv/bin/python` from the main
checkout. `cellb_readout.py` reproduces line for line; I then recomputed the pieces it does
not print.

**Reproduced from `cellb_readout.py` (exact):**

| quantity | cell B | comparison |
|---|---|---|
| 1D MAP / mean / σ_h / edge-peak | **0.7450** / 0.7320 / 0.0255 / 1.1e−8 | A = 0.7299, C(r1) MAP 0.7400 |
| 2D MAP / mean / σ_h / edge-peak | **0.7900** / 0.7962 / 0.0187 / 1.2e−2 (interior) | A = 0.7300, C(r1) 0.8133 |
| B − A (estimator) | **+0.0151 (1D) / +0.0600 (2D)** | |
| C − B (scatter) | **−0.0050 (1D) / +0.0233 (2D)** | |
| class-summed argmax, 1D | in-cat **0.860**, dark **0.640** | identical structure to every #53 run |
| class-summed argmax, 2D | in-cat 0.860, dark **0.800** | |
| channel difference 0.73→0.81 | in-cat **−1.80**, dark **+18.00**, total **+16.20** | C(r1): +2.97 / +15.83 / +18.80 |
| combined per-event in-cat 1D argmax at 0.86 | **53/76 = 69.7%** | C(r1) 57.9%; #51 estimator 5.3% |

**Recomputed by me, not printed by the readout script:**

- **Catalogue-leg-only rail** (`L_cat_no_bh` argmax, in-cat events, positive on the whole
  grid): **B 68/75 = 90.7%** vs **C(r1) 66/74 = 89.2%**. Median 0.86 in both. This is the
  90.7 / 89.2 pair the readout quotes, and it reproduces.
- **`w_G(h)` bit-identity**: B vs `real_r1`, all **41** grid points, `max|Δ| = 0.0`,
  element-wise `==` **True**. The pre-registered "expected bit-identical" read is exact, not
  approximate.
- **In-cat median `L_cat` share of the 1D mixture at h = 0.73**: B **0.9628** vs C(r1)
  **0.9631** — Ch 7's "96.3%" is cell-B-stable.
- **C4-amended observables, B vs C(r1)** *(reviewer-computed; not in any adjudicated
  artifact — flag as such if used)*:

  | | C = r1 (scattered) | B (unscattered) |
  |---|---|---|
  | dark frac with `L_cat_with_bh == 0` at 0.73 | 0.647 | **0.855** |
  | events with a live 1D catalogue leg anywhere | 1095 | 982 |
  | of those, 2D-zero at every h (dark) | 488 (487) | **688 (687)** |
  | dark mean catalogue mixture weight 1D → 2D | 0.0354 → 0.0061 (×5.79) | **0.0361 → 0.0043 (×8.39)** |
  | dark channel-diff carried by always-2D-zero events | +0.24 = **1.5%** | +3.46 = **19.2%** |
  | …by both-dead events | 0.00 | 0.00 |
  | …by survivors | +15.60 = **98.5%** (n = 534) | +14.53 = **80.7%** (n = 219) |

  Reading: the mass sieve deletes **more**, and de-weights **harder**, without any realized
  scatter. C4's amended mechanism ("de-weighting, not deletion") **survives** on an
  independent configuration — but the headline 98.5%/1.5% split is *r1-specific* and moves
  to 80.7%/19.2% in B. Ch 8 currently states 98.5% as if it were the mechanism's signature.

**Other gates re-run (all reproduce exactly):** EMRI-889 CRB row → σ_dL = 7.9843e−5 Gpc,
σ_dL/dL = **8.9833e−4**, M = 724631.5, μ = 10, SNR = 1424.72, host 859360, in_catalog True,
76/1590 in-cat. C7 law `[1+√(1+12ε²)]/2` reproduces every `leg_a` row of `ch07_c7.json`;
solving it for the 0.86 edge gives **0.2644**; host ε quartiles 0.3794/0.5193/0.6445;
frac ε > 0.256 = 0.9868; observed in-cat tilt median +0.3082, 93.24% positive.
C1/C2/C3 on r1: 1D +2.48/−11.77/−9.30, 2D +5.45/+4.06/+9.51, channel diff
+2.97/+15.83/+18.80. C4 observables 64.7% / 32.5% / 1095 / 488 / 487, f̄ 0.0354→0.0061.
C9: 76/1590 + 88/1545 = 164/3135 = 0.052313, binomial z = **−11.86**, mass-aware z = **+0.21**.
G4 counterfactual: 1D 0.7321→0.6430, 2D 0.8123→0.7433. C5 Poisson: max|shift| 0.024904
(seed61000/r5) vs 0.000006 / 0.000015 idealized. 889's swing +1.9832 / −2.0347 / −3.3004.
1D MAP range over the ten delivered runs: 0.700–0.740, exactly as Ch 11's table says.

---

## [BLOCKER]

### BL-1 — ch11, "The questions with no answer key", item 1: the book's closing honesty device is now the book's biggest false statement

**Page/anchor:** `ch11-honest-state.html:1313–1342` (`<h2>The questions with no answer key</h2>`,
`div.selfcheck.ch11-noanswer`, list item 1).

**Observed.** Item 1 reads *"Did cell B land at B ≈ C or B ≈ A? Which of C7, C9 transferred
to the delivered posterior?"*, inside a block whose closing line is *"There is no Show
answer control on this block, and adding one would be the single most dishonest thing this
book could do."*

**Expected.** As of 2026-07-31 the project **has** an answer, measured and written up. The
block's own rule ("the absence is the point: this is the live edge of the work on the day
the book was written") now inverts: leaving item 1 unanswered is the dishonest act.

**Fix.** Delete item 1 from the no-answer list and replace it with a *dated resolution
card* immediately above the block, so the arc survives:

> **Resolved after this chapter was drafted — 2026-07-31.** Cell B landed (jobs
> 6103219/6103220, code `7fd60bb`, unscattered parent catalogue through the #53 estimator).
> **1D MAP 0.7450, 2D MAP 0.7900.** B − A = **+0.0151 / +0.0600**; C − B = **−0.0050 /
> +0.0233**. The pre-registered "estimator owns it" outcome. The remaining four questions
> still have no answer.

Then renumber the block to four items and change the closing line to *"Four of the five
questions this chapter opened still have no answer. The fifth was answered by running the
control — which is the only reason it is no longer on this list."* Do **not** delete the
original question text from the page: keep it visible as the question that *was* open,
struck through or in a "what this chapter asked" line. The pre-registration beat is
strictly stronger when the reader can see the question, the registered prediction, and then
the answer.

---

### BL-2 — ch11 §5: the 2×2 is shipped with an empty cell and an [OPEN] stamp

**Page/anchor:** `ch11-honest-state.html:869` (`<h2>5. The confound, and the control that is
still running</h2>`) → `div.ch11-prereg` at `:920–959`, table cell `:934`
(`B = <em>this run — in flight</em>`), stamp `:956–957` (`<span class="badge open">OPEN</span>
— at the time of writing this chapter, cell B had not landed. The book does not report a
result it does not have.`).

**Observed.** The chapter's centrepiece table has a hole, and asserts as a matter of policy
that the result does not exist.

**Expected.** It exists, and the book *does* have it.

**Fix.** Keep the entire pre-registration block **verbatim** — it is the best pedagogy in
the chapter and it must not be edited into hindsight — and append a second, visually
distinct **readout block** below it:

```
                point / generator_marginal    volume_deconv / absolute_marginal
unscattered     A = #51: 1D 0.7299, 2D 0.7300  B = 1D 0.7450, 2D 0.7900   ← landed 2026-07-31
scattered       forbidden by guard             C = #53 r1: 1D 0.7400, 2D 0.8133
```

with the two differences printed as the chapter's own arithmetic:
**B − A (estimator) = +0.0151 / +0.0600**, **C − B (scatter) = −0.0050 / +0.0233**,
and the 2D apportionment **+0.060 of the +0.0833 total = 72% estimator / 28% scatter**.

Change the section title to **"5. The confound, and the control that resolved it"** and the
badge from `OPEN` to `RESOLVED 2026-07-31` (or add a `finding` badge beside the retained
`open` one with the date). Chip it to
`CELLB_READOUT_20260731.md` and to `CLAIM_2D_BIAS_20260730.md` C6 *as amended 2026-07-31*.

**Two things the revision must not get wrong here** (see BL-4 and MJ-1).

---

### BL-3 — ch07 §6 "Both sides, and who decides": "It has not landed" is false, and the correct replacement is *not* "C7 is settled"

**Page/anchor:** `ch07-redshift.html:775` (`<h3>Both sides, and who decides</h3>`),
paragraph `:819–831`, ending **"It has not landed."**; provenance panel item `:1194–1195`
(`OPEN … not yet landed`); rail pip `:1280`; closing paragraph `:1155–1161` ("the deciding
experiment pre-registered and not yet run"); Trap 7.B `:1051–1052`; `data/ch07_c7.json →
conflict.decider = "… registered 2026-07-30, pending"`.

**Observed.** Five separate assertions on one page that the decider has not run.

**Expected.** It ran, and it delivered exactly the reading the paragraph registered in
advance — *for the C7 magnitude*. It did **not** resolve the G2b↔C7 collision, which is a
derivation-level conflict that no posterior can settle.

**Fix.** Replace the final sentence and add a measured block, keeping the registered
readings above it untouched:

> **It landed, 2026-07-31.** With **exact host redshifts** and the true cluster parent's
> `z_error` column — the staleness-free check this section was waiting for — the
> catalogue leg's per-event argmax sits at the top of the prior for **68 of 75** in-catalogue
> events (**90.7%**), against **66 of 74 (89.2%)** in the scattered run and **5.3%** under the
> idealised estimator. The in-catalogue *class* argmax is **0.860**, exactly as registered.
> The realized scatter is not what rails the catalogue leg; if anything it damps the
> *combined* rail (69.7% in B against 57.9% in C).
> <chip>CELLB_READOUT_20260731.md</chip> <chip>CLAIM_2D_BIAS C6 (resolved 2026-07-31)</chip>

Then, immediately, the constraint that survives:

> What cell B settles is the **magnitude and the attribution** of C7's effect, not the
> collision. G2b's confirmation is untouched: the fix is still author-gated, must
> **explicitly supersede G2b**, and — a constraint added by the cell-B readout — **must not
> be the historically-exonerated "`p_det` inside the numerator alone" form**
> (`CELLB_READOUT_20260731.md` §Consequences / Next steps 1b). Chapter 7 still may not say
> "the kernel is wrong, settled".

That last clause is new information the book does not currently carry anywhere, and it
belongs in ch07 §6 and in ch11's C7 subsection (`ch11-honest-state.html:706–729`), which
ends with "The decider is the pre-registered cell B … (§5)".

**Also update, with the same numbers:** the rail pip (`:1280`) from grey/"not landed" to
amber/"cell B: 90.7% catalogue-leg rail with exact z (2026-07-31)"; the provenance panel
`OPEN` line to a dated `FINDING`; Trap 7.B's "precisely what cell B was pre-registered to
decide" → "…was pre-registered to decide — and did, on 2026-07-31"; and
`ch07_c7.json → conflict.decider` from `"pending"` to the landed values, since that string
is read by the widget.

**One honest nuance the fix must carry** (this is the chapter's own flag culture applied to
itself): the *indicative* local `z_error` column predicted **75/76 = 98.7%** of hosts
peaking above 0.86 (`ch07_c7.json → hosts.frac_peak_above_086`, which the noscript fallback
quotes). The staleness-free measurement gives **90.7%**. The two are not the same statistic
(reconstructed unclipped single-host peak vs delivered clipped `L_cat` argmax), so this is
not a contradiction — but the honest sentence is *"the staleness caveat resolves in the
confirming direction, with the delivered rail somewhat weaker than the stale column
implied"*, not *"98.7% confirmed"*.

---

### BL-4 — nobody has flagged that the pre-registered **1D band was missed**; "confirmed on every read" is one step too strong

**Page/anchor:** `ch11-honest-state.html:943–952` (the dated pre-readout statement,
"1D ≈ 0.70–0.74 as a crossing"); `ch10-calibration.html:877–880`; the reveal at
`ch11-honest-state.html:976–980`.

**Observed.** The registered prediction has three numeric reads: 2D ∈ 0.78–0.82,
in-cat class argmax ≈ 0.86, **1D ∈ 0.70–0.74**. Cell B delivered 2D = **0.7900** ✓,
in-cat class argmax = **0.860** ✓, 1D MAP = **0.7450** — **one grid step above the
registered band**. `CELLB_READOUT_20260731.md` says "confirmed on every pre-registered
read", but it silently substitutes "an interior crossing" for the numeric 1D band it is
scoring against.

**Expected.** A book whose thesis is *pre-register, then report what you registered* cannot
reproduce a softened self-scoring. This is not a defect in the result — it is the most
valuable teaching beat available, and it makes the pre-registration stronger, not weaker.

**Fix.** In ch11 §5's readout block, score the prediction explicitly, three lines:

> **2D ∈ [0.78, 0.82] → 0.7900 ✓**
> **in-cat class argmax ≈ 0.86 → 0.860 ✓**
> **1D ∈ [0.70, 0.74] → 0.7450 ✗ by one grid step** (the 1D *mean*, 0.7320, is inside the
> band, and the band was derived from the ten delivered runs' 1D **MAPs**, 0.700–0.740 —
> so the miss is real on the convention the band was written in).

Then the sentence that earns the chapter its title: *the registered prediction was right on
two reads out of three, and the third missed by the smallest resolvable amount — which is
what a pre-registration is for, and is why the readout's "confirmed on every read" is the
one sentence in it this book should not copy.* Add a one-line note that the project's own
readout should carry the same correction (out of book scope, but worth a flag file entry).

---

### BL-5 — four pages assert σ_dL/dL = 8.0×10⁻⁵ as fact, with no flag; only ch07 prints the measured value

**Page/anchor:** `ch08-mass-channel.html:801`, `ch10-calibration.html:735`,
`ch11-honest-state.html:1018`, `museum.html:1316` — all in the EMRI-889 dossier row:
`d_L 88.9 Mpc (σ_dL/dL = 8.0×10⁻⁵)`. Ch 7 prints **8.98×10⁻⁴** with FLAG-1
(`ch07-redshift.html:894–908`).

**Observed.** BUILD_REPORT §5.1 item 1 states *"Chapters print both values everywhere"* and
§8 asks the expert reviewer to *"verify no page quietly prefers one"*. Four pages prefer
the spec value, silently, in the running example's own identity card — the single most
repeated number in the book.

**Verified independently** (row 889 of
`seed61000/prepared_cramer_rao_bounds.csv`): σ_dL = **7.9843×10⁻⁵ Gpc**, d_L = 0.0888792
Gpc, **σ_dL/d_L = 8.9833×10⁻⁴**. The spec's figure is the absolute Gpc value carried under
a fractional label. Event identity is not in doubt (M, μ, SNR, host index all match).

**Fix.** In all four dossiers, print `σ_dL/dL = 8.98×10⁻⁴ (spec card: 8.0×10⁻⁵ — see
ch07 FLAG-1)`, or at minimum `8.0×10⁻⁵ [disputed → Ch 7 flag]` with a link to the ch07
provenance note. One of the two must be visible on every page that quotes it; a dossier row
is exactly where a reader forms the number they will carry.

---

### BL-6 — ch11's Adjudication Board no longer transcribes the claim file verbatim, which is the board's own advertised contract

**Page/anchor:** `ch11-honest-state.html:219–229` (the "How to read the board" block:
*"Every status string in the board below is transcribed **verbatim** from the claim file's
section heading or the adjudication's per-claim verdict"*), driving `data/ch11_board.json`.

**Observed.** `CLAIM_2D_BIAS_20260730.md` was amended on 2026-07-31. C6's section heading
is now:

> `## C6 — ~~Attribution is confounded; the decisive control was never run~~ **RESOLVED 2026-07-31: cell B ran — THE ESTIMATOR OWNS IT** [DOC + INFER → LOCAL, MEASURED]`

The board still carries `"status": "FINDING (confirmed by Gate A1); resolution in flight"`,
`"badge": "confounded"`, `"tag": "[DOC + INFER]"`, `"live": true`, and an `adjudication`
field ending *"jobs 6101146 / 6101147"*. C9 still carries
`"FINDING [LOCAL, VERIFIED] — live, gated on cell B"`; that gate is released
(`CELLB_READOUT §Consequences`: "cell B removes its last external gate"). C7's
`adjudication` still ends *"Cell B is the staleness-free magnitude check."*

**Expected.** Verbatim means verbatim as of the artifact's current version, or the contract
sentence must be re-scoped to a date.

**Fix (in `gen_ch11.py` → `ch11_board.json`, per claim):**

- **C6** — `status` → the new heading text; `badge` → `finding` (or a new `resolved`
  tone); `tag` → `[DOC + INFER → LOCAL, MEASURED]`; `live` → `false`;
  `adjudication` → the C6 resolution paragraph (B 1D 0.7450 / 2D 0.7900; B−A +0.015/+0.060;
  C−B −0.005/+0.023; 72% of the 2D displacement; catalogue-leg rail 90.7% vs 89.2%; dark
  channel difference +18.0 nats unscattered; `w_G` bit-identical); `refute_by` → *"run the
  unscattered catalogue through the #53 estimator — DONE, and the pre-registered prediction
  held"*; chips → add `CELLB_READOUT_20260731.md`.
- **C9** — `status` → `FINDING [LOCAL, VERIFIED] — live; the cell-B gate is released, the
  fix is author-gated`; keep the leverage discount and the ledger-#61 re-litigation guard
  verbatim.
- **C7** — append to `adjudication`: *"Cell B (2026-07-31) supplied the staleness-free
  magnitude check: 90.7% catalogue-leg rail at the true parent widths, with exact host
  redshifts. The G2b collision is unchanged; the fix must supersede G2b and must not take
  the exonerated 'p_det inside the numerator alone' form."*
- The board's **live count** falls from five to four (C5, C7, C8, C9). Both the
  `<noscript>` fallback and the widget's count line must move together.

---

### BL-7 — ch10 §5 tells the reader to go to Ch 11 for an unresolved prediction that is now resolved

**Page/anchor:** `ch10-calibration.html:838` (`<h2>5. The control that was never run</h2>`),
paragraph `:873–881` — *"It is in flight. Chapter 11 shows you that prediction from the
inside, unresolved, because the project has not resolved it either."*; the following
callout `:883–891` (*"one confounded attribution whose decisive control has not landed"*);
provenance items `:1069–1070`; the rail pip `:1149–1150`; Q10.5's answer `:1978` — *"you can
conclude **nothing** about which one"*.

**Observed.** A forward promise the destination chapter can no longer keep.

**Fix.** Keep §5 exactly as the design lesson it is — the *"design the missing control
yourself"* paragraph is the best thing in ch10 and must not be spoiled by an early reveal.
Change only the last two sentences of `:873–881` to:

> It was submitted, and it landed on 2026-07-31 — after the pre-registration, and after
> this chapter's prediction was written down. Chapter 11 shows you the prediction from the
> inside first, and then what it got right and what it missed.

Callout `:883–891`: *"…and one attribution that was confounded until the control was run"*.
Rail pip: grey/"not landed" → a dated entry. Provenance `OPEN … in flight` → `FINDING —
landed 2026-07-31`.

**Q10.5** is a different case and must be handled as history, not overwritten: the answer
("you can conclude nothing about which one") was **correct at the state the question
describes**. Keep it, and append a dated postscript:

> *Postscript, 2026-07-31.* The missing cell was run. With the catalogue held unscattered
> and only the estimator switched, the 2D MAP moves 0.7300 → **0.7900** — **72%** of the
> full displacement. The answer to "which one" is now *the estimator configuration*; the
> answer to "what could you conclude before the control" is still **nothing**, and that is
> the transferable half.

The same pattern applies to **Q11.6** (`ch11-honest-state.html:1280–1309`, "Run the
control"): the discipline answer stays, with a postscript noting the control was run and
that a *confirmed* prediction is the weakest kind of confirmation — which is exactly why
the fix program is still author-gated rather than shipped.

---

## [MAJOR]

### MJ-1 — the 2×2's C cell mixes a **mean** with a **MAP**; once B is filled the reader will compute the wrong scatter effect

**Page/anchor:** `ch11-honest-state.html:936` (`C = #53 r1–r5: 1D 0.732, 2D 0.813`),
inherited from `PREREGISTRATION_2x2_cellB.md`.

**Observed / verified.** For `seed61000/real_r1`: 1D **MAP = 0.740**, 1D **mean = 0.7321**;
2D MAP = 0.8133. So the C row currently prints the 1D **mean** beside the 2D **MAP**. Cell
B: 1D MAP 0.7450, 1D mean 0.7320. If the reader fills B with 0.7450 next to C = 0.732 they
get a scatter effect of **−0.013**; the readout's is **−0.0050**. And on the *mean*
convention the 1D scatter effect is **−0.0001** (0.7320 vs 0.7321) — i.e. the estimator
owns essentially **100%** of the 1D displacement, not 100%+ or 150%.

**Expected.** One convention per table, stated.

**Fix.** Print the table in MAPs throughout (A 0.7299/0.7300, B 0.7450/0.7900, C
0.7400/0.8133 — r1, the cell the diagnostics belong to) and add a one-line footnote giving
the means (A —, B 0.7320/0.7962, C 0.7321/0.8123) and noting that the 1D estimator/scatter
split is **+0.0151 / −0.0050 on MAPs** and **≈ +0.002 / ≈ 0.000 on means**. Also note
explicitly that the 1D estimator share **exceeds 100%** of the total 1D displacement
(+0.0151 of +0.0101) because the realized scatter pushes the *other way* — the "72%"
headline is a **2D-only** number and must never be printed as a both-channel summary.

### MJ-2 — the delivered run is jobs **6103219/6103220**, not 6101146/6101147

**Page/anchor:** `ch11-honest-state.html:927` and `:1350`; `ch10-calibration.html:1150`;
`ch11_board.json` C6 `adjudication`.

**Observed.** 6101146/6101147 failed on a missing raw-CRB symlink in the run-dir setup and
were resubmitted as **6103219/6103220**; the pre-registration was unchanged and the failure
was pure plumbing (`CELLB_READOUT_20260731.md` header, `cellb_readout.py` docstring).

**Fix.** Where the *pre-registration* is quoted verbatim, keep 6101146/6101147 (that is
what was registered). Where the *result* is reported, cite 6103219/6103220, and add one
sentence: *"the first submission failed on a missing symlink in the run-dir setup and was
resubmitted without touching the test design — recording that is cheaper than being asked
about it later."* That detail is on-message for this book.

### MJ-3 — "Cell B" is now two different objects with nearly identical 2D numbers

**Page/anchor:** `data/museum_ledger.json` row **#88** (`"…Cell B (broadened volume_deconv
numerator + generator norm): 1D = 0.73 truth, 2D = 0.80 INTERIOR HIGH, +29.4 ln"`, venue
seed1000 deep), surfaced by the ledger search panel on every chapter page; vs the 2×2 cell B
(2D = **0.7900**, campaign venue). Ch 11 `:991–1001` calls the *same* historical run
**"cell A′"** (following `BIAS_HISTORY_LEDGER.md` §3), while ch07 §5 cites ledger #88 for
the 85.3%/86.7% split.

**Observed.** A reader who types "cell B" into the ledger search after reading ch11 §5 gets
a row whose 2D number is 0.80 on a different venue and whose "cell B" means a different leg
of a different A/B. With B = 0.7900 landing, the collision becomes actively misleading.

**Fix.** Rename the 2×2 object consistently as **"the 2×2 cell B"** wherever it appears in
ch07/ch09/ch10/ch11, and add a disambiguation line to museum ledger row #88's verdict
string: *"(this 'Cell B' is the three-way A/B's leg B on the seed1000 deep venue — not the
2026-07-31 2×2 cell B; see Ch 11 §5)"*. Ch 11's off-venue-analogue paragraph should say
"ledger #88's cell B, which `BIAS_HISTORY_LEDGER` §3 relabels A′".

### MJ-4 — ch08 §4–§5 needs cell B: the mass de-weighting is **estimator-borne**, and the 98.5%/1.5% split is r1-specific

**Page/anchor:** `ch08-mass-channel.html:603–610` (*"Chapter 7's realistic redshift kernel
is what switched the mass kernel on … which is also, precisely, why the attribution is
confounded ⏭ Ch 11"*); §5 `:633` onward; Q8.3's answer; the "487 events carry 1.5%"
statement at `:670–680` and in the C4 block.

**Observed.** ch08 correctly attributes the switch-on to the *kernel*, but ends by handing
the reader a confound that no longer exists, and it presents the 98.5%/1.5% survivors/deleted
split as the mechanism's signature.

**Expected.** Cell B gives ch08 the two things it currently has to defer:
(i) the dark channel difference is **+18.00 nats with a completely unscattered catalogue**
(vs +15.83 scattered) — the de-weighting channel is estimator-borne, closing the "is the
dark de-weighting scatter-induced?" question definitively; (ii) a **second diagnostics CSV**,
i.e. the first non-r1 C4 partition the adjudication asked for.

**Fix.** In §5, after the "what replicates and what does not" block, add:

> **Cell B, 2026-07-31 — the same accounting with the scatter removed.** With the
> unscattered parent catalogue the dark class's channel difference is **+18.00 nats**
> (in-cat −1.80, total +16.20), against +15.83 / +2.97 / +18.80 scattered. The mass channel
> does not need realized noise to do this.
> <chip>CELLB_READOUT_20260731.md</chip>

and in the C4 block, replace *"98.5% is carried by the 534 survivors"* with the
configuration-scoped pair — *"on r1, 534 survivors carry 98.5% and the 487 always-zero
events carry 1.5%; on cell B's unscattered configuration the same partition reads 219
survivors / 80.7% and 688 zeroed / 19.2%. Deletion is a minority carrier in both; the
precise share is not a constant of the mechanism."* Mark this as recomputed for the book
(it is not in any adjudicated artifact — I computed it in §0 above) and raise it in
`ch08_FLAGS.md` so the author can decide whether it becomes a claim-file amendment.
Finally, change the §4 closing pointer from *"why the attribution is confounded"* to
*"why the attribution was confounded — until the 2×2's cell B separated them ⏭ Ch 11 §5"*.

### MJ-5 — ch09's re-litigation guard and rail pip still gate C9 on a landed control

**Page/anchor:** `ch09-universe-factory.html:898–906` (*"C9 is live; the fix is gated on the
2×2 cell-B control, which had not landed when this chapter was written"*), rail pip `:1167`,
provenance `:1068` (`OPEN … in flight at the time of writing`), verdict string `:1300`
(*"it is gated on cell B"*).

**Observed.** The gate is released. `CELLB_READOUT_20260731.md` §Consequences: *"cell B
removes its last external gate; the remaining input is the author's leg-adjudication."*

**Expected.** C9 stays **live** (the defect is not fixed) but is no longer *blocked* — the
remaining gate is `/physics-change` plus the author's leg adjudication, and the fix must be
the **joint C9+C8 mass-consistent mixture**, never a piecewise patch.

**Fix.** Rewrite the guard's last sentence: *"C9 is live. The cell-B gate was released on
2026-07-31; what remains is the author's leg-adjudication and a `/physics-change`
derivation — and the adjudication is binding that C9 and C8 must be fixed **jointly**,
because their counterfactuals act on different terms of the same mixture and are not
additive."* Same edit at `:1300`.

**And give ch09 its own cell-B win**, because it is the cleanest pre-registration payoff in
the book and it is *this chapter's quantity*: §6's paragraph at `:908–917` quotes the
registered line *"w_G(h): expected bit-identical to the #53 runs … If it differs, that
itself is a finding."* It is now checked against cell B and it is **exactly bit-identical
across all 41 grid points** (I verified `max|Δ| = 0.0`, element-wise equality True;
0.1625175 / 0.1215039 / 0.1038732 at h = 0.60/0.73/0.81). Say so, with the date. A
prediction registered at 7 significant figures and met bit-for-bit is worth a sentence.

### MJ-6 — ch11's headline scoreboard quietly prefers the un-reproduced 2D pull range that ch08 flagged

**Page/anchor:** `ch11-honest-state.html:160–161` — 2D row, `pull vs truth`:
**+3.4 … +4.5 (mean +4.04)**, chipped `REALISTIC_READOUT.md §6`, no flag.

**Observed.** `ch08_FLAGS.md` F-ch08-1 records that this range does not reproduce, and
ch08 prints the recomputed one (`ch08-mass-channel.html:203`: *"pulls +2.47…+4.74 (mean
+4.04)"*). I re-verified from `realistic_scores.csv` column `pull_2d`: n = 10,
mean **+4.0388**, min **+2.474**, max **+4.735**, |pull| > 2 in 10/10, 2D MAP range
0.78–0.82, mean 0.807, bias +0.077. So the mean and the count reproduce; the **range does
not**.

**Expected.** BUILD_REPORT §8's rule — no page quietly prefers one side of a flagged pair —
applied to the chapter whose entire thesis is that rule.

**Fix.** Ch 11's table row → `+2.47 … +4.74 (mean +4.04)` with a footnote *"the readout
prints +3.4 … +4.5; recomputed from `realistic_scores.csv` the range is +2.474 … +4.735 —
mean and 10/10 count reproduce exactly. See ch08 FLAG F-ch08-1."* This is the highest-value
single-line fix in the chapter: the opening table is where a reader calibrates how much to
trust the rest.

### MJ-7 — "Four are live: C5, C6, C7, C8, C9" lists five

**Page/anchor:** `ch11-honest-state.html:297` (noscript static fallback); cross-check
`:305–307` (the GW-reader block lists the same five without a count) and
`ch11_board.json` (`live: true` on C5, C6, C7, C8, C9 → **five**).

**Observed.** Plain counting error, visible to every no-JS reader and to anyone who counts.

**Fix.** After BL-6 the correct number is **four** (C5, C7, C8, C9) — so this is fixed by
the same edit, but the count word must be changed deliberately rather than left to
coincide. Also check the widget's count line and `index.html:145–149`, which says *"three
live, measured inconsistencies … plus one confounded attribution (C6 … the control is in
flight)"* — after cell B that reads *"three live, measured inconsistencies … plus a fourth
(C5), and an attribution that was confounded until the control was run on 2026-07-31."*

### MJ-8 — museum: two cell-B-dependent statements, one of which is the museum's own live-claim marker

**Page/anchor:** `museum.html:586–591` (*"the measured half of claim **C9**, which the
adjudication leaves open pending the cell-B control"*) and `museum.html:1305–1310`
(*"**w_G is not on the list** — C9 is live, gated on cell B"*).

**Observed.** Both are correct statements about the 2026-07-30 adjudication and false about
the current state.

**Fix.** Keep both sentences and date-scope them: *"…which the 2026-07-30 adjudication left
open pending the cell-B control; that control landed on 2026-07-31 and released the gate —
C9 remains live and unfixed, and `w_G` stays off the exonerated list."* The museum's meta-rule
("nothing interactive lets a dead hypothesis look alive") has a mirror image that matters
here: nothing static should let a *resolved* question look open.

---

## [MINOR]

### MN-1 — `ch07_c7.json → conflict.decider` is a data-layer string the widget reads

`"decider": "cell B (PREREGISTRATION_2x2_cellB.md, registered 2026-07-30, pending)"`.
Update in `gen_ch07.py` to carry the landed values so the page and the data agree; the JSON
is the thing a future reader will grep.

### MN-2 — `ch07_c7.json → hosts.staleness_caveat` ends "Cell B is the staleness-free magnitude check"

Same file. Append the answer: `"resolved_by_cellB": {"date": "2026-07-31", "lcat_rail_frac":
0.907, "n": "68/75", "comparison_scattered": 0.892}`. The noscript fallback at
`ch07-redshift.html:597–603` should gain one sentence with the same numbers, since the
static reader currently gets the stale-column prediction with no resolution.

### MN-3 — ch11 `<meta name="description">` and subtitle

`ch11-honest-state.html:11` and `:133–135` both end on *"the control still in flight" /
"the control still running"*. Update to *"…and the control that landed the night the book
was built"*. Cosmetic, but it is the page's social-card text and the first line the reader
sees.

### MN-4 — ch11 §7's closing verdict paragraph

`:1150–1161`: *"the one experiment that separates them was pre-registered, submitted, and
had not landed when this was written."* → *"…was pre-registered, submitted, and landed:
the estimator configuration owns 72% of the 2D displacement, and the realism layer is
largely exonerated. That is a result. What to do about it is still a position."* Note the
rhetorical structure of the paragraph survives intact — it just gets a better ending.

### MN-5 — grey "not landed" rail pips on four pages

`ch07:1280`, `ch09:1167`, `ch10:1149–1150`, and ch11's equivalent. All four should become
a single dated amber/finding pip with identical wording so the rail reads consistently
across chapters: `"cell B (2026-07-31): estimator owns +0.060 of the 2D +0.083"`.

### MN-6 — ch07 §5's ledger #88 citation, after MJ-3

`ch07-redshift.html:640–642` quotes *"the δ-kernel carries 85.3% (1D) / 86.7% (2D) of the
total ln movement"* from ledger #88. That is the seed1000 deep venue. Cell B now gives the
**on-venue** number for the same question — the whole #53 estimator (kernel + normalization
together, not the kernel alone) carries **+0.060 of +0.083 = 72%** of the 2D displacement.
Worth one sentence, carefully scoped: the two are not the same decomposition (per-leg vs
whole-estimator, deep vs campaign venue), and the chapter should say so rather than let a
reader read 86.7% and 72% as a disagreement.

---

## [PRAISE]

### PR-1 — ch07 FLAG-2 is exactly right, and I could not break it

`ch07-redshift.html:833–845`. The chapter prints the artifacts' 0.256 as *the* threshold,
draws the reader's crossing where the published law actually puts it, states the ~3% gap,
and refuses to reconcile. I independently solved `[1+√(1+12ε²)]/2 · 0.73 = 0.86` → **0.2644**
and interpolated the delivered per-host medians (0.8476 at ε = 0.25, 0.9390 at 0.35) →
**≈0.264**. The flag is correct, the disposition is correct, and the closing line — *"A book
about estimator honesty does not get to round its own numbers into agreement"* — is the best
single sentence in chapters 7–11. Keep it verbatim.

### PR-2 — ch07 §6's three-bullet "why this is a measurement, not a prediction that matches"

`:706–732`. Driver-is-the-code / σ_z→0 gate / confronted-with-delivered-data is a reusable
epistemic template, and every one of its numbers reproduces (I re-ran the law against all
eight `leg_a` rows and the production-confrontation median/positive-fraction). The
deliberate contrast with the `volume_trunc` interlude two sections later — *"That is why one
of these two results is a finding and the other is an exhibit"* (`:1144–1150`) — is the
chapter earning its structure.

### PR-3 — ch08's C4 refutation is the best physics writing in the book

`:667–680` and Q8.2/Q8.3. Leading with the spectacular wrong answer ("64.7% of dark events
have an identically-zero 2D catalogue leg"), then *"That sentence is wrong, and the way it
was refuted is the best thing in this chapter"*, then the exact ln C cancellation and the
1.5%/98.5% budget — that is how to teach that a striking number is not a mechanism. The
budget closes to the digit in my recomputation (+15.83 = 0 + 19.10 − 3.27; f̄ 0.0354 →
0.0061). Cell B strengthens this section rather than threatening it (MJ-4).

### PR-4 — ch11's fair-framing discipline on C5

Both halves of the amended claim appear together everywhere I checked (ch07 `:757–773`,
ch11 `:1188–1208`, Q7.4), and Q11.2 makes the *reason* both are true into the lesson
(coherent same-signed displacements add linearly, class σ grows as √N). The book obeys a
binding amendment it could easily have quoted half of. Ch 11's F-ch11-1/F-ch11-2
handling — quote the adjudicated 1500–2400×, show the recomputed 142–2458× beside it,
and quote the 0.025 Poisson figure that reproduces to 1e-6 instead of the σ_h gloss that
does not — is exactly the right call. I re-ran the Poisson reweight: 0.024904 vs 0.000006 /
0.000015 idealized. Exact.

### PR-5 — ch09's four-quadrant generator/estimator bench refuses to invent data

`:1298–1312`: two of the four cells return *"No such run exists"* and the widget declines to
draw a binomial z for a configuration nobody realized. An interactive that can be dragged
into a state it then refuses to score is a much stronger honesty device than one that
greys the control out.

### PR-6 — ch10 §4 as the model of an exoneration

`<h2>4. C11 — an exoneration done right</h2>` and the summary at `:828–834`: plausible
mechanism, right operating point, third-party instrument, monotone across two decades, a
control reading zero, a numeric margin, *and* a stated scope limit. Contrasting it with the
museum ("full of suspects eliminated by assertion and later found innocent for reasons
nobody had checked") is the correct use of the annex. Cell B does not touch this section —
it is the only one of ch10's five that needs no edit.

---

## Suggested order of work

1. **BL-1, BL-2, BL-4** together (one editing pass over ch11 §5 + the closing question
   block) — this is the cliffhanger's resolution and it must land as one coherent arc:
   confound → registered prediction → control → **two hits and one one-step miss**.
2. **BL-6** (`gen_ch11.py` → `ch11_board.json`) — the board is the chapter's data spine and
   BL-1/BL-2 will read as inconsistent until the board agrees.
3. **BL-3, BL-7, MJ-5, MJ-8** — the four other pages that assert "in flight".
4. **BL-5, MJ-6** — the two "quietly prefers one side of a flagged pair" violations; these
   are independent of cell B and are the ones an examiner will find fastest.
5. **MJ-1 … MJ-4, MJ-7**, then the MINORs.

Nothing in chapters 7–11 needs a *conclusion* changed. Every verdict, every exoneration,
every refutation I checked still stands. What changed is that one question the book left
open has an answer, and the book's own rules say it has to print it.
