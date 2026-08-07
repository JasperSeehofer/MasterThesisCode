# CLAIM — Who owns M-2's matched 2D overlap residual? (stage-0 intake)

**Status: CLAIM, NOT ESTABLISHED. Written to be attacked.**
**Date/time of writing:** 2026-08-07, 19:52 CEST.
**Cycle:** `docs/RESEARCH_CYCLE.md` stage 0 → stage 1 (free reads D-1/D-2). Author approval of
record: the ch13 Part A proposal (`book/site/ch13-unowned-residual.html`), approved 2026-08-07
("lets start the research cycle, its approved"). This intake **implements** that proposal; it does
not reinvent it. The one deliberate extension over ch13's four hypotheses is **H-e
(chance/multiplicity)**, added per the task mandate so the null is a first-class hypothesis, not a
footnote.
**Blindness statement (honest, verbatim commitment):** this file is being written **in parallel
with** the stage-1 reads D-1/D-2. At the time of writing, no number from D-1 or D-2 exists in this
session; every "expected signature" in §5 is derived from the *structure* of the hypothesis and
from already-committed 2026-08-05/07 artifacts, not from peeking at read outputs. If a D-1/D-2
number contradicts a signature below, the signature stands as written and the discrepancy is
reported — no silent edits above the verdict line.
**Governing value ruling (binding, author, 2026-08-05, `BIAS_HISTORY_LEDGER.md` §5):**

> "our overarching goal is a scientifically sound project with novel insights and not to get rid
> of the bias by any means — scientific correctness and new insights are valued higher."

Corollary inherited from the cross-term thread (author, 2026-08-05): *measure, never refute by
convenience.* "The residual is confounding" and "the residual is a real mechanism" are both
acceptable endings; declaring an owner without the component-level measurement, or burying an
unowned effect because it is small, are not.

## Provenance legend

| tag | meaning |
|---|---|
| **[LOCAL]** | re-measured in this session from repo artifacts; reproducible offline now |
| **[AGENT]** | measured by a subagent, not independently reproduced here — **none used in this file** |
| **[DOC]** | read from a committed artifact (readout JSON, ledger, register, prereg, code) |
| **[LIT]** | quoted verbatim from already-cited literature (ar5iv rendering of arXiv:2212.08694 re-fetched this session for quote-verification; Gray et al. 2020 via its repo-registered quotes), with section numbers |
| **[INFER]** | inference from [DOC]/[LOCAL]/[LIT]; no new measurement |

---

## 1. The claim, precisely

**C1 — The residual exists: a matched, cluster-robust +0.021–0.022 nat/event low-h preference
of overlap-involved events in the 2D channel, at both venues. [DOC, independently adjudicated
CONFIRMED]**

Statistic: per-event chord `ln L(h=0.60) − ln L(h=0.73)` on `combined_with_bh` (2D) /
`combined_no_bh` (1D) from `results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/
event_likelihoods.csv` (65,108 data rows = 1588 events × 41 h each; events 1203, 1356 dropped at
evaluate time). **Positive chord = the event prefers low h.** Strata: 385 overlap-involved events
(C-4 census: 1620 sky pairs → 279 sky+2σ-d_L pairs → 385 touched events of 1590 CRB rows;
`recon_c4_census.py` recipe, r = 2√λ_max(JΣJᵀ) chord + 2σ d_L-window intersection) vs 1203
controls. Matching: 1-NN with replacement, control→overlap, on standardized (log₁₀ ball-radius
chord, SNR); balance SMD 1.227 → 0.0027 (log₁₀ radius), −0.381 → 0.0152 (SNR); 234 unique
controls serve 385 pairs.

Measured (matched mean paired diff, 2D):
- **iiib: +0.02225 nats/event**, sign-flip p = 1.5e-04, cluster-robust p = 0.0050
- **joint_r1: +0.02070 nats/event**, sign-flip p = 1.0e-04, cluster-robust p = 0.0042

(cluster-robust = secondary sign-flip flipping all pairs sharing a control together, 234
clusters, addressing control re-use.) Independent adjudication (`m2_adjudicate.py`, own
implementations, seed 777777, 40k perms vs original 20k/seed 20260805): **CONFIRMED** — every
quoted digit reproduced; discrepancies Monte-Carlo-level only (e.g. cluster-robust 0.00465/0.0029
vs 0.0050/0.0042). Scale context: +0.021 nats/event × 385 events ≈ **8 nats aggregate if
coherent** — small vs the hundreds-of-nats class sums of this thread, but one-directional and
present at both venues, including the unscattered-catalogue venue.

Sources: `../crossterm_instrument/m2_results.json` (incl. `robustness_cluster_signflip`),
`../crossterm_instrument/m2_adjudication.json`, `../crossterm_instrument/
session_reads_summary_20260805.json` (m2 + m2_verify blocks), `m2_overlap_stratified.py`,
`m2_adjudicate.py`.

*Refute by (free):* re-run `m2_overlap_stratified.py` from repo root (`cd
/home/jasper/Repositories/MasterThesisCode && uv run python …`) — pure CSV read; or the
already-run independent adjudication. Both are zero-compute.

**C2 — The 1D channel does not carry it. [DOC]**
iiib 1D matched p = 0.098 → **NULL**. joint_r1 1D matched p = 0.0414 (primary NON-NULL by the
pre-stated α = 0.0455 criterion) but **cluster-robust p = 0.137 fails** → marginal-and-fragile,
effectively unresolved toward NULL. The 1D conclusion *flips* from unmatched to matched — the
unmatched 1D "effect" was selection.
*Refute by (free):* same re-run as C1.

**C3 — The unmatched read (+0.044…+0.050 nats/event, perm p at floor everywhere) is mostly
selection — the [A2] trap, discharged. [DOC]**
Overlap events are by construction the large-ball/dense-sky events (SMD 1.227 in log₁₀ radius
before matching). Matching collapses the imbalance and takes ~55–60% of the 2D "effect" and the
whole 1D effect with it. The finding of record is the matched +0.021, nothing larger.
*Refute by:* n/a — this claim protects against inflation; attacking it means showing the matched
number is the artifact, which is C1's refutation route plus H-c below.

**C4 — The leading candidate owner is EXCLUDED: the Eq. (31) pairwise cross-term. [DOC]**
NEGLECT-WITH-NUMBER in all four venue × channel cells (ledger §1 row 96; author ruling
2026-08-07): composed T = 5.648753e-05 (joint_r1/2d) / 5.379542e-06 (iiib/2d) nats vs locked
X = 2.78 — minimum margin **4.92e+04×**; adjudicated bit-identical. The composed 2D chord
moreover has the **opposite sign** (high-h) to the M-2 residual. Closure is **conditional** on
the six-trigger register `../crossterm_instrument/NEGLECT_TRIGGER_REGISTER.md`; per its §5, if
this hunt re-implicates likelihood-factorization structure or the mixture composition, that is
trigger-(b)/(f) territory and re-opens that register by construction — it does not silently
re-open here.
*Refute by:* only a named register trigger firing. Not re-litigated in this thread.

**C5 — The live clue: the low-h coherence physically exists in the raw catalogue leg and the
mixture composition annihilates it. [DOC]**
Raw catalogue-leg 2D class-summed chords are **positive (low-h): +2.507 nats (joint_r1) /
+0.0116 nats (iiib)**. The posterior never consumes the raw leg: per-event catalogue share
r_e = w_G·L_cat/combined ≈ 3e-04–1.6e-03, per-pair composition factor F = r_i·r_j median
~1.5e-07–2.5e-06 (≈5 orders of suppression), and w_G's h-fall (0.0957 → 0.0556 across the grid,
×1.72) **reverses the class-level chord sign** (register §1, facts 1–3). ⇒ **The owner must act
outside the annihilated catalogue path, or through the composition weights themselves.** A
mechanism living purely inside the raw catalogue-leg coupling cannot reach the posterior at the
measured size.

**C6 — Structural constraint (new, this intake): any event-common h-dependent factor cancels
EXACTLY out of the matched-difference statistic. [INFER]**
The statistic is a mean over pairs of (chord_overlap − chord_matched_control). A multiplicative
likelihood factor common to all events at each h adds the same constant to every event's chord
and cancels identically in every paired difference. Consequences, each anchored to a measured
fact: (i) the **f_k pool-substitution level/slope** cannot own the residual — M-3 measured its
per-event L_cat chord as **event-independent to ~1e-12** (0.0177/0.0176 nats uniform;
`session_reads_summary_20260805.json` m3); (ii) an **N-2-class global completion correction**
(S̄_φ event-common factor) cannot own it either; (iii) **w_G itself is event-independent at
fixed h** (asserted nunique==1 per h per venue) — so w_G can matter **only through its
interaction with per-event leg ratios** (that interaction is exactly H-d, and it does NOT
cancel). This constraint is what makes the owner-hunt tractable: the owner must be a
**per-event, stratum-correlated** object.
*Refute by (free):* exhibit an event-common factor that does not cancel in the paired statistic
— i.e., check the algebra; or show the matched pairs are evaluated at different h grids
(they are not; same CSV, same 41-h grid).

**The claim in one sentence:** a matched, cluster-robust +0.021–0.022 nat/event low-h preference
of overlap-involved events exists in the 2D channel at both venues, its leading candidate
mechanism is measured-excluded, and **its mechanism is unidentified**.

---

## 2. Two-layer exoneration check (stage-0 rule 1) — RECORDED

Layers grepped before anything opened: **(1)** local Exonerated lists —
`../CLAIM_2D_BIAS_20260730.md:721-745` ("Exonerated — do NOT re-open without new evidence") and
the Hitchhiker draft's §4/§7; **(2)** project-wide —
`../gate_b_20260730/BIAS_HISTORY_LEDGER.md` §1 chronology (incl. August rows 90–96), §2
DO-NOT-RE-TRY (⚠ items), §3 history-vs-current-claims, §4 open threads, §5 author rulings. The
binding set is the union. Exonerations are **venue-scoped**: M-2's venues are the
`run_20260804_postfix` iiib + joint_r1 family; the seed600-scoped exonerations (#70, #72
scoping rule, §2 closing note) do not automatically transfer and are not leaned on.

Per adjacent thread — does it bear on this claim, and is re-litigation occurring?

| adjacent thread | what is settled | bears on this claim? | re-litigation? |
|---|---|---|---|
| **gate (vii) catalogue tilt** (ledger row 90 Aug; A2 exoneration void; `run_20260804_postfix/gate_vii/`) | dark-class catalogue-leg 2D/1D tilt GREW post-fix (−604.8 nats, N=534); it is a **DOWN-pull muted by w̃_G**, composition-dominated (81% from 316 scatter-resurrected events) | YES — it independently established that the catalogue leg's channel structure is composition-muted, consistent with C5; and its [A2] lesson is *why* M-2 was matched at all | **NO.** M-2's object is an overlap-vs-control **stratum difference of the combined likelihood**, not a class-summed channel difference. Nothing from gate (vii) is re-measured; its paired-read discipline is inherited, not re-derived |
| **N-2 selection numerator** (row 93; §4 item 14; adoption OPEN = queue item 3, author rulings R-0..R-5) | S̄_φ-inside-1D is a real, positive, **bounded** correction (+24.6/+22.7 nats/h chord, in band) that does NOT un-rail 1D; adoption undecided | YES, twice: (i) structurally — as an (approximately) event-common 1D completion factor it **cannot own M-2's residual** (C6); (ii) procedurally — adoption fires cross-term register trigger (b) and changes the composition weights H-d watches | **NO.** The adoption ruling stays the author's open queue item 3; this thread neither measures S̄_φ again nor presumes the ruling. If adoption lands mid-thread, D-1's weight-piece read is re-run on the post-adoption emits (minutes) |
| **D1 / f_k pool coupling** (rows 94, 96-adjacent M-3; §4 item 15; queue item 5, its own future intake) | D1 bounded-null via the tilt route (m_S=0.032, m_R=0.011 ≪ 0.25); D1→g_frac route dead at machine precision; f_k pool substitution moves L_cat by **one global h-dependent scalar** (per-event chord 0.0177/0.0176, event-independent to ~1e-12); the ⚠ "p_det inside/outside" ledger items sit adjacent to that thread | YES — the M-3 event-independence measurement is exactly what C6 needs to *exclude* the f_k level/slope as this residual's owner; the f_k thread's honesty caveat (event-common factors still tilt a joint posterior via N_events) is a different estimand from M-2's paired difference | **NO.** f_k stays its own stage-0 intake (queue item 5) with its own exoneration check ("p_det inside/outside" ⚠ items). This thread only *uses* M-3's committed number. No pool substitution is re-run |
| **g_frac / C9** (rows 91–92; author ruling R-A 2026-08-05: g_frac(h) h-slope = **correct physics**; C9 = w_G 2.3–2.5× miscalibration vs generator, live, gated on cell B, `CLAIM_2D_BIAS_20260730.md` C9) | the 2D carrier is legitimate spectral-siren physics (closed-loop MIXED/not-REFUTE supports it); w_G calibration question is separately live | YES — H-a/H-b run **through g_frac** (1587 distinct per-event values; per-event ⇒ can carry a stratum difference). Ruled-correct physics **does not preclude** g_frac *correlating with overlap-stratum membership* — ch13 Part A states this scoping explicitly and this intake inherits it | **NO on both prongs.** The derivation question is closed by R-A and is not re-opened: no re-derivation, no defect claim. C9 (w_G calibration) is gated on cell B and untouched: H-d measures w_G's *stratum-interacting composition role*, not its calibration level |
| **Eq. (31) cross-term** (row 96; NEGLECT_TRIGGER_REGISTER) | excluded owner (C4) | YES — as the exclusion and as the source of C5's clue | **NO** — conditional closure respected; only a named trigger re-opens it |
| **§2 ⚠ adjacency sweep** (items 1, 3, 10, 13) | `mass_trunc`/truncated-lognormal as 2D *driver* exonerated twice (#72, #89); `w_G = β_G/D` bookkeeping *as the fix* refuted (#61); `L_comp`/`B_num` *as a defective integral* exonerated (#80, #87 — B_num is the residual carrier but not a shown defect); "information starvation" OVERTURNED (#41/#52) | YES — these scope what H-a/H-d may claim | **NO, by construction of the hypotheses:** H-a claims a **stratum-differential h-slope of the completion leg**, not a defective integral; H-d claims a **stratum-dependent composition interaction**, not a bookkeeping fix; no mass-kernel re-derivation anywhere; starvation is not resurrected as an explanation |

**Verdict of the check: no re-litigation is occurring.** Every hypothesis below either measures a
new estimand (stratum-differential, paired) on an already-exonerated-for-a-different-estimand
object, or uses a committed measurement as an exclusion. The two live author-gated questions this
thread touches (N-2 adoption, issue #53 ball width) are left to their owners; both are named
re-read triggers, priced at minutes.

---

## 3. R0 sweep (amendment A5, mandatory) — already-cited literature re-read for warnings

**Scope discipline:** ring **R0 only** — papers already cited in this repo. arXiv:2212.08694 was
re-fetched (ar5iv rendering, same source as the 2026-08-05 intake) solely to quote-verify
passages **beyond §2.3**; Gray et al. 2020 is read *as cited in the repo* (its registered quotes
— a full Stage L section for it remains open per `docs/LITERATURE_WARNINGS.md` "Other sources").
No fresh external search, no R1–R4 rings. Quote-verification before mapping, per A5: **quote
first, map second.**

### [LIT-1] arXiv:2212.08694 §4.2, Inconsistency 5 — the strongest signature match in the sweep

Verbatim (§4.2, "Inconsistency 5, simulating dark siren events along the same line of sight";
re-verified this session; also registered in the 2026-08-05 intake):

> "we find that a low-[H₀] bias will arise if there are low-redshift overdensities along the
> line of sight and the full likelihood analysis is not used. When simulating multiple events
> along the same line of sight, if there is an overdensity of galaxies in that direction, the
> [H₀] posterior is expected to display a peak at an [H₀] lower than the input value, that
> corresponds to the redshift of the overdensity … the presence of the same galaxies across
> multiple GW events needs to be taken into account with the full likelihood analysis so as not
> to incur in a bias."

**Mapping (after the quote):** M-2's residual is a **low-h** preference of exactly the
shared-sky, d_L-compatible (same-line-of-sight-like) stratum. This is a **direction and
population match** to Inconsistency 5. Two honest caveats: (i) the paper's stated remedy is the
full Eq. (31) likelihood — whose leading pairwise term we **measured** as composition-annihilated
with the *opposite* composed sign (C4/C5); so if an Inconsistency-5-type effect is operating
here, it must reach the posterior by a **non-pairwise route** — e.g. the density structure inside
each event's own ball shaping its per-event likelihood — which is precisely the C5 constraint
and the territory of H-b/H-c; (ii) `docs/LITERATURE_WARNINGS.md` row H-e registered this warning
**UNCHECKED** and its note assessed direction only against the *high*-H₀ rail ("candidate
partial-cancellation term"). **R0 finding: for THIS claim the direction matches; the register
row's reasoning note is stale with respect to thread 16 and should gain a dated addendum when
this thread reaches stage 6.** (No register edit is made now — rails.)

### [LIT-2] arXiv:2212.08694 §4.2, Inconsistency 4 — a concrete mechanism candidate for H-c

Verbatim (§4.2, "Inconsistency 4, GW likelihood mismodeling"; quote-verified this session —
this passage was NOT in the 2026-08-05 intake and is not in the warnings register):

> "Another possible source of error is treating the GW likelihood as if the standard deviation
> was not dependent on the true value of [d_L], although the simulations are made assuming this
> dependence … By dropping the overall normalization factor in the GW likelihood, one is in
> practice ignoring a part of the likelihood that depends on the true luminosity distance. This
> causes a biased dependence of the [H₀] posterior on the luminosity distance uncertainty …
> We find that in this case the inconsistency has the effect of biasing [H₀] towards lower
> values for increasing values of [the d_L uncertainty]."

**Mapping:** a bias toward **low H₀ that grows with the event's d_L uncertainty** is exactly the
shape of a **residual selection confound**: overlap events are the large-ball (≈ large
localisation/d_L-uncertainty) stratum, and 1-NN matching on log₁₀ radius + SNR balances the
covariate *means* but not necessarily a *nonlinear* response in the covariate. This gives H-c a
concrete, literature-documented mechanism with the right sign. Our adjacent measurement of this
class is ledger #67 (σ(d_L^obs)-vs-σ(d_L^true) noise model, const-σ floor a real asymptotic
bias ~+0.0005 in the harness) — measured on the harness, **not** on the production venues, and
never as a stratum read. **R0 finding: new candidate register row; feeds D-2's covariate list
(d_L and its error explicitly included).**

### [LIT-3] arXiv:2212.08694 §4.2, Inconsistencies 1–3 — adjacent, mostly heeded

Verbatim anchors (quote-verified this session): Inconsistency 1 ("double counting" —
generator/estimator prior compounding; "this type of mismatch usually results in a bias towards
lower values of [H₀]") — registered as warnings row H-d, **UNCHECKED**; the direction (low)
is noted for completeness since our residual is a low-h preference, but the mechanism is global
(event-common prior mismatch ⇒ largely cancels in the paired statistic per C6, except insofar as
the compounding is density-dependent — folded into H-b's reading). Inconsistency 2 ("TH21
assumes that the detection probability is a Heaviside step function of the true luminosity
distance … correct only in the limit that there are no errors on the luminosity distance
estimation"; "can bias the estimation of the Hubble constant to lower values") — our p_det is a
survival estimator, not Heaviside, and the noise-model class was measured (#67); as a
*stratum-differential* object it collapses into H-c/D-2. Inconsistency 3 ("GW events are drawn
from galaxies with [z < z_draw], but this aspect is not accounted for in the analysis … might
introduce a high [H₀] bias") — this is the D1/generator-filter class, measured to bounded-null
via the (ii-d) count audit and the D1 three-arm read (rows 83-era, 94). **Heeded; no new row
needed beyond H-d's existing UNCHECKED.**

### [LIT-4] arXiv:2212.08694 §4.1 + Appendix A — clustering per se is NOT a bias source (tempering for H-b)

Verbatim (§4.1, "The absence of clustering"; quote-verified this session):

> "The absence of clustering significantly weakens the result, rendering it nearly uninformative
> in the limit of large numbers of galaxies, but does not incur in significant biases." …
> "contrary to Trott & Huterer (2021), we conclude that the absence of clustering is not
> expected to introduce a bias on the [H₀] estimation. In reality, the Universe is not uniform
> in comoving volume on smaller scales … but the effect of clustering is to enhance the [H₀]
> constraint, rather than being essential to it."

**Mapping:** clustering by itself, treated consistently, is constraint-enhancing, not biasing —
so H-b cannot rest on "overlap events sit in denser sky" alone; it must name a **coupling**
(g_frac or the composition weights correlating with density) through which the estimator treats
dense-sky events differently from the generator. That is how H-b is framed below. **Tempering,
not exclusion.**

### [LIT-5] arXiv:2212.08694 §2.1 + Discussion — population/rate terms and the completeness-rate correlation

Verbatim (quote-verified this session): "we will neglect the rate term in Eq. 5 in order to
align more closely with the approach taken in TH21. **As long as this rate assumption is treated
self-consistently when generating the mock data and analyzing it, this will not introduce any
bias to the results.**" And from the Discussion: "On the galaxy catalog side, there are several
challenges related to the **completeness correction** of the galaxy catalog and the presence of
a **possible correlation between galaxy intrinsic luminosity and merger rates** (Gray et al.,
2020, 2022)."

**Mapping:** the self-consistency clause is our measured territory (w_pop tilt ≤ +0.0004,
ledger #64 — do not re-open). The completeness-correction/rate-correlation warning maps onto the
composition weights (w_G, α_G_φ, r_Malm, D̃_φ) and the rate weight w_pop inside the balls —
the objects H-d and H-b watch. No quantitative claim imported; it is a warrant that the
weight-structure route is a field-recognized failure surface.

### [LIT-6] Gray et al. 2020 (arXiv:1908.06050), as cited in this repo — the completion framework's photo-z blind spot

Repo-registered quote (`docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:171-174`; register "Other
sources" row): Gray et al. 2020's photo-z handling is an **unexercised equation** — "validated
at σ_z = 0 only ('ignore these crucial redshift uncertainties altogether')"; under flat p_det
the same-kernel denominator degenerates to a constant N. Also ledger row 26: our L_cat was
brought onto Gray Eqs. A.9/A.10 after a misreading fix.

**Mapping:** the completion leg (L_comp/B_num) and the partition weights are Gray-derived
structure operating at σ_z/z ~ O(1), a regime the source never validated. This does not name a
stratum mechanism by itself, but it is the standing warrant that the **completion-leg path**
(H-a) — the leg the composition does *not* annihilate — is exactly where an undocumented
regime-dependence would live. Register status stays UNCHECKED-as-section; no new claim imported.

**Absences (recorded so they are not re-hunted):** the 2026-08-05 sweep's negative results stand
— arXiv:2212.08694 contains **no** statement on shared-population-hyperparameter correlations,
catalogue-realization (shared-noise) correlations, or shared injection-pool/selection-MC
correlations (`CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT` §1 "Statements that do NOT
exist"). If the owner turns out to be a pool- or realization-coupling, **the field has no
documented warning for it** — that would itself be a reportable novelty per the Stage L exit
rules.

**Is a fuller Stage L (R1–R4) warranted?** **Conditionally yes — deferred, not skipped.** If
D-1/D-2 leave a density/clustering-coupled mechanism standing (H-b, or H-c refuted as *sole*
owner with a density-shaped remainder), a timeboxed R1 pass over the forward citations of
arXiv:2212.08694 filtered on *line-of-sight / overdensity / clustering bias / cross-correlation*
is the indicated next ring — Inconsistency 5 is plainly the field's nearest documented failure
mode and its neighborhood has not been searched. If the residual dissolves into confounding
(H-c) or chance (H-e), no fuller Stage L is needed. This conditional is recorded here so the
stage-5 decision inherits it explicitly.

---

## 4. Hypotheses H-a … H-e

Framed per ch13 Part A ("deliberately not ranked; none pre-judged"), each with the observation
motivating it and a `Refute by:` naming a **FREE** read. The D-1 component decomposition is
designed to discriminate H-a/H-b/H-d in one read; D-2 targets H-c; H-e has its own free
machinery. All reads operate on the identical 385 matched pairs and identical sign-flip +
cluster-robust machinery as M-2 unless stated.

**H-a — Completion-leg difference. [INFER on DOC]**
The completion leg is the path the composition does **not** annihilate (C5). If overlap-stratum
events' `L_comp` — or, 2D-specifically, the completion mass factor `g_frac` (ruled correct
physics, which does not preclude stratum correlation) — carries a systematically different
h-slope, the residual lives there. The 2D-only/1D-null pattern (C2) is naturally accommodated:
`g_frac` enters only the with-BH channel.
*Refute by (FREE — D-1):* matched stratified read on the `L_comp` and `g_frac` chord columns of
`event_likelihoods.csv`. If neither completion-leg column reproduces a positive matched residual
surviving the cluster-robust test, H-a is refuted.

**H-b — g_frac–clustering correlation. [INFER on DOC]**
`g_frac` has 1587 distinct per-event values; overlap events sample denser sky. A correlation
between the mass-factor slope and local clustering would make the overlap stratum inherit a
different effective g(h) **with no defect anywhere** — under the governing ruling this would be
a *novel-insight* ending, not a bias to remove. Tempered by [LIT-4]: the claim must be a
coupling, not "density biases per se".
*Refute by (FREE — D-1 + D-2 jointly):* (i) matched read on the `g_frac` chord column shows no
stratum difference → refuted; or (ii) the g_frac-column residual exists but does **not**
attenuate when density covariates (ball galaxy count from the frozeng ball emits
`results/run_20260804_frozeng/<venue>/posteriors_with_bh_mass/h_0_73.json`, local catalogue
density, sky position) enter the matching → the *clustering-correlation* reading is refuted
(leaving a bare H-a).

**H-c — Residual selection confounding. [INFER on DOC + LIT-2]**
The matching balances two covariates (log₁₀ ball-radius chord, SNR). If overlap-stratum
membership is driven by further covariates that independently shape the likelihood chord — sky
position, catalogue density in the ball, d_L and its error ([LIT-2] gives the documented
low-H₀-with-growing-σ_dL mechanism), or a nonlinear response to the already-matched radius that
1-NN mean-balancing does not remove — then +0.021 is a matching artifact, not physics. Finding
this "costs minutes and is a success, not a failure" (ch13).
*Refute by (FREE — D-2):* extend the matching covariate set (ball galaxy count / local
catalogue density / qS, phiS / d_L and δd_L from `prepared_cramer_rao_bounds.csv`) and re-run
the identical machinery. If the combined-column 2D residual does **not** attenuate materially
under every richer matching (signature bands in §5), confounding-as-sole-owner is refuted.

**H-d — Composition-weight h-dependence. [INFER on DOC]**
The annihilation (C5) is built on w_G(h) and the per-event shares r_e. w_G is event-independent
at fixed h (C6), so it cannot act alone — but a **stratum-dependent interaction** between
w_G's h-fall (0.0957→0.0556) and the per-event mixture (r_e spans ~3e-04–1.6e-03, and overlap
events plausibly have systematically different L_cat/L_comp ratios) could move overlap events
differently — the "through the weights" route the trigger register's §5 explicitly flags. The
h-structure of the weight objects (`w_G`, `alpha_G_phi`, `r_Malm`, `D_tilde_phi`, `w_tilde_G`)
differing by stratum is the general form.
*Refute by (FREE — D-1):* decompose each event's combined chord into the w_G-composed pieces
(`w_G·L_cat` and `(1−w_G)·L_comp` at each h, all columns in/derivable from
`event_likelihoods.csv`). If the matched residual is visible in **neither** composed piece
beyond what the corresponding bare leg already carries — equivalently, if the per-event mixture
share s_e(h) = w_G·L_cat/combined shows no stratum-differential h-slope — H-d is refuted. (If
H-d survives to a mechanism claim, note it is register-trigger-(b)/(f) adjacent by
construction.)

**H-e — Chance / multiplicity. [INFER on DOC]**
The M-2 criterion was **pre-registered NULL-expected** (Hitchhiker draft §6, M-2 spec: "the two
strata should be statistically indistinguishable"), with the covariates and α = 0.0455 stated
in-script before reading results — so this was not a forking-paths hunt. The honest
look-elsewhere, quantified:
- *Family size.* The session's pre-specified read family: 4 venue × channel cells in M-2 (plus
  M-1/M-3/M-4, which carry no stratum p-values). Bonferroni ×4 on the primary matched p: 2D
  1.5e-04→6e-04 (iiib), 1.0e-04→4e-04 (joint_r1) — both survive. On the cluster-robust
  secondary: 0.0050→0.020, 0.0042→0.017 — survive. Widening the family to the draft's six
  hypotheses (×6) still leaves the 2D cells < 0.05. The joint_r1 **1D** cell (p = 0.0414)
  survives **no** correction and is already treated as null-ish (C2).
- *What multiplicity cannot explain away, and what it still could.* The two venues are **not**
  independent replications of the matching (identical control assignment from the shared CRB
  covariates — caveat carried verbatim from m2_results.json); they replicate the likelihoods
  only. And the primary/secondary tests assume the 385 paired differences are exchangeable
  units: **control re-use** was corrected (234 clusters), but **overlap-event correlation was
  not** — the 385 overlap events are linked by the 1620-sky-pair graph into connected sky
  clusters whose chords are plausibly correlated (shared sky, shared catalogue structure). If
  the effective number of independent pairs is far below 385, the sign-flip floor p-values
  overstate the evidence. That is the one live chance route, and it is testable for free.
- *The honest posture:* the cluster-robust-by-control test was a secondary added when the
  re-use issue was noticed (documented in m2_results.json), i.e. one robustness layer was
  post-hoc; the by-overlap-component layer has never been run at all.
*Refute by (FREE):* (i) sign-flip permutation flipping **connected components of the C-4
overlap graph** as units (components computable from `prepared_cramer_rao_bounds.csv` alone) —
if the 2D residual survives component-level flips at both venues, the intra-stratum-correlation
chance route is refuted; (ii) jackknife-over-components and re-matching under different RNG
seeds for stability; (iii) structural evidence — D-1 localizing the residual coherently to one
component at both venues is itself strong evidence against chance (H-e predicts no coherent
localization; see §5).

---

## 5. Stage-1 plan (exactly as approved in ch13 Part A) + pre-stated expected signatures

**Timestamp and blindness, restated:** written 2026-08-07, 19:52 CEST, in parallel with the
reads; no D-1/D-2 output has been seen. Signatures below are derived from hypothesis structure
and committed artifacts only. Bands are stated **before** any read; the reading session appends
verdicts below the line at the end of this section and edits nothing above it.

### D-1 — Component decomposition of the matched residual (FREE)

Re-run the **identical** matched stratified read — same 385 pairs, same 1-NN matching (same
covariates and seed), same sign-flip + cluster-robust machinery — separately on each likelihood
component emitted per event and per h in
`results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/event_likelihoods.csv`:
`L_cat_no_bh`, `L_cat_with_bh`, `L_comp`, `g_frac`, `B_num`, `B_num_wbh`, and the w_G-composed
pieces `w_G·L_cat` and `(1−w_G)·L_comp` (plus the weight objects `w_tilde_G`, `alpha_G_phi`,
`r_Malm`, `D_tilde_phi` as h-chords where informative). The +0.021 was measured on the combined
column; this localizes which term carries it.

*Pre-declared handling constraint (stated blind):* `L_cat_with_bh` is **zero** for 981/1588
(joint_r1) and 1294/1588 (iiib) events, and `L_cat_no_bh` for 493/606 (M-4 V3-validated counts)
— ln-chords are undefined there. Component reads run on the defined-per-column subsets with the
per-column pair count reported honestly; a component whose defined subset loses most of the 385
pairs cannot be read at the same power, and that attrition is reported, not hidden. No
imputation, no zero-replacement.

*Cost:* CPU-minutes; inputs all on disk. *Exit:* the residual localized to component(s), or
spread thin (see H-e signature).

### D-2 — Extended-covariate confounding check (FREE)

Add covariates to the matching and re-run the identical machinery on the **combined** 2D chord:
ball galaxy count (from `results/run_20260804_frozeng/<venue>/posteriors_with_bh_mass/
h_0_73.json` ball sets — M-4 V2 validated them h-independent), a local catalogue density proxy,
sky position (qS, phiS), and d_L + δd_L (from `prepared_cramer_rao_bounds.csv`). Report balance
(SMDs) before/after per covariate set, and the matched residual + both p-values per enrichment.

*Cost:* minutes. *Exit:* residual robust to enrichment, or attenuated into confounding.

### Expected signatures per hypothesis — declared BEFORE the reads

| hyp | D-1 signature if TRUE | D-2 signature if TRUE | additional discriminator |
|---|---|---|---|
| **H-a** (completion leg) | Positive matched residual **reproduces on `L_comp` and/or `g_frac`** chords (2D-relevant pieces), surviving cluster-robust; `L_cat` columns and the `w_G·L_cat` piece do NOT carry it beyond noise | Residual **persists** (≤ ~30% attenuation) under density/sky enrichment | 1D/2D asymmetry: the carrier column should be one that enters only (or much more strongly) the with-BH channel — `g_frac`/`B_num_wbh` — matching C2 |
| **H-b** (g_frac–clustering) | `g_frac`-column matched residual present (as H-a) | AND the g_frac-column (and combined) residual **attenuates materially (≳50% toward 0, or loses cluster-robust significance)** when density covariates enter | Free correlation read: per-event g_frac h-chord vs ball galaxy count, positive association concentrated in the overlap stratum |
| **H-c** (residual confounding) | **No single component carries it**: several components show small same-sign shifts none of which survives cluster-robust alone | Combined-column residual **shrinks toward zero** (≳50% attenuation and/or cluster-robust p ≥ 0.0455) under the richer matching — the ch13 wording: "attenuates toward zero under the richer matching is evidence for confounding, not mechanism" | Dose-response: residual size should track the pre-match imbalance of the *added* covariate (e.g. ball count SMD), and re-matching on radius-nonlinear terms alone should already bite |
| **H-d** (composition weights) | Residual visible in the **composed pieces** (`w_G·L_cat`, `(1−w_G)·L_comp` chords) but in **neither bare leg**; equivalently the mixture-share s_e(h) h-slope differs by stratum (cluster-robust) | Persists under enrichment (it is a likelihood-structure effect, not a covariate artifact) | The effect should scale with the event's leg ratio L_cat/L_comp; stratifying pairs by r_e(0.73) tercile should show the residual concentrated at high r_e |
| **H-e** (chance/multiplicity) | **No coherent localization**: no component survives cluster-robust; magnitudes scattered in sign across components and venues | Unstable: attenuation/inflation moves erratically with covariate-set choice and matching seed, without dose-response | **The decisive free read:** component-level sign-flip over C-4 overlap-graph connected components. H-e predicts p ≥ 0.0455 at one or both venues; every mechanism hypothesis predicts survival at both. Also: jackknife-over-components shows the mean driven by ≤ a few components under H-e |

*Mixed/undetermined branch (first-class, per stage-2 discipline inherited early):* if the reads
return a split picture — e.g. partial attenuation 30–50% under D-2, or a component carrier that
fails cluster-robust at one venue — the split is **read directly and reported as MIXED**; no
branch is forced. A MIXED outcome routes to stage 2 (pre-registration of whatever surviving
candidate needs a locked band) — and two consecutive MIXED/UNDETERMINED verdicts on this thread
auto-trigger a **full Stage L** per trigger L-b.

*If the residual dissolves entirely (H-c or H-e confirmed):* the cycle stops at stage 1 and the
chronicle records a confounding/chance verdict — per ch13, "that is a success, not a failure."

---

## 6. What is explicitly NOT claimed

1. **Not claimed:** that the residual is a defect. Under the governing ruling, a real
   clustering–g_frac correlation (H-b) and a confounding artifact (H-c) are both acceptable
   endings; the claim is only that the mechanism is unidentified.
2. **Not claimed:** any H₀ displacement number attributable to the residual. The ~8 nats
   class-scale is an if-coherent bound, not a measured posterior displacement.
3. **Not claimed:** that the Eq. (31) cross-term neglect is unsafe — its conditional closure
   (C4) stands; this thread does not fire any register trigger by existing.
4. **Not claimed:** that g_frac's h-slope derivation is wrong (R-A closed that), that w_G's
   calibration level is settled (C9 stays live and gated elsewhere), that `B_num`/`L_comp` is a
   defective integral (#80/#87 stand), or that the 1D rail owner is identified (open thread
   §4 item 14, untouched here).
5. **Not claimed:** venue-independence of the matching. The matched control assignment is
   identical across venues (shared CRB covariates); the venues replicate the likelihoods, not
   the matching (caveat carried verbatim from m2_results.json).

## Exonerated — do NOT re-open without new evidence (local list, this thread)

Inherited binding union (see §2): everything in `../CLAIM_2D_BIAS_20260730.md:721-745` plus
`BIAS_HISTORY_LEDGER.md` §2 items 1–17. Specifically load-bearing here: **Eq. (31) pairwise
cross-term as this residual's owner** (row 96, NEGLECT-WITH-NUMBER, min margin 4.92e+04×,
opposite composed sign — re-openable only via the named triggers) · **f_k pool-substitution
level/slope as a per-event carrier** (M-3: event-independent to ~1e-12 ⇒ cancels in the paired
statistic, C6) · **g_frac h-slope as a defect** (author ruling R-A) · **`L_comp`/`B_num` as a
defective integral** (#80, #87) · **`w_G = β_G/D` bookkeeping as a fix** (#61) ·
**mass-kernel family / mass_trunc as the 2D driver** (#72, #89 — venue-scoped, not universal) ·
**information starvation as an explanation** (#41/#52 OVERTURNED).

## Errors made this session — do not inherit them

1. **Near-miss:** an early draft of §3 treated the WebFetch small-model summary of §4.2 as
   quotable. It paraphrased and truncated (Inconsistency 3 cut off mid-item). All [LIT] quotes
   above were re-extracted from the raw ar5iv HTML text instead. Rule inherited from A5 stands:
   quote-verify from the source rendering, never from a summarizer.
2. **Ambiguity to not propagate:** m2_results.json labels the 2D direction
   `overlap_high_revives_H2` — "high" there refers to the **chord statistic** (more positive),
   which means the stratum prefers **low h**. Any downstream text must say "low-h preference"
   and never bare "pulls high", or the sign will eventually be flipped in prose. (ch13 already
   uses the correct "pulls toward low h".)
3. **Scope discipline:** this file initially risked folding the f_k intake (queue item 5) into
   H-a's completion-leg territory; they are separated — f_k is catalogue-leg and its own
   thread. Kept apart in §2.

---

## 7. Intake completeness checklist (stage-0 template)

- [x] **Two-layer exoneration check before opening anything** (rule 1) — §2, both layers
      grepped, union taken, ⚠ items swept, venue-scoping honored. No re-litigation.
- [x] **Status header** "CLAIM, NOT ESTABLISHED. Written to be attacked." (rule 2).
- [x] **Every statement tagged** [LOCAL]/[AGENT]/[DOC]/[LIT]/[INFER] (rule 3) — no [AGENT]
      claims used; the M-2 numbers are [DOC] with an independent CONFIRMED adjudication.
- [x] **Every numbered claim carries the measured statement, method, and a `Refute by:`
      naming the cheapest decisive test** (rule 4) — C1–C6, H-a–H-e; every Refute-by is FREE.
- [x] **`## What is explicitly NOT claimed`, Exonerated list, and `## Errors made this
      session`** present (rule 5).
- [x] **A5 mandatory R0 sweep** — §3, six [LIT] findings, quote-verification before mapping,
      absences recorded, fuller-Stage-L conditional stated.
- [x] **[A1] free re-reads before compute** — the entire stage-1 plan (D-1, D-2, H-e's
      component-flip) is free reads of existing CSVs/JSONs; no cluster, no `--confirm-run`
      instruments, no new likelihood evaluations proposed.
- [x] **[A2] paired/stratified read alongside any aggregate** — constitutive of the design:
      the claim object is itself the matched paired read, and D-1/D-2 inherit the machinery.
- [x] **Blindness recorded** — header + §5; expected signatures pre-stated; append-only
      verdict discipline declared.
- [ ] **Author gate** — stage-0/1 output to be presented with the D-1/D-2 verdicts; STOP
      before any stage-2 pre-registration. No ledger rows appended by this intake, no edits to
      any existing claim/prereg/ledger/register file, no commits (final Synthesis agent owns
      the commit).

---

*Verdicts from the stage-1 reading session append below this line; nothing above it is edited.*
