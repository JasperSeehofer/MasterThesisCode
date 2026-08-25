# Stage-L literature search — [HIER] thread (symptom card only, blind to campaign mechanism claims)

**Date:** 2026-08-25. **Scope:** research-cycle Stage L, rings R0–R4, timeboxed ~40 min of
search effort. Input was the symptom card only (ensemble H₀ from N≈40–200 standard sirens vs a
photometric catalogue, median σ_z/z≈49%, posteriors rail at grid edges, score-at-truth tilt
z-structured 0→−1 across z≈0.4→0.9, N-coherent multi-σ ensemble bias, p_det≈1). No campaign
mechanism document was read for this pass, per instruction.

**Quote-verification discipline applied inconsistently below** — WebFetch on raw arXiv PDFs
returned degraded/partially-hallucinated text in several cases (flagged inline as `UNVERIFIED`).
Any candidate promoted to a proposal or a code decision needs a raw-PDF or arXiv-HTML re-fetch
before its quote is trusted, per this repo's existing quote-verification convention.

---

## Ring R0 — already cited in this repo (`docs/LITERATURE_WARNINGS.md`)

Re-read for photometric-z / hierarchical content already on file. Nothing in R0 already proposes
or executes ensemble self-calibration of a photo-z error model jointly with H₀ — this confirms
the symptom card's target is a genuine gap in our own citation set, not a re-tread.

- **Gray et al. 2020** (arXiv:1908.06050) row `G20-b`: the per-galaxy redshift kernel `p(z_i)` is
  *derived but never exercised* — the paper's own mock data challenges run at σ_z = 0 ("ignore
  these crucial redshift uncertainties altogether"). No self-calibration content.
- **Gray et al. 2023** (arXiv:2308.02281) rows `G23-a`–`d`: Gaussian kernel, footnote notes no
  strict Gaussianity requirement, but no truncation/joint-inference-with-H0 treatment.
- **Gair et al. 2023 "Hitchhiker's Guide"** (arXiv:2212.08694) row `H-a`: **already VIOLATED** in
  our pipeline — perfect-z separability fails; this is the row our symptom card's premise sits on.
  No self-calibration escape route is offered by this paper; its only stated escape clauses are
  perfect z or the large-volume uniform-density argument, both already ruled out for us.
  Row `H-b` is the closest existing content to "jointly account for imperfect z across the
  ensemble" — but it is a **cross-term correction to the same fixed (assumed-correct) kernel**,
  not inference of the kernel's own parameters. Different problem class from the symptom card.
- **Alfradique/Bom/Castro 2025, Borghi 2025, VanWyngarden 2025** (rows `ABC25`, `B25`, `VW25`):
  all study **catalogue incompleteness** bias direction/magnitude, not **photo-z error-model
  misspecification**. The register's own "documented field gap" note (2026-08-21) already flags
  that none of the surveyed papers decomposes catalogue-sector vs completion-sector bias — a
  related but distinct absence from the one this Stage-L pass is chartered to find.
- **MFG 2019** (arXiv:1809.02063): selection-consistency, not error-model self-calibration.

**R0 verdict:** confirmed gap — no paper in our own citation register proposes or executes
joint ensemble inference of a photometric error model with H₀.

---

## Ring R1/R2 candidates (ranked)

### 1. Hanselman, Vijaykumar, Fishbach & Holz 2024 — arXiv:2405.14818 (ApJ, "Gravitational-wave
   Dark Siren Cosmology Systematics from Galaxy Weighting")

**Signature match: HIGH.** This is the single closest paper found to the symptom card's ask, and
it is a **named-but-deferred** result, not an executed method:

> §IV.5 "Diagnosing incorrect assumptions" (WebFetch-extracted, **re-verify against source before
> citing in a proposal**): "we suggest using hierarchical analysis ... to test for model
> misspecification such as an incorrect galaxy weighting scheme" but "we emphasize that using
> hierarchical analysis does not diminish the bias, nor does it give any information on what the
> correct weighting scheme should be." Then: "it should be possible to simultaneously infer the
> weighting scheme as well as H₀ by generalizing the idea laid out in [86]; however, we do not
> investigate this here."

Mapping hypothesis: their target parameter is the **host-galaxy weighting scheme** (luminosity
band/exponent), not our target (**the photo-z error curve** — bias/scatter/outlier fraction). The
mechanism they gesture at (population-level hierarchical model with a shared nuisance drawn from a
Gaussian hyper-distribution, diagnosed via the ensemble) is structurally the right shape for our
symptom card, but has **never been built for the error-KERNEL case**, only proposed by analogy for
the weighting-SCHEME case, and even that analogy is unexecuted.

**Reference [86] is UNVERIFIED** — my WebFetch pass returned "Vijaykumar, A., Fishbach, M.,
Adhikari, S., & Holz, D. E. 2024, ApJ, 972, 157, arXiv:2407.XXXXX" with a literal placeholder
arXiv suffix — this is almost certainly a **model-fabricated citation**, not read from source.
**Do not cite [86] anywhere until the actual bibliography entry is pulled from the arXiv-HTML
reference list** (`arxiv.org/html/2405.14818` §References) with a real id.

**Cost-to-test:** MEDIUM. Re-fetch the HTML version's reference list (cheap, minutes) to nail down
[86]'s real identity — this is the highest-value single follow-up from this whole search, because
it is likely either (a) the one paper that already builds the joint-inference machinery, in which
case it answers the symptom card directly, or (b) a host-weighting-only precedent that still
leaves the error-kernel case fully open. Either answer is decisive and cheap to get.

### 2. Zhang, Bean, Knox 2010 ("Self-calibration of photometric redshift scatter in weak-lensing
   surveys") — MNRAS 405, 359, arXiv:0910.4181

**Signature match: MEDIUM (math class, R3).** The origin of "self-calibration" as a term of art
in this literature: uses **shear cross-correlations between photo-z bins** to constrain the
scatter of the photo-z error distribution jointly with cosmology, without external spectroscopic
truth. Structurally closest existing *method*, but built for **survey-scale statistics**
(hundreds of thousands–millions of source galaxies, a smooth two-point correlation function
estimator) — not obviously portable to N≈40–200 discrete GW events with per-event likelihoods.

**Cost-to-test:** HIGH — would require inventing a cross-correlation-analog statistic over
O(100) sparse GW events rather than reusing the paper's actual estimator; the paper gives no
small-N validity statement (their N is always survey-scale) so there is no off-the-shelf
threshold to check us against.

### 3. Outlier self-calibration in weak lensing — arXiv:2007.12795 ("Photo-z outlier
   self-calibration in weak lensing surveys")

**Signature match: MEDIUM (math class, R3).** Infers outlier fraction η and an offset
distribution as shared hyperparameters via cross-correlation with lensing maps. WebFetch
(degraded — **re-verify**) suggests validity around **η ≲ 10–15%** — i.e., this machinery is
built for a *small tail* of badly-mismeasured redshifts sitting on top of an otherwise-good
photo-z sample. **Our symptom card's σ_z/z≈49% median is not a tail — it is the typical case.**
If this reading holds up, it is a clean, reportable **regime mismatch**: the self-calibration
literature's own stated domain (mostly-good-z, small-outlier-tail) is structurally different from
our regime (uniformly bad z), which would mean none of the outlier-self-calibration machinery
transfers without a domain extension nobody has published.

**Cost-to-test:** LOW to confirm the η-range claim from source text (one more targeted fetch);
MEDIUM to assess whether the underlying math (cross-correlation estimator) degrades gracefully
or catastrophically outside its stated η range — likely requires derivation, not literature.

### 4. "Taking the Weight Off: Mitigating Parameter Bias from Catastrophic Outliers in 3×2pt
   Analysis" — arXiv:2509.08052

**Signature match: MEDIUM (math class, R3).** General statement found: fixing (rather than
marginalizing over) an outlier fraction in the likelihood biases the downstream cosmological
parameter; marginalizing recovers unbiasedness. This is the generic *justification* for why joint
inference should work in principle — orthogonal confirmation of the symptom card's premise, not a
directly portable method (again built at survey/3×2pt scale, not O(100)-event scale).

**Cost-to-test:** LOW — cheap confirmatory read, does not need a build.

### 5. "Dark sirens and the impact of redshift precision" — arXiv:2502.17747

**Signature match: LOW-as-solution, HIGH-as-contrast.** Abstract (verified verbatim via WebFetch):
"we show that redshift outliers (as occur in realistic photometric redshift catalogues), do not
introduce bias into the measurement of H₀ ... In all three scenarios, we obtain unbiased
estimates of H₀." Important: this paper's "unbiased" result is obtained with a **correctly
specified** assumed error model (their photo-z-like scenario uses the *true* generating error
distribution in the estimator). Our symptom card's premise (railing, N-coherent bias) lives in the
**mis-specified-kernel** regime — exactly the regime our own register already logs as VIOLATED
(`H-a`). This paper is therefore evidence that **the field's default finding ("photo-z doesn't
bias H₀") is conditioned on the estimator knowing its own error model correctly** — a validity
condition worth a `LITERATURE_WARNINGS.md` row on its own, independent of the self-calibration
question. Cost-to-test: LOW (already quote-extracted).

### 6. "A robust cosmic standard ruler from the cross-correlations of galaxies and dark sirens" —
   arXiv:2412.00202 (R2, not fetched this pass — time-boxed)

**Signature match: LOW-MEDIUM.** Different observable class entirely (cross-correlation ruler,
not per-event catalogue matching) — flagged as an R2 forward-citation lead, not opened this pass.
Cost-to-test: unknown, not assessed.

### 7. "One Year and One Night to 1% in H₀: Efficient Spectroscopic Strategy for Dark Siren
   Cosmology" — arXiv:2607.17284 (title only, not opened this pass)

**Signature match: LOW as a self-calibration method** — its framing ("efficient *spectroscopic*
strategy") suggests the field's answer to our symptom is typically "get spectroscopic follow-up",
not "self-calibrate the photometric error model". Worth a one-line note in the "what the field
does not have" section: the dominant field response to bad photo-z appears to be *avoid it*
(spectroscopic follow-up, hybrid catalogues) rather than *characterize and marginalize it*.

---

## What the field does NOT have (reportable absence)

1. **No paper found builds genuine ensemble-level joint hierarchical inference of a photometric
   redshift error MODEL (bias curve + scatter + outlier fraction) simultaneously with H₀ in a
   standard-siren / dark-siren GW context.** The one paper that gestures at this exact shape
   (Hanselman et al. 2024, §IV.5) explicitly declines to build it and defers to an unverified
   reference [86] whose actual target (per the paper's own framing) is host-*weighting*-scheme
   self-calibration, not error-*kernel* self-calibration — a narrower, adjacent problem.
2. **No self-calibration method found states a validity condition at small N (~40–200 events).**
   Every self-calibration/shrinkage method located (Zhang 2010, the 2020 outlier variant) is
   built and validated at survey scale (10⁴–10⁶ galaxies, smooth correlation-function statistics).
   None of them has a stated small-N failure mode or extension — this repo would be extending
   into genuinely untested territory, not applying a validated small-N method.
3. **No paper found treats "most of the sample is badly measured" (our σ_z/z≈49% median) as the
   outlier-self-calibration regime.** The outlier-calibration literature's own stated domain is a
   *minority* mismeasured tail (η ≲ 10–15%, UNVERIFIED — recheck) riding on an otherwise-good
   photo-z sample — the inverse of our situation.
4. **No paper found documents a z-structured score-at-truth tilt (≈0 below z≈0.4, ≈−1 by z≈0.9)
   or grid-edge railing as a diagnostic signature of a mis-specified redshift kernel** in dark/
   standard-siren H₀ posteriors. This specific shape did not turn up in any of R0–R2's material —
   candidate novel diagnostic, consistent with this repo's existing "no literature precedent
   found" note on the O2/O3 three-channel decomposition (`LITERATURE_WARNINGS.md`, 2026-08-21).
5. **The field's stated "photo-z doesn't bias H₀" result (arXiv:2502.17747) is conditioned on a
   correctly-specified error model** — nowhere found is that claim tested under a mis-specified
   kernel, which is our own register's already-measured regime (`H-a`, VIOLATED).

---

## Proposed `docs/LITERATURE_WARNINGS.md` rows

```
## Hanselman, Vijaykumar, Fishbach & Holz 2024 — arXiv:2405.14818 (ApJ ad9393), "Gravitational-
wave Dark Siren Cosmology Systematics from Galaxy Weighting"

| # | warning/condition (location) | what it requires | our status | evidence |
|---|---|---|---|---|
| HVFH24-a | §IV.5 "Diagnosing incorrect assumptions" (WebFetch-extracted, quote UNVERIFIED — raw
fetch needed): hierarchical population-level analysis can DIAGNOSE a misspecified galaxy-weighting
scheme but "does not diminish the bias, nor does it give information on what the correct weighting
scheme should be" without a further step | a further self-calibration step, gestured at via
unverified ref [86], never built in this paper | `UNCHECKED` — closest documented near-miss to
the [HIER] symptom card's self-calibration ask; target is host-weighting scheme, not the photo-z
error kernel we need | Stage-L [HIER] sweep, `results/campaign51_20260728/realistic_20260729/
STAGE_L_HIER_20260825.md`; ref [86] identity itself UNVERIFIED, needs re-fetch |

## Dark sirens and the impact of redshift precision — arXiv:2502.17747

| # | warning/condition (location) | what it requires | our status | evidence |
|---|---|---|---|---|
| DSIRP25-a | Abstract (WebFetch-verified verbatim): "redshift outliers ... do not introduce bias
into the measurement of H0 ... In all three scenarios, we obtain unbiased estimates of H0" | the
estimator's assumed error model must be CORRECTLY specified relative to the generating process
(not stated as a caveat in the fetched abstract, but implied by the forecast-not-diagnose framing)
| `N-A`-pending — result obtained under a correctly-specified kernel; our venue's own register
already measures the opposite condition as VIOLATED (`H-a`); full-text check needed to confirm the
"correctly specified" reading before using this as a contrast citation | Stage-L [HIER] sweep,
same file as above |
```

---

## Not reached this pass (time-boxed)

R2 forward-citation sweep of Gray/Hitchhiker/MFG citing papers (beyond what search-engine ranking
surfaced organically) was not run as a formal citation-graph traversal — relied on keyword search
instead, which is a weaker instrument. R4 (numerics of railing under misspecified kernels,
generic Bayesian-misspecification literature) turned up only generic robustness/KSD-Bayes results,
none specific to redshift kernels or GW dark sirens — logged as OPEN, not as an absence, since the
generic-misspecification literature is large and was only lightly skimmed. arXiv:2412.00202 and
arXiv:2607.17284 were found but not opened.

## Recommended next action

Re-fetch `arxiv.org/html/2405.14818` specifically for its bibliography (cheap) to resolve
reference [86]'s real identity — this is the single highest-value unresolved thread from this
search and could either hand the [HIER] thread a ready-built method or close the "field has
nothing" finding with higher confidence.

---

## ORCHESTRATOR VERIFICATION APPENDIX (2026-08-25; quote-verification discipline)

The flagged reference [86] is RESOLVED at source (arXiv HTML bibliography, fetched directly):
the Hanselman+ 2024 §IV.5 sentence verbatim — "However, it should be possible to
simultaneously infer the weighting scheme as well as H0 by generalizing the idea laid out in
[86]" — and **[86] = Vijaykumar, A., Fishbach, M., Adhikari, S., & Holz, D. E. 2024, ApJ,
972, 157, doi:10.3847/1538-4357/ad6140** (neighbors [85] Uddin+ 2024, [87] SciPy — context
confirms). The searcher's fabrication flag on its own extraction was correct caution; the
resolved citation supersedes it. Follow-up for the proposal: read [86]'s actual method (the
"idea" to generalize) before the prereg — queued as the proposal's stage-L obligation.
