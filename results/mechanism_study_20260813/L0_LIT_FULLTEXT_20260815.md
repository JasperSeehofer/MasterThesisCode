# L0-LIT full-text verification: Gray et al. 2020 §2 / Appendix 2, Gray et al. 2023 §2.1–§2.2

**Status: PRESENTED, NOT ADJUDICATED.** Task L0-LIT (ledger row #105). This is a full-text,
equation-level read of the two papers `docs/LITERATURE_WARNINGS.md` flagged `UNCHECKED` in its
"Other sources" table, and that `STAGE_L_SWEEP_20260815.md` Q1 recommended as the necessary
follow-up beyond report-level search. No branch call, repair, or register status change on our own
pipeline is made here — that is the author's, against this artifact.

**Sources read.** Both papers' arXiv HTML render (`arxiv.org/html/...`) was incomplete/truncated
(the Gray 2020 appendix and exact equation text were not present in the HTML fetch; the Gray 2023
HTML fetch refused verbatim equation transcription citing copyright). Both full PDFs were
downloaded directly (`arxiv.org/pdf/1908.06050v4`, `arxiv.org/pdf/2308.02281v2`) and converted with
`pdftotext -layout`, then read directly (not summarized by an intermediate model) for this report.
Every equation quoted below was read in that converted text by me, not inferred. Where notation
could not be rendered faithfully in Markdown (Greek letters, hats, subscripts), I keep the
`pdftotext` rendering as closely as possible and flag anything ambiguous.

---

## 1. Gray et al. 2020, arXiv:1908.06050 (PRD 101, 122001), "Cosmological Inference using
Gravitational-Wave Standard Sirens: A Mock Data Challenge"

### What was read

Main text §II (Bayesian Framework, incl. §II C "The galaxy catalog method" and §II C 2 "The
counterpart method"), and the full **Appendix ("Detailed methodology")**, specifically:
- Appendix 1 "A note on luminosity weighting and redshift evolution" (Eqs. A.1–A.4)
- Appendix 2 "A detailed breakdown of the galaxy catalog case" (Eqs. A.5–A.19), which is where the
  single-event, per-galaxy likelihood — the object our venue's per-candidate construction is
  compared against — is actually derived. Main text explicitly defers here: *"We leave the details
  of this derivation to Appendix 2."*
- Appendix 3 "The catalog patch case" (Eqs. A.20)
- Appendix 5 "GW selection effects" (Eqs. A.22 onward)

### Transcribed equations

Main text, Eq. (6) (top-level posterior):
```
p(H0 | {xGW}, {DGW}) ∝ p(H0) p(Ndet|H0) ∏_i p(xGWi | DGWi, H0)
```

Main text, Eq. (9) (marginalize over host-in-catalogue G vs Ḡ):
```
p(xGW | DGW, H0) = Σ_{g=G,Ḡ} p(xGW | g, DGW, H0) p(g | DGW, H0)
                  = p(xGW|G,DGW,H0) p(G|DGW,H0) + p(xGW|Ḡ,DGW,H0) p(Ḡ|DGW,H0)
```

Appendix, Eq. (A.2) (prior factorization for a galaxy at z, Ω, M):
```
p(z, Ω, M, m | s, H0) = δ(m − m(z,M,H0)) · [p(s|z)p(z)/p(s)] · p(Ω) · [p(s|M,H0)p(M|H0)/p(s|H0)]
```
p(z) here is stated explicitly as "the prior distribution of galaxies in the universe, taken to be
uniform in comoving volume-time."

Appendix, Eq. (A.5) — the marginalization defining the in-catalogue likelihood:
```
p(xGW | G, DGW, s, H0) = [1 / p(DGW|G,s,H0)] ∫∫∫∫ p(xGW|z,Ω,s,H0) p(z,Ω,M,m|G,s,H0) dz dΩ dM dm
```
— **the integration variable is z, not D_L.** `p(xGW|z,Ω,s,H0)` is written directly as a function
of z (the GW likelihood evaluated at the corresponding D_L(z,H0) internally, not shown as an
explicit change-of-variables step here).

Appendix, Eqs. (A.6)–(A.9) reduce this (via the same steps as A.1–A.2, expanding
`p(z,Ω,M,m|G,s,H0)` as a sum of delta functions at each galaxy's catalogued z, Ω, m) to a discrete
sum over the N catalogued galaxies, **with no redshift uncertainty yet**:
```
                    Σ_i p(xGW|zi,Ωi,s,H0) p(s|zi) p(s|M(zi,mi,H0))
p(xGW|G,DGW,s,H0) = ---------------------------------------------------      (A.9)
                    Σ_i p(DGW|zi,Ωi,s,H0) p(s|zi) p(s|M(zi,mi,H0))
```

Appendix, **Eq. (A.10)** — the equation that adds per-galaxy redshift uncertainty p(zi), which is
the direct structural analogue of our venue's per-candidate kernel:
```
                     Σ_i ∫ p(xGW|zi,Ωi,s,H0) p(s|zi) p(s|M(zi,mi,H0)) p(zi) dzi
p(xGW|G,DGW,s,H0) = ------------------------------------------------------------      (A.10)
                     Σ_i ∫ p(DGW|zi,Ωi,s,H0) p(s|zi) p(s|M(zi,mi,H0)) p(zi) dzi
```
Text immediately preceding: *"Notably, in the case the galaxies in the catalogs are provided along
with their redshift uncertainties p(zi), these can be implemented in the above equations as..."*
followed directly by Eq. A.10. **No Jacobian factor of any kind (`|dD_L/dz|`, `(1+z)`, or
otherwise) appears anywhere in Eq. A.10, its derivation (A.5–A.9), or the surrounding text.** A
`grep` of the full converted paper text for "Jacobian" returns exactly one hit, in Appendix 5/6
(mass-frame transform, `M_z = (1+z)M`, Eq. A.25 area) — an unrelated detector-frame/source-frame
mass Jacobian, not a distance–redshift one.

Appendix, Eq. (A.19) — the out-of-catalogue analogue (Ḡ term), same structure, integration limits
`z(M, mth, H0)` to `∞` instead of a discrete galaxy sum; **no Jacobian here either.**

Appendix, Eq. (A.22) (GW selection normalization):
```
p(DGW|H0) = ∫ p(DGW|xGW,H0) p(xGW|H0) dxGW
```
with `p(DGW|xGW,H0)` a binary detection-threshold indicator. The text notes this term is evaluated
in the *same expanded form* as the numerator (i.e. the denominators of Eqs. A.9/A.10/A.19 already
*are* this term, conditioned further on z, Ω) — **the selection normalization is not a separate
global multiplicative or log-subtracted factor; it is the same per-galaxy-summed integral,
evaluated with `p(DGW|·)` in place of `p(xGW|·)`, appearing as the denominator of the same
fraction as the numerator, per event.**

Footnote 3 (main text, just before §III): *"While uncertainties on the galaxy sky-coordinates can
be safely ignored, the error on the redshift can be modeled with a Gaussian or a more complicated
distribution."* — this is the only place a Gaussian form for p(zi) is even suggested; it is never
written as an explicit equation (no `G(z − ẑ; σ̂)`-style closed form anywhere in this paper), and
**no truncation or renormalization of p(zi) is discussed anywhere in the paper** (a full-text
`grep` for "truncat" and "renormali" returns zero hits).

Critically, the mock data analyses actually run in this paper **do not exercise p(zi) at all**:
main text states plainly, *"Our present mock data analyses ignore these crucial redshift
uncertainties altogether, and the impact of their magnitudes, profiles, and other systematic
artefacts are left aside for possible future study"* and again in §III, *"we neglect the effects of
large-scale structure and redshift uncertainties in the mock catalogs."* Eq. A.10 with a nontrivial
`p(zi)` is therefore a **derived-but-unexercised** equation in this paper — exactly the
characterization already flagged in `docs/LITERATURE_WARNINGS.md`'s "Other sources" row for this
paper, now confirmed at the equation level.

### Answers — Gray et al. 2020

**(a) Measure present/absent/implicit-in-variable-choice.** **Absent as an explicit term, and not
implicit either** in any way that resolves the question either way, because this paper's mock data
challenge never turns on p(zi) at all (σ_z = 0 throughout). Eq. A.10 — the one equation that *would*
carry the measure if it were needed — integrates `dzi` directly against `p(xGW|zi,Ωi,s,H0)` written
as a function of z, with no `|dD_L/dz|` or `(1+z)` factor anywhere in it or its derivation. Because
the GW term `p(xGW|zi,...)` is used throughout Appendix 2 as a **likelihood** (a weighting function
of z, obtained by evaluating the GW measurement's distance likelihood at `D_L(zi,H0)`), not as a
density that is itself being re-expressed from D_L into z, no explicit change-of-variables step
appears anywhere for it to carry a Jacobian on. The paper's marginalization is a standard
`∫ likelihood(z) · density(z) dz` — the object requiring proper normalization in z is `p(zi)`
(a bare, unequationed density) times `p(s|zi)`, not the GW term. This structurally does **not**
require a `|dD_L/dz|` factor **as long as p(xGW|zi,...) is genuinely only ever used as an
unnormalized-in-z likelihood weight and never separately asserted to integrate to 1 over z.**
Whether that holds for our venue's actual code is outside the scope of a literature read (see §3).

**(b) Kernel normalization convention.** Not specified. No closed-form for p(zi) is given (Gaussian
is only suggested in a footnote, never equationed), and no truncation/renormalization is discussed
anywhere in the paper. This paper simply provides no precedent either way for the
truncated-unrenormalized-vs-renormalized question.

**(c) Comparison to our venue's construction.** **Not comparable at the level of a functioning
counter-example or confirming precedent**, because this specific equation (A.10) is never actually
run in this paper — it is a derived-but-idle placeholder for a feature (redshift uncertainty) the
paper explicitly declines to model. What **is** transferable is the structural template: our
venue's kernel-times-GW-likelihood-integrated-dz form is the *same shape* as Eq. A.10 (both are
`∫ [GW likelihood as fn of z] × [redshift kernel density(z)] dz`), and neither, on its face, carries
a stated Jacobian requirement in this paper's own math — which is a genuine complication for the
"missing Jacobian" framing (see the novelty-verdict table, §4): if the GW term is legitimately used
as an unnormalized likelihood throughout (as Gray 2020's derivation treats it), the Jacobian
requirement as literally stated in the mechanism-study's own framing does not follow from this
paper's math. This paper is silent on selection-normalization placement questions relative to ours
beyond confirming the "same per-event fraction, not a separate global log-subtracted term"
structure, which **does differ** from our venue's `N ln α(h)` global-term placement as described in
the task brief.

---

## 2. Gray et al. 2023, arXiv:2308.02281 (v2), "Joint cosmological and gravitational-wave
population inference using dark sirens and galaxy catalogues" (gwcosmo update)

### What was read

§2 in full: §2 lead-in (Eqs. 2.1–2.4, the top-level posterior and pixel/z marginalization), **§2.1
"The line of sight redshift prior"** with all five subsections (§2.1.1 in-catalogue part, §2.1.2
out-of-catalogue part, §2.1.3 full LOS-prior expression, **§2.1.4 "The separability of GW mass and
rate evolution hyper-parameters from the line-of-sight redshift prior"**, §2.1.5 resolution), and
**§2.2 "Gravitational-wave selection effects"** (Eqs. 2.23–2.25).

### Transcribed equations

Eq. (2.3)/(2.4) — top-level posterior with explicit z-marginalization (θ = {z, θ′}):
```
p(Λ|{xGW},{DGW},I)
  ∝ p(Λ|I) p(Ndet|Λ,I) [ ∫∫ p(DGW|z,θ′,Λ,I) p(θ′|Λ,I) Σ_j p(z|Ωj,Λ,I) dθ′ dz ]^{−Ndet}
    × ∏_i [ ∫∫ Σ_j p(xGWi|Ωj,z,θ′,Λ,I) p(θ′|Λ,I) p(z|Ωj,Λ,I) dθ′ dz ]                     (2.4)
```
Note: `p(z|Ωj,Λ,I)` is the LOS *prior*, and `p(xGWi|Ωj,z,θ′,Λ,I)` is again written directly as a
function of z (i.e. the GW likelihood evaluated at `D_L(z,H0)` internally) — same pattern as Gray
2020. **No `|dD_L/dz|` appears in Eq. 2.4.**

Eq. (2.8)–(2.9) — the per-galaxy redshift kernel:
```
p(z, m | G, Ωi, I) = [1/N_gal(Ωi)] Σ_k p(z|ẑk) δ(m − m̂k)                     (2.8)
p(z|ẑk) = G(z − ẑk ; σ̂k)                                                     (2.9)
```
Text: *"i.e. that a Gaussian centered at ẑk with standard deviation σ̂k is a reasonable
approximation for the redshift posterior of galaxy k."* Footnote 9: *"there is no strict
requirement that the galaxy uncertainties be Gaussian, or even that they follow the same
distribution."* Footnote 10 additionally notes that if `p(z|ẑk)` is actually a *likelihood* rather
than a posterior, a prior (e.g. uniform-in-comoving-volume) ought to be applied — flagged as an
assumption the paper is choosing to skip by treating the catalogue values as posteriors already.
**No truncation of this specific kernel (e.g. at z=0) is discussed.**

Eq. (2.10) — in-catalogue LOS-prior contribution (weighted sum of Gaussians):
```
∫∫ p(z,M,m|G,Ωi,Λ,s,I) dM dm
  = [1/(p(s|G,Ωi,Λ,I) N_gal(Ωi))] Σ_k p(z|ẑk) p(s|z,M(z,m̂k,Λ),Λ,I)
```

Eq. (2.14) — out-of-catalogue contribution (the term that **is** explicitly discussed for
renormalization, see below); integrates a uniform-in-comoving-volume prior `p(z,M|Λ,I)` over M
between redshift-dependent limits `M(z,mth(Ωi),Λ)` and `Mmax(H0)` (or `Mmin(H0)` to `Mmax(H0)` above
`zcut`).

**Eq. (2.18)** — probability a galaxy is inside the catalogue:
```
p(G|Ωi,Λ,I) = ∫_0^{zcut} ∫_{Mmin(H0)}^{M(z,mth(Ωi),Λ)} p(z,M|Λ,I) dM dz
```

**The explicit renormalization/truncation discussion (§2.1.3, directly after Eq. 2.18):**
> "The fact that all probability densities must be properly normalised may cause the reader to
> wonder whether the choice of redshift range over which this is done so has an impact on this
> analysis. After all the term `p(G|Ωi,Λ,I)` will take a very different value if `p(z,M|Λ,I)` is
> normalised over a redshift range which is truncated at `zmax = 2` and one where `zmax = 10`.
> However, it is useful to note that as long as the same expression for `p(z,M|Λ,I)` is also be
> used in Eq. 2.14, the normalisation will come out the front as a constant. As such, the choice of
> what redshift range over which to normalise `p(z,M|Λ,I)` has no bearing on the result."

This is the paper's own, explicit statement of the truncation/renormalization convention: **the
kernel does not need to be independently renormalized after truncation, provided the exact same
truncated-and-unrenormalized expression is used consistently everywhere it appears** (here:
`p(G|Ωi,Λ,I)`, Eq. 2.18, and inside the out-of-catalogue term, Eq. 2.14) — the missing normalization
constant then cancels as a shared prefactor. This is a *conditional* statement, not a blanket "no
renormalization needed" claim — see the novelty-verdict discussion in §4.

Eq. (2.19) — full LOS prior (in-catalogue + out-of-catalogue, with the `1/p(s|Ωi,Λ,I)` prefactor
noted to cancel against an identical denominator term once substituted back into Eq. 2.4 — i.e. the
selection-normalization cancellation the LITERATURE_WARNINGS.md "Other sources" row already
flagged).

**§2.1.4** (Eq. 2.20–2.22) restates Eq. 2.19 with `p(s|z,M,Λrate,I)` factored out front and
explicitly names the H0-cancellation claim already recorded in `docs/LITERATURE_WARNINGS.md`:
> "Despite the dependence of individual terms on H0, when computed for different values of H0 the
> LOS prior is independent of its value down to some normalisation constant. Because the same prior
> is used to evaluate the GW likelihood and GW selection effects, this dependence cancels, meaning
> that the LOS prior can in fact be computed just once and used in an analysis in which
> {H0, Λmass, Λrate} vary."
**No Jacobian appears anywhere in §2.1.1–§2.1.4.** A full-text `grep` for "Jacobian" in this paper
returns exactly one hit, at line ~910 (§2.2, below) — again a mass-frame, not a distance–redshift,
use.

Eq. (2.23)–(2.25), **§2.2 "Gravitational-wave selection effects"** — this is the only place in the
paper a `∂d_L/∂z` Jacobian is written explicitly, and it is in the **injection-reweighting step of
the Monte-Carlo Pdet estimator**, not in the event-likelihood redshift marginalization of §2.1:
```
Nexp/N(Λ) = ∫∫∫ p(DGW|ms1,ms2,z,Λ,I) p(ms1,ms2,z|Λ,I) dms1 dms2 dz
          ≈ (1/Nsim) Σ_i [ p(ms1,i,ms2,i,zi|Λ,I) / π_inj^s(ms1,i,ms2,i,zi|Λ,I) ]              (2.23)

π_inj^s(ms1,i,ms2,i,zi|Λ) = π_inj(md1,i,md2,i,dL,i) · (1+zi)^2 · [∂dL/∂z]_{z=zi}               (2.24)

Nexp/N(Λ) ≈ (1/Nsim) Σ_i  p(ms1,i,ms2,i,zi|Λ) / { π_inj(md1,i,md2,i,dL,i) (1+zi)^2 [∂dL/∂z]_{z=zi} }   (2.25)
```
Text: *"where the Jacobians for transforming between source frame and detector frame, and dL and z
have been written explicitly."* This is a mass/rate-density reweighting Jacobian for converting an
injection set sampled in `(m^d_1, m^d_2, d_L)` into source-frame `(m^s_1, m^s_2, z)` densities for
the Monte-Carlo Pdet sum — a different (though physically related, same `∂d_L/∂z` quantity)
Jacobian use-case than a distance-likelihood-to-redshift-density conversion in the event term.

### Answers — Gray et al. 2023

**(a) Measure present/absent/implicit-in-variable-choice.** **Absent in §2.1 (event/LOS-prior
term), present-but-elsewhere in §2.2 (selection term, Pdet Monte-Carlo estimator).** The event
likelihood's z-marginalization (Eq. 2.4) has no Jacobian, for the same structural reason as Gray
2020: `p(xGWi|Ωj,z,θ′,Λ,I)` is used as a likelihood/weighting function of z (implicit
evaluate-at-`D_L(z,H0)`), multiplied against a genuinely-normalized-in-z prior/kernel
(`p(z|Ωj,Λ,I)`, built from the Gaussian `p(z|ẑk)` kernel plus the uniform-in-comoving-volume
out-of-catalogue term), and integrated. The `∂d_L/∂z` Jacobian *is* explicit — but only in §2.2's
different context (reweighting a d_L-sampled injection pool into source-frame-mass-and-z density
for the Pdet Monte-Carlo sum), not in the per-candidate galaxy-redshift kernel construction of
§2.1.

**(b) Kernel normalization convention.** `p(z|ẑk) = G(z−ẑk; σ̂k)` (Eq. 2.9) is a plain, unequationed
Gaussian with no stated truncation of that specific per-galaxy kernel. The paper *does* explicitly
discuss truncation/renormalization, but for a different object — the **out-of-catalogue
uniform-in-comoving-volume prior `p(z,M|Λ,I)`**, whose redshift range of normalization (`zmax=2` vs
`zmax=10`, quoted verbatim above) is stated to not matter **conditional on the identical truncated
expression appearing consistently in both `p(G|Ωi,Λ,I)` (Eq. 2.18) and the out-of-catalogue LOS
term (Eq. 2.14)** — the shared unrenormalized constant then cancels.

**(c) Comparison to our venue's construction.** Same structural-parallel finding as Gray 2020: the
shape "`kernel(z) × GW-likelihood(z) integrated dz, no Jacobian`" matches this paper's Eq. 2.4/2.10
construction on its face, given that both papers use the GW term as an unnormalized likelihood
throughout §2.1-equivalent material. **The truncation/renormalization question is where a real,
checkable difference could live**: Gray 2023's own escape clause for skipping renormalization is
conditional — it requires the *same* truncated-and-unrenormalized kernel expression to appear
identically in both the event numerator and wherever the equivalent normalizing constant is used
downstream (here, `p(G|Ωi,Λ,I)`, which propagates into the selection/Pdet side via the LOS prior
being shared between numerator and Pdet denominator, §2.1.4's cancellation argument). Our venue's
per-candidate window (`±4σ_d, ±5σ_z`) is **candidate-specific** — it depends on each candidate's own
`z_obs` and `σ_z`, which differ galaxy-to-galaxy and event-to-event — so if our selection-side
`α(h)` term does *not* apply the identical per-candidate truncation window (the task brief
describes it as an `N ln α(h)` global term, not a per-candidate-window-matched one), Gray 2023's own
cancellation condition would **not** be met, and the "constant cancels" escape this paper documents
would not straightforwardly transfer to our construction. This is a structural, not a magnitude,
finding — whether it actually applies requires reading our own selection-term code, which is out of
scope for this literature-only task (see novelty table, §4).

---

## 3. What could and could not be read

**Read directly, full text, by me:** Gray 2020 main text §I–§III + full Appendix (via
`pdftotext -layout` on the arXiv PDF, `arxiv.org/pdf/1908.06050v4`); Gray 2023 §1–§2.2 in full,
including all subsections of §2.1 and §2.2 (via `pdftotext -layout` on `arxiv.org/pdf/2308.02281v2`).
Both PDFs are open-access arXiv preprints, not paywalled. Every equation and quoted sentence above
was located and read in the converted text directly, not summarized by an intermediate model —
where an intermediate-model summary (WebFetch on the arXiv HTML render) was used earlier in this
pass, its claims were **re-verified against the raw PDF text** before being included here; two
summary-stage errors were caught this way (the HTML-render pass mis-numbered/garbled Eq. 2.22 and
mis-attributed the §2.2 Jacobian to §2.1.4 — both corrected against the PDF transcription above).

**Not read:** Gray 2020's §I, IV, V (results/discussion) beyond what's quoted for context; Gray
2023's §3 onward (data, results, discussion) and §2.1.5/§4 in detail beyond the excerpts above —
neither is needed for the redshift-marginalization/Jacobian/normalization questions this task
asked. No section of either paper was paywalled or unreachable; nothing here is inferred from an
abstract or a review-paper's characterization.

---

## 4. Novelty-claims verdict table

**PRESENTED, NOT ADJUDICATED.** This table states what the full-text read shows and does not
attempt to close the mechanism-study thread's open questions; branch calls remain the author's.

| Thread's novelty claim (from `STAGE_L_SWEEP_20260815.md` and the task brief) | What this full-text read found | Verdict after full-text read |
|---|---|---|
| "The `\|dD_L/dz\|` Jacobian is standard/implicit in every published pipeline's z-marginalization, just not separately named" (Stage-L Q1 hypothesis (a)) | **Not confirmed as stated.** In both Gray papers, the event-term z-marginalization (Gray 2020 App. 2 Eq. A.10; Gray 2023 Eq. 2.4/2.10) carries **no Jacobian at all**, explicit or implicit-by-derivation-step — because the GW term is used as an unnormalized likelihood function of z, not as a density being converted from D_L into z. The only explicit `∂d_L/∂z` Jacobian found (Gray 2023 Eq. 2.24) is in the **selection-effect Monte-Carlo Pdet reweighting**, a structurally different use. | **Refuted as literally stated** for these two papers specifically: it is not that the Jacobian is present-but-unnamed in the event term; it is genuinely absent from the event term in both papers' own math, for a structural reason (likelihood vs. density usage) that may or may not also hold for our venue's code — that check is out of scope here. |
| "A missing-Jacobian bias in dark-siren H0, of the magnitude we measured (~48% of total bias), is apparently novel — no published quantification found" | Consistent with, and now reinforced by, the full-text read: since neither template paper's own math needs this Jacobian in the event term the way our venue's construction was described as needing it, there is *more* reason (not less) to treat this as either (i) a genuine departure specific to our code's treatment of the GW term as a density rather than a likelihood, or (ii) a different mechanism than a literal missing-Jacobian defect that happens to respond to adding a Jacobian-shaped correction. Full text cannot distinguish these; it only sharpens that Gray's derivation does not supply a "yes, add the Jacobian" precedent the way the earlier report-level sweep assumed. | **Survives, with a new caveat that widens the claim's uncertainty**, not narrows it: full-text reading found no precedent *confirming* the Jacobian-shaped fix is the standard-practice repair; the question of *why* adding it worked empirically in our venue (48% bias reduction) is now less explained by "we were out of step with standard practice," since standard practice (per these two template papers) does not carry this term either. |
| "Unrenormalized-truncation bias" (kernel truncated at ±4σ_d/±5σ_z without renormalizing over retained support) is a departure from standard practice, magnitude unquantified in the literature (Stage-L Q4) | Gray 2023 §2.1.3 **explicitly discusses and defends** skipping renormalization after truncation — but only **conditional on the identical truncated expression appearing consistently in both the numerator and wherever the corresponding normalizing constant is used downstream** (Eq. 2.18 / Eq. 2.14 pairing, propagating via §2.1.4's H0-cancellation argument). Our venue's truncation window is per-candidate (depends on each candidate's own z_obs, σ_z) rather than a single global range shared identically across the numerator and the selection term as described in the task brief. | **Not settled by this read, but sharpened**: Gray 2023 supplies a *named, explicit* condition under which skipping renormalization is provably harmless (identical truncated kernel used symmetrically numerator vs. denominator) — which is a genuine literature anchor Stage-L's Q4 did not have. Whether our venue's per-candidate, non-shared truncation window satisfies that condition is a **checkable structural question about our own selection-term code**, not something this literature-only task can adjudicate. If it does *not* satisfy the condition, Gray 2023's own escape clause would not apply to us, which would support (not refute) treating our truncation defect as real; if our α(h) term *does* apply the same per-candidate window, Gray's cancellation argument may apply and the "bias" claim would need re-examination. |
| Kernel normalization convention: truncated Gaussian kernels "must be rescaled" (Q4, citing arXiv:2302.12037) vs. Gray's own template papers | Gray 2020: **silent** — no truncation/renormalization discussion at all (p(zi) is never equationed, only footnoted, and is unexercised in the paper's own mock data challenges). Gray 2023: **discusses it explicitly but reaches the opposite operational conclusion** from the "must be rescaled" framing — under the stated symmetric-usage condition, *not* rescaling is fine because the missing constant cancels. | **Genuine tension surfaced, not present in the earlier report-level Stage-L pass**: Q4's cited paper (2302.12037) frames truncation-then-renormalize as a requirement in general; Gray 2023 frames it as *conditionally optional*, not universally required. Both can be true simultaneously (the condition differs: 2302.12037's is a physical-boundary truncation at z=0 for an isolated per-galaxy kernel; Gray 2023's is a shared-range truncation of a distribution appearing symmetrically twice). This is a nuance for the pre-registration to account for, not a resolved contradiction. |
| Selection-normalization term enters "relative to the event term" as our venue's `N ln α(h)` global subtracted term | Gray 2020: the selection-normalization is **not** a separate global term at all — it is the *same per-galaxy-summed integral* as the numerator, with `p(DGW|·)` substituted for `p(xGW|·)`, forming the denominator of the same per-event fraction (Eq. A.9/A.10). Gray 2023: the selection term is `p(DGW|Λ,I)` raised to the `−Ndet` power multiplying the product of per-event likelihoods (Eq. 2.4) — i.e. also a genuinely shared object (built from the *same* LOS prior, per §2.1.4's explicit cancellation argument), not a separately-parameterized global correction. | **Both template papers place the selection normalization in a way that shares structure/parameters with the event numerator by construction** (same LOS prior feeding both, in Gray 2023; same per-galaxy sum structure, in Gray 2020). A venue where the event-term truncation window is per-candidate but the `α(h)` selection term is a single global correction is, on its face, a **structural departure from both templates' shared-object design** — this is a comparison worth carrying into the pre-registration, though again this read cannot confirm what our own code actually does; only what the task brief describes. |

**Overall reading of this pass.** The full-text read does not hand the mechanism-study thread a
confirmed external bug report or a confirmed external vindication for either the Jacobian claim or
the truncation-renormalization claim. What it does supply, newly, relative to the report-level
Stage-L sweep: (1) neither template paper's event-term math structurally needs a Jacobian the way
the earlier sweep assumed "standard practice" would; (2) Gray 2023 supplies a *named, explicit,
conditional* argument for why skipping kernel renormalization can be harmless — and the condition
it states (symmetric usage of the identical truncated expression in numerator and the shared
normalizing object) is a concrete, checkable structural question about our own code, not decided
here. Both findings sharpen, rather than close, the open novelty questions.

---

*Task L0-LIT, ledger row #105. No branch call is made in this artifact.*
