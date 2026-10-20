# Literature reference for the class-G S̄_φ de-double-weight fix

**Purpose:** supply the literature-reference item of the six-item `/physics-change`
package for the author-granted, not-yet-run "class-G S̄_φ de-double-weight fix"
(`RUNBOOK_NEXT_SESSION_34.md:17`, `RUNBOOK_NEXT_SESSION_35.md:73`; measured effect
13.5% (identity) / 16.0% (BR), `p3_2d_forensic_20260826/C2_star_review.md:107-108`).
**No code was written or edited to produce this package.**

---

## 0. First — what the fix actually is and where it lives (load-bearing correction)

Before the literature question can be answered honestly, one fact has to be stated
precisely, because it changes what kind of reference is even the right one to look
for.

**Location, confirmed by reading the code directly**
(`darksiren_emri/validation/correspondence_1d.py:1605-1756`,
`_draw_2d_accepted_latents`): this is the **class-G mock-universe / twin generator**
used by the `[P3-2D]` correspondence/coverage-testing harness
(`darksiren_emri/validation/`), **not** `bayesian_inference/bayesian_statistics.py`
(the production catalogue likelihood). `correspondence_1d.py` is **not** on
`CLAUDE.md`'s physics-change trigger-file list (`physical_relations.py`,
`constants.py`, `LISA_configuration.py`, `parameter_estimation.py`,
`bayesian_statistics.py`, `simulation_detection_probability.py`,
`cosmological_model.py`) — this is a fact for whoever runs the gate to weigh, not a
ruling by this package.

**Mechanism, confirmed by reading the function body:**

1. `_draw_kernel_survival_redshifts` (called at `correspondence_1d.py:1687-1696`)
   draws `z_true` from a density already proportional to `k̄_g·S̄_φ(z)` — the
   survival factor is baked into the *sampling density* once.
2. The function then draws a latent mass and evaluates `S_4D(d_L(z_true), M_z_true)`
   at the drawn triple, and applies **Bernoulli(S_4D)** rejection on top
   (`correspondence_1d.py:1711-1723`, `accept_mask = u_batch < s4d_batch`).

The function's own docstring (`:1616-1628`) states this "reproduces exactly the
target joint law up to the (unchanged) z-marginal's own existing survival
weighting," registering it as an *intentional additional selection layer* per
`PREREGISTRATION_P3_2D_20260825.md`. The forensic review found empirically that this
construction does **not** reproduce the mixture's own predictive law — the
accepted-event law realized by the harness is "(model class-G law) × S̄_φ(z_ev),
renormalized" (`C2_star_review.md:45`), i.e. **one extra factor of the survival term**
relative to the target `ḡ₂ = B_num_wbh/β̄_Ḡ_φ`, which requires
`M̂|z ~ g_sel(z,·)/S̄_φ(z)` with the S̄_φ appearing **once** (`C2_star_review.md:53,131`).
The granted fix (`C2_star_review.md:167-170`, task 3(b1)) is stated as:

> "remove the double survival weight in the 2D branch (z-draw from `k̄_g·w_pop`
> without the S̄_φ factor when the Bernoulli(S_4D) layer is active, or keep the
> z-draw and drop the Bernoulli in favor of an S̃-reweighting — **one of the two,
> not both**); harness-only (`correspondence_1d.py`)"

So: this is a **generative Monte-Carlo double-counting bug** — a survival/detection
probability baked into an importance-weighted draw density, then applied a second
time as an independent accept/reject step on the same triple — inside a
mock-universe validation generator, not a numerator-vs-normalization placement bug
inside the production likelihood itself. This distinction matters for which
literature is actually load-bearing (§2 below) versus which literature is the
project's own established anchor for a related-but-different question (§1).

---

## 1. What the project has already established (read from the repo, not re-derived)

`docs/LITERATURE_WARNINGS.md` and `docs/derivations/fixb_pathA_phi_marginal_selection.md`
were read in full. Relevant established rows, quoted as they stand in the register
(not re-verified by me except where marked):

- **G23-b** (`LITERATURE_WARNINGS.md`): Gray et al. 2023 §2.1.3, after Eq. 2.18 —
  a **conditional** escape clause: skipping renormalization after truncating the
  redshift range is harmless **only if** "the identical truncated expression is
  used consistently in both the numerator (Eq. 2.14) and the normalizing object
  (Eq. 2.18) that propagates into the selection side." Status before my pass:
  `UNCHECKED` against our own code (not against the paper text — the paper text
  itself had not been independently re-fetched in this register entry).
- **G23-c**: Gray et al. 2023 §2.1.4 — the comoving-volume LOS-prior H0-dependence
  cancellation between numerator and selection denominator requires "the **same
  LOS prior object**... so the H0-dependent normalization constant cancels."
- **G23-c-check**: the project's *production* fused selection term
  (`completion_mass_factor_g_sel` + S̄_φ numerator insertion, commit `2b10b8b8`) was
  independently code-audited and found **CHECKED — SAME-OBJECT**: one survival
  accessor and one `phi_survival_table`, built once, feed both the S̄_φ numerator
  slot and the Σ⁴ᴰ/Σᶲ denominator slots. This establishes that the *production*
  numerator/normalization pairing already satisfies Gray 2023's condition — it is
  evidence about `bayesian_statistics.py`, **not** about the
  `correspondence_1d.py` harness bug this package is about.
- **`docs/derivations/fixb_pathA_phi_marginal_selection.md:48-50,265-267`**: the
  Fix-B/path-A package that defined S̄_φ in the first place cites "Mandel, Farr &
  Gair (2019), arXiv:1809.02063, Eqs. (5)–(7) — selection α must use the *same*
  population and detection model as every numerator (A2)" as the violated
  principle for the *original* class-of-defect this codebase treats S̄_φ
  consistency problems under.
- **MFG-a** (`LITERATURE_WARNINGS.md`): flags that this exact MFG19 A2 citation is
  a **repo paraphrase, not verbatim-verified** against the arXiv text — status
  `UNCHECKED` prior to this pass. I verify it directly in §2 below (partial
  resolution of MFG-a, restricted to Eqs. 5–7; the paraphrase's wording is closely
  but not identically matched, see caveats).
- **Absence precedent** already on record (`LITERATURE_WARNINGS.md`, end of the
  Borghi/VanWyngarden block): "no surveyed paper... decomposes a mixture
  posterior's bias into catalogue-sector vs completion-sector conditionals" and,
  separately (fixb doc, item 4 of §7): "no published dark-siren analysis carries a
  compact-object mass observable in the catalogue/completion split." Both are cited
  by the task prompt as the project's established absence-reporting discipline;
  §3 below extends this list with a new absence specific to this fix.

---

## 2. What the literature says — verified directly, this pass

### 2a. Gray et al. 2023 (arXiv:2308.02281v2, JCAP) §2.1.3 — quote-verified

Fetched directly (`arxiv.org/html/2308.02281`). Verbatim, in order (each segment as
returned by the fetch, under its 125-char-per-quote constraint but reproduced here
in full reading order):

> "The fact that all probability densities must be properly normalised may cause
> the reader to wonder whether the choice of redshift range over which this is
> done so has an impact on this analysis. After all the term p(G|Ωi,Λ,I) will take
> a very different value if p(z,M|Λ,I) is normalised over a redshift range which
> is truncated at zmax=2 and one where zmax=10. However, it is useful to note that
> as long as the same expression for p(z,M|Λ,I) is also be used in Eq. [16], the
> normalisation will come out the front as a constant. As such, the choice of what
> redshift range over which to normalise p(z,M|Λ,I) has no bearing on the result."

Equation numbers as printed on the page: Eq. 2.17 (the full LOS prior), Eq. 2.18
(`p(G|Ωi,Λ,I) = ∭ p(G|z,M,m,Ωi,Λ,I) p(z,M,m|Ωi,Λ,I) dz dM dm`), Eq. 2.19 (the
catalogue-membership indicator, a product of Heaviside functions in apparent
magnitude and redshift cutoff), Eq. 2.20 (the same integral restricted to the
in-catalogue region). This confirms the G23-b register row's paraphrase
essentially verbatim — the escape clause is **conditional on object identity**,
not unconditional.

### 2b. Gray et al. 2023 §2.1.4 — quote-verified

> "Despite the dependence of individual terms on H0, when computed for different
> values of H0 the LOS prior is independent of its value down to some
> normalisation constant... Because the same prior is used to evaluate the GW
> likelihood and GW selection effects, this dependence cancels, meaning that the
> LOS prior can in fact be computed just once..."

Confirms G23-c essentially verbatim.

**What this establishes, precisely:** Gray 2023's own correctness argument for
letting a normalization/truncation choice be invisible to the final posterior is a
**same-object identity** condition applied *across* two structurally different
roles (an in-catalogue numerator integral, Eq. 2.18/2.20, and the selection-side
normalizing object, "Eq. [16]"/the analogous selection integral). It is **not** a
statement about a single survival/detection factor being applied twice *within*
the construction of one term (which is the actual defect in §0). Gray 2023 answers
"when is reusing the same object in two different roles correct" (answer: always,
provided it is truly the same object) — it does not by itself answer "what happens
if a survival factor is baked into a sampling density and then reapplied as an
independent accept/reject step on the same draw," because that is not a
numerator/normalization placement question at all; it is a **within-one-term**
double-multiplication question. See §3.

### 2c. Mandel, Farr & Gair (2019) (arXiv:1809.02063) Eqs. (5)–(7) — quote-verified,
resolving MFG-a partially

Fetched directly (`arxiv.org/html/1809.02063`). As rendered from the paper's LaTeX:

> Eq. (5): `p(d⃗|λ⃗′) = ∫dθ⃗ p(d⃗|θ⃗) p_pop(θ⃗|λ⃗′) / α(λ⃗′)`
>
> Eq. (6): `α(λ⃗′) ≡ ∫dθ⃗ [∫_{d⃗>threshold} dd⃗ p(d⃗|θ⃗)] p_pop(θ⃗|λ⃗′)`
>
> Eq. (7): `p({d⃗ᵢ}|λ⃗′) = ∏ᵢ ∫dθ⃗ p(d⃗ᵢ|θ⃗) p_pop(θ⃗|λ⃗′) / ∫dθ⃗ p_det(θ⃗) p_pop(θ⃗|λ⃗′)`

**Structurally, this confirms the repo's paraphrase's substance** (α must reuse the
same `p_pop` — and, by Eq. 6/7's own definition, the same detection/survival
integrand — as the numerator uses), but note the caveat under §4: the fetched
rendering did not return the paper's own prose statement of "assumption A2" in
Claude's words, only the equations; the repo's specific phrase "same population
*and* the same detection model" is a fair reading of Eqs. (5)-(7) (survival/`p_det`
appears **once**, exclusively inside α, and never as a second factor multiplying
the per-event term `p(d⃗ᵢ|θ⃗)` in Eq. 7's numerator) but was not literally located
as a single quoted sentence in this pass — MFG-a should stay flagged `UNCHECKED`
for the exact-sentence form and be updated to **partially resolved (structural)**
for the equations, not closed outright.

**What this establishes, precisely, and this is the closest the surveyed
literature comes to the actual bug:** in the canonical hierarchical/selection-effects
form (Eqs. 5-7), the survival/detection probability `p_det(θ)` (≡ this pipeline's
`S̄_φ`/`S_4D`) is a **single** multiplicative object, tied to the definition of
"belongs to the detected sub-population," and it appears **exactly once** — inside
the normalizing integral α. It does not additionally multiply the per-event
numerator term `p(d⃗ᵢ|θ⃗)` in Eq. (7); an event that has already been counted as
"detected" (i.e., already survived the accept/reject decision, real or synthetic)
is not multiplied by the detection probability a second time when its own term is
evaluated. This is the general algebraic fact the class-G bug violates: the
harness's `_draw_2d_accepted_latents` bakes survival into the sampling density
*and* re-applies it as an accept/reject step on the same event, which is formally
equivalent to counting a detected/accepted event's own survival probability twice
— the same "detection probability enters once per event" structure Eqs. (5)-(7)
encode, just violated in a generative-sampling implementation rather than in an
analytic normalization term.

---

## 3. Reportable absences (stated explicitly, per the task's own precedent)

- **No surveyed dark-siren methodology paper (Gray 2020, Gray 2023, the Hitchhiker's
  Guide, Mandel/Farr/Gair 2019, Alfradique/Bom/Castro 2025, Borghi et al. 2025,
  VanWyngarden et al. 2025 — the full roster in `docs/LITERATURE_WARNINGS.md`)
  discusses, names, or gives a correctness criterion for the specific configuration
  this fix corrects: a survival/selection factor folded into an importance-weighted
  Monte-Carlo *sampling density* for synthetic/mock "detected" events, combined
  with an *additional*, independent Bernoulli accept/reject step evaluating the
  same (or a related) survival quantity on the same drawn triple.** This is a
  synthetic-data-generation (twin/mock-universe/coverage-harness) procedure, and
  none of the surveyed papers discuss how to generate mock detected catalogues at
  all — they are inference papers, not injection/mock-generation papers. This
  extends the project's own established absence-reporting precedent (no cited
  paper treats mass-covariate deconvolution; none decomposes bias into
  catalogue/completion conditionals) with a new, adjacent absence: **none treats
  mock-detected-population generation's own selection-consistency requirement.**
- Correspondingly, **no literature reference constrains the correct form of this
  specific fix beyond first-principles/elementary probability** (§2c's structural
  reading of Eqs. 5-7, plus the trivial fact that rejection sampling with
  acceptance probability `S(x)` applied to a draw already made from a density
  `∝ p(x)·S(x)` realizes an effective target `∝ p(x)·S(x)²`, not `∝ p(x)·S(x)` —
  this is elementary rejection-sampling/importance-sampling arithmetic, not a
  citable domain-specific result, and needs no reference beyond a standard
  Monte-Carlo-methods text such as Robert & Casella, *Monte Carlo Statistical
  Methods* (2004), which was **not** fetched or quote-verified in this pass — named
  here as a candidate general-methods citation only, not verified).
- Gray 2023 §2.1.3/§2.1.4 (verified §2a-b above) is the project's correct anchor for
  a **different, already-resolved** question: why the *production* S̄_φ
  numerator/normalization pairing in `bayesian_statistics.py` is self-consistent
  (G23-c-check, already `CHECKED`). It is **not** the literature that fixes the
  correct form of the class-G harness bug, and citing it as such in the
  physics-change package's reference item would overstate what was verified here.

---

## 4. Caveats

- The WebFetch tool renders arXiv HTML through an intermediate summarizing model;
  the equation and prose quotes above were cross-checked for internal consistency
  across two separate fetch calls each (Gray 2023: one call for §2.1.3/§2.1.4
  equation numbers, one for the full paragraph; MFG19: one call for
  title/abstract, one for Eqs. 5-7) and match. This is the same "confirmed across
  two independent fetches" discipline the register already uses for
  WebFetch-sourced quotes (e.g. ABC25-a) but is **not** a raw-PDF cross-check — the
  register's own convention (see the ABC25 section header) is to raw-PDF-verify
  before a quote is used in the paper itself; that step was not performed here.
- MFG-a in `LITERATURE_WARNINGS.md` should be updated to reflect this pass:
  Eqs. (5)-(7) are now quote-verified and structurally support the repo's
  paraphrase, but the single-sentence "A2" wording was not located verbatim — the
  row should move to a qualified status (e.g. "equations verified; prose
  paraphrase not literally located"), not to a bare `CHECKED`. I have not edited
  the register myself, per the no-code/no-file-edits constraint on this package
  (only the file named in the task was written).
- This package establishes the *general correctness principle* (survival enters
  once, consistently) and traces it to the two literature anchors the project
  already uses elsewhere (MFG19 Eqs 5-7; Gray 2023 §2.1.3/§2.1.4 for the
  cousin numerator/normalization-identity question). It does **not** derive the
  fix's exact replacement formula (which of the two disjunctive options in
  `C2_star_review.md:168-170` — drop the S̄_φ weighting from the z-draw, or drop
  the Bernoulli(S_4D) layer — is the correct one for the harness's registered
  purpose) — that choice is a code-level/harness-design decision for the
  physics-change author, not something the surveyed literature adjudicates between.
- Per CLAUDE.md's physics-change trigger list, `correspondence_1d.py` is not a
  listed trigger file; whether this fix nonetheless warrants the full six-item
  gate (it does encode a generative physical/statistical model, even though it
  lives under `validation/`) is a scope question for the author/orchestrator, not
  resolved by this literature package.

---

## 5. Sources fetched this pass

- `https://arxiv.org/abs/2308.02281` — title/author/abstract/version confirmation.
- `https://arxiv.org/html/2308.02281` — §2.1.3 (Eqs. 2.17-2.21) and §2.1.4, quoted
  above.
- `https://arxiv.org/abs/1809.02063` — title/author/abstract confirmation.
- `https://arxiv.org/html/1809.02063` — Eqs. (5)-(7), quoted above.

## 6. Repo evidence read this pass

- `docs/LITERATURE_WARNINGS.md` (full file).
- `docs/derivations/fixb_pathA_phi_marginal_selection.md` (full file).
- `docs/derivations/PROPOSAL_SIGMA_PHI_DIVISOR_20260822.md` (full file — a related
  but distinct, still author-gated, production physics-change proposal for the
  no-BH catalogue divisor; not the subject of this package).
- `results/campaign51_20260728/RUNBOOK_NEXT_SESSION_34.md` (full file).
- `results/campaign51_20260728/realistic_20260729/p3_2d_forensic_20260826/C2_star_review.md`
  (full file).
- `results/campaign51_20260728/realistic_20260729/gate_b_20260730/BIAS_HISTORY_LEDGER.md`
  rows #209-#210 (grep-located, read in place).
- `darksiren_emri/validation/correspondence_1d.py:1595-1770`
  (`_draw_2d_accepted_latents` and its docstring — read directly, not summarized
  from a derivative doc).
