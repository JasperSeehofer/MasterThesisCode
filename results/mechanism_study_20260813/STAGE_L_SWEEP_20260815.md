# Stage L literature sweep (R0 ring) — T_res / score-balance registration

**Status: PRESENTED, NOT ADJUDICATED.** Per research-cycle amendment A5 (Stage L: external
consult, `docs/RESEARCH_CYCLE.md`), this is the ring-R0 pass: search the published literature for
what is already known before the next preregistration (locating the unlocated `T_res` bias
component and testing a score-balance hypothesis for it) re-derives or mis-attributes it. No
branch call, repair, or register status change is made here — that is the author's, against this
artifact. Follows `docs/LITERATURE_WARNINGS.md` conventions: every claim below is attributed to a
specific paper by arXiv ID, with a stated relevance verdict and predicted-behaviour-for-our-setup,
kept separate from what we have actually measured.

**Context this sweep targets** (not asserted as literature findings — this is the shape of the
open question the search was aimed at): our validation-venue estimator smears galaxy-catalogue
photo-z with a per-candidate Gaussian kernel `N(z; z_obs, sigma_z)` multiplying a GW distance
likelihood in `d_L/d_obs`, integrates `dz` **without** the `|d d_L/dz|` Jacobian, truncates the
integration window (±4σ_d, ±5σ_z) without renormalizing over the retained support, and subtracts
an `N ln alpha(h)` selection-normalization term. Measured (this thread, not this sweep): MAP
displaced ≈ +1×σ_z in `h` (H0 biased high), coverage 0/N (posterior ≈8.5× overconfident);
restoring the Jacobian removes ≈48% of the bias; the remainder rides an uncancelled aggregate
log-posterior tilt at truth ≈ the α-term's own tilt, plus an unlocated dose-dependent residual
(`T_res`).

---

## Q1 — Measure/Jacobian treatment in published dark-siren pipelines

**Papers found:**
- **gwcosmo 3.0 / GPU systematics** — arXiv:2605.23538, *Scalable Dark Siren Cosmology with
  gwcosmo: GPU Acceleration, Validation and Systematics* (2026).
- **icarogw** — Mastrogiovanni et al. 2021/2023 (no single settled arXiv ID surfaced by this
  search; icarogw 2.0 is cited via its companion papers, e.g. the GWTC-4/5 cosmology releases
  below). Not independently confirmed at the equation level in this pass — flagged `UNCHECKED`
  below, not asserted.
- **GWTC-4.0 cosmology** — arXiv:2509.04348, *Constraints on the Cosmic Expansion Rate and
  Modified Gravitational-wave Propagation* (LVK, 2025).
- **GWTC-5.0 cosmology** — arXiv:2605.27227, companion release (2026).
- **Echoes from the dark** — arXiv:2509.18243 / A&A companion, *Galaxy catalog incompleteness in
  standard siren cosmology* (2026).

**Relevance verdict.** None of the abstracts/summaries retrieved in this pass state the
`|d d_L/dz|` Jacobian explicitly as a named quantity — the standard hierarchical-likelihood
derivation (Gray et al. 2020/2023 form, which our `docs/LITERATURE_WARNINGS.md` already registers
as our template) is written with `p(z | ...)` and `d_L(z)` linked through the fixed cosmological
relation `d_L(z) = (1+z) ∫ c dz'/H(z')`, and standard practice in every published pipeline this
sweep touched is to sample/marginalize *in z* with the GW likelihood re-expressed as a function of
`z` via this relation — which requires the Jacobian to keep the density normalized in the sampling
variable. This sweep did **not** turn up a published erratum or discussion of a *missing*-Jacobian
bias specifically — no paper says "an earlier pipeline dropped `|d d_L/dz|` and this biased H0."
That silence is notable given how basic the requirement is: it suggests either (a) it is treated as
too elementary to warrant a dedicated erratum, folded silently into "we integrate over the
posterior in z with the standard change-of-variables," or (b) no publicly documented pipeline has
made this specific error before. This sweep cannot distinguish (a) from (b) — full-text
verification of gwcosmo/icarogw source or the Gray 2020/2023 method papers at the equation level
was not performed here (report-level web search only).

**Predicted behaviour for our setup.** If (a) holds — the Jacobian is standard and just not
separately discussed — then the ≈48% bias reduction from restoring it in our estimator is exactly
what standard practice would predict: we were the ones out of step, not the literature. Nothing
found here contradicts our measurement; nothing found here explains the residual 52%.

**Status:** `UNCHECKED` at the source-equation level for gwcosmo/icarogw specifically — recommend a
follow-up equation-level read of Gray et al. 2020 (arXiv:1908.06050) §2 and Gray et al. 2023
(arXiv:2308.02281) §2.1 already flagged `UNCHECKED` in `docs/LITERATURE_WARNINGS.md`, rather than
re-deriving from a fresh Stage-L pass.

---

## Q2 — Photo-z smearing in dark-siren H0: direction and scaling of the systematic

**Papers found:**
1. **arXiv:2302.12037 / MNRAS stad3110** — Turski? (unresolved first author in this pass; cited
   as *"Impact of modelling galaxy redshift uncertainties on the gravitational-wave dark standard
   siren measurement of the Hubble constant"*), MNRAS 526, 6224 (2023).
2. **arXiv:2503.18887 / Phys. Rev. D vd36-3mys** — *Systematic bias in dark siren statistical
   methods and its impact on Hubble constant measurement* (2025).
3. **arXiv:2502.17747** — *Dark sirens and the impact of redshift precision*, PASA (2025/2026).
4. **arXiv:2505.13568** — *The Luminosity of the Darkness: Schechter function in dark sirens*
   (tangential — galaxy weighting, not directly photo-z scaling; listed for completeness).

**Relevance verdict.** Paper 1 (2302.12037) is the most directly on-point: it compares standard
Gaussian, modified-Lorentzian, and no-uncertainty redshift-error models and reports that *"not
using redshift uncertainties at all can lead to a potential bias comparable with other potential
systematic effects previously considered for GWTC-3 H0 measurements, though still small compared
to the overall statistical error."* This is a bias-from-*omitting* σ_z statement, not a
bias-from-*including* σ_z-at-the-wrong-order statement — it does not isolate whether the bias it
reports scales as O(σ_z) or O(σ_z²); the abstract-level summary explicitly could not resolve this
scaling question in this pass. Paper 3 (2502.17747) reports the *opposite* framing —
"unbiased estimates of H0" across all three redshift-precision scenarios tested, including
realistic photometric outliers — which is a **direct precedent claim** that Gaussian photo-z
smearing done *correctly* (their pipeline) should not bias H0 at all, only degrade precision and
interact with completeness. Paper 2 (2503.18887) is squarely about *dark-siren systematic bias*
by name but the retrieved summary attributes its bias mechanism to catalogue incompleteness and
**incorrect host weighting schemes**, not explicitly to the photo-z kernel's functional form or
order.

**No paper surfaced in this pass makes an explicit "O(σ_z) linear H0 bias" claim as distinct from
O(σ_z²) variance inflation.** This is a real gap relative to our need (Q2 asked for exactly this),
not a null result to over-read: it may exist in full text (e.g. as a derivative expansion) that a
report-only search did not surface, or it may genuinely not be a claim anyone has made in these
terms — dark-siren papers to date frame redshift-uncertainty impact primarily as "included vs
not," "photometric vs spectroscopic," and "weighting-scheme correct vs incorrect," not as an
order-of-σ_z Taylor decomposition of the resulting H0 shift.

**Predicted behaviour for our setup.** Paper 3's "unbiased with realistic photo-z outliers" result
is in tension with our measured +1σ_z MAP displacement — but Paper 3's pipeline is not confirmed to
share our specific defects (no-Jacobian, unrenormalized truncation); if their kernel treatment is
Jacobian-correct and properly renormalized, their null-bias result is consistent with our own
Jacobian-restoration halving the bias, and would predict the *remaining* (Jacobian-corrected)
component should also vanish under a fully standard treatment — i.e. it predicts our residual is a
**second, still-uncorrected defect** (truncation renormalization and/or the α-tilt interaction),
not an inherent property of Gaussian photo-z smearing itself.

**Status:** `UNCHECKED` for the specific O(σ_z) vs O(σ_z²) scaling claim (Q2's literal ask); the
direction/no-bias-if-done-right claim (Paper 3) is a genuine external data point worth folding into
`docs/LITERATURE_WARNINGS.md` as a new row.

---

## Q3 — Selection-term/score balance: misspecified-likelihood MAP displacement

**Papers found (general statistics):**
1. **White (1982)** and **Kleijn & van der Vaart (2012)** — foundational misspecification
   asymptotics (pre-arXiv-era / classical refs, confirmed via multiple 2025-26 papers citing them
   verbatim in this pass, e.g. arXiv:2604.03398 *Robust Standard Errors for Bayesian Posterior
   Functionals via the Infinitesimal Jackknife*).
2. **Müller (2013)**, Econometrica — *Risk of Bayesian Inference in Misspecified Models, and the
   Sandwich Covariance Matrix*.
3. **arXiv:2604.03398** — sandwich-form posterior-functional standard errors under
   misspecification (2026), useful as a recent restatement of the White/KvdV result.

**Papers found (GW-specific):**
4. **Mandel, Farr & Gair (2019), arXiv:1809.02063** — *Extracting distribution parameters from
   multiple uncertain observations with selection biases*, MNRAS 486, 1086. The standard reference
   for combining per-event measurement uncertainty with population-level selection effects in a
   hierarchical Bayesian framework — the general form our per-candidate-kernel × selection-α
   structure is an instance of.
5. **Essick & Fishbach (2024), arXiv:2310.02017** — *"DAGnabbit!" Ensuring Consistency between
   Noise and Detection in Hierarchical Bayesian Inference*, ApJ 962, 169. Shows that several
   selection-effect approximations used in the GW population-inference literature correspond to
   detection processes that are internally inconsistent (violate the implied DAG), which
   *generically* biases inference — the assumption that detectability is independent of the
   observed data given the true source parameters is the specific failure mode named.

**Relevance verdict.** The general statistics result is exactly what Q3 asked for: under
misspecification, the Bayesian posterior concentrates near the KL-minimizing pseudo-true
parameter (White 1982; Kleijn & van der Vaart 2012), and the MLE/posterior variance under the
*wrong* model differs from the sandwich-form true variance `H⁻¹JH⁻¹`. This is the textbook
mechanism for "MAP displaced by an amount related to a misspecification tilt, scaled by the
model's own (wrong) posterior variance" — structurally the same shape as our "tilt × posterior
variance" hypothesis for the uncancelled residual, though this sweep found no paper stating the
relation as literally "displacement ≈ tilt × variance" in closed form; that specific functional
form reads as a plausible **local (first-order) approximation** to the KL-argmin shift rather than
a named, citable result. Essick & Fishbach (arXiv:2310.02017) is the most targeted GW-specific hit
for the *selection*-term half of Q3: it shows that internally-inconsistent selection-effect
treatments (e.g. detectability assumed independent of observed data given true parameters, which
is close in spirit to a per-candidate kernel integrated against a detection probability that does
not itself carry the same truncation/renormalization defects as the numerator) generically bias
inference — but it is framed for population/rate inference (mass/redshift distributions), not
H0-from-single-event-catalogue dark sirens, so the mapping to our α(h) selection-normalization term
is structural, not literal.

**Predicted behaviour for our setup.** The general misspecification result predicts exactly the
observed shape: since our α(h) selection term is (per the thread's framing) itself "correct" while
the event-term (unrenormalized truncated kernel, and pre-fix, no Jacobian) is misspecified, a
KL-argmin displacement between the true and fitted models is expected, and it should scale with
the *fitted* model's own posterior variance (hence "8.5× overconfident" and "displaced" being
linked phenomena rather than independent facts) — consistent with, though not proof of, the
score-balance hypothesis motivating the next preregistration.

**Status:** the qualitative mechanism (KL-tilt → MAP shift, scaled by the misspecified posterior's
own variance) is `documented` in general statistics (White 1982; Kleijn & van der Vaart 2012); the
literal "displacement ≈ tilt × variance" functional form used operationally in this thread is
**not found as a named citable result** in this pass and should be treated as a working
approximation to be derived/justified from first principles at pre-registration, not cited as an
established theorem.

---

## Q4 — Truncated-and-unrenormalized kernels

**Papers found:**
1. **arXiv:2302.12037 / MNRAS stad3110** (same as Q2#1) — redshift-error models with the note that
   error distributions truncated at z=0 (unphysical negative redshift) "must be rescaled" — i.e.
   this paper explicitly names renormalization-after-truncation as required.
2. Search did not surface a distinct paper specifically quantifying the *bias magnitude* from
   skipping that renormalization step (as opposed to stating that it should be done).

**Relevance verdict.** Direct hit on the requirement, indirect on the consequence. The retrieved
summary states plainly that truncating a Gaussian redshift-error kernel at the physical boundary
(z=0) requires rescaling so the retained probability integrates to 1 — this is the standard
truncated-normal renormalization our estimator's ±5σ_z window (and separately, the GW-likelihood's
±4σ_d window) is not doing. No paper in this pass went on to quantify what happens to H0 inference
*specifically* when that step is skipped — the literature treats it as a modeling requirement
stated in passing, not as a systematic-bias case study in its own right.

**Predicted behaviour for our setup.** Consistent with treating this as a real, documented
requirement we are violating (not a novel concern) — but the literature gives no quantitative
prior for how much of our residual bias this specific omission should own. This sweep cannot
adjudicate whether the unrenormalized-truncation defect is folded into the ≈48%-already-fixed
Jacobian piece, into the residual, or is itself negligible; it only confirms the requirement is
real and named elsewhere.

**Status:** requirement `documented` (2302.12037); bias-magnitude-from-violation `UNCHECKED` in
the literature — no external quantitative benchmark exists to compare our residual against.

---

## Q5 — Posterior overconfidence / coverage failure in GW cosmology validation

**Papers found:**
1. **Gray et al. 2020, arXiv:1908.06050** — *Cosmological Inference using Gravitational-Wave
   Standard Sirens: A Mock Data Challenge*. Already registered in `docs/LITERATURE_WARNINGS.md` as
   our partition-norm template, validated at σ_z = 0 ("a simplified universe with no redshift
   uncertainties or galaxy clustering" per this pass's summary). This pass could not confirm
   whether it runs an explicit P-P/coverage test from the abstract-level summary retrieved —
   flagged `UNCHECKED` for that specific sub-claim, not asserted absent.
2. **arXiv:2502.14164 / MNRAS stag828** — *Implementing a Robust Test of Galaxy Catalogue
   Completeness for Dark Siren Measurements of the Hubble Constant* (2025/2026) — completeness
   testing, not a P-P/coverage test of the H0 posterior itself; tangential.
3. **arXiv:adda3a (ApJ) / Blinded Mock Data Challenge I** — *Assessing the Robustness of Methods
   Using Binary Black Hole Mass Spectrum* (2025) and its companion **arXiv:2604.26273**, *Blinded
   Mock Data Challenge: Is the Spectral Siren Technique Robust for Measuring the Hubble Constant?*
   (2026) — both are spectral-siren (mass-spectrum) mock-data-challenge papers, not catalogue dark
   sirens; still relevant as the closest published analogue to "run the pipeline on mock truth and
   check whether it recovers it," including reported "systematic biases due to mismatches between
   the astrophysical population model used for analysis and the true population model used to
   simulate BBH mock events" — a mismatched-model → biased-recovery result structurally parallel
   to our misspecified-kernel → biased/overconfident result, but for population (mass) parameters,
   not H0-from-catalogue.
4. General note from this pass: coverage/HPD diagnostics are described in the search summary as
   "a well-established metric" for GW-inference validation generally, but no dark-siren-specific
   H0 paper reporting a **coverage failure** (as opposed to a coverage *pass*) was surfaced.

**Relevance verdict.** The published dark-siren mock-data-challenge line (Gray 2020 and its
descendants) is method-validation-oriented and, as far as this pass could determine, has not
published a coverage failure mode resembling ours (confident-but-displaced, 0/N at nominal
credibility). The nearest documented analogue is on the *population* (mass-spectrum/spectral-siren)
side, where model mismatch between generator and estimator is explicitly shown to bias recovery —
supporting the general principle (mismatch → bias, and by the Q3 mechanism, mismatch → miscoverage)
without being a direct dark-siren-H0 precedent.

**Predicted behaviour for our setup.** No paper found predicts our specific coverage number or
confirms/refutes the 8.5× overconfidence figure — this appears to be new ground for the
catalogue/H0 side specifically, though the *general* mismatch→miscoverage principle is well
established in the adjacent spectral-siren MDC line and in the general misspecification literature
(Q3).

**Status:** `UNCHECKED` — no dark-siren-H0-specific coverage-failure precedent found; closest
analogue is population-side (mass spectrum) MDC papers, which is a structural parallel, not a
direct measurement of the same failure mode.

---

## Summary table — known vs novel here

| Our measured/hypothesized fact | Literature status | Basis |
|---|---|---|
| Standard hierarchical-likelihood z-marginalization requires the `|d d_L/dz|` Jacobian (implicit in every published pipeline's cosmological d_L(z) relation) | **known** (standard practice, not separately named as an erratum) | Q1 — no explicit "missing Jacobian" erratum found; requirement is structurally implicit in every retrieved pipeline description |
| A *missing*-Jacobian bias in dark-siren H0 specifically, of the magnitude we measured (≈48% of our total bias) | **apparently novel** — no published quantification found | Q1 |
| Omitting redshift uncertainty entirely biases dark-siren H0 (direction unspecified in retrieved summaries) | **known** (2302.12037) | Q2 |
| Photo-z uncertainty, modeled *correctly*, need not bias H0 at all (only degrades precision, interacts with completeness) | **known**, and in tension with our raw (pre-Jacobian-fix) result | Q2 (2502.17747) |
| O(σ_z) linear vs O(σ_z²) quadratic scaling of the photo-z-induced H0 bias | **not found / apparently open** | Q2 |
| Misspecified-likelihood MAP displacement toward the KL-minimizing pseudo-true value, with posterior variance following the (wrong) sandwich form | **known** (White 1982; Kleijn & van der Vaart 2012) | Q3 |
| "Displacement ≈ tilt × posterior variance" as an operational, literal closed-form relation | **not found as a named citable theorem** — reads as an unverified local-approximation working hypothesis | Q3 |
| Internally-inconsistent selection-effect treatment generically biases hierarchical GW inference (structural analogue to our α(h)-vs-event-term split) | **known**, but for population/rate inference, not catalogue H0 | Q3 (2310.02017) |
| Truncated Gaussian redshift kernels require renormalization over the retained support | **known**, stated as a modeling requirement (2302.12037) | Q4 |
| Quantified H0 bias specifically from *skipping* that renormalization | **not found** | Q4 |
| A dark-siren-H0-specific published coverage-failure mode resembling ours (confident-but-displaced, ~0/N) | **not found** — nearest analogue is population/mass-spectrum MDC papers showing mismatch→bias, not H0-catalogue coverage failure | Q5 |
| Our overall pipeline defect bundle (no-Jacobian + unrenormalized truncation + α-tilt interaction) as a *named, integrated* mechanism in one paper | **not found anywhere as a single treatment** | Q1–Q5 jointly |

**Caveat on this sweep's depth.** All findings above are report-level (search-result summaries and
abstract-level WebFetch extractions), not full-text equation-level verification. Per
`docs/LITERATURE_WARNINGS.md` conventions, `UNCHECKED` items above are genuinely unchecked, not
absent — several (Q1's Jacobian handling in gwcosmo/icarogw source, Q2's O(σ_z) scaling claim, Q4's
bias-magnitude quantification) would need a full-text or source-code read to move past
report-level confidence. This sweep's purpose was to prevent re-deriving what is already published
and to flag where our score-balance/T_res hypothesis has no existing literature anchor to lean on —
both purposes are served by the table above regardless of that depth limit.
