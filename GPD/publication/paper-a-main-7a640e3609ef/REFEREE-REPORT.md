---
reviewed: 2026-07-02T20:52:13Z
scope: manuscript
target_journal: mnras
recommendation: major_revision
confidence: medium
major_issues: 9
minor_issues: 14
---

# Referee Report

**Scope:** Full manuscript review (staged six-agent panel, final adjudication, round 1)
**Manuscript:** `paper_a/main.tex` (sha256 `af60a4304bfca7e325a5934f291509e93439a954df61afe0c76f78094d4dc7e1`, 21-pp MNRAS draft)
**Date:** 2026-07-02T20:52:13Z
**Target Journal:** MNRAS

## Summary

The paper reports the diagnosis and repair of a railed Hubble-constant posterior in an independently implemented Gray-et-al.-family galaxy-catalogue dark-siren estimator, applied to a simulated LISA EMRI campaign (injected h = 0.73) against the real GLADE+ catalogue. Its diagnostic arc is genuinely strong: three normalization defects are identified under a single prior-consistency invariant; a falsifiable sign-flip prediction (rail at h = 0.86 → flip to h = 0.60 under the 1/(4π)-only fix → interior peak at 0.73 under either prior-consistent repair) is stated and then observed on identical data; a two-factor ablation cleanly separates kernel consistency from the denominator choice; and an Eddington-in-z σ_z² bias law is derived, numerically verified across a factor 11 in σ_z² with a stable coefficient, and shown on known-truth synthetic universes to collapse frequentist coverage to 0–3 per cent for the bare-Gaussian kernel while the volume-deconvolved kernel calibrates close to nominal in the clean single-host limit. The mathematics is unusually well verified: the math stage and the adversarial proof red-team independently recomputed essentially every quoted coefficient and traced every headline number to its artifact, finding no mathematical error.

Three problem classes nonetheless prevent any recommendation stronger than major revision. First, the manuscript is literally incomplete: the full-scale confirmation of the headline de-rail (all 3375 events, 38-value grid) is an explicit `[RESULT PENDING]` placeholder (realdata.tex:73), so the central real-data demonstration currently rests on a 494-event subsample and a seven-point grid whose injected truth coincides with the grid midpoint. Second, the claim scope materially exceeds the evidence in the paper's significance backbone: the abstract designates σ_z/z ~ 0.7 as "the regime probed here" while the coverage evidence lives at σ_z/z ~ 0.1–0.2 and the real-data sample is mostly near-spectroscopic; the introduction's mechanism sentence wrongly attributes σ_z² suppression to all three defects (two are zeroth order in σ_z); one supporting sentence in Section 6 ("3331 of 3355 events... numerator evaluates to zero") is demonstrably a misreading of a diagnostic log and is false as written; and the same-round proof red-team fails closed (`gaps_found`) on seven scoping/disclosure gaps across the theorem-bearing claims. Third, the literature positioning must be rebuilt: Cross-Parkin et al. 2025 — a published, uncited, known-truth photometric-redshift dark-siren bias study with a comoving-volume prior — falsifies the introduction's implicit universal, and the citation horizon ends mid-2023 in a niche with at least five directly adjacent 2025–2026 works.

Crucially, none of the required repairs replaces the central result. The rail-is-a-curable-prior-consistency-artifact conclusion, the Eddington-in-z law, the coverage-collapse demonstration, and the prior-consistency invariant all survive; the negative novelty claim survives adversarial literature search under its full three-part conjunction, and Cross-Parkin et al. corroborates rather than preempts the repair. The narrowed, honestly scoped paper is a genuine and timely methods contribution that fits MNRAS well. The recommendation is therefore major revision, not rejection.

## Panel Evidence

| Stage | Artifact | Assessment | Key blockers or concerns |
| ----- | -------- | ---------- | ------------------------ |
| Read | review/STAGE-reader.json | adequate manuscript, ceiling major_revision | [RESULT PENDING] slot (REF-R001); σ_z/z regime bookkeeping (REF-R002); internal tension at realdata.tex:54 (REF-R004) |
| Literature | review/STAGE-literature.json | positioning weak, ceiling major_revision | Uncited Cross-Parkin et al. 2025 (REF-L001); citation horizon mid-2023 (REF-L002); negative claim survives its full conjunction |
| Math | review/STAGE-math.json | strong verification, ceiling major_revision | "3331/3355" sentence false as written (REF-M001); pending confirmation (REF-M002); every recomputed number reproduced |
| Proof red-team | review/PROOF-REDTEAM.md | status gaps_found (blocks favorable recommendation) | 7 coverage gaps + 3 missing hypotheses; all wording/disclosure/scoping; no mathematical counterexample |
| Physics | review/STAGE-physics.json | core physics well-founded, ceiling major_revision | Mechanism misattribution in intro (REF-P004); regime gap sharpened (REF-P001); dt² argument adjudicated sound |
| Significance | review/STAGE-interestingness.json | conditionally MNRAS-worthy, ceiling major_revision | Regime designation never occupied (REF-S001); rail arc is prophylactic, bias law is the venue case (REF-S004) |

All five stage artifacts and the proof red-team bind the same manuscript snapshot (path and sha256 verified on disk this run) and round 1; they are mutually consistent. Stage findings were spot-checked against the manuscript: the `[RESULT PENDING]` placeholder (realdata.tex:73), the "3331 of 3355" sentence (realdata.tex:54), the mechanism sentence (introduction.tex:81–87), the universal quantifier (introduction.tex:100–104), the full-machinery coverage residuals (coverage.tex:75–77), and the h-neutrality corollary (estimators.tex, "The adopted default") were all confirmed verbatim.

## Recommendation

**MAJOR REVISION**

The core technical result survives — and is unusually well verified — but the paper cannot be published as staged. (i) A central results slot is an explicit `[RESULT PENDING]` placeholder, which alone caps the recommendation. (ii) The significance backbone (abstract, introduction, conclusions) claims a regime (σ_z/z ~ 0.7) that none of the paper's evidence occupies, misattributes the silence mechanism to all three defects, and states a literal universal ("every independently implemented dark-siren pipeline") that the paper's own four-pipeline audit contradicts; these are claim-scope failures in exactly the sections referees read first, so minor revision is not available. (iii) The literature positioning must be rebuilt around an uncited, directly relevant 2025 known-truth photometric study and the 2025–2026 adjacent literature. (iv) The same-round proof red-team reports `gaps_found`; by review policy a non-passing proof audit forecloses any favorable recommendation until its follow-ups land and it is re-run to `passed`. Against rejection: novelty survives adversarial search under the narrowed conjunction (the Eddington-in-z H0-bias law and the known-truth coverage-collapse demonstration at photometric width have no published counterpart), the physical mechanism is real and verified, venue fit is good in kind (MNRAS has direct topical precedent in Turski et al. 2023), and every required repair narrows or completes the paper without changing its central claim.

## Evaluation

### Strengths

1. **A falsifiable diagnostic chain, stated then observed.** The sign-flip prediction (up-rail → down-rail under the 1/(4π)-only fix → interior peak under prior-consistent repair) is genuine diagnostic physics, and the two-factor ablation (realdata.tex, Table 2/Fig. 2) separates kernel consistency (the de-railing factor) from the global-vs-local denominator (+0.03 residual, quantitatively consistent with the independently measured −17 per cent global tilt).
2. **Mechanism-level verification, not curve fitting.** The Eddington-in-z law was independently re-derived and recomputed by two panel stages: q(0.05) = 38.1, K(0.25) = 20.1, K_meas = 17–20 stable to ±8 per cent across a factor 11 in σ_z², with the non-perturbative regime (σ_z·q > 1 at z_g = 0.05) identified and handled by exact quadrature rather than pushed beyond validity.
3. **An independent, deterministic, released calibration harness** (`master_thesis_code.validation.pp_coverage`): no production-code imports, estimator re-derived from the written formulas, single master seed, paired per-realization seeds, regression-pinned — the calibration claim has a defensible independence argument and the harness is a community tool.
4. **Term-by-term mapping onto Gray et al. (2020)** with every deviation declared deliberate; the assembled-likelihood rearrangement was verified as an exact identity, finite at f → 1 where Gray's A19 form is 0/0.
5. **Unusual claim hygiene at the mechanics level:** per-number artifact sourcing, disclosed unfavourable items (the dt² SNR-scale defect, timeout selection, Ω_m fiducial), explicit simulated-events scope statements, and a systematics budget with dispositions.
6. **The pinned-version four-pipeline audit** (gwcosmo, CHIMERA, icarogw, DarkSirensStat) converts a private failure into field-level information and is honest in both directions (the rail is structurally excluded from gwcosmo; the bare kernel is nonetheless live there by unenforced declaration).

### Major Issues

#### Issue REF-R001: Headline confirmation is an explicit [RESULT PENDING] placeholder

**Dimension:** completeness
**Severity:** Major revision required (blocking)
**Location:** paper_a/sections/realdata.tex:73; paper_a/sections/conclusions.tex:73–77

**Description:** The full-scale confirmation of the headline de-rail — the production pipeline with the volume-deconvolved default on all 3375 seed600 events over the complete 38-value h grid — is an unfilled placeholder ("its posterior maximum is [PENDING]"; cluster jobs 5698617/5698618). All interior-peak evidence currently rests on a 494-event subsample over a seven-point grid whose resolution near the truth is ±0.03. Merges REF-M002 (math) and REF-S002 (significance).

**Impact:** The central real-data demonstration cannot be finally adjudicated; a seven-point grid with 0.73 as a grid point cannot distinguish an interior peak at 0.73 from one at 0.74–0.75. If the full-grid result is not peaked near 0.73, the de-rail narrative must be revised, not patched.

**Suggested fix:** Fill the placeholder with the completed full-grid combined-posterior result before any submission decision; re-check the abstract's de-rail sentence against it.

**Quoted claim:** "A full-scale confirmation ... is running on the compute cluster at the time of writing, and its posterior maximum is [PENDING]."

**Missing evidence:** The completed 3375-event, 38-point-grid combined posterior.

#### Issue REF-R002: The abstract designates a regime (σ_z/z ~ 0.7) that no evidence bundle in the paper occupies

**Dimension:** significance
**Severity:** Major revision required (blocking)
**Location:** paper_a/sections/abstract.tex (lines 7–9, 30–35); coverage.tex:12–13; realdata.tex:71; conclusions.tex:44–50

**Description:** The abstract sells "σ_z/z ~ 0.7 in the GLADE+ regime probed here," but the known-truth coverage calibration runs at σ_z = 0.035 with median detected z ~ 0.3 (σ_z/z ~ 0.1–0.2, the perturbative Eddington regime, σ_z·q ~ 0.2), and the real-data section itself states most GLADE+ hosts in the sample carry near-spectroscopic uncertainties. At the designated order-unity ratio the effect is non-perturbative (σ_z·q = 1.33 at z_g = 0.05, per the paper's own Appendix B) and the volume kernel's calibration there is never coverage-tested — by this paper or any other. Merges REF-P001 (physics) and REF-S001 (significance).

**Impact:** The paper's central selling point (probing, or closing, the order-unity-ratio validation gap) is not what the paper delivers; the gap it names remains open, including for its own harness. For a paper whose venue case is a field-level warning, this is the difference between a significant methods paper and an oversold one.

**Suggested fix:** Either add a coverage run in the non-perturbative regime the abstract names (hosts at z ~ 0.05–0.1, σ_z ~ 0.035, catalogue-dominated), or scope the abstract/introduction/conclusions to state explicitly which σ_z/z each evidence bundle occupies and that the order-unity regime remains untested for every pipeline including this one. Recast the headline contribution as mechanism-plus-methodology at photometric absolute width.

**Quoted claim:** "The method's known-truth validations are almost entirely spectroscopic, while the catalogues actually used are photometric: σ_z/z ∼ 0.7 in the GLADE+ regime probed here."

**Missing evidence:** A known-truth coverage test at σ_z/z ≳ 0.5 in a catalogue-information-dominated configuration.

#### Issue REF-M001: The "3331 of 3355 events" sentence is false as written (misread diagnostic log)

**Dimension:** correctness
**Severity:** Major revision required (blocking)
**Location:** paper_a/sections/realdata.tex:54

**Description:** The sentence claims "for 3331 of the 3355 events the in-catalogue completion numerator evaluates to zero at working precision (more than 5 per cent of the quadrature weight falls outside the p_det grid)". Direct inspection of the diagnostic code (bayesian_statistics.py, STAT-04 diagnostic at the commission-base tag) shows the logged "numerator=0.000" is the *fraction of quadrature weight outside the p_det grid* — zero means full grid coverage, the opposite of a vanishing numerator. The sentence contradicts the same section's statement that all 3355 events fall inside the p_det grid (realdata.tex:9), the in-catalogue weight w_G = 0.8175 (realdata.tex:10), and the catalogue-only row of Table 2 peaking at 0.73. Merges REF-R004 (reader) and REF-P003 (physics).

**Impact:** The sentence's inference — "on these data the completion machinery, not the host matches, carries most of the constraint" — is unsupported for the post-fix configuration; the paper's own numbers show the opposite (catalogue-only mean 0.737 vs full 0.740). Notably, correcting it *strengthens* the paper: the in-catalogue numerators were never zero, so "the in-catalogue information was present all along" becomes fully consistent with the artifacts, and pre-fix completion dominance is already established by the 10³–10⁵ B_num inflation without this false statistic.

**Suggested fix:** Delete or rewrite the sentence to state what the diagnostic actually measured. Recast channel dominance as a pre-fix property (peak-density inflation explains the coherent rail) versus post-fix catalogue dominance (quote w_G = 0.8175 and the catalogue-only vs full comparison). If any post-fix completion-dominance claim is retained, support it with a per-event decomposition of the log-likelihood h-dependence into catalogue and completion channels.

**Quoted claim:** "for 3331 of the 3355 events the in-catalogue completion numerator evaluates to zero at working precision"

**Missing evidence:** A per-event channel decomposition; the cited log actually measures off-grid quadrature-weight fractions.

#### Issue REF-L001: Cross-Parkin et al. 2025 is uncited and falsifies the validation-gap narrative as stated

**Dimension:** literature_context
**Severity:** Major revision required (blocking)
**Location:** paper_a/sections/introduction.tex:57–64; abstract.tex; codes.tex; conclusions.tex

**Description:** Cross-Parkin et al. 2025 (PASA, arXiv:2502.17747) is a published known-truth photometric-redshift dark-siren bias study: Gray-family estimator, uniform-in-comoving-volume prior applied to photo-z likelihoods (this paper's prior-consistent kernel), 200 realizations × 200 events, outlier models, incompleteness subsampling, unbiased H0 at σ_z up to 10⁻² (σ_z/z ≲ 0.08). It falsifies the introduction's implicit universal ("subsequent photometric-redshift studies on real data could quantify information loss but not bias, because real data come without a truth") and strains the abstract's "almost entirely spectroscopic". The gap claim survives under its full three-part conjunction — Cross-Parkin does not occupy the catalogue-dominated, order-unity-σ_z/z regime — and their volume-prior unbiasedness independently corroborates the paper's repair. Absorbs REF-S003 (significance-layer restatement).

**Impact:** The paper claims more novelty than it owns until the framing is rebuilt; because the negative universal is the significance backbone, this is a substantial reframing of abstract, introduction, codes section, and conclusions, not a citation insertion. Citing the work is strictly favourable to the paper.

**Suggested fix:** Cite Cross-Parkin et al. 2025 and rewrite the validation-gap narrative as a three-way taxonomy (spectroscopic known-truth: Gray 2020, Laghi 2021, Borghi et al. 2026; photometric known-truth at moderate σ_z/z: Borghi 2024, Cross-Parkin 2025 — both prior-consistent, both unbiased; photometric real-data: Turski 2023, Palmese 2023). Restate the surviving gap precisely and re-derive the abstract's significance sentences from the narrowed claim.

**Quoted claim:** "subsequent photometric-redshift studies on real data ... could quantify information loss but not bias, because real data come without a truth."

**Missing evidence:** Engagement with the published known-truth photometric study that contradicts the sentence as an implicit universal.

#### Issue REF-L002: Citation horizon ends mid-2023; the negative-existence claim lacks its adjacent-work map

**Dimension:** literature_context
**Severity:** Major revision required (blocking)
**Location:** paper_a/sections/codes.tex:4–9, 39–43

**Description:** At least five 2025–2026 works in the immediate niche are unexamined: the gwcosmo-team Blinded MDC I (ApJ 2025; spectral sirens only), Borghi et al. 2026 "Echoes from the dark" (A&A 706, A199; spectroscopic-only, photometric explicitly deferred — which supports the gap claim), the CHIMERA v2 code paper (Tagliazucchi et al. 2025 — uncited although the audit pins tag v2.2), Alfradique et al. 2025, and VanWyngarden et al. 2025. The literature stage verified none fills the gap.

**Impact:** A negative universal at MNRAS needs "here is the complete map and the cell is empty," not "we found nothing"; referees from this community will name these works.

**Suggested fix:** Add a compact adjacent-known-truth-studies passage to Section 8 stating, for each, why it does not occupy the catalogue-dominated σ_z/z ~ 0.7 regime; cite the audited CHIMERA v2 code paper.

**Quoted claim:** "no known-truth validation of any published pipeline exists at σ_z/z ∼ 0.7."

**Missing evidence:** Explicit positioning against the 2025–2026 known-truth studies.

#### Issue REF-P004: The introduction's mechanism sentence is physically wrong for two of the three defects

**Dimension:** correctness
**Severity:** Major revision required (blocking)
**Location:** paper_a/sections/introduction.tex:81–87 (and 100–104; abstract.tex:7–9)

**Description:** The introduction attributes the defects' invisibility to their being "second order in the redshift kernel width and therefore invisible in the spectroscopic regime where the method's validations live." That holds only for Defect 1 (bare-kernel Eddington bias, O(σ_z²)). Defect 2's peak-density over-weight 2/(σ_φ σ_θ) contains no σ_z at all, and Defect 3's −17 per cent global tilt is measured with p_det point-evaluated at catalogue redshifts, already the σ_z = 0 configuration. Their historical invisibility is a validation-configuration property (complete-catalogue mocks suppress the completion channel via f → 1; code-vs-code comparisons are blind to common modes) — which Section 8 documents correctly. Concurs with proof red-team missing hypothesis CLM-013-H-validation-configuration.

**Impact:** The misattributed σ_z² invisibility props up the hazard claim's generality: "calibrating perfectly on spectroscopic-quality validation inputs" is demonstrated only for the kernel defect in the clean single-host harness; no in-paper spectroscopic-quality calibration of the full pre-fix estimator exists, and a spectroscopic validation on an incomplete catalogue would still expose Defect 2.

**Suggested fix:** Rewrite the mechanism sentence: σ_z² suppression for the kernel defect alone; validation-configuration invisibility for Defects 2 and 3 (import the correct physics from Section 8); qualify the "calibrating perfectly" clause accordingly.

**Quoted claim:** "a superposition of normalization defects ... whose effects are second order in the redshift kernel width and therefore invisible in the spectroscopic regime where the method's validations live."

**Missing evidence:** Either a spectroscopic-quality calibration of the full pre-fix estimator (which would in fact fail on an incomplete catalogue), or the corrected configuration-based attribution.

#### Issue REF-X001: Proof red-team fails closed — remaining theorem-to-proof scoping repairs

**Dimension:** technical_soundness
**Severity:** Major revision required (blocking)
**Location:** review/PROOF-REDTEAM.md; estimators.tex (secs. 4.1, 4.5); appendix_sky_marginal.tex:95–99; framework.tex vs appendix_beta_g.tex; introduction.tex:100–104

**Description:** The same-round adversarial proof audit reports `status: gaps_found` across the seven theorem-bearing claims. No mathematical counterexample was found and every recomputed number reproduced exactly, but five conclusion clauses are narrower than claimed and three hypotheses are unstated. Items not otherwise ledgered: (a) the "prior-consistent" label for `local_ratio` satisfies only requirement (i) of the paper's own two-part definition (framework sec. 2.6) — the retained bare kernel violates requirement (ii), as estimators.tex sec. 4.5 itself concedes — and the supporting lemma "every per-galaxy constant ... cancels row by row" is false for genuinely per-galaxy factors in a ratio of sums (two-galaxy counterexample; appendix_beta_g's "ball-common factors cancel" is the correct phrasing); (b) the sky-marginal bound "valid to relative error ≲ 10⁻⁵" drops the source derivation's "sub-percent, h-smooth drift" qualifier (unbounded |ρ_θu|); (c) completeness notation is inconsistent (f(z,Ω) vs f(z,Ω,h)) and the h-dependence under the smooth-threshold rewriting is unpinned; (d) the introduction's literal universal "every independently implemented dark-siren pipeline" is contradicted by the paper's own four-pipeline audit and survives only in the conclusions' ex-ante form. The remaining red-team items are ledgered as REF-M004 (CLM-015 mass-domain clamp) and REF-M005 (CLM-019 corollary).

**Impact:** By review policy, a non-passing same-round proof red-team prevents any favorable recommendation; a theorem-bearing clause that outruns its proof is never compatible with minor revision, however local the fix.

**Suggested fix:** Execute PROOF-REDTEAM.md Required Follow-Up items 1–6 verbatim, then re-run the proof red-team to `status: passed`. The registry-level follow-up 7 (CLAIMS.json wording) belongs to the next-round Stage 1, not to the manuscript.

**Quoted claim:** e.g. "Two prior-consistent estimator repairs are constructed" (section title / registry) versus the abstract's correctly scoped "numerator–denominator prior consistency."

**Missing evidence:** The narrowed statements themselves; each gap is a wording/disclosure repair, not new analysis.

#### Issue REF-R003: "Calibrated close to nominal" is scoped only by the clean single-host limit

**Dimension:** completeness
**Severity:** Major (non-blocking)
**Location:** paper_a/sections/coverage.tex:75–77; abstract.tex:26–29; conclusions.tex:48–50

**Description:** Inside the full completion machinery, the corrected estimator's synthetic coverage is 0.40/0.54/0.82 with a residual −0.013 bias and a MAP-vs-truth slope of ~0.35 shared by all non-railing variants; the paper attributes these residuals to the synthetic's completeness/interloper modelling *without an ablation* and designates the clean single-host test "the decisive calibration verdict" by assertion. On the real data the injected truth 0.73 coincides with the grid midpoint (0.60+0.86)/2, so a shrinkage-type miscalibration would land where genuine truth recovery lands. Merges REF-M007, REF-P002, REF-S006.

**Impact:** The paper's own prescriptive conclusion — coverage tests as a field prerequisite — is a standard the adopted estimator would not pass at nominal inside the full machinery; a referee can turn the paper's standard against it. Scoped honestly, the same fact becomes evidence for how demanding the standard is.

**Suggested fix:** Scope the calibration claim to the clean single-host limit wherever it appears at abstract/conclusion level; add either a synthetic ablation showing the residuals move with the synthetic's completeness/interloper model, or one full-machinery injection at an off-centre truth in a GLADE-like high-w_G configuration; note the truth-equals-midpoint coincidence in Section 6.

**Quoted claim:** "the volume-deconvolved kernel is calibrated close to nominal."

**Missing evidence:** Full-machinery calibration evidence, or a demonstrated (not asserted) attribution of the residuals to the synthetic's modelling.

#### Issue REF-S004: The significance framing weights the self-inflicted arc over the field-live result

**Dimension:** significance
**Severity:** Major (non-blocking)
**Location:** paper_a/sections/abstract.tex:14–25; introduction.tex:100–104; conclusions.tex:36–47

**Description:** Two of the three diagnosed defects (peak-density sky factor, global-denominator tilt) are implementation errors the paper's own audit shows are structurally excluded from all four published pipelines — the dramatic rail/flip/de-rail arc is prophylactic. The genuinely field-live result — the bare-kernel Eddington bias present in gwcosmo by an unenforced declaration (−3.3 per cent on H0 at σ_z = 0.035, coverage collapse) — is quieter but is the actual venue case. codes.tex states this split correctly ("two precise halves"); the abstract and introduction lead with the rail.

**Impact:** If the rail remains the headline, a referee can fairly ask whether the core is a code post-mortem; foregrounding the transferable elements clears the MNRAS bar comfortably.

**Suggested fix:** Rebalance abstract and introduction: lead with the bias law, coverage collapse, and validation gap; present the rail arc as the exposing case study; scope the hazard sentence to the ex-ante reading the conclusions already use.

**Quoted claim:** "is a field-level hazard for every independently implemented dark-siren pipeline, not a code anecdote."

**Missing evidence:** The literal universal is contradicted by the paper's own audit; only the ex-ante form is supported.

### Minor Issues

#### REF-R005: Abstract quotes the interior peak without the grid-resolution qualifier
**Dimension:** clarity — **Location:** abstract.tex:22–25; realdata.tex:18
Peak-location agreement is demonstrated only at ±0.03 resolution on a seven-point grid containing 0.73. **Fix:** qualify, or update with the full-grid result (with REF-R001).

#### REF-R006: Negative novelty claim restated without its full three-part conjunction
**Dimension:** literature_context — **Location:** abstract.tex; conclusions.tex
**Fix:** restate "known-truth AND catalogue-information-dominated AND σ_z/z ~ 0.7" everywhere the claim appears.

#### REF-R007: CMB-frame convention statement contradicted by the Section 6 baselines
**Dimension:** clarity — **Location:** framework.tex:21–23; budget.tex:29
Committed baselines used heliocentric redshifts (+0.15 per cent measured effect). **Fix:** one-line qualification.

#### REF-L003: Boundary against Borghi 2024 / Cross-Parkin 2025 not quantified in the paper's own σ_z/z metric
**Dimension:** literature_context — **Location:** codes.tex:36
**Fix:** quantify their σ_z/z ranges and state the recovery-vs-coverage distinction.

#### REF-L004: Flagship application dated (GWTC-3); Palmese-line photometric real-data work omitted
**Dimension:** literature_context — **Location:** introduction.tex:29–41
**Fix:** cite the GWTC-4.0 cosmology paper (arXiv:2509.04348) and Palmese et al. 2023.

#### REF-M003: β_G tilt convention undefined; one endpoint misquoted
**Dimension:** correctness — **Location:** pitfall.tex:71; appendix_beta_g.tex:74–79
"−17.2 per cent" is an endpoint-difference (multiplicative −15.8 per cent); "+8.7" should be +8.5. **Fix:** define the convention once; correct the endpoint.

#### REF-M004: Eddington-in-M quadrature domain clamp undisclosed
**Dimension:** technical_soundness — **Location:** appendix_eddington_m.tex:50–61
Implementation clamps at max(M_g − 5σ_M, 10³ M_⊙), active for essentially every galaxy at σ_rel ~ 1; "width-conservative" asserted. **Fix:** disclose the clamp (as the z-channel did), bound its effect, justify or soften "width-conservative". (Proof red-team follow-up 2.)

#### REF-M005: h-neutrality corollary outruns the lemma
**Dimension:** technical_soundness — **Location:** estimators.tex, "The adopted default", third reason
h-neutrality excludes only a spurious normalization tilt; the no-manufacture conclusion rests on the Section 5 calibration plus two-estimator agreement. **Fix:** narrow per proof red-team follow-up 3. (Merges REF-P008.)

#### REF-M006: Table C1 "exact" label vs the declared w_g-outside-the-integral approximation
**Dimension:** clarity — **Location:** appendix_gray_mapping.tex:61 vs 161–168
**Fix:** qualify the N_g/D_g row status.

#### REF-P005: dt² caveat's interpretive footprint unstated
**Dimension:** significance — **Location:** realdata.tex:12–14; budget.tex
The catalogue-dominated character (w_G = 0.8175) is a pre-dt² depth artifact; a corrected-depth GLADE+ campaign would be completion-dominated. **Fix:** one sentence noting the LISA-era designation assumes LSST/Euclid-depth catalogues.

#### REF-P006: Up-rail sign is empirical, not derived
**Dimension:** completeness — **Location:** pitfall.tex:96–105; realdata.tex:47–49
**Fix:** explain physically why the inflated completion channel prefers high h, or state the up-rail direction as an empirical input to the chain.

#### REF-P007: Gaussian photo-z likelihood assumption unscoped in the regime where it matters most
**Dimension:** technical_soundness — **Location:** framework.tex:159–183; abstract.tex:30–35
At σ_z/z ~ 0.7 real photo-z errors are skewed with catastrophic outliers. **Fix:** one scope sentence separating the (kernel-agnostic) invariant from the (Gaussian-specific) quantification.

#### REF-S005: "LISA-era" framing borrows realism the sample does not have
**Dimension:** significance — **Location:** realdata.tex:12–14; conclusions.tex:57–61
**Fix:** audit every "LISA-era" occurrence to read as motivation for the methodology, not description of the analysed sample.

#### REF-R008: Submission placeholders (author, affiliation, acknowledgements TBD)
**Dimension:** presentation_quality — **Location:** main.tex:13–17, 85–86
**Fix:** fill before submission.

### Suggestions

1. **REF-R009 — Reconcile the two grids:** a one-line note relating the [0.60, 0.86] seven-point diagnostic grid to the [0.60, 0.87] 0.01-spaced full-sample grid (postmortem.tex).
2. **REF-L005 — EMRI lineage:** cite MacLeod & Hogan 2008 and Del Pozzo, Sesana & Klein 2018 in the EMRI dark-siren lineage sentence.
3. **REF-L006 — Bibliography hygiene:** prune the 9 uncited orphan entries; resolve the Babak:2023lro texkey/year inconsistency (arXiv:2108.01167 is 2021).

## Claim-Evidence Audit

| claim | claim_type | manuscript_location | direct_evidence | support_status | overclaim_severity | required_fix |
|---|---|---|---|---|---|---|
| CLM-001 rail at 0.86 | main_result | realdata.tex §6.2 | crux_results.json, verified by math stage | supported | none | — |
| CLM-002 flip to 0.60 | main_result | realdata.tex §6.2 | derail_matrix_results.json, verified | supported | minor (sign half-derived, REF-P006) | state empirical status of up-rail sign |
| CLM-003 de-rail to interior peak at truth | main_result | realdata.tex §6.2, abstract | 494-event/7-point matrix, verified; means 0.730/0.740 | partially_supported | major (grid resolution; pending confirmation) | REF-R001, REF-R005 |
| CLM-004 coverage collapse / repair | main_result | coverage.tex, abstract | clean single-host tables verified; full-machinery 0.40/0.54/0.82 | partially_supported (abstract-level) | major | REF-R003 scoping or new evidence |
| CLM-005 three-defect decomposition | method | pitfall.tex | derivations + measurements, verified | supported | minor (mechanism sentence in intro) | REF-P004 |
| CLM-006 two prior-consistent repairs | method | estimators.tex | constructions verified incl. code default | partially_supported (label) | major (proof gate) | REF-X001(a) |
| CLM-007 Eddington-in-z law | physical_interpretation | appendix B | independently recomputed to 3 s.f. by two stages | supported | none | — |
| CLM-008 sky marginal | method | appendix A | derivation verified; error bound overstated | partially_supported (bound clause) | minor–major (proof gate) | REF-X001(b) |
| CLM-009 global-tilt failure | method | appendix E | G1_beta_g_check.json verified | supported | minor (conventions) | REF-M003 |
| CLM-010 misdiagnosis postmortem | physical_interpretation | postmortem.tex | discriminating observables, verified | supported (after REF-M001 fix) | minor | REF-M001 |
| CLM-011 negative novelty claim | novelty | codes.tex, abstract | survives adversarial 2025–26 search under full conjunction | partially_supported as framed | major | REF-L001/L002, REF-R006 |
| CLM-012 four-pipeline audit | generality | codes.tex | pinned-version audit, spot-checked by literature stage | supported | none | — |
| CLM-013 field-level hazard | significance | abstract, introduction | mechanism half verified; universal + regime + "calibrating perfectly" outrun evidence | partially_supported | major (central framing) | REF-R002, REF-P004, REF-X001(d), REF-S004 |
| CLM-014 systematics budget | significance | budget.tex | gate artifacts | supported | minor | REF-R007 |
| CLM-015 Eddington-in-M | method | appendix D | identity re-derived; numbers verified | partially_supported (domain clause) | minor (proof gate) | REF-M004 |
| CLM-016 two-factor ablation | main_result | realdata.tex §6.3 | G3_ablation_cube.json verified | supported | none | — |
| CLM-017 full-grid confirmation | main_result | realdata.tex:73 | none (explicit placeholder) | unclear | blocking | REF-R001 |
| CLM-018 independent harness | method | coverage.tex §5.1 | released module + tests | supported | none | — |
| CLM-019 h-neutrality | method | estimators.tex §4.4/4.5 | lemma verified exactly; corollary overdrawn | partially_supported (corollary) | minor (proof gate) | REF-M005 |
| CLM-020 Gray mapping | method | appendix C | rearrangement re-verified symbolically | supported (notes) | minor | REF-M006, REF-X001(c) |
| CLM-021 dt² caveat | method | realdata.tex §6.1 | rescaling-equivalence argument adjudicated sound | supported | minor (footprint) | REF-P005, REF-S005 |

## Theorem-Proof Alignment Audit

Theorem-bearing set (bound by the proof red-team and Stage 3 proof_audits): CLM-006, CLM-007, CLM-008, CLM-013, CLM-015, CLM-019, CLM-020. Stage 3 coverage is complete (one audit per claim, no duplicates); the same-round PROOF-REDTEAM.md reports `gaps_found`, so alignment is not adequate for a favorable recommendation.

| claim | uncovered assumptions | uncovered parameters | alignment_status | required_fix |
|---|---|---|---|---|
| CLM-006 | prior-consistency requirement (ii) violated by local_ratio's bare kernel; cancellation lemma false as phrased | — | partially_aligned | REF-X001(a) |
| CLM-007 | K population-averaging underived (disclosed); −0.002 floor empirical | — | partially_aligned (all numerics verified) | optional strengthening only |
| CLM-008 | unbounded \|ρ_θu\| drift vs stated ≲10⁻⁵ bound | — | aligned per Stage 3; bound clause narrowed by red-team | REF-X001(b) |
| CLM-013 | validation-configuration hypothesis unstated; "calibrating perfectly" undemonstrated for full pre-fix estimator | universal quantifier fails literally | partially_aligned | REF-P004, REF-X001(d) |
| CLM-015 | M ≤ 0 domain clamp undisclosed | — | partially_aligned | REF-M004 |
| CLM-019 | corollary outruns lemma | — | partially_aligned | REF-M005 |
| CLM-020 | f h-dependence under threshold rewriting unpinned | — | partially_aligned | REF-X001(c), REF-M006 |

## Detailed Evaluation

### 1. Novelty: ADEQUATE

The Eddington-in-z σ_z² H0-bias law for dark sirens, the known-truth coverage-collapse demonstration at photometric absolute width, and the sign-flip de-rail diagnostic have no published counterpart (verified by adversarial literature search through 2026, including works past the manuscript's citation horizon). The volume kernel itself is correctly not claimed as novel. However, the novelty *framing* must be rebuilt around Cross-Parkin et al. 2025 and the 2025–2026 adjacent literature (REF-L001/L002); the claim survives only in its narrowed three-part conjunction.

### 2. Correctness: MOSTLY CORRECT

The central mathematics is sound and unusually well verified. One quantitative sentence in Section 6 is false as written (REF-M001, a misread diagnostic log — its correction strengthens the narrative), the introduction's mechanism sentence misattributes σ_z² suppression to all three defects (REF-P004), and the β_G tilt numbers mix conventions (REF-M003).

**Equations/derivations checked (by the math stage and proof red-team, adjudicated here):**

| Equation | Location | Dimensional | Limits | Status |
| -------- | -------- | ----------- | ------ | ------ |
| Assembled likelihood p_i = [β_G L_cat + B_num]/D | framework.tex eq:assembled | ok (Mpc³ sr⁻² both channels) | f→1 finite (verified) | pass |
| Eddington-in-z: δ_z = σ_z² q, Δ_h ≈ −K σ_z² | appendix B | ok | σ_z→0, low-z q≈2/z (verified) | pass (numerics reproduced to 3 s.f.) |
| Sky marginal p̄_GW(u) = (sin θ̂/4π) N(u; 1, Σ_uu) | appendix A | ok | f→1, f→0, σ_sky→0, σ_dL→0 (all verified) | pass; error-bound clause narrowed |
| h⁻³ neutrality of volume deconvolution | estimators.tex / appendix B | ok | spectroscopic limit p_g→δ (verified) | pass; corollary narrowed |
| Eddington-in-M shifted Gaussian | appendix D | ok | σ_M→0, α_g=0 (verified) | pass; domain clamp undisclosed |
| Gray eq. 9 / A14–A19 rearrangement | appendix C | ok | f→1 (verified) | pass |

**Numerical results checked:**

| Result | Claimed Value | Verified | Agreement | Status |
| ------ | ------------- | -------- | --------- | ------ |
| De-rail matrix (6 rows MAP/mean/edge mass) | Table 2 | traced to derail_matrix_results.json / crux_results.json / G3 cube | exact | pass |
| K coefficients | q(0.05)=38.1; K(0.25)=20.1; K(0.05)=569 | independently recomputed twice | 3 s.f. | pass |
| Coverage tables | 0–3% bare vs near-nominal volume; scan −0.0016/−0.0064/−0.023/−0.046 | traced to clean_pp_summary.json / NOTE | exact | pass |
| Sky over-weight | 1.6e3–1.8e5; median 1.15, mean π/2 | recomputed analytically | exact | pass |
| β_G tilt | −17.2% end-to-end; +8.7% at h=0.60 | recomputed from G1 JSON | convention mismatch; +8.5 not +8.7 | fail (REF-M003) |
| "3331/3355 zero numerator" | realdata.tex:54 | code inspection at commission-base tag | contradicted | fail (REF-M001) |
| Eddington-M impact | 2D mean 0.790→0.770; edge 0.216→0.023 | traced to G7row9 JSON | exact | pass |
| Full-grid confirmation | [RESULT PENDING] | not available | — | unfilled (REF-R001) |

### 3. Clarity: GOOD

The diagnose-predict-test arc is easy to follow, notation is defined and mostly consistent, and per-number source comments are exemplary. Deductions: the completeness notation inconsistency (f(z,Ω) vs f(z,Ω,h)), the CMB/heliocentric convention contradiction (REF-R007), and the undefined tilt convention (REF-M003).

### 4. Completeness: GAPS

The headline confirmation slot is empty (REF-R001); the designated regime is never coverage-tested (REF-R002); full-machinery calibration evidence is residual-laden and unresolved by ablation (REF-R003); the up-rail sign is empirical (REF-P006). Everything else promised is delivered, with error analysis and disclosed unfavourable items.

### 5. Significance: MEDIUM

Conditionally MNRAS-worthy. The transferable half — the bare-kernel bias live in the flagship pipeline by unenforced declaration, the coverage methodology and released harness, the prior-consistency invariant, the validation-gap map — is a genuine, timely methods advance. The dramatic rail arc is prophylactic (self-inflicted defects structurally excluded from the audited pipelines) and must not carry the headline (REF-S004). The regime designation must be honest (REF-R002) for the significance case to survive referee scrutiny.

### 6. Reproducibility: MOSTLY REPRODUCIBLE

The coverage harness is released, deterministic from a single master seed, and regression-pinned; every headline number traces to a named artifact; reproducibility-manifest.json validates. Deductions: the pending full-grid run; the undisclosed M-channel quadrature clamp (REF-M004); β_G tilt numbers not reproducible without the undefined convention (REF-M003).

### 7. Literature Context: INCOMPLETE

The pinned-version code audit is exemplary, but the citation horizon ends mid-2023: Cross-Parkin et al. 2025 (directly on-point, uncited), the GWTC-4.0 cosmology paper, the CHIMERA v2 code paper (audited yet uncited), Borghi et al. 2026, and the Palmese-line photometric work are all missing (REF-L001/L002/L004). This is the weakest dimension.

### 8. Presentation Quality: NEEDS POLISHING

MNRAS-appropriate structure and depth; figures carry complete captions with source annotations. Author/affiliation/acknowledgements placeholders remain (REF-R008); minor grid-labeling and table-labeling tensions (REF-R009, REF-M006).

### 9. Technical Soundness: MOSTLY SOUND

Methodology is appropriate and its core assumptions are stated; the dt² rescaling-equivalence argument was adjudicated physically sound. Deductions: the Gaussian photo-z assumption is unscoped in the regime where it matters most (REF-P007); the "decisive calibration verdict" designation is a choice, not a demonstration (REF-R003); several theorem-bearing clauses outrun their proofs (REF-X001).

### 10. Publishability: MAJOR REVISION

The narrowed, completed paper — placeholder filled, regime bookkeeping explicit, mechanism sentence corrected, false Section 6 sentence rewritten, literature rebuilt around the 2025–2026 record, proof red-team follow-ups landed — is a publishable MNRAS methods paper with genuine novelty and a defensible field-level warning. As staged, it cannot be sent to a journal.

## Physics Checklist

| Check | Status | Notes |
| ----- | ------ | ----- |
| Dimensional analysis | pass | Both numerator channels Mpc³ sr⁻²; 1/(4π) and h⁻³ bookkeeping verified |
| Limiting cases | pass | f→1, f→0, σ_sky→0, σ_dL→0, σ_z→0, σ_M→0, spectroscopic kernel limit all verified |
| Symmetry preservation | pass | Isotropic sky prior handled exactly (marginal vs peak-density is the paper's own point) |
| Conservation laws | unchecked | Not applicable in the usual sense; probability normalization (measure counted once) is the paper's central invariant and was audited |
| Error bars present | pass (with gaps) | Coverage/bias tables with realization counts; binomial bands on P–P; but full-machinery residuals unresolved (REF-R003) |
| Approximations justified | fail (partial) | Narrow-beam and expansion regimes quantified; Gaussian photo-z unscoped (REF-P007); M-domain clamp undisclosed (REF-M004); sky error bound overstated (REF-X001b) |
| Convergence demonstrated | pass (with note) | Deterministic quadrature with regression pinning; grid resolution ±0.03 near truth is the open item (REF-R005/R001) |
| Literature comparison | fail | Uncited directly relevant 2025–2026 known-truth studies (REF-L001/L002) |
| Reproducible | pass (with gaps) | Released seeded harness; per-number artifact trail; pending run and clamp disclosure outstanding |

---

### Actionable Items

```yaml
actionable_items:
  - id: "REF-R001"
    finding: "Full-grid de-rail confirmation is an explicit [RESULT PENDING] placeholder"
    severity: "major"
    specific_file: "paper_a/sections/realdata.tex"
    specific_change: "Fill line 73 with the completed 3375-event, 38-point-grid combined-posterior MAP from cluster jobs 5698617/5698618; revise de-rail claims if the peak is not near 0.73; update conclusions.tex:73-77"
    estimated_effort: "small"
    blocks_publication: true
  - id: "REF-R002"
    finding: "Abstract designates sigma_z/z ~ 0.7 as the regime probed; no evidence bundle occupies it"
    severity: "major"
    specific_file: "paper_a/sections/abstract.tex"
    specific_change: "State which sigma_z/z each evidence bundle occupies (coverage ~0.1-0.2; real data mostly near-spectroscopic; low-z photometric subset ~0.7) in abstract, introduction, and conclusions; either add a non-perturbative coverage run (hosts z ~ 0.05-0.1, sigma_z ~ 0.035) or present the 0.7 regime as mapped-but-open for all pipelines including this one"
    estimated_effort: "medium"
    blocks_publication: true
  - id: "REF-M001"
    finding: "'3331 of 3355 events... numerator evaluates to zero' misreads a diagnostic log and is false as written"
    severity: "major"
    specific_file: "paper_a/sections/realdata.tex"
    specific_change: "Rewrite line 54: the log value is the off-grid quadrature-weight fraction (zero = full coverage); recast channel dominance as pre-fix (10^3-10^5 B_num inflation) vs post-fix (catalogue channel dominates, w_G = 0.8175, catalogue-only mean 0.737 vs 0.740); support any retained post-fix completion-dominance claim with a per-event channel decomposition"
    estimated_effort: "small"
    blocks_publication: true
  - id: "REF-L001"
    finding: "Cross-Parkin et al. 2025 uncited; validation-gap narrative falsified as implicit universal"
    severity: "major"
    specific_file: "paper_a/sections/introduction.tex"
    specific_change: "Cite arXiv:2502.17747; rewrite introduction.tex:57-64 as a three-way taxonomy of known-truth validations; adjust the abstract's 'almost entirely spectroscopic'; restate the surviving gap with its full three-part conjunction; note their volume-prior unbiasedness corroborates the repair"
    estimated_effort: "medium"
    blocks_publication: true
  - id: "REF-L002"
    finding: "Negative-existence claim argued against a mid-2023 literature snapshot"
    severity: "major"
    specific_file: "paper_a/sections/codes.tex"
    specific_change: "Add an adjacent-known-truth-studies passage citing the gwcosmo Blinded MDC I, Borghi et al. 2026, Tagliazucchi et al. 2025 (CHIMERA v2), and Alfradique et al. 2025, stating for each why it does not occupy the catalogue-dominated sigma_z/z ~ 0.7 regime"
    estimated_effort: "medium"
    blocks_publication: true
  - id: "REF-P004"
    finding: "Mechanism sentence attributes sigma_z^2 suppression to all three defects; wrong for defects 2 and 3"
    severity: "major"
    specific_file: "paper_a/sections/introduction.tex"
    specific_change: "Rewrite lines 81-87: sigma_z^2 suppression for the kernel defect only; validation-configuration invisibility (complete-catalogue mocks, code-vs-code common modes) for the sky-factor and global-tilt defects; qualify 'calibrating perfectly on spectroscopic-quality validation inputs'"
    estimated_effort: "small"
    blocks_publication: true
  - id: "REF-X001"
    finding: "Proof red-team gaps_found: remaining theorem-scoping repairs (CLM-006 label+lemma, CLM-008 bound, CLM-020 notation, CLM-013 universal)"
    severity: "major"
    specific_file: "paper_a/sections/estimators.tex"
    specific_change: "Execute PROOF-REDTEAM.md follow-ups 1-6: restore the sub-percent-drift qualifier in appendix_sky_marginal.tex:95-99 or bound |rho_theta-u|; scope the local_ratio label to numerator-denominator prior consistency and fix the row-by-row cancellation lemma; harmonize f(z,Omega) vs f(z,Omega,h) and pin the h-dependence; replace the introduction universal with the conclusions' ex-ante form; then re-run the proof red-team to status passed"
    estimated_effort: "medium"
    blocks_publication: true
  - id: "REF-R003"
    finding: "'Calibrated close to nominal' supported only in the clean single-host limit; full-machinery coverage 0.40/0.54/0.82 unresolved"
    severity: "major"
    specific_file: "paper_a/sections/abstract.tex"
    specific_change: "Scope the abstract/conclusion calibration claim to the clean single-host limit; add a synthetic completeness/interloper ablation or one off-centre-truth full-machinery injection; note the truth-equals-grid-midpoint coincidence in Section 6"
    estimated_effort: "medium"
    blocks_publication: false
  - id: "REF-S004"
    finding: "Significance framing leads with the self-inflicted rail arc instead of the field-live bias law"
    severity: "major"
    specific_file: "paper_a/sections/abstract.tex"
    specific_change: "Rebalance abstract/introduction to lead with the bare-kernel bias law, coverage collapse, and validation gap; present the rail arc as the exposing case study; scope the hazard sentence to the ex-ante reading"
    estimated_effort: "medium"
    blocks_publication: false
  - id: "REF-R005"
    finding: "Abstract interior-peak claim lacks grid-resolution qualifier"
    severity: "minor"
    specific_file: "paper_a/sections/abstract.tex"
    specific_change: "Qualify with the seven-point-grid resolution or update with the full-grid result"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-R006"
    finding: "Negative novelty claim restated without its full three-part conjunction"
    severity: "minor"
    specific_file: "paper_a/sections/abstract.tex"
    specific_change: "Restate 'known-truth AND catalogue-information-dominated AND sigma_z/z ~ 0.7' everywhere the claim appears"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-R007"
    finding: "CMB-frame convention statement contradicted by heliocentric Section 6 baselines"
    severity: "minor"
    specific_file: "paper_a/sections/framework.tex"
    specific_change: "Qualify the convention statement with the disclosed +0.15 per cent heliocentric effect"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-L003"
    finding: "Nearest counterexample-candidates not bounded in the paper's own sigma_z/z metric"
    severity: "minor"
    specific_file: "paper_a/sections/codes.tex"
    specific_change: "Quantify Borghi 2024 / Cross-Parkin 2025 sigma_z/z ranges; state the recovery-vs-coverage distinction"
    estimated_effort: "small"
    blocks_publication: false
  - id: "REF-L004"
    finding: "Flagship application dated; photometric real-data line incomplete"
    severity: "minor"
    specific_file: "paper_a/sections/introduction.tex"
    specific_change: "Cite the GWTC-4.0 cosmology paper and Palmese et al. 2023"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-M003"
    finding: "beta_G tilt convention undefined; '+8.7' should be '+8.5'"
    severity: "minor"
    specific_file: "paper_a/sections/appendix_beta_g.tex"
    specific_change: "Define the endpoint-difference convention once; correct the endpoint; relabel or replace the raw '+93 per cent'"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-M004"
    finding: "Eddington-in-M quadrature domain clamp undisclosed"
    severity: "minor"
    specific_file: "paper_a/sections/appendix_eddington_m.tex"
    specific_change: "Disclose the M >= max(M_g - 5 sigma_M, 1e3 Msun) clamp, bound its effect on the first moment, justify or soften 'width-conservative'"
    estimated_effort: "small"
    blocks_publication: false
  - id: "REF-M005"
    finding: "h-neutrality corollary outruns the lemma"
    severity: "minor"
    specific_file: "paper_a/sections/estimators.tex"
    specific_change: "Narrow to: the deconvolution cannot inject a spurious h-tilt through its own normalization; calibration of the relocated peak is established by Section 5 and the two-estimator agreement"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-M006"
    finding: "Table C1 'exact' label vs declared w_g approximation"
    severity: "minor"
    specific_file: "paper_a/sections/appendix_gray_mapping.tex"
    specific_change: "Qualify the N_g/D_g row: 'exact up to w_g held outside the z-integral (Section C3)'"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-P005"
    finding: "dt^2 caveat's regime provenance unstated (catalogue dominance is a pre-dt^2 artifact)"
    severity: "minor"
    specific_file: "paper_a/sections/realdata.tex"
    specific_change: "Add one sentence noting the LISA-era regime designation assumes LSST/Euclid-depth catalogues"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-P006"
    finding: "Up-rail sign empirical, presented as predicted"
    severity: "minor"
    specific_file: "paper_a/sections/pitfall.tex"
    specific_change: "Explain the up-rail direction physically or state it as an empirical input to the sign-flip chain"
    estimated_effort: "small"
    blocks_publication: false
  - id: "REF-P007"
    finding: "Gaussian photo-z assumption unscoped at order-unity sigma_z/z"
    severity: "minor"
    specific_file: "paper_a/sections/framework.tex"
    specific_change: "Add one scope sentence separating the kernel-agnostic invariant from the Gaussian-specific quantification; note skew/outliers are outside the present calibration"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-S005"
    finding: "'LISA-era' framing borrows realism the testbed sample does not have"
    severity: "minor"
    specific_file: "paper_a/sections/conclusions.tex"
    specific_change: "Audit every 'LISA-era' occurrence to read as methodology motivation, not sample description"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-R008"
    finding: "Author/affiliation/acknowledgements placeholders"
    severity: "minor"
    specific_file: "paper_a/main.tex"
    specific_change: "Fill author list, affiliations, acknowledgements"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-R009"
    finding: "Two h-grids ([0.60,0.86] vs [0.60,0.87]) unreconciled"
    severity: "suggestion"
    specific_file: "paper_a/sections/postmortem.tex"
    specific_change: "One-line note reconciling the diagnostic and full-sample grids"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-L005"
    finding: "EMRI lineage omits MacLeod & Hogan 2008 and Del Pozzo et al. 2018"
    severity: "suggestion"
    specific_file: "paper_a/sections/introduction.tex"
    specific_change: "Add both citations to the EMRI dark-siren lineage sentence"
    estimated_effort: "trivial"
    blocks_publication: false
  - id: "REF-L006"
    finding: "9 orphan bib entries; Babak texkey/year inconsistency"
    severity: "suggestion"
    specific_file: "paper_a/references.bib"
    specific_change: "Prune orphans; rekey Babak:2023lro to Babak:2021mhe or record the deliberate choice"
    estimated_effort: "trivial"
    blocks_publication: false
```

### Confidence Self-Assessment

| Dimension | Confidence | Notes |
|-----------|-----------|-------|
| Novelty | MEDIUM | Rests on the literature stage's adversarial 2025–2026 search; negative-existence claims can never be fully certified |
| Correctness | HIGH | Two independent recomputation passes (math stage + proof red-team) reproduced every checked number; REF-M001 verified at code level |
| Clarity | HIGH | Direct manuscript reading |
| Completeness | HIGH | The [RESULT PENDING] slot is unambiguous |
| Significance | MEDIUM | Venue-fit judgment for MNRAS; a human editor's calibration would strengthen this |
| Reproducibility | MEDIUM | Harness verified at spec level; production-code implementation of eq:assembled not line-audited by the panel |
| Literature context | MEDIUM | 2025–2026 works verified by the literature stage; recommend a human check of the Cross-Parkin characterization before the rewrite |
| Presentation quality | HIGH | Direct inspection |
| Technical soundness | HIGH | dt² argument and approximation budget independently adjudicated |
| Publishability | MEDIUM | Conditional on the pending result landing as expected |

### Validator Record

- `gpd validate review-ledger review/REVIEW-LEDGER.json`: PASSED (exit code 0; ledger parsed and echoed with no errors).
- `gpd validate referee-decision --strict --ledger`: exit code 1, `valid: False`, `proposed_recommendation: major_revision`, `most_positive_allowed_recommendation: major_revision`. Verbatim reasons: "PROOF-REDTEAM.md must report `status: passed` for theorem-bearing review", "Claim scope must be materially narrowed before the manuscript can be reconsidered.", "Central theorem-bearing claims are missing explicit proof-audit coverage.", "Theorem statements and proofs are not yet aligned on explicit assumptions or parameters.", "Literature comparison remains too weak for anything better than major revision.", "Blocking referee issues remain open.", "Unresolved major issues remain in the referee ledger." With a `gaps_found` proof red-team the strict validator withholds clearance by design; this is the correct fail-closed state for this round, not an artifact defect. Strict clearance requires re-running the proof red-team to `passed` after the PROOF-REDTEAM follow-ups land.

---

_Reviewed: 2026-07-02T20:52:13Z_
_Reviewer: GPD referee agent (final panel adjudicator, round 1)_
_Disclaimer: This is an AI-generated mock referee report. It supplements but does not replace expert peer review._
