# Literature warnings register

**Purpose.** The dark-siren literature documents its own pitfalls: validity conditions
attached to equations, "inconsistency" catalogues from code-comparison and mock-data
papers, errata, and caveat sections. This file is the **assumption register** — every
such documented pitfall we are aware of, mapped onto *our* pipeline with an explicit
status and an evidence link.

Established by research-cycle amendment **A5 — Stage L: external consult**
(`docs/RESEARCH_CYCLE.md`, amendment ledger, 2026-08-05). Motivating case: the
decisive passage for the 2026-08-05 Hitchhiker thread had been sitting quoted inside
`docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:174-179` for weeks. **Already-cited ≠
already-heeded.** This register exists so that a paper's warnings are checked once,
recorded, and re-findable — instead of being re-derived from author memory.

**Status vocabulary**

| status | meaning |
|---|---|
| `CHECKED` | we measured or argued the condition and it **holds** at the named venue(s) |
| `VIOLATED` | we measured the condition and it **fails**; the consequence is stated and linked |
| `UNDER MEASUREMENT` | an instrument is specified/pre-registered; verdict pending |
| `OPEN` | recognised as live, no instrument yet |
| `N-A` | structurally inapplicable to this pipeline, with the reason stated |
| `UNCHECKED` | we know the warning exists and have **not** looked. Say so; never leave a row blank |

**Rules**
1. Every row cites its evidence (file, ledger row, commit) **or** says `UNCHECKED`.
   A status with no evidence link is not a status.
2. Rows are added at Stage L intake (ring R0) and at `/physics-change` gate item 6
   (source-equation validity conditions registered at import, checked per venue).
3. Statuses change by measurement, not by opinion. Append the dated verdict; do not
   silently rewrite a row.
4. Venue scoping is mandatory: "holds at the idealized venue" is not "holds".

---

## arXiv:2212.08694 — Gair et al. 2023, *The Hitchhiker's Guide to Dark Siren Cosmology* (AJ)

Our per-event-independent catalogue likelihood is this paper's approximate form. The
paper states its own validity conditions in §2.3 and catalogues generator/estimator
inconsistencies in §4.2. Full statement-by-statement intake:
`results/campaign51_20260728/realistic_20260729/CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT`.

| # | warning (location) | what it requires | our status | evidence |
|---|---|---|---|---|
| H-a | **Eq. 15 perfect-z validity** — §2.3, paragraph after Eq. (30): the selection denominator's dependence on the latent field `{z_g}` breaks separability of the integrals **unless the true galaxy redshifts are perfectly known**; only then does the hierarchical likelihood reduce to Eq. 15 | perfect (δ-kernel) host redshifts, **or** the large-detection-volume / uniform-in-comoving-volume escape clause | **VIOLATED — every venue** | ledger row 95 (`.../gate_b_20260730/BIAS_HISTORY_LEDGER.md` §1); M-1 read `m1_kernel_delta_check.json`: `host_z_kernel: "volume_deconv"` pinned in iiib's own `run_metadata_0.json`, and the parent catalogue's parse-time peculiar-velocity floor (`galaxy_catalogue/handler.py:434-479`) gives 0.0% exactly-zero σ_z, median σ_z/(1+z) = **0.0330** ≈ 0.94× the GLADE photo-z scale. The perfect-z escape clause fails on the "idealized" venue too; every earlier use of the "idealized ⇒ exact z" shorthand needs kernel-level re-scoping |
| H-b | **Eq. 31 multi-event cross-terms / large-N_gal** — §2.3 leading to Eq. (31), and §3.3: the product-of-per-event-marginals form drops cross-terms, suppressed by `1/N_gal`, arising **only** with imperfect galaxy redshifts **and** a host shared by ≥2 events | large N_gal in the detection volume; no significant shared-host population | **UNDER MEASUREMENT** | cross-term instrument specified in `CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT` § "The cross-term instrument" (per-galaxy-resolved re-evaluation over the 385 overlap-involved events / 279 `d_L`-compatible shared-sky pairs, h ∈ {0.60, 0.73, 0.81, 0.86}); negligibility band to be locked at pre-registration. Author mandate 2026-08-05: measure, do not refute by convenience. RUNBOOK-8 §5 |
| H-c | **§2.3 / Eq. 30 selection-denominator separability (P1)** — the single-event statement: a detection object depending on the latent galaxy field may be factored out of the per-event integral **only** under the large-detection-volume argument. Needs **no** shared host | detection volume large enough that the galaxy-redshift average is ≈ uniform in comoving volume | **OPEN** | converges with open thread §4 item 15 — the catalogue leg's `f_k` completeness callable is **pool-fed** and applied inside a *ball-sized* volume, exactly where the large-volume premise does not hold (`BIAS_HISTORY_LEDGER.md` §4 item 15; discovered via D1's N2-null failure, ledger row 94). P1 gives item 15 a literature warrant; no instrument yet |
| H-d | **§4.2 Inconsistency 1 — z²→z⁴ double counting** — sampling hosts ∝ z² from a catalogue already distributed ∝ z² gives an event distribution ∝ z⁴; "this type of mismatch usually results in a bias towards lower values of H_0" | generator host-sampling prior and estimator host prior must not compound | **UNDER MEASUREMENT** | not looked at as a register row. Adjacent (not a substitute): the STANDARD bare-Gaussian doubly-smeared `dV_c` de-rail to 0.600 is noted as being in this direction, `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:262-266`; and the p_sample ≠ p_comp prior mismatch tracked in the inference-consistency audit. Neither is a check of this condition. 2026-08-17: the 2-channel production-calibration harness build (ledger row #120, [A3] spec) is the designated instrument for this condition; status promoted from `UNCHECKED` to `UNDER MEASUREMENT` per the H-b precedent (row above — a specified-but-not-yet-run instrument suffices for this status; verdict pending) |
| H-e | **§4.2 Inconsistency 5 — LOS overdensities / same-line-of-sight events** — multiple events drawn along a line of sight with a low-z overdensity produce a **low-H_0** peak at the overdensity redshift unless the full likelihood is used; acute where few galaxies sit at z ≲ 0.3 | either no repeated-LOS structure, or the full Eq. 31 likelihood | **UNCHECKED** | not measured. Note the direction is *opposite* to our observed high-H₀ rail, which makes it a candidate partial-cancellation term rather than a candidate rail owner — but this is reasoning, not a measurement |
| H-f | **§3.2 operating rule** — "In the limit that we have N_GW ≥ N_gal, the simplified framework cannot be used" | N_gal per detection volume ≫ N_GW | **UNCHECKED** | our candidate balls are large (multi-candidate), so this is expected to hold comfortably — but the count has not been registered per venue. Cheap to check from existing diagnostics |
| H-g | **§4.2 Inconsistency 4 — GW likelihood mismodeling, dropped σ(d_L^true) dependence** (verbatim, quote-verified; not in the 2026-08-05 intake): "Another possible source of error is treating the GW likelihood as if the standard deviation was not dependent on the true value of [d_L], although the simulations are made assuming this dependence … By dropping the overall normalization factor in the GW likelihood, one is in practice ignoring a part of the likelihood that depends on the true luminosity distance. This causes a biased dependence of the [H₀] posterior on the luminosity distance uncertainty … We find that in this case the inconsistency has the effect of biasing [H₀] towards lower values for increasing values of [the d_L uncertainty]" | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **UNCHECKED** | added at Stage L intake, thread 16's R0 sweep (`m2_residual_owner/CLAIM_M2_RESIDUAL_OWNER_20260807.md` [LIT-2]). Mapped as the concrete mechanism candidate for H-c in the M-2 matched 2D overlap residual hunt: a low-H₀ bias growing with d_L uncertainty is the documented shape of the residual selection confound the thread traced to the collinear d_L-geometry + ball-density bundle. The ratified stage-5 verdict (`BIAS_HISTORY_LEDGER.md` §5, AUTHOR RULING 2026-08-08; row #97) found this bundle confounding-absorbable to ~2/3 by a smooth, verified d_L-functional completion-leg response (A2, R²=0.88) with the remaining ~1/3 density-coupled at joint_r1 (specification-fragile) and not significant at iiib — the dissolution of thread 16's residual into this bundle is evidence *adjacent to* H-g's mechanism, not a direct measurement of the dropped-normalization condition itself; still **not** independently checked against production code |

**Addendum (2026-08-08, thread 16 stage 5/6 close-out).** H-e's note above assessed
direction only against the *high*-H₀ rail ("candidate partial-cancellation term")
and was flagged stale by the thread-16 R0 sweep (`CLAIM_M2_RESIDUAL_OWNER_20260807.md`
[LIT-1]). Update: thread 16's low-h stratum reading (the M-2 matched 2D overlap
residual, +0.02070 joint_r1 / +0.02225 iiib nats/event, **low-H₀-preferring**) is a
**direction match** for H-e's mechanism, not the earlier-assessed high-H₀ rail.
Per the ratified stage-5 verdict (`BIAS_HISTORY_LEDGER.md` §5, AUTHOR RULING
2026-08-08; row #97), the residual is attributed to DISSOLUTION in modified-H-c form
(completion-leg-carried, d_L-geometry/ball-density confounding), with H-e's
mechanism entering as component-coherent CONDITIONAL on the overlap-among-overlap
exchangeability model (qualification q1) — this status remains **UNCHECKED as an
independent, unconditional measurement of H-e** and should not be read as H-e
itself having been confirmed; see thread 16 for the full qualification.

**Absences recorded as negative results** (searched, not present in this paper — do not
attribute them to it): correlations from shared *population hyperparameters*; catalogue
*realization* (shared-noise) correlations; correlations from a shared Monte-Carlo
selection/injection-pool estimate. Source:
`CLAIM_HITCHHIKER_INDEPENDENCE_20260805.md.DRAFT` §1 "Statements that do NOT exist".

---

## Gray et al. 2020 — arXiv:1908.06050 (PRD 101, 122001), "Cosmological Inference using
Gravitational-Wave Standard Sirens: A Mock Data Challenge" (our partition-norm template)

Task L0-LIT (ledger row #105) full-text read of main text §II and the full Appendix (2020's
detailed derivation lives in the appendix, not §2). Full transcription, equation numbers, and
answers: `results/mechanism_study_20260813/L0_LIT_FULLTEXT_20260815.md` §1.

| # | warning/condition (location) | what it requires | our status | evidence |
|---|---|---|---|---|
| G20-a | **App. 2, Eq. A.10** — the equation that would carry a `\|dD_L/dz\|` Jacobian if the paper's z-marginalization needed one | none stated: no Jacobian appears anywhere in the derivation (Eqs. A.5–A.19); the GW term is used as an unnormalized likelihood function of z throughout, not a density converted from D_L | `N-A` — this paper's own math does not carry the condition at all, so it cannot warrant a Jacobian requirement on our pipeline one way or the other | L0_LIT_FULLTEXT_20260815.md §1, Eq. A.10 transcription |
| G20-b | **Footnote 3 / App. 1** — per-galaxy redshift kernel `p(zi)` | no closed form given (Gaussian only suggested in a footnote, never equationed); no truncation/renormalization discussed anywhere (`grep` for "truncat"/"renormali" returns zero hits) | `UNCHECKED` — paper supplies no convention to check against; this equation is **derived but never exercised** (paper's own MDAs run at σ_z = 0, "ignore these crucial redshift uncertainties altogether") | L0_LIT_FULLTEXT_20260815.md §1, footnote 3 quote |
| G20-c | **App. 5, Eq. A.9/A.10 denominator** — selection-normalization placement | in this paper the selection term is the *same per-galaxy-summed fraction* as the numerator (`p(DGW\|·)` in place of `p(xGW\|·)`), not a separate global log-subtracted term | our venue's `N ln α(h)` global term is a **structural departure** from this paper's shared-object design, per the task brief's description (not independently re-checked against our own code in this pass) | L0_LIT_FULLTEXT_20260815.md §1, Eq. A.9/A.10 |

**Absence recorded as a negative result:** no `|dD_L/dz|` Jacobian anywhere in the full paper text
except one unrelated detector-frame mass-Jacobian hit — do not attribute a Jacobian requirement to
this paper's derivation.

---

## Gray et al. 2023 — arXiv:2308.02281 (v2), "Joint cosmological and gravitational-wave population
inference using dark sirens and galaxy catalogues" (gwcosmo update)

Task L0-LIT (ledger row #105) full-text read of §2 in full (§2.1.1–§2.1.5, §2.2). Full
transcription, equation numbers, and answers: `results/mechanism_study_20260813/L0_LIT_FULLTEXT_20260815.md` §2.

| # | warning/condition (location) | what it requires | our status | evidence |
|---|---|---|---|---|
| G23-a | **§2.1, Eq. 2.4/2.9/2.10** — LOS-prior event-term z-marginalization | no `\|dD_L/dz\|` Jacobian anywhere in §2.1 (same likelihood-not-density usage as Gray 2020). The one explicit `∂dL/∂z` Jacobian in the whole paper is in **§2.2 Eq. 2.24**, the injection-reweighting step of the selection-effect (Pdet) Monte-Carlo estimator — a different use-case, not the event-term redshift marginalization | `N-A` for the event term — this paper's event-term math does not carry a distance-to-redshift Jacobian requirement either; **`CHECKED` that a Jacobian is used, but only in the selection/Pdet term**, structurally distinct from our venue's per-candidate event kernel | L0_LIT_FULLTEXT_20260815.md §2, Eqs. 2.4, 2.9–2.10, 2.24 |
| G23-b | **§2.1.3, after Eq. 2.18** — truncation/renormalization of the out-of-catalogue prior `p(z,M\|Λ,I)` | explicit, **conditional** escape clause (quoted verbatim in the L0 report): skipping renormalization after truncating the range is harmless **only if the identical truncated expression is used consistently in both the numerator (Eq. 2.14) and the normalizing object (Eq. 2.18) that propagates into the selection side** | `UNCHECKED` against our own code — our per-candidate truncation window (`±4σ_d, ±5σ_z`, varying per candidate's own z_obs/σ_z) is not obviously the same shared, identically-used object this condition describes; whether it satisfies Gray 2023's condition is a structural question about our selection-term (`α(h)`) code, not decided by this literature-only pass | L0_LIT_FULLTEXT_20260815.md §2, §2.1.3 quote |
| G23-c | **§2.1.4** — comoving-volume LOS-prior H₀-dependence cancellation between numerator and selection denominator | requires the **same LOS prior object** to be used to evaluate both the GW likelihood and the GW selection effect, so the H0-dependent normalization constant cancels | `UNCHECKED` against our own code — condition confirmed to exist and be load-bearing in Gray 2023's own derivation (quoted verbatim in the L0 report), but whether our venue's event term and `α(h)` selection term share the same object the way Gray's do is not checked here | L0_LIT_FULLTEXT_20260815.md §2, §2.1.4 quote |
| G23-d | **§2.1.1, Eq. 2.9** — per-galaxy redshift kernel `p(z\|ẑk) = G(z−ẑk; σ̂k)` | plain Gaussian, footnote 9 notes no strict requirement it be Gaussian; **no truncation of this specific kernel discussed** | `UNCHECKED` — matches our venue's Gaussian-kernel choice at the functional-form level; truncation convention for *this* kernel specifically (as opposed to the out-of-catalogue prior, G23-b) is not addressed by the paper either way | L0_LIT_FULLTEXT_20260815.md §2, Eq. 2.9 |
| G23-c-check | **Fused selection term (commit `2b10b8b8`) vs the G23-c cancellation condition** — `completion_mass_factor_g_sel` + `S̄_φ` numerator insertion | whether the fused selection object and the per-candidate event-term LOS prior are the *same object* (G23-c's condition, above) | `CHECKED` — **SAME-OBJECT** (row #120 Q-1 audit, 2026-08-17, orchestrator spot-verified): one survival accessor (`detection_probability_with_bh_mass_interpolated`) serves the S̄_φ table build (`:1936-1946`), the fused g_sel `_s_query` (`:4572-4580`), and the Σ⁴D denominator (`:2675-2685`); `phi_survival_table` built once (`:3394`) and passed by reference to both denominator precomputes (`:3401`,`:3413`) and read, never rebuilt, by the fused 1D integrand (`:4521`); same catalogue handler + rate weight and same `comoving_volume_element` population leg on both sides | `darksiren_emri/bayesian_inference/bayesian_statistics.py:2155`, `:4496-4531`, `:3394-3414`, `:2675-2685`; this file's G23-c row above |

**Novelty-claims verdict** (mechanism-study thread's Jacobian/truncation claims, per Stage-L
Q1/Q4): see the verdict table in `L0_LIT_FULLTEXT_20260815.md` §4 — **presented, not adjudicated**.
Summary: the "missing Jacobian is standard-and-implicit" framing from the report-level Stage-L pass
is **refuted as literally stated** (neither template paper's event term carries this Jacobian at
all, implicit or explicit); the truncation-renormalization claim is **sharpened, not settled** —
Gray 2023 supplies a named conditional escape clause our venue's per-candidate (non-shared)
truncation window does not obviously satisfy, but confirming that requires a code-level check not
performed in this literature-only pass.

**G5b staleness note (2026-08-17).** `docs/gates/G5b_chimera_icarogw_inspection.md` (2026-07-02)
is the closest prior inspection of the P1/P2/P3 numerator/selection-denominator consistency
questions that G23-b and G23-c raise against our own code (its §0 P3 definition — "whether the
numerator and the selection denominator ... use the *same* galaxy distribution model" — is the
G23-c condition in the third-party-code comparison it actually ran). It predates the selection
fusion (commit `2b10b8b8`, 2026-08-17) and the ledger rows #117–#119 verdicts that landed on top
of it by six weeks, and its P1/P2/P3 findings were computed against pre-fusion production code.
Flagged **STALE** relative to current production code. Follow-up — re-check G5b's P1/P2/P3
questions against current production code — registered **OPEN**; re-check launched 2026-08-17
(ledger row #120, Q-2).

**P3 re-check ANSWERED (2026-08-17, row #120 Q-2 audit).** Two findings: (1) G5b never actually
answered P3 for our own code — its §0 P3 definition was applied only to the three external codes;
"stale" understates it, the slot was empty. (2) Against current HEAD the answer is: numerator and
selection denominator use the **same** galaxy/population objects in both flavors (discrete
catalogue rows + rate weight; continuous `comoving_volume_element` population leg; shared
survival accessor/table — see G23-c-check row for the anchors). The fusion **removed** a
pre-existing P3-type asymmetry rather than adding one: pre-fusion, `beta_Gbar(h)` carried the
survival factor (`bayesian_statistics.py:1263-1278`) while the legacy completion numerator
(`:4452-4493`, still the `off`-cell path) carried none (implicit S≡1); rows #117–#118's fused
S̄_φ/g_sel insertions close exactly that gap. Status: **CHECKED** for P3 at commit-of-record
`2b10b8b8`+; P1/P2 re-checks remain OPEN.

---

## Essick & Fishbach (2024) — arXiv:2310.02017, "DAGnabbit!" (selection-effect consistency
in hierarchical Bayesian GW inference)

Provenance: report-level only in this repo —
`results/mechanism_study_20260813/STAGE_L_SWEEP_20260815.md` Q3. Full paper text **NOT yet read**
here; the quote below is transcribed from the report-level intake, not verified against source.

| # | warning/condition | what it requires | our status | evidence |
|---|---|---|---|---|
| EF24-a | internally-inconsistent selection-effect treatments in hierarchical Bayesian GW inference generically bias inference; named failure mode: detectability assumed independent of the observed data given the true parameters (quote **UNVERIFIED** — not read from source in this pass) | selection function must condition consistently on the same latent/observed split used by the rest of the likelihood | `UNCHECKED` | `results/mechanism_study_20260813/STAGE_L_SWEEP_20260815.md` Q3 (report-level only); mapping target: our fused `g_sel` term (`completion_mass_factor_g_sel`, commit `2b10b8b8`) — not yet checked against this condition |

---

## Talts et al. (2018), arXiv:1804.06788, and Cook, Gelman & Rubin (2006) — SBC / coverage
method's own validity conditions

Cited as the SBC/coverage **method** at `darksiren_emri/validation/calibration_gate.py:176-177`,
but never interrogated in this repo for their own stated validity conditions or failure modes.

| # | warning/condition | what it requires | our status | evidence |
|---|---|---|---|---|
| SBC-a | SBC/coverage method's own stated validity conditions and failure modes (Talts et al. 2018 §2–3; Cook, Gelman & Rubin 2006) — cited as method only, never interrogated | whether rank-statistic SBC as specified can detect a selection term that is wrong **identically** in generator and estimator (the D1-class blind spot); complements the mandatory absolute detected-count audit leg | `UNCHECKED` | `darksiren_emri/validation/calibration_gate.py:176-177` |

---

## Other sources — rows to be opened

Named here so the gaps are visible; each becomes its own section when a Stage L ring-R0
pass reads it for warnings rather than for equations.

| source | why it needs a section | status |
|---|---|---|
| Barausse 2012, arXiv:1201.5888 (M1 population) | the fiducial cosmology import; its stated regime of validity is what makes the WMAP-era constants a design choice rather than a bug (G11) | partially covered by `docs/gates/G7_systematics_budget.md` row 6 |
| Vallisneri 2008, arXiv:gr-qc/0703086 (Fisher validity) | the Fisher/Cramér-Rao high-SNR validity conditions behind our error bars | `UNCHECKED` |

---

## Mandel, Farr & Gair (2019), arXiv:1809.02063 — selection-consistency principle

Load-bearing for the Gray-convention finding and the [C-SYM] front (the coded convention's
`bayesian_statistics.py` comment cites it; the paper thread will cite it), but the repo's
working quote is a paraphrase.

| # | warning/condition | what it requires | our status | evidence |
|---|---|---|---|---|
| MFG-a | the consistency principle as quoted in `docs/derivations/fixb_pathA_phi_marginal_selection.md` §1 ("the selection normalisation must use the same population model and the same detection model as every numerator", attributed to Eqs. (5)–(7)/assumption A2) is a **repo paraphrase, not verbatim-verified** against the arXiv text | verbatim verification (section/eq. numbers) before the paper quotes or blockquotes it; until then cite as supported claim only (same treatment as the G-3 Gray gloss caveat) | `UNCHECKED` | R0 sweep 2026-08-18 ([C-SYM]/[P3] front stage 0); also confirmed: no cited paper states the data-deterministic/latent-thresholded fork or a σ_z→0 validity condition for this bias class — both are repo-internal results |

---

**Amendment log.** 2026-08-17, five rows added/updated per ledger row #120 item 4. 2026-08-18,
MFG section + row MFG-a added (Stage-L R0 exit, [C-SYM]/[P3] front stage 0; autonomous-cycle
session, flagged).
