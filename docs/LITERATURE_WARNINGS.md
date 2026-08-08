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
| H-d | **§4.2 Inconsistency 1 — z²→z⁴ double counting** — sampling hosts ∝ z² from a catalogue already distributed ∝ z² gives an event distribution ∝ z⁴; "this type of mismatch usually results in a bias towards lower values of H_0" | generator host-sampling prior and estimator host prior must not compound | **UNCHECKED** | not looked at as a register row. Adjacent (not a substitute): the STANDARD bare-Gaussian doubly-smeared `dV_c` de-rail to 0.600 is noted as being in this direction, `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:262-266`; and the p_sample ≠ p_comp prior mismatch tracked in the inference-consistency audit. Neither is a check of this condition |
| H-e | **§4.2 Inconsistency 5 — LOS overdensities / same-line-of-sight events** — multiple events drawn along a line of sight with a low-z overdensity produce a **low-H_0** peak at the overdensity redshift unless the full likelihood is used; acute where few galaxies sit at z ≲ 0.3 | either no repeated-LOS structure, or the full Eq. 31 likelihood | **UNCHECKED** | not measured. Note the direction is *opposite* to our observed high-H₀ rail, which makes it a candidate partial-cancellation term rather than a candidate rail owner — but this is reasoning, not a measurement |
| H-f | **§3.2 operating rule** — "In the limit that we have N_GW ≥ N_gal, the simplified framework cannot be used" | N_gal per detection volume ≫ N_GW | **UNCHECKED** | our candidate balls are large (multi-candidate), so this is expected to hold comfortably — but the count has not been registered per venue. Cheap to check from existing diagnostics |
| H-g | **§4.2 Inconsistency 4 — GW likelihood mismodeling, dropped σ(d_L^true) dependence** (verbatim, quote-verified; not in the 2026-08-05 intake): "Another possible source of error is treating the GW likelihood as if the standard deviation was not dependent on the true value of [d_L], although the simulations are made assuming this dependence … By dropping the overall normalization factor in the GW likelihood, one is in practice ignoring a part of the likelihood that depends on the true luminosity distance. This causes a biased dependence of the [H₀] posterior on the luminosity distance uncertainty … We find that in this case the inconsistency has the effect of biasing [H₀] towards lower values for increasing values of [the d_L uncertainty]" | the GW likelihood's normalization must retain its σ(d_L^true) dependence, not just its mean-value dependence | **measured-adjacent** | added at Stage L intake, thread 16's R0 sweep (`m2_residual_owner/CLAIM_M2_RESIDUAL_OWNER_20260807.md` [LIT-2]). Mapped as the concrete mechanism candidate for H-c in the M-2 matched 2D overlap residual hunt: a low-H₀ bias growing with d_L uncertainty is the documented shape of the residual selection confound the thread traced to the collinear d_L-geometry + ball-density bundle. The ratified stage-5 verdict (`BIAS_HISTORY_LEDGER.md` §5, AUTHOR RULING 2026-08-08; row #97) found this bundle confounding-absorbable to ~2/3 by a smooth, verified d_L-functional completion-leg response (A2, R²=0.88) with the remaining ~1/3 density-coupled at joint_r1 (specification-fragile) and not significant at iiib — the dissolution of thread 16's residual into this bundle is evidence *adjacent to* H-g's mechanism, not a direct measurement of the dropped-normalization condition itself; still **not** independently checked against production code |

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

## Other sources — rows to be opened

Named here so the gaps are visible; each becomes its own section when a Stage L ring-R0
pass reads it for warnings rather than for equations.

| source | why it needs a section | status |
|---|---|---|
| Gray et al. 2020, arXiv:1908.06050 (our partition-norm template) | its photo-z handling is an *unexercised* equation — validated at σ_z = 0 ("ignore these crucial redshift uncertainties altogether"); under flat p_det the same-kernel denominator degenerates to a constant N | `UNCHECKED` as a register section; partial note at `docs/BIAS_RESOLUTION_ATTEMPTS_REPORT.md:170-174` |
| Gray et al. 2023, arXiv:2308.02281 (GWcosmo) | states the comoving-volume LOS-prior H₀-dependence **cancels** between numerator and denominator (§2.1.4) — a validity condition on our normalisation choice | `UNCHECKED` |
| Barausse 2012, arXiv:1201.5888 (M1 population) | the fiducial cosmology import; its stated regime of validity is what makes the WMAP-era constants a design choice rather than a bug (G11) | partially covered by `docs/gates/G7_systematics_budget.md` row 6 |
| Vallisneri 2008, arXiv:gr-qc/0703086 (Fisher validity) | the Fisher/Cramér-Rao high-SNR validity conditions behind our error bars | `UNCHECKED` |
