# COMPARISON: photo-z in-catalogue dark-siren normalisation — OURS vs the literature

Status: **pre-derivation reverse-engineering.** This file extracts, method-by-method, how each
published dark-siren H0 pipeline handles the in-catalogue photo-z normalisation, across the five
scope dimensions, and contrasts them with OUR partition-norm pipeline. Equation numbers are quoted
from each paper. Where the adversarial verification pass **corrected, downgraded, or refused to
verify** a claim, it is flagged inline — unverified/refuted claims are NOT laundered as fact.

The five scope dimensions, fixed for every section:

1. **Per-galaxy z term**: is `N(z; z_g, σ_z)` used as a measurement LIKELIHOOD or as a PRIOR, and is
   it regularised by the background prior `p_bg(z) ∝ dV_c/dz` (and normalised by `Z_g`)?
2. **Selection β with photo-z** — including the decisive **num=denom same-kernel identity question**:
   does the IDENTICAL redshift density appear in both the numerator and the selection denominator?
3. **Normalisation level**: per-event frozen scalar, vs hierarchical/ensemble (shared object across
   events).
4. **Photo-z injection self-consistency**: is the sim↔inference loop matched object-for-object
   (draw true z; report `z_g = z_true + N(0,σ_z)`; source at true z; inference convolves around `z_g`)?
5. **Validated regime**: is OUR corner — `z ~ 0.05`, `σ_z/z ~ 0.7`, `p_det ~ 1` (flat), locally
   near-complete — inside the demonstrated-unbiased range?

---

## A. Gray et al. 2020 — "Mock Data Challenge" (arXiv:1908.06050, PRD 101, 122001)
**The paper our partition-norm pipeline is directly modelled on (in/out-of-catalogue split, GW
selection denominator).**

1. **Per-galaxy z term — PRIOR slot.** The catalogue redshift enters the `p(z,Ω,M,m | G,s,H0)`
   prior on the source redshift (Eq. A.2). Spectroscopic runs use it as a sum of (smoothable) delta
   functions; photo-z, *where present*, enters as a per-galaxy `p(z_i)` (Eq. A.10, footnote 3:
   "Gaussian or a more complicated distribution"). The smooth comoving-volume population prior `p(z)`
   is a SEPARATE object that survives only in the out-of-catalogue (Eq. A.19) and completeness
   (Eq. A.14) terms.
2. **Selection / same-kernel identity — YES within each window, with the decisive caveat.** Eq. A.10:
   the SAME per-galaxy kernel `p(z_i)`, the SAME galaxy sum, the SAME z-range appear in numerator
   (weighted by `p(x_GW|z_i)`) and in-catalogue denominator (weighted by `p_det(z_i)`); only the GW
   factor differs. The global completeness term `p(G|D_GW,s,H0)` (Eq. A.14) is the full-range
   `∫ p_det(z) p(z) dz` ratio — **this is exactly OUR global Option-A scalar D(h)**.
3. **Normalisation level — PER-EVENT.** Eq. (6): product of per-event likelihoods, each divided by
   its own selection denominator `p(D_GW|H0)` (Eq. 7/8). Population (mass, rate) assumed known
   exactly → **no hierarchical coupling, no shared object beyond H0.**
4. **Photo-z injection — NONE.** Every MDA injects delta-function (true) redshifts. The photo-z slot
   (Eq. A.10) is **never numerically exercised.**
5. **Validated regime — OUTSIDE ours, decisively.** Validated only at `σ_z = 0` (spectroscopic).

> **Verification flags (HIGH reliability):** all five load-bearing claims CONFIRMED verbatim.
> - **CLAIM 1 CONFIRMED** (decisive): "Our present mock data analyses ignore these crucial redshift
>   uncertainties altogether … left aside for possible future study." This paper is **NOT** the
>   working unbiased photometric method — photo-z is an unexercised equation.
> - **CLAIM 2 CONFIRMED** (with refinement): under flat `p_det`, Eq. A.10's denominator
>   `Σ_i ∫ p_det(z_i) p(z_i) dz_i → Σ_i ∫ p(z_i) dz_i = N` — the same-kernel ratio **degenerates**.
>   *Refinement:* it reduces to exactly `N` only if the rate/luminosity weights are H0-independent;
>   the Schechter luminosity weight `M(z,m,H0)` leaves a residual H0 dependence — but **not** the
>   local redshift-density-gradient term the rail needs. Conclusion unchanged.
> - **CLAIM 3 CONFIRMED:** the anti-rail mechanism is a VARYING `p_det` (Eqs. A.22–A.24, "drops from
>   1 to 0 over a range of z") plus sharp delta redshifts and a completeness gradient — none of which
>   exist in our flat-`p_det`, broad-photo-z corner. Highest-stakes (b) is an interpretive
>   extrapolation (sound, not a direct quote).

---

## B. Hitchhiker's Guide — Gair, Chen, Fishbach et al. 2023 (arXiv:2212.08694, AJ 166, 22)
**The most important method: it explicitly derives the per-event form as an approximation and gives
the exact hierarchical object.**

1. **Per-galaxy z term — fundamentally a LIKELIHOOD, repackaged as a prior in the approximation.**
   `L_red(ẑ_g|z)` (Eq. 17, Gaussian, `σ_z = 0.013(1+z)^3 ≤ 0.015` — **this is the exact scaling of
   our CLAUDE.md Known Bug 9, datamodels/galaxy.py:64**). The per-galaxy POSTERIOR is
   `p_red(z|ẑ_g) = L_red(ẑ_g|z) p_bg(z) / Z` (Eq. 16); these are averaged into `p_CBC ≈ p_cat`
   (Eq. 13). The background prior `p_bg(z) = (dV_c/dz)/∫(dV_c/dz)dz` (Eq. 32) is **H0-independent**.
2. **Selection / same-kernel identity — YES in the approximate Eq. 3, but it is only an
   APPROXIMATION.** Eq. (3): IDENTICAL `p_CBC(z)` in numerator and denominator (Hint-1 form
   confirmed). The exact object is the hierarchical Eq. (31) / one-galaxy Eq. (33): the true galaxy
   redshifts `{z_g}` are SHARED latent variables marginalised ONCE across all `N_obs` events, with
   `p_det(H0,{z_g})` (Eq. 30) a function of the true redshifts INSIDE the z-integral. Text: the
   `{z_g}`-dependence of the denominator "breaks the separability of the integrals" — **factoring it
   into a frozen scalar is precisely the approximation our pipeline makes.**
3. **Normalisation level — PER-EVENT (Eq. 3) is approximate; HIERARCHICAL (Eq. 31/33) is exact.**
4. **Photo-z injection — object-for-object self-consistent (Sec. 3.1).** Matches Hint 3 exactly:
   MICEcat true `{z_g}`; observed `ẑ_g` drawn via Eq. 17; source at host's TRUE z; inference rebuilds
   `p_red` around `ẑ_g` with `p_bg ∝ dV_c/dz` (Eq. 32). **"Inconsistency 1" (Sec. 4.2):** since
   MICEcat is already `∝ dV_c/dz`, applying an EXTRA `dV_c/dz` weight double-counts (`z² → z⁴`) and
   biases H0 **LOW** — this is Hint 4.
5. **Validated regime — OUTSIDE ours.** Photo-z model `σ_z ≤ 0.015`; controlled one-galaxy stress
   test at `δz/z = 0.3%` and `3%` only. At `3%` the per-event Eq. (3) is **biased HIGH** (Sec. 3.3,
   Fig. 8); Eq. (33) removes the bias. Our `σ_z/z ~ 0.7` is ~20–200× larger.

> **Verification flags (HIGH reliability, three refinements — no refutations):**
> - **Eq. 3 is an approximation: CONFIRMED** ("in the limit that the number density of CBCs is much
>   lower than the number density of galaxies … reduced to Eq. 3").
> - **PRIMARY validity condition CORRECTED:** the dominant licensing condition is "perfect (or
>   negligible-error) galaxy redshifts AND CBC density ≪ galaxy density," **not** chiefly the
>   large-volume / near-uniform argument. **For us the condition that breaks is the photo-z (perfect
>   redshifts), not the volume/count argument.**
> - **Eq. 3 biases HIGH at 3%: CONFIRMED** ("posterior shifted significantly to the right … large
>   bias"); full likelihood Eq. 33 gives a diagonal PP-plot.
> - **`p_det → const` degeneracy: CONFIRMED but flagged as OUR inference**, mathematically sound and
>   consistent with the paper's "the detection probability is effectively an average of the galaxy
>   redshift distribution" — not a verbatim paper claim.
> - **"Full hierarchical solves OUR regime": DOWNGRADED to extrapolation.** Eq. 31/33 is validated
>   unbiased only up to the 3% test; `σ_z/z ~ 0.7` is **not** tested. Also the Fig. 8 failure is the
>   ONE-GALAXY (sparse) limit; our regime is many-galaxy. The shared physical driver is imperfect
>   redshifts, but sparsity ≠ large-σ_z — these are distinct axes.
> - **Label uncertainty:** "Inconsistency 1" as an exact heading is UNVERIFIED (physics CONFIRMED);
>   the general-form label "Eq. 31" carries small residual uncertainty (Eq. 33 firmly attested).

---

## C. GWcosmo / LOS prior — Gray et al. 2023 (arXiv:2308.02281, JCAP 12 (2023) 023)
**Pixelated catalogue, joint population+cosmology, ensemble `−N_det` normalisation.**

1. **Per-galaxy z term — PRIOR, treated as a galaxy redshift POSTERIOR.** The catalogue sits in the
   `p(θ|Λ)` prior slot ("line-of-sight redshift prior," Eq. 2.19/2.22). Per-galaxy
   `p(z|ẑ_k) = G(z−ẑ_k; σ̂_k)` (Eq. 2.9) is "the posterior on the redshift of galaxy k" (under
   Eq. 2.8). **Footnote 10 (decisive for Hint 4):** because the catalogue value is treated as a
   posterior, **NO additional `dV_c/dz` prior is applied to in-catalogue hosts** — the `dV_c` prior
   appears ONLY in the out-of-catalogue/completeness term (Eq. 2.13).
2. **Selection / same-kernel identity — NUANCED, NOT pointwise identical.** Eq. 2.4 uses the SAME LOS
   prior symbol in numerator (per-pixel) and denominator (summed over ALL pixels). But the
   denominator's all-sky sum reduces to the smooth, GW-detection-weighted POPULATION prior; the local
   per-line-of-sight gradient lives ONLY in the numerator. **Sec. 2.1.4: the comoving-volume LOS
   prior's H0-dependence "drops out when normalised" and "cancels" between GW likelihood and
   selection** — so the denominator carries no local gradient and no net H0-tracking from the prior.
3. **Normalisation level — POPULATION/ENSEMBLE.** Single selection denominator raised to `−N_det`
   (Eq. 2.2), shared across the whole event set, joint cosmology+mass+rate inference.
4. **Photo-z injection — NO controlled object-for-object photo-z mock.** Validation is GWTC-3 BBH
   reanalysis with real heterogeneous GLADE+ photo-z (notes ~0.7% of GLADE+ have `σ_z > z`, flagged
   as artefacts) + icarogw cross-check. The self-consistent draw-true/report-noisy loop is NOT
   exercised; only the analytic `σ̂ → 0` limit is examined.
5. **Validated regime — OUTSIDE ours.** GLADE+ markedly INCOMPLETE at BBH distances (out-of-cat term
   dominates); `p_det` VARIES strongly across the z-range; `H0 = 69(+12,−7)`. No flat-`p_det`,
   `σ_z/z ~ 0.7`, `z ~ 0.05` near-complete photo-z mock.

> **Verification flags (HIGH on equations; equation NUMBERS corrected):**
> - **Same-kernel-but-denominator-washes-out: CONFIRMED**, and STRENGTHENED: Sec. 2.1.4 shows the
>   LOS-prior H0-dependence **cancels in BOTH** numerator and denominator, so "de-railing protection
>   cannot come from the prior normalisation at all" — H0 sensitivity comes entirely from the
>   `d_L(z;H0)` GW overlap with the DISCRETE fixed-z catalogue in the numerator. **Hint 1's degeneracy
>   caveat is REAL.**
> - **No `dV_c` on in-catalogue hosts: CONFIRMED**, but **CONDITIONAL** (footnote 10): valid only
>   because catalogue redshifts are treated as POSTERIORS. **If GLADE photo-z entries are actually
>   likelihoods, even GWcosmo prescribes a uniform-in-comoving-volume prior.** The "our pipeline
>   double-counts" half cannot be verified from the paper — must be checked in-repo.
> - **Equation-number REFUTATION:** the shared `−N_det` denominator is **Eq. 2.2** (not 2.3); the
>   population/rate realisation is Sec. 2.2 / 3.1 and **Eq. 4.2** (Madau-Dickinson), **not Eqs.
>   2.23–2.25**. Cosmetic — no physics claim affected. Corrected anchors: in-catalogue posterior sum
>   Eq. 2.8 + kernel Eq. 2.9 + weighted Eq. 2.10; out-of-cat `dV_c` term Eq. 2.13; assembled LOS
>   prior Eq. 2.22; footnote 10.

---

## D. Cross-Parkin, Howlett, Davis & Khetan 2025 (arXiv:2502.17747) — "redshift precision"
**The one paper that actually injects and tests PHOTOMETRIC redshifts in a per-event same-kernel
ratio (Hint-1 form in its purest realisation).**

1. **Per-galaxy z term — LIKELIHOOD + `dV_c/dz` PRIOR, combined by Bayes (Hint 4 validated).**
   `L_red(z_g|z) = N(z | σ_z(1+z)+A)` (Eq. 10, redMaGiC fit `σ_z=0.019, A=−0.013`); prior
   `p_bg(z) = (dV_c/dz)/norm` (Eq. 11); per-galaxy posterior `p_red(z|z_g)` (Eq. 9). dV_c counted
   exactly once.
2. **Selection / same-kernel identity — YES, the purest Hint-1 form.** Eq. (3): the IDENTICAL
   photo-z-broadened catalogue density `p_CBC(z)` (Eq. 8, built from Eqs. 9–10) appears in numerator
   (against `L_GW`) AND in the selection denominator (against `p_det^GW`). `p_rate` cancels (Eq. 7).
   **CRUCIAL:** their per-event same-kernel ratio de-rails ONLY because `p_det` VARIES across the
   catalogue — the threshold `d_L^thr = 1550 Mpc (z~0.29)` sits INSIDE `[0.15,0.7]`, so
   `∫ p_det(z,H0) p_cat(z) dz` is a genuine H0-dependent, density-weighted quantity.
3. **Normalisation level — PER-EVENT.** Simple product over events (Eq. 2). **NOT hierarchical** — no
   shared cross-event object beyond H0. Unbiasedness rides on varying `p_det`, not on any ensemble
   constraint.
4. **Photo-z injection — object-for-object self-consistent (Hint 3 exactly).** MICEcat true z;
   `z_g = z_true + N(0, σ_z(1+z)+A)` (Eq. 1) or real redMaGiC crossmatch (~6% outliers); source at
   TRUE z; inference convolves around `z_g`. 200 events × 200 realisations. **This is the consistent
   loop OUR pipeline violates.**
5. **Validated regime — OUTSIDE ours on three/four axes.** `z ∈ [0.15,0.7]` (never `z~0.05`);
   `σ_z/z ~ 0.014–0.07` (we are `~0.7`); `p_det` VARIES (threshold inside range). Deeper: their
   `σ_z(photo) ~ 0.01–0.02 ≲ σ_z^GW ~ 0.1z ~ 0.02–0.07`, so the **GW distance still dominates host
   localisation**; in our regime `σ_z(photo) ~ 0.035 ≈ 17× σ_z^GW`, so photo-z dominates and the GW
   can no longer localise.

> **Verification flags (HIGH reliability):**
> - **Same kernel in num AND denom: CONFIRMED** (Eq. 3 prints identical `p_CBC` in both integrals;
>   no hidden distinct `p_pop`). *Residual caveat:* verified via ar5iv HTML, model-summarised text —
>   the verbatim selection-function paragraph carries minor uncertainty (equation structure robust).
> - **Photo-z as likelihood, dV_c as prior: CONFIRMED** (Eqs. 9–12).
> - **Unbiasedness requires varying `p_det` AND sub-dominant photo-z; does NOT transfer: CONFIRMED.**
>   *Numerical correction:* within `[0.15,0.7]`, `p_det` falls `1 → ~0.5` at the `z~0.29` threshold,
>   `→ 0` only at `z_draw=0.7` (the extraction's "1→0 at z~0.29" was imprecise).
> - **`p_det~1 ⇒ ∫p_det p_cat → ∫p_cat = const` degeneracy: CONFIRMED, flagged as OUR derivation**
>   layered on their Eq. 3 (the paper is silent on the flat-`p_det` limit; it neither confirms nor
>   denies de-railing there). More precise framing: "their mechanism is structurally inapplicable
>   when `p_det` is flat, leaving our obstruction intact and pointing toward Hint 2."

---

## E. Echoes from the dark — Borghi, Moresco, Tagliazucchi, Cuomo 2025 (arXiv:2509.18243, CHIMERA)
**Incompleteness-focused; structurally Option-A (global scalar `ξ` + per-event-normalised catalogue
prior) — the same structure that rails for us.**

1. **Per-galaxy z term — BOTH.** Gaussian `N(z; z̃_g, σ̃²)` (measurement likelihood) × background
   prior `p_bkg(z) ∝ dV_c/dz × mass function`, renormalised PER GALAXY (Eq. 8) into a posterior;
   summed with weights (Eq. 7) to build the source-redshift PRIOR multiplying the GW likelihood
   (Eq. 1). `σ_z = 0.001(1+z)` — spectroscopic only.
2. **Selection / same-kernel identity — NO, decisively.** `ξ(λ) = ∫ P_det(θ,λ) p_pop(θ|λ) dθ`
   (Eq. 2), MC-estimated from a SMOOTH population over the full horizon (Eq. 3); the discrete catalog
   density `p_cat` (Eq. 7) appears ONLY in the numerator. **This is NOT Hint 1** — it is Option-A,
   our structure. De-railing comes from (a) the spectroscopically sharp posterior `Eq. 8 → δ` as
   `σ→0`, and (b) a VARYING `P_det` (BBH at `z≲1`).
3. **Normalisation level — HIERARCHICAL** (single global `ξ(λ)` divides the whole product, Eq. 1) +
   per-event-normalised catalogue prior. Structurally identical to our Option-A.
4. **Photo-z injection — self-consistent BUT SPECTROSCOPIC ONLY** (MICECATv2; `z̃_g = z + N(0,σ²)`,
   `σ_z=0.001(1+z)`). Broad photometric errors **explicitly deferred to future work.**
5. **Validated regime — OUTSIDE ours.** `σ_z/z ~ 0.001–0.01` (70–700× smaller); `z≲1`; `P_det`
   varies strongly; EMRIs/LISA never mentioned.

> **Verification flags (HIGH reliability):**
> - **User's premise REFUTED by the paper itself:** "the correct unbiased photometric form is in
>   this paper" is **false** — photometric treatment is explicitly deferred. **CONFIRMED:** Sect. 3.5
>   uses `σ_z=0.001(1+z)`; "we plan to assess the impact of photometric redshift measurements … in
>   future work."
> - **Not same-kernel (Option-A): CONFIRMED** (`p_cat` numerator-only; `ξ` smooth `p_pop`).
> - **Large-σ limit CORRECTED:** Eq. 8 as `σ→∞` does **NOT** "cancel to unity" — the flat Gaussian
>   cancels its own constant, leaving the normalised background `p_bkg ∝ n_theo dV_c/dz` (Eq. 13),
>   i.e. **the rising prior.** This makes the obstruction STRONGER, not weaker.
> - **Our regime outside on two independent axes (σ_z/z and `P_det` shape): CONFIRMED.** CHIMERA's
>   own structure is the structure that rails for us; it escapes only because both spectroscopic
>   sharpness and z-varying `P_det` hold.

---

## F. OURS — partition-norm in-catalogue likelihood (tag photoz-railing-v1, commit ee98f71)

1. **Per-galaxy z term — used as a BARE per-host density, missing `p_bg/Z_g`.** Constructed at
   `bayesian_statistics.py:1623` as `norm(loc=host_z, scale=host_z_error)`. In Gray/Hitchhiker terms
   this object is the host redshift LIKELIHOOD, but the code uses it as the redshift *weighting
   density* inside the integrands with **NO `p_bg(z)` and NO `1/Z_g`** (EQ-Ng, `:1641-1646`). The
   correct object is the regularised posterior `p_red = norm·p_bg/Z_g` (Hitchhiker Eq. 16). **This is
   Deviation 1 — the bare-Gaussian defect.** `σ_z ≈ 0.035` for GLADE flag-1 photometric hosts.
2. **Selection / same-kernel identity — NO, three DIFFERENT densities.** (1) numerator convolves the
   BARE Gaussian, no `p_bg` (`:1646`); (2) the GLOBAL in-catalogue selection POINT-EVALUATES
   `p_det(z_g)` with no convolution (`:472,:490`); (3) the outer `D(h)`/`B_num` use the smooth
   `p_bg = (1/(1+z)) dV_c/dz` (`:260,:1480`). This violates the literature same-kernel requirement
   (Gray 2020 Eq. A.10; Hitchhiker Eq. 3). **Two asymmetries:** (A) bare Gaussian missing `p_bg/Z_g`
   vs `p_bg`-weighted `D(h)` — LOAD-BEARING; (B) convolve-vs-point — a PROVEN no-op since `p_det≈1`
   over the window and `Sum_global = C·β_G` cancels (Deviation 2).
3. **Normalisation level — PER-EVENT with a single GLOBAL SCALAR denominator `D(h)`** (Option-A,
   `:701-707`). `p_i(h) = (β_G·L_cat + B_num)/D(h)` (EQ-PI, `:1501-1504`); joint posterior = product
   of per-event `p_i`. **NO hierarchical/population coherence** beyond shared scalar H0 and the shared
   global `D(h)`/`Sum_global` — the missing Hint-2 object.
4. **Photo-z injection — object-for-object INCONSISTENT (Hint-3 violation).** In-catalogue events are
   injected at the host's EXACT catalogue z (`_bridge_lib.py:351-353`; true host returned 99%, median
   `|z_cand−z_true| = 0.000`), but the catalogue LABELS that exact z with `σ_z = 0.035` and inference
   convolves `norm(z;z_g,0.035)` around it — **smears a delta**, biasing H0. The self-consistent
   draw-true/report-noisy/source-at-true setup is NOT implemented. A pure inference-side spec-z filter
   (`flag==3`) FAILS (removes the actually-injected photo-z hosts → host mismatch).
5. **Validated regime — this IS our operating regime, and it is INSIDE the FAILING range.** Cluster
   MAP 0.86 vs truth 0.73 (+0.13, upper grid edge); reproduced object-for-object by the bridge Rung G
   photo-z convolution. UNBIASED only in the spectroscopic/delta-z limit (`σ_z→0` recovers MAP
   0.725) and the synthetic spec-z forecast arm. `σ_z/z ~ 0.7`, `p_det ~ 1` flat at `z~0.05` —
   **outside every literature method's demonstrated-unbiased range** (see Sections A–E).

> **Confidence (HIGH):** every field read from production code, cross-checked against the project's
> own derivation docs and the reproduced bridge ablations. The bare-Gaussian (missing `p_bg/Z_g`)
> defect, the global point-eval selection, and the delta-sharp injection were each confirmed at exact
> file:line.

---

## Cross-cutting summary of where verification CORRECTED / DOWNGRADED claims

| Paper | Claim as extracted | Verification verdict |
|---|---|---|
| Gray 2020 | photo-z denom `→ N` under flat `p_det` | CONFIRMED, refined (residual H0-dependent luminosity weight; still not the gradient term) |
| Gray 2020 | "our regime outside their range" | CONFIRMED as sound INTERPRETIVE extrapolation, not a paper quote |
| Hitchhiker | "Eq. 3 valid in large-volume/many-galaxy limit" | CORRECTED — primary condition is *perfect redshifts AND CBC≪galaxy density*; volume is secondary |
| Hitchhiker | "full hierarchical solves OUR regime" | DOWNGRADED to extrapolation (validated only to 3%; sparse ≠ large-σ_z axis) |
| Hitchhiker | "Inconsistency 1" heading | label UNVERIFIED (physics CONFIRMED) |
| GWcosmo | shared denom = "Eq. 2.3"; pop = "Eq. 2.23–2.25" | REFUTED (cosmetic): correct anchors Eq. 2.2 / Eq. 4.2 |
| GWcosmo | "no dV_c on in-cat hosts" | CONFIRMED but CONDITIONAL on posterior interpretation (footnote 10) |
| GWcosmo | "our pipeline double-counts dV_c" | UNVERIFIABLE from paper — must check in-repo |
| 2502.17747 | "p_det 1→0 at z~0.29" | CORRECTED: 1 → ~0.5 at threshold, → 0 at z_draw=0.7 |
| 2502.17747 | "paper confirms our obstruction" | DOWNGRADED — paper is SILENT on flat-`p_det`; degeneracy is OUR derivation |
| Echoes 2509 | "correct photometric form is in this paper" | REFUTED by the paper (photometric deferred to future work) |
| Echoes 2509 | "Eq. 8 large-σ cancels to unity" | CORRECTED — reverts to rising `p_bkg ∝ dV_c/dz` (obstruction stronger) |
