# F1 Literature Audit: h-Independent Bin Edges for p_det Histogram Estimator

**Date:** 2026-05-11
**Scope:** Focused audit (not full review) — does proposed fix F1 align with consensus practice in GW dark-siren / hierarchical Bayesian H0 inference?
**Time budget:** ~20 min, ~6-10 sources.

---

## (a) Question

The pipeline builds a histogram estimator of p_det(d_L; h) over an injection set. Currently bin edges are `np.linspace(0, max(d_L_target(h)) * 1.1, 61)` — they drift with the trial Hubble constant h because the SNR-rescaled injection mapping z -> d_L(z; h) is h-dependent. F1 replaces this with a fixed support `np.linspace(0, DL_GLOBAL_MAX, 61)` computed once over the full h-grid. The question: does F1 align with how the leading dark-siren / hierarchical Bayesian pipelines handle the support discretization?

## (b) Survey

Across the modern GW dark-siren / population inference literature, the dominant pattern for evaluating the selection integral β(Λ) = ∫ p_det(θ) p_pop(θ|Λ) dθ is **option (c) of the original taxonomy: a fixed Monte-Carlo injection set with per-hyperparameter reweighting** (the Farr 2019 / Tiwari 2018 / Mandel-Farr-Gair 2019 framework). The decisive feature shared by **all** modern pipelines is that the *underlying stochastic object* — the injection sample — is generated once and held fixed; the hyperparameter dependence enters only through analytic per-injection weights, not through regenerating the discretization support.

This is the strongest possible form of "h-independent support": the stochastic discretization is not just bin-edge-stable across h, it is *literally the same sample of injections* at every h. When pipelines additionally interpolate p_det onto a grid for fast posterior evaluation (e.g. gwcosmo's `RegularGridInterpolator`-based detection probability over (z, M, H0), and the ICAROGW GPU-accelerated weight grids), that grid is constructed in **detector-frame or absolute** coordinates whose definition does not depend on the trial cosmology — masses are detector-frame, distances are either redshift or detector-frame d_L drawn from the injection prior, and H0-dependence enters only through reweighting at evaluation time.

Crucially, **none of the surveyed papers describe an h-dependent histogram support like the one currently used in the audited code**. The pattern of letting bin edges drift with the trial cosmological parameter is, as far as this audit can tell, not standard practice — and the failure mode the user describes (5-25% jumps when an injection crosses a bin boundary as h shifts by 0.001) is precisely the artifact that the fixed-injection-set + reweighting paradigm was designed to avoid (Farr 2019, arXiv:1904.10879, derives the n_eff > 4N condition explicitly because such discretization noise propagates coherently into hierarchical posteriors).

The closest thing to an explicit discussion of "support stability" appears in Talbot, Thrane & Farr (2024, arXiv:2404.16930), which advocates closed-form Marcum-Q approximations precisely to bypass discretization concerns. Mandel, Farr & Gair (2019, arXiv:1809.02063) formulate the selection integral abstractly and do not prescribe a specific discretization, but their Monte-Carlo realization (their Eq. 18) sums over a *fixed* injection ensemble.

## (c) Specific Citations

| Reference | arXiv ID | Approach to p_det discretization | Bin-edge / support behavior |
|---|---|---|---|
| Mandel, Farr & Gair 2019 | 1809.02063 | MC integral over fixed injection set; reweight per Λ | Implicitly h-independent (single fixed sample) |
| Farr 2019 | 1904.10879 | Fixed injection set; analytic accuracy criterion (n_eff > 4N) | Fixed; explicitly warns about discretization noise propagating into population inference |
| Tiwari 2018 | 1712.00482 | Weighted Monte-Carlo integration of sensitive volume | Fixed injection sample, reweighted |
| Talbot, Thrane & Farr 2024 ("Quick recipes") | 2404.16930 | Closed-form Marcum-Q for p_det(θ, ξ); avoids grids | Analytic — no discretization at all |
| Gray et al. 2020 (gwcosmo MDC) | 1908.06050 | Injection-based p_det(z, M, H0) on grid in (z, M, H0); pre-computed once | Grid axes are *redshift and detector-frame mass*, not d_L(h); fixed across H0 |
| Mastrogiovanni et al. 2024 (ICAROGW) | 2306.17671 (A&A 682, A167) | Fixed N_gen injections from π_inj(θ); per-Λ reweighting on GPU | Single fixed sample, no per-h rebuild |
| Palmese et al. 2020 (DES+GW190814) | 2006.14961 | Standard hierarchical Bayes with Λ-dependent selection via fixed injections | Fixed support |
| Gray et al. 2023 (joint pop+cosmo) | 2308.02281 | Injection reweighting (Farr 2019) | Fixed support |
| Loredo 2004 (foundational) | astro-ph/0409387 | Formal Bayesian framework with truncation/selection | No grid prescription; framework only |

## (d) Verdict

**F1 aligns with consensus practice.** The proposed fix — using a fixed luminosity-distance support `DL_GLOBAL_MAX` that does not depend on the trial h — is the *minimum* implementation of a principle that every modern pipeline implements more strongly via fixed injection sets and per-hyperparameter reweighting. The current h-dependent bin-edge construction is non-standard and produces exactly the kind of artifact (coherent 5-25% jumps summing across 1473 events to ±1-3 in log p_joint) that Farr 2019 (arXiv:1904.10879) identifies as a controlling source of bias in hierarchical inference.

There is **no published justification** for using h-dependent histogram support in dark-siren H0 inference, and **no published reason to prefer it** for better resolution near small-h — the standard argument (Farr 2019, Talbot-Thrane-Farr 2024) goes the other way: instability of the discretization is far more harmful to the inference than improved nominal resolution. If small-h support coverage is a concern, the standard remedy is to extend `DL_GLOBAL_MAX` slightly beyond the worst-case h support rather than rebuild per h.

The fix is consistent with the project's existing adoption of the **Mandel-Farr-Gair selection-function-at-hypothesis principle** and clears the bar for invoking `/physics-change`.

## (e) Caveats

1. **Implementation choice, not derivation.** None of the surveyed papers (Mandel-Farr-Gair, Farr 2019, Loredo 2004) *derive* the requirement that support be h-stable. It is universally treated as an unstated implementation choice — every pipeline does it, but no one calls it out by name. The audit found no paper that explicitly states "bin edges must not depend on hyperparameters," so F1 is *consensus practice* rather than a theorem.

2. **EMRI-specific gap.** All cited papers concern stellar-mass CBC dark sirens (LIGO/Virgo band). The EMRI dark-siren literature is sparser, and this audit did not identify an EMRI-specific reference that explicitly addresses p_det grid construction. The mechanism (hierarchical Bayes + injection-based selection) is identical in structure, so transferability is well-supported, but a strict EMRI precedent is not available.

3. **Histograms vs. reweighting.** The strongest form of the consensus practice is *not* "fix the bin edges" but rather "use a fixed injection set and reweight." If the project later moves from histogram-binning to per-injection reweighting (Farr 2019 form), that would be an additional improvement beyond F1 and is what the production pipelines actually do. F1 should be regarded as a necessary intermediate step, not the final form.

4. **Verification suggestion.** The 5-25% jump magnitude is suspicious for a converged 61-bin histogram with 1473 events — it suggests the underlying injection sample may be undersized for this discretization regardless of the bin-edge fix. A Farr 2019 n_eff check would be worthwhile after F1 lands.

---

## Sources

- [Mandel, Farr & Gair 2019, arXiv:1809.02063](https://arxiv.org/abs/1809.02063)
- [Farr 2019, arXiv:1904.10879](https://arxiv.org/abs/1904.10879)
- [Tiwari 2018, arXiv:1712.00482](https://arxiv.org/abs/1712.00482)
- [Talbot, Thrane & Farr 2024 ("Quick recipes"), arXiv:2404.16930](https://arxiv.org/abs/2404.16930)
- [Gray et al. 2020 (gwcosmo MDC), arXiv:1908.06050](https://arxiv.org/abs/1908.06050)
- [Mastrogiovanni et al. 2024 (ICAROGW)](https://www.aanda.org/articles/aa/full_html/2024/02/aa47007-23/aa47007-23.html)
- [Palmese et al. 2020 (DES+GW190814), arXiv:2006.14961](https://arxiv.org/abs/2006.14961)
- [Gray et al. 2023, arXiv:2308.02281](https://arxiv.org/abs/2308.02281)
- [gwcosmo injection user guide](https://lscsoft.docs.ligo.org/gwcosmo/injections.html)
