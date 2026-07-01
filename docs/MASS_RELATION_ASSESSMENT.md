# Stellar-mass → BH-mass relation — assessment (provenance, scatter, validity, code)

**Scope.** Research-track assessment (paper-milestone §4) of the host stellar-mass →
central-BH-mass relation used by the with-BH-mass (2-D) H₀ inference channel. Verified by
fetching the primary literature (not from memory) in workflow `wf_689bad79-8be`. Feeds the
σ_M-axis realism of the precision forecast [`docs/SIGMA_Z_SIGMA_M_FORECAST.md`].

**Date:** 2026-06-30 · **Code:** `master_thesis_code/galaxy_catalogue/handler.py`
(`:30-33`, `:1033-1052`, applied at `:801`).

---

## 1. Provenance — confirmed EXACT: Reines & Volonteri 2015

The constants `alpha = 7.45·ln10`, `beta = 1.05`, `d_alpha = 0.08·ln10`, `d_beta = 0.11`
(`handler.py:30-33`) and the form `M_BH = exp(α + β·ln(M_*/10¹¹))` ⇔
**log₁₀(M_BH/M_⊙) = 7.45 + 1.05·log₁₀(M_*/10¹¹ M_⊙)** match **digit-for-digit** the
broad-line-AGN fit of

> **Reines & Volonteri (2015)**, *"Relations between Central Black Hole Mass and Total
> Galaxy Stellar Mass in the Local Universe"*, ApJ **813**, 82 — arXiv:1508.06274,
> **Eq. (5)**: log₁₀(M_BH) = (7.45 ± 0.08) + (1.05 ± 0.11) log₁₀(M_*/10¹¹).

- **This is the M_BH–M_*,total relation** (total galaxy stellar mass), broad-line AGN sample,
  z < 0.055, M_* ∼ 10⁸–10¹² M_⊙. **It is the correct choice** to pair with GLADE+ *total*
  stellar masses (the pipeline's input).
- **NOT McConnell & Ma 2013** (arXiv:1211.2816). MM13 shares the *slope* (β ≈ 1.05) but is an
  M_BH–M_*bulge* relation with intercept **8.46** (≈ 1 dex higher). The slope coincidence is a
  red herring; the **intercept 7.45 + total-stellar-mass definition** uniquely identify R&V15.
  (R&V15's own dynamical elliptical/bulge fit is their Eq. (6): α = 8.95, β = 1.40 — *not* used.)
- **Code encoding is correct.** Natural-log internal form (`α = 7.45·ln10`, `d_α = 0.08·ln10`)
  reproduces R&V Eq. (5) in log₁₀ exactly; the unit pivot is right (`stellar_mass` in 10¹⁰ M_⊙
  → `/10` = M_*/10¹¹). The *forward* central value is faithful to the paper.

**BibTeX** (for the paper .bib):
```bibtex
@article{2015ApJ...813...82R,
  author  = {{Reines}, Amy E. and {Volonteri}, Marta},
  title   = {{Relations between Central Black Hole Mass and Total Galaxy Stellar Mass in the Local Universe}},
  journal = {Astrophys. J.}, year = {2015}, volume = {813}, number = {2}, eid = {82},
  doi = {10.1088/0004-637X/813/2/82}, eprint = {1508.06274}, archivePrefix = {arXiv}, primaryClass = {astro-ph.GA}
}
```

## 2. Intrinsic scatter — the dominant uncertainty, currently OMITTED

R&V15 §4.1 (Kelly 2007 Bayesian regression): the rms deviation about the AGN relation is
**0.55 dex**, decomposed as a virial measurement error of 0.50 dex and a **best-fit intrinsic
scatter ε₀ = 0.24 dex** added in quadrature (√(0.50² + 0.24²) = 0.555 ✓). For predicting the
M_BH of a *new* host the appropriate spread is the **total predictive rms ≈ 0.55 dex** (floor
≥ ε₀ = 0.24 dex intrinsic).

**The code's `BH_mass_error` omits this entirely** — it propagates only the fit-parameter
uncertainties (`d_α`, `d_β`) and the input stellar-mass error. Near the pivot it yields
≈ 0.08 dex, vs the true 0.24–0.55 dex — a **3–7× under-estimate** that *dominates* the
with-BH-mass error budget. In fractional (linear) terms, with CV = √(exp((ln10·s)²) − 1):

| scatter s | linear σ_M/M |
|---|---|
| 0.08 dex (code's intercept term) | 0.19 |
| 0.24 dex (intrinsic floor) | 0.60 |
| 0.50 dex (measurement) | 1.66 |
| **0.55 dex (total predictive)** | **1.99** |

## 3. Connection to the σ_z/σ_M forecast (F5) — strengthens "no rescue"

The with-BH-mass channel only adds H₀ information when the host BH mass is known to
**σ_M ≲ 1–2%** (F5). The realistic floor from this relation is **σ_M ≈ 60% (intrinsic) to
≈ 170–200% (total scatter)** — i.e. **30–100× above the threshold**, and it is an *irreducible
astrophysical* floor (the true scatter of M_BH at fixed M_*; it does **not** shrink with better
fit parameters, larger samples, or smaller stellar-mass errors). Two consequences:

1. **The 2-D channel offers no H₀ rescue** for GLADE-like photometric hosts — the auxiliary
   BH-mass observable is broader than the signal by 1.5–2 orders of magnitude. This **confirms
   and deepens** the F5 conclusion (the realistic σ_M sits *off the top* of the F5 grid).
2. **The linear-Gaussian mass model breaks down** at 0.55 dex (a factor-~3.5 uncertainty). Both
   the production likelihood and F5's kernel are linear-Gaussian in M_z; at this scatter a
   **log-normal** (Gaussian-in-log-mass) treatment is the correct model. (F5 already flags this;
   it does not change the no-rescue verdict — at large σ_M the anchor is uninformative regardless
   of kernel shape — but it is the physically correct fix if the channel is ever used.)

## 4. Low-mass validity — usable, R&V is the right choice

R&V15 is calibrated at z < 0.055, M_* ∼ 10⁸–10¹², **M_BH ∼ 10⁵–10⁹** (well-populated
∼10⁶·⁵–10⁸; the sub-10⁶ regime anchored by a few IMBH/dwarf AGN, e.g. RGG 118 ∼ 5×10⁴,
Pox 52 ∼ 3×10⁵). EMRI hosts (M_BH ∼ 10⁵–10⁷ M_⊙) are therefore **inside-to-edge** of the
calibration — a *mild* extrapolation at the lowest masses, **not** a wild one. This is precisely
why R&V15 (AGN, total mass) is correct and the bulge/dynamical relations (MM13; R&V15 Eq. 6)
are **not** — those lie ≈ 1 dex higher and would over-predict EMRI host M_BH by ~10×.

Scatter *persists/grows* toward low mass (no flattening): Greene, Strader & Ho 2020 (ARA&A 58,
257, arXiv:1911.09678) — the relation "continues unbroken to M_BH ∼ 10⁵ M_⊙, albeit with large
scatter"; Baldassare et al. 2020 (arXiv:2006.15150). Selection bias (broad-line/dynamical samples
favour over-massive BHs at fixed M_*) may bias the low-mass anchor high and under-represent the
true scatter — both effects **deepen** the no-rescue conclusion. Reference for a low-mass-regime
relation/scatter if ever updating: **Greene+2020**.

## 5. Code findings (all `[PHYSICS]` — NOT yet implemented; proposed below)

Verified at `handler.py` (forward `:1033-1042`, inverse `:1045-1052`); the forward result feeds
`_map_stellar_masses_to_BH_masses` (`:801`) → `host_M_error` →
`bayesian_statistics.py:1774` (`sigma_gal_frac = host_M_error·(1+z)/_det_M`) → the 2-D
likelihood mass-prior width (`:1780`).

| # | severity | finding | fix |
|---|---|---|---|
| 1 | **HIGH** | **Missing intrinsic scatter** (forward + inverse). The ≈0.55 dex relation scatter — the dominant term — is absent; `BH_mass_error` is ~3–7× too tight, biasing the 2-D channel and mis-calibrating any σ_M number. | Add `(σ_int·ln10)²` inside the `sqrt` (σ_int ≈ 0.55 dex total, or ≥0.24 dex intrinsic). At this magnitude prefer a **log-normal** mass error rather than the linear-Gaussian `BH_mass·sqrt(...)`. |
| 2 | **HIGH** | **Operator-precedence bug** (forward `:1040`): `beta / stellar_mass / 10 * stellar_mass_error` parses as `(β/stellar_mass)/10·σ_*`, but d ln M_BH/d M_* = β/M_* (no `/10`). Understates the stellar-mass term by **100× in variance**. | `(beta / stellar_mass * stellar_mass_error) ** 2` — drop the spurious `/10`. |
| 3 | LOW (**dead code**) | **Inverse error term wrong** (`:1049`): has `(β·σ_MBH/M_BH)²`; d ln M_*/d ln M_BH = 1/β ⇒ correct is `(σ_MBH/(M_BH·β))²`. Off by **β⁴ ≈ 1.22**. | `(MBH_mass_error / (MBH_mass * beta)) ** 2`. |
| 4 | LOW (**dead code**) | Inverse also omits the intrinsic scatter (same as #1). | Add the term consistently if the inverse is ever used. |

**On-path check (verified by grep):** the **forward** relation IS on-path — applied in
`_map_stellar_masses_to_BH_masses` (`:802`), called from catalog init (`:208`), so #1 and #2
affect production host masses. The **inverse** `_empiric_MBH_to_M_stellar_relation` has **zero
usages in the package** → it is **dead code**, so #3/#4 are correctness-only (no result impact)
and lowest priority (fix or delete).

**Note on #2 vs #1:** once the intrinsic scatter (#1, ≈0.55 dex) is added it dominates, so #2 is
sub-dominant in the *forward* total — but it is an unambiguous bug and should be fixed.

## 6. Recommendation

1. **Cite R&V15** as the host-mass relation (it is correctly chosen and implemented at the
   central-value level). Add the BibTeX (§1).
2. **`[PHYSICS]` change (bring to the user under the protocol):** add the intrinsic/total scatter
   (≈0.55 dex) to `BH_mass_error` (and the inverse), fix the `/10` precedence bug and the inverse
   β term, and — given the ≈0.55 dex magnitude — model the host-mass error as **log-normal**, not
   linear-Gaussian, in both `handler.py` and the with-BH-mass likelihood.
3. **Paper:** state plainly that the with-BH-mass channel is *uninformative for H₀* with
   GLADE-like hosts because the M_BH–M_* relation's intrinsic scatter (≈0.55 dex ⇒ σ_M ∼ 60–200%)
   is ≫ the ~1–2% the channel needs — an irreducible astrophysical floor, independent of the
   photo-z problem. Flag the lowest-mass hosts (10⁵–10⁶) as edge/extrapolated.

## 7. Sources
- Reines & Volonteri 2015, ApJ 813, 82 (arXiv:1508.06274) — Eq. (5), §4.1, Table 3. **[source relation]**
- McConnell & Ma 2013, ApJ 764, 184 (arXiv:1211.2816) — ruled out (M_bulge, intercept 8.46).
- Greene, Strader & Ho 2020, ARA&A 58, 257 (arXiv:1911.09678) — low-mass/IMBH scatter.
- Baldassare et al. 2020, ApJL 898, L3 (arXiv:2006.15150) — dwarf-AGN low-mass extrapolation.
- Suh et al. 2020, ApJ 889, 32 (arXiv:1912.02824) — total-stellar-mass relation to z∼2.5 (if high-z hosts).
