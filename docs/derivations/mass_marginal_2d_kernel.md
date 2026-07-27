# 2D host-mass marginal kernel for real-data mode (#40 remainder)

**Status: RATIFIED (2026-07-27) — all seven [RATIFY-Mn] gates approved by
the author with the stated recommendations ("all approved with your
recommendations — implement via /physics-change"): (M1) lognormal family,
median M_g, σ_lnM = host_M_error/M_g, flat-in-lnM reference; (M2) truncate +
renormalize on [10⁴, 10⁷] = ParameterSpace.M; (M3) numerator = Option N-A
(GH-24) WITH the mandatory small-σ crossover to the analytic Gaussian
product; (M4) denominator = same kernel, GL-in-lnM, erf-sum retired on this
path; (M5) R_eff shape-only in the kernel, w_g stays point-evaluated;
(M6) "necessary, not established sufficient" — real-data 2D mode stays
CANDIDATE/OPEN pending the §3.8 branch discriminators; (M7) catalogue-mass
provenance filed in G7, model-or-bound gates real-data promotion. This
document closes the "2D-OPEN" clause of the ratified host-z kernel
derivation (`hostz_pv_photoz_kernel.md`, RATIFY-5) at the derivation level;
the 2D channel itself remains OPEN per RATIFY-M6 until the discriminators
run.**

**Scope.** Derive the per-galaxy host-BH-mass kernel for the 2D
(with-BH-mass) channel in *real-data* mode — where the catalogue mass is a
Reines & Volonteri (2015) stellar-mass proxy with ≈0.58 (1σ, natural-log)
scatter, not a measurement — and specify how it enters (i) the 2D numerator
(the analytic M_z-fraction marginal), (ii) the 2D selection denominator
(counted-once-in-M), and (iii) the joint (M, z) treatment under the ratified
*broadened* host-z kernel, including the mechanism by which z-broadening
exposes the mass-kernel defect that the mock's δ-kernel masks. The measured
driving record is the three-way A/B (2026-07-26,
`results/lcat_h_dependence_20260725/threeway_ab/THREEWAY_AB_READOUT.md`) and
EXP-45 (`results/mass_kernel_truncation_20260713/FINDINGS.md`). This is a
derivation document only: no `.py` file changes, no mode promotion, no
commit until ratified.

---

## 0. Why this is needed (measured, not hypothetical)

From the three-way per-leg A/B (seed-1000 deep venue, 3454 events, fused
7-point h-grid {0.60…0.86}; metric Σᵢ ln pᵢ(h) − Σᵢ ln pᵢ(0.73); artifacts
at commit 906284c0b, 2026-07-26):

- **Cell A** (absolute_marginal baseline, volume_deconv numerator, n̂_w/D
  normalization): 2D MAP = **0.86 RAIL**, +128.2 ln over truth.
- **Cell B** (generator_marginal normalization legs + broadened
  volume_deconv numerator): 2D MAP = **0.80 INTERIOR, +29.4 ln over truth**
  (profile −204.4, −98.2, −29.4, 0, +20.2, +29.4, +13.4 over the 7-point
  grid) — while the **1D channel in the same cell is at truth** (MAP 0.73,
  monotone fall-off, −85.4 at 0.86).
- **Cell C** (production point/point pairing): 2D MAP = 0.73 = truth,
  −735.4 ln at 0.86.
- Attribution: of the −863.6 ln total 2D movement A→C, the δ-kernel alone
  (B→C) carries −748.8 ln = **86.7%**. The 2D channel is the exception to
  the 1D story: normalization substitutions alone cure 1D but leave 2D with
  an interior HIGH tilt; **only the δ-kernel brings the 2D MAP to truth**.
  The point/point pairing is load-bearing for 2D specifically, and real-data
  mode cannot use it (hostz doc §0: the δ-kernel is generator-exact for the
  mock only).

From EXP-45 (mass-kernel truncation, 2026-07-13):

- The stored mass error is **linear** 1σ = M_g·σ_lnM with σ_lnM floored at
  √(0.553² + 0.184²) ≈ **0.58** (Reines & Volonteri 0.24 dex intrinsic
  scatter dominant); typical σ_M/M ≈ 0.6–0.8. The production 2D likelihood
  marginalises this with a **linear Gaussian** N(M; M_g, σ_M).
- At σ_M/M = 0.6: **P(M < 0) = 4.8%** of every host's kernel mass;
  **29%** of kernel mass below M_min = 10⁴ for low-mass hosts, **24%**
  above M_max = 10⁷ for high-mass hosts.
- G2d (the Eddington-in-M moment-matched shift) is validated **< 1%** in
  the mass interior (10⁵–3×10⁶) even at σ_rel = 0.6, but is **−15% (wrong
  sign)** near M_min and **+22%** near M_max — and **65% of R_eff-weighted
  EMRI hosts sit in the M_max boundary zone** (median host mass 4.55×10⁶).
- The controlled H0 toy (production linear-Gaussian+G2d vs truncated
  lognormal×R_eff; diff = production − correct): **+0.0004** at σ_z/z=0.05,
  **+0.0025±0.0006** at 0.15, **+0.0081±0.0021** at 0.30, **+0.0165±0.0055**
  at 0.50, **+0.0214±0.0078** at 0.75. Sign HIGH everywhere (~3σ); grows
  with photo-z leverage.

The two records are consistent **in sign** (2D tilts HIGH) but have **no
quantitative bridge** — the toy's per-event +0.0004…+0.021 vs the measured
cell-B MAP displacement of ≈+0.073 (Laplace fit, §3.8). This gap is the
crux of this derivation and is treated head-on in §3.8, not papered over.

## 1. Current state of the code (what already exists)

All anchors `master_thesis_code/bayesian_inference/bayesian_statistics.py`
(= bs.py) unless noted.

**Default (production) 2D path.** Per detection, a 4D Gaussian in fraction
coordinates x = (φ, θ, d_L_frac, M_z_frac) with mean [φ, θ, 1, 1]
(bs.py:2097) and CRB covariance (bs.py:2032–2059); `_det_M` is the
detector-frame measured M_z (bs.py:2006). Gaussian conditioning (Bishop
2006 Eq. 2.81–2.82; internal Eqs. 14.23–14.28) precomputes `sigma2_cond`
and `proj` (bs.py:2112–2119). The numerator mass marginal is the **analytic
Gaussian product** (bs.py:3473–3488): μ_gal_frac = M_g^eff(1+z)/M_det,
σ_gal_frac = σ_M(1+z)/M_det, mz = N(μ_cond; μ_gal_frac, σ²_cond+σ²_gal) —
an **untruncated linear-Gaussian mass kernel** (formally including M < 0).
M_g^eff carries the G2d Eddington-in-M moment-matched shift
(`eddington_shifted_host_mass`, bs.py:266–300; gated
`_use_volume_deconv and not _use_mass_trunc`, bs.py:3424–3428). The
denominator inner-M integral is the exact erf-sum against the same Gaussian
prior (`_bh_mass_denominator_inner_m_integral`, bs.py:2957–3031; Owen 1980
first-moment identities), with **constant-clamp p_det tails spanning the
full real line** — a p_det-grid clamp, not a prior truncation. Production
default `generator_marginal` collapses the z-kernel to the point path
(bs.py:3496–3511) but **retains the Gaussian mass kernel** (issue #24 note,
bs.py:3498–3500).

**The candidate kernel already exists as a non-default experimental mode.**
`mass_trunc` (EXP-45) implements the truncated lognormal × R_eff prior:
`_mass_trunc_lnM_weight` (bs.py:303–332, LN(M; M_g, σ_lnM)·R_eff(M) as a
d ln M density), `_mass_trunc_sigma_lnM` (bs.py:335–349, recovers
σ_lnM = host_M_error/M_g, floor 10⁻⁶), truncation to
[M_min, M_max] = [10⁴, 10⁷] (bs.py:175–176, mirroring
`ParameterSpace.M`, `datamodels/parameter_space.py:53–58`), per-host Z_M
(bs.py:376–394), Gauss–Hermite n=24 numerator marginal
(`_mass_trunc_mz_integral`, bs.py:397–444) and Gauss–Legendre n=64 ln M
denominator (`_mass_trunc_denominator_inner_m_integral`, bs.py:447–528).
Both legs share the ONE prior (counted-once-in-M).

Unlike the host-z case — where the broadened kernel was already the
production `volume_deconv` quadrature — the candidate mass kernel exists
**only as an underived experimental mode**. This document is its
derivation and ratification path. The question analogous to hostz Q1:

> **Q1-M — Is the `mass_trunc` kernel as implemented (truncated lognormal
> median-M_g × R_eff, GH/GL quadrature) the correct real-data 2D mass
> kernel, and is adopting it sufficient to de-rail the 2D channel?**

Candidate answers developed below: (a) YES structurally, with the measure
and truncation choices ratified in §3.1–§3.2 — then real-data 2D mode is
this kernel on the ratified real-data legs and the remaining work is
validation; (b) NO — family/measure wrong (settled in §3.1); (c) YES on
form but **NOT sufficient** — the cell-B residual has additional drivers
(§3.8 branches). The recommendation is (a) on form — with one amendment
to "as implemented": the GH numerator needs the §3.3 small-σ crossover
(RATIFY-M3), and the point×mass_trunc combination needs a guard (§3.3
note) — and **agnostic-leaning-(c) on sufficiency**, with pre-registered
discriminators.

**Mass-error provenance** (`galaxy_catalogue/handler.py`): Reines &
Volonteri (2015) constants in ln units (lines 32–43: α = 7.45·ln10,
β = 1.05, dα = 0.08·ln10, dβ = 0.11, σ_int = 0.24·ln10);
`_empiric_stellar_mass_to_BH_mass_relation` (lines 1127–1141) stores the
**linear** error host_M_error = M_g·σ_lnM at parse time (lines 896–901,
`BH_MASS_ERROR` column) — a first-order linearization of the lognormal.

**2D channel assembly caveat.** In production `generator_marginal` the
per-host D_g is diagnostic-only; the catalogue term is
L_cat_wbh = (Σ w_g N_g)/n̂_w (bs.py:2695–2699). Global modes divide by
Σ_glob_wbh (point-evaluated, isotropic p_det(d_L(z_g), M_g(1+z_g)) with
w_g = R_eff(M_g)/(1+z_g); bs.py:1392, 1411–1425). The z×M_z composition of
Σ_glob_wbh is owned by the FIX-3 §7.1 thread
(`results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md`) —
coordinated with, not duplicated by, this derivation (§3.4, §3.8 branch d).

**Harness gap.** `validation/pp_coverage.py` has **no mass channel**
(pp_coverage.py:156, 276–278); the P–P full-power harness
(`results/pp_fullpower_20260727/FULLPOWER_READOUT.md`) is 1D-only. No
existing calibration harness can currently test this kernel (§4 item 5).

## 2. Verified literature anchors (2026-07-27 scan)

| Source | Content used here | Verified |
|---|---|---|
| Reines & Volonteri 2015 (arXiv:1508.06274) | Eq. (4): log(M_BH/M☉) = α + β log(M_*/10¹¹ M☉); Eq. (5): α = 7.45±0.08, β = 1.05±0.11, **intrinsic scatter 0.24 dex** (rms 0.55 dex incl. 0.50 dex measurement error), local broad-line AGN sample | **full text (ar5iv, 2026-07-27)** — matches repo constants in handler.py |
| Babak et al. 2017 (arXiv:1703.09722) | per-MBH EMRI rate R_eff(M) [Gyr⁻¹], Eqs. (31)×(34) surrogate | repo-verified (`emri_rate.py:235–261` reference comments; G2d doc) |
| Bishop 2006 PRML Eq. 2.81–2.82 | multivariate-Gaussian conditioning (σ²_cond, proj) | standard textbook; repo-verified (internal derivation Eqs. 14.23–14.28) |
| Owen 1980, Commun. Statist. B9(4) 389–419 | Gaussian zeroth/first-moment identities (erf-sum denominator) | repo-cited (bs.py:2994–2996); standard table |
| Eddington 1913 | noisy-estimate bias correction (via G2d) | repo-verified (G2d doc) |
| Mandel, Farr & Gair 2019 (arXiv:1809.02063) | one selection factor; no numerator p_det | repo-verified (hostz doc §2) |
| Abramowitz & Stegun 25.4.46 | Gauss–Hermite identity ∫N(a;μ,σ)f(a)da = π^{-1/2}Σ w_k f(μ+√2σt_k) | standard; repo-cited (bs.py:418) |
| Gray et al. 2020 (arXiv:1908.06050) | structure only: in-catalogue numerator carries GW likelihood × host priors; selection separate | functional form only — **Eq. A.10/A.19 numbers appear in repo code comments and G2d §1 but were NOT independently re-verified (ar5iv excerpt lacks the appendix); do not cite an eq. number without opening the PDF** — same discipline as hostz doc §2 |

Truncated-lognormal partial expectations are derived inline in §3.2 (no
external citation needed; complete-the-square). No source was found (or
needed) prescribing a *linear*-Gaussian BH-mass kernel at σ_M/M ≈ 0.6; the
scatter is defined in dex by the source relation, which by itself settles
the family question (§3.1).

Consistent with all verified sources and the ratified hostz derivation:
(i) the kernel is **not h-dependent**; (ii) it enters the **per-galaxy
numerator** (and the per-galaxy D_g with the same prior); (iii) the
selection/normalization keeps the smooth population prior — the discrete
kernel is never double-applied there.

## 3. The derivation

### 3.1 Error model → kernel family and reference measure **[RATIFY-M1]**

The catalogue BH mass is not a measurement of M; it is the RV15 relation
evaluated at the (noisy) stellar mass. The relation is **fit in log space**
with Gaussian scatter in log₁₀M_BH (0.24 dex intrinsic + calibration
terms), so the honest measurement model is

    ln M | M_*  ~  N( ln M̂(M_*),  σ_lnM² ),
    σ_lnM² = (ln10)²·[σ_int² + dα² + (log₁₀(M_*/10¹¹)·dβ)²] + (β·σ_M*/M_*)²,

i.e. the catalogue M_g = exp(α + β ln(M_*/10¹¹)) is the **median**
predictor, and the stored linear error M_g·σ_lnM is a first-order
linearization only. At σ_lnM ≈ 0.58 the linearization is not benign: the
symmetric linear Gaussian N(M; M_g, M_g σ_lnM) puts 4.8% of its mass at
M < 0 and misrepresents the heavy upper tail — it is the **wrong
distribution family**, not merely a mis-set width.

**Error-budget decomposition (per-host vs coherent).** The σ_lnM²
quadrature sum mixes two statistically different things. σ_int and
β·σ_M*/M_* are genuinely per-host (independent between hosts). dα and dβ
are calibration (fit-parameter) errors: **fully coherent across the
entire catalogue**, mutually correlated (the α–β fit covariance is
dropped by the quadrature sum), and not small at EMRI-host masses — at
M_BH ≈ 10⁵ (log₁₀(M_*/10¹¹) ≈ −2.3) the dβ term alone is 0.26 dex
≈ 0.59 in ln, comparable to σ_int = 0.553. A coherent mass-scale error
does not average down over 3454 events; inflating each per-host width
is a documented **conservative width choice**, not a substitute for
tracking it as a systematic — the coherent component (dα, dβ + their
covariance) is filed in the G7 systematics budget. A second coherent
systematic, one sentence: RV15 is calibrated on local broad-line AGN
(§2), and its applicability to the mostly-inactive GLADE hosts is an
untested extrapolation — also a G7 entry, not a per-host width.

The likelihood term in M at fixed catalogue estimate is
L(M) ∝ exp(−(ln M − ln M_g)²/(2σ_lnM²)) (symmetric in ln M ↔ ln M_g; the
1/M_g Jacobian of the density in M_g is constant in M and drops). The
posterior needs a reference measure for the unmodeled catalogue mass
function (G2d §6: dn_cat/dM is not separately modelled):

- **flat-in-lnM** (π ∝ R_eff(M) dM/M): p_g(M) ∝ LN(M; M_g, σ_lnM)·R_eff(M)
  — lognormal with **median M_g**. This is the implemented `mass_trunc`
  form (bs.py:303–332, 441).
- **flat-in-M** (π ∝ R_eff(M) dM): completing the square in
  u = ln M gives a lognormal with median M_g·e^{σ_lnM²} — the median
  shifts **up by e^{0.336} ≈ 1.40** at σ_lnM = 0.58. Material, not a
  technicality.

**Recommendation: flat-in-lnM (median-M_g lognormal) — and for the
dominant term this is a DERIVATION, not a measure preference.** Under
the generative statement above, ln M | M_* ~ N(ln M̂, σ_lnM²) IS the
RV15 conditional: for the dominant intrinsic-scatter component the
median-M_g lognormal is directly p(M | M_*) from the fit — no reference
measure is being chosen at all. The flat-in-M alternative corresponds to
re-reading the fit as a likelihood plus an unmotivated abundance prior —
an inconsistent re-reading, not a competing measure. The measure
question survives only for the residual catalogue-mass-function
gradient, and there: (i) flat-in-lnM is the scale-invariant
(Jeffreys-type) measure for a scale parameter spanning 3 decades;
(ii) — stated honestly — the explicit R_eff(M) tilt models the EMRI
**rate** gradient only; the catalogue mass-function gradient dn_cat/dM
is unmodelled (G2d §6) and is SET TO ZERO by the flat-in-lnM choice,
whereas flat-in-M would set it to an unmotivated +1 per e-fold; the
residual genuine assumption is that the catalogue host population's
mass function does not differ materially from the RV15 calibration
sample's. Consequence (not a ground): the choice matches the
implemented `mass_trunc` weight, so ratification requires no
kernel-shape change. **[RATIFY-M1: kernel family = lognormal, median
M_g, width σ_lnM = host_M_error/M_g (floor 10⁻⁶); reference measure
flat-in-lnM (equivalently: the RV15 conditional plus dn_cat/dM ≡ 0,
documented). Recommendation: adopt.]**

Anti-tuning note (feedback memory, first-principles rule): both the family
and the measure are fixed by the source relation and an invariance
argument — nothing here is anchored to the observed tilt direction.

### 3.2 The kernel, its normalization, truncation, and the exact relation to G2d **[RATIFY-M2]**

The per-galaxy real-data mass kernel is

    p_g(M) = 1[M_min ≤ M ≤ M_max] · LN(M; M_g, σ_lnM) · R_eff(M) / Z_M ,   (M1)

    LN(M; M_g, σ_lnM) = exp(−(ln M − ln M_g)²/(2σ_lnM²)) / (M σ_lnM √(2π)) ,

    Z_M = ∫_{M_min}^{M_max} LN(M; M_g, σ_lnM) · R_eff(M) dM .              (M2)

R_eff(M) = C_NORM·κ(M)·Γ(M)·R₀(M) [Gyr⁻¹] (`emri_rate.py:235–261`, Babak
et al. 2017 Eqs. 31×34). Because Z_M divides it out, **only the shape of
R_eff across the ±few-σ_lnM scatter range enters the kernel**; its
magnitude R_eff(M_g) lives elsewhere (sum-level w_g, §3.4) — this is what
makes the R_eff appearance here non-double-counting.

**Truncation bounds.** [M_min, M_max] = [10⁴, 10⁷] M☉ = the EMRI
population support (`ParameterSpace.M`, parameter_space.py:53–58; mirrored
at bs.py:175–176 and asserted against ParameterSpace in kernel tests).
The ground, stated for real data where there is no injector: this is the
**astrophysical population-support assumption** — the Babak et al. 2017
M1 EMRI population (whose mass range `ParameterSpace.M` mirrors) assigns
zero EMRI rate outside these bounds, so a host whose true central mass
lay outside could not host a detected EMRI; the truncation is the
population prior's support, not a numerical convenience.
(Mock-consistency remark, secondary: in the mock this is also exactly
the generator-truth injection range.) It is distinct from (i) the p_det injection-grid M_z clamp
(grid coverage; stays as is) and (ii) the draw-side
`galaxy_catalogue/handler.py:28` M_max = 10⁶ constant (catalogue/draw
machinery; a one-line consistency verification item in §4, NOT changed
here — note the measured median EMRI host mass 4.55×10⁶ exceeds it, so it
evidently does not cap the mock host population). **[RATIFY-M2: truncate
and renormalize on [10⁴, 10⁷] = ParameterSpace.M bounds. Recommendation:
adopt.]**

**Truncated-lognormal moments (for tests and the §3.3 option analysis).**
With R_eff frozen (const) and A = ln(M_min/M_g)/σ_lnM,
B = ln(M_max/M_g)/σ_lnM, completing the square in u = ln M:

    P_in = Φ(B) − Φ(A) ,
    E[M | in] = M_g · e^{σ_lnM²/2} · [Φ(B − σ_lnM) − Φ(A − σ_lnM)] / P_in . (M3)

With R_eff included, moments are by quadrature (the same GL-in-lnM nodes
as Z_M). Dimension check: (M3) carries [M☉] ✓.

**Exact sense in which (M1) subsumes G2d — stated precisely, because the
naive "interior unchanged" phrasing is not exactly achievable.** In the
interior (|ln(M_bound/M_g)| ≫ σ_lnM) with a locally log-linear
ln R_eff = const + α_g ln M, completing the square gives
p_g = LN(M; M_g e^{α_g σ_lnM²}, σ_lnM) and posterior mean

    E[M] = M_g · e^{(α_g + 1/2)σ_lnM²}  ≈  M_g (1 + (α_g + ½)σ_lnM²) .     (M4)

G2d's exact-moment helper on the *Gaussian* family gives
M_g(1 + α_g σ_rel²) + O(σ⁴) with σ_rel = σ_lnM numerically (linear error
= M_g σ_lnM). So:

- the **R_eff rate-tilt** — G2d's entire content — is reproduced inside
  (M1) exactly (the α_g σ² term), and G2d's separate point shift is NOT
  applied on this path (already the `mass_trunc` gating, bs.py:3424–3428):
  counted once;
- the **additional +½σ_lnM² term is the family (log→linear convexity)
  correction** the linear Gaussian cannot represent: e^{σ²/2} ≈ **+18%**
  in the interior effective mass at σ_lnM = 0.58. It is derived, intended,
  and pre-registered (§3.9 P1) — *not* a regression. The EXP-45 §2
  interior agreement "<1%" was measured within the Gaussian family
  (G2d vs exact truncated Gaussian×R_eff) and remains valid for the
  Gaussian-family modes; it does not contradict (M4).
- FINDINGS.md §5's constraint "interior unchanged" is hereby **refined**:
  interior R_eff-tilt behavior unchanged; interior family correction
  e^{σ²/2} intentional and bounded; boundary behavior (the −15%/+22% G2d
  failures) replaced by the exact truncated posterior. As σ_lnM → 0 both
  **exact** kernels → δ(M − M_g) and all differences vanish at O(σ²), so
  the spec-mass F5 regime is untouched **at the level of the exact
  integrals**. The implemented GH numerator does NOT realize this limit
  on its own (it has a validity floor, §3.3): the C0-continuity minimum
  bar (feedback memory) is met by the exact kernel plus the **mandatory
  small-σ crossover** specified in §3.3/RATIFY-M3, not by the GH
  quadrature alone.

### 3.3 Numerator: how (M1) replaces the analytic Gaussian product **[RATIFY-M3]**

The 2D in-catalogue numerator is (hostz §3.2 structure + mass marginal;
no p_det in the numerator, MFG 2019):

    N_g^{2D} = ∫ dz  p_g(z) ∫ dM  p(x_GW | z, Ω_g, M(1+z), h) · p_g(M) ,    (M5)

with p_g(z) the ratified §3.1-hostz kernel × volume weight / Z_g. Using the
Gaussian conditioning of the 4D CRB likelihood (Bishop 2.81–2.82; internal
14.23–14.28), the inner integral in the fraction coordinate
a = M(1+z)/M_det is

    mz(z) = ∫ N(a; μ_cond(z), σ_cond) · p_a(a | z) da ,
    p_a(a | z) = p_g(M = a·M_det/(1+z)) · M_det/(1+z)   (pushforward, |dM/da|). (M6)

For a Gaussian p_g this is the current closed form (bs.py:3481–3488). For
(M1) there is no closed form. Two candidate schemes:

**Option N-A — per-galaxy quadrature on the GW peak (implemented).**
Substitute a = μ_cond + √2 σ_cond t (A&S 25.4.46): Gauss–Hermite nodes
land ON the narrow GW M_z peak, so the narrow GW factor is never aliased
over the 3-decade mass window (`_mass_trunc_mz_integral`, bs.py:397–444;
n = 24, truncation mask applied at the nodes, exact pushforward).
**Validity condition (stated, not optional):** the scheme is exact to GH
order only while the prior is at least as wide as the GH node coverage —
σ_lnM·a_gal ≳ σ_cond with a_gal = M_g(1+z)/M_det. Below that the roles
invert: the PRIOR becomes the spike and falls between the GW-centred
nodes, and GH-24 aliases it (measured against brute-force quadrature at
σ_cond = 10⁻²: exact at σ_lnM = 5×10⁻³; wrong by O(1) at 10⁻³; returns
exactly 0 at the 10⁻⁶ floor, vs the finite analytic Gaussian-product
limit). The production catalogue (σ_lnM ≥ 0.58 ≫ σ_cond ~ 10⁻²) sits
deep inside the validity region; an F5-style sweep into small mass
errors does not, so a **crossover is required**: for
σ_lnM ≤ k·σ_cond/a_gal (k = O(5), pinned by the §3.7 case-1 test) fall
back to the analytic Gaussian product — there the family difference is
O(σ_lnM) (§3.7 case 8) and truncation is negligible for interior hosts.
(Prior-centred nodes are an acceptable alternative crossover; the
fallback is simpler and closed-form.)
**IMPLEMENTATION CORRECTION (found by the kernel-parity goldens at
implementation, 2026-07-27 — the G2d §4 discipline).** The stated width
condition ALONE misfires for **mass-mismatched hosts**: a_gal ≪ 1 makes
the *linearized* width σ_gal = σ_lnM·a_gal tiny even when the prior is
broad (σ_lnM ≈ 0.7) — and there the GH quadrature was CORRECT (it
integrates the smooth fat lognormal tail at the GW peak; no spike exists
to alias), while the Gaussian fallback replaces that fat tail with
exp(−thousands) (measured: golden `near_lowmass_bound_mt_4d` numerator
0.061 → 7×10⁻¹⁵). Genuine aliasing requires a Gaussian-like *spike*,
which requires σ_lnM itself small: an in-span spike with moderate σ_lnM
is impossible (a_gal ≈ μ_cond forces σ_gal ≈ σ_lnM > k·σ_cond). The
implemented crossover therefore requires BOTH σ_gal ≤ k·σ_cond AND a
family-validity cap **σ_lnM ≤ 0.1** (`_MASS_TRUNC_GH_CROSSOVER_SIGMA_LNM_MAX`).
Consequence: the catalogue regime (σ_lnM ≥ 0.58) NEVER crosses over — all
pre-correction GH golden values are preserved exactly — while the
σ_lnM → 0 spec-mass limit is restored as ratified. The fat-tail case is
pinned by `test_mz_integral_broad_mismatched_host_keeps_fat_tail`. **Truncation-edge caveat:** for
hosts with M*(z) within ~±(node span)·σ_cond·M_det/(1+z) of a bound, the
hard truncation puts a step INSIDE the GH node span; convergence there
degrades from spectral to ~O(1/n) — it is not "resolved" by locality.
The §4 item-3 GH 24-vs-48 test must therefore stratify on
|ln M*(z_g) − ln M_bound| so bound-straddling events (the operative 65%
boundary-host population) are explicitly covered.
Cost: 24 kernel/R_eff evaluations per z-node — ≈50×24 = 1200
vectorized evaluations per host per h-value on the volume_deconv path
(R_eff is analytic numpy; the mz factor is a subdominant share of the
numerator next to `dist_vectorized` + the 3D MVN). Estimated ≤×5 on the
2D per-host legs; a measured benchmark is §4 item 7.

**Option N-B — moment-matched truncated Gaussian.** Compute (m₁, v) of
(M1) once per host (z-independent — the kernel is in source-frame M;
one 401-pt quadrature like `eddington_shifted_host_mass`), then keep the
closed-form Gaussian product with μ_gal = m₁(1+z)/M_det,
σ_gal = √v(1+z)/M_det. Near-zero marginal cost; generalizes G2d (matches
mean AND variance).

**Decisive argument — the narrow-GW limit.** The GW M_z likelihood is
narrow (σ_cond ~ 10⁻² in fraction units ≪ σ_lnM ≈ 0.6), so

    σ_cond → 0:   mz(z) → p_a(μ_cond(z)) = p_g(M*(z)) · M_det/(1+z),
                  M*(z) = μ_cond(z) · M_det/(1+z) .                        (M7)

The numerator evaluates the kernel's **local density at M***, not its
moments. A moment-matched Gaussian is the wrong approximation in exactly
the operative regime: for boundary hosts (65% of the R_eff-weighted
population) the truncated kernel is one-sided/strongly skewed, and N-B
re-leaks mass across the bound — reintroducing a smaller copy of the
defect being fixed. **[RATIFY-M3: numerator scheme = Option N-A
(per-galaxy GH quadrature, order 24) PLUS the mandatory small-σ
crossover to the analytic Gaussian product at σ_lnM ≤ k·σ_cond/a_gal —
N-A as currently implemented lacks the crossover and provably fails the
σ_lnM→0 limit (§3.7 case 1), so ratification is of N-A-with-crossover,
not of the bare implementation; N-B rejected for the numerator on the
(M7) pointwise argument; GH-order adequacy pinned by a convergence test
(n = 24 vs 48) stratified on bound-straddling hosts. Recommendation:
adopt N-A with the crossover.]**

Note for A/B design — **a reachable counted-once violation, must be
guarded**: `host_z_kernel = "point"` composes freely with
`normalization_mode = "mass_trunc"` from the CLI (`resolve_host_z_kernel`
has no guard, bs.py:110–137). In that combination the numerator point
path uses the bare untruncated Gaussian at host_M — with neither the G2d
shift (gated off under mass_trunc, bs.py:3424–3428) nor truncation
(bs.py:3503–3510) — while the denominator uses the truncated LN×R_eff
prior (bs.py:3536–3542), and in mass_trunc mode D_g is load-bearing
(ratio-of-sums assembly): N_g and D_g silently carry DIFFERENT mass
priors, violating the §3.4 "identical in N_g and D_g" invariant. The
cell is expressible but prior-inconsistent. The §4 item-1 flag design
must raise (ValueError) on the mismatched combination (or force the
matching Gaussian denominator with it); the A/B matrix in §4 must not
claim that cell.

### 3.4 Denominator and the counted-once-in-M ledger **[RATIFY-M4] [RATIFY-M5]**

The per-galaxy selection term keeps the (M5) priors with p_det in place of
the GW likelihood:

    D_g = ∫ dz  p_g(z) ∫ dM  p_det(d_L(z; h), M(1+z)) · p_g(M) .           (M8)

With a Gaussian p_g the inner integral is the exact erf-sum (piecewise-
linear p_det in M_z; Owen 1980). With (M1) the erf-sum does not apply
(Gaussian-prior-only); the implemented replacement is Gauss–Legendre
n = 64 in ln M over the peak-aware window: the CENTER ln M_g is first
clipped into [ln M_min, ln M_max], then ±10σ_lnM is applied and
intersected with the bounds (bs.py:355–373). The center-clip order
matters: for an out-of-bounds M_g (RV15 can produce M_BH > 10⁷) with
small σ_lnM, clipping the finished window instead would invert it
(negative Z_M); the implemented center-clip cannot. This is the **same
support and weight as Z_M**
(bs.py:447–528), so numerator, denominator, and normalization share ONE
prior. The peak-aware window is the volume_trunc lesson (a full-window GL
would alias a narrow spike; bs.py:355–373). Accuracy: p_det is smooth and
bounded in [0,1]; GL-64 against a lognormal on its own ±10σ window is
spectrally convergent; a knot-refinement test (n = 64 vs 128) is §4
item 3. **[RATIFY-M4: denominator = same kernel (M1), GL-in-lnM inner
integral; the erf-sum path is retired on the new-kernel path (it remains
the exact method for the Gaussian-family modes). Recommendation: adopt.]**

**Counted-once-in-M ledger.** Mass information appears at exactly three
sites; each piece is counted once:

1. **Per-galaxy kernel (M1)** — the σ_lnM scatter and the *shape* of
   R_eff over the scatter range; identical in N_g (M5) and D_g (M8), so
   the per-host ratio is internally consistent (proposal = prior lineage,
   bs.py:3411–3423).
2. **Sum-level hosting weight** w_g = R_eff(M_g)/(1+z_g) — the
   *magnitude* of R_eff, point-evaluated (bs.py:606–629; G2c/G2d §6
   status). The exact marginal would be the scatter-averaged expected
   rate Z_M/(1+z_g) — Z_M as defined in (M2) is already computed against
   the normalized LN (the 1/(σ_lnM√2π) lives inside
   `_mass_trunc_lnM_weight`, bs.py:330), so no extra normalization
   factor exists; conditioning on in-support mass would instead give
   Z_M/(P_in·(1+z_g)) with P_in = Φ(B)−Φ(A) from (M3). Keeping the
   point evaluation is a deliberate next-order deferral, consistent
   with G2d §6.
3. **Selection-side Σ_glob_wbh composition** — point-evaluated
   p_det(d_L(z_g), M_g(1+z_g)) with weight R_eff(M_g)/(1+z_g)
   (bs.py:1392, 1411–1425). The z×M_z-resolved variant (Σ_glob ×0.546,
   slope +0.106 vs +0.505; −58 ln predicted gap effect) is **owned by the
   FIX-3 §7.1 thread** (DERIVATION_ZRESOLVED_SURVIVAL.md), which also
   established that FIX-2/FIX-3 must ship and be gated TOGETHER. This
   derivation does not duplicate it; it records the dependency (§3.8
   branch d, §4 item 8).

**[RATIFY-M5: R_eff enters the kernel shape-only (Z_M-normalized); w_g
stays point-evaluated R_eff(M_g)/(1+z_g) at this order; any upgrade of
site 2/3 is coordinated with FIX-3, not folded into this kernel change.
Recommendation: adopt, with the deferral documented as above.]**

### 3.5 Joint (M, z) structure — and why the broadened z-kernel exposes the mass defect

**Factorization.** The host priors factorize, p_g(z)·p_g(M): catalogue z
(spectroscopy/photometry) and M (stellar-mass proxy) are treated as
independent measurements — an assumption, stated here, and **NOT merely
second-order for real data** (see the provenance paragraph below): for
photo-z hosts the z error propagates into ln M_g at first order. The
kernel (M1) is z-independent (source-frame M). The **integrand does not
factorize**: the GW likelihood couples (z, M) through d_L(z) and
M_z = M(1+z), and through the CRB d_L–M_z cross-covariance (proj). The
correct treatment is therefore the nested quadrature exactly as
implemented (outer z × inner GH in M) — (M5) makes **no separability
approximation**, and none may be introduced (e.g. precomputing the mass
marginal at z_g and reusing it across z would drop the (1+z) coupling).

**Catalogue-mass provenance (real data) — first-order caveats
[RATIFY-M7].** GLADE+ stellar masses are luminosity-derived at a fixed
fiducial cosmology, so M_* ∝ d_L(z_cat; h_fid)² and hence
δ ln M_BH ≈ 2β·δ ln d_L ≈ 2.1·δ ln d_L. Two consequences the bare
p_g(z)·p_g(M) statement misses:

- **(z, M) covariance for photo-z hosts.** With σ_z/z ~ O(1) at the
  venue, the catalogue-z error enters ln M_g at FIRST order, fully
  correlated with the z-kernel — the factorization fails at first order
  for precisely the host class where z-broadening gives the 2D channel
  its leverage (this section's own mechanism). Before real-data
  promotion this covariance must be modelled (M_g as a function of the
  hypothesis z along the kernel) or explicitly bounded.
- **Hidden h-dependence of the datum M_g.** Across the 0.60–0.86 h-grid,
  M_g ∝ h⁻²β implies δ ln M_g up to ≈ 0.4 relative to h_fid —
  comparable to σ_lnM = 0.58. (M1) itself contains no h, but the datum
  it is centred on does; the §3.7 case-6 h-independence argument is
  structural about the kernel, not complete about the data.

**[RATIFY-M7: record both provenance channels in the G7 systematics
budget with the magnitude estimates above; before real-data 2D
promotion, either model the d_L²-induced (z,M) covariance for photo-z
hosts (and the M_g(h) rescaling) or bound their H0 impact.
Recommendation: adopt — G7 entry now, model-or-bound as a gate on
real-data promotion, coordinated with the §3.8 branch discriminators.]**

**Why the mock's δ-kernel masks the defect.** In the mock, the generator
injects the EMRI at the host's catalogue values — the M-side pairing is
generator-near-exact just as the z-side is (hostz §0). Under the point
kernel z is pinned at z_g: the mass factor mz(z_g; h) varies with h only
through μ_cond(d_L(z_g; h)) — the narrow CRB cross-correlation channel —
against a FIXED μ_gal(z_g) that equals truth by construction. The wide
mass-kernel shape (family, tilt, truncation) is never swept: kernel
defects enter only as a nearly-h-inert per-host weight plus the narrow
μ_cond channel centred on truth. Cell C measures exactly this: 2D at
truth, defect invisible.

**Why z-broadening exposes it.** Under the broadened p_g(z), the numerator
convolves along the trajectory M*(z) = μ_cond(z)·M_det/(1+z): as z sweeps
the host/GW window, M* sweeps **through the mass kernel's shape**, even
with a generator-exact M_g. In the narrow-GW limit (M7),

    ln mz(z) = ln p_g(M*(z)) − ln(1+z) + const ,
    d ln mz/dz = [d ln p_g/d ln M](M*) · (d ln μ_cond/dz − 1/(1+z)) − 1/(1+z) . (M9)

mz(z) is a *second z-likelihood*. If the kernel shape is wrong
(linear-Gaussian vs truncated LN×R_eff), the error is a multiplicative
z-tilt Δ(z) = ln mz_prod − ln mz_true, which shifts the numerator's
effective redshift by δz ≈ σ_w²·Δ′(z_g) (Laplace; σ_w = the combined
z-width of gw_3d·p_g(z)), hence d_L, hence h. Two consequences, both
testable:

- the shift scales ~**σ_w²** at leading (Laplace) order — consistent in
  DIRECTION with the toy's leverage growth (+0.0004 → +0.021 as σ_z/z
  goes 0.05 → 0.75), but the toy's measured growth is sub-quadratic and
  saturates at large leverage (per-step ratios ×3.2, ×2.0, ×1.3 vs
  quadratic ×4, ×2.8, ×2.25) — the σ_w² law is a small-leverage
  statement, not a quantitative match across the range;
- mass appears **only in the 2D channel**, so the defect is 2D-only —
  the measured info-monotonicity violation (2D +0.025 > 1D +0.013) and
  the cell-B "2D is the exception" finding are both explained
  qualitatively by (M9).

The sign of the net tilt is NOT cleanly derivable by a point argument (the
naive argument gives LOW; the full marginalisation over the truncated
kernel's shape gives HIGH — FINDINGS §4). The sign statement this
derivation relies on is the **measured** toy sign (HIGH, ~3σ, all
leverages) with the shape difference decomposed into three separable
sub-effects — family (linear vs log), tilt (R_eff), truncation (bounds) —
whose individual attribution is a validation deliverable (§4 item 4), not
an assumption.

### 3.6 Dimensional analysis (every defined object)

- σ_lnM, α_g, A, B, a, μ_cond, σ_cond, t: dimensionless. ✓
- LN(M; M_g, σ_lnM): [M☉⁻¹] (density in M). R_eff(M): [Gyr⁻¹].
- Z_M (M2): [Gyr⁻¹] — a scatter-averaged rate; dividing by it makes
  p_g(M) = LN·R_eff·1[·]/Z_M: [M☉⁻¹] — a normalized density in M. ✓
- `_mass_trunc_lnM_weight` = LN·R_eff·M: [Gyr⁻¹] as a d ln M density;
  ∫ w d ln M = Z_M. ✓ (bs.py:308–311.)
- Pushforward p_a = p_g·M_det/(1+z): [M☉⁻¹]·[M☉] = dimensionless density
  in the dimensionless a — same units as the Gaussian-product mz it
  replaces (both are densities in a). ✓
- mz(z) (M6): dimensionless density in a; N_g^{2D} carries
  gw_3d [density in (φ,θ,d_L_frac)] × mz × p_g(z) [z⁻¹, dimensionless]
  — identical unit structure to the current path. ✓
- (M8) inner integral: p_det ∈ [0,1] averaged over a normalized density →
  dimensionless; D_g = window-averaged detection probability ∈ [0,1]. ✓
- (M3): [M☉]. (M4): [M☉]. w_g: [Gyr⁻¹]. ✓

### 3.7 Limiting cases (minimum set; each becomes a test in §4)

1. **σ_lnM → 0** (floor 10⁻⁶): the EXACT kernel gives p_g → δ(M − M_g)
   for interior M_g; mz → N(μ_cond; M_g(1+z)/M_det, σ_cond) — the
   current Gaussian-product formula's σ_gal → 0 limit; D_g inner →
   p_det(d_L, M_g(1+z)). The bare GH-24 quadrature does NOT recover this
   limit (it aliases the prior spike for σ_lnM·a_gal ≲ σ_cond and
   returns exactly 0 at the floor, §3.3): the test therefore targets the
   §3.3 crossover path — it must pin (i) the fallback value against the
   Gaussian-product formula and (ii) continuity of mz across the
   crossover threshold k·σ_cond/a_gal. The spec-mass limit (mass analog
   of `test_volume_deconv_numerator_collapses_to_point_as_sigma_to_zero`)
   holds for the switched implementation, not the bare GH kernel.
2. **Bounds → (0, ∞), log-linear R_eff**: p_g = tilted lognormal;
   effective mass M_g e^{(α_g+½)σ²} — G2d's rate tilt + the derived
   family term (M4), to stated tolerance (<1% vs the analytic form for a
   synthetic log-linear R_eff).
3. **R_eff = const, no truncation**: bare median-M_g lognormal — the pure
   measurement kernel; mz = exact LN–Gaussian convolution (pinned
   numerically by GH vs brute-force quadrature).
4. **Interior hosts** (|ln(M_bound/M_g)| ≥ kσ_lnM): truncation
   corrections ≤ Φ(−k) per side (2.3% of kernel mass at k = 2, smaller on
   the mean); interior behavior = case 2 within tolerance.
5. **σ_cond → 0**: mz → pointwise kernel density (M7) — the limit that
   ratifies N-A over N-B; test at σ_cond = 10⁻⁴.
6. **h-independence of the kernel — structural.** (M1) is a function of
   (M; M_g, σ_lnM, bounds, R_eff) only: **no h anywhere**, so
   ∂p_g(M)/∂h ≡ 0. But the marginal mz is NOT h-independent at fixed z:
   h enters through μ_cond, because luminosity_distance_fraction =
   d_L(z; h)/_det_d_L feeds x_obs → μ_cond (bs.py:3443–3465) — the CRB
   cross-correlation channel that §3.5 and §3.8 branch (b) are built on.
   This is intended physics, not a defect (mock-mode h-dependence of the
   kernel truly is forbidden; h-dependence of μ_cond is not). h enters
   the 2D numerator through d_L(z; h) in BOTH gw_3d and μ_cond (and the
   z-window mapping), and the denominator through p_det's d_L(z; h)
   argument, where selection h-dependence belongs (MFG). Test:
   ∂p_g(M)/∂h ≡ 0 and ∂mz/∂h = 0 at fixed (z, μ_cond) — structurally,
   `_mass_trunc_mz_integral` takes no h argument; the fixed-z derivative
   of mz is nonzero by design and must not be asserted zero.
7. **(1+z) → 1**: a = M/M_det, trivial pushforward; mz reduces to the
   z-free convolution.
8. **Small-σ continuity with the current default**: for σ_lnM ≪ 1,
   LN(M; M_g, σ_lnM) → N(M; M_g, M_g σ_lnM)(1 + O(σ_lnM)) — the current
   linear-Gaussian path is recovered continuously; realized in the
   implementation via the §3.3 crossover (which falls back to exactly
   that path), so no discontinuity is introduced anywhere in the F5
   sweep's small-error corner.

### 3.8 Scale reconciliation — the crux **[RATIFY-M6]**

The honest quantitative position, stated before any A/B:

**The measured cell-B failure.** A Laplace fit through the top of the
cell-B 2D profile ({0.76, 0.80, 0.86} → {+20.2, +29.4, +13.4} ln) gives
peak ĥ ≈ 0.803, curvature A ≈ 5.0×10³ ln·h⁻² (combined σ_h ≈ 0.010).
Displacement from truth: **δ_obs ≈ +0.073**. Per-event budget: 29.4/3454
≈ 0.0085 ln at MAP — a tiny per-event asymmetry, coherently aggregated.

**What the toy supplies.** Per-event mass-kernel shift
δ_toy = +0.0004 (spec-z-like leverage) … +0.021±0.008 (σ_z/z = 0.75);
+0.016…+0.02 at shallow-shell leverage σ_z/z ≈ 0.5–0.65. The toy is
single-host, no selection D(h), flat-prior photo-z anchor, moderate-z
hosts — its own caveats say "sign and rough magnitude, not a
campaign-grade number".

**The bridge does not close.** If the cell-B venue's effective z-leverage
is spec-z-like (deep-venue catalogue σ_eff, small), the toy predicts
δ ≈ 0.0004–0.003 — explaining ≲4% of δ_obs. Even at the toy's *maximal*
leverage, δ_toy ≈ 0.021 is **≥3.5× short** of 0.073. Sources assert
sign-consistency only; no existing measurement establishes that the
truncated-lognormal fix closes the cell-B gap. Additionally, cell B pairs
a normalization derived for the point/point generator pairing with a
broadened numerator (per-leg attribution instrumentation, not a candidate
mode) — part of the +29.4 ln may be attribution-instrumentation mismatch
rather than mass-kernel physics.

**Conclusion (recommended position): the (M1) kernel is NECESSARY —
the current family is wrong by construction at σ_lnM ≈ 0.6, the toy
measures a real HIGH contribution, and no correct real-data 2D mode can
be built on an untruncated linear Gaussian — but it is NOT ESTABLISHED AS
SUFFICIENT for 2D real-data mode.** The residual is enumerated as
derivation branches, each with a decisive discriminator:

- **(a) Kernel defect at real leverage** (this fix is the whole story
  iff production leverage ≫ toy leverage, e.g. because the operative
  scale is the GW window, not the host σ_eff). *Discriminator:* the §4
  item-4 A/B — switch the cell-B (and cell-A′) 2D numerator+denominator
  to (M1) and read the MAP/ln movement against §3.9 P3.
- **(b) CRB cross-correlation channel**: μ_cond(z) drift (proj) along the
  d_L direction interacting with the kernel shape — operates on the GW
  window scale even for spec-z hosts. *Discriminator:* diagnostic run
  with proj zeroed (cross-covariances dropped) in the 2D numerator;
  attribution of the residual tilt with/without.
- **(c) Attribution-instrumentation mismatch in cell B**: the
  generator-consistent legs are not derived for a broadened numerator.
  *Discriminator:* run the kernel A/B inside `absolute_marginal`
  (the ratified real-data normalization): cells A′ (Gaussian-M) vs A″
  ((M1)-M), both with volume_deconv z-kernel. The real-data-relevant
  comparison; removes the mismatch by construction.
- **(d) The M-leg of the selection normalization**: Σ_glob_wbh is
  point-evaluated in M with no scatter/truncation; a numerator-side
  kernel change without the matching selection-side M treatment can
  leave a composition mismatch. *Owned by FIX-3* (z×M_z variant already
  measured: Σ_glob ×0.546, −58 ln predicted); coordinate, don't
  duplicate; the FIX-2/FIX-3 ship-together rule applies.
- **(e) 2D numerator quadrature resolution**: fixed_quad n = 50 over the
  wide GW window with a narrow spec-z p_g(z) spike, now modulated by
  mz(z) — the 1D channel shares the z-quadrature and is clean in cell B,
  so any 2D-specific aliasing must come through the mz modulation.
  *Discriminator:* n = 50 vs 200 convergence on the cell-B 2D numerator
  for a stratified event sample.
- **(f) B_num residual** (+0.004…+0.006 HIGH at h ≥ 0.72, 1D P–P
  harness): small, same sign, common to both channels — bounded by the
  existing harness finding; not 2D-specific.

**[RATIFY-M6: adopt the position "necessary, not established sufficient";
real-data 2D mode = absolute_marginal normalization + ratified
volume_deconv host-z kernel (hostz §3) + (M1) mass kernel, designated as
CANDIDATE pending the branch discriminators; the 2D channel remains OPEN
until the cell-A′/A″ residual is attributed. Recommendation: adopt — do
not promote the kernel to a production-mode claim on sign-consistency
alone.]**

### 3.9 Pre-registered predictions (BEFORE any A/B with the ratified kernel)

- **P1 (kernel level, no posterior run needed).** Boundary-host effective
  masses move as measured in EXP-45 §2: vs G2d, ≈−15% at M_g ≈ 1.5×10⁴,
  ≈+22% at M_g ≈ 7×10⁶ (posterior-mean sense); interior effective masses
  move by the family term e^{σ_lnM²/2} ≈ +18% at σ_lnM = 0.58 (M4). The
  numerator-operative quantity — kernel density at the GW peak,
  p_g(M*) vs N(M*; M_g, σ_M) — changes by O(1) factors for boundary
  hosts with M* beyond a bound (hard zero vs Gaussian tail).
- **P2 (direction).** Switching the 2D legs to (M1) moves the 2D MAP
  **DOWN (toward truth)** on the seed-1000 venue, in every cell where the
  z-kernel is broadened (A′→A″ and B→B″). Sign firm from the toy (~3σ).
- **P3 (magnitude, conditional).** If the toy's leverage mapping
  transfers (leverage = host σ_eff scale): MAP movement −0.003…−0.02,
  i.e. recovery of roughly 2–13 of the +29.4 ln and a residual 2D MAP
  ≥ 0.78 in the cell-B configuration — the fix would be measurably
  helpful and measurably insufficient. If instead the MAP lands at
  0.73 ± 0.01, the operative leverage is the GW-window scale and the toy
  under-measured production leverage — sufficiency established, leverage
  model revised. **Middle band, assigned:** a MAP in (0.74, 0.78)
  falsifies the leverage-scale dichotomy itself — neither leverage model
  holds cleanly — and forces mixed attribution across §3.8 branches
  (a)+(b)/(c): pre-registered as "partial closure; mandatory branch
  decomposition before any further claim". Every outcome region now has
  a pre-registered interpretation; the prediction is the full partition,
  not a dichotomy with an unassigned gap.
- **P4 (falsifiers).** (i) Post-fix 2D MAP moves UP or the +29.4 ln gap
  grows → the toy's sign does not transfer; §3.1–§3.3 re-open. (ii)
  Post-fix 2D MAP > 0.76 with 1D at truth in the absolute_marginal
  cells → necessary-not-sufficient confirmed; branches (b)/(d)/(e)
  escalate to first priority. (iii) Interior-host golden events shift by
  more than the P1 family term → implementation defect, not physics.
- **P5 (structural).** ∂p_g(M)/∂h ≡ 0 (the kernel carries no h) and
  ∂mz/∂h = 0 at fixed (z, μ_cond) hold structurally —
  `_mass_trunc_mz_integral` takes no h argument, so at the function
  level this is checkable and any h reaching the mass marginal other
  than through μ_cond(d_L(z; h)) is a bug, full stop. At fixed z, mz DOES
  vary with h through μ_cond (§3.7 case 6, intended CRB channel); a
  fixed-z ∂mz/∂h = 0 assertion would condemn a correct implementation
  and is NOT a prediction of this derivation.

## 4. Validation plan (after ratification)

1. **Expressibility.** Decouple the M-kernel from `normalization_mode`
   with a `host_mass_kernel ∈ {auto, gaussian, trunc_lognormal}` flag
   mirroring `host_z_kernel` (#48 precedent, `resolve_host_z_kernel`,
   bs.py:110–137), so the real-data combination (absolute_marginal ×
   volume_deconv-z × (M1)-M) and the A/B matrix are expressible without
   mode aliasing. The point×(M1) numerator cell is expressible but
   prior-inconsistent (§3.3): the flag resolution MUST raise
   (ValueError) on `host_z_kernel="point"` combined with the (M1)/
   mass_trunc denominator, so the mismatched-prior configuration cannot
   run silently.
2. **Golden discipline.** Default path byte-identical
   (pipeline-parity golden); mass_trunc-family kernel goldens change ⇒
   REGEN_KERNEL_GOLDEN as a REVIEWED value-update step; any touch of the
   stored `BH_MASS_ERROR` provenance would trigger the reduced-CSV
   regeneration + cluster re-staging discipline (hostz implementation
   decision) — none is proposed here.
3. **Limiting-case tests** per §3.7 (8 cases, including the case-1
   small-σ crossover: fallback value + continuity at the threshold),
   plus quadrature convergence: GH 24 vs 48 (numerator), GL 64 vs 128
   (denominator, Z_M), **stratified on |ln M*(z_g) − ln M_bound|** so
   bound-straddling events (step inside the GH node span, ~O(1/n)
   convergence, §3.3) are explicitly covered — not just generic samples
   — and scalar-vs-batch bit-identity (bs.py:3898–3933, 3935–3957
   twins).
4. **Kernel A/B on the seed-1000 venue** (per-leg ln attribution, §3.8
   branches a/c): cells A′/A″ (absolute_marginal) and B/B″
   (generator-marginal instrumentation), scored against P2/P3/P4 BEFORE
   any production claim — the volume_trunc regression-gate lesson.
   Plus the branch-(b) proj-ablation and branch-(e) n=50/200 diagnostics.
5. **P–P harness mass extension.** pp_coverage currently has no mass
   dimension (pp_coverage.py:156, 276–278); add a 2D channel with
   synthetic lognormal mass scatter and known truth, kernel-switchable
   (gaussian vs (M1)) — the decisive calibration test that the kernel is
   *calibrated*, not merely different. Named as the next quantitative
   step by EXP-45's own caveats.
6. **Blind alternative-truth mock (#39)** remains the anti-tuning gate.
7. **Cost benchmark.** Measured wall-clock of the (M1) legs vs the
   analytic/erf-sum path on a venue slice (estimate §3.3: ≤×5 on the 2D
   per-host legs; verify).
8. **Coordination.** The Σ_glob_wbh M-composition (branch d) ships with
   FIX-3 under the FIX-2/FIX-3 joint gate, not with this kernel; the
   one-line handler.py:28 M_max=10⁶ consistency check is filed with the
   implementation PR.

## References

- Three-way A/B: `results/lcat_h_dependence_20260725/threeway_ab/THREEWAY_AB_READOUT.md`.
- EXP-45: `results/mass_kernel_truncation_20260713/FINDINGS.md`
  (`mass_trunc_probe.py`, `mass_kernel_h0_toy.py`).
- Host-z kernel (ratified): `docs/derivations/hostz_pv_photoz_kernel.md`.
- G2d: `docs/derivations/G2d_host_mass_rate_prior.md`.
- FIX-2/FIX-3: `results/lcat_h_dependence_20260725/DERIVATION_ZRESOLVED_SURVIVAL.md`.
- P–P full-power (1D-only): `results/pp_fullpower_20260727/FULLPOWER_READOUT.md`.
- Reines & Volonteri 2015, arXiv:1508.06274, Eqs. (4)–(5) (verified
  2026-07-27); Babak et al. 2017, arXiv:1703.09722, Eqs. (31), (34);
  Bishop 2006, PRML, Eqs. (2.81)–(2.82); Owen 1980, Commun. Statist.
  B9(4) 389–419; Eddington 1913; Mandel, Farr & Gair 2019,
  arXiv:1809.02063; Abramowitz & Stegun 25.4.46; Gray et al. 2020,
  arXiv:1908.06050 (structure only — eq. numbers unverified).
