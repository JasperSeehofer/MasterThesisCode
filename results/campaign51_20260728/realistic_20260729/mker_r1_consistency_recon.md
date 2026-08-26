# [P3-MKER] R1 — numerator/normalization consistency recon (Refute-by(a))

**Scope:** zero-compute code recon only. Answers: does the catalogue-side BH mass and its
uncertainty get treated the SAME way on the numerator side (per-host with-BH-mass likelihood)
and the normalization side (selection/completeness terms) of the dark-siren ratio, under the
CLAIM_P3_MKER_20260826.md §5 Refute-by(a) test (Gray 2023 G23-b bridge: truncation/renormalization
is harmless only under numerator/normalization consistency)?

**Verdict: INDETERMINATE — split by which "normalization" object the bridge means.** There are
TWO distinct normalization objects in the with-BH-mass chain, and they disagree:

1. **Per-host in-catalogue ratio N_g/D_g** (the object `single_host_likelihood` returns,
   `L_cat_with_bh` in the row #205 exhibit): mass-uncertainty-CONSISTENT. Both sides convolve
   the SAME Gaussian (mean `host_M_eff`, width `host_M_error`) — or, under the experimental
   `mass_trunc` mode, the SAME truncated-lognormal×R_eff prior (`sigma_lnM`, `Z_M`). This part
   of the chain supports demoting claim (a) toward "design choice" — but not for the reason
   claim (a) states (see caveat below).
2. **Population-level Σ_global(h)** (`precompute_global_catalog_selection`, the in-catalogue
   completeness-fraction normalization that determines how `L_cat` combines with the
   missing-galaxy completion term): under the DEFAULT production setting
   (`sigma4d_mass_kernel="point"`), it evaluates p_det at the EXACT catalogue mass
   `M_g*(1+z_g)` — a point, with NO mass-uncertainty convolution at all. This directly
   contradicts the convolved treatment in (1). It is self-documented in the code as a KNOWN,
   un-defaulted gap ("issue #24"; an instrument, `sigma4d_mass_kernel="kernel"`, exists to fix
   it but is not on by default).

So the falsifier's precise wording — "the catalogue mass is treated as exact BY DESIGN
elsewhere in the chain, CONSISTENTLY in numerator AND normalization" — is **not met**: the
pipeline does not treat catalogue mass as exact consistently. It treats it as a distribution in
one normalization object and as a point in the other. Whether that defeats Refute-by(a) depends
on which object the G23-b bridge is actually about (see §5 below).

**Caveat on claim (a) itself:** the claim's own premise — "the with-BH mass likelihood weights
candidates by a width dominated by GW-conditional σ_cond and does NOT convolve the full
uncertainty budget of the catalogue-side mass" — is only half right. Object (1) DOES convolve
catalogue mass uncertainty (`host_M_error`) into both numerator and denominator; it is dominated
by σ_cond because `host_M_error` is numerically small (the separately-tracked "host-mass errors
~3-7x too tight" / deferred log-normal-refactor issue — memory: `mass-relation-reines-volonteri`,
commit `555f018`), not because the code fails to convolve it at all. That is a magnitude/input
problem (already partly fixed, refactor deferred), distinct from an architectural
numerator/normalization mismatch. Object (2) is where an architectural mismatch genuinely lives.

---

## 1. NUMERATOR side — per-host with-BH-mass likelihood

File: `darksiren_emri/bayesian_inference/bayesian_statistics.py`

### 1a. Default production kernel (Gaussian product, `host_mass_kernel="gaussian"`)

Defaults confirmed:
- `single_host_likelihood_batch(..., normalization_mode: str = "generator_marginal", ...)` (line 6764,
  commented "production default since 2026-07-26 (MULTISEED_READOUT_20260726.md)").
- `host_mass_kernel: str = "auto"` (line 6770); `resolve_host_mass_kernel` resolves `"auto"` to
  `"gaussian"` **unless** `normalization_mode == "mass_trunc"` (lines 298-302):
  ```
  resolved = (
      ("trunc_lognormal" if normalization_mode == "mass_trunc" else "gaussian")
      if host_mass_kernel == "auto"
      else host_mass_kernel
  )
  ```
  Since production `normalization_mode` is `"generator_marginal"` (not `"mass_trunc"`), the
  DEFAULT with-BH-mass kernel is the analytic **Gaussian product**, not the experimental
  `mass_trunc` truncated-lognormal path.

Numerator (lines 6601-6616):
```
                mu_gal_frac = _host_M_eff * (1 + z) / _det_M
                sigma_gal_frac = host_M_error * (1 + z) / _det_M

                # Analytic Gaussian product integral:
                # ∫ N(x; μ_cond, σ²_cond) · N(x; μ_gal, σ²_gal) dx
                #   = N(μ_cond; μ_gal, σ²_cond + σ²_gal)
                # Eq. (14.31) in derivations/dark_siren_likelihood.md
                sigma2_sum = _sigma2_cond + sigma_gal_frac**2
                mz_integral = np.exp(-0.5 * (mu_cond - mu_gal_frac) ** 2 / sigma2_sum) / np.sqrt(
                    2 * np.pi * sigma2_sum
                )
```
`host_M_error` is the raw per-galaxy catalogue mass-error column, passed straight into
`single_host_likelihood` as a function argument (used unchanged here). So **the catalogue mass
IS treated as a distribution of width `host_M_error`** (converted to fraction-coordinate
`sigma_gal_frac`), not an exact point — contradicting claim (a)'s literal wording that it is
"not convolved."

### 1b. `mass_trunc` experimental mode (EXP-45)

Setup (lines 6519-6523):
```
        if _use_mass_trunc:
            # sigma_lnM (recovered from the stored linear error) + per-host Z_M for
            # the truncated lognormal x R_eff prior (see _mass_trunc_* helpers).
            _sigma_lnM = float(_mass_trunc_sigma_lnM(host_M, host_M_error))
            _Z_M = _mass_trunc_log_normalisation(host_M, _sigma_lnM).item()
```
Numerator call (lines 6557-6562):
```
            if _use_mass_trunc:
                # Truncated lognormal x R_eff mass marginal via Gauss-Hermite on the
                # narrow GW M_z peak (EXP-45). Supersedes the analytic Gaussian product.
                mz_integral = _mass_trunc_mz_integral(
                    mu_cond, math.sqrt(_sigma2_cond), 1.0 + z, _det_M, host_M, _sigma_lnM, _Z_M
                )
```
The narrow/wide crossover inside `_mass_trunc_mz_integral` (lines 857-865) always includes the
catalogue-mass width one way or the other — it is a numerical-quadrature switch, not a physics
omission:
```
    narrow = (sigma_gal <= _MASS_TRUNC_GH_CROSSOVER_K * sigma_cond) & (
        np.asarray(sigma_lnM, dtype=np.float64).reshape(np.shape(sigma_lnM) + (1,))
        <= _MASS_TRUNC_GH_CROSSOVER_SIGMA_LNM_MAX
    )  # (..., K)
    sigma2_sum = sigma_cond**2 + sigma_gal**2
    mz_gauss = np.exp(-0.5 * (mu_cond - mu_gal) ** 2 / sigma2_sum) / np.sqrt(
        2.0 * np.pi * sigma2_sum
    )
    mz: npt.NDArray[np.float64] = np.where(narrow, mz_gauss, mz_gh)
```
Both `mz_gauss` (analytic) and `mz_gh` (Gauss-Hermite over the truncated-lognormal prior) carry
`sigma_gal`/`sigma_lnM`; the `narrow` branch is a fast-path approximation of the SAME integral,
not a fallback to exact-mass.

## 2. NORMALIZATION side

Two distinct normalization objects were traced.

### 2a. Per-host denominator D_g — CONSISTENT with the numerator

Default (non-`mass_trunc`) path, `_bh_mass_denominator_inner_m_integral` (lines 5676-5763),
called at line 6725-6727:
```
            else:
                inner_m = _bh_mass_denominator_inner_m_integral(
                    z, detection_probability, host_phiS, host_qS, _host_M_eff, host_M_error, h
                )
```
Its docstring states the integral explicitly:
```
        g(z) = \int p_\mathrm{det}\big(d_L(z),\, M(1+z)\big)\,
               \mathcal{N}(M;\, M_g^\mathrm{eff},\, \sigma_M)\, dM .
```
and the body (lines 5748-5749):
```
    mu = host_M_eff
    sigma = host_M_error
```
followed by an exact erf-sum convolution of the piecewise-linear `p_det` interpolant against
`N(M; host_M_eff, host_M_error)` (Owen 1980 zeroth/first-moment identities). **Same `_host_M_eff`
mean and `host_M_error` width as the numerator (1a)** — single shared parameters, not
independently recomputed. The class docstring at the call site (lines 6702-6704) states the
formal object:
```
        # Denominator D_g = INTEGRAL p_gal(z) [ INTEGRAL p_det(d_L(z), M(1+z)) N(M) dM ] dz.
```
`mass_trunc` mode is likewise symmetric — numerator (1b) and denominator both draw
`_sigma_lnM`/`_Z_M` from the same single computation (lines 6522-6523) and both integrate the
SAME truncated-lognormal×R_eff prior (lines 6717-6723):
```
            if _use_mass_trunc:
                # Same truncated lognormal x R_eff prior as the numerator, so N_g and
                # D_g share ONE mass prior (Gauss-Legendre in ln M; the erf-sum closed
                # form is Gaussian-prior-only and does not apply).
                inner_m = _mass_trunc_denominator_inner_m_integral(
                    z, detection_probability, host_phiS, host_qS, host_M, _sigma_lnM, _Z_M, h
                )
```
**Conclusion for D_g/N_g: numerator/normalization consistent** in both the default Gaussian
kernel and the experimental `mass_trunc` kernel.

### 2b. Population-level Σ_global(h) — INCONSISTENT with the numerator under the default flag

`precompute_global_catalog_selection` (line 2657) computes `Σ_global(h) = Σ_g w_g p_det`, the
in-catalogue selection weight used to derive the catalogue-vs-missing-galaxy completeness split
(the "normalization" that determines how much weight `L_cat` itself receives relative to the
completion term — line 5413 lists it alongside `D(h)`/`beta_Gbar(h)` as one of the selection
normalization triad).

Default with-BH-mass branch (`sigma4d_mass_kernel` default is `"point"`, confirmed at line 2671:
`sigma4d_mass_kernel: str = "point"`), lines 2915-2932:
```
            else:
                M_z_g = M_g * (1.0 + z_g)  # observer-frame mass (P_det grid axis)
                # [PHYSICS] FIX-3 §7.1 [RATIFY-Z1/Z5]: Sigma_glob_wbh's averaging
                # measure is the CATALOGUE's joint (z_g, M_z,g) — when the flag is
                # on the galaxy's own z_g conditions the query, S(d_L(z_g;h) |
                # z_g, M_g(1+z_g)); sky stays isotropic (unchanged decision).
                # docs/derivations/fix3_zmz_catalog_selection.md §3.1 (K1)/(K2).
                p_det = np.asarray(
                    detection_probability_obj.detection_probability_with_bh_mass_interpolated(
                        d_L_g,
                        M_z_g,
                        phi_iso,
                        theta_iso,
                        h=h,
                        **_wbh_z_kwargs(detection_probability_obj, z_g),
                    ),
                    dtype=np.float64,
                )
```
`M_z_g = M_g * (1.0 + z_g)` — a single point evaluation at the RAW catalogue mass, with no
`host_M_error`/`sigma_lnM` term anywhere in this branch. This is a POINT-mass treatment.

The gap is self-documented in the codebase, not something I am inferring. The
`_smeared_global_pdet_expectation` docstring (lines 1654-1657), which implements the
Σ_global redshift-smearing symmetry fix for a *different* risk (num/denom σ_z symmetry, issue
#30), explicitly flags that the analogous MASS fix was intentionally NOT done there:
```
    With-BH-mass channel: the observer-frame mass tracks the smeared redshift,
    ``M_z(z) = M_g (1+z)`` (consistent z-propagation). The galaxy MASS-ERROR
    kernel of the numerator is intentionally NOT mirrored here (pre-existing
    point-``M_g`` treatment retained; tracked separately under issue #24).
```
An instrument to close the gap exists but is opt-in, not default (lines 2890-2906):
```
            if sigma4d_mass_kernel == "kernel":
                # [PHYSICS] Instrument J registered kernel (results/
                # prod2d_closure_20260818/PREREGISTRATION_TILT_BATTERY.md §1,
                # P2): replaces the point evaluation at M_z_g = M_g(1+z_g) by
                # the expectation over the per-galaxy Eddington-shifted mass
                # prior, via the SAME erf-sum inner-M machinery production's
                # own D_g uses. sigma_g = catalogue BH_MASS_ERROR. M_eff_g
                # carries the SAME Eddington-in-M shift D_g uses, gated by
                # --eddington_m (instrument E); NO R_eff/mass_trunc lognormal
                # inside the kernel (w_g stays the point rate weight computed
                # above; Sigma^phi is untouched -- it contains no per-galaxy
                # mass evaluation).
                sigma_g = M_error_all[eligible]
                M_eff_g = (
                    _eddington_shifted_host_mass_batch(M_g, sigma_g) if eddington_m == "on" else M_g
                )
                p_det = _sigma4d_mass_kernel_expectation(
                    z_g, M_eff_g, sigma_g, phi_iso, theta_iso, h, detection_probability_obj,
                )
```
The comment explicitly names the object it is fixing: "replaces the point evaluation at
`M_z_g = M_g(1+z_g)` by the expectation ... via the SAME erf-sum inner-M machinery production's
own D_g uses" — i.e. the pipeline authors already recognize `D_g` (convolved) and `Σ_global`
default (point) as being on different footings, and built "Instrument J" specifically to make
them match. That instrument is not the default.

### 2c. `detection_probability_with_bh_mass_interpolated` itself carries no uncertainty model

File: `darksiren_emri/bayesian_inference/simulation_detection_probability.py`, lines 2018-2058.
Docstring:
```
        """Detection probability including BH mass dependence (survival form).

        Interpolates the 2D detection-horizon survival grid
        ``p_det(d_L, M_z) = K_M-weighted P(d_hor >= d_L)`` with a linear
        ``RegularGridInterpolator`` (``bounds_error=False``, ``fill_value=None``).
        ...
        Args:
            d_L: Luminosity distance in Gpc.
            M_z: Observer-frame (redshifted) BH mass in solar masses.
```
This function is a pure point-wise interpolator: it takes ONE `M_z` value per query and returns
one `p_det`. It has no internal notion of mass uncertainty at all — whether the caller convolves
over a mass distribution (as `_bh_mass_denominator_inner_m_integral` and
`_mass_trunc_denominator_inner_m_integral` do, calling it many times on a quadrature grid) or
evaluates it once at a point (as the default Σ_global branch does) is entirely a caller-side
decision. This confirms §2a/§2b are architecturally independent call sites that happen to have
been given DIFFERENT treatments, not two paths through one shared uncertainty model.

## 3. Answer to the decision question

**Not a single yes/no.** Two normalization objects exist in the with-BH-mass chain:

| Object | Mass treatment | Matches numerator (N_g)? |
|---|---|---|
| N_g (numerator, per host) | Gaussian (`host_M_eff`, `host_M_error`) or truncated-lognormal (`sigma_lnM`) | — |
| D_g (per-host denominator) | SAME Gaussian / SAME truncated-lognormal, same params | **YES** |
| Σ_global(h) (population selection normalization, default `sigma4d_mass_kernel="point"`) | Point at `M_g*(1+z_g)`, no uncertainty | **NO** — self-documented gap ("issue #24"), instrument built (`sigma4d_mass_kernel="kernel"`) but not defaulted on |

If the G23-b bridge's "normalization" means the per-host `D_g` that directly forms `L_cat_with_bh
= N_g/D_g` (the object in the row #205 exhibit, `1.39e-85`), then Refute-by(a) is **SUPPORTED**:
the current kernel IS the same conditional-likelihood family on both sides of that ratio, and the
narrow-kernel exhibit's k~19σ pull is a magnitude problem (small `host_M_error`, tracked
separately), not a numerator/normalization mismatch.

If "normalization" means the completeness/selection normalization that determines how `L_cat`'s
weight is set relative to the missing-galaxy completion term (Σ_global), then Refute-by(a)
**FAILS**: catalogue mass is treated inconsistently across the with-BH-mass chain (convolved in
D_g, point in Σ_global by default), and the code's own comments concede this ("issue #24").

**This ambiguity is exactly the kind the task brief asked me to surface rather than paper over.**
I cannot resolve which normalization object the G23-b passage (Gray 2023 §2.1.3,
truncation/renormalization harmless only under numerator/normalization consistency) is actually
about without re-reading the source text itself, which was outside this recon's scope (the R0
sweep already banked the citation as "status UNCHECKED against our mass-kernel code" — this recon
checked the CODE side only, per the task's framing; the LITERATURE side of the bridge is still
open).

## 4. What this recon did NOT check (explicit gaps)

- Whether `host_M_error` (the shared width in §2a) itself is dimensionally/physically correct
  post-555f018 (i.e., whether the deferred log-normal refactor / R&V15 0.55 dex scatter would
  change these conclusions) — out of scope; tracked separately.
- The magnitude of the Σ_global point-vs-convolved discrepancy in nats/H0-effect — not measured
  here (this is a zero-compute recon; measuring it is Instrument J's job, `sigma4d_mass_kernel`
  A/B, already built but unrun in this recon).
- Whether Σ_global's role in the full posterior makes its mass-treatment choice material at all
  (it multiplies a completeness fraction, not the per-candidate likelihood ratio directly — its
  leverage on H0 could be much smaller than D_g/N_g's).
- The Gray (2023) §2.1.3 source text itself, to confirm what "normalization" it refers to.
