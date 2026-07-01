# PHYSICS-CHANGE-PROTOCOL — Sky-Aware Selection Function

**Status:** Proposed (paper-blocker fix). Requires user approval before any code edit.
**Trigger files touched:** `bayesian_inference/bayesian_statistics.py`,
`bayesian_inference/simulation_detection_probability.py`,
`galaxy_catalogue/pixel_completeness.py` — all on the `/physics-change` trigger list (CLAUDE.md).
**Commit prefix (mandatory):** `[PHYSICS]`.
**Audit source:** `.planning/PSAMPLE-PCOMP-AUDIT-20260701.md` (R1 = most acute term).

---

## 0. The defect, in one paragraph

The **generator** draws hosts on an **anisotropic real sky** (in-catalog hosts keep the GLADE
catalog sky; dark hosts concentrate in low-completeness / Zone-of-Avoidance pixels) and SNR-selects
them through the **sky-dependent LISA TDI response** (`SNR = sqrt(<h|h>)`,
`parameter_estimation.py:455`, via `ResponseWrapper`, `waveform_generator.py:56-71`). The
**inference selection function** instead evaluates an **isotropic, sky-marginalized** `p_det`
(`phi = theta = 0`) in all three selection integrals — `D(h)`
(`bayesian_statistics.py:246-248`), `beta_Gbar(h)` (`:350-372`), and the global catalog denominator
(`:473-474`). Hence `<p_det>_iso != <p_det>_population`, `D(h)` is mis-shaped in the H0-carrying
variable `d_L(z,h)`, and H0 is biased. Mandel–Farr–Gair (2019, arXiv:1809.02063, Eq. 6): the
selection normalization MUST marginalize `p_det` over the **same** population measure the numerator
/ generator uses. The code even flags this itself at `bayesian_statistics.py:365-366`
("*Valid because p_det is sky-uniform; if real LISA sky dependence is restored this must become
`sum_k f_k p_det(z,Omega_k)`*"). **This is a genuine self-consistency / regime-of-validity failure
of the current paper, correctly rated PAPER-BLOCKER — but its numerical H0 impact is bounded small
(see §7).**

---

## 1. Recommended fix — one line

Build a **sky-resolved detection probability** `p_det(d_L, [M_z], Omega)` as an **empirical
ecliptic-latitude-band survival function** re-binned from the **existing isotropic injection pool**
(no new simulation), and wire it — as the **same shared object**, with the **same pixel grid and
quadrature** — into **all three** selection integrals plus the global catalog sum. Verified to
recover the current code exactly in the isotropic limit; expected H0 shift `<~1%`,
sign-indeterminate.

**Route selection (verification-mandated):**

- **Route A — empirical per-latitude survival `S_band(d_L, M_z) = P(d_hor >= d_L | beta-band)`.
  RECOMMENDED. This is the ONLY route free of a separation assumption** — it bins the *real*
  `fastlisaresponse` SNRs, so it carries the full sky × inclination × polarization × frequency
  covariance. Retains only the weak, literature-supported assumption of azimuthal symmetry
  `R = R(beta)` after multi-year averaging.
- **Route B — deconvolution for an analytic `S0`, then `S0(d_L / R(beta))`.** Cross-check only.
- **Route C — direct substitution `p_det_iso(d_L / R(beta))`.** Cross-check only.

> **CORRECTION (both verification lenses):** Do **NOT** present
> `p_det(d_L, M_z | Omega) = S0(d_L / R(beta))` (or Route C `p_det_iso(d_L / R(beta))`) as
> **EXACT** or "**sharpened**". The full-TDI sky×inclination×polarization coupling breaks the
> pointwise factorization `SNR(Omega, xi) = R(Omega)·omega(xi)`: because `g+ = (1+cos^2 i)/2 != gx =
> cos i` away from face-on, the *shape* of the extrinsic-factor distribution is itself
> `Omega`-dependent, not merely rescaled. **Only the `1/d_L` amplitude scaling is exact**
> (Hogg 1999, arXiv:astro-ph/9905116 Eq. 16). Routes B/C carry an `O(Var[R])` **plus**
> shape-change **plus** near-`f* ~ 19 mHz` frequency error and are demoted to cross-checks with
> explicitly stated approximation error.
>
> **CORRECTION (reference attribution):** Finn–Chernoff (1993, arXiv:gr-qc/9301003), Finn (1996),
> and "Quick recipes" (arXiv:2404.16930) justify a **single joint** projection factor `Theta`/`omega`
> marginalized over **all** extrinsic angles jointly — **not** the further split
> `omega = R(Omega)·omega0(orientation)`. arXiv:2404.16930 **explicitly excludes orbiting/LISA
> detectors**. Cite instead **Cutler (1998, arXiv:gr-qc/9703068)** and **arXiv:1201.3684** for the
> LISA orbit-averaged `R(beta)`, and make clear **Route A derives the sky dependence empirically**,
> not from any separation ansatz.

---

## 2. Symbols and frame

| Symbol | Meaning | Units |
|---|---|---|
| `d_L(z,h)` | luminosity distance | Gpc (code) |
| `M_z = M_source·(1+z)` | observer-frame MBH mass | `M_sun` |
| `Omega = (phi, theta)` | ecliptic sky direction (`BarycentricTrueEcliptic` J2000) | rad |
| `beta = pi/2 - theta` | ecliptic latitude; `beta = pi/2 - qS = pi/2 - theta_pixel` | rad |
| `SNR = sqrt(<h\|h>)` | matched-filter SNR (`parameter_estimation.py:455`) | — |
| `threshold` | `SNR_THRESHOLD = 20` (`constants.py`) | — |
| `d_hor = SNR·d_L/threshold` | h-invariant detection horizon (`sim_det_prob.py:262`) | Gpc |
| `f_k(z,h) ∈ [0,1]` | pixel-k completeness (`pixel_completeness.py:233`) | — |
| `f_bar = (1/Npix) Σ_k f_k` | sky-averaged completeness (`:287`, verified) | — |
| `Npix` | `12288` (NSIDE=32, equal area) | — |
| `p_pop(z)` | `dVc/dz · 1/(1+z)` (mass-integrated rate cancels in ratios) | Mpc³/sr |

**Frame fact (verified):** galaxy `THETA_S` becomes `pi/2 - beta` (colatitude) after
`_map_angles_to_spherical_coordinates`, `handler.py:905-907`; `ang2pix` uses `lat = pi/2 - theta`,
`pixel_completeness.py:308`; the response uses `is_ecliptic_latitude=False` with `qS` = ecliptic
colatitude, `waveform_generator.py:61-64`. All three share the ecliptic frame, so
`beta = pi/2 - qS = pi/2 - theta_pixel` is exact.

---

## 3. Per-change protocol

Each change below gives: **(1)** old formula + `file:line`, **(2)** new formula,
**(3)** reference, **(4)** dimensional analysis, **(5)** isotropic-sky limiting case.

### CHANGE 1 — Sky-resolved `p_det` estimator (the enabling object)

**File:** `bayesian_inference/simulation_detection_probability.py`

**(1) OLD.** Horizon-survival, sky discarded:
`p_det(d_L, M_z) = P(d_hor >= d_L)` with `d_hor = SNR·d_L/threshold` (`:262`), built from the
pooled isotropic injections. Accessors explicitly marginalize the sky:
`detection_probability_with_bh_mass_interpolated(..., phi, theta, h)` — "*Sky angles (phi, theta)
are accepted for API compatibility but are marginalized over internally (D-02)*" (`:776-777`); same
for `detection_probability_without_bh_mass_interpolated_zero_fill` (`:929-930`). `required_cols`
(`:240`) is `{z, M, SNR, h_inj, luminosity_distance}` — `phiS`, `qS` are **present in the CSV**
(`main.py:602`, written `:787-788`) **but not extracted**.

**(2) NEW.** Add an ecliptic-latitude axis. For each injection compute `beta = pi/2 - qS`; assign to
one of `Nband` **equal-|sin beta| (equal-solid-angle)** bands; build **one sorted-horizon survival
per band** via the existing `searchsorted` path (`:271-276`, `_build_grid_1d/2d`):

```
p_det(d_L, [M_z] | Omega) = S_{band(beta)}(d_L, [M_z]) = P(d_hor >= d_L | band(beta))     [Route A]
```

Accessors map `(phi, theta) -> beta -> band` and return that band's survival; interpolate **linearly
in sin beta** across band centres to avoid step artefacts. Keep the existing pooled (isotropic) grid
as the regression fallback.

**(3) REFERENCE.** Empirical CCDF of the detection horizon per sky band — the survival construction
of Finn–Chernoff (1993, arXiv:gr-qc/9301003) applied per latitude band; azimuthal symmetry
`R = R(beta)` from Cutler (1998, arXiv:gr-qc/9703068) and arXiv:1201.3684; `1/d_L` exactness from
Hogg (1999, arXiv:astro-ph/9905116, Eq. 16). **No separation ansatz** (Route A is empirical).

**(4) DIMENSIONAL ANALYSIS.** `beta` [rad], bands dimensionless. `S_band: [Gpc] -> [0,1]`, identical
signature/units to the current 1D/2D survival. `d_hor` [Gpc] unchanged. No new dimensional factor.

**(5) ISOTROPIC LIMIT.** If SNR is `beta`-independent, every band's sorted-horizon set equals the
pooled set, so `S_band = S_iso` for all bands and the accessor returns the current marginalized
`p_det` bit-for-bit. ✓

---

### CHANGE 2 — Full selection `D(h)`

**File:** `bayesian_inference/bayesian_statistics.py`, `precompute_completion_denominator`

**(1) OLD** (`:246-248`, inside `_denom_integrand`):
```python
phi = np.zeros_like(z)   # marginalized; value does not matter
theta = np.zeros_like(z)
p_det = ..._without_bh_mass_interpolated_zero_fill(d_L, phi, theta, h=_h)
```
i.e. `D(h) = INTEGRAL <p_det>_iso(d_L(z,h)) · dVc/(1+z) dz` (`:239-262`).

**(2) NEW** (equal-area pixel sum, `Omega_k` = pixel centres):
```
D(h) = INTEGRAL  (1/Npix) Σ_k p_det(d_L(z,h), Omega_k) · dVc/(1+z) dz
```

**(3) REFERENCE.** Gray, Gerosa et al. (2023, arXiv:2308.02281, Eq. 2.3) — GW selection as an
explicit **sum over pixels** with the per-pixel population prior; Gray–Messenger–Veitch (2022,
arXiv:2111.04629, Eqs. 4–5); Mandel–Farr–Gair (2019, arXiv:1809.02063, Eq. 6) self-consistency.

**(4) DIMENSIONAL ANALYSIS.** Swap a dimensionless scalar `<p_det>_iso` for a dimensionless sky
average `(1/Npix) Σ_k p_det(Omega_k)` (`Σ_k 1/Npix = 1`, equal-area weight). Integrand keeps
`[dimensionless]·(dVc/dz)[Mpc³/sr]·1/(1+z)`; `D(h)` stays **Mpc³/sr** (docstring `:230`). Inert. ✓

**(5) ISOTROPIC LIMIT.** `p_det(Omega_k) -> <p_det>_iso` ⇒ `(1/Npix) Σ_k <p_det>_iso = <p_det>_iso`
⇒ current `:246-260` verbatim. ✓

---

### CHANGE 3 — Missing-completion selection `beta_Gbar(h)`

**File:** `bayesian_inference/bayesian_statistics.py`, `precompute_missing_completion_denominator`

**(1) OLD** (`:350-372`, `_missing_denom_integrand`): sky-uniform, factorizes `<(1-f)·p_det>_Omega`
into `<1-f>_Omega · <p_det>_Omega`:
```
beta_Gbar(h) = INTEGRAL (1 - f_bar(z,h)) · <p_det>_iso(d_L(z,h)) · dVc/(1+z) dz
```
(uses `completeness.f_bar`, `:367-368`; the code's own caveat is at `:365-366`).

**(2) NEW** (exactly the caveat's prescription — per-pixel `Σ_k (1-f_k) p_det(Omega_k)`):
```
beta_Gbar(h) = INTEGRAL  (1/Npix) Σ_k (1 - f_k(z,h)) · p_det(d_L(z,h), Omega_k) · dVc/(1+z) dz
```
ZoA / empty pixels (`f_k = 0`) contribute the **full** `p_det(Omega_k)` — this is exactly where dark
hosts concentrate (audit R1). Use per-pixel `f_k` (`pixel_completeness.py:233`) and pixel centres
`Omega_k`.

**(3) REFERENCE.** Gray et al. (2020, arXiv:1908.06050, Eq. 33) out-of-catalog denominator;
Gray–Messenger–Veitch (2022, arXiv:2111.04629, Eq. 5) per-pixel in/out split;
the code caveat `bayesian_statistics.py:365-366`.

**(4) DIMENSIONAL ANALYSIS.** `f_k`, `p_det`, `1/Npix` dimensionless; integrand and result stay
**Mpc³/sr**, identical to Change 2. Inert. ✓

**(5) ISOTROPIC LIMIT.** `p_det(Omega_k) -> <p_det>_iso` pulls out; `(1/Npix) Σ_k (1-f_k) =
1 - f_bar` (verified identity `pixel_completeness.py:287`) ⇒ current `:367-372` verbatim. ✓

---

### CHANGE 4 — Global in-catalog denominator (real galaxy sky)

**File:** `bayesian_inference/bayesian_statistics.py`, `precompute_global_catalog_selection`

**(1) OLD** (`:473-474`): the loop has each galaxy's REAL ecliptic sky available
(`INternalCatalogColumns.PHI_S/THETA_S`, `handler.py:171-172`, ecliptic after COORD-03), yet sets
```python
phi = np.zeros_like(z_g)   # sky-marginalized, matching D(h)
theta = np.zeros_like(z_g)
```
⇒ `Sigma_global(h) = Σ_g w_g · <p_det>_iso(d_L_g)` (`:490`), and the with-BH branch (`:475-482`)
likewise.

**(2) NEW** (each galaxy's real `Omega_g`):
```
Sigma_global(h) = Σ_{g: z_g < z_max(h)} w_g · p_det(d_L(z_g,h), [M_z,g], Omega_g)
```
Extract `phi_g = REDUCED[PHI_S][eligible]`, `theta_g = REDUCED[THETA_S][eligible]` at `:441-449`,
pass at `:479` (4D) and `:485-487` (3D). `w_g = R_eff_per_mbh(M_g)/(1+z_g)` and the `z < z_max(h)`
eligibility (`:457-471`) **unchanged**. The catalog galaxies ARE the Monte-Carlo sky sampling of the
in-catalog channel (they trace LSS), so this is the correct MC estimator of
`beta_G = INTEGRAL f·p_det dVc/(1+z)`.

**(3) REFERENCE.** Gray et al. (2020, arXiv:1908.06050, §II, Eq. 8) antenna response "*varies over
sky position and polarization*"; Gray et al. (2023, arXiv:2308.02281, Eq. 2.3).

**(4) DIMENSIONAL ANALYSIS.** `w_g` [rate/(1+z)] unchanged, `p_det` dimensionless — same units as the
current `:490` (`R_eff` constants cancel in the `L_cat = Sigma_local/Sigma_global` ratio). Inert. ✓

**(5) ISOTROPIC LIMIT.** `p_det(Omega_g) -> <p_det>_iso` ⇒ `Σ_g w_g <p_det>_iso` = current
`:484-490`. ✓

---

### CHANGE 5 (RESOLVED — **OUT OF SCOPE**) — `B_num` per-event numerator

**File:** `bayesian_inference/bayesian_statistics.py:1444-1480`

Angle A proposed feeding the event-pixel `beta` into `p_det` here. **The consistency lens overrides
this: `B_num` contains the GW likelihood `p_GW` (`_mvn_pdf`, `:1456-1461`), NOT `p_det`**, and it
**already** evaluates `f_k` at the event's real sky pixel (`_event_pixel = ang2pix(detection.phi,
detection.theta)`, `:1444`). It is already sky-aware and needs **no `p_det` change**. Left unchanged.
(Recorded here only to close the A-vs-B discrepancy.)

---

### CHANGE 6 (support, **not** a physics change) — pixel-centre helper

**File:** `galaxy_catalogue/pixel_completeness.py`

Add `pixel_centers() -> (phi_k, theta_k)` via `self._healpix.healpix_to_lonlat(np.arange(npix))`
(default `dx=dy=0.5` = centres), `theta_k = pi/2 - lat`. Pure geometry (mirrors
`sample_sky_in_pixels`, `:315-329`); **no computed physical value changes**, so this file's edit is a
software change (the physics-change protocol still applies to Changes 2–4 in the same commit).

---

## 4. How `R(Omega)` / the sky axis is computed (and caching)

**Primary (Route A):** the sky dependence is **measured**, not modelled. Re-bin the existing
`_d_hor` array (`sim_det_prob.py:262`) by `beta = pi/2 - qS` into `Nband` equal-|sin beta| bands and
build one survival per band. **Verified statistics:** 504,000 injections across 560 CSVs
(`simulations/injections/*.csv`); at 6 equal-|sin beta| bands each holds ~83.8k–84.2k injections
(ample). A full NSIDE=32 sky-resolved survival would thin to ~41 inj/pixel (**too noisy** — do NOT
do it); use `beta`-bands for the response and keep `f_k` at NSIDE=32 for catalog structure.

**Cross-check only:** compute an analytic `R(beta)` **once** from the LISA antenna patterns
(`LISA_configuration.py` F+, Fx, orbit-averaged; Cutler 1998 / arXiv:1201.3684), cache it like the
`m_th` map (a small `.npy`), and **assert** the empirical per-band rms matches within tolerance. The
analytic `R(beta)` is also the **fallback** for any under-populated polar band. It is **never** the
production `p_det`.

**Caching:** the per-band survival interpolators are built once in `SimulationDetectionProbability`
(same lifecycle as today's shared grid); the analytic `R(beta)` table is a small cached array. No
per-MCMC-step cost.

---

## 5. Injection campaign / re-sim — **NOT required** for the recommended path

- **Route A `p_det`:** **NO new campaign.** `phiS`, `qS` already exist in the injection CSVs
  (`main.py:602`); the estimator merely discards them (`sim_det_prob.py:240`). Fix = add `qS` to
  `required_cols` + re-bin existing arrays. **CPU-only post-processing** on the dev machine.
- **Closure test:** **NO new campaign mandatory.** The generator already produces anisotropic-sky
  detected events (real catalog sky + ZoA dark draw), and the CRB CSVs already retain `qS`, `phiS`
  (`main.py:1098-1103`). The closure test changes only the **inference** side (sky-aware selection),
  reusing existing detected events + existing isotropic injection pool.
- **When a re-sim WOULD be needed (NOT recommended):** a full NSIDE-resolved `p_det` or a
  well-sampled 4D `sky × M_z` grid, or the alternative of re-drawing injection sky from the catalog
  (which also introduces mild circularity). Cost: `~Npix × N_per_bin` GPU-SNR evaluations
  (GPU-hours on the cluster). Azimuthal symmetry collapses the problem to 1-D in `beta`, making this
  unnecessary.

---

## 6. Closure / validation tests to add

**T1 — Isotropic-limit regression (MANDATORY, machine precision).** Feed a **constant** `p_det` and
assert:
`D_fixed == D_iso`, `beta_Gbar_fixed == beta_Gbar_current`, `Sigma_global_fixed ==
Sigma_global_current` to machine precision. Protects the already-reconciled redshift / rate / mass /
completeness axes (audit §2) from any global-scale shift.

**T2 — Partition identity (MANDATORY).** Assert `D(h) == beta_G(h) + beta_Gbar(h)` numerically
post-fix (holds per (pixel, z) node since `f_k + (1-f_k) = 1`), with `beta_G := D - beta_Gbar`.

**T3 — Sky-marginal invariance (MANDATORY).** Assert `<p_det(·|Omega)>_iso == legacy p_det_iso`
(equivalently `<R^2>_iso = 1`). Guaranteed **only if** the inference `pixel -> band` grouping uses
the **identical** equal-|sin beta| band edges as the injection `p_det` build.

**T4 — North–south (|beta|) symmetry (verify, do NOT assume).** Empirically check the survival is
symmetric under `beta -> -beta` from the injections before folding to `|beta|`.

**T5 — Band-count convergence.** Confirm the sky-resolved `p_det` (and the resulting `D`,
`beta_Gbar`) are stable as `Nband` = 4 -> 6 -> 8. Cap `Nband <= 8` against injection Poisson noise
(~0.3% per-band). Flag under-populated polar bands (analogous to the existing `n_total >= 10` check,
`sim_det_prob.py:710`).

**T6 — Anisotropic closure (MANDATORY acceptance criterion, + negative control).**
The existing Change-5 closure test **cannot** witness this fix (both sides share the marginalized
`p_det`; audit §5, line 82). Add:
- **Positive:** inject anisotropic (catalog + ZoA) sky **and** infer with the sky-aware `p_det`;
  assert the H0 residual is within the claimed `<~1%` (within MC error) — i.e. the correction
  **removes** bias without injecting a larger stochastic error.
- **Negative control:** inject anisotropic sky but infer with the **old isotropic** `p_det`; the H0
  residual should reproduce the (small) bias the fix is meant to remove — demonstrating the test
  actually has discriminating power on the sky axis.

**T7 — Frame assertion.** Assert the `fastlisaresponse` ecliptic **pole** coincides with
`BarycentricTrueEcliptic(J2000)` (only the pole matters under azimuthal symmetry; longitude/equinox
offset is immaterial because latitude is invariant under azimuthal rotation). Add a startup
shape/frame/md5 check on `m_th_map_nside32.npy` (audit R4).

---

## 7. Expected magnitude of the H0 shift (Angle C, from the pipeline's own frozen artifacts)

Measured from 504,000 isotropic injections + the frozen `m_th_map_nside32.npy` (12,288 px, ecliptic,
6.11% ZoA):

- **Response modulation in ecliptic latitude:** mean response `<R>(beta)` ~3% rms (peak/trough
  1.08–1.09); **detection fraction** `P(SNR>=20)(beta)` **~8% rms**, max/min = **1.23**, enhanced at
  `|beta| < 40°` (near the ecliptic plane), suppressed at `|beta| > 56°` (poles). Consistent with the
  Cutler-1998 LISA pattern (best near plane / normal ±30°, worst at poles).
- **Population/completeness anisotropy in the ECLIPTIC frame:** essentially **uncorrelated** with
  ecliptic latitude — `corr(|ecl.lat|, is_ZoA) = -0.003`, `corr(|ecl.lat|, m_th_depth) = -0.021`;
  mean `|ecl.lat|` of ZoA pixels 32.4° ≈ finite 32.7° ≈ isotropic 32.7°. **Physical reason:** the
  GLADE ZoA is **Galactic**-plane-aligned, inclined ~60° to the ecliptic, so it projects nearly
  uniformly onto ecliptic latitude — the very frame where the LISA response modulation lives.

⇒ The bias-driving covariance `Cov_Omega[p_det, p_sky^true] ~ (<=8% rms) × (~0 correlation) ~ 0`.

**Verdict:** **H0 shift bounded `<~1%` (plausibly sub-percent), sign-indeterminate.** This matches
the field norm (Gray 2020 arXiv:1908.06050; Gray–Messenger–Veitch 2022 arXiv:2111.04629 — who
**explicitly** justify sky-uniform `p_det` by this same smallness; Laghi et al. 2021
arXiv:2102.01708 — LISA-EMRI selection treated as a ~1% correction).

> **CORRECTION (both lenses):** This is a **measured bound for THIS catalog realization and injection
> set**, not a theorem, and it is **comparable in size to the effect being fixed**. Report it as a
> **bounded systematic in the H0 error budget**, not a dismissal. Frame the fix as **rigor /
> self-consistency closure** (MFG Eq. 6) that lets the closure test witness the sky axis — **do NOT
> overclaim a large de-biasing**. It remains a legitimate paper-blocker for **formal correctness**.

---

## 8. Ordered implementation plan

1. **`pixel_completeness.py`** — add `pixel_centers()` (Change 6, pure geometry). *(software)*
2. **`simulation_detection_probability.py`** — add `qS` to `required_cols` (`:240`) and extract it
   (`:247-255`); compute `beta`, assign equal-|sin beta| bands (`Nband` config, default 6); build a
   per-band survival (reuse `_survival_at`/`searchsorted`, `:271-276`; add band as an outer index to
   `_build_grid_1d/2d`, `:506-663`); make the two accessors (`:751-819`, `:900-1003`) map
   `(phi,theta) -> beta -> band` with linear-in-`sin beta` interpolation; keep the pooled grid as
   the isotropic fallback. *(PHYSICS — Change 1)*
3. **`bayesian_statistics.py`** — convert `D(h)` (`:239-262`) and `beta_Gbar(h)` (`:343-372`)
   **JOINTLY** to the per-pixel sum using the **same** `p_det(Omega_k)` object, the **same** pixel
   grid, the **same** quadrature order (`_DH_QUAD_ORDER`) and `z_max(h)`; compute `beta_G := D -
   beta_Gbar`. *(PHYSICS — Changes 2 & 3, atomic)*
4. **`bayesian_statistics.py`** — `precompute_global_catalog_selection` (`:441-490`): extract real
   `phi_g/theta_g`, pass into the 3D and 4D accessors. *(PHYSICS — Change 4)*
5. **Cross-check artefact:** analytic `R(beta)` table (cached `.npy`), assert vs empirical bands.
6. **Tests T1–T7.** T1 (isotropic regression) and T2 (partition) must pass before anything else.
7. **Physics-change protocol at edit time:** present old/new formula + this derivation + dimensional
   analysis + isotropic limit for each of Changes 1–4; add reference comments above each changed
   line (Cutler 1998; Gray 2023 Eq. 2.3; GMV 2022 Eq. 5; MFG 2019 Eq. 6); commit with `[PHYSICS]`.

**Guardrails folded in (verification `required_corrections`):**
- Antenna anisotropy lives **only** in `p_det(Omega)`; `f_k(z,Omega)` stays pure EM completeness; sky
  prior `p(Omega) = 1/Npix` stays **uniform** — **no double-counting**. Do NOT reweight pixels by
  galaxy counts inside `p_det`.
- The **same** `p_det(Omega)` object must serve `D`, `beta_Gbar`, `beta_G`, **and** the global sum —
  a divergence between the discrete catalog sum (`:490`) and the continuum `beta_G` reintroduces the
  exact bias being removed.
- **No circularity:** Route A `p_det` is built from **isotropic** injections (response geometry
  only, no catalog contents). The rejected re-draw-from-catalog alternative WOULD be circular.
- **4-yr vs 5-yr flag (pre-existing):** the confusion-noise PSD uses `t_obs_years = 4.0`
  (`LISA_configuration.py:67`) while the signal/SNR integrates over `T = 5 yr`
  (`parameter_estimation.py:88`). Both `> 1 yr`, so azimuthal symmetry holds; the annual-averaging
  argument should reference the **5-yr** signal-integration time. The 4-vs-5 mismatch is a **separate
  item to flag**, not fixed here.

---

## 9. Open decisions the user MUST approve

1. **Route choice.** Approve **Route A** (empirical band survival) as production, with analytic
   `R(beta)` as cross-check/polar fallback only? (Recommended.) Routes B/C rejected as production
   because their exactness claim is invalid.
2. **Band resolution.** `Nband = 6` equal-|sin beta| bands (verified ~84k inj/band)? Fold to `|beta|`
   after T4 confirms N–S symmetry, or keep signed `beta`?
3. **With-BH-mass 4D branch** (`bayesian_statistics.py:475-482`). The 4D `sky × M_z` survival is
   **statistics-starved**. Approve **keeping it isotropic and explicitly FLAGGING** the residual in
   the with-BH-mass posterior (rather than trusting a noisy coarse 4D sky grid)? (Recommended.) This
   affects only the with-BH-mass posterior, not the primary result; it compounds the pre-existing
   3D/4D mixing item (audit R2, orthogonal to this fix).
4. **Closure-test data.** Reuse existing anisotropic detected events for T6, or generate a small
   controlled mock catalog for a cleaner positive/negative control? (Reuse is sufficient and needs no
   new GPU time; a controlled mock is optional polish.)
5. **Scope of the paper claim.** Approve reporting the sky-selection effect as a **bounded (`<~1%`),
   sign-indeterminate systematic** in the H0 budget for **this** catalog/injection realization
   (with the T6 residual shown), rather than as a large de-biasing?
6. **Deferral option.** If the fix is deferred, approve folding the measured `<~1%` bound into the
   reported systematics (it may **not** simply be dropped, per both lenses).

---

## 10. Reference ledger (fetched this session, not asserted from memory)

- **Mandel, Farr & Gair (2019)** arXiv:1809.02063, Eq. 6 — selection normalization over the true
  population measure (the self-consistency requirement the code violates).
- **Gray, Gerosa et al. (2023)** arXiv:2308.02281, Eq. 2.3 — GW selection as an explicit per-pixel
  sum (the template for Changes 2–4).
- **Gray, Messenger & Veitch (2022)** arXiv:2111.04629, Eqs. 4–5 — pixelated `f_k`, `f_bar`, in/out
  split; explicitly justifies sky-uniform `p_det` by run-averaged smallness.
- **Gray et al. (2020)** arXiv:1908.06050, Eqs. 8, 32, 33 — antenna response varies over sky /
  polarization; selection denominator and out-of-catalog `(1-f)` weight.
- **Cutler (1998)** arXiv:gr-qc/9703068 — LISA antenna patterns with annual orbital motion; sensitive
  near the ecliptic plane, weakest at the poles → basis for `R(beta)` (correct citation for the sky
  dependence).
- **arXiv:1201.3684** — closed-form orbit/orientation-averaged LISA response vs source sky angles
  (analytic route to `R(beta)`).
- **Robson, Cornish & Liu (2019)** arXiv:1803.01944 (cited `LISA_configuration.py:63`) —
  sky/pol/inclination-averaged response factor; anchors `<R^2>_iso = 1`.
- **Laghi et al. (2021)** arXiv:2102.01708 — LISA MBHB/EMRI dark-siren H0; sky-marginalized selection
  as a ~1% correction (magnitude corroboration).
- **Hogg (1999)** arXiv:astro-ph/9905116, Eq. 16 — `h ∝ 1/d_L` (exactness of the amplitude scaling).
- **Finn & Chernoff (1993)** arXiv:gr-qc/9301003; **Finn (1996)** arXiv:gr-qc/9601048 — single joint
  projection factor `Theta` survival (NOT the `R(Omega)·omega0` split; see §1 correction).
- **"Quick recipes for GW selection effects"** arXiv:2404.16930 — CCDF of the projection factor;
  **explicitly excludes orbiting/LISA detectors** (cannot justify `R(Omega)` for LISA).
