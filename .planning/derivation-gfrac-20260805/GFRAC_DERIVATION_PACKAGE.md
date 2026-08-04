# `g_frac(h)` — derivation package: is the completion-leg mass factor's h-slope physics or defect?

**Date:** 2026-08-05 · **Author of record:** Jasper Seehofer (rules via `/physics-change`;
this document derives and proposes, it decides nothing).
**Question of record** (set by `results/run_20260804_postfix/gate_vii/PREREGISTRATION_FROZEN_GFRAC.md`,
verdict CONFIRM both venues): the h-slope of
`g_frac = B_num_wbh/B_num` carries the entire residual 2D high-h displacement
(2D MAP 0.780/0.800 → 0.660/0.640 frozen; live == CSV proxy exactly). **Is that
h-slope correct physics, or a measure/normalisation defect?**

**Code base:** `main` @ `87bc7771` (+ the frozen-g instrumentation commits
`930a9484`, `c917ed87`). All line numbers below are
`master_thesis_code/bayesian_inference/bayesian_statistics.py` unless prefixed.

---

## VERDICT CANDIDATE

> **(i) CORRECT PHYSICS**, with two named residuals that are *not* about `g`.
>
> The h-slope of `g_i(z;h)` is the mass–redshift (spectral-siren) term of a
> correctly constructed MFG completion numerator. It is **not** a Jacobian or
> measure asymmetry: the x_M measure enters the 2D numerator only, as it must,
> and the matching normalisation must **not** carry it — the completion leg's
> measure invariance is not merely analogous to the catalogue leg's, it is
> *stronger* (exact, algebraic, and h-independent). The `(1+z)` mass Jacobian is
> present exactly once, at `:1989`/`:1994`, and it does not cancel — correctly,
> because it is a real coordinate change between `dM` and `dM_z`.
>
> The slope's **magnitude is reproduced in closed form to 0.1 %** by
> `dln g_i/dh = (−s_dex) · dln(1+z*(d_L,h))/dh` with `s_dex = −0.43` the
> log-slope of φ per dex (§6.3), and it is within **8 %** of the tilt the *true*
> event-generating population would produce — **in the adverse direction** (the
> true population wants a *larger* slope, §6.5). Neither φ's band edges (§6.4)
> nor any band-pass are active.
>
> The residual 2D displacement is therefore the algebraic sum of a genuine
> +243.5 nats/h population term and the **same** negative bias that rails the 1D
> channel at 0.600. The arithmetic closes (§7). Deciding "is 0.78 wrong" is
> therefore not a question about `g` — it is the 1D-rail question
> (`[[h0-railing-rootcause-photoz]]`).

**Two findings that are new, sharp, and NOT `g`'s fault** — the author should
rule on them separately:

* **N-1 (measurement, no formula):** gate (i) — the 2D measure-invariance proof
  — is **near-vacuous as evidence**. The 2D catalogue leg carries a median
  share of **0.000** of the mixture and is *identically zero for 81.5 % of
  events* (iiib) / 61.8 % (joint r1); mean share 5.4 % / 5.7 % (§6.6).
  `dMAP/dlnC = 0.0` was measured on a term that is zero for four events in five.
  The measure consistency *between* the two 2D legs remains untested by
  measurement (it is proven algebraically here, §3.3, which is why this is a
  provenance finding and not a physics one).
* **N-2 (candidate defect, 1D-side, DIFFERENT object):** the 1D completion
  numerator `B_num` is the marginal of the 2D numerator over **all** `M_z^obs`
  (§5, limiting case c — exact). That is the correct marginal *only if the
  detection cut does not act in the discarded coordinate*. It does: `S_4D`
  depends on `M_z`. The MFG-correct 1D marginal restricts the `M_z^obs`
  integral to the detectable region, which introduces `S̄_φ(z;h)` **inside**
  the 1D quadrature. This is a candidate defect of `B_num`, moves the **1D**
  channel, and leaves the 2D channel untouched. It is raised, not proposed —
  see §9.

---

## 1. The implemented object (exact expressions, file:line)

### 1.1 The 1D completion numerator

`completion_numerator_integrand`, `:4055-4096`; assembled `:4186-4193`:

```python
p_gw = norm.pdf(d_L_fraction, loc=_comp_mean_dLfrac, scale=_comp_sigma_dLfrac) \
       * np.sin(self.detection.theta) / (4.0 * np.pi)          # :4073-4077
...
return (1.0 - f_z) * p_gw * dVc / (1.0 + z)                     # :4096
```

with `d_L_fraction = dist_vectorized(z, h)/_comp_det_d_L` (`:4059-4062`),
`_comp_sigma_dLfrac = sqrt(inv(cov_inv_3d)[2,2])` (`:4044-4045`),
`_comp_mean_dLfrac = _comp_mean_3d[2]` (`= 1`, `:4046`).

```
B_num(h) = ∫_{z_lo(h)}^{z_hi(h)} (1 − f_k(z;h)) · p_gw(x_dL(z;h)) · dV_c/dz /(1+z) dz
```

### 1.2 The 2D companion (N8)

`completion_numerator_integrand_with_bh_mass`, `:4118-4147`:

```python
base = completion_numerator_integrand(z, h_eval)                 # :4122
g_i  = completion_mass_factor_g(z, d_L_mass/_comp_det_d_L,
                                _g_det_M_z, _g_proj, _g_sigma)   # :4130-4136
return base * g_i                                                # :4147
```

`completion_mass_factor_g`, `:1935-1995`, body:

```python
x_nodes, x_weights = roots_hermite(n_hermite)                    # :1987  (n=64)
scale   = det_M_z / (1.0 + z_nodes)                              # :1989   dM/dx_M
mu_cond = 1.0 + proj_d_L_to_M * (d_L_fraction - 1.0)             # :1990
x_M     = mu_cond[:,None] + sqrt(2)*sigma_cond_M*x_nodes[None,:] # :1992
M_source = x_M * scale[:,None]                                   # :1993   M = x_M M_z,det/(1+z)
phi_x    = dark_mass_density_per_mass(M_source) * scale[:,None]  # :1994   φ_x = φ(M)·dM/dx_M
return (phi_x @ x_weights) / sqrt(pi)                            # :1995
```

Kernel scalars, `:3255-3263`:
`proj_d_L_to_M = cov_4d[2,3]/cov_4d[2,2]`,
`sigma_cond_M = sqrt(cov_4d[3,3] − cov_4d[2,3]²/cov_4d[2,2])`.

φ, `:1750-1779` (`dark_mass_density_per_mass`), normalised on
`[M_SOURCE_FRAME_MIN, M_SOURCE_FRAME_MAX] = [1e4, 1e7] M_sun`
(`constants.py:125-126`), hard-zero off-band (`:1776-1779`); the density itself
is `dark_siren_injection.dark_mass_log10_density_unnormalised` (`:332-365`) =
`mbh_mass_function × R_eff_per_mbh` — the **same** function
`_draw_dark_masses` (`:368-392`) samples the injected dark hosts from.

```
B_num_wbh(h) = ∫ (1 − f_k) · p_gw(x_dL(z;h)) · dV_c/(1+z) · g_i(z;h) dz      (:4194-4205)
g_frac       = B_num_wbh / B_num                                             (:4213)
```

### 1.3 Where it enters the posterior (path A, `absolute_marginal`)

`:4296-4334`:

```python
B_scale       = beta_Gbar_phi / beta_Gbar                        # :4324
B_num_phi     = B_num     * B_scale                              # :4325
B_num_wbh_phi = B_num_wbh * B_scale                              # :4326
combined_without_bh_mass = (beta_G_phi  * L_cat_no_bh   + B_num_phi)     / D_tilde_phi  # :4330
combined_with_bh_mass    = (alpha_G_phi * L_cat_with_bh + B_num_wbh_phi) / D_tilde_phi  # :4333
```

with `D̃^φ = α_G^φ + β_Ḡ^φ` (`path_a_mixture_objects`, `:1998-2069`) **shared
by both channels**, `β_G^φ`/`β_Ḡ^φ` from `precompute_phi_selection_integrals`
(`:1916-1924`) built on `S̄_φ(z;h) = ∫φ(log₁₀M) S_4D(d_L(z;h), M(1+z)) dlog₁₀M`
(`precompute_phi_marginal_survival`, `:1782-1830`).

---

## 2. The first-principles target

Population (hybrid model, `GATE_PACKAGE_FINAL.md` Appendix A), dark leg:

```
n_Ḡ(z, Ω, M | h) = (1 − f(z,Ω)) · w_pop(z;h) · φ(M) / 4π ,     ∫φ(M) dM = 1
```

MFG (arXiv:1809.02063 Eqs. 5–7) with data `d`, parameters `θ = (z, Ω, M)`:

```
p(d | det, h) = [ ∫dθ L(d|θ) n(θ|h) ] / α(h) ,   α(h) = ∫dθ P(det|θ) n(θ|h)
```

**2D data** `d^{2D} = (Ω^obs, d_L^obs, M_z^obs)`; the Fisher likelihood is the
4-D Gaussian, whose isotropic-sky marginal is the 2-D Gaussian
`N₂(d_L^obs, M_z^obs; d_L(z;h), M(1+z), Σ_{(dL,Mz)})`. Hence, dark leg:

```
B^{2D}(d_L^obs, M_z^obs; h)
  = ∫dz (1−f_k(z)) (dV_c/dz)/(1+z) ∫dM φ(M) N₂( (d_L^obs, M_z^obs) ; (d_L(z;h), M(1+z)) )   (T1)
```

with matching normalisation

```
α_Ḡ(h) = ∫dz (1−f̄) w_pop(z;h) ∫dM φ(M) S_4D(d_L(z;h), M(1+z))
       = ∫dz (1−f̄) w_pop(z;h) S̄_φ(z;h)  =  β_Ḡ^φ(h)                                        (T2)
```

**1D data** `d^{1D} = (Ω^obs, d_L^obs)`: marginalise (T1) over the discarded
`M_z^obs`; `∫dM_z^obs N₂ = N₁` and `∫dM φ(M) = 1`, so

```
B^{1D}(d_L^obs; h) = ∫dz (1−f_k) (dV_c/dz)/(1+z) N₁(d_L^obs; d_L(z;h))                       (T3)
```

with the **same** α (T2) — α is a property of the population and the detector,
not of which observables the analyst chooses to use.

---

## 3. Comparison: implemented vs target

### 3.1 The numerator is exactly (T1)

Write the code in physical units. `x_M ≡ M_z/M_z,det,i`, so
`M = x_M · M_z,det,i/(1+z)` and `dM/dx_M = M_z,det,i/(1+z)` — **exactly `scale`,
`:1989`**. Then `:1994` is

```
φ_x(x_M; z) = φ( x_M M_z,det/(1+z) ) · M_z,det/(1+z) = φ(M) · dM/dx_M
```

i.e. φ pushed forward from the `dM` measure to the `dx_M` measure. Line `:1995`
Gauss-Hermites `E_{x_M ~ N(μ_cond, σ_cond)}[φ_x]`, so

```
g_i(z;h) = ∫dx_M N(x_M; μ_cond(z;h), σ_cond) φ_x(x_M; z) = ∫dM φ(M) N(x_M(M,z); μ_cond, σ_cond)
```

and `base · g_i` (`:4147`) is

```
p_gw(x_dL) · ∫dM φ(M) N(x_M | x_dL) = ∫dM φ(M) · [ p_gw(x_dL) N(x_M|x_dL) ] = ∫dM φ(M) N₂(x_dL, x_M)
```

**This factorisation is exact only if `p_gw`'s scale is the *marginal* variance
of `x_dL` in the same 2-D block whose conditional supplies `(μ_cond, σ_cond)`.
Verified in code:** `cov_obs = cov_4d[:3,:3]` (`:3242`) and
`_cov_inv_3d[slot] = inv(cov_obs)`, so `inv(cov_inv_3d)[2,2] = cov_4d[2,2]`
(`:4044-4045`) — the same `_s_dd = cov_4d[2,2]` that `:3258-3263` conditions on.
✔ `base × N_cond` **is** the 2-D marginal joint, not an approximation.

⇒ `B_num_wbh` = (T1) up to the per-event, **h-independent** constant
`M_z,det,i` (the price of working in the fractional coordinate). ✔

### 3.2 The normalisation is exactly (T2)

`β_Ḡ^φ` (`:1916-1924`) contracts the **same** φ against the **same** `(1+z)`
mass lift inside `S̄_φ` (`:1796-1797`). ⇒ numerator and denominator both carry
`∫φ dM`; neither carries a dangling x_M density. ✔

### 3.3 The measure question, answered

The interpretation thread's sharp question was: *does the 2D channel's
normalisation carry the matching x_M measure, or is the h-dependent mass-density
weight in the numerator unmatched?*

**Answer: the normalisation must NOT carry an x_M measure, and it does not.**
`α(h)` is a pure number (`∫ p_det n dθ`), not a density in any data coordinate.
The x_M measure lives in the numerator alone because `B^{2D}` is a *density in
the data* `(d_L^obs, M_z^obs)` while `B^{1D}` is a density in `d_L^obs` only.
The dimensional difference between them is exactly `[x_M]⁻¹`, and its
coefficient is `M_z,det,i` — **data, hence h-independent, hence invisible to the
H₀ posterior**. Formally, for any constant `C`,

```
x_M → x_M/C  ⇒  φ_x → C φ_x  ⇒  B_num_wbh → C·B_num_wbh  ⇒  ln p_i^{2D} → ln p_i^{2D} + ln C
```

with `∂/∂h (ln C) = 0`. **Completion-leg measure invariance is therefore not
merely "analogous" to gate (i)'s catalogue-leg result — it is exact and
algebraic, with no measurement required.** The h-slope cannot be a measure
asymmetry.

### 3.4 The `B_scale` observation (raised, not a finding against `g`)

`B_scale = β_Ḡ^φ/β_Ḡ` (`:4324`) is applied identically to `B_num` and
`B_num_wbh` (`:4325-4326`). It is a *common* factor on the completion leg of
both channels, so it cancels out of `g_frac` and cannot contribute to the
question of record. (It does re-weight completion-vs-catalogue h-dependently;
that is ratified path-A scope, `FIXB_PATHA_PACKAGE.md` §3.2, and is out of scope
here.)

---

## 4. Dimensional analysis

| object | units | check |
|---|---|---|
| `p_gw` (`:4073`) | `[x_dL]⁻¹ · sr⁻¹` | `norm.pdf` in `x_dL`; `sinθ/4π` is the isotropic sky measure |
| `dVc/(1+z) dz` | comoving volume | shared with `D`, `β_Ḡ`, the event sampler |
| `1 − f_k` | dimensionless | `clip(...,0,1)` `:4088-4095` |
| **`B_num`** | `[x_dL]⁻¹ · V` | density in the 1D data |
| `φ(M)` (`:1778`) | `M_sun⁻¹` | `∫φ dM = 1` on `[1e4,1e7]` |
| `scale = M_z,det/(1+z)` (`:1989`) | `M_sun` per unit `x_M` | `dM/dx_M` |
| `φ_x = φ·scale` (`:1994`) | `[x_M]⁻¹` | pushforward ✔ |
| `N(x_M; μ_cond, σ_cond)` | `[x_M]⁻¹` | conditional Gaussian |
| **`g_i`** (`:1995`) | `[x_M]⁻¹` | `∫dx_M [x_M]⁻¹·[x_M]⁻¹ · [x_M] = [x_M]⁻¹` ✔ (docstring `:1979` agrees) |
| **`B_num_wbh`** | `[x_dL]⁻¹ [x_M]⁻¹ · V` | density in the 2D data ✔ |
| `g_frac` | `[x_M]⁻¹` | *not* dimensionless — a **conditional density**, §5 |
| `S̄_φ`, `β^φ`, `α_G^φ`, `D̃^φ` | dimensionless / `p_pop dz` | **no x_M density anywhere** ✔ |

Target (T1)/(T3) carry precisely the same units. ✔ **No dimensional defect.**

The one item worth stating explicitly: `g_frac` having units `[x_M]⁻¹` means
the 1D and 2D `combined_*` columns are **not** on a common scale and must never
be summed or compared in absolute value — only their h-shapes are meaningful.
That is already the practice; it is recorded here because §6.6 shows the
catalogue leg is near-absent from the 2D mixture and a future reader may be
tempted to compare the channels' absolute likelihoods.

---

## 5. Limiting cases (worked, with numbers)

Setup: measured `σ_cond ≈ 8.8e-8` (`FIXB_PATHA_PACKAGE.md` §3.5 L5) makes the
Gauss-Hermite average a point evaluation, so

```
g_i(z;h) ≃ φ( μ_cond(z;h) · M_z,det/(1+z) ) · M_z,det/(1+z) ,   μ_cond = 1 + proj·(x_dL−1)
```

### (a) Flat population mass function — **the premise is REFUTED**

*Claim under test: "flat φ over the support ⇒ `g` must lose all h-dependence."*

Take `φ(M) = c` (flat in `M`). Then

```
g_i(z) = c · M_z,det/(1+z)  ∝ (1+z)^{-1}
```

**`g` retains a full `(1+z)⁻¹` dependence** — the mass Jacobian `dM/dM_z`. It
is z-dependent, hence h-dependent through the quadrature's reweighting of `z`.
And this is *correct*: it is the exact statement that a source-frame-flat mass
function is **not** detector-frame flat. Confirming from the target: with
`φ = c`, substituting `u = M(1+z)` in (T1) gives
`∫dM c N₂(·, M(1+z)) = (c/(1+z)) N₁(·)` — the `1/(1+z)` is *in the target*.

Sign: flat φ gives `dln g/dln(1+z) = −1 < 0`, so `ḡ(h)` would **fall** with h.
The measured `ḡ(h)` **rises** (0.134769 → 0.141337, `adjudicate_g_frac.py`).
⇒ **the observed slope is not the Jacobian — it is φ's curvature, and it beats
the Jacobian.** This limiting case *localises the slope to the population*,
which is the whole question, and it does so in the direction "physics".

Closed form for a locally power-law φ, `φ_M(M) ∝ M^{-p}`:

```
g_i(z;h) = C_i · (1+z)^{p−1} ,  C_i = M_z,det,i^{1−p}/Z_φ ,  d ln g_i/d ln(1+z) = p − 1 = −s_dex
```

With `dn/dlog₁₀M ∝ M^{-0.30}` (`emri_rate.py:96`, Babak 2017 Eq. 5) and
`R_eff ∝ M^{-0.13}` (`emri_rate.py:235-261`, Eqs. 31×34): `s_dex = −0.43`,
`p = 1.43`, so **`dln g/dln(1+z) = +0.43`** — positive, matching the sign of the
measurement.

**AMENDMENT (Gate B, 2026-08-05):** φ is not a single power law over its
support — it is a **broken** power law. `kappa_cap` (`emri_rate.py:169-198`)
applies a `(M/M_turn)^{1/2}` roll-off below `M_turn = 1e5 M_sun`
(`emri_rate.py:169`), multiplying the `M^{-0.30}×M^{-0.13}` product used above
(`emri_rate.py:198` applies the factor). Above the kink `s_dex = −0.43` as
derived; **below it, the roll-off flips the local slope to `s_dex = +0.07`**.
Event 953's source-frame mass trajectory straddles `M_turn` across the h grid,
giving a **98 % deviation** from the single-power-law closed form for that
event, and `g_frac,953(h)` **turns over** (changes sign of curvature) at
`h ≈ 0.733`. The 0.1 % / 8 % closure figures in §6.3/§6.5 are aggregate
statistics over 1588 events and are not invalidated in the mean, but the
single-power-law closed form is not exact event-by-event near the kink — see
the amended §6.4, §6.5, and pin P1 below.

### (b) `σ_Mz → 0`

`g_i` → the point evaluation above: finite, non-zero, no `1/σ` blow-up. A dark
host's mass is *never measured* — only the model's density is read at the
GW-implied point. ✔ (unchanged from `FIXB_PATHA_PACKAGE.md` L5).

### (c) The 1D channel as the marginal of the 2D channel — **EXACT, constant = 1**

```
∫dM_z^obs B^{2D}(d_L^obs, M_z^obs; h)
  = ∫dz W(z;h) ∫dM φ(M) ∫dM_z^obs N₂( (d_L^obs, M_z^obs); (d_L(z;h), M(1+z)) )
  = ∫dz W(z;h) ∫dM φ(M) N₁(d_L^obs; d_L(z;h))
  = ∫dz W(z;h) N₁(d_L^obs; d_L(z;h))  =  B^{1D}(d_L^obs; h)
```

using `∫dM φ = 1` (guaranteed by `:1774`, the *identical* `Z_φ` for the
`log₁₀M` and `M` forms). **Constant = 1, for every `h`, exactly.** The channels
are therefore consistent as unrestricted marginals, and `g_frac,i(·;h)` is a
**properly normalised conditional density** `p(M_z^obs | d_L^obs, h)` (in the
`x_M` measure, times the h-independent `M_z,det,i`).

⇒ *No inconsistency in the gate-(vii)-measured direction.* The sign question
posed in the task brief does not arise: the constant is 1, not a function of h.

**But note the caveat this raises (N-2, §9):** the integral above runs over
**all** `M_z^obs`, whereas the detection cut `S_4D` acts in that coordinate. The
MFG-exact 1D marginal restricts it, which inserts `S̄_φ(z;h)` into (T3). This is
a candidate defect of **`B_num`**, not of `g`.

### (d) `f → 1` and `f → 0`

Unchanged from `GATE_PACKAGE_FINAL.md` §2.2 item 5 / `FIXB_PATHA_PACKAGE.md`
§3.5; `g` is untouched by either limit's algebra.

---

## 6. Measurements (all cheap, run this session against the on-disk diagnostics)

Data: `results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/event_likelihoods.csv`
(41 h × 1588 events) and `.../prepared_cramer_rao_bounds.csv`.

### 6.1 The slope, per event

`dln g_frac/dh` at h = 0.73: **all 1588 events positive**, quantiles
(0/5/25/50/75/95/100 %) = 0.0094 / 0.0598 / 0.1297 / 0.1608 / 0.1857 / 0.2124 /
0.2388. `corr(slope, z) = 0.978`.

### 6.2 Σ over events

`Σ_i dln g_frac,i/dh = +243.5 nats per unit h` at h = 0.73
(+244.4 as a chord over the full grid; `Δln ḡ = 0.047586`,
`adjudicate_g_frac.py`).

### 6.3 The closed form reproduces it to **0.1 %**

Prediction from §5(a): `dln g_i/dh = 0.43 · dln(1 + z*(d_L,i; h))/dh`, with
`z*` the redshift solving `dist(z,h) = d_L,det,i` (no free parameter).

| | measured | predicted | ratio |
|---|---|---|---|
| median per-event slope @0.73 | 0.16032 | 0.16057 | **0.9990** (IQR 0.9985–0.9995) |
| Σ over 1588 events | 243.5 | 243.9 | 0.9984 |

**The mechanism is analytically closed.** The entire h-slope of `g_frac` is
`(−s_dex) × dln(1+z*)/dh`: the source-frame mass function's log-slope times the
rate at which a fixed observed `d_L` maps onto redshift as H₀ moves. This is the
textbook mass–redshift (spectral-siren) term — Chernoff & Finn (1993);
Taylor, Gair & Mandel (2012) arXiv:1108.5161; Farr et al. (2019)
arXiv:1908.09084; Ezquiaga & Holz (2022) arXiv:2202.08240.

### 6.4 Band edges are NOT active — φ's support is never touched

Source-frame masses of the 1588 analysed events, `M_src = M_z,det/(1+z(h=0.73))`:
min 9.89e4, 5 % 2.14e5, median 3.87e5, 95 % 7.35e5, max 1.47e6 `M_sun`.
Support is `[1e4, 1e7]`. **Fraction above 1e7: 0.000. Below 1e4: 0.000.** The
support-exit warning (`:4137-4146`) has nothing to fire on. Over the whole h
grid the events stay ≥ 6× from the nearest edge.

⇒ The **hard band-edge/D1 mechanism does not act through `g`.** (Cross-reference,
not conflation: D1 — the stale `ParameterSpace.p0` window,
`FIXB_PATHA_PACKAGE.md` §4 gate (vi) — is a *selection* distortion that feeds
`S_4D`, hence `S̄_φ`, `β^φ`, `Σ⁴ᴰ`, hence `D̃^φ` and `B_scale`. It touches
`B_num_wbh` only through those shared objects, i.e. through factors that are
**common to both channels** and therefore cancel from `g_frac`. D1 cannot be the
carrier of the g-slope through the hard support edges; it remains live for the
shared bias, §7.)

**AMENDMENT (Gate B, 2026-08-05):** the statement above is about the *hard*
support edges `[1e4, 1e7]` only, and those are confirmed inactive. It does
**not** extend to the internal `kappa_cap` kink at `M = 1e5` (§5(a) amendment,
`emri_rate.py:169-198`), which sits well inside the support and **is** active:
the 1588-event source-frame mass distribution (median 3.87e5) straddles it,
and event 953 crosses it across the h grid. The kink is a feature of φ itself,
not of D1 or the hard band, so it does not reopen the D1-cancellation argument
above — but the g-slope closed form is not kink-blind, and **the D1 exclusion
from `g` is therefore CONDITIONAL on N-2's resolution** (§9 R-C): if the
correct 1D marginal turns out to require `S̄_φ(z;h)` inside the quadrature
(N-2), that object is built from the same broken φ and the kink's effect on
the shared normalisation would need to be re-examined together with D1, not
assumed independent of it.

### 6.5 Population check: φ vs the true event-generating mass density

The analysed CRB events were drawn by `Model1CrossCheck` from
`emri_distribution(M,z) = dN_dz_of_mass(M,z) · R_emri(M)`
(`cosmological_model.py:249-283`), source-frame band-rejected on the same
`[1e4,1e7]` (`:293-303`). The estimator's φ is
`mbh_mass_function × R_eff_per_mbh` (`emri_rate.py:72-97`, `:235-261`) — a
**different** parametrisation, and **z-independent** where the generator's is
not (`dN_dz_of_mass` slope at `M = 6e5`: −0.02 at z = 0.2, −0.86 at z = 0.8).

| quantity | value |
|---|---|
| φ's `s_dex` above the `kappa_cap` kink (`M ≥ 1e5`) | **−0.4300** |
| φ's `s_dex` below the `kappa_cap` kink (`M < 1e5`) | **+0.0700** |
| true `s_dex`, event-weighted mean | −0.4253 (median −0.4798) |
| true `s_dex`, **information-weighted** (weight = `dln(1+z*)/dh`) | **−0.4646** |
| implemented total tilt | +243.5 nats/h |
| tilt the true population implies | **+264.0 nats/h** |
| excess (implemented − true) | **−19.6 nats/h (−8.0 %)** |

**The implemented slope is 8 % too SMALL, not too large.** Correcting the
population mass model would move the 2D MAP *further up*, by ≈ +0.01–0.02 in h.
⇒ Population mis-specification is **not** the source of the high displacement,
and this is a falsification of the most natural "defect" hypothesis. (It *is* a
real, separately documentable systematic: the estimator's dark-host mass model
and the analysed events' mass model are two different Barausse-M1 fits that
happen to agree in mean slope to 1 %.)

**AMENDMENT (Gate B, 2026-08-05):** the −0.4300 row is φ's slope only above the
`kappa_cap` kink at `M = 1e5` (§5(a) amendment); below it φ's slope is +0.0700,
not −0.4300 — so "flat over the band" in the original text was wrong, and the
event-weighted/information-weighted means above implicitly average over the
break. Event 953 straddles the kink across the h grid, deviating from the
single-power-law closed form of §6.3 by 98 %, and `g_frac,953(h)` turns over
at `h ≈ 0.733`. This is a single-event outlier against the 1588-event
aggregate (§6.3's 0.1 % median closure is unaffected), but it is not covered
by the "flat φ" framing this subsection used, and it is the reason pin P1
(§9) must be kink-aware rather than a blanket ≤3e-3 tolerance.

### 6.6 Provenance finding N-1: the 2D catalogue leg is near-absent

At h = 0.73, share of the 2D mixture carried by `α_G^φ · L_cat_with_bh`:

| venue | mean share | median | fraction of events with `L_cat_with_bh == 0` | fraction with share > 1 % |
|---|---|---|---|---|
| iiib | 0.0543 | **0.000** | **0.815** | 0.082 |
| joint_r1 | 0.0573 | **0.000** | **0.618** | 0.116 |

The 2D channel is ≈ 95 % completion leg. Gate (i)'s `dMAP/dlnC = 0.0` therefore
measured invariance of a term that is *identically zero for most events*.
(Consistent with `[[mass-relation-reines-volonteri]]`: host BH-mass errors are
3–7× too tight and omit the ~0.55 dex R&V15 intrinsic scatter, so the mass
kernel rejects the true host.) **This does not weaken the present verdict** —
§3.3 proves the completion leg's measure invariance algebraically — but it does
mean gate (i) should not be quoted as measured evidence for 2D measure
consistency.

---

## 7. Does the arithmetic close? — REFUTED at Gate B, replaced (2026-08-05)

~~`ln p_i^{2D} = ln p_i^{1D} + ln g_frac,i(h) + const` for the completion-dominated
events (95 % of the 2D weight, §6.6); `D̃^φ` and `B_scale` are common and cancel.
Hence the two channels' MAPs differ by exactly the tilt divided by the curvature:
the 2D displacement decomposes as
`0.780 = 0.600 (the shared 1D bias, at the rail) + 0.060 (frozen-2D residual
over 1D) + 0.120 (the genuine spectral-siren tilt from φ)`, and the tilt
*overshoots* only because the thing it is added to is 0.13 too low.~~
**[REFUTED, `GATEB_REFUTATION_REPORT.md` claim #4]** — this additive
decomposition violates the standing never-add-MAP-displacements rule
(`GATE_PACKAGE_FINAL.md:609`, `DERIVATION_C7_HOSTZ_KERNEL.md:545-547`);
`ln p^{2D} = ln p^{1D} + ln g + const` was checked directly and fails by
33.8/23.5 nats; and the "0.13 too low" framing treats a flat argmax as a
load-bearing MAP value (see below). The curvature figures quoted in the
original table were not reproduced independently (a sign change was found).

**The defensible statement, in its place:**

The frozen-2D likelihood is **flat over `h ∈ [0.63, 0.745]`** — the argmax
beats `h = 0.640` by only 0.021 nats, i.e. it is statistically unlocalised to
better than the grid spacing, not a sharp MAP. Against that flat baseline, the
tilt is verified directly and exactly, not inferred from a curvature argument:

```
(ln L_2D,live − ln L_2D,frozen) = Σ_i Δ(ln g_frac,i)   to ≤ 0.8 of 63.5 nats
```

i.e. the live-minus-frozen likelihood difference equals the summed g_frac tilt
to better than 1.3%. The freeze is surgically clean and the tilt is real. What
does **not** survive is turning that into a MAP-space picture: an identical
tilt, applied to the two venues' differently-shaped flat frozen likelihoods,
produces `0.120` (iiib) vs `0.160` (joint) — two different numbers from the
same closed-form tilt — because the argmax of a flat function is measure-zero
and not a stable target for arithmetic. Per the never-add-MAP-displacements
rule (`GATE_PACKAGE_FINAL.md:609`), MAP-space shifts of this kind are not
decomposable into additive rail/residual/tilt terms, and this package does
not attempt to reconstruct one.

**Consequence for R-A (§9):** because the additive decomposition that
attributed "0.78 vs 0.73" to "the 1D rail + a correctly-sized tilt" is
refuted, the claim "the question 'is 0.78 wrong?' reduces to the 1D-rail
question" is **NOT ESTABLISHED** by this package (see the ruling in §9). The
tilt itself (§3–§6) is unaffected — it is the log-likelihood identity above,
not a MAP-arithmetic story, that survives.

---

## 8. Why this is *not* a defect — the standing objection and its answer

**Objection.** `g_frac` is a normalised conditional density (§5c), so the score
identity should give `E[∂_h ln g_frac] = 0` at `h_true`; instead **1588 of 1588
events have a strictly positive score**. A zero-mean score cannot produce
1588/1588 same-sign values.

**Answer.** The score identity applies to the *full* per-event log-likelihood
`ln(numerator) − ln α`, not to the factor `ln g_frac` in isolation. The
observed data are distributed as `p(d|det,h) ∝ B^{2D}(d)·1[det(d)]/α`, i.e. the
*detected* conditional in `M_z^obs` is `∝ g_frac · S`, not `∝ g_frac`. Then

```
E_det[ ∂_h ln g_frac ] = (1/⟨S⟩) ∂_h ∫ dM_z S(d_L, M_z) g_frac(M_z; h) = ∂_h ln ⟨S⟩_h
```

which is non-zero and, by construction, is exactly the per-event image of
`∂_h ln S̄_φ` — the object the estimator carries at population level in
`β_Ḡ^φ(h)` inside `D̃^φ`. For scale, `β_Ḡ^φ` falls by `Δln = −0.2856` over the
grid (`FIXB_PATHA_PACKAGE.md` §5 pins: 1.024526e9 → 7.703527e8), i.e. **−1098
nats/h per event-normalisation**, ×1588 ≫ the +243.5 in question. All-positive
per-event scores in a single factor are therefore expected, not anomalous.

The residual question this leaves — *does the population-level `β_Ḡ^φ(h)`
correctly absorb the per-event `∂_h ln⟨S⟩_i`, or does the per-event
heterogeneity of `⟨S⟩_i` leave a net tilt?* — is the honest boundary of what a
derivation can settle. It is a closure question, and §9's regression names the
measurement that answers it. Under MFG it is answered "yes" by construction
(α is population-level by design); the only way it fails is if the events'
`(d_L, M_z)` distribution is not the model's — which §6.5 measures at the 8 %
level, adverse.

---

## 9. What the author must rule on

No `/physics-change` gate package is drafted, because **no formula change is
proposed for `g`**. Three rulings are requested instead:

**R-A — Accept verdict (i) for `g_frac(h)`.** The h-slope is the spectral-siren
term; derivation §3, closed form §6.3 (0.1 %), limiting cases §5, population
check §6.5 (8 %, adverse). **Consequence clause — NOT ESTABLISHED (Gate B,
2026-08-05):** the original text re-attributed the residual 2D displacement
from "the mass factor" to "the shared bias that rails the 1D channel"; that
re-attribution rested on the §7 additive decomposition, which is **refuted**
(§7 above, `GATEB_REFUTATION_REPORT.md` claim #4). Verdict (i) itself (the
h-slope is genuine spectral-siren physics, not a measure defect) survives on
its own evidence (§3, §6.3, §5), but the further claim that this *explains* why
the live 2D MAP sits at 0.78/0.80 rather than 0.73 does not follow from
anything in this package. **The pre-registered §9 closed-loop test below is
the deciding measurement for that question** (being built and run tonight,
2026-08-05/06 — see "Incomplete checks" in `GATEB_REFUTATION_REPORT.md`).
`BIAS_HISTORY_LEDGER` and the suspect queue should record verdict (i) as
accepted but the MAP-displacement re-attribution as pending the closed-loop
result, not as settled. D1 stays on the list for the **shared** bias
(§6.4 shows it cannot act through `g`'s hard support edges; the internal
`kappa_cap` kink is active but is a property of φ, not of D1 — see the §6.4
amendment), with the D1-exclusion-from-`g` conditional on N-2's resolution
(R-C below).

**R-B — Rule on N-1 (documentation/evidence, no code).** Whether
`GATE_PACKAGE_FINAL.md` gate (i) / `FIXB_PATHA_PACKAGE.md` §4 gate (i) may
continue to be quoted as *measured* 2D measure-invariance evidence, given that
the term rescaled is identically zero for 81.5 %/61.8 % of events (§6.6).
Recommendation: replace the measured claim with the algebraic proof of §3.3
(which is strictly stronger), and record the 82 %-zero catalogue leg as a
first-class provenance caveat on every 2D result.

**R-C — Open N-2 as a separate `/physics-change` question, RE-SCOPED (Gate B,
2026-08-05): a both-channels selection-in-numerator question, not a 1D-only
one.** Is `B_num` (`:4096`) the correct MFG marginal? §5(c) shows it is the
marginal over **all** `M_z^obs`, while `S_4D` — the detection cut — acts in
that very coordinate. The MFG-exact reduced-data marginal is

```
B^{1D}_corr(d_L^obs; h) = ∫dz (1−f_k) (dV_c/dz)/(1+z) · N₁(d_L^obs; d_L(z;h)) · S̄_φ(z;h)
```

i.e. `S̄_φ` **inside** the 1D quadrature. This is *not* the MFG "p_det in the
numerator" mistake — it is the standard truncation that appears when an
observable entering the selection function is discarded from the likelihood.
The original text stated this moves only the 1D channel. **That scoping is
too narrow**: the detection model is θ-deterministic
(`simulation_detection_probability.py:175-179`), so the correct hierarchical
numerator carries `p_det(θ)` in **both** channels — as `S_4D` inside `g_i`'s
`M`-integral on the 2D side, versus `S̄_φ(z;h)` on the 1D side — and this
factor does **not** cancel out of `g_frac` in general. §6.4's claim that D1
"cannot reach `g`" is therefore **conditional on how N-2 resolves**: if the
2D-side correction is required as well as the 1D-side one, D1's channel into
`S_4D`/`S̄_φ` becomes a channel into `g_frac` too, through the same shared
objects (`β^φ`, `D̃^φ`) that §6.4 currently treats as cancelling. If adopted,
`g_frac`'s definition becomes `B_num_wbh_corr/B_num_corr` with both numerators
carrying their respective selection factor, restoring the per-event score
identity (§8) on both legs. Direction and magnitude unknown and un-derived
here.

**Pre-registered acceptance criteria, should the author want R-A tested rather
than accepted on the derivation** (the one measurement that decides):

* **Instrument:** `validation/pp_coverage.py` (G4b harness), extended with the
  mass channel — a synthetic universe whose dark hosts are drawn from *the
  estimator's own* φ and `w_pop`, selected by *the estimator's own* `S_4D`, then
  scored by the 2D completion leg alone (catalogue leg off, which §6.6 shows is
  nearly the production configuration anyway).
* **Pre-registration:** ≥ 200 seeds, `h_true = 0.73`, canonical 41-h grid.
  * **CONFIRM (i):** the 2D MAP distribution is centred on 0.73 within
    `±0.010` (MC error at 200 seeds ≈ 0.005 given the 0.03 per-realisation
    spread of record) and the P–P curve is inside the 90 % band. ⇒ `g`'s tilt
    is self-consistent; the production displacement is inherited, R-A stands.
  * **REFUTE:** the 2D MAP is displaced by ≥ +0.03 in a closed loop where the
    data provably follow the model. ⇒ the 2D leg's normalisation is defective;
    R-C becomes blocking and the fix is derived there.
  * **MIXED:** displacement in `(0.010, 0.030)` ⇒ report the split; do not force
    a branch. Quote the measured `Σ ∂_h ln g_frac` of the synthetic set against
    the production +243.5 nats/h.
* **Regression pins (must-not-move if any of this ever changes code):**
  * `P1` — `g_frac` closed-form pin, **kink-aware** (amended, Gate B
    2026-08-05): for the shipped `roots_hermite(64)` path, `dln g_frac,i/dh`
    must equal `−s_dex(M_src,i) · dln(1+z*(d_L,i,h))/dh` to `≤ 3e-3` relative,
    for all events whose source-frame mass trajectory over the h grid does
    **not** straddle the `kappa_cap` kink at `M = 1e5` (`emri_rate.py:169`) by
    more than the grid's per-step mass excursion. Events that straddle the
    kink (event 953 is the known case, 98 % deviation) must be excluded from
    this pin or tracked in a separately-banded pin with its own tolerance,
    since φ's local slope there is +0.07, not −0.43. As measured today: 40 of
    1588 events exceed `3e-3` relative deviation — this is the straddle
    population, not a failure of the closed form away from the kink.
  * `P2` — flat-φ regression: with `dark_mass_log10_density_unnormalised`
    stubbed to a constant, `g_i(z;h)·(1+z)/M_z,det` must be h-independent and
    z-independent to `≤ 1e-12` (the §5(a) Jacobian-only limit).
  * `P3` — marginal identity: `∫dM_z^obs B_num_wbh(M_z^obs) = B_num` to
    `≤ 1e-5` relative (amended, Gate B 2026-08-05, relaxed from `≤ 1e-6`) at
    h = 0.60, 0.73, 0.86 (the §5(c) exact limit; a direct numerical test of
    the normalisation, and the cheapest possible guard against a future
    measure error). Reason for the relaxation: the code's own `∫φ dM` closes
    to `1 + 7.7e-7`, not exactly 1 (finite-support/finite-quadrature residual
    of `dark_mass_density_per_mass`'s own normalisation, `:1774`), which
    leaves no headroom under a `1e-6` pin — the identity would be measuring
    quadrature noise in φ's own normalisation, not a defect in `B_num_wbh`.
  * `P4` — support-exit counter (`:4137-4146`) must stay at **0** for the
    production CRB set (§6.4); a non-zero count invalidates the "band edges
    inactive" premise of this package.
  * `P5` — `β_Ḡ^φ`, `S̄_φ` pins of `FIXB_PATHA_PACKAGE.md` §5 unchanged.

---

## 10. References

* Mandel, Farr & Gair (2019), arXiv:1809.02063, Eqs. (5)–(7) — hierarchical
  selection; α uses the same population and detection model as every numerator
  (A2). Applied to the hybrid density of `GATE_PACKAGE_FINAL.md` Appendix A.
* Gray et al. (2020), arXiv:1908.06050, Eqs. (32), (A.10), (A.19) — the
  catalogue/completion partition and the completion numerator's structure.
* Turski, Bilicki, Dálya, Gray & Ghosh (2023), arXiv:2302.12037, Eq. (8) —
  the completion numerator and denominator carry the population
  mass/luminosity density.
* Babak et al. (2017), arXiv:1703.09722, Eqs. (5), (23), (31)×(34) — φ and
  `R_eff` (`emri_rate.py:72-97`, `:235-261`).
* Bishop (2006), *PRML*, Eqs. (2.81)–(2.82) — the Gaussian conditional used at
  `:3258-3263` and `:1990`.
* Chernoff & Finn (1993), ApJ 411, L5; Taylor, Gair & Mandel (2012),
  arXiv:1108.5161; Farr, Fishbach, Ye & Holz (2019), arXiv:1908.09084;
  Ezquiaga & Holz (2022), arXiv:2202.08240 — the mass-scale/spectral-siren H₀
  mechanism that §5(a)/§6.3 identify `g`'s h-slope as.
* Honest gap retained (unchanged from `FIXB_PATHA_PACKAGE.md` §3.3): no
  published dark-siren analysis carries a compact-object mass observable in the
  catalogue/completion split. The 2D channel is this project's extension; the
  s-sweep (±0.036 in h) remains the quoted dark-class systematic.

## 11. Provenance

Numbers in §6 and §7 were computed this session from
`results/run_20260804_postfix/{iiib,joint_r1}/diagnostics/event_likelihoods.csv`
and `prepared_cramer_rao_bounds.csv` (1590 rows, 1588 analysed), plus direct
calls to `master_thesis_code.physical_relations.dist_to_redshift`,
`master_thesis_code.emri_rate`, and `master_thesis_code.cosmological_model`.
No source file was modified; no run was launched; no commit was made.

---

## Gate B adjudication (2026-08-05)

Independent opus-tier adversarial review, full record in
`GATEB_REFUTATION_REPORT.md`. Overall: **goes to the author WITH AMENDMENTS**
(not rework). Per-claim verdict:

| # | claim | verdict |
|---|---|---|
| 1 | Measure argument (x_M measure numerator-only; α(h) pure number; completion-leg measure invariance exact) | **SURVIVES** (caveat: conditional on N-2) |
| 2 | 0.1 % slope closure, `s_dex = −0.43` | **SURVIVES + AMENDED** (φ is a broken power law; kink active; P1 amended) |
| 3 | Exact 1D-marginal identity `∫(2D compl. num.) dM_z^obs = 1D compl. num.` | **SURVIVES (verified independently)**; P3 relaxed to ≤1e-5 |
| 4 | Decomposition "0.600 rail + 0.060 + 0.120 genuine tilt" and its "re-attribute to 1D rail" consequence | **REFUTED** (§7 rewritten; survives only: live−frozen ln L = Σ Δln g_frac to ≤0.8/63.5 nats) |
| 5 | Flat-φ limiting case (slope flips negative) | **SURVIVES** |
| N-1 | Gate (i) near-vacuous (2D catalogue leg dead for most events) | **SURVIVES exactly** |
| N-2 | 1D-side selection-marginal defect candidate | **REAL in structure, MIS-SCOPED** — both-channels question, not 1D-only (R-C amended) |
| §6.5 | Population cross-check (implemented tilt 8 % too small, adverse) | **SURVIVES** |

Required amendments 1–5 of the refutation report have been applied in place
(§5(a), §6.4, §6.5, §7, P1, P3, R-A, R-C above). R-A's consequence clause is
downgraded to NOT ESTABLISHED pending the §9 closed-loop synthetic-universe
test, which is the deciding measurement and was not run as part of this
package. See `GATEB_REFUTATION_REPORT.md` for the full evidence trail,
including the two incomplete checks (posterior-weighted-z vs plug-in z*;
the §9/G4b run itself).
