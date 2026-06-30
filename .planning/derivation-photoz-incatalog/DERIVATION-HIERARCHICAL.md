# DERIVATION — the hierarchical shared-latent in-catalogue likelihood, adapted to the partition-norm closure

Status: **CANDIDATE for approval. NOT to be coded yet.** This file derives the Hint-2 object
(Hitchhiker Eq. 31 / 33, the full shared-latent likelihood) re-expressed for OUR weighted
partition-norm closure, proves the mandatory `sigma_z -> 0` gate, gives the dimensional analysis,
and delivers an **honest** mechanism analysis of whether it can de-rail at `p_det ~ 1`,
`sigma_z/z ~ 0.7`. The honest bottom line is stated up front and is not oversold.

Date: 2026-06-30. Conventions (closure): `_OMEGA_M = 0.25`, `_OMEGA_DE = 0.75`, `h = H0/100`,
truth `h = 0.73`, distances in Gpc, redshift dimensionless, `p_bg(z) ∝ (dV_c/dz)/(1+z)`,
photo-z model Gaussian `N(z; z_cat, sigma_z)` (closure Eq. 17 analogue).

> **One-sentence verdict.** The literal shared-latent likelihood adapted to our closure is the
> correct theoretical object and passes the `sigma_z -> 0` gate exactly, but the honest mechanism
> analysis predicts it **degenerates** to the already-falsified clean-numerator form at `p_det ~ 1`
> because the entire difference between the full hierarchical form and the per-event form is
> governed by the **gradient of `p_det` across the catalogue**, which is `~ 0` in our regime; the
> only ingredient that is not guaranteed to degenerate is the photo-z **symmetrisation of the
> selection denominator**, and whether its residual `O(sigma_z^2)` effect moves the rail interior
> is a quantitative question only the prototype can settle.

---

## 0. Exact source equations (quoted)

**Hitchhiker (Gair et al. 2023, arXiv:2212.08694), verified via ar5iv.**

- **Eq. 13** (approximate catalogue redshift prior built from per-galaxy posteriors):
  ```
  p_CBC(z) ≈ p_cat(z|{ẑ_g}) = (1/N_gal) Σ_i^{N_gal} p_red(z|ẑ_g^i)
  ```
- **Eq. 16** (per-galaxy regularised posterior — `p_bg` applied ONCE):
  ```
  p_red(z|ẑ_g^i) = L_red(ẑ_g^i|z) p_bg(z) / ∫ L_red(ẑ_g^i|z) p_bg(z) dz
  ```
- **Eq. 17** (Gaussian redshift likelihood; in TH21/Hitchhiker `sigma_z = 0.013(1+z)^3 ≤ 0.015`):
  ```
  L_red(ẑ_g^i|z) = (1/(√(2π) sigma_z)) exp[ −(ẑ_g^i − z)^2 / (2 sigma_z^2) ]
  ```
- **Eq. 30** (detection probability as a function of the TRUE galaxy redshifts):
  ```
  p_det(H0,{z_g}) = (1/N_gal) Σ_{j=1}^{N_gal} ∫_{−∞}^{d_L^thr} L_GW(x|d_L(z_g^j,H0)) dx
                  = (1/N_gal) Σ_{j=1}^{N_gal} p_det^GW(d_L(z_g^j, H0))
  ```
- **Eq. 31** (the FULL hierarchical shared-latent likelihood — the Hint-2 target):
  ```
  p(H0|{x}) ∝ p(H0) ∫ d{z_g}
        [ ∏_{j=1}^{N_obs} L_GW(x_j|{z_g},H0) / p_det(H0,{z_g}) ]
      × [ ∏_{k=1}^{N_gal} L_red(ẑ_g^k|z_g^k) ] p_bg({z_g}),
  with  L_GW(x_j|{z_g},H0) = (1/N_gal) Σ_{i=1}^{N_gal} L_GW(x_j|d_L(z_g^i,H0)),
  and   p_det(H0,{z_g}) given by Eq. 30,  p_bg({z_g}) = ∏_k p_bg(z_g^k).
  ```
  The paper's own commentary on Eq. 31 is decisive for our analysis: *"This dependence of the
  denominator of the GW likelihood on `{z_g}` breaks the separability of the integrals ... unless
  the true redshifts of the galaxies are perfectly known"*, and *"if that volume is sufficiently
  large ... the dependence of the detection probability on the actual galaxy redshifts will be
  relatively weak and so this term can be factored out, reducing the result to the simpler
  expression."* The cross-terms (same host for two events) are *"1/N_gal times fewer ... we will
  essentially never be in a regime where these corrections matter."*
- **Eq. 32** (the `p_bg` used — `dV_c` once, H0-independent):
  ```
  p_bg(z) = (dV_c/dz) / ∫_0^∞ (dV_c/dz) dz
  ```
- **Eq. 33** (the ONE-galaxy reduction of Eq. 31; the form they actually de-bias with in Sec. 3.3):
  ```
  p(H0|{x}) ∝ p(H0) ∫ dz [ ∏_{j=1}^{N_obs} L_GW(x_j|d_L(z,H0)) / p_det(H0,z) ] × [ L_red(ẑ|z) ] p(z)
  ```

**GWcosmo (Gray et al. 2023, arXiv:2308.02281), verified via ar5iv — Eq. 2.2** (population-level
ensemble normalisation; cross-reference for the `−N_det` power):
```
p(Λ|{x_GW},{D_GW},I) ∝ p(Λ|I) p(N_det|Λ,I)
        [ ∫ p(D_GW|θ,Λ,I) p(θ|Λ,I) dθ ]^{−N_det}
      × ∏_i^{N_det} ∫ p(x_GWi|θ,Λ,I) p(θ|Λ,I) dθ.
```
The selection integral `p(D_GW|Λ) = ∫ p(D_GW|θ,Λ) p(θ|Λ) dθ` *"applies to the whole population
of CBCs (as opposed to being event-specific) it can be taken outside the product"* — i.e. one
population selection object raised to `−N_det`, exactly the Hitchhiker `∏_j 1/p_det(H0,{z_g})`
structure with `p_det` evaluated on a shared population.

---

## 1. The current per-event object (what we are replacing)

From `rung_I_prior_domination.py` / `_rungI_verify_B.py` and `bayesian_statistics.py`:

```
L_cat,i(h)  = [ Σ_g w_g N_g,i(h) ] / D(h)                                   (per event i)
N_g,i(h)    = ∫ p_GW(d_L(z,h); d_meas_i, sdL_i) · N(z; z_cat_g, sigma_z) dz   (BARE Gaussian)
D(h)        = Σ_g w_g p_det^GW(d_L(z_cat_g, h))      (precompute_global_catalog_selection; POINT)
w_g         = R_eff(M_g) / (1 + z_cat_g)            (rate weight; generalises Hitchhiker's 1/N_gal)
posterior:    p({x}|h) = ∏_i L_cat,i(h),   then normalise over the h-grid.
```

Two facts about this object, established by the prior work:
1. It is the **per-event approximation** of Eq. 31: the host true redshift is marginalised
   **separately for each event** (one Gaussian convolution per event), and the selection is a
   **single event-independent scalar** `D(h)` — the `{z_g}` denominator collapsed to a constant.
2. The numerator carries the **bare** Gaussian (no `p_bg`); equivalently the in-cat host redshift
   density carries **net `dV_c = 0`** (NORMALISATION-FIX / CATALOG-INTERPRETATION).

The product form `∏_i L_cat,i = (∏_i Σ_g w_g N_g,i) / D(h)^{N_obs}` already contains the `−N_obs`
power of GWcosmo Eq. 2.2 — that power is **not** the missing ingredient. What is collapsed relative
to Eq. 31 is (i) the `{z_g}`-dependence **inside** `p_det`, and (ii) the **shared** (rather than
per-event) marginalisation of the host true redshifts.

---

## 2. THE CANDIDATE — adapted shared-latent in-catalogue likelihood

### 2.1 Literal adaptation of Eq. 31 to the weighted partition-norm closure

Replace Hitchhiker's flat `(1/N_gal) Σ` by our rate weights `w_g`, and identify
`ẑ_g^k = z_cat_g` (the catalogue photo-z), `L_red = N(·; ·, sigma_z)`,
`p_bg(z) ∝ (dV_c/dz)/(1+z)`. The host TRUE redshifts `{z_g}` are SHARED LATENT variables,
marginalised **once** across **all** `N_obs` events:

```
p(h|{x}) ∝ p(h) ∫ [ ∏_{k=1}^{N_gal} dz_k  L_red(z_cat_k | z_k) p_bg(z_k) ]
                 × ∏_{i=1}^{N_obs}  [ Σ_g w_g L_GW(x_i | d_L(z_g, h)) ]
                                    ─────────────────────────────────────
                                    [ Σ_g w_g p_det^GW(d_L(z_g, h)) ]          (★ FULL form)
```

with `L_GW(x_i|d_L(z_g,h)) = p_GW(d_meas_i; d_L(z_g,h), sdL_i)` (Gaussian in `d_L`). Compared to
the current object, **three things change**, and **only these three**:

| ingredient | current per-event | candidate (★) |
|---|---|---|
| host true-z marginalisation | per-event Gaussian convolution (independent per event) | **shared** integral over `{z_g}`, ONE marginalisation for all events |
| selection denominator | scalar `D(h) = Σ_g w_g p_det^GW(d_L(z_cat_g,h))` (POINT, frozen) | `Σ_g w_g p_det^GW(d_L(z_g,h))` **inside** the `{z_g}` integral, on the TRUE redshifts |
| population prior `p_bg` (`dV_c` once) | absent in numerator (net `dV_c = 0`) | `p_bg(z_k)` on each true redshift, applied **exactly once** (Eq. 32 / Hint 4) |

The `dV_c` is counted once: each galaxy carries one `p_bg(z_k)` in the prior; no second `dV_c`
appears at re-injection (Hitchhiker "Inconsistency 1" avoided), and the out-of-catalogue branch
`B_num` keeps its own single `dV_c` exactly as today (CATALOG-INTERPRETATION §3.2).

### 2.2 Tractable (mean-field) reduction — the form to prototype

The full `{z_g}` integral over `N_gal` latents is the genuinely shared object but is intractable
verbatim (millions of galaxies). Two standard, paper-sanctioned reductions make it computable
**without changing the small-`sigma_z` limit**:

- **Cross-terms negligible** ("space is big", Eq. 31 commentary): drop the same-host terms in the
  expansion of `∏_i Σ_g`. They are `1/N_gal` suppressed. Then the numerator's `{z_g}` integral
  **factorises per galaxy**, and each event integrates against the per-galaxy regularised posterior
  ```
  p_red(z | z_cat_g) = L_red(z_cat_g|z) p_bg(z) / Z_g ,   Z_g = ∫ L_red(z_cat_g|z) p_bg(z) dz  (Eq.16)
  ```
- **Selection sum concentrates** (law of large numbers over `N_gal` i.i.d. latents — exactly the
  paper's "factor it out" argument, but done **consistently with the same `p_red`** rather than at
  a point): the random selection sum `S(h,{z_g}) = Σ_g w_g p_det^GW(d_L(z_g,h))` concentrates on
  its expectation under the same per-galaxy posteriors,
  ```
  D_sm(h) ≡ E[S] = Σ_g w_g ∫ p_det^GW(d_L(z,h)) p_red(z | z_cat_g) dz      (SMEARED selection)
  ```
  and is pulled out of the integral, raised to `−N_obs`.

The resulting **tractable candidate** (THE object to code behind a flag) is:

```
p(h|{x}) ∝ p(h) · ∏_{i=1}^{N_obs} Ñ_i(h) / D_sm(h)^{N_obs}                       (★★ tractable)

Ñ_i(h)  = Σ_g w_g ∫ L_GW(x_i|d_L(z,h)) · p_red(z | z_cat_g) dz       (clean numerator, dV_c once)
D_sm(h) = Σ_g w_g ∫ p_det^GW(d_L(z,h)) · p_red(z | z_cat_g) dz       (photo-z-SMEARED selection)
p_red(z|z_cat_g) = N(z; z_cat_g, sigma_z) · p_bg(z) / Z_g ,   p_bg(z) ∝ (dV_c/dz)/(1+z).
```

**What differs from the current `L_cat,i/D(h)`, made explicit:**
- numerator `N_g,i` (bare Gaussian) → `Ñ_i` uses the regularised posterior `p_red` (adds the one
  `dV_c`; the Hint-4 / NORMALISATION-FIX "Angle A/C" numerator);
- denominator `D(h)` (point selection at `z_cat_g`) → `D_sm(h)` (selection averaged over the **same**
  `p_red` kernel — the photo-z symmetrisation that CATALOG-INTERPRETATION §3.3 flagged as missing);
- both num and denom now use the identical per-galaxy density `p_red` (the genuine same-kernel
  property, Hint 1, but now realised at the **global/population** level inside the partition norm,
  not the failed LOCAL per-event same-kernel of `consistent_denom`).

`(★★)` is the **factorised** image of `(★)`. The **only** information in `(★)` that `(★★)` discards
is (a) the same-host cross-terms and (b) the **fluctuations** of `S` about `D_sm` (the Jensen gap
`E[S^{−N_obs}] − D_sm^{−N_obs}`). Both are `O(1/N_gal)` AND proportional to the variance of
`p_det^GW` across the catalogue — see §5, where this is the crux of the honest de-rail analysis.

---

## 3. What is shared across events (the coupling object)

The object that couples the events is the **population selection functional evaluated on the
shared latent host-redshift field**:
```
S(h, {z_g}) = Σ_g w_g p_det^GW(d_L(z_g, h)) ,   entering as  ∏_i 1/S = S^{−N_obs}.
```
The **same** field `{z_g}` enters every event's numerator (`Σ_g w_g L_GW(x_i|d_L(z_g,h))`) and the
**single** shared denominator `S`. This is structurally different from a per-event scalar
denominator: the current code has `N_obs` copies of one frozen number `D(h)`; the candidate has one
**functional** of the entire latent field, common to all events, whose value correlates with which
galaxies the numerators select. In the tractable reduction this functional is replaced by its
expectation `D_sm(h)` — **a single h-dependent scalar** — and at that moment it again becomes
numerically equivalent to a per-event scalar. **That equivalence is the degeneracy** discussed in
§5: the coupling survives as a genuine event-coupling only to the extent that `S` fluctuates with
`{z_g}`, i.e. to the extent `p_det^GW` varies across the catalogue.

Cross-reference: this is exactly GWcosmo's `[∫ p(D_GW|θ,Λ)p(θ|Λ)dθ]^{−N_det}` — one population
object raised to `−N_det`, "taken outside the product" — except GWcosmo integrates `θ` (incl.
redshift) against the **population** `p(θ|Λ)`, whereas the in-catalogue version integrates against
the **catalogue's** per-galaxy posteriors `p_red`. In both, the shared object is a population/
catalogue selection normalisation, not a per-event quantity.

---

## 4. MANDATORY GATE — `sigma_z -> 0` reduction to the standard Option-A global form

**Claim.** As `sigma_z -> 0`, the candidate `(★★)` (and the full `(★)`) reduces *identically* to the
current standard per-event Option-A object `∏_i L_cat,i(h) = (∏_i Σ_g w_g N_g,i)/D(h)^{N_obs}` with
the bare numerator and the point global selection. Hence it inherits the verified small-`sigma_z`
behaviour (`sigma_z = 0.002 -> 0.7438`, peaked, ~unbiased).

**Proof, step by step.**

1. **Photo-z likelihood -> delta.** `L_red(z_cat_g|z) = N(z; z_cat_g, sigma_z) -> δ(z − z_cat_g)`
   as `sigma_z -> 0` (nascent delta).

2. **Regularised posterior -> delta, `p_bg` cancels.** From Eq. 16,
   ```
   p_red(z|z_cat_g) = N(z;z_cat_g,sigma_z) p_bg(z) / Z_g ,   Z_g = ∫ N(z;z_cat_g,sigma_z) p_bg(z) dz.
   ```
   As `sigma_z -> 0`: `Z_g -> p_bg(z_cat_g)` and
   `p_red(z|z_cat_g) -> δ(z − z_cat_g) p_bg(z)/p_bg(z_cat_g) = δ(z − z_cat_g)`.
   **The `p_bg` (the one `dV_c`) cancels identically** — this is the same cancellation proven for
   "Angle A/C" in NORMALISATION-FIX §2 (measured gate PASS, 0.7478 vs standard 0.7438). So in the
   limit the candidate numerator loses its `dV_c` and coincides with the **bare** numerator.

3. **Numerator -> bare Option-A sum.**
   ```
   Ñ_i(h) = Σ_g w_g ∫ L_GW(x_i|d_L(z,h)) p_red(z|z_cat_g) dz
          -> Σ_g w_g L_GW(x_i|d_L(z_cat_g,h)) = Σ_g w_g N_g,i^{point}(h),
   ```
   i.e. the current bare catalogue sum evaluated at the catalogue redshifts (the
   `N_g,i` convolution also collapses to a point evaluation since its own kernel -> δ). Identical
   to the standard numerator.

4. **Smeared selection -> point selection.**
   ```
   D_sm(h) = Σ_g w_g ∫ p_det^GW(d_L(z,h)) p_red(z|z_cat_g) dz
           -> Σ_g w_g p_det^GW(d_L(z_cat_g,h)) = D(h),
   ```
   exactly `precompute_global_catalog_selection` (the current global Option-A scalar).

5. **Shared integral collapses, denominator factors out exactly.** In the full form `(★)`, every
   `L_red -> δ(z_k − z_cat_k)` forces `z_k = z_cat_k` for all `k`, so the `{z_g}` integral collapses
   to the single point `{z_cat_g}`. Then `S(h,{z_g}) -> S(h,{z_cat_g}) = D(h)` is a **constant**
   (no residual latent), pulled out with **zero** Jensen gap; the same-host cross-terms vanish
   because distinct galaxies have distinct `δ`-supports. Thus
   ```
   p(h|{x}) -> p(h) · ∏_i [ Σ_g w_g L_GW(x_i|d_L(z_cat_g,h)) ] / D(h)^{N_obs}
             = p(h) · ∏_i L_cat,i(h).                                         ∎
   ```

This is **exactly** the current standard global Option-A posterior. Gate **PASS by construction**;
the candidate cannot change the (correct) small-`sigma_z` behaviour. [CONFIDENCE: HIGH — this is an
algebraic limit, cross-checked against the measured `regularised_kernel` gate value 0.7478.]

---

## 5. HONEST mechanism analysis — can it de-rail at `p_det ~ 1`, `sigma_z/z ~ 0.7`?

This is the load-bearing section. I do **not** oversell it.

### 5.1 The de-rail lever is the `{z_g}`-dependence of the selection — and it is ~0 for us

The **entire** difference between the full hierarchical `(★)` and the per-event form is the
`{z_g}`-dependence of the shared denominator `S(h,{z_g})` (Eq. 31 commentary: this dependence "is
what breaks the separability"; when it is weak, "this term can be factored out, reducing the result
to the simpler expression"). Quantitatively, expanding the shared factor about the mean,
```
E[ S(h,{z_g})^{−N_obs} ]  =  D_sm(h)^{−N_obs} · [ 1 + (N_obs(N_obs+1)/2) Var(S)/D_sm^2 + ... ],
Var(S)/D_sm^2  ~  (1/N_gal) · Var_g[ w_g p_det^GW(d_L(z_g,h)) ] / (mean_g[ w_g p_det^GW ])^2 .
```
The correction is `O(1/N_gal)` **and** proportional to the **variance of `p_det^GW` across the
catalogue**. In our regime hosts sit at `z ~ 0.05 << z_horizon`, so `p_det^GW(d_L(z_g,h)) ≈ 1` for
all in-cat galaxies; therefore `Var_g[p_det^GW] ≈ 0`, `S(h,{z_g}) ≈ Σ_g w_g = const`, and the
hierarchical correction **vanishes on both counts** (large `N_gal` and flat `p_det`). The full `(★)`
collapses to the mean-field `(★★)`, and the de-railing channel Hitchhiker relies on in Sec. 3.3
(where their `p_det` does vary across the EM-uncertain host redshift) is **absent**.

### 5.2 Does the mean-field candidate `(★★)` itself de-rail? Likely NOT — it degenerates

Even granting `(★★)`, the new ingredient relative to the already-tested variants is the **smeared**
selection `D_sm(h)` versus the **point** selection `D(h)`. Their difference is a second-order
photo-z effect:
```
D_sm(h) − D(h) = Σ_g w_g [ ∫ p_det^GW(d_L(z,h)) p_red(z|z_cat_g) dz − p_det^GW(d_L(z_cat_g,h)) ]
              ≈ Σ_g w_g · (1/2) sigma_eff^2 · d²/dz²[ p_det^GW(d_L(z,h)) ]|_{z_cat_g} + (drift),
```
which is `O(sigma_z^2 × p_det'')`. With `p_det^GW ≈ 1` and nearly flat over the in-cat support,
`p_det'' ≈ 0`, so `D_sm(h) ≈ D(h)`. The candidate `(★★)` therefore numerically approaches
**"clean `p_red` numerator + (≈) global point denominator"** — which is exactly the
NORMALISATION-FIX "Angle A/C" / "Angle B" variant **measured to rail UP to 0.8700** at
`sigma_z = 0.035`. So the honest mechanistic prediction is: **`(★★)` likely rails up to ~0.87**,
because its only genuinely new piece (`D_sm − D`) is `O(sigma_z^2 p_det'') ≈ 0` here.

### 5.3 The object that *would* supply the cancellation — and why we lack it

For a per-event/global-scalar structure, the local gradient of the numerator's effective redshift
prior at `z*(h) = dist_to_redshift(d_meas_i, h)` is cancelled only by a denominator that **tracks
that same local gradient**. The only object that does so is the **local** selection
`∫ p_det^GW(d_L(z,h)) p_red(z) dz` over the event ball — but with `p_det ≈ 1` this is just the local
catalogue count `Σ w_g`, which reintroduces the rising `dV_c` density and rails at **all**
`sigma_z` (the `consistent_denom` control, measured 0.8700/0.8700, gate FAIL). So the cancellation
must come from **selection variation**, which we do not have, or from **ensemble coherence** through
the shared `S` — which §5.1 shows is `O(1/N_gal) × Var[p_det] ≈ 0`. **There is no object in the
adapted likelihood that supplies the missing local cancellation when `p_det ≈ 1`.** This is the
honest, mechanism-level statement of the obstruction.

### 5.4 The strongest argument AGAINST de-railing (do not oversell)

- **Information-starvation, not normalisation.** With `sigma_z ≈ 17 × sigma_z^GW`, the photo-z
  dominates the GW by `~17×`; per-event host localisation is genuinely lost. No literature method
  has demonstrated de-railing at `sigma_z/z ~ 0.7`: Hitchhiker validated the hierarchical fix only
  to `delta z/z ≈ 3%` and demonstrated the failure on the **sparse one-galaxy** axis (Eq. 33), not
  our many-galaxy axis; Cross-Parkin requires `sigma_z(photo) ≲ sigma_z^GW`. The corner is outside
  every validated range.
- **The gate proof is also the bad-news proof.** Precisely because the candidate reduces exactly to
  the (railing) standard form as `sigma_z -> 0`, and the only deformation away from it scales as
  `O(sigma_z^2 p_det'')` with `p_det'' ≈ 0`, there is no large lever between the gate limit and our
  `sigma_z`. A clean gate here is evidence the de-rail will be weak, not strong.
- **Degeneracy with already-falsified variants.** §5.2 shows `(★★) ≈` the measured 0.87-rail
  variant. Unless the prototype reveals a non-negligible `D_sm − D`, this is the expected outcome.

### 5.5 The (narrow, honest) hope

The one component **not** guaranteed to vanish is the photo-z **symmetrisation** `D_sm` vs `D`.
The closure's `p_det^GW` (`_p_det_of_dl`) has a soft logistic edge; a non-trivial fraction of the
catalogue's photo-z-scattered `z_cat_g` (with `sigma_z = 0.035`) can probe the rising part of that
edge, so `Var_g[p_det^GW]` and `p_det''` may be small-but-nonzero rather than exactly zero — and
the **direction** of `D_sm − D` (smearing pushes selection mass toward higher `d_L` where `p_det`
falls, lowering `D_sm` at high `h`) is the right sign to pull the 0.87 rail back toward 0.73.
Whether the magnitude suffices is **purely quantitative** and must be measured; the mechanism does
**not** promise it. [CONFIDENCE: the gate is HIGH; the de-rail is MEDIUM-leaning-NEGATIVE — most
likely a degenerate ~0.87 rail, with a residual chance the symmetrisation moves it interior.]

---

## 6. Dimensional analysis

Units: `z` dimensionless, `d_L` in Gpc, `sigma_z` dimensionless, `sdL` in Gpc, `w_g` an arbitrary
relative rate weight `[w]` (cancels overall in the normalised h-posterior), `p_det^GW`
dimensionless in `[0,1]`.

| quantity | expression | units | note |
|---|---|---|---|
| `L_GW(x_i|d_L)` | `N(d_meas_i; d_L, sdL_i)` | `Gpc^{−1}` | Gaussian in `d_L` |
| `p_bg(z)` | `(dV_c/dz)/(1+z)` normalised | (per unit `z`) = `1` | PDF in dimensionless `z` |
| `L_red(z_cat|z)` | `N(z; z_cat, sigma_z)` | (per unit `z`) = `1` | Gaussian in `z` |
| `Z_g` | `∫ L_red p_bg dz` | `1` | normaliser of `p_red` |
| `p_red(z|z_cat_g)` | `L_red p_bg / Z_g` | (per unit `z`) = `1` | proper z-PDF, `∫ = 1` |
| `Ñ_i(h)` | `Σ_g w_g ∫ L_GW p_red dz` | `[w]·Gpc^{−1}·1·1 = [w] Gpc^{−1}` | |
| `D_sm(h)` | `Σ_g w_g ∫ p_det^GW p_red dz` | `[w]·1·1 = [w]` | |
| `Ñ_i / D_sm` | per-event factor | `Gpc^{−1}` | **matches a density in `d_meas_i`** ✓ |
| `∏_i Ñ_i / D_sm^{N_obs}` | full posterior `∝` | `Gpc^{−N_obs}` | one `Gpc^{−1}` per detected distance ✓ |

Numerator and denominator are dimensionally consistent: each event contributes `Gpc^{−1}`
(a likelihood density in its distance datum), the `[w]` rate-weight units cancel between `Ñ_i` and
`D_sm`, and the overall `Gpc^{−N_obs}` is the correct dimension for `N_obs` independent distance
measurements. The shared-latent prior factors `∫ ∏_k dz_k L_red p_bg` are individually
dimensionless (`∫ dz · 1 · 1`), so they do not alter the dimension — consistent with `(★)` and
`(★★)` agreeing dimensionally.

---

## 7. Bridge implementation spec (prototype only — do NOT touch `bayesian_statistics.py`)

**Where.** `scripts/bridge_closure/_rungI_verify_B.py`, function `run_closure_photoz`, behind a new
boolean flag `hierarchical_shared_latent=False` (mutually exclusive with the existing
`regularised_kernel` / `global_voldecount` / `consistent_denom` flags).

**What to compute.**
1. **Numerator** = the existing `regularised_kernel` path: replace the bare Gaussian `nm` by the
   regularised posterior `nm * pbg / Z_g` (already implemented at lines ~97–101). This supplies
   `p_red` (the one `dV_c`).
2. **Denominator (the new ingredient)** = a **global, photo-z-smeared** selection scalar `D_sm(h)`,
   computed **once per `h`** over the **whole** catalogue (or a fixed representative subsample for
   speed), replacing `gd = gdenom[h]`:
   ```
   # per-h precompute, over all catalogue galaxies g (vectorised in blocks):
   #   D_sm(h) = Σ_g w_g ∫ p_det^GW(d_L(z,h)) · p_red(z | z_cat_g) dz
   # using the SAME N(z; z_cat_g, sigma_z) kernel and the SAME p_bg as the numerator.
   ```
   Use a shared `z`-grid spanning the catalogue support; reuse `B._p_det_of_dl`,
   `B.comoving_volume_element`, `dist_vectorized`. Note this is the **global** smeared selection,
   distinct from the failed **local** `consistent_denom` (which integrates only over the event ball
   and reintroduces `Σ w_g`).
3. Assemble `L_cat,i = Ñ_i / D_sm(h)`; `logpost[i] = Σ_i log L_cat,i` (the `D_sm^{N_obs}` power is
   the product over events, unchanged).

**Complexity.** Precompute `D_sm`: `O(n_h × N_gal × n_zgrid)`. For the reduced settings
(`n_h = 28`, `N_gal = 12000`, `n_zgrid ~ 60`) this is `~2×10^7` evaluations — seconds, comparable to
`precompute_global_catalog_selection` plus one inner z-integral. Numerator cost is unchanged from
the existing `regularised_kernel` run. No GPU needed.

**Gate + de-rail test (the acceptance criterion).** Run `run_closure_photoz(0.73, sz, seed=1,
hierarchical_shared_latent=True)` for `sz ∈ {0.002, 0.035}`:
- **GATE (must pass):** `sz = 0.002 -> h ≈ 0.73`, peaked, matching standard `0.7438` (±0.01).
  Predicted PASS by §4.
- **DE-RAIL (the open question):** `sz = 0.035 -> ?`
  - **Honest prediction:** interior failure — likely `h ≈ 0.87` (degenerate with the measured
    Angle-A/C/B rail), because `D_sm − D = O(sigma_z^2 p_det'') ≈ 0` (§5.2).
  - **Success (the narrow hope):** an **interior** peak near `0.73`, strictly between the 0.60 and
    0.87 rails, would prove the photo-z symmetrisation `D_sm` supplies a non-negligible
    cancellation. Report `h_refined`, `railed`, and `D_sm(h)/D(h)` across the grid to **measure**
    the smearing residual directly (this number is the whole ballgame).
- **Diagnostic to log:** `Var_g[p_det^GW(d_L(z_cat_g,h))]` and `(D_sm − D)/D` vs `h`. If both are
  `< 10^{−3}` the degeneracy of §5 is confirmed empirically and the honest project output is the
  spec-z forecast arm plus a caveated GLADE limitation (GAP-ANALYSIS §6).

**Optional second rung (only if `(★★)` degenerates but you want to exhaust `(★)`):** Monte-Carlo the
full shared-latent `(★)` — draw `L` realisations of `{z_g}` from the per-galaxy `p_red`, evaluate
`∏_i Ñ_i({z_g})/S(h,{z_g})^{N_obs}` directly (no mean-field), average over draws. This restores the
Jensen/cross-term corrections `(★★)` drops; §5.1 predicts they change nothing at `p_det ≈ 1`, so a
null result here would be the definitive information-starvation verdict.

---

## 8. Summary table — candidate vs the falsified search space

| variant | numerator prior | denominator | `sz=0.002` | `sz=0.035` | status |
|---|---|---|---|---|---|
| STANDARD (current) | bare `N` (net dV_c 0) | global POINT `D(h)` | 0.7438 ✓ | 0.6000 rail DOWN | baseline |
| Angle A/C | `p_red` (dV_c once) | global POINT `D(h)` | 0.7478 ✓ | 0.8700 rail UP | falsified |
| Angle B | `p_red` (de-count) | global POINT `D(h)` | 0.7439 ✓ | 0.8700 rail UP | falsified |
| consistent_denom | any | LOCAL `∫ p_det N` | 0.8700 rail | 0.8700 rail | gate FAIL |
| **CANDIDATE (★★)** | **`p_red` (dV_c once)** | **global SMEARED `D_sm(h)`** | **~0.74 (predicted ✓)** | **~0.87 predicted (hope: interior)** | **to test** |

The candidate is the **only** point in the table that changes the **global** denominator
(point -> photo-z-smeared) while keeping the partition norm intact. It is distinct from every
falsified variant, it provably passes the gate, and its de-rail outcome hinges on a single
measurable number `D_sm(h)/D(h)` whose mechanism analysis says is `≈ 1` here — hence the honest
"likely degenerates" verdict.

---

## Sources

- Gair, Ghosh, Gray, et al., *"A Hitchhiker's Guide to the Galaxy Catalog Approach for Dark Siren
  Gravitational-wave Cosmology"*, arXiv:2212.08694 — Eq. 13, 16, 17, 30, **31** (full shared-latent
  likelihood + commentary), 32, 33 (one-galaxy); Sec. 3.3 (one-galaxy photo-z bias). Verified via
  ar5iv 2026-06-30.
- Gray, Gair, et al. (GWcosmo), arXiv:2308.02281 — Eq. 2.2 (`[∫ p(D_GW|θ,Λ)p(θ|Λ)dθ]^{−N_det}`
  ensemble normalisation; "taken outside the product"); footnote 10 (posterior-vs-likelihood fork).
  Verified via ar5iv 2026-06-30.
- Project: `.planning/derivation-photoz-incatalog/GAP-ANALYSIS.md`, `NORMALISATION-FIX.md`,
  `CATALOG-INTERPRETATION.md`; `scripts/bridge_closure/rung_I_prior_domination.py`,
  `_rungI_verify_B.py`, `_bridge_lib.py`.

---

## Verification addendum (independent adversarial verifier, 2026-06-30)

An independent verifier checked the two load-bearing claims. Net: **the σ_z→0 gate is CONFIRMED,
but this document's "likely degenerates to 0.87" verdict is REFUTED — in the optimistic direction.
The honest call is a genuine COIN-FLIP, and the recommendation is PROCEED to prototype.**

- **GATE — CONFIRMED (HIGH).** The candidate reduces rigorously to standard global Option-A as
  σ_z→0 (p_red→δ, dV_c cancels, `D_sm→D(h)`). Crucially `D_sm` is a GLOBAL sum, so unlike the
  gate-FAILING *local* `consistent_denom` (an event-window count) it collapses to the true `D(h)`.

- **DE-RAIL pessimism — REFUTED.** §5.2's dismissal "`(D_sm−D)/D = O(σ_z²·p_det'') ≈ 0`" is
  **invalid**: it Taylor-expands around the *event hosts* (z~0.05, p_det flat), but `D` and `D_sm`
  are GLOBAL sums **dominated by catalogue EDGE galaxies** (z~0.15–0.25, where `dV_c/(1+z)` peaks
  against the `p_det` cutoff). At the edge, σ_z=0.035 → σ_dL≈0.14 Gpc **>** the logistic width
  0.096 Gpc, so the photo-z smearing of the selection edge is **ORDER ONE** and h-dependent — a real
  lever the per-event/local "p_det≈1" argument missed. The "p_det≈1 ⇒ Var[p_det]≈0" claim conflates
  the event windows with the edge-dominated global denominator (whose Var IS large).

- **MECHANISM — relabeled.** This is **NOT** Hint-2 ensemble/hierarchical coherence: the cross-event
  coherence genuinely vanishes via the 1/N_gal "space is big" suppression (robustly, regardless of
  p_det). The surviving lever is the **mean-field GLOBAL photo-z-smeared same-kernel SELECTION**
  `D_sm(h)` — a population-level Hint-1-flavoured symmetrisation, distinct from every falsified
  variant (the frozen point `D`, the local `consistent_denom`, the unsmeared Angle-A/C/B).

- **SURVIVING OBSTRUCTION (sound).** A global per-h scalar still cannot track the LOCAL numerator
  gradient at z*(h), so de-rail is **not guaranteed** even with an order-one `D_sm−D`. Hence
  coin-flip, settled only by measurement.

- **CORRECTED DIAGNOSTICS for the prototype** (the absolute `(D_sm−D)/D` cancels in the normalised
  posterior — do not gate on it): log **`d/dh log(D_sm(h)/D(h))` vs `d/dh log Ñ_i(h)`** (the
  h-GRADIENT is the whole ballgame), and the **edge-galaxy (z>0.12) fractional contribution** to
  `D_sm` and to `(D_sm−D)`. Keep σ_z=0.002→~0.74 as a mandatory pass/abort gate. **Do NOT** pursue
  the optional full-(★) Monte-Carlo rung — 1/N_gal kills the coherence regardless.

- Minor: dimensional analysis CONFIRMED; a §5.5 sign-wording slip ("lowering D_sm at high h" — the
  net effect is `D_sm/D` RISING with h) does not change the claimed final sign; Hitchhiker eq. numbers
  were taken on the derivation's authority (not independently re-fetched in verification).
