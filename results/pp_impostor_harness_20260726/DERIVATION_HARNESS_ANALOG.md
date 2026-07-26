# P–P harness impostor-ball universe and the production-stack analog — derivation note

**Date:** 2026-07-26 · **Branch:** `feat/pp-impostor-harness` (branched from
`physics/absolute-mass-marginal`) · **Module:**
`master_thesis_code/validation/pp_coverage.py`
**Status:** derived + implemented + unit-tested + smoke-run. `pp_coverage.py` is not on the
`/physics-change` trigger list, but this note applies the same protocol: generative model stated
first, estimator derived from it, dimensional analysis, limiting cases, and a correspondence
table naming every production term and every remaining simplification.

**Inputs.** `results/lcat_h_dependence_20260725/DERIVATION_ESTIMATOR_REDESIGN.md` (V1,
`absolute_marginal`), `.../DERIVATION_GENERATOR_CONSISTENT_NORM.md` (FIX-3,
`generator_marginal`), `.../DERIVATION_ZRESOLVED_SURVIVAL.md` (FIX-2, `S(d_L|z)`),
Gray et al. 2020 (arXiv:1908.06050) Eqs. 29/32 and A.9/A.10, Mandel, Farr & Gair 2019
(arXiv:1809.02063).

---

## 0. The gap this note closes

The P–P/coverage harness generates universes in which **every detected event has exactly one
candidate host** — its own photo-z-scattered redshift. Commit `7c513dd` added
`mixture_mode="absolute"` (the V1 harness analog) and immediately hit the wall: the mode showed
no coverage or bias change in any of three cells, because *the harness has no mechanism that V1
was designed to remove.* V1's stated purpose (`DERIVATION_ESTIMATOR_REDESIGN.md` §3.4(b)) is to
stop a candidate ball that contains **only foreground/background impostors** from carrying O(1)
in-catalogue weight through the self-normalized ratio-of-sums `L_cat = Σ_ball w N / Σ_ball w D_g`.
A one-candidate universe cannot build such a ball.

This note derives the harness extension that can: a discrete, frozen, shared galaxy catalogue
plus hard sky-localization balls, and the three estimators (`lcat`, `absolute`,
`generator_marginal`) that operate on it.

---

## 1. Harness generative model (catalogue mode)

All of the following is drawn by the harness and none of it is visible to the estimator except
where stated.

**(G1) Galaxies.** `n_galaxies` galaxies with true redshifts drawn from the comoving-volume
**number** density

```
n_gal(z) ∝ dV_c/dz  =  (1+z) · w_pop(z)          on [Z_MIN, Z_MAX_POP]
```

and directions uniform on the sphere. Here `w_pop(z) ∝ (dV_c/dz)/(1+z)` is the harness's
pre-existing population **rate**-weight (`population_weight_of_z`). The split is the physical
one: constant comoving galaxy density × per-galaxy EMRI rate suppression.

**(G2) Per-galaxy rate weight.**

```
w(z) = 1/(1+z)          (observer-frame time dilation)
```

so `n_gal(z) · w(z) ≡ w_pop(z)` identically. This is the harness analog of production's
`w_g = R_eff_per_mbh(M_g)/(1+z_g)` with the mass factor dropped (no mass dimension here).

**(G3) Completeness.** A galaxy is *catalogued* iff `z_true < z_support`, i.e. the harness's
sky-averaged completeness is the hard step `f̄(z) = 1[z < z_support]`. Catalogued galaxies carry
an observed redshift `z_obs = z_true + N(0, σ_z)` (optionally clamped at `Z_MIN`, existing
`clamp_zgal`). The estimator sees `{z_obs, direction}` for catalogued galaxies and the value
`z_support`; it never sees `z_true`, the catalogued/uncatalogued flag of the *host*, or which
ball member is the host.

**(G4) Hosts and detection.** A host galaxy is drawn from **all** galaxies with probability
∝ `w(z_g)`, then detected with probability `p_det(A(z_g)/h_true)`. The detected-host redshift
density is therefore

```
∝ n_gal(z) w(z) p_det(A(z)/h_true) = w_pop(z) p_det(A(z)/h_true),
```

**identical to the continuum harness's `_sample_detected_redshifts`.** (Unit-tested by a
two-sample KS test.) This is what makes catalogue mode a controlled extension rather than a
different experiment.

**(G5) GW data.** `d_L^obs = d_L^true + N(0, σ_f d_L^true)` (unchanged), plus a **sky datum**:
a spherical cap of solid-angle fraction `Ω_frac = ΔΩ/4π` (`sky_frac`). The cap centre is drawn
uniformly inside the cap of the same half-angle about the true host direction. By the symmetry
of the "within angle θ_c" relation this makes the host **uniform inside the cap given the
centre**, so the flat in-cap sky likelihood

```
p(sky data | direction n̂)  ∝  1[n̂ ∈ cap]          (normalized: 1/ΔΩ inside)
```

is **exact**, not an approximation. The candidate ball `B_i` is every *catalogued* galaxy inside
the cap. When the host is uncatalogued, the ball contains impostors only (or is empty) — the
misassociation configuration production faces at `z > z_support`.

---

## 2. The estimator, derived

### 2.1 Per-galaxy measure

A catalogued galaxy `g` is an object of unknown true redshift with prior `n_gal(z)` (G1) and
likelihood `N(z_obs,g; z, σ_z)` (G3). Its normalized true-z posterior is

```
p(z | z_obs,g) = n_gal(z) N(z_obs,g; z, σ_z) / Z_g ,
Z_g = ∫_{Z_MIN}^{Z_MAX_POP} n_gal(z) N(z_obs,g; z, σ_z) dz .
```

Multiplying by the host-selection rate weight (G2) defines the **per-galaxy rate-weight measure**

```
dμ_g(z) = p(z | z_obs,g) · w(z) dz .                                          (1)
```

Every catalogue object in the estimator is an integral of (1):

```
w_g            = ∫ dμ_g                              (galaxy's expected rate weight)
N_g(h)         = ∫ dμ_g p_GW(d_L^obs,i | A(z)/h)     (GW-data density; MFG: no p_det)
D_g(h)         = ∫ dμ_g p_det(A(z)/h)                (per-host selection denominator)
W_cat          = Σ_{g ∈ cat} ∫ dμ_g
Σ_glob(h)      = Σ_{g ∈ cat} ∫ dμ_g p_det(A(z)/h)
```

**σ_z→0 limit:** `dμ_g → δ(z − z_obs,g) w(z_obs,g) dz`, so `Σ_glob → Σ_g w_g p_det(A(z_g)/h)` —
exactly production's point-evaluated `precompute_global_catalog_selection`. The harness realizes
the **(σ_z-kernel numerator ↔ σ_z-smeared Σ_glob)** member of the two internally consistent
pairings named in `DERIVATION_GENERATOR_CONSISTENT_NORM.md` §4.3; production, whose mock
catalogue redshifts *are* the true redshifts, realizes the (point ↔ point) member. The choice is
**forced here**, not selected: the harness's catalogue redshifts are genuinely noisy, so `z_true`
is not an observable and the point form is not available to the estimator.

Note the kernel difference from the single-candidate modes. There, `z_gal` is the noisy
observation of an object *already selected as a detected host*, so its prior is the host
population `w_pop` — hence the existing `kernel="volume"` normalization `N·w_pop/∫N·w_pop`. Here,
ball members are *catalogue entries*, not hosts, so the prior is `n_gal` and the host-selection
weight `w(z)` multiplies afterwards. Both follow from the same rule (prior = density of the
object being conditioned on); they differ because different objects are being conditioned on.
This is why catalogue mode rejects `kernel="bare"` and does not reuse the `"volume"` branch.

### 2.2 Marginal over the host

Let the model universe at hypothesis `h` be: the same frozen catalogue, the same `f̄`, cosmology
at `h`. The uncatalogued ("dark") population has rate-weight density `n̂_w (1−f̄(z)) w_pop(z)`
per unit `z`, isotropic on the sky, where `n̂_w` is the absolute rate-weight density per unit
`w_pop`-volume (fixed in §2.3). With the MFG convention (one selection factor dividing the
marginal, no `p_det` inside the numerator; Mandel, Farr & Gair 2019):

```
numerator_i(h) = Σ_{g∈cat} w_g N_g(h) · [1/ΔΩ] 1[g ∈ cap_i]
               + ∫dΩ ∫dz  n̂_w (1−f̄) w_pop(z) /(4π) · p_GW(z;h) · [1/ΔΩ] 1[Ω ∈ cap_i]
               = (1/ΔΩ) Σ_{g∈B_i} w_g N_g(h)  +  (1/4π) n̂_w ∫(1−f̄) w_pop p_GW dz .
```

Multiplying through by `4π/n̂_w` (an `i`-independent, `h`-independent constant that cancels
against the same factor in the normalization) gives the implementable form

```
p_i(h) = [ A_i(h) + B_num,i(h) ] / Den(h) ,                                    (2)

A_i(h)     = Σ_{g∈B_i} w_g N_g(h) / ( n̂_w · Ω_frac ) ,                        (3)
B_num,i(h) = ∫ (1−f̄(z)) w_pop(z) p_GW(d_L^obs,i | A(z)/h) dz ,                (4)
```

with (4) *bit-identically* the harness's pre-existing `_completion_numerator` (its integration
window `[max(Z_MIN, z_support, z_GW,lo), min(Z_MAX_POP, z_GW,hi)]` is exactly `(1−f̄)` times the
GW support).

**The `Ω_frac` factor is the harness analog of production's pixel solid angles.** Its role is
fixed by the tiling requirement, not chosen: since the expected in-cap catalogue rate weight is
`Ω_frac · n̂_w · f̄(z) w_pop(z) dz`,

```
E_cap[ A_i(h) ] = ∫ f̄(z) w_pop(z) p_GW(z;h) dz ,                              (5)
```

so `A_i + B_num,i` has expectation `∫ w_pop p_GW dz` — **independent of `z_support`**. That is
the dimensional-consistency statement of this mode: the discrete catalogue sum and the continuum
completion integral are on the same absolute scale, and the two terms tile the population
support `[Z_MIN, Z_MAX_POP]` exactly. (Verified numerically, §5.1.)

### 2.3 The three normalizations

`Den(h)` is the probability that a draw at hypothesis `h` is detected, in the same reduced
(`×4π/n̂_w`) convention:

```
Den(h) = Σ_glob(h)/n̂_w + β_Ḡ(h) ≡ D_gen(h) ,
β_G(h)  = ∫_{Z_MIN}^{z_support} p_det(A(z)/h) w_pop(z) dz ,
β_Ḡ(h) = ∫_{z_support}^{Z_MAX_POP} p_det(A(z)/h) w_pop(z) dz = D(h) − β_G(h) .
```

The three implemented modes differ **only** in how the discrete sum is scaled and which
denominator is used:

| `mixture_mode` | in-catalogue term | denominator |
|---|---|---|
| `lcat` | `β_G(h) · (Σ_ball w N)/(Σ_ball w D_g)` | `D = β_G + β_Ḡ` |
| `absolute` | `(Σ_ball w N)/(n̄_w(h) · Ω_frac)`, `n̄_w = Σ_glob/β_G` | `D = β_G + β_Ḡ` |
| `generator_marginal` | `(Σ_ball w N)/(n̂_w · Ω_frac)`, `n̂_w = W_cat/V_f` | `D_gen = Σ_glob/n̂_w + β_Ḡ` |

with `V_f = ∫ f̄(z) w_pop(z) dz`. `lcat` is the Gray-A9 self-normalized ratio-of-sums that
production's default `volume_deconv` implements; `absolute` is production `absolute_marginal`
(V1, Option-A calibration `n̄_w = Σ_glob/β_G`); `generator_marginal` is the FIX-3 stack, Eqs.
(3)–(5) of `DERIVATION_GENERATOR_CONSISTENT_NORM.md`.

**Empty balls** give `Σ_ball w N = 0` in all three, so `p_i = B_num/Den` emerges as the
continuous limit, not a separate `#29` branch — the same structural property V1 claims.

### 2.4 Where the `h³` went

Production's `n̂_w(h) = W_cat/V_f(h)` scales as `h³` because `V_f` is a physical comoving volume
(`∝ (c/H₀)³`). The harness's `population_weight_of_z` deliberately drops the common
`(c/100)³h⁻³`. This is legitimate here and in production for the same reason: **every one of the
four terms in (2) — `A_i`, `B_num,i`, `Σ_glob/n̂_w`, `β_Ḡ` — is homogeneous of degree one in
`w_pop`.** Hence any multiplicative function `c(h)` of `w_pop` cancels between numerator and
denominator of `p_i`, including `c(h) = h⁻³`. Consequences: (i) `n̂_w` is `h`-INDEPENDENT in the
harness while production's carries `d ln n̂_w/dh = 3/h`, and the two give identical posteriors;
(ii) the harness must nonetheless keep the `n_gal`/`w_pop` bookkeeping straight, since `W_cat` is
a discrete rate-weight sum and `V_f` a `w_pop` integral — their ratio depends on the `w_pop`
normalization convention, and only the degree-one homogeneity makes the dependence cancel. This
is unit-tested (§5.2) by rescaling the module's `_W_POP` table and demanding bit-identical
output.

---

## 3. Production ↔ harness correspondence table

| Production object (`bayesian_statistics.py` / derivation packets) | Harness object (`pp_coverage.py`) | Simplification |
|---|---|---|
| GLADE+ pruned catalogue rows | `SyntheticCatalogue.z_true/direction/z_obs` | drawn from the harness's own `n_gal(z)`; no magnitudes, no clustering |
| `w_g = R_eff_per_mbh(M_g)/(1+z_g)` | `w(z) = 1/(1+z)`, `host_rate_weight_of_z` | mass factor dropped (no mass dimension) |
| pixelated completeness `f_k(z)`, `f̄(z)` from `m_th` map | `f̄(z) = 1[z < z_support]` | hard sky-uniform step; no pixelation |
| BallTree sky query → candidate ball `B_i` | `cKDTree` over 3D unit vectors, chord radius from `sky_frac` | hard cap, flat in-cap sky likelihood (exact by construction, §G5) |
| pixel solid angle bookkeeping in `A_i` | scalar `Ω_frac = sky_frac` in Eq. (3) | one global cap size for all events |
| `N_g(h)` = `single_host_likelihood` (σ_z kernel over `d_L`, sky, `M_z`) | `N_g(h) = ∫ dμ_g p_GW(d_L^obs \| A(z)/h)` | 1-D in `d_L` only; sky handled by the hard ball, no `M_z` channel |
| `Σ_glob(h)` = `precompute_global_catalog_selection` (point-evaluated) | `Σ_glob(h) = Σ_g ∫ dμ_g p_det` (σ_z-smeared) | the *other* consistent σ_z pairing of §4.3; forced by the harness's noisy catalogue |
| `W_cat = Σ_{z_g<1.5} w_g` | `W_cat = Σ_g ∫ dμ_g` | same object, smeared measure |
| `V_f(h) = ∫ f̄ (dV_c/dz)/(1+z) dz` | `V_f = ∫ f̄ w_pop dz` | common `h⁻³` dropped (§2.4) |
| `n̂_w(h) = W_cat/V_f(h)` | `n_hat_w = W_cat/V_f` | `h`-independent here (§2.4) |
| `n̄_w(h) = Σ_glob(h)/β_G(h)` (Option A) | identical formula | — |
| `β_Ḡ(h)` = `precompute_missing_completion_denominator` | `∫_{z_support}^{Z_MAX_POP} p_det w_pop dz` | sky-uniform |
| `D(h) = β_G + β_Ḡ` | identical | — |
| `D_gen(h) = Σ_glob/n̂_w + β_Ḡ` | identical | 4D-vs-3D `p_det` convention question is moot (no `M_z` axis) |
| `B_num,i(h) = ∫ (1−f_k) p_GW,iso p_pop dz` | `_completion_numerator` (pre-existing) | sky-uniform, no pixel |
| `L_cat = Σ_ball w N / Σ_ball w D_g` (Gray A9) | `mixture_mode="lcat"` | — |
| `absolute_marginal` (V1) | `mixture_mode="absolute"` | — |
| `generator_marginal` (FIX-3) | `mixture_mode="generator_marginal"` | — |
| `S(d_L\|z)` z-resolved survival (FIX-2) | `detection_probability(A(z)/h)` | **vacuous — see §4** |
| oracle selection / SNR≥20 threshold | probabilistic `p_det(d_L)` erfc roll-off | latent-detection convention already covered by the existing `pdet_in_numerator` probe |

---

## 4. FIX-2 (z-resolved survival) is vacuous in this harness — and that is the correct analog

Production's FIX-2 replaces the **pooled** survival `S(d_L) = P(d_hor ≥ d_L)` by the
z-conditional `S(d_L | z)`. The pooled form is wrong because the injection pool's horizon
distribution is z-dependent: the detector-frame mass lift `M_z = M(1+z)` drags the median horizon
0.89 → 1.59 Gpc across `z ≈ 0.18 → 0.9`, so the marginal survival overestimates the true SNR≥20
rate at fixed z by +30–45 % (`DERIVATION_ZRESOLVED_SURVIVAL.md` §1–2).

In this harness the detection probability is the **exact deterministic function**
`p_det(d_L) = ½ erfc((d_L − d₅₀)/(√2 w))` of luminosity distance alone. There is no intrinsic
parameter scatter, no horizon distribution, and no `M_z` lift, so

```
S_harness(d_L | z) = p_det(d_L) = S_harness(d_L)   for every z,
```

i.e. the pooled and z-conditional survivals coincide **identically**. The harness therefore sits
in FIX-2's *fixed* state by construction, and the faithful analog of "production + FIX-2" is
simply the harness's existing `p_det(A(z)/h)` evaluated at the integration redshift — which is
what every selection integral (`D`, `β_G`, `β_Ḡ`, `Σ_glob`, `D_g`) already does. Implementing a
"pooled" alternative would require first *manufacturing* a horizon-scatter + `(1+z)` lift
mechanism that the harness's generative model does not have; that is a different experiment
(deliberately out of scope) and is recorded as the top open item in §6.

---

## 5. Checks

### 5.1 Absolute-scale / tiling identity (Eq. 5)

Averaging `A_i(h)` over uniformly placed caps at fixed `(d_L^obs, σ_dL)` against the analytic
`∫_{Z_MIN}^{z_support} w_pop p_GW dz`, with `n_galaxies = 3·10⁵`, `sky_frac = 2·10⁻³`,
`z_support = 0.30`:

| σ_z | mean ratio over the h grid |
|---|---|
| 0.001 | 0.995 |
| 0.035 | 0.993 |

The residual few-per-mille deficit is the **above-edge kernel leak** (§6 item ii), not a scale
error: the per-galaxy posterior (1) is deliberately *not* truncated at `z_support`
(production-faithful), so a catalogued galaxy near the edge puts posterior mass above it, where
`w(z)` is smaller. Measured directly: `n̂_w` sits 1.5 % below the analytic galaxy density
`N_gal/∫n_gal dz` at `σ_z = 0.035, z_support = 0.30`, and → 0 as σ_z → 0. The per-h scatter (±3 %
at σ_z = 0.001) is fixed-catalogue shot noise: it does **not** shrink when the number of sampled
caps is raised ×10, confirming it is catalogue realization noise rather than a bias.

### 5.2 Homogeneity / normalization invariance (§2.4)

Rescaling the module `_W_POP` table by an arbitrary constant leaves every catalogue-mode result
bit-identical (unit test `test_generator_marginal_wpop_normalization_invariance`), confirming
degree-one homogeneity and hence the cancellation of `h⁻³`.

### 5.3 Complete-catalogue exact identity

At `z_support ≥ Z_MAX_POP`: `β_Ḡ = 0`, `B_num = 0`, and

```
absolute:            (Σ w N)/(n̄_w Ω_frac) / β_G  =  (Σ w N) β_G /(Σ_glob Ω_frac β_G) = (Σ w N)/(Σ_glob Ω_frac)
generator_marginal:  (Σ w N)/(n̂_w Ω_frac) / (Σ_glob/n̂_w) = (Σ w N)/(Σ_glob Ω_frac)
```

so the two modes are **algebraically identical** there. Verified numerically to machine precision
in the log-likelihood (unit test).

### 5.4 Generative-model consistency

Catalogue-mode detected-host redshifts are KS-consistent with the continuum harness's
`_sample_detected_redshifts` at the same `h_true` (unit test).

### 5.5 Option-A compliance (a harness *limitation*, measured)

Because the harness catalogue is drawn from exactly the density the estimator models, production's
Option-A identity `Σ_glob = n̂_w β_G` nearly holds here. Measured at `z_support = 0.30`,
`n_galaxies = 4·10⁵`, `σ_z = 0.035`: `n̄_w/n̂_w ∈ [0.919, 0.986]` across the h grid — an 8 %
`h`-dependent residual sourced entirely by the σ_z asymmetry (`Σ_glob` smeared vs `β_G` a point
model integral) and the above-edge leak, not by catalogue structure. **The harness therefore
cannot adjudicate FIX-3 against V1 on catalogue-structure grounds** (the real GLADE+ catalogue
violates Option A far more strongly); it *can* measure the σ_z-asymmetry channel, which is the
f9c58f4 `--smear_global_selection` motivation.

### 5.6 End-to-end: unbiased under maximal impostor load

The strongest check is the estimator's own output. With the completion fraction driven to
exactly zero (`z_support = 0.60` and `0.95`) but the candidate balls at their *most*
impostor-dominated (14.3 candidates/ball at 93 % impostors; 41.1 at 97.6 %), the
`generator_marginal` MAP bias at `h_true = 0.72` is `−0.0006` and `+0.0024` against a
`sd/√n = 0.002` standard error, with 68 % coverage 0.70 and 0.63 against a nominal 0.68
(`SMOKE_SUMMARY.md`, `zsupport_sweep.json`). Since the `n̂_w · Ω_frac` normalization has no
free parameter, an error in the absolute scale of Eq. (3) would have shown up directly as a
bias in exactly these rows. The residual bias at finite completion fraction (+0.021 at
`z_support = 0.30`, +0.049 at 0.15) is therefore attributable to the `B_num` completion
channel, not to the discrete-catalogue term.

---

## 6. Remaining simplifications (honest list)

1. **No FIX-2 content** (§4): no horizon scatter, no detector-frame mass lift, hence no
   pooled-vs-conditional survival discrepancy to repair. To exercise FIX-2 the harness would need
   a latent per-event horizon `d_hor,k` with a `(1+z)^κ` lift and a pooled-survival estimator
   option. Not implemented.
2. **Above-edge kernel leak kept** (§5.1): the per-galaxy posterior is not truncated at
   `z_support`, exactly as production does not truncate. The *exact* posterior would truncate
   (catalogue membership is informative about `z`). Keeping the leak is the production-faithful
   choice and lets the harness measure its cost; `mixture_mode="exact"` studies the truncated
   alternative in the single-candidate universe.
3. **Option-A-compliant catalogue** (§5.5): the harness cannot manufacture the real catalogue's
   density/completeness mismatch.
4. **No mass dimension**: no `M_z` channel, no host-mass prior, no 2D-vs-3D likelihood channels,
   so the 3D-vs-4D `p_det` convention question inside `D_gen` (packet §7 author decision 1) is
   moot here.
5. **Hard, uniform sky ball**: one `sky_frac` for all events; no GW sky-likelihood shape, no
   sky-area/SNR correlation, no clustering (galaxies are Poisson on the sphere, so impostor
   counts are Poisson rather than clustered — this makes the harness's impostor problem *easier*
   than production's).
6. **Shared frozen catalogue by default**: catalogue shot noise is common-mode across
   realizations (production-faithful: there is one GLADE+). `resample_catalogue_per_realization`
   switches to fully independent universes at ~50× the cost.
7. **Binned `Σ_glob`/`W_cat`**: the global catalogue density `K̂(z)` is built by binning `z_obs`
   at `σ_z/16` and convolving, an O((σ_z/16)²) smoothing error far below catalogue Poisson noise;
   the *ball* numerators use exact per-galaxy Gaussians. Cross-checked against a direct
   per-galaxy sum in a unit test (rel. error < 10⁻³).

---

## 7. What the harness can and cannot decide

**Measured (`SMOKE_SUMMARY.md`):** `lcat` is worse than both absolute-mass forms in every
cell of the smoke (MAP bias +0.0349 vs +0.0232 at `h_true = 0.72`; 68 % coverage 0.415 vs
0.465; HIGH rail 0.775 vs 0.640 at `h_true = 0.84`), on identical paired universes — the
first harness evidence for V1's core claim. `absolute` and `generator_marginal` agree to 4
decimals, as §5.5 predicts. The surviving bias is completion-sourced (§5.6).

**Can:** whether `lcat`'s self-normalization damages coverage/bias in the presence of genuine
impostor balls, and whether the absolute-mass forms repair it — the V1 claim of
`DERIVATION_ESTIMATOR_REDESIGN.md` §3.4(b), which the pre-existing one-candidate harness
structurally could not test. Also: the empty-ball continuity claim, the complete-catalogue
unbiasedness claim, and the σ_z-asymmetry size in `n̄_w`.

**Cannot:** adjudicate FIX-3 vs V1 on catalogue-structure grounds (§5.5); say anything about
FIX-2 (§4); or substitute for the production-code gates (packet §6 gates 2–3, seed600/seed1000
re-evaluations).
