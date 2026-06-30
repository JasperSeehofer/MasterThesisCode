# Photo-z-marginalised in-catalogue likelihood: correct derivation and exact pipeline deviation

Status: synthesis of three independent derivation angles (A: full hierarchical; B: selection-
dominated/D(h)-scaling; C: numerator/denominator same-kernel) and their adversarial verdicts.
Dataset under test: seed-600 dark events, `/tmp/seed600_local/simulations/`, 3361 detections,
true H0/100 = 0.73, real pipeline MAP = 0.86 (rails to upper grid edge).

---

## 1. The correct likelihood (canonical catalogue method)

Single-event in-catalogue likelihood is a **single ratio with the SAME catalogue redshift density
top and bottom** (Gray et al. 2020 arXiv:1908.06050 Eq. 25; Gair/Ghosh/Gray/Fishbach/Chen et al.
2024 "Hitchhiker's Guide" arXiv:2212.08694 Eq. 3):

```
                INT dz  L_GW(x | d_L(z,h))  p_CBC(z)
  L_i(h)  =  ---------------------------------------------
                INT dz  P_det(d_L(z,h))     p_CBC(z)
```

The catalogue redshift density is built from per-galaxy redshift **POSTERIORS** (not bare
likelihoods), partitioned into in/out of catalogue (Echoes from the dark, A&A 2026,
arXiv:2509.18243 Eq. 15):

```
  p_CBC(z) = f(z) p_cat(z) + (1 - f(z)) p_bg(z)
  p_cat(z) = (1/W) Sum_g  w_g  p_red(z | z_g),     w_g = R_eff(M_g)/(1+z_g),  W = Sum_g w_g
  p_bg(z)  proportional to  (1/(1+z)) dVc/dz        [population prior, uniform in comoving volume]
```

**The crucial object** — the per-galaxy redshift POSTERIOR (Hitchhiker's Guide Eq. 16/17/32):

```
  p_red(z | z_g) = norm(z_g; z, sigma_z) * p_bg(z) / Z_g,     Z_g = INT norm(z_g; z', sigma_z) p_bg(z') dz'
                 = norm(z; z_g, sigma_z) * p_bg(z) / Z_g       (Gaussian symmetric in z<->z_g)
  norm(z_g; z, sigma_z) = (1/(sqrt(2pi) sigma_z)) exp(-(z_g - z)^2 / (2 sigma_z^2))
```

The `p_bg(z)` factor is the **comoving-volume regulariser**. It is what makes the large-sigma_z
limit degrade each host gracefully to the population prior (the SAME prior already used by the
completion B_num and the selection D(h)), instead of amplifying the raw catalogue density gradient.

### Reduction to the project's partition-norm structure

Substituting the split and marginalising sky as a per-host weight `sky_w_g` recovers the project's
per-event ratio **unchanged in structure**:

```
  p_i(h) = ( beta_G(h) * L_cat(h) + B_num(h) ) / D(h)

  L_cat = Sum_local / Sum_global
  Sum_local  = Sum_{g in cone}  w_g sky_w_g * INT p_GW(d_L(z,h))  p_red(z|z_g) dz
  Sum_global = Sum_{g, z_g<z_max}  w_g       * INT P_det(d_L(z,h)) p_red(z|z_g) dz   <-- SAME p_red kernel
  B_num   = INT (1 - f_k(z)) p_GW(d_L(z,h)) p_bg(z) dz
  D(h)    = INT P_det(d_L(z,h)) p_bg(z) dz   (= INT P_det (1/(1+z)) dVc/dz dz)
  beta_G  = D(h) - beta_Gbar(h)
  p_GW(d_L) = exp(-0.5 (d_meas/sigma_dL)^2 (d_L(z,h)/d_meas - 1)^2)
```

The correct in-catalogue ratio convolves the **IDENTICAL** regularised posterior `p_red(z|z_g)` in
BOTH numerator and selection denominator. This is the same-kernel requirement that the literature is
unanimous about (Gray 2020 Eq. 25; Hitchhiker's Eq. 3/15; arXiv:2502.17747 Eq. 3).

---

## 2. The EXACT deviation in the current pipeline

There are two deviations. Only the **first is load-bearing**; the second is a proven no-op.

### Deviation 1 (LOAD-BEARING) — bare Gaussian kernel instead of the regularised posterior

The per-host redshift PDF in the in-catalogue numerator is the **bare, volume-unregularised
Gaussian** `norm(z; z_g, sigma_z)`, missing the `p_bg(z)` comoving-volume factor and the per-host
normalisation `1/Z_g`.

- Bridge: `scripts/bridge_closure/_bridge_sky.py:246-248`
  ```python
  nm = np.exp(-0.5 * ((zgrid[None, :] - zg[:, None]) / szg[:, None]) ** 2) / (
      np.sqrt(2 * np.pi) * szg[:, None]
  )                          # = norm(z; z_g, sigma_z), NO p_bg, NO 1/Z_g
  N_dL = nm @ (gw * dzg)
  ```
- Production: `master_thesis_code/bayesian_inference/bayesian_statistics.py:~1646`
  (`single_host_likelihood` numerator convolves `norm(host_z, host_z_error)`).

**Why this biases H0 high.** The effective in-catalogue density becomes the raw smoothed catalogue
histogram `n_K(z) = Sum_g w_g norm(z; z_g, sigma_z)`, which inherits the catalogue's intrinsic
comoving-volume / clustering rise. Meanwhile the denominator `D(h)` is `p_bg`-weighted (line 174).
The two redshift densities **do not match**. In the large-sigma_z regime (sigma_z = 0.035 >>
sigma_z^GW ~ 0.002) the sharp GW factor collapses the numerator to `A(h) * n_K(z*(h))`, where
`z*(h)` is the GW-implied redshift (`d_L(z*,h) = d_meas`, increasing with h). Because `n_K` rises at
z ~ 0.046, higher h is rewarded -> MAP runs to the upper grid edge (the +0.13 rail). This is
Hitchhiker's "Inconsistency 1" (comoving-volume double-counting): the catalogue already follows
`dVc/dz`, and the bare kernel never divides it back out.

### Deviation 2 (PROVEN NO-OP) — asymmetric selection kernel

The global selection denominator `Sum_global w_g D_g` uses the POINT value `D_g = P_det(z_g)`
(`bayesian_statistics.py:386-490`), with NO convolution, while the numerator convolves sigma_z.
This *looks* like the same-kernel violation, but convolving `D_g` is a no-op for two independent
reasons, both verified:
1. The in-catalogue galaxies are nearby (z ~ 0.046 << z_horizon), so `P_det ~= 1` across the whole
   sigma_z = 0.035 window: `INT P_det norm dz ~= P_det(z_g)`.
2. Option A self-consistency: `Sum_global(h) = C * beta_G(h)` (line 490 vs line 260), so `beta_G`
   cancels and the effective normalisation of the in-catalogue term is `D(h)` alone. Leaving
   `Sum_global` unchanged is therefore correct.

Convolving the selection denominator alone does **not** remove the bias. The bias lives entirely in
the numerator's redshift prior (Deviation 1).

### What is NOT the deviation (eliminated by the bridge — do not touch)

D(h) sign/scaling (Angle B impossibility bound below), candidate ball-tree/frame (true host returned
99%, median |z_cand - z_true| = 0.000), MVN correlations, n(z) shape, measurement scatter,
completion f_k/B_num, p_det survival, the 1.5-sigma candidate radius. Each recovers 0.73 when added
alone.

---

## 3. Angle B's firm NEGATIVE result: D(h) is NOT the lever

Re-deriving, rescaling, or sign-flipping `D(h)` / `beta` **cannot** remove the rail. Proof
(impossibility bound, from `.planning/HANDOFF-RAILING-INVESTIGATION-20260629.md:16-30`):

- Summed log-likelihood deficit to be cancelled: Sum_logL 1.624M (h=0.73) -> 2.171M (h=0.86),
  i.e. **+546,650** over dh = 0.13.
- D(h) only moves 3.30M -> 3.15M, contributing N*dlogD ~= -157 (0.03% of the deficit).
- To cancel +546,650 with a rescale D -> D*(h/0.73)^p requires **p ~= 1000**.
- The physically motivated comoving-volume normalisation is p = 3, which supplies only ~1640 of the
  deficit — short by a factor ~330.

`dD/dh < 0` is correct physics (fixed-d_L^max detection horizon; every (1+z) factor and the upper
limit `z_max(h)` decrease with h). The handoff's "D(h) must increase with h" framing is a red
herring. **Do not chase D(h).** A cheap falsification leg (sweep an artificial power `D*(h/0.73)^P`)
will confirm only absurd P ~ 1000 moves the MAP.

---

## 4. Limiting cases (all three angles agree)

| Limit | Bare Gaussian (current) | Regularised posterior (proposed) |
|---|---|---|
| sigma_z -> 0 | -> delta(z - z_g), recovers delta-z MAP 0.725 | -> delta(z - z_g) (Z_g -> p_bg(z_g) cancels), **same 0.725** |
| sigma_z -> inf | -> uniform-in-z (WRONG prior, mismatches D(h)) | -> p_bg(z) = uniform in comoving volume (**matches** B_num, D(h)) -> in-cat ratio -> [INT p_GW p_bg]/[INT P_det p_bg] = Mandel-Farr-Gair empty-catalogue form, posterior -> prior, **unbiased** |
| matched density (p_cat = p_bg, f=1, P_det=1) | residual O(sigma_z^2 * curvature) bias | exactly unbiased |

**Dimensional check:** `norm` [1/z] * `p_bg` [1/z] / `Z_g` [1/z] = [1/z], unit area. `INT p_GW
(dimensionless) p_red dz` is dimensionless — identical units to the bare-Gaussian version, so `p_i`
is dimensionally unchanged.

**Double-counting guard (passes):** `w_g = R_eff/(1+z)` (per-host rate weight on discrete galaxies)
and `p_bg ∝ (dVc/dz)/(1+z)` (continuous within-host redshift prior) are distinct axes; the per-host
`1/Z_g ~= 1/p_bg(z_g)` division ensures the volume factor `(1/(1+z)) dVc/dz` enters **exactly once**.

---

## 5. Honest verdict: is the bias removable within the partition-norm structure?

**The kernel regularisation is the physically correct fix and is the unanimous recommendation of all
three angles and the entire literature.** It is structure-preserving (only the per-host redshift PDF
inside `L_cat`'s numerator changes), dimensionally clean, and provably preserves delta-z while
giving the unbiased empty-catalogue limit as sigma_z -> inf.

**BUT there is one decisive, dataset-specific caveat that the adversarial review surfaced, and it
must be stated plainly:**

> **The seed-600 bridge events were injected AT the catalogue redshift, with NO actual photo-z
> scatter.** Confirmed in `scripts/bridge_closure/_bridge_lib.py:353`:
> `z_host = dist_to_redshift_vec(d_true, TRUE_H)`. The catalogue stores the *exact* true host z but
> *labels* it with sigma_z = 0.035 (the bridge reports median |z_cand - z_true| = 0.000, true host
> returned 99%). So the statistically correct host prior for this dataset is **delta-z**, and
> convolving with ANY kernel of width 0.035 smears an actually-sharp truth.

Consequences:
- At sigma_z = 0.035 the regularised posterior `p_red` is still a localised (tilted) bump — neither
  a delta nor `p_bg` — so it does **not** restore delta-z sharpness. It does divide out the smooth
  volume overdensity of high-z candidates (the `1/Z_g ~= 1/p_bg(z_g)` reweighting), which is a
  genuine corrective piece pulling the MAP **down** from 0.857. Whether that lands at ~0.73 or merely
  at an interior-but-still-biased value (e.g. > 0.78) **cannot be settled analytically** — the
  finite-sigma residual `psi(z) * n_smooth(z) != psi(z)` still carries catalogue gradient.
- This fix shares the structural form of the already-failed "Fix-A" (a pure inference-side change).
  Fix-A failed because of the sim<->inference inconsistency (events injected from photo-z hosts).
  The kernel fix is a *likelihood correction* rather than a *catalogue cut*, so it is not identically
  Fix-A, but it does **not by itself repair the inconsistency** that the truth is delta-sharp while
  the inference assumes a 0.035-wide kernel.

**The two honest possibilities:**
1. **Partial-removability (Angle C verdict, "yes" at sigma_z=0.035):** at sigma_z/z ~ 0.5-0.7 the
   catalogue is deep in the degraded regime where the deconvolution -> p_bg -> MFG limit is
   approached; the fix de-rails to ~0.72-0.74 (possibly recovery-by-prior, i.e. the posterior
   flattens toward the prior median ~0.735 on the [0.60, 0.87] grid, with a visibly **broadened**
   posterior — photo-z costs precision, not accuracy: arXiv:2502.17747, Echoes 2509.18243).
2. **Not-removable-by-kernel-alone (Angle A verdict, "no"):** because the truth is delta-sharp,
   imposing a 0.035-wide volume-tilted prior generically re-biases high; the robust resolution is
   **sim<->inference consistency** — re-inject events with genuine photo-z scatter
   (`true z = z_g + N(0, sigma_z)`), at which point the regularised same-kernel scheme recovers
   cleanly (this is the literature's actual demonstration regime).

**Synthesis position:** the kernel regularisation is necessary and correct physics that must be
implemented regardless; it is the highest-probability single fix. But it should be validated on the
bridge **together with a re-injection leg**, because the existing seed-600 CRBs encode a delta-sharp
truth that no inference-side kernel can fully match. If the kernel fix recovers on the re-injected
(photo-z-consistent) event set but only partially de-rails on the as-injected set, that is the
diagnostic signature that the residual is the sim<->inference inconsistency, not a remaining
likelihood error — and the production fix is then "regularised kernel + consistent injection," not
the kernel alone.

**What is firmly settled regardless of which possibility holds:**
- D(h)/beta is the wrong lever (impossibility bound, Angle B).
- The bare-Gaussian kernel is a real, literature-confirmed defect that must be corrected.
- Candidate selection, MVN, n(z), completion, measurement scatter are all eliminated.

---

## References

- Gray et al. 2020, arXiv:1908.06050 (Eq. 25 in-cat ratio, Eq. 29 beta_G)
- Gair, Ghosh, Gray, Fishbach, Chen et al. 2024, arXiv:2212.08694 (Eq. 3 same-kernel ratio,
  Eq. 13 catalogue prior, Eq. 15 perfect-z limit, Eq. 16/17 posterior, Eq. 32 p_bg; Sec. 4.2
  Inconsistency 1 double-counting)
- arXiv:2502.17747 "Dark sirens and the impact of redshift precision" (Eq. 3; photometric
  catalogues unbiased, variance-only)
- arXiv:2503.18887 "Systematic bias in dark siren statistical methods" (numerator/selection
  inconsistency)
- Echoes from the dark, A&A 2026, arXiv:2509.18243 (Eq. 15 completeness coupling, same p_bg)
- Turski et al. 2023, MNRAS 526, 6224, arXiv:2302.12037 (photo-z shifts n(z) peak)
- Code: `scripts/bridge_closure/_bridge_sky.py:246-248` (bare kernel),
  `_bridge_lib.py:353` (injection at true z), `bayesian_statistics.py:174` (D(h)),
  `:289` (beta_Gbar), `:386-490` (Sum_global), `:~1646` (production numerator kernel)
- `.planning/HANDOFF-RAILING-INVESTIGATION-20260629.md:16-30` (impossibility-bound numbers)
- `scripts/bridge_closure/BRIDGE-FINDINGS.md`
