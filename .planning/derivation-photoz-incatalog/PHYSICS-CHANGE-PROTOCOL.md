# Physics Change Protocol — photo-z-marginalised in-catalogue kernel

These are the proposed physics changes in the CLAUDE.md Physics-Change-Protocol format. They are
**not yet approved for production**. They must first be validated on the bridge harness (see
`BRIDGE-PROTOTYPE-SPEC.md`). When ported to `bayesian_statistics.py`, each requires `/gpd:physics-change`
and the `[PHYSICS]` commit prefix.

---

## Change 1 (PRIMARY) — regularise the per-host photo-z kernel with the comoving-volume prior

### 1. Old formula

Per-host redshift PDF in the in-catalogue numerator is the bare Gaussian:

```
  k_g(z) = norm(z; z_g, sigma_z) = (1/(sqrt(2pi) sigma_z)) exp(-(z - z_g)^2 / (2 sigma_z^2))
  N_g    = INT p_GW(d_L(z,h)) k_g(z) dz
```

- Production: `master_thesis_code/bayesian_inference/bayesian_statistics.py:~1646`
  (`single_host_likelihood`, `norm(host_z, host_z_error)` convolution)
- Bridge mirror: `scripts/bridge_closure/_bridge_sky.py:246-248`

### 2. New formula

Per-host redshift PDF is the comoving-volume-regularised POSTERIOR:

```
  p_red(z | z_g) = norm(z; z_g, sigma_z) * p_bg(z) / Z_g
  Z_g            = INT norm(z; z_g, sigma_z) p_bg(z) dz                  (per-host normalisation)
  p_bg(z)        proportional to (1/(1+z)) dVc/dz   (the SAME population prior D(h) integrates)
  N_g^corr       = INT p_GW(d_L(z,h)) p_red(z | z_g) dz
                 = ( INT p_GW(d_L(z,h)) norm(z;z_g,sigma_z) p_bg(z) dz ) / Z_g
```

`p_bg(z)` is evaluated at the h under test (it is h-dependent, exactly matching `D(h)`).

### 3. Reference

- Gair/Ghosh/Gray/Fishbach/Chen et al. 2024 "Hitchhiker's Guide", arXiv:2212.08694,
  **Eq. (16)** (posterior `p_red = L_red p_bg / norm`), **Eq. (17)** (Gaussian likelihood),
  **Eq. (32)** (`p_bg ∝ dVc/dz`, uniform in comoving volume).
- Gray et al. 2020, arXiv:1908.06050, **Eq. (25)** (per-galaxy redshift PDF convolved in the ratio).
- Echoes from the dark, A&A 2026, arXiv:2509.18243, **Eq. (15)** (same `p_bg` in in-cat, completion,
  selection).

### 4. Dimensional analysis

| symbol | units |
|---|---|
| `norm(z; z_g, sigma_z)` | 1/z |
| `p_bg(z)` (normalised) | 1/z |
| `Z_g = INT norm p_bg dz` | 1/z |
| `p_red = norm p_bg / Z_g` | 1/z, integrates to 1 |
| `p_GW` | dimensionless |
| `N_g^corr = INT p_GW p_red dz` | dimensionless |

Identical units to the old `N_g` -> the partition-norm ratio `p_i = (beta_G L_cat + B_num)/D(h)` is
dimensionally unchanged.

### 5. Limiting cases

- **sigma_z -> 0:** `norm -> delta(z - z_g)`, `Z_g -> p_bg(z_g)` cancels the `p_bg(z)` factor, so
  `p_red -> delta(z - z_g)` and `N_g^corr -> p_GW(d_L(z_g, h))`. Identical to the exact-host
  (delta-z) numerator -> preserves the bridge's delta-z MAP 0.725.
- **sigma_z -> inf:** `norm -> flat`, so `p_red -> p_bg(z)` for every host. The in-catalogue term
  collapses to `[INT p_GW p_bg dz] / [INT P_det p_bg dz]` = the Mandel-Farr-Gair empty-catalogue
  likelihood — the SAME population-prior integral as B_num and D(h). Posterior -> prior, unbiased.
- **Contrast (old kernel) sigma_z -> inf:** `norm -> uniform-in-z`, NOT uniform-in-volume; mismatches
  D(h)/B_num -> railed prior. This is exactly the +0.13 bias being fixed.

### Post-implementation checks (to report after approval)

- Sign convention: `p_red >= 0` everywhere; renormalisation makes it a proper PDF.
- Dimensional consistency: as table above.
- Reference comment to add above the changed line:
  ```python
  # p_red posterior kernel: Eq. (16)/(32) in Gair et al. (2024), arXiv:2212.08694
  ```

---

## Change 2 (OPTIONAL, consistency leg — expected NO-OP) — same kernel in the global selection

### 1. Old formula

```
  D_g       = P_det(d_L(z_g, h))                        (point evaluation, no convolution)
  Sum_global = Sum_{z_g < z_max(h)} w_g D_g
```
- `bayesian_statistics.py:386-490`

### 2. New formula

```
  D_g^conv  = INT P_det(d_L(z, h)) p_red(z | z_g) dz    (SAME regularised kernel as the numerator)
  Sum_global = Sum_{z_g < z_max(h)} w_g D_g^conv
```

### 3. Reference

Same-kernel requirement: Hitchhiker's Guide Eq. (3); Gray 2020 Eq. (25).

### 4. Dimensional analysis

`P_det` dimensionless, `p_red` [1/z], `INT P_det p_red dz` dimensionless — same as `D_g`. Unchanged.

### 5. Limiting case / expected effect

`P_det` is broad over sigma_z = 0.035 (in-cat galaxies at z ~ 0.046 << z_horizon, `P_det ~= 1`), so
`INT P_det p_red dz ~= P_det(z_g)`. Additionally Option A gives `Sum_global = C * beta_G`, so `beta_G`
cancels regardless. **Predicted change to the MAP: negligible.** This leg exists only to *prove* the
no-op and attribute any recovery to Change 1, not to fix anything. Do not port to production unless
the bridge shows a non-negligible effect (it should not).

---

## Change 3 (DO NOT IMPLEMENT — documented negative result) — D(h) rescale / re-derivation

Rejected by the impossibility bound (Angle B). Cancelling the +546,650 summed-log deficit by a
`D -> D*(h/0.73)^p` rescale requires `p ~= 1000`; the physical comoving-volume normalisation `p = 3`
supplies < 0.3% of it. `dD/dh < 0` is correct physics (fixed-d_L^max horizon). **No D(h)/beta change
is warranted.** Recorded here so it is not re-attempted.

---

## Routing note

Change 1 alters computed likelihood values (the in-catalogue redshift prior) -> it is a **physics
change**. When ported from the bridge to `bayesian_statistics.py:single_host_likelihood`, route
through `/gpd:physics-change`, add the reference comment, and prefix the commit `[PHYSICS]`. Per the
DERIVATION's honest caveat, the production fix may need to be paired with sim<->inference-consistent
injection (genuine photo-z scatter at simulation time); if so, the simulation-side change is a
separate physics change to the host-redshift assignment.
