# [P3-MKER] R1 — 900121:20 analytic one-formula recompute (Refute-by / dissolve test)

**Scope:** zero-compute (reads + small analytic Python, one ~48s galaxy-catalogue load — see
§0 caveat). Task: CLAIM_P3_MKER_20260826.md §5 item (ii) — recompute the 900121:20 exhibit's
with-BH-mass kernel weight under a convolved effective width (R&V15 mass-relation scatter) and
state whether the −176.6-nats / e^−176-scale exhibit dissolves.

**Bottom line: NO, the exhibit does not dissolve — but not for the reason the claim assumes.**
Direct reconstruction of the two window-passed candidates' mass-kernel factors, cross-checked
two independent ways, shows the with-BH **mass kernel is already wide, not narrow**, for this
specific exhibit (pulls of 0.03σ and 0.31σ, weights O(0.3–0.6)) — nowhere near "~19σ / −176.6
nats." That figure's magnitude is not reproduced anywhere on the mass axis. The likely true
source is flagged in §5 as an open thread requiring further verification, not banked as fact.

---

## 0. Venue identification and caveat on compute scope

The claim's exhibit (`L_cat_with_bh = 1.39e-85`) matches the **`bt_900121`** (P3-2D twin,
`catalogue_numerator_survival_2d = "mz_sel"`) venue, not `bc_900121` (baseline, `"off"`):

- `bt_900121`: `event_likelihoods.csv` row `event_idx=20` →
  `L_cat_with_bh = 1.392199591828355e-85` (file:
  `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/diagnostics/event_likelihoods.csv`,
  column 10) — matches "1.39e-85" essentially exactly.
- `bc_900121` (baseline): same row, `L_cat_with_bh = 1.431038452316417e-85` — same order of
  magnitude, ~3% different (the `mz_sel` twin cell composes with the production Gaussian-product
  branch, per `bayesian_statistics.py:7090-7098` guard, so the two venues' with-BH numerators
  differ only by the twin's extra `E[S_4D]` factor — confirmed numerically in §2).

Venue flags, cite-verified from
`results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_meta.json`:
`catalogue_global_selection="phi"`, `catalogue_numerator_survival="phi"`,
`selection_in_completion_numerator="fused"`, `mass_filter_sigma="symmetric"`,
`catalogue_numerator_survival_2d="mz_sel"`, `catalogue_numerator_survival_2d_center="eff"`,
`git_commit="fb4ac4eea8bb415e38d542f6f458b3dd259060f0"`.

**Caveat (compute scope):** items 1–3 below required one read of the 1.68 GB
`darksiren_emri/galaxy_catalogue/reduced_galaxy_catalogue.csv` through the production
`GalaxyCatalogueHandler` loader (to resolve the two candidates' catalogue rows correctly —
`catalog_index` in the banked JSON is a post-prune, post-`reset_index()` row position, not a raw
file line, so it cannot be resolved without running the same load+prune+reset the pipeline runs).
This took 47.7s wall time on a `.venv` Python process — over the nominal "<30s" guideline but
still a **pure CPU read of an existing file** (`pd.read_csv` + vectorized pandas ops), no
simulation, no GPU, no `--evaluate`/`--simulation_steps` invocation, no cluster job. Flagging
per the "quote-verify" instruction rather than silently exceeding the guideline.

---

## 1. The two window-passed candidates: banked values, cited

### 1a. Per-candidate numerator/denominator (banked, no reconstruction needed)

File: `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bt_900121_work/seed900121/simulations/posteriors_with_bh_mass/h_0_73.json`,
key `["galaxy_likelihoods"]["20"]` — a list of `[galaxy_catalog_index, [num_no, den_no, num_w,
den_w, quad_w_out_num, quad_w_out_den]]` (6-column schema per
`bayesian_statistics.py:7372-7379`):

| galaxy_catalog_index | num_no (num_without_bh) | den_no | num_w (num_with_bh) | den_w |
|---|---|---|---|---|
| 6791158 | 9.813974564808528e-231 | 0.9822158525445621 | 3.1640186792771146e-231 | 0.8731263812985358 |
| 6791138 | 1.0824148618119582e-78 | 0.9748260343322143 | 5.873853242962365e-79 | 0.9100129585865653 |

(`bc_900121` values for the same event/candidates are essentially identical up to the `mz_sel`
twin factor — `num_w` = `3.25213693020771e-231` / `6.037718947254922e-79` respectively, ratio
to `bt`'s `num_w` ≈ 0.973 for both, consistent with one shared, near-unity `E[S_4D]` twin
factor. Confirms `bt`/`bc` candidate identity match.)

### 1b. The underlying GW detection (event 20 = CRB row 20, 0-based)

File: `results/campaign51_20260728/realistic_20260729/p3_2d_fleet_20260825/bc_900121_work/seed900121/simulations/prepared_cramer_rao_bounds.csv`,
row 20 (`df.iloc[20]`, 0-based positional row of a freshly-read, unfiltered 200-row CSV — see
`bayesian_statistics.py:3257` `self.cramer_rao_bounds = pd.read_csv(PREPARED_CRAMER_RAO_BOUNDS_PATH)`,
and `:3906-3915` for the label-preserving SNR/quality filters that run afterward without
resetting the index, so `detection_index=20` still names this exact row):

| field | value | source column |
|---|---|---|
| `M` (M_z,det, M_sun) | 1,333,246.128 | `M` |
| `d_L` (Gpc) | 0.2831422160 | `luminosity_distance` |
| `d_L` frac. unc. | 0.505632% | `sqrt(delta_luminosity_distance_delta_luminosity_distance)/luminosity_distance` |
| `M` frac. unc. | 3.891e-9 | `sqrt(delta_M_delta_M)/M` |
| `phiS` | 2.6140511580 rad | `phiS` |
| `phiS` unc. | 5.70697e-4 rad | `sqrt(delta_phiS_delta_phiS)` |
| `qS` | 3.0710658714 rad | `qS` |
| `qS` unc. | 9.97107e-4 rad | `sqrt(delta_qS_delta_qS)` |
| SNR | 235.8558 | `SNR` |
| `host_galaxy_index` (true injected host) | 6791134 | `host_galaxy_index` |

Detection field mapping verified against `darksiren_emri/datamodels/detection.py:134-151`.

### 1c. Catalogue-side host data (needs the catalogue load — §0 caveat)

Loaded via `GalaxyCatalogueHandler(M_min=M_SOURCE_FRAME_MIN, M_max=M_SOURCE_FRAME_MAX,
z_max=HOST_DRAW_Z_MAX)` (`darksiren_emri/constants.py:111,125,126`:
`M_SOURCE_FRAME_MIN=1e4`, `M_SOURCE_FRAME_MAX=1e7`, `HOST_DRAW_Z_MAX=1.5` — the exact
constructor call `main.py:154-160` uses), then `.reduced_galaxy_catalog.iloc[gid]`:

| catalog_index | PHI_S (rad) | THETA_S (rad) | REDSHIFT | host_M (M_sun, post-R&V15) | host_M_error (M_sun) |
|---|---|---|---|---|---|
| 6791158 | 2.595631 | 3.071807 | 0.031403 | 709,540.709 | 1,570,331.0 |
| 6791138 | 2.602957 | 3.069987 | 0.057443 | 709,540.709 | 894,866.276 |

`host_M`/`host_M_error` are `InternalCatalogColumns.BH_MASS`/`BH_MASS_ERROR`, which
`GalaxyCatalogueHandler._map_stellar_masses_to_BH_masses` overwrites **in place** with the
R&V15-converted BH mass (`handler.py:1137-1142`, calling `_empiric_stellar_mass_to_BH_mass_relation`
at `handler.py:1368-1382`) — these are **already BH masses**, not raw GLADE stellar masses.

**Key fact, verified by direct code read (`handler.py:1371-1381`):**

```python
BH_mass = np.exp(alpha + beta * np.log(stellar_mass / 10))
BH_mass_error = BH_mass * np.sqrt(
    sigma_int**2
    + d_alpha**2
    + (np.log(stellar_mass / 10) * d_beta) ** 2
    + (beta / stellar_mass * stellar_mass_error) ** 2
)
```

with `sigma_int = 0.24 * np.log(10)` (`handler.py:40`, R&V15 §4.1 intrinsic scatter) —
**the R&V15 0.24-dex intrinsic scatter is already the dominant term inside `host_M_error`**,
folded in at catalogue-load time, post the 555f018 fix (`/10` precedence bug on the
propagated-stellar-mass-error term, not on `sigma_int`). Confirmed numerically: for candidate
6791138, `host_M_error/host_M = 1.2612` — i.e. `ln(1.2612+1)`-ish scale, consistent with
`sigma_int=0.5526` (0.24 dex in nats) being dominant plus non-negligible propagated-error and
`d_beta` terms. **This directly contradicts CLAIM_P3_MKER §1(a)'s premise** ("the R&V15
mass-relation intrinsic scatter (~0.55 dex) is omitted") for the *current, post-555f018* code —
see §5 caveat.

---

## 2. Which kernel branch does the exhibit take?

Per the companion recon `mker_r1_consistency_recon.md` §1a (already banked, re-verified here
against the same line numbers): production `normalization_mode` defaults to
`"generator_marginal"` (`bayesian_statistics.py:6764`), and
`resolve_host_mass_kernel` (`:298-302`) resolves `host_mass_kernel="auto"` to `"gaussian"`
**unless** `normalization_mode == "mass_trunc"`. Neither `bc_900121` nor `bt_900121`'s meta
stamps show `mass_trunc` engaged (no such key in either `a22_stamp`), so **the exhibit uses the
analytic Gaussian-product kernel** (`bayesian_statistics.py:6601-6616`, scalar path; the
identical formula in the batch path used for the actual run is
`bayesian_statistics.py:7228-7244` minus the `_use_mass_trunc` branch):

```python
mu_gal_frac = _host_M_eff * (1 + z) / _det_M            # :6603
sigma_gal_frac = host_M_error * (1 + z) / _det_M         # :6604
sigma2_sum = _sigma2_cond + sigma_gal_frac**2            # :6612
mz_integral = exp(-0.5*(mu_cond-mu_gal_frac)**2/sigma2_sum) / sqrt(2*pi*sigma2_sum)  # :6613-6615
```

**Correction to the task brief:** the "mass-truncation / Gauss-Hermite block around lines
780–900" (`_mass_trunc_mz_integral`, its `narrow`/`mz_gh` crossover at `:857-865`) belongs to
the **experimental `mass_trunc` mode**, which is **not** engaged in production
(`normalization_mode="generator_marginal"` here) and is **not** what produced this exhibit's
numbers. The `narrow`-vs-`mz_gh` crossover, and its `_MASS_TRUNC_GH_CROSSOVER_SIGMA_LNM_MAX=0.1`
threshold (`:467`), is inert for this exhibit. The production formula (above) has **no
narrow/wide branch at all** — it is unconditionally the single analytic Gaussian product, always
"the narrow branch's formula" in form (`sigma2_sum = sigma_cond**2 + sigma_gal**2` — same
structure as `:861`), just without a fallback. This is a genuine framing gap in the task/claim,
flagged per instruction 2 ("if the exhibit already takes the convolved branch... say so") in the
opposite direction: there is no branch selection to make — production is always on this one
analytic formula, and CLAIM_P3_MKER's evidence citation of "the narrow kernel" is this same
formula, correctly identified in substance even though the specific code lines the task pointed
at (780–900) are the wrong ones.

---

## 3. Current (production, un-convolved) mass-kernel pull and weight

### 3a. `sigma_cond`, `mu_cond` — event-level, from the CRB Fisher matrix alone (no catalogue needed)

Built the 4×4 covariance `cov_4d` exactly as `bayesian_statistics.py:4219-4247` from §1b's row,
then the with-BH-mass Schur complement exactly as `:4300-4315`:

```
cov_obs = cov_4d[:3,:3]; cov_cross = cov_4d[3,:3]; cov_mz = cov_4d[3,3]
sigma2_cond = cov_mz - cov_cross @ pinv(cov_obs) @ cov_cross
proj        = cov_cross @ pinv(cov_obs)
```

Result: **`sigma_cond = 3.3103e-9`** (fractional/dimensionless, in the `a = M(1+z)/M_det`
coordinate) — matches the order of magnitude in CLAIM_P3_MKER §1(a)'s "production p50
fractional ~1e-8" (this specific event is unusually high-SNR = 235.9, so tighter than a typical
p50 event; consistent, not a contradiction). `proj = [2.9917e-6, 8.6802e-7, 3.4279e-9]`.

`mu_cond` (`:7228-7235`, point-mode since `_use_generator_point=True` under
`generator_marginal`): `mu_cond = mu_obs_4d[3] + (x_obs[:3] - mu_obs_4d[:3]) @ proj`, with
`x_obs = [host_phiS, host_qS, dist(host_z; h=0.73)/d_L]`. Because `proj`'s coefficients are
`O(1e-6)`–`O(1e-9)`, `mu_cond ≈ 1` for **both** candidates regardless of their (very different)
sky/z offsets — a projection this small cannot pull `mu_cond` away from 1 by a meaningful
amount:

- candidate 6791158: `mu_cond = 0.99999994`
- candidate 6791138: `mu_cond = 0.99999997`

### 3b. `mu_gal`, `sigma_gal` — per candidate, catalogue-side

`M_eff` = `eddington_shifted_host_mass(host_M, host_M_error)` (`bayesian_statistics.py:602-632`,
default `eddington_m="on"`), then `mu_gal_frac = M_eff*(1+z)/M`, `sigma_gal_frac =
host_M_error*(1+z)/M` (`:6603-6604`, `M = 1,333,246.128`):

| candidate | M_eff | mu_gal_frac | sigma_gal_frac |
|---|---|---|---|
| 6791158 | 1,343,486.517 | 1.039325 | **1.214812** |
| 6791138 | 987,783.026 | 0.783444 | **0.709749** |

### 3c. Current pull and weight

```
sigma2_sum = sigma_cond^2 + sigma_gal_frac^2      # sigma_cond^2 = 1.096e-17, totally negligible
pull = (mu_cond - mu_gal_frac) / sqrt(sigma2_sum)
mz_integral = exp(-0.5*pull^2) / sqrt(2*pi*sigma2_sum)
```

| candidate | sigma2_sum | pull (σ) | mz_integral (weight) | ln(mz_integral) |
|---|---|---|---|---|
| 6791158 | 1.47577 | **−0.0324** | **0.3282** | −1.114 |
| 6791138 | 0.50374 | **+0.3051** | **0.5365** | −0.623 |

**Independent cross-check (no Fisher reconstruction, banked-numbers-only):** since
`num_w = (gw_3d * mz_integral)` exactly (`:7323`, no `S_bar_phi` confound on the with-BH side —
that factor only multiplies `numerator_without_bh_mass`, `:7108`), and `num_no = gw_3d ×
S_bar_phi(host_z)` (confound present, `S_bar_phi ∈ [0,1]`), `num_w/num_no = mz_integral /
S_bar_phi(host_z) ≥ mz_integral`:

| candidate | num_w/num_no (bt, §1a) | mz_integral (§3c, direct) | implied S_bar_phi = ratio⁻¹×mz |
|---|---|---|---|---|
| 6791158 | 0.32188 (=3.1640e-231/9.8140e-231) | 0.3282 | ≈1.02 (self-consistent, near unity) |
| 6791138 | 0.54269 (=5.8739e-79/1.0824e-78) | 0.5365 | ≈0.99 (self-consistent, near unity) |

The two fully independent routes (Fisher-matrix-only reconstruction vs. the pipeline's own
banked `num_w`/`num_no`) agree to ≤2%. **Neither candidate's mass-kernel pull is anywhere near
"~19σ"; neither weight is anywhere near `e^-176`.** The mass kernel for this exhibit is wide
(`sigma_gal_frac` ~0.71–1.21, i.e. 71%–121% *fractional* catalogue-mass uncertainty — itself
already inclusive of the R&V15 0.24-dex intrinsic scatter per §1c) and dominates `sigma_cond`
(~3e-9) by **8 orders of magnitude**. A "narrow kernel" this is not.

---

## 4. Convolved recompute: add R&V15 scatter, explicit unit conversion

Per the task instruction, `sigma_eff^2 = sigma_cond^2 + sigma_gal^2 + sigma_scatter^2`, with
`sigma_scatter` converted to the kernel's own (dimensionless, fraction-coordinate `a`) units.
R&V15 §4.1 (already banked in the R0 sweep, §2 of CLAIM_P3_MKER): 0.24 dex intrinsic, 0.55 dex
total rms (0.50 dex measurement ⊕ 0.24 dex intrinsic in quadrature). A dex is a log10 unit:

```
sigma_scatter [nats, i.e. natural-log/relative units] = sigma_scatter [dex] * ln(10)
  0.24 dex -> 0.24 * ln(10) = 0.552620  (matches sigma_int already inside host_M_error, §1c)
  0.55 dex -> 0.55 * ln(10) = 1.266422
```

This is a *relative* (log-space) width; to express it in the kernel's `a`-coordinate (the same
linearization the production code already uses for `sigma_gal_frac` — a first-order
`sigma_M/M ≈ sigma_lnM` map, see `_mass_trunc_sigma_lnM` docstring, `bayesian_statistics.py:726-740`,
for the same convention applied to the (inactive-here) `mass_trunc` kernel), multiply by the
same `mu_gal_frac` center used above:

```
sigma_scatter_frac = sigma_scatter[nats] * mu_gal_frac
```

**Caveat flagged explicitly (per instruction 2's spirit):** using the FULL 0.55-dex figure here
**double-counts** the 0.24-dex intrinsic-scatter component, because §1c already showed
`host_M_error` (hence `sigma_gal_frac`) contains `sigma_int = 0.24 dex` as its dominant term.
Reporting both the literal-instruction case (0.55 dex, on top of the already-inclusive
`sigma_gal_frac`, as asked) and the double-count-avoiding case (0.24 dex on top, i.e. maximally
generous to the claim) below.

### 4a. `sigma_scatter_frac` per candidate

| candidate | mu_gal_frac | 0.24-dex → frac (0.552620×μ) | 0.55-dex → frac (1.266422×μ) |
|---|---|---|---|
| 6791158 | 1.039325 | 0.574406 | 1.316231 |
| 6791138 | 0.783444 | 0.432997 | 0.992176 |

### 4b. New sigma_eff, pull, weight

```
sigma2_eff = sigma_cond^2 + sigma_gal_frac^2 + sigma_scatter_frac^2
pull_new   = (mu_cond - mu_gal_frac) / sqrt(sigma2_eff)
mz_new     = exp(-0.5*pull_new^2) / sqrt(2*pi*sigma2_eff)
```

| candidate | case | sigma2_eff | sigma_eff | pull_new (σ) | mz_new | vs. narrow mz |
|---|---|---|---|---|---|---|
| 6791158 | +0.24 dex | 1.80557 | 1.34372 | −0.02927 | **0.2968** | 0.3282 → **0.90×** |
| 6791158 | +0.55 dex | 3.20892 | 1.79135 | −0.02195 | **0.2226** | 0.3282 → **0.68×** |
| 6791138 | +0.24 dex | 0.69117 | 0.83137 | +0.26053 | **0.4638** | 0.5365 → **0.86×** |
| 6791138 | +0.55 dex | 1.48864 | 1.21998 | +0.17747 | **0.3220** | 0.5365 → **0.60×** |

**The convolution makes both candidates' mass-kernel weight go DOWN, not up** — by 10–40%,
because these two candidates were never being suppressed on the mass axis (pulls already
`<0.31σ`): widening an already-good-fit kernel only dilutes its peak density (the
`1/sqrt(2*pi*sigma_eff^2)` normalization term falls faster than the `exp` numerator's near-1
value rises). A convolution can only rescue candidates that a too-narrow kernel was wrongly
suppressing; there is nothing here to rescue.

---

## 5. Does the exhibit dissolve?

**No — the −176.6-nats / e^−176-scale figure is not reproduced anywhere on the mass axis, with
or without the convolution, so the convolution cannot be what "dissolves" it.**

Direct arithmetic check on the exhibit's own headline number: `ln(1.39e-85) = -195.39` — **not**
`-176.6`. So "−176.6 nats" was never a description of the overall `L_cat_with_bh`; it must
describe some sub-component. The only sub-component this recompute can locate anywhere near that
scale is **not on the mass axis**: `ln(num_no)` for candidate 6791138 (the without-BH-mass
numerator, i.e. the **sky+distance** Gaussian `gw_3d` times the `S_bar_phi(z)` survival factor —
zero mass information) = `ln(1.0824e-78) = -179.52`, within ~2% of `-176.6`. And the sky-position
offset for that same candidate is `(phiS_det - phiS_gal)/phi_error = 19.44σ` —
**numerically matching the exhibit's own "~19σ" figure almost exactly**, on an axis (sky
position, not mass) that a mass-kernel convolution cannot touch at all.

**This is flagged as an open thread, not banked as fact:** an independent by-hand reconstruction
of the full 3-parameter `gw_3d` Gaussian (`bayesian_statistics.py:3137-3140` `_mvn_pdf` formula,
using the exact `cov_3d` from §1b) gave `ln(gw_3d) = -496.8` for this candidate — more suppressed
than the banked `ln(num_no)=-179.5`, which is impossible under `num_no = gw_3d × S_bar_phi(z)`,
`S_bar_phi ∈ [0,1]` (would require `S_bar_phi > 1`). There is an unresolved inconsistency in that
specific by-hand step (likely a bug in this recompute's manual `cov_3d`/`x_obs` construction, or
a `host_z_kernel`/scattered-catalogue resolution subtlety not accounted for), which this
zero-compute pass could not close out. **It does not affect the mass-axis conclusion in §3–4**,
which used a completely separate, doubly-cross-validated computation path (Fisher-matrix Schur
complement + independent `num_w`/`num_no` ratio check, agreeing to ≤2%, neither of which depends
on the `gw_3d`/`cov_3d` reconstruction that produced the inconsistency).

**Net verdict:**

1. The with-BH **mass kernel** for this exhibit is not narrow (`sigma_gal_frac` 71–121%
   fractional, dominating `sigma_cond`≈3e-9 by 8 orders of magnitude) and its weight is
   `O(0.3–0.6)`, not `e^-176`, under the *current, un-convolved* kernel — confirmed two
   independent ways.
2. Convolving in the R&V15 scatter (0.24 or 0.55 dex, explicit unit conversion in §4) moves the
   mass-kernel weight the WRONG way for this exhibit (down 10–40%), because there was no
   mass-axis suppression to rescue.
3. `L_cat_with_bh` therefore does **not** move toward `O(1)`-ish under this convolution — the
   claim's own diagnostic ("weight becomes O(1)-ish rather than e^-180") is not met, but not
   because the fix failed: it is because the mass kernel was never the exhibit's bottleneck.
4. **CLAIM_P3_MKER's characterization of this specific exhibit as a mass-kernel ~19σ/−176.6-nats
   event is not supported by direct decomposition of its own cited artifacts.** If the ~19σ/
   −176.6-nats figure is real (plausible — the numerical coincidence in this section is
   striking), its most likely locus, per this recompute, is the sky-position axis of the
   *without*-mass-information `gw_3d`/`S_bar_phi` term — a pre-existing, coarse-vs-tight
   filter/kernel mismatch on sky position and/or redshift (exactly the row #196 "kernel-zero"
   mechanism, just on a different observable than CLAIM_P3_MKER (a) names), not the with-BH-mass
   uncertainty budget this claim is about. This is a candidate correction to the exhibit's
   framing for the author to rule on, not a banked finding — the open inconsistency in §5's
   second paragraph means it needs a follow-up pass (a second independent `gw_3d` reconstruction,
   or a targeted print statement in the actual pipeline) before it can be banked.

---

## 6. Caveats

- The catalogue load (§0) added a 47.7s CPU-only step beyond the nominal "<30s one-liner"
  guidance; flagged rather than hidden. No simulation/GPU/cluster compute was run.
- §5's sky-axis lead (candidate explanation for "~19σ / −176.6 nats") is explicitly **not**
  verified to closure — an internal inconsistency (`S_bar_phi` would need to exceed 1) was found
  and reported rather than papered over, per the task's quote-verify discipline. Do not cite it
  as confirmed without a follow-up pass.
- §4's convolution used the *linearized* `sigma_scatter_frac = sigma_scatter[nats] × mu_gal_frac`
  map (consistent with the production `_mass_trunc_sigma_lnM` convention for the — inactive here
  — `mass_trunc` mode); a rigorous log-normal treatment (the "deferred log-normal refactor" named
  in CLAIM_P3_MKER §1(a)) could differ at the tens-of-percent level for these already-`O(1)`
  fractional errors, where the linear approximation is at its weakest. This does not change the
  §5 verdict's order of magnitude.
- This read used `bc_900121`'s CRB CSV (`prepared_cramer_rao_bounds.csv`) for the Fisher matrix;
  `bt_900121` should carry a byte-identical CRB for the same injection seed (the twin only
  touches the catalogue leg), consistent with the ≤2% cross-check in §3c, but the two files were
  not diffed directly.
- Per CLAUDE.md's approval-scope convention: this file is a **[DO]**-scope zero-compute read
  with no [RULE]/[STANDING] content; the §5 item 4 framing correction is presented as a candidate
  for author ruling, not self-ratified.
