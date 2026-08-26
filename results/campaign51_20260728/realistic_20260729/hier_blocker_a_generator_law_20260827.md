# [HIER] Blocker A: generator law, truth-theta, and hook placement (PA-HIER-1 / PA-HIER-2)

**Scope.** Zero-compute code-reading analysis only. No production files edited, no runs executed.
Repo root `/home/jasper/Repositories/darksiren-emri`, all line numbers verified by `Read`/`grep`
on 2026-08-27 against the working tree at the commit checked out at analysis time. This document
does not rule on any [DO]/[RULE]/[STANDING] item — it turns PA-HIER-1 and PA-HIER-2 into single-line
decisions for the author.

---

## 1. Every `host_mode` the mirror harness supports

`darksiren_emri/validation/correspondence_1d.py:1897-1903` (`draw_realization`'s `Literal` union) is
the authoritative enumeration — five values, each with a generating law for `z_true` in the same
method (:1993-2138) and a per-arm binding in `ARM_HOST_MODE` (:452-473):

| `host_mode` | arm | generating law for `z_true` | file:line |
|---|---|---|---|
| `"catalogue"` | `b0`, `bsig005`, `bsig025`, `eden05`, `eden2`, `bf1` | **delta** at the catalogue's own listed `z`: `host_z = pool.z[host_idx]`, no separate `z_true` draw at all — `host_z` is used directly as truth (`true_d_L = dist_vectorized(host_z, h=H_TRUE)` at :2141) | `:2003-2012`, `:2141` |
| `"population"` | `bout` | `host_z = draw_population_redshifts(rng, n, h=H_TRUE)` — an independent population-model draw, **no catalogue host or `z_g`/`z_error` involved at all** | `:2013-2020` |
| `"population_selected"` | `bsel`, `bself`, `bden` | `host_z = draw_selected_population_redshifts(rng, n, completeness, phi_survival_table, h=H_TRUE)` — population × (1−completeness) × survival weighting; **again no catalogue `z_g`/`z_error`** | `:2021-2052` |
| `"catalogue_selected"` | `b0i` | `z_true` drawn per-event from `k_g(z)·S_bar_phi(z;H_TRUE)` on the drawn host's own kernel window via `_draw_kernel_survival_redshifts`, where `k_g(z) ∝ N(z; z_g, z_error_eff_g) · w_pop(z) · f_k(z)` | `:2053-2088`, kernel at `:1440-1499` |
| `"catalogue_selected_2d"` | `b0i2d` | **byte-for-byte the same law as `"catalogue_selected"`** (`_B0i2DLatents.z_true` docstring, `:1572-1575`: "UNCHANGED from the 1D catalogue_selected mode — the mass-law extension does not perturb the z-draw law"), plus an orthogonal latent-mass extension | `:2090-2131`, z-draw reused from `:1440-1499` |

`"catalogue"` and `"catalogue_selected"`/`"catalogue_selected_2d"` are the only three that touch a
catalogue `z_g`/`z_error` pair at all — the two `"population*"` modes have no photo-z object for
θ=(b,s) to act on in the first place.

---

## 2. Truth-theta per mode, derived

θ = (b, s) is defined against the ESTIMATOR's assumed per-host Gaussian photo-z kernel:
`galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error_eff)`
(`darksiren_emri/bayesian_inference/bayesian_statistics.py:6247`; `host_z_error_eff` computed
immediately above at `:6223-6224`). The question per mode is whether the **generator's** law for
`z_true` is that same Gaussian (centered at the catalogue's listed `z_g`, width the catalogue's own
`host_z_error_eff`) at `(b,s)=(0,1)`, or something else.

- **`"catalogue"`** — the generator's law is a delta at `z_g` (0 variance). Since the estimator's
  kernel is a Gaussian of width `s·σ_eff`, matching a delta requires `s→0` — **truth is `s→0`, not
  `s=1`; `b` is undefined (there is no shift to speak of at zero width).** This is exactly PA-HIER-1's
  finding, now traced to the concrete code branch. `b0` — the arm the default `host_mode` binds to —
  is this mode.

- **`"population"` / `"population_selected"`** — no catalogue `z_g`/`z_error` pair exists for these
  hosts (`host_index_col = -1`, `in_catalog=False`, :2019-2020/:2034-2035); the θ=(b,s) photo-z
  kernel is **not a term in this generative law at all**. Truth-theta is not `(0,1)`, `(0,0)`, or any
  other value on this axis — the axis is **inapplicable**: a `[HIER]` run against `bout`/`bsel`/
  `bself`/`bden` cannot test whether θ recovers anything, because nothing in the data-generating
  process depends on θ.

- **`"catalogue_selected"` / `"catalogue_selected_2d"`** — the generator's density is
  `k_g(z)·S_bar_phi(z;H_TRUE)/Z`, `k_g(z) ∝ N(z; z_g, z_error_eff_g)·w_pop(z)·f_k(z)`. Isolating just
  the Gaussian factor: `_draw_kernel_survival_redshifts` (`:1490-1498`) builds
  `kernel_i = norm.pdf(z_i_grid, loc=host_z[i], scale=z_error_eff[i])` — `loc` is the **unshifted**
  catalogue `z_g` (`b=0`) and `scale` is the **unscaled** `host_z_error_eff` (`s=1`). The remaining
  factors `w_pop(z)·f_k(z)·S_bar_phi(z)` are **not extraneous to the estimator's side** — see §3 —
  so, subject to the identity check in §4, **truth-theta = (0,1) genuinely holds for this mode.**

---

## 3. Which mode makes truth-theta = (0,1) genuinely hold

**`"catalogue_selected"` (arm `b0i`) — and its byte-identical-on-the-z-axis sibling
`"catalogue_selected_2d"` (arm `b0i2d`) — are the only modes where truth-theta = (0,1) is a genuine
statement, not a bookkeeping artifact.** Reasoning, closing the loop the reviewer opened:

1. **The Gaussian factor is exactly the estimator's kernel at `(0,1)`.** Confirmed in §2: `loc=z_g`,
   `scale=host_z_error_eff` unscaled/unshifted, both sides using the byte-identical
   `host_z_error_eff` functional form (§6).

2. **The extra `w_pop(z)·f_k(z)` factor is not a generator-only artifact — it's the estimator's OWN
   `volume_deconv` prior weight.** Production's `galaxy_redshift_prior_pdf(z)` under
   `_use_volume_deconv` (`bayesian_statistics.py:6335-6339`) is `base·w_pop_eff(z)/Z_g`, `base` being
   exactly `galaxy_redshift_normal_distribution.pdf(z)` — the **same multiplicative structure** the
   generator's `k_g(z)` builds. `PRODUCTION_FLAGS["--host_z_kernel"] = "volume_deconv"`
   (`correspondence_1d.py:330`), so this factor is live on the estimator side under the run flags
   `[HIER]` inherits.

3. **The extra `S_bar_phi(z)` factor is not a generator-only artifact either — it's production's own
   `catalogue_numerator_survival="phi"` term, ACTIVE under `PRODUCTION_FLAGS`.**
   `PRODUCTION_FLAGS["--normalization_mode"] = "absolute_marginal"` (`correspondence_1d.py:329`).
   `BayesianStatistics.evaluate`'s `catalogue_numerator_survival` parameter defaults to `"auto"`
   (`bayesian_statistics.py:3421`), and `"auto"` resolves to `"phi"` exactly when
   `normalization_mode == "absolute_marginal"` (`:3535-3541`) — which it is. With `_cat_num_surv ==
   "phi"`, `numerator_integrant_without_bh_mass` (`:6344-6370`) multiplies by
   `np.interp(z, z_grid, s_phi)` off `catalogue_survival_table` (`:6363-6369`) — the same
   `S_bar_phi(z;h)` factor the generator's `k_g(z)·S_bar_phi(z)` law samples from.

4. **Both `w_pop`/`f_k` and the completeness pixel lookup are literally shared code, not
   reimplementations.** `correspondence_1d.py:227-238` imports `_completeness_at_host_nodes`,
   `_host_pixels`, and `precompute_phi_marginal_survival` directly from
   `bayesian_inference.bayesian_statistics`; `:262` imports `comoving_volume_element` from
   `physical_relations` (the single production source). There is no parallel/duplicated
   `w_pop`/completeness implementation on the generator side to drift from the estimator's.

So under `"catalogue_selected"`, the generator draws exactly the θ=(0,1) conditional the estimator's
own numerator kernel represents (Gaussian symmetry: drawing `z_true ~ N(z_true; z_g,σ)·π(z_true)`
given a fixed observed `z_g` is the same conditional a forward model `z_true~π(·), z_g=z_true+N(0,σ)`
would produce, because `N(z_g;z_true,σ)=N(z_true;z_g,σ)`) — **provided** the two independently-built
`phi_survival_table` objects used on the two sides are numerically the same table (§4, unverified).

**`"catalogue"` gives `s→0`; `"population"`/`"population_selected"` give an inapplicable axis; no
mode other than `"catalogue_selected"`/`"catalogue_selected_2d"` gives `(0,1)`.** This is the
decisive structural finding: **the `[HIER]` prereg's implicit default (`host_mode="catalogue"`, arm
`b0`) is the one mode in the whole enumeration that cannot support the thread's premise; the fix is
a one-line `host_mode` change to `"catalogue_selected"` (arm `b0i`), not a redesign.**

---

## 4. What must be checked to CERTIFY generator/estimator identity at truth-theta=(0,1)

Recommended mode: `"catalogue_selected"` (`b0i`). Open certification items, in order of risk:

1. **`phi_survival_table` value-identity.** The generator's table is built by a *separate*
   construction (`build_bsel_selection_objects`, called before any mirror event is drawn — module
   docstring `:91-106` discloses this is "a SEPARATE object ... not the literal same Python instance
   `evaluate()` builds internally"). Both constructions call `precompute_phi_marginal_survival` with
   `h_true` and should be deterministic given identical `(injection_dir, snr_threshold, dl_bins,
   mass_bins, estimator, expected_z_max, z_max_cap)` — but this equality of *inputs* has not been
   asserted anywhere in the code read for this task. **Not certified; needs an explicit runtime
   equality assertion or a shared-instance refactor**, one line, before the first `[HIER]` run banks
   a number.

2. **Quadrature-family alignment for the S̃_φ,g normalization vs. the estimator's `Z_g`.** These
   *are* aligned, contrary to the PA-2D-2/PA-2D-3 precedent's default risk: `kernel_smeared_survival`
   uses `_B0I_KERNEL_QUAD_N = 50` Gauss-Legendre nodes (`:1156-1157`, comment "mirrors `_HOST_QUAD_N`'s
   default"), and production's own `Z_g`/numerator quadrature is `_HOST_QUAD_N = 50`
   (`bayesian_statistics.py:409`, `FIXED_QUAD_N = _HOST_QUAD_N` at `:6139`) — same rule (Gauss-
   Legendre via `roots_legendre`), same node count. **This particular pairing is certified by code
   inspection, no PA-2D-2/3-style mismatch here.**

3. **The 401-point uniform inverse-CDF *draw* grid is a genuinely different numerical operation from
   item 2 and is un-audited.** `_B0I_ZTRUE_GRID_N = 401` (`:1164`) builds a per-host **uniform**
   grid inside `_draw_kernel_survival_redshifts` (`:1491-1498`) to actually sample `z_true` — this is
   not the GL-50 rule used for the S̃_φ,g normalizing scalar (the per-host normalizing constant
   "cancels in the inverse-CDF normalization", per the docstring `:1462-1465`, so item 2's alignment
   does not automatically cover this). This is precisely the PA-2D-2/PA-2D-3 failure shape one axis
   over: a borrowed/independent discretization approximating a density that itself has structure
   (kinks in `S_bar_phi(z)` from its own construction, or in `w_pop_eff(z)` from the completeness
   pixel edges) that a coarse-ish grid can under-resolve in the wide-window/near-horizon regime —
   both prior PA-2D amendments were exactly this class of failure, found only by an explicit
   convergence spot-check (401 vs. a much finer grid, or a brute-force arbiter) on the
   widest-window/highest-z hosts. **Not certified; recommend the same spot-check discipline PA-2D-2/
   PA-2D-3 used before any `b0i`-mode `[HIER]` number is banked** — this is compute, out of this
   task's zero-compute scope, but should be a named pre-registration gate item.

4. **`SIGMA_V_PEC_KM_S` no-op check.** Both sides' `host_z_error_eff` reduce to `sqrt(z_error^2)`
   only while `SIGMA_V_PEC_KM_S == 0.0`; both import the same `constants.SIGMA_V_PEC_KM_S`
   (`correspondence_1d.py:246`), so this is a shared-constant guarantee, not a live risk — worth one
   assertion in the harness for documentation, not a real certification gap.

---

## 5. Generator-side vs. estimator-side sites; where the θ hook must go

**Generator-side (constructs the synthetic DATA — must NOT be theta's hook):**

| site | role |
|---|---|
| `correspondence_1d.py:1167-1188` `host_z_error_eff()` | generator's effective-width helper |
| `correspondence_1d.py:1323` (inside `kernel_smeared_survival`) | caller #1 — feeds the host-draw WEIGHT (`S̃_φ,g`) |
| `correspondence_1d.py:1485` (inside `_draw_kernel_survival_redshifts`) | caller #2 — feeds the per-event `z_true` DRAW |
| `correspondence_1d.py:2003-2138` (`draw_realization` host_mode branches) | assembles the mirror-universe DataFrame (`z_true`, `host_z`, `obs_d_L`, …) |

PA-HIER-2's finding stands and is now fully traced: `host_z_error_eff` in `correspondence_1d.py` has
exactly these two callers and both are on the path that produces the synthetic data, not the path
that scores it. A θ hook placed here would rescale/shift what gets **drawn as truth**, not what the
posterior **assumes** — the mirror would silently stop testing "does jointly inferring θ absorb the
tilt" and start testing "does injecting a different photo-z error recreate the tilt", a different
(and already-explored, see `sigma_z_scale` in `host_pool_for_sigma_scale`, `:1860-1891`) question.

**Estimator-side (production's own posterior evaluation, reached via `bs.evaluate(...)` at
`correspondence_1d.py:2844` under `PRODUCTION_FLAGS`, in-scope for a hook):**

| site | role |
|---|---|
| `bayesian_statistics.py:6223-6224` | `host_z_error_eff` inline (the `absolute_marginal` numerator/`Z_g` path) |
| `bayesian_statistics.py:6247` | `galaxy_redshift_normal_distribution = norm(loc=host_z, scale=host_z_error_eff)` — **the object θ is meant to reparametrize** |
| `bayesian_statistics.py:6335-6339` | `galaxy_redshift_prior_pdf(z)` — the Gaussian × `w_pop_eff(z)/Z_g` prior the numerator integrant consumes |
| `bayesian_statistics.py:6878-6879` | `host_z_error_eff` inline, vectorized sibling (denominator/quadrature batch path feeding `:6888-6923`) |
| `bayesian_statistics.py:6923`, `:7031` | `_gaussian_pdf(·, host_z[:,None], host_z_error_eff[:,None])` — batched evaluations of the same kernel |
| `bayesian_statistics.py:7518-7519` | a further `host_z_error_eff` inline copy (a different caller context) |
| `bayesian_statistics.py:1669-1672` | a further `sigma_z_pv`/`sigma_eff` inline copy in a different quadrature-weight builder |

**Hook placement:** θ=(b,s) must reparametrize `host_z`→`host_z + b·(1+host_z)` and
`host_z_error_eff`→`s·host_z_error_eff` at **every one of the estimator-side sites above** (there
are at least six near-duplicate inline copies of the same formula/kernel-construction pattern, not
one canonical function — a genuine landmine for a partial hook that only patches the first site
found). The cleanest implementation is a single new parameter threaded into `evaluate()`
(mirroring how `normalization_mode`/`host_z_kernel` are already threaded) that all six sites read
from, rather than patching each call site independently — but that is an implementation decision for
whoever authors the physics-change proposal, not this task.

---

## 6. Docstring parity claim at `correspondence_1d.py:1173` — verified

The docstring (`:1170-1177`) claims `host_z_error_eff`'s functional form is "byte-identical ...
to production's per-host sigma (`bayesian_statistics.py:5908-5909`)". Current content at that cited
line range (read `2026-08-27`):

```
5908	            theta_g[sl],
5909	            M_eff_g[sl],
```

— unrelated code (a mass-marginal denominator batch call, `_bh_mass_denominator_inner_m_integral_batch`).
**The reviewer's "stale line number" flag is CONFIRMED: `:5908-5909` is not the formula.** The
formula the docstring describes now lives at (all verified byte-identical to the generator's form,
`np.sqrt(z_error**2 + ((1+z)*SIGMA_V_PEC_KM_S/SPEED_OF_LIGHT_KM_S)**2)`):

- `bayesian_statistics.py:6223-6224` (the `absolute_marginal` numerator/`Z_g` path — the one
  `PRODUCTION_FLAGS` actually reaches, and the recommended citation target)
- `bayesian_statistics.py:6878-6879` (vectorized sibling)
- `bayesian_statistics.py:7518-7519` (a further caller)
- `bayesian_statistics.py:1669-1672` (`sigma_z_pv`/`sigma_eff`, same formula, different quadrature-weight builder)

**The functional-form parity claim itself is CONFIRMED TRUE** (all four production sites and the
generator's `:1186-1188` are the identical formula) — **only the specific cited line number is
stale** and should be corrected to `bayesian_statistics.py:6223-6224` (with a note that three further
byte-identical copies exist at `:6878-6879`, `:7518-7519`, `:1669-1672`).

---

## Summary for the author

- **PA-HIER-1 is CONFIRMED and now fully diagnosed**: `host_mode="catalogue"` (the `[HIER]`
  prereg's implicit default, arm `b0`) generates `z_true` as a delta at the catalogue `z`; truth on
  the `s` axis is `s→0`, not `s=1`.
- **The fix is a one-line mode change, not a redesign**: `host_mode="catalogue_selected"` (arm
  `b0i`) is the only mode in the five-way enumeration where truth-theta genuinely holds at `(0,1)`
  — derived, not assumed, in §2-§3.
- **PA-HIER-2 is CONFIRMED and now fully diagnosed**: the reviewed hook site
  (`correspondence_1d.host_z_error_eff`, called only at `:1323`/`:1485`) is generator-side; the θ
  hook must instead reparametrize the six estimator-side sites enumerated in §5, of which
  `bayesian_statistics.py:6247` is the canonical one.
- **Two certification items remain open before a number can be banked** (§4 items 1 and 3): the
  two independently-built `phi_survival_table` objects' value-identity, and a PA-2D-2/PA-2D-3-style
  convergence spot-check on the 401-point inverse-CDF draw grid for the widest-window hosts. Both
  are compute, out of this zero-compute task's scope, and should be named pre-registration gate
  items for whoever runs `b0i`-mode `[HIER]`.
- **Item 6 (docstring line-number staleness) is a trivial fix**: the functional-form claim holds;
  only the cited line number needs updating.
